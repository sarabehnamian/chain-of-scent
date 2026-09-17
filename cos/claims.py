"""Turning free-text reasoning into testable claims.

A claim is a triple: (functional group, odour descriptor, direction), attached
to the molecule the agent proposed in that round. Example: "the aldehyde should
give it a green top note" becomes (aldehyde, green, increase) on that molecule.

Two extractors:

  RuleExtractor  deterministic co-occurrence within a sentence, using the group
                 alias table and the descriptor vocabulary. No API cost, fully
                 reproducible, and it serves as a validation check on the LLM
                 extractor. Report agreement between the two in the paper.
  LLMExtractor   an LLM asked for strict JSON over the same closed vocabulary.
                 Catches paraphrase the rule extractor misses.

Both are constrained to a closed vocabulary, so the downstream intervention is
always well defined. Anything outside the vocabulary is dropped and counted,
so you can report extraction coverage rather than hiding it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

from . import chem

NEGATIONS = ("without", "avoid", "remove", "lose", "not ", "no ", "reduce",
             "suppress", "less ", "instead of", "drop")


@dataclass
class Claim:
    target_id: str
    arm: str
    round_index: int
    molecule: str
    group: str            # canonical group name from chem.GROUPS
    descriptor: str
    direction: str        # "increase" or "decrease"
    sentence: str
    extractor: str

    def to_json(self) -> dict:
        return asdict(self)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;!?])\s+|\n+", text) if s.strip()]


class RuleExtractor:
    name = "rule"

    def __init__(self, descriptors: list[str]):
        self.descriptors = sorted(descriptors, key=len, reverse=True)
        self.alias_to_group = {
            a.replace("_", " "): g.name for a, g in chem.GROUP_BY_NAME.items()
        }

    def __call__(self, text: str, molecule: str, target_id: str, arm: str,
                 round_index: int) -> list[Claim]:
        out: list[Claim] = []
        seen: set[tuple[str, str]] = set()
        for sent in _sentences(text):
            low = sent.lower()
            groups = {g for a, g in self.alias_to_group.items()
                      if re.search(rf"\b{re.escape(a)}\b", low)}
            descs = {d for d in self.descriptors
                     if re.search(rf"\b{re.escape(d)}\b", low)}
            if not groups or not descs:
                continue
            direction = "decrease" if any(n in low for n in NEGATIONS) else "increase"
            for g in groups:
                for d in descs:
                    if (g, d) in seen:
                        continue
                    seen.add((g, d))
                    out.append(Claim(target_id, arm, round_index, molecule, g, d,
                                     direction, sent, self.name))
        return out


EXTRACT_SYSTEM = """You extract structured claims from a chemist's reasoning text.

A claim links ONE functional group to ONE odour descriptor with a direction.
Only use these group names: {groups}
Only use these descriptors: {descriptors}

Return STRICT JSON, an array of objects, nothing else, no markdown fence:
[{{"group": "...", "descriptor": "...", "direction": "increase"|"decrease",
   "sentence": "the sentence it came from"}}]

Rules:
- Only extract claims the text actually makes about THIS molecule's smell.
- direction is "increase" if the group is said to produce or strengthen the
  descriptor, "decrease" if it is said to remove or weaken it.
- Drop anything you cannot map onto the two vocabularies above.
- Return [] if there are no claims."""


class LLMExtractor:
    name = "llm"

    def __init__(self, llm, descriptors: list[str]):
        self.llm = llm
        self.descriptors = set(descriptors)
        self.system = EXTRACT_SYSTEM.format(
            groups=", ".join(sorted({g.name for g in chem.GROUPS})),
            descriptors=", ".join(sorted(descriptors)),
        )

    def __call__(self, text: str, molecule: str, target_id: str, arm: str,
                 round_index: int) -> list[Claim]:
        if not text.strip():
            return []
        resp = self.llm(self.system, [{"role": "user", "content": text}], None)
        body = resp.text.strip()
        body = re.sub(r"^```(?:json)?|```$", "", body, flags=re.M).strip()
        try:
            items = json.loads(body)
        except Exception:
            return []
        out = []
        for it in items if isinstance(items, list) else []:
            g = str(it.get("group", "")).lower().strip()
            d = str(it.get("descriptor", "")).lower().strip()
            if g not in chem.GROUP_BY_NAME or d not in self.descriptors:
                continue
            out.append(Claim(target_id, arm, round_index, molecule,
                             chem.GROUP_BY_NAME[g].name, d,
                             "decrease" if str(it.get("direction")) == "decrease"
                             else "increase",
                             str(it.get("sentence", ""))[:400], self.name))
        return out


def extract_from_logs(logs: list[dict], extractor) -> list[Claim]:
    claims: list[Claim] = []
    for log in logs:
        for r in log["rounds"]:
            if not r.get("proposal") or not r.get("reasoning"):
                continue
            claims.extend(extractor(r["reasoning"], r["proposal"],
                                    log["target_id"], log["arm"], r["index"]))
    return claims


def agreement(a: list[Claim], b: list[Claim]) -> dict:
    """Jaccard agreement between two extractors on (molecule, group, descriptor)."""
    key = lambda c: (c.molecule, c.group, c.descriptor)
    sa, sb = {key(c) for c in a}, {key(c) for c in b}
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return {"n_rule": len(sa), "n_llm": len(sb), "jaccard": inter / union,
            "recall_of_llm_by_rule": inter / (len(sb) or 1)}
