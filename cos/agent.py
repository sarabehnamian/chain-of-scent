"""The design loop and its ablation arms.

Five arms, each given the same target, the same round budget and the same
oracle. The comparison between them is the experiment: with a scoring oracle
in the loop an agent can improve by hill-climbing with no useful reasoning at
all, so "reasoning correlates with success" means nothing until these are on
the table.

  full            reasoning + tool feedback  (the system under test)
  no_reasoning    proposals only, no explanation, same feedback
  scrambled       its own reasoning is swapped for reasoning written for a
                  different target; proposals continue from there
  no_feedback     reasoning, but oracle scores are withheld until the end
  random_mutation no LLM at all; RDKit mutation hill-climb on the same budget

Everything is logged to JSONL: one record per round, including the verbatim
reasoning text, which is the raw material for the faithfulness analysis.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict

import numpy as np

from . import chem
from .llm import LLMResponse
from .oracle import FingerprintOracle, Prediction
from .targets import Target

ARMS = ("full", "no_reasoning", "scrambled", "no_feedback", "random_mutation")

TOOL_SPEC = [{
    "name": "predict_odor",
    "description": (
        "Predict the odour descriptor profile of a molecule from its SMILES "
        "string. Returns the top predicted descriptors with scores in [0,1], "
        "the scores for the target descriptors, and a confidence note."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "smiles": {"type": "string",
                       "description": "SMILES string of the candidate molecule"},
        },
        "required": ["smiles"],
    },
}]

SYSTEM_FULL = """You are designing a small volatile molecule with a specified smell.

Target odour profile: {target}

You have a tool, predict_odor, that scores a SMILES string against odour \
descriptors. You have {rounds} rounds. Each round:

1. State your chemical reasoning FIRST, in plain prose, before proposing \
anything. Say which structural features you are adding, keeping or removing \
and which odour descriptor each one is meant to produce. Name functional \
groups explicitly (aldehyde, ester, thiol, lactone, alkene, phenol, and so on) \
and name the descriptor each is meant to drive.
2. Then call predict_odor with exactly one candidate SMILES.

Keep candidates plausible as odorants: molecular weight under 300, neutral, \
no metals, synthesisable. Do not propose the same molecule twice.
When you are done, write FINAL: <smiles> on its own line."""

SYSTEM_NO_REASONING = """You are designing a small volatile molecule with a \
specified smell.

Target odour profile: {target}

You have a tool, predict_odor, that scores a SMILES string. You have {rounds} \
rounds. Each round, call predict_odor with exactly one candidate SMILES. \
Output no explanation, no commentary and no reasoning of any kind, only the \
tool call. Keep candidates plausible as odorants: molecular weight under 300, \
neutral, no metals. Do not propose the same molecule twice.
When you are done, write FINAL: <smiles> on its own line."""

SYSTEM_NO_FEEDBACK = """You are designing a small volatile molecule with a \
specified smell.

Target odour profile: {target}

Propose {rounds} candidate molecules, one at a time. State your chemical \
reasoning first each time, naming functional groups and the descriptor each is \
meant to drive, then give the candidate as: CANDIDATE: <smiles>
You will not receive any scoring feedback between candidates. \
Keep candidates plausible as odorants: molecular weight under 300, neutral.
When you are done, write FINAL: <smiles> on its own line."""

SMILES_RE = re.compile(r"(?:FINAL|CANDIDATE):\s*([^\s`]+)")


@dataclass
class Round:
    index: int
    reasoning: str
    proposal: str | None
    valid: bool
    target_match: float
    top_predicted: list = field(default_factory=list)
    nn_similarity: float = 0.0
    sa_score: float | None = None
    plausible: bool = True
    max_ensemble_std: float = 0.0
    note: str = ""


@dataclass
class RunLog:
    target_id: str
    target: list[str]
    arm: str
    model: str
    seed: int
    rounds: list[Round] = field(default_factory=list)
    best_smiles: str | None = None
    best_score: float = 0.0
    final_smiles: str | None = None
    usage: dict = field(default_factory=lambda: {"input": 0, "output": 0})
    error: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def _score(pred: Prediction, target: list[str]) -> float:
    return pred.target_score(target) if pred.valid else 0.0


def _record(i: int, reasoning: str, smi: str | None, pred: Prediction | None,
            target: list[str]) -> Round:
    if pred is None or not pred.valid:
        return Round(index=i, reasoning=reasoning, proposal=smi, valid=False,
                     target_match=0.0, note=(pred.note if pred else "no proposal"))
    return Round(
        index=i, reasoning=reasoning, proposal=pred.smiles, valid=True,
        target_match=_score(pred, target), top_predicted=pred.top(6),
        nn_similarity=pred.nn_similarity, sa_score=pred.sa_score,
        plausible=pred.plausible,
        max_ensemble_std=max(pred.ensemble_std.values()) if pred.ensemble_std else 0.0,
        note=pred.note,
    )


# --------------------------------------------------------------------------
# LLM arms
# --------------------------------------------------------------------------
def run_llm(llm, oracle: FingerprintOracle, target: Target, arm: str = "full",
            rounds: int = 6, seed: int = 0,
            scramble_pool: list[str] | None = None) -> RunLog:
    tgt = ", ".join(target.descriptors)
    system = {"full": SYSTEM_FULL, "scrambled": SYSTEM_FULL,
              "no_reasoning": SYSTEM_NO_REASONING,
              "no_feedback": SYSTEM_NO_FEEDBACK}[arm].format(target=tgt, rounds=rounds)
    log = RunLog(target_id=target.id, target=target.descriptors, arm=arm,
                 model=getattr(llm, "model", "unknown"), seed=seed)
    rng = np.random.default_rng(seed)
    messages: list[dict] = [{"role": "user",
                             "content": f"Design a molecule smelling of: {tgt}. Begin."}]
    use_tools = arm != "no_feedback"

    try:
        for i in range(rounds):
            resp: LLMResponse = llm(system, messages,
                                    TOOL_SPEC if use_tools else None)
            log.usage["input"] += resp.usage.get("input", 0)
            log.usage["output"] += resp.usage.get("output", 0)
            reasoning = resp.text.strip()

            if use_tools and resp.tool_calls:
                call = resp.tool_calls[0]
                smi = str(call["input"].get("smiles", "")).strip()
                pred = oracle.predict(smi)
                log.rounds.append(_record(i, reasoning, smi, pred, target.descriptors))

                # what the model is allowed to remember about its own thinking
                if arm == "scrambled" and scramble_pool:
                    reasoning_shown = str(rng.choice(scramble_pool))
                else:
                    reasoning_shown = reasoning

                assistant_blocks = []
                if reasoning_shown:
                    assistant_blocks.append({"type": "text", "text": reasoning_shown})
                assistant_blocks.append({"type": "tool_use", "id": call["id"],
                                         "name": call["name"], "input": call["input"]})
                messages.append({"role": "assistant", "content": assistant_blocks})
                messages.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": call["id"],
                    "content": json.dumps(
                        pred.as_tool_result(target.descriptors)),
                }]})
            else:
                hits = SMILES_RE.findall(resp.text)
                smi = hits[-1] if hits else None
                pred = oracle.predict(smi) if smi else None
                log.rounds.append(_record(i, reasoning, smi, pred, target.descriptors))
                messages.append({"role": "assistant", "content": resp.text})
                messages.append({"role": "user",
                                 "content": "Next candidate." if i < rounds - 1
                                 else "Give your FINAL molecule."})
                if resp.stop_reason == "end_turn" and "FINAL:" in resp.text and use_tools:
                    break
    except Exception as exc:  # keep partial logs rather than losing the run
        log.error = f"{type(exc).__name__}: {exc}"

    valid = [r for r in log.rounds if r.valid]
    if valid:
        best = max(valid, key=lambda r: r.target_match)
        log.best_smiles, log.best_score = best.proposal, best.target_match
        log.final_smiles = valid[-1].proposal
    return log


# --------------------------------------------------------------------------
# no-LLM baseline
# --------------------------------------------------------------------------
def run_random_mutation(oracle: FingerprintOracle, target: Target, rounds: int = 6,
                        seed: int = 0, start: str = "CCCCCC=O",
                        children: int = 4) -> RunLog:
    """Hill-climb with random RDKit edits, same oracle-call budget per round as
    the LLM arms times `children`. Report the budget honestly in the paper."""
    rng = np.random.default_rng(seed)
    log = RunLog(target_id=target.id, target=target.descriptors,
                 arm="random_mutation", model="none", seed=seed)
    cur = start
    for i in range(rounds):
        cands = []
        for _ in range(children):
            c, label = chem.random_mutate(cur, rng)
            if c:
                cands.append((c, label))
        if not cands:
            break
        scored = [(oracle.predict(c), lab) for c, lab in cands]
        pred, lab = max(scored, key=lambda p: _score(p[0], target.descriptors))
        r = _record(i, f"random edit: {lab}", pred.smiles, pred, target.descriptors)
        log.rounds.append(r)
        if _score(pred, target.descriptors) >= _score(oracle.predict(cur),
                                                      target.descriptors):
            cur = pred.smiles
    valid = [r for r in log.rounds if r.valid]
    if valid:
        best = max(valid, key=lambda r: r.target_match)
        log.best_smiles, log.best_score = best.proposal, best.target_match
        log.final_smiles = valid[-1].proposal
    return log


def append_log(log: RunLog, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(log.to_json()) + "\n")


def read_logs(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]
