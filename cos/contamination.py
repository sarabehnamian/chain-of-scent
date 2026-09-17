"""Memorisation probe.

GoodScents and Leffingwell descriptor tables are public and have been scraped
into many corpora, so a frontier model has very likely seen molecule-descriptor
pairs from exactly this dataset. Any claim about "reasoning" has to survive
that, and the honest way to handle it is to measure it rather than hope.

Three probes, all cheap:

  recall     given a SMILES and no tool, list its odour descriptors. Scored
             against the curated labels. High recall = memorised.
  name       given a SMILES, name the compound. A model that can name it can
             probably recall its odour too.
  scrambled  same recall question on a molecule that does not exist in the set
             (an ablated variant). Establishes the floor from chemical
             plausibility alone rather than recall.

Condition every faithfulness result on the recall score of the molecules
involved. If faithfulness is high only on memorised molecules, that is the
finding.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

import numpy as np

from . import chem
from .data import OdorData

RECALL_SYSTEM = """You are an expert flavour and fragrance chemist.
Given a SMILES string, list the odour descriptors that this compound is \
conventionally described with. Use only single words or short standard terms.
Answer with STRICT JSON and nothing else:
{"compound_name": "name or null", "descriptors": ["...", "..."], \
"confidence": "high"|"medium"|"low"}"""


@dataclass
class ProbeResult:
    smiles: str
    in_dataset: bool
    true_labels: list[str]
    predicted: list[str]
    named: str | None
    confidence: str
    precision: float
    recall: float
    f1: float

    def to_json(self) -> dict:
        return asdict(self)


def _f1(pred: set[str], true: set[str]) -> tuple[float, float, float]:
    if not pred and not true:
        return 1.0, 1.0, 1.0
    tp = len(pred & true)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(true) if true else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def probe(llm, smiles: str, true_labels: set[str], vocabulary: set[str],
          in_dataset: bool = True) -> ProbeResult:
    resp = llm(RECALL_SYSTEM, [{"role": "user", "content": smiles}], None)
    body = re.sub(r"^```(?:json)?|```$", "", resp.text.strip(), flags=re.M).strip()
    try:
        data = json.loads(body)
    except Exception:
        data = {"descriptors": [], "compound_name": None, "confidence": "low"}
    pred = {str(d).lower().strip() for d in data.get("descriptors", [])}
    pred &= vocabulary  # score only on the shared vocabulary
    p, r, f = _f1(pred, true_labels)
    return ProbeResult(
        smiles=smiles, in_dataset=in_dataset, true_labels=sorted(true_labels),
        predicted=sorted(pred), named=data.get("compound_name"),
        confidence=str(data.get("confidence", "low")), precision=p, recall=r, f1=f,
    )


def run(llm, data: OdorData, n: int = 100, seed: int = 0,
        include_controls: bool = True) -> list[ProbeResult]:
    """Probe n test-scaffold molecules, plus matched off-dataset controls."""
    rng = np.random.default_rng(seed)
    vocab = set(data.descriptors)
    idx = np.where(data.split == "test")[0]
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
    known = set(data.smiles)
    out: list[ProbeResult] = []
    for i in pick:
        smi = data.smiles[i]
        true = {d for d, v in zip(data.descriptors, data.Y[i]) if v}
        out.append(probe(llm, smi, true, vocab, in_dataset=True))
        if include_controls:
            groups = chem.groups_present(smi)
            if groups:
                alt = chem.ablate(smi, str(rng.choice(groups)))
                if alt and alt not in known:
                    out.append(probe(llm, alt, set(), vocab, in_dataset=False))
    return out


def summarise(results: list[ProbeResult]) -> dict:
    ins = [r for r in results if r.in_dataset]
    outs = [r for r in results if not r.in_dataset]
    named = [r for r in ins if r.named and str(r.named).lower() != "null"]
    return {
        "n_in_dataset": len(ins),
        "n_off_dataset_controls": len(outs),
        "mean_f1_in_dataset": float(np.mean([r.f1 for r in ins])) if ins else None,
        "mean_recall_in_dataset": float(np.mean([r.recall for r in ins])) if ins else None,
        "named_rate": len(named) / len(ins) if ins else None,
        "mean_f1_when_named": float(np.mean([r.f1 for r in named])) if named else None,
        "mean_f1_when_unnamed": float(
            np.mean([r.f1 for r in ins if r not in named])) if ins else None,
        "high_recall_fraction": float(np.mean([r.recall > 0.5 for r in ins])) if ins else None,
    }
