"""The odour oracle: the tool the agent calls, and the measuring instrument
for the interventions.

Two backends share one interface:

  FingerprintOracle  bagged one-vs-rest logistic regression on Morgan counts,
                     trained here on the train scaffolds only. Fast, CPU-only,
                     fully reproducible, and it exposes ensemble disagreement
                     which we use as an out-of-distribution flag.
  OpenPOMOracle      wraps the pretrained openpom MPNN ensemble if installed.

Two properties matter more than raw accuracy:

1. It must be trained on train scaffolds only, so that targets drawn from
   test scaffolds are genuinely out of sample.
2. It must report its own uncertainty. An agent given 10 rounds against a
   learned proxy will find molecules that score well because the proxy is
   wrong there. Ensemble std + nearest-neighbour Tanimoto let us detect that
   instead of reporting it as success.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field, asdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import chem
from .data import OdorData


@dataclass
class Prediction:
    smiles: str
    valid: bool
    probs: dict[str, float] = field(default_factory=dict)
    ensemble_std: dict[str, float] = field(default_factory=dict)
    nn_similarity: float = 0.0
    nn_neighbour: str | None = None
    sa_score: float | None = None
    plausible: bool = True
    note: str = ""

    def top(self, k: int = 8) -> list[tuple[str, float]]:
        return sorted(self.probs.items(), key=lambda kv: -kv[1])[:k]

    def as_tool_result(self, target: list[str] | None = None, k: int = 8) -> dict:
        """Compact JSON the agent sees. Deliberately does not reveal the
        ground-truth label set, only model scores."""
        out: dict = {"smiles": self.smiles, "valid": self.valid}
        if not self.valid:
            out["error"] = self.note or "could not parse SMILES"
            return out
        out["top_predicted_odors"] = [
            {"descriptor": d, "score": round(p, 3)} for d, p in self.top(k)
        ]
        if target:
            out["target_scores"] = [
                {"descriptor": t, "score": round(self.probs.get(t, 0.0), 3)}
                for t in target
            ]
            out["target_match"] = round(self.target_score(target), 3)
        out["confidence_note"] = (
            "low confidence: molecule is unlike the training set"
            if self.nn_similarity < 0.4
            else "ok"
        )
        return out

    def target_score(self, target: list[str]) -> float:
        """Mean predicted probability over the target descriptors. This is the
        agent's objective; analysis also reports rank-based measures that are
        harder to game."""
        if not target:
            return 0.0
        return float(np.mean([self.probs.get(t, 0.0) for t in target]))


class FingerprintOracle:
    name = "fingerprint_ensemble"

    def __init__(self, descriptors: list[str], n_members: int = 5, seed: int = 0):
        self.descriptors = descriptors
        self.n_members = n_members
        self.seed = seed
        self.models: list[list[LogisticRegression | float]] = []
        self.novelty: chem.NoveltyIndex | None = None
        self._cache: dict[str, Prediction] = {}

    # -- training ----------------------------------------------------------
    def fit(self, smiles: list[str], Y: np.ndarray) -> "FingerprintOracle":
        X = chem.fp_matrix(smiles)
        rng = np.random.default_rng(self.seed)
        n = len(smiles)
        for m in range(self.n_members):
            idx = rng.integers(0, n, n)  # bootstrap
            Xb, Yb = X[idx], Y[idx]
            member: list[LogisticRegression | float] = []
            for j in range(Y.shape[1]):
                y = Yb[:, j]
                if y.sum() < 5 or y.sum() == len(y):
                    member.append(float(Y[:, j].mean()))
                    continue
                clf = LogisticRegression(
                    C=1.0, max_iter=300, solver="liblinear", random_state=self.seed + m
                )
                clf.fit(Xb, y)
                member.append(clf)
            self.models.append(member)
        self.novelty = chem.NoveltyIndex(smiles)
        return self

    # -- prediction --------------------------------------------------------
    def predict(self, smiles: str) -> Prediction:
        key = smiles.strip()
        if key in self._cache:
            return self._cache[key]
        canon = chem.canonical(key)
        if canon is None:
            p = Prediction(smiles=key, valid=False, note="invalid SMILES")
            self._cache[key] = p
            return p
        x = chem.fp_matrix([canon])
        per_member = np.zeros((self.n_members, len(self.descriptors)))
        for m, member in enumerate(self.models):
            for j, clf in enumerate(member):
                if isinstance(clf, float):
                    per_member[m, j] = clf
                else:
                    per_member[m, j] = clf.predict_proba(x)[0, 1]
        mean = per_member.mean(axis=0)
        std = per_member.std(axis=0)
        sim, nb = self.novelty.nearest(canon) if self.novelty else (0.0, None)
        ok, why = chem.odorant_plausible(canon)
        p = Prediction(
            smiles=canon,
            valid=True,
            probs={d: float(v) for d, v in zip(self.descriptors, mean)},
            ensemble_std={d: float(v) for d, v in zip(self.descriptors, std)},
            nn_similarity=sim,
            nn_neighbour=nb,
            sa_score=chem.sa_score(canon),
            plausible=ok,
            note=why,
        )
        self._cache[key] = p
        return p

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "FingerprintOracle":
        with open(path, "rb") as fh:
            return pickle.load(fh)


class OpenPOMOracle:
    """Optional wrapper around the pretrained openpom MPNN ensemble.

    Install per https://github.com/ARY2260/openpom . Kept behind a lazy import
    so nothing else in the project depends on deepchem/dgl being present.
    Note the pretrained weights were trained on the *whole* curated set, so if
    you use this backend your scaffold split no longer protects the oracle,
    only the LLM analysis. Say so in the paper.
    """

    name = "openpom"

    def __init__(self, model_dir: str, descriptors: list[str]):
        from openpom.feat.graph_featurizer import GraphFeaturizer  # noqa
        import deepchem as dc  # noqa

        self.model_dir = model_dir
        self.descriptors = descriptors
        raise NotImplementedError(
            "Fill in per openpom's predict_odors.py once you have the weights; "
            "return the same Prediction object as FingerprintOracle.predict."
        )


def evaluate(oracle: FingerprintOracle, smiles: list[str], Y: np.ndarray) -> dict:
    """Held-out AUROC (macro, over descriptors with both classes present)."""
    from sklearn.metrics import roc_auc_score

    P = np.array([[oracle.predict(s).probs.get(d, 0.0) for d in oracle.descriptors]
                  for s in smiles])
    aucs = []
    for j in range(Y.shape[1]):
        if 0 < Y[:, j].sum() < len(Y):
            aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    return {
        "n_molecules": len(smiles),
        "n_descriptors_scored": len(aucs),
        "macro_auroc": float(np.mean(aucs)),
        "median_auroc": float(np.median(aucs)),
    }


if __name__ == "__main__":
    from .data import load

    d = load()
    tr_s, tr_Y = d.subset("train")
    te_s, te_Y = d.subset("test")
    orc = FingerprintOracle(d.descriptors).fit(tr_s, tr_Y)
    print(json.dumps(evaluate(orc, te_s, te_Y), indent=2))
    for s in ("CC/C=C\\CC=O", "CCCCCC=O", "CCS"):
        p = orc.predict(s)
        print(s, "->", [(d_, round(v, 2)) for d_, v in p.top(5)],
              f"nn={p.nn_similarity:.2f}")
