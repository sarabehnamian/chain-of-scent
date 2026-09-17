"""Stratified targets: separating retrieval from reasoning.

The contamination probe shows the agent can name 75% of test molecules from
SMILES and recalls their descriptors far above the frequency floor. So a high
target-match score is ambiguous: the agent may have recognised a compound that
fits the profile, or it may have reasoned about structure. This module builds
target strata that separate the two, using measured recognisability rather
than an assumption about what is "well known".

Three strata:

  recognised    reference molecule was named AND recalled well in the probe.
                Retrieval is available. Expect high scores.
  unrecognised  reference molecule was not named and recalled poorly.
                A real molecule realises the profile, but the agent shows no
                sign of knowing it. Retrieval is unavailable; structural
                inference is not.
  unrealisable  descriptor combinations that NO molecule in the curated set
                carries, assembled from descriptors that individually are
                common. Nothing can be retrieved because nothing exists to
                retrieve. This is the floor for retrieval and the ceiling for
                compositional reasoning.

The prediction, if the agent is retrieving rather than reasoning: performance
tracks the strata (recognised > unrecognised > unrealisable) and collapses
toward the random-mutation baseline in the last two. If the agent is reasoning
about structure, the strata should matter much less.

Note the confound and state it in the paper: unrecognised molecules are also
rarer, and rarer chemistry may simply be harder for reasons unrelated to
recall. The unrealisable stratum is not subject to that confound, which is why
it carries the argument.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .data import OdorData
from .targets import Target


@dataclass
class Strata:
    recognised: list[Target]
    unrecognised: list[Target]
    unrealisable: list[Target]

    def all(self) -> list[Target]:
        return self.recognised + self.unrecognised + self.unrealisable

    def label_of(self, target_id: str) -> str:
        for name in ("recognised", "unrecognised", "unrealisable"):
            if any(t.id == target_id for t in getattr(self, name)):
                return name
        return "unknown"


def load_probe(path: str) -> dict[str, dict]:
    """Read results/contamination.jsonl into {smiles: record}."""
    out: dict[str, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("in_dataset"):
                out[r["smiles"]] = r
    return out


def _named(rec: dict) -> bool:
    n = rec.get("named")
    return bool(n) and str(n).lower() not in ("null", "none", "unknown", "")


def build(data: OdorData, probe_path: str, n_per_stratum: int = 15,
          size: tuple[int, int] = (2, 4), seed: int = 0,
          recall_hi: float = 0.5, recall_lo: float = 0.2) -> Strata:
    rng = np.random.default_rng(seed)
    probe = load_probe(probe_path)
    if not probe:
        raise FileNotFoundError(
            "results/contamination.jsonl not found. Run scripts/04_contamination.py "
            "first — the strata are defined by measured recognisability.")

    test_idx = np.where(data.split == "test")[0]
    d2i = {d: i for i, d in enumerate(data.descriptors)}
    tr = np.where(data.split == "train")[0]

    rec_pool, unrec_pool = [], []
    for i in test_idx:
        smi = data.smiles[i]
        r = probe.get(smi)
        if r is None:
            continue
        labs = [d for d, v in zip(data.descriptors, data.Y[i]) if v]
        if not (size[0] <= len(labs) <= size[1]):
            continue
        cols = [d2i[l] for l in labs]
        support = int(data.Y[np.ix_(tr, cols)].all(axis=1).sum())
        item = (i, sorted(labs), support)
        if _named(r) and r["recall"] >= recall_hi:
            rec_pool.append(item)
        elif (not _named(r)) and r["recall"] <= recall_lo:
            unrec_pool.append(item)

    def take(pool, prefix):
        rng.shuffle(pool)
        seen, out = set(), []
        for i, labs, support in pool:
            if data.scaffolds[i] in seen:
                continue
            seen.add(data.scaffolds[i])
            out.append(Target(
                id=f"{prefix}{len(out):03d}", descriptors=labs,
                reference_smiles=data.smiles[i],
                reference_scaffold=data.scaffolds[i],
                n_train_molecules_with_profile=support))
            if len(out) >= n_per_stratum:
                break
        return out

    recognised = take(rec_pool, "R")
    unrecognised = take(unrec_pool, "U")

    # unrealisable: common descriptors that never co-occur in the whole set
    # "odorless" and other meta terms pair vacuously with anything, which makes
    # an incoherent target rather than an unrealisable one
    EXCLUDE = {"odorless", "odourless", "tasteless", "weak", "mild"}
    counts = Counter()
    for j, d in enumerate(data.descriptors):
        if d not in EXCLUDE:
            counts[d] = int(data.Y[:, j].sum())
    common = [d for d, c in counts.most_common(40)]
    existing = {frozenset(d for d, v in zip(data.descriptors, row) if v)
                for row in data.Y}
    pairwise = data.Y.T @ data.Y  # co-occurrence counts

    unrealisable: list[Target] = []
    tries = 0
    while len(unrealisable) < n_per_stratum and tries < 20000:
        tries += 1
        k = int(rng.integers(size[0], size[1] + 1))
        pick = sorted(rng.choice(common, size=k, replace=False).tolist())
        cols = [d2i[p] for p in pick]
        # every pair must be genuinely incompatible in the data
        if any(pairwise[a, b] > 0 for a in cols for b in cols if a != b):
            continue
        if any(set(pick) <= s for s in existing):
            continue
        if pick in [t.descriptors for t in unrealisable]:
            continue
        unrealisable.append(Target(
            id=f"X{len(unrealisable):03d}", descriptors=pick,
            reference_smiles="", reference_scaffold="",
            n_train_molecules_with_profile=0))

    return Strata(recognised, unrecognised, unrealisable)


def save(strata: Strata, path: str) -> None:
    payload = {name: [t.to_json() for t in getattr(strata, name)]
               for name in ("recognised", "unrecognised", "unrealisable")}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def load(path: str) -> Strata:
    with open(path) as fh:
        p = json.load(fh)
    return Strata(**{k: [Target(**t) for t in v] for k, v in p.items()})


def compare(logs: list[dict], strata: Strata) -> dict:
    """Arm performance within each stratum, plus the recognised-minus-other gap."""
    from .analysis import bootstrap_ci

    rows: dict[tuple[str, str], list[float]] = {}
    for log in logs:
        lab = strata.label_of(log["target_id"])
        if lab == "unknown":
            continue
        rows.setdefault((lab, log["arm"]), []).append(log.get("best_score", 0.0))

    out: dict = {"by_stratum": {}}
    for (lab, arm), vals in sorted(rows.items()):
        m, lo, hi = bootstrap_ci(vals)
        out["by_stratum"].setdefault(lab, {})[arm] = {
            "n": len(vals), "mean": m, "ci": [lo, hi]}

    st = out["by_stratum"]
    for lab in ("unrecognised", "unrealisable"):
        if "recognised" in st and lab in st and "full" in st.get(lab, {}):
            a = st["recognised"].get("full", {}).get("mean")
            b = st[lab]["full"]["mean"]
            floor = st[lab].get("random_mutation", {}).get("mean")
            out[f"recognised_minus_{lab}"] = {
                "gap": None if a is None or b is None else a - b,
                "stratum_full": b, "stratum_floor": floor,
                "full_above_floor": None if (b is None or floor is None) else b - floor,
            }
    out["reading"] = (
        "A large recognised-minus-unrealisable gap, with full collapsing "
        "toward random_mutation in the unrealisable stratum, is evidence the "
        "agent succeeds by recognition rather than compositional reasoning."
    )
    return out
