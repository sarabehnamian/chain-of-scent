"""Target odour profiles for the design task.

Targets are drawn from held-out (test-scaffold) molecules, so the reference
molecule that realises the profile is one the oracle never saw. We keep the
reference SMILES only for analysis; it is never shown to the agent.

Profiles are filtered to be neither trivial (a single very common descriptor)
nor impossible (rare descriptor combinations with no support anywhere).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

from .data import OdorData


@dataclass
class Target:
    id: str
    descriptors: list[str]
    reference_smiles: str      # analysis only, never shown to the agent
    reference_scaffold: str
    n_train_molecules_with_profile: int

    def to_json(self) -> dict:
        return asdict(self)


def build(data: OdorData, n: int = 30, size: tuple[int, int] = (2, 4),
          seed: int = 0, require_train_support: int = 3,
          focus: list[str] | None = None) -> list[Target]:
    """Sample `n` target profiles from test-scaffold molecules.

    focus: optionally restrict to profiles containing one of these descriptors
    (e.g. ["green"]) for a narrow first-pass study.
    """
    rng = np.random.default_rng(seed)
    idx = np.where(data.split == "test")[0]
    tr = np.where(data.split == "train")[0]
    d2i = {d: i for i, d in enumerate(data.descriptors)}

    cands = []
    for i in idx:
        labs = [d for d, v in zip(data.descriptors, data.Y[i]) if v]
        if not (size[0] <= len(labs) <= size[1]):
            continue
        if focus and not any(f in labs for f in focus):
            continue
        cols = [d2i[l] for l in labs]
        support = int((data.Y[np.ix_(tr, cols)].all(axis=1)).sum())
        if support < require_train_support:
            continue
        cands.append((i, labs, support))

    if not cands:
        raise ValueError("no candidate profiles; relax size/support/focus")

    rng.shuffle(cands)
    seen_scaffold: set[str] = set()
    out: list[Target] = []
    for i, labs, support in cands:
        if data.scaffolds[i] in seen_scaffold:
            continue  # one target per scaffold, keeps the sample independent
        seen_scaffold.add(data.scaffolds[i])
        out.append(Target(
            id=f"T{len(out):03d}",
            descriptors=sorted(labs),
            reference_smiles=data.smiles[i],
            reference_scaffold=data.scaffolds[i],
            n_train_molecules_with_profile=support,
        ))
        if len(out) >= n:
            break
    return out


def save(targets: list[Target], path: str) -> None:
    with open(path, "w") as fh:
        json.dump([t.to_json() for t in targets], fh, indent=2)


def load(path: str) -> list[Target]:
    with open(path) as fh:
        return [Target(**t) for t in json.load(fh)]
