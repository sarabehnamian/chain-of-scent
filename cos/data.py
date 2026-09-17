"""Dataset loading, canonicalisation and scaffold splitting.

Source: the curated GoodScents + Leffingwell merge distributed with openpom
(4,983 molecules x 138 odour descriptors, multi-label binary).
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

DATA_URL = (
    "https://raw.githubusercontent.com/ARY2260/openpom/main/openpom/data/"
    "curated_datasets/curated_GS_LF_merged_4983.csv"
)
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "curated_GS_LF_merged_4983.csv",
)


@dataclass
class OdorData:
    """Canonicalised molecules, label matrix and scaffold-based split."""

    smiles: list[str]
    Y: np.ndarray  # (n_mol, n_desc) binary
    descriptors: list[str]
    scaffolds: list[str]
    split: np.ndarray  # array of "train" / "valid" / "test"

    def subset(self, which: str) -> tuple[list[str], np.ndarray]:
        idx = np.where(self.split == which)[0]
        return [self.smiles[i] for i in idx], self.Y[idx]

    def labels_for(self, smi: str) -> list[str] | None:
        try:
            i = self.smiles.index(smi)
        except ValueError:
            return None
        return [d for d, v in zip(self.descriptors, self.Y[i]) if v]


def download(path: str = DEFAULT_PATH) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def canonical(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def murcko(smi: str) -> str:
    """Bemis-Murcko scaffold; acyclic molecules get their own bucket."""
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        s = ""
    return s if s else f"ACYCLIC::{smi}"


def scaffold_split(
    scaffolds: list[str], frac=(0.8, 0.1, 0.1), seed: int = 0
) -> np.ndarray:
    """Deterministic scaffold split: whole scaffold groups go to one side.

    Big groups are placed first (standard practice) so the test set stays
    structurally distinct from train. This is the split that matters for any
    claim about generalisation, and for the contamination analysis.
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(scaffolds):
        groups.setdefault(s, []).append(i)
    order = sorted(groups.values(), key=lambda g: (-len(g), scaffolds[g[0]]))
    n = len(scaffolds)
    caps = [frac[0] * n, (frac[0] + frac[1]) * n]
    split = np.empty(n, dtype=object)
    n_tr = n_va = 0
    for grp in order:
        if n_tr + len(grp) <= caps[0]:
            tag, n_tr = "train", n_tr + len(grp)
        elif n_tr + n_va + len(grp) <= caps[1]:
            tag, n_va = "valid", n_va + len(grp)
        else:
            tag = "test"
        for i in grp:
            split[i] = tag
    # tiny shuffle guard so ties are not alphabetical artefacts
    rng.random()
    return split


def load(path: str = DEFAULT_PATH, min_positives: int = 30, seed: int = 0) -> OdorData:
    """Load, canonicalise, drop rare descriptors, and scaffold-split."""
    df = pd.read_csv(download(path))
    desc_cols = [c for c in df.columns if c not in ("nonStereoSMILES", "descriptors")]

    canon, keep = [], []
    for i, smi in enumerate(df["nonStereoSMILES"]):
        c = canonical(smi)
        if c is not None:
            canon.append(c)
            keep.append(i)
    df = df.iloc[keep].reset_index(drop=True).copy()
    df["canonical"] = canon

    # deduplicate on canonical SMILES, OR-ing the label rows
    df = df.groupby("canonical", as_index=False)[desc_cols].max()

    counts = df[desc_cols].sum()
    desc = [c for c in desc_cols if counts[c] >= min_positives]

    smiles = df["canonical"].tolist()
    Y = df[desc].to_numpy(dtype=np.int8)
    scaf = [murcko(s) for s in smiles]
    split = scaffold_split(scaf, seed=seed)
    return OdorData(smiles=smiles, Y=Y, descriptors=desc, scaffolds=scaf, split=split)


if __name__ == "__main__":
    d = load()
    print(f"{len(d.smiles)} molecules, {len(d.descriptors)} descriptors")
    for s in ("train", "valid", "test"):
        print(f"  {s}: {(d.split == s).sum()}")
    print("example:", d.smiles[0], d.labels_for(d.smiles[0]))
