"""Chemistry utilities: fingerprints, novelty, synthesisability, and the
functional-group catalogue used for causal interventions.

The catalogue is the heart of the faithfulness test. Each entry pairs a SMARTS
pattern that *detects* a group with a reaction SMARTS that *neutralises* it
(a minimal edit that removes the feature while perturbing the rest of the
molecule as little as possible). Every entry is self-tested at import time
against a probe molecule; entries whose reaction does not fire are dropped
rather than silently producing garbage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator as rfg

RDLogger.DisableLog("rdApp.*")

_MORGAN = rfg.GetMorganGenerator(radius=2, fpSize=2048)


# --------------------------------------------------------------------------
# basic molecule handling
# --------------------------------------------------------------------------
def mol(smi: str):
    return Chem.MolFromSmiles(smi)


def canonical(smi: str) -> str | None:
    m = mol(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def is_valid(smi: str) -> bool:
    return mol(smi) is not None


def fingerprint(smi: str):
    m = mol(smi)
    if m is None:
        return None
    return _MORGAN.GetFingerprint(m)


def fp_matrix(smiles: list[str]) -> np.ndarray:
    X = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, s in enumerate(smiles):
        fp = fingerprint(s)
        if fp is not None:
            arr = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X[i] = arr
    return X


def tanimoto(a: str, b: str) -> float:
    fa, fb = fingerprint(a), fingerprint(b)
    if fa is None or fb is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fa, fb)


class NoveltyIndex:
    """Nearest-neighbour Tanimoto against a reference set (usually the oracle's
    training molecules). Used both as a novelty measure and as a cheap
    out-of-distribution flag for proposals."""

    def __init__(self, reference: list[str]):
        self.reference = reference
        self.fps = [fingerprint(s) for s in reference]
        self.fps = [f for f in self.fps if f is not None]

    def nearest(self, smi: str) -> tuple[float, str | None]:
        fp = fingerprint(smi)
        if fp is None or not self.fps:
            return 0.0, None
        sims = DataStructs.BulkTanimotoSimilarity(fp, self.fps)
        j = int(np.argmax(sims))
        return float(sims[j]), self.reference[j]


# --------------------------------------------------------------------------
# synthesisability / drug-like sanity
# --------------------------------------------------------------------------
def sa_score(smi: str) -> float | None:
    """Ertl-Schuffenhauer synthetic accessibility (1 easy .. 10 hard).

    Uses RDKit's contrib sascorer when available; otherwise falls back to a
    crude ring-complexity + size heuristic so the pipeline never hard-fails.
    """
    m = mol(smi)
    if m is None:
        return None
    try:
        from rdkit.Chem import RDConfig
        import os
        import sys

        p = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if p not in sys.path:
            sys.path.append(p)
        import sascorer  # type: ignore

        return float(sascorer.calculateScore(m))
    except Exception:
        nring = rdMolDescriptors.CalcNumRings(m)
        nstereo = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
        return float(
            1.0 + 0.05 * m.GetNumHeavyAtoms() + 0.4 * nring + 0.3 * nstereo
        )


def odorant_plausible(smi: str) -> tuple[bool, str]:
    """Very loose physical filter for 'could this even be a smell'.

    Volatility roughly bounds odorants: MW under ~300, no charges, no metals.
    This is not a rigour claim, it is a guard against the agent proposing
    peptides to game the oracle.
    """
    m = mol(smi)
    if m is None:
        return False, "invalid"
    mw = Descriptors.MolWt(m)
    if mw > 300:
        return False, f"MW {mw:.0f} > 300"
    if any(a.GetFormalCharge() != 0 for a in m.GetAtoms()):
        return False, "charged"
    allowed = {"C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "Si", "P"}
    bad = {a.GetSymbol() for a in m.GetAtoms()} - allowed
    if bad:
        return False, f"element {sorted(bad)}"
    if Crippen.MolLogP(m) > 6:
        return False, "logP > 6"
    return True, "ok"


# --------------------------------------------------------------------------
# functional-group catalogue + interventions
# --------------------------------------------------------------------------
@dataclass
class Group:
    name: str
    smarts: str          # detection
    reaction: str        # neutralising edit, reaction SMARTS
    probe: str           # molecule that must contain the group
    aliases: tuple[str, ...] = ()

    def __post_init__(self):
        self.patt = Chem.MolFromSmarts(self.smarts)
        self.rxn = AllChem.ReactionFromSmarts(self.reaction)


_RAW_GROUPS = [
    Group("aldehyde", "[CX3H1](=O)[#6]", "[#6:1][CX3H1]=O>>[#6:1][CH3]",
          "CCCCCC=O", ("formyl", "cho")),
    Group("ketone", "[#6][CX3](=O)[#6]", "[#6:1][CX3:2](=O)[#6:3]>>[#6:1][CH1:2]([OH])[#6:3]",
          "CCC(=O)CC", ("carbonyl",)),
    Group("primary_alcohol", "[CX4;H2][OX2H]", "[CX4;H2:1][OX2H]>>[CH3:1]",
          "CCCCO", ("hydroxyl", "alcohol", "oh")),
    Group("carboxylic_acid", "[CX3](=O)[OX2H1]", "[#6:1][CX3](=O)[OX2H1]>>[#6:1][CH3]",
          "CCCC(=O)O", ("acid", "cooh")),
    Group("ester", "[#6][CX3](=O)[OX2][#6]",
          "[#6:1][CX3](=O)[OX2][#6:2]>>[#6:1][CH2][O][#6:2]",
          "CCCC(=O)OCC", ("acetate", "ester_linkage")),
    Group("thiol", "[#6][SX2H]", "[#6:1][SX2H]>>[#6:1][CH3]",
          "CCCS", ("mercaptan", "sh", "sulfhydryl")),
    Group("sulfide", "[#6][SX2][#6]", "[#6:1][SX2][#6:2]>>[#6:1][CH2][#6:2]",
          "CCSCC", ("thioether",)),
    Group("primary_amine", "[CX4][NX3;H2]", "[CX4:1][NX3;H2]>>[CX4:1][CH3]",
          "CCCCN", ("amine", "nh2")),
    Group("nitrile", "[#6][CX2]#[NX1]", "[#6:1][CX2]#[NX1]>>[#6:1][CH3]",
          "CCCC#N", ("cyano", "cn")),
    Group("alkene", "[CX3;!R]=[CX3;!R]", "[CX3:1]=[CX3:2]>>[CX4:1][CX4:2]",
          "CC=CCCO", ("double_bond", "unsaturation", "cis_double_bond")),
    Group("ether", "[#6;!$(C=O)][OX2][#6;!$(C=O)]",
          "[#6:1][OX2][#6:2]>>[#6:1][CH2][#6:2]", "CCOCC", ("alkoxy",)),
    Group("methoxy_aromatic", "[c][OX2][CH3]", "[c:1][OX2][CH3]>>[c:1][CH3]",
          "COc1ccccc1", ("anisole", "methoxy")),
    Group("phenol", "[c][OX2H]", "[c:1][OX2H]>>[c:1][CH3]",
          "Oc1ccccc1", ("aromatic_hydroxyl",)),
    Group("lactone", "[#6;R][CX3;R](=O)[OX2;R]",
          "[#6:1][CX3;R](=O)[OX2;R][#6:2]>>[#6:1][CH2][O][#6:2]",
          "O=C1CCCCO1", ("cyclic_ester",)),
]


def _self_test(groups: list[Group]) -> list[Group]:
    ok = []
    for g in groups:
        m = mol(g.probe)
        if m is None or not m.HasSubstructMatch(g.patt):
            continue
        prods = g.rxn.RunReactants((m,))
        good = False
        for p in prods:
            try:
                Chem.SanitizeMol(p[0])
                if Chem.MolToSmiles(p[0]) != Chem.MolToSmiles(m):
                    good = True
                    break
            except Exception:
                continue
        if good:
            ok.append(g)
    return ok


GROUPS: list[Group] = _self_test(_RAW_GROUPS)
GROUP_BY_NAME: dict[str, Group] = {}
for _g in GROUPS:
    GROUP_BY_NAME[_g.name] = _g
    for _a in _g.aliases:
        GROUP_BY_NAME.setdefault(_a, _g)


def groups_present(smi: str) -> list[str]:
    m = mol(smi)
    if m is None:
        return []
    return [g.name for g in GROUPS if m.HasSubstructMatch(g.patt)]


def ablate(smi: str, group_name: str) -> str | None:
    """Apply the neutralising edit for `group_name`. Returns canonical SMILES
    of the edited molecule, or None if the group is absent or the edit fails."""
    g = GROUP_BY_NAME.get(group_name.lower().strip())
    if g is None:
        return None
    m = mol(smi)
    if m is None or not m.HasSubstructMatch(g.patt):
        return None
    for prod in g.rxn.RunReactants((m,)):
        cand = prod[0]
        try:
            Chem.SanitizeMol(cand)
        except Exception:
            continue
        out = Chem.MolToSmiles(cand)
        if out and out != Chem.MolToSmiles(m) and mol(out) is not None:
            return out
    return None


def edit_size(a: str, b: str) -> int:
    """Heavy-atom distance between two molecules, used to match an intervention
    against its placebo control."""
    ma, mb = mol(a), mol(b)
    if ma is None or mb is None:
        return 99
    return abs(ma.GetNumHeavyAtoms() - mb.GetNumHeavyAtoms())


# --------------------------------------------------------------------------
# random mutation baseline
# --------------------------------------------------------------------------
_MUTATIONS = [
    ("[CH3:1][CH2:2]>>[CH3:1][CH2][CH2:2]", "extend chain"),
    ("[CH2:1][CH2:2][CH3]>>[CH2:1][CH3:2]", "shorten chain"),
    ("[CH3:1][CH2:2][OH]>>[CH3:1][CH:2]=O", "alcohol to aldehyde"),
    ("[CH:1]=[CH:2]>>[CH2:1][CH2:2]", "saturate"),
    ("[CH2:1][CH2:2]>>[CH:1]=[CH:2]", "desaturate"),
    ("[#6:1][OX2H]>>[#6:1]OC(C)=O", "acetylate"),
    ("[#6:1][CH3]>>[#6:1]C(C)C", "branch"),
]
_MUT_RXN = [(AllChem.ReactionFromSmarts(s), lab) for s, lab in _MUTATIONS]


def random_mutate(smi: str, rng: np.random.Generator) -> tuple[str | None, str]:
    m = mol(smi)
    if m is None:
        return None, "invalid parent"
    order = rng.permutation(len(_MUT_RXN))
    for i in order:
        rxn, label = _MUT_RXN[i]
        prods = rxn.RunReactants((m,))
        cands = []
        for p in prods:
            try:
                Chem.SanitizeMol(p[0])
            except Exception:
                continue
            s = Chem.MolToSmiles(p[0])
            if s and s != Chem.MolToSmiles(m) and mol(s) is not None:
                cands.append(s)
        if cands:
            return str(rng.choice(cands)), label
    return None, "no mutation applied"


if __name__ == "__main__":
    print(f"{len(GROUPS)}/{len(_RAW_GROUPS)} groups passed self-test:")
    for g in GROUPS:
        print(f"  {g.name:20s} {g.probe:12s} -> {ablate(g.probe, g.name)}")
    print()
    hexenal = "CC/C=C\\CC=O"
    print("cis-3-hexenal groups:", groups_present(hexenal))
    print("  minus aldehyde:", ablate(hexenal, "aldehyde"))
    print("  minus alkene:  ", ablate(hexenal, "alkene"))
    print("  SA:", round(sa_score(hexenal), 2), "plausible:", odorant_plausible(hexenal))
    rng = np.random.default_rng(0)
    print("  mutation:", random_mutate(hexenal, rng))
