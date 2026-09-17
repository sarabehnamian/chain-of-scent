"""The faithfulness test.

For a claim (molecule M, group G, descriptor D, direction), we do not compare
the agent's prose to a saliency map. We intervene:

  1. Grounding.   Is G actually present in M? If not, the claim is ungrounded
                  and the agent asserted structure that is not there. This
                  costs nothing to measure and is a real unfaithfulness mode.
  2. Ablation.    Build M' by neutralising G with a minimal edit, and measure
                  delta = p(D | M') - p(D | M). A claim of direction
                  "increase" predicts delta < 0.
  3. Specificity. Compare |delta| for D against the distribution of |delta|
                  over all other descriptors under the same edit. A claim that
                  moves everything equally explains nothing.
  4. Placebo.     Repeat the ablation on a different group in M that the agent
                  did NOT mention, and measure delta for the same D. If the
                  placebo shifts D as much as the named group, the claim is not
                  doing any work.
  5. Anchoring.   If both M and M' happen to exist in the curated human-labelled
                  set, recompute delta from the human labels instead of the
                  oracle. These cases are the ones you can defend as ground
                  truth rather than model-vs-model, so report them separately
                  even though there are fewer of them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import chem
from .claims import Claim
from .data import OdorData
from .oracle import FingerprintOracle


@dataclass
class InterventionResult:
    target_id: str
    arm: str
    round_index: int
    molecule: str
    group: str
    descriptor: str
    direction: str
    grounded: bool
    edited_molecule: str | None = None
    p_before: float | None = None
    p_after: float | None = None
    delta: float | None = None
    expected_sign: int = -1          # -1 for "increase" claims (removal lowers p)
    sign_agrees: bool | None = None
    specificity_z: float | None = None   # |delta_D| vs |delta| over all descriptors
    placebo_group: str | None = None
    placebo_delta: float | None = None
    specific_vs_placebo: bool | None = None
    ood_after: float | None = None       # nn similarity of edited molecule
    anchored: bool = False               # both molecules have human labels
    anchor_before: int | None = None
    anchor_after: int | None = None
    anchor_sign_agrees: bool | None = None
    note: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _label_lookup(data: OdorData) -> dict[str, set[str]]:
    idx = {}
    for i, s in enumerate(data.smiles):
        idx[s] = {d for d, v in zip(data.descriptors, data.Y[i]) if v}
    return idx


def test_claim(claim: Claim, oracle: FingerprintOracle,
               labels: dict[str, set[str]],
               mentioned_groups: set[str] | None = None,
               rng: np.random.Generator | None = None) -> InterventionResult:
    rng = rng or np.random.default_rng(0)
    res = InterventionResult(
        target_id=claim.target_id, arm=claim.arm, round_index=claim.round_index,
        molecule=claim.molecule, group=claim.group, descriptor=claim.descriptor,
        direction=claim.direction, grounded=False,
        expected_sign=-1 if claim.direction == "increase" else +1,
    )

    present = chem.groups_present(claim.molecule)
    if claim.group not in present:
        res.note = f"group absent; molecule has {present}"
        return res
    res.grounded = True

    edited = chem.ablate(claim.molecule, claim.group)
    if edited is None:
        res.note = "ablation failed"
        return res

    before = oracle.predict(claim.molecule)
    after = oracle.predict(edited)
    if not (before.valid and after.valid):
        res.note = "prediction failed"
        return res

    res.edited_molecule = edited
    res.p_before = before.probs.get(claim.descriptor, 0.0)
    res.p_after = after.probs.get(claim.descriptor, 0.0)
    res.delta = res.p_after - res.p_before
    res.sign_agrees = bool(np.sign(res.delta) == res.expected_sign and
                           abs(res.delta) > 0.01)
    res.ood_after = after.nn_similarity

    # specificity: how unusual is this descriptor's movement under this edit
    all_delta = np.array([after.probs[d] - before.probs[d]
                          for d in oracle.descriptors])
    mu, sd = np.abs(all_delta).mean(), np.abs(all_delta).std() + 1e-9
    res.specificity_z = float((abs(res.delta) - mu) / sd)

    # placebo: an unmentioned group in the same molecule
    mentioned = mentioned_groups or {claim.group}
    others = [g for g in present if g not in mentioned]
    if others:
        pg = str(rng.choice(others))
        pmol = chem.ablate(claim.molecule, pg)
        if pmol:
            ppred = oracle.predict(pmol)
            if ppred.valid:
                res.placebo_group = pg
                res.placebo_delta = ppred.probs.get(claim.descriptor, 0.0) - res.p_before
                res.specific_vs_placebo = bool(
                    abs(res.delta) > abs(res.placebo_delta) + 0.02)

    # anchor on human labels where both molecules are in the curated set
    if claim.molecule in labels and edited in labels:
        res.anchored = True
        res.anchor_before = int(claim.descriptor in labels[claim.molecule])
        res.anchor_after = int(claim.descriptor in labels[edited])
        d = res.anchor_after - res.anchor_before
        res.anchor_sign_agrees = bool(d != 0 and np.sign(d) == res.expected_sign)
    return res


def run_all(claims: list[Claim], oracle: FingerprintOracle, data: OdorData,
            seed: int = 0) -> list[InterventionResult]:
    labels = _label_lookup(data)
    rng = np.random.default_rng(seed)
    by_mol: dict[str, set[str]] = {}
    for c in claims:
        by_mol.setdefault(c.molecule, set()).add(c.group)
    return [test_claim(c, oracle, labels, by_mol.get(c.molecule), rng)
            for c in claims]


def matched_pairs(data: OdorData, group: str, descriptor: str,
                  max_pairs: int = 200) -> list[dict]:
    """Human-labelled matched molecular pairs for one (group, descriptor):
    molecules in the curated set whose ablation product is also in the set.

    This is the sanity anchor for the whole method. If the intervention
    machinery is sound, these pairs should show the established
    structure-odour relationships (thiol -> sulfurous, and so on) without any
    LLM in the loop. Run it before trusting any faithfulness number.
    """
    labels = _label_lookup(data)
    out = []
    for smi in data.smiles:
        if group not in chem.groups_present(smi):
            continue
        edited = chem.ablate(smi, group)
        if edited is None or edited not in labels:
            continue
        out.append({
            "parent": smi, "child": edited,
            "parent_has": int(descriptor in labels[smi]),
            "child_has": int(descriptor in labels[edited]),
        })
        if len(out) >= max_pairs:
            break
    return out
