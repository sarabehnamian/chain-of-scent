"""Step 1: build the dataset split, train the oracle, and sanity-check the
intervention machinery against human labels before any LLM is involved.

    python scripts/01_prepare.py

Writes results/oracle.pkl, results/oracle_eval.json, results/matched_pairs.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cos import interventions
from cos.data import load
from cos.oracle import FingerprintOracle, evaluate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

# established structure-odour relationships used as positive controls
CONTROLS = [
    ("thiol", "sulfurous"),
    ("ester", "fruity"),
    ("aldehyde", "green"),
    ("aldehyde", "aldehydic"),
    ("lactone", "creamy"),
    ("phenol", "phenolic"),
    ("alkene", "green"),
    ("carboxylic_acid", "sour"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    data = load(seed=args.seed)
    print(f"{len(data.smiles)} molecules, {len(data.descriptors)} descriptors")
    for s in ("train", "valid", "test"):
        print(f"  {s}: {(data.split == s).sum()}")

    tr_s, tr_Y = data.subset("train")
    te_s, te_Y = data.subset("test")
    print("training oracle (train scaffolds only)...")
    orc = FingerprintOracle(data.descriptors, n_members=args.members,
                            seed=args.seed).fit(tr_s, tr_Y)
    ev = evaluate(orc, te_s, te_Y)
    print(json.dumps(ev, indent=2))
    orc.save(os.path.join(RES, "oracle.pkl"))
    with open(os.path.join(RES, "oracle_eval.json"), "w") as fh:
        json.dump(ev, fh, indent=2)

    # positive controls: does the intervention recover known chemistry from
    # HUMAN labels, with no model in the loop?
    print("\nmatched-pair positive controls (human labels):")
    report = {}
    for group, desc in CONTROLS:
        if desc not in data.descriptors:
            continue
        pairs = interventions.matched_pairs(data, group, desc)
        if not pairs:
            report[f"{group}->{desc}"] = {"n_pairs": 0}
            print(f"  {group:18s} -> {desc:12s}  no pairs")
            continue
        keeps = sum(p["parent_has"] and p["child_has"] for p in pairs)
        loses = sum(p["parent_has"] and not p["child_has"] for p in pairs)
        gains = sum((not p["parent_has"]) and p["child_has"] for p in pairs)
        loss_rate = loses / (keeps + loses) if (keeps + loses) else None
        report[f"{group}->{desc}"] = {
            "n_pairs": len(pairs), "parent_has": keeps + loses,
            "loses_descriptor_on_ablation": loses,
            "keeps": keeps, "gains": gains, "loss_rate": loss_rate,
        }
        print(f"  {group:18s} -> {desc:12s}  {len(pairs):4d} pairs, "
              f"loss rate {loss_rate if loss_rate is None else round(loss_rate, 2)}")

    with open(os.path.join(RES, "matched_pairs.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote results/oracle.pkl, oracle_eval.json, matched_pairs.json")


if __name__ == "__main__":
    main()
