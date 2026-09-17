"""Step 5: tables and figures.

    python scripts/05_analyze.py

Writes results/table_arms.json, results/table_gaming.json,
results/table_faithfulness.json, results/fig_arms.png and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cos import agent, analysis

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    logs = agent.read_logs(os.path.join(RES, "runs.jsonl"))
    inters = load_jsonl(os.path.join(RES, "interventions.jsonl"))

    arms = analysis.arm_table(logs)
    gaming = analysis.gaming_report(logs)
    faith = analysis.faithfulness_table(inters) if inters else {}

    analysis.write(arms, os.path.join(RES, "table_arms.json"))
    analysis.write(gaming, os.path.join(RES, "table_gaming.json"))
    if faith:
        analysis.write(faith, os.path.join(RES, "table_faithfulness.json"))
    if not args.no_figure:
        try:
            analysis.plot_arms(arms, os.path.join(RES, "fig_arms.png"))
        except Exception as exc:
            print(f"(figure skipped: {exc})")

    print("=" * 62)
    print("A. DESIGN PERFORMANCE BY ARM")
    for arm, row in arms["per_arm"].items():
        ci = row["best_target_match"]["ci"]
        print(f"  {arm:16s} best={row['best_target_match']['mean']:.3f} "
              f"[{ci[0]:.3f},{ci[1]:.3f}]  "
              f"gain={row['improvement_over_first']:+.3f}  "
              f"valid={row['smiles_validity']:.2f}  "
              f"nn={row['mean_nn_similarity_to_train']:.2f}")
    print("\n  contrasts (paired bootstrap):")
    for k, v in arms["contrasts"].items():
        if v:
            print(f"    {k:28s} {v['mean_delta']:+.3f} "
                  f"[{v['ci_low']:+.3f},{v['ci_high']:+.3f}] p={v['p_two_sided']:.3f}")

    print("\nB. ORACLE GAMING")
    for k, v in gaming.items():
        if k != "interpretation":
            print(f"  {k}: {v}")

    if faith:
        print("\nC. FAITHFULNESS")
        o = faith["overall"]
        print(f"  claims={o['n_claims']} grounded={o['grounding_rate']:.2f} "
              f"tested={o['n_tested']}")
        sa = o["sign_agreement"]
        if sa["mean"] is not None:
            print(f"  sign agreement (oracle) = {sa['mean']:.2f} "
                  f"[{sa['ci'][0]:.2f},{sa['ci'][1]:.2f}]  (chance 0.5)")
        print(f"  beats placebo = {o['beats_placebo_rate']}")
        print(f"  human-anchored n={o['n_human_anchored']} "
              f"agreement={o['anchored_sign_agreement']}")
    print("=" * 62)
    print("wrote tables to results/")


if __name__ == "__main__":
    main()
