"""Step 6: the retrieval-vs-reasoning experiment.

Builds three target strata from the measured contamination probe, runs every
arm on all of them, and reports performance within each stratum.

    # build the strata (needs results/contamination.jsonl)
    python scripts\\06_strata.py build --per-stratum 12

    # run the agents on them
    python scripts\\06_strata.py run --backend anthropic --model claude-sonnet-5 --rounds 4

    # compare
    python scripts\\06_strata.py compare

Writes results/strata.json, results/runs_strata.jsonl, results/table_strata.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cos import agent, strata as st
from cos.data import load as load_data
from cos.llm import get_llm
from cos.oracle import FingerprintOracle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
STRATA = os.path.join(RES, "strata.json")
RUNS = os.path.join(RES, "runs_strata.jsonl")


def cmd_build(args):
    data = load_data(seed=args.seed)
    s = st.build(data, os.path.join(RES, "contamination.jsonl"),
                 n_per_stratum=args.per_stratum, seed=args.seed,
                 recall_hi=args.recall_hi, recall_lo=args.recall_lo)
    st.save(s, STRATA)
    for name in ("recognised", "unrecognised", "unrealisable"):
        group = getattr(s, name)
        print(f"\n{name}: {len(group)} targets")
        for t in group[:4]:
            print(f"  {t.id}  {'+'.join(t.descriptors)}"
                  f"{'  ref=' + t.reference_smiles if t.reference_smiles else ''}")
        if len(group) > 4:
            print(f"  ... and {len(group) - 4} more")
    print(f"\nwrote {STRATA}")


def cmd_run(args):
    s = st.load(STRATA)
    orc = FingerprintOracle.load(os.path.join(RES, "oracle.pkl"))
    llm = get_llm(args.backend, model=args.model, seed=args.seed, max_tokens=3000)
    targets = s.all()
    print(f"{len(targets)} targets across 3 strata")

    scramble_pool: list[str] = []
    if os.path.exists(RUNS):
        for log in agent.read_logs(RUNS):
            if log["arm"] == "full":
                scramble_pool += [r["reasoning"] for r in log["rounds"]
                                  if r.get("reasoning")]

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    order = [a for a in ("full", "no_reasoning", "no_feedback", "scrambled",
                         "random_mutation") if a in arms]
    for arm in order:
        for t in targets:
            if arm == "random_mutation":
                log = agent.run_random_mutation(orc, t, rounds=args.rounds,
                                                seed=args.seed)
            else:
                if arm == "scrambled" and not scramble_pool:
                    print("skipping scrambled: run the full arm first")
                    break
                log = agent.run_llm(llm, orc, t, arm=arm, rounds=args.rounds,
                                    seed=args.seed, scramble_pool=scramble_pool)
                if arm == "full":
                    scramble_pool += [r.reasoning for r in log.rounds if r.reasoning]
            agent.append_log(log, RUNS)
            print(f"{s.label_of(t.id):14s} {arm:16s} {t.id} "
                  f"best={log.best_score:.3f} {log.error}")
    print(f"\nwrote {RUNS}")


def cmd_compare(args):
    s = st.load(STRATA)
    logs = agent.read_logs(RUNS)
    table = st.compare(logs, s)
    with open(os.path.join(RES, "table_strata.json"), "w") as fh:
        json.dump(table, fh, indent=2, default=float)

    print("=" * 66)
    print("PERFORMANCE BY STRATUM (best target match)")
    for lab in ("recognised", "unrecognised", "unrealisable"):
        block = table["by_stratum"].get(lab)
        if not block:
            continue
        print(f"\n  {lab}")
        for arm, v in block.items():
            print(f"    {arm:16s} n={v['n']:3d}  {v['mean']:.3f} "
                  f"[{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
    for k in ("recognised_minus_unrecognised", "recognised_minus_unrealisable"):
        if k in table:
            v = table[k]
            print(f"\n  {k}: gap={v['gap']:+.3f}  "
                  f"full_above_floor={v['full_above_floor']}")
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--per-stratum", type=int, default=12)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--recall-hi", type=float, default=0.5,
                   help="min probe recall for the 'recognised' stratum")
    b.add_argument("--recall-lo", type=float, default=0.2,
                   help="max probe recall for the 'unrecognised' stratum")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("run")
    r.add_argument("--backend", default="mock", choices=["mock", "anthropic"])
    r.add_argument("--model", default="claude-sonnet-5")
    r.add_argument("--rounds", type=int, default=4)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--arms", default=",".join(agent.ARMS))
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
