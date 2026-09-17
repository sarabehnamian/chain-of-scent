"""Step 2: build target profiles and run every arm.

    # offline dry run, no API key needed, exercises the whole pipeline
    python scripts/02_run_agents.py --backend mock --n-targets 8 --rounds 4

    # real run
    export ANTHROPIC_API_KEY=...
    python scripts/02_run_agents.py --backend anthropic --model claude-sonnet-5 \
        --n-targets 30 --rounds 6 --focus green

Appends to results/runs.jsonl (one JSON object per run).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cos import agent, targets as tg
from cos.data import load
from cos.llm import get_llm
from cos.oracle import FingerprintOracle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=["mock", "anthropic"])
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--n-targets", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--focus", default=None,
                    help="restrict targets to profiles containing this descriptor")
    ap.add_argument("--arms", default=",".join(agent.ARMS))
    ap.add_argument("--out", default=os.path.join(RES, "runs.jsonl"))
    args = ap.parse_args()

    data = load(seed=args.seed)
    orc = FingerprintOracle.load(os.path.join(RES, "oracle.pkl"))

    target_path = os.path.join(RES, "targets.json")
    if os.path.exists(target_path):
        targets = tg.load(target_path)
        print(f"reusing {len(targets)} targets from {target_path}")
    else:
        targets = tg.build(data, n=args.n_targets, seed=args.seed,
                           focus=[args.focus] if args.focus else None)
        tg.save(targets, target_path)
        print(f"built {len(targets)} targets -> {target_path}")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    llm = get_llm(args.backend, model=args.model, seed=args.seed, max_tokens=3000)

    # pool of reasoning text for the scrambled arm, harvested from full runs
    scramble_pool: list[str] = []
    if os.path.exists(args.out):
        for log in agent.read_logs(args.out):
            if log["arm"] == "full":
                scramble_pool += [r["reasoning"] for r in log["rounds"]
                                  if r.get("reasoning")]

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
            agent.append_log(log, args.out)
            print(f"{arm:16s} {t.id} best={log.best_score:.3f} "
                  f"rounds={len(log.rounds)} {log.error}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
