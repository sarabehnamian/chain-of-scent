"""Step 4: memorisation probe.

    python scripts/04_contamination.py --backend anthropic --n 100

Writes results/contamination.jsonl and results/contamination_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cos import contamination as ct
from cos.data import load
from cos.llm import get_llm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=["mock", "anthropic"])
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load(seed=args.seed)
    llm = get_llm(args.backend, model=args.model, seed=args.seed)
    results = ct.run(llm, data, n=args.n, seed=args.seed)

    with open(os.path.join(RES, "contamination.jsonl"), "w") as fh:
        for r in results:
            fh.write(json.dumps(r.to_json()) + "\n")
    summary = ct.summarise(results)
    with open(os.path.join(RES, "contamination_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
