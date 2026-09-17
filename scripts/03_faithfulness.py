"""Step 3: extract claims from the logged reasoning and test each one by
intervention.

    python scripts/03_faithfulness.py --extractor rule
    python scripts/03_faithfulness.py --extractor both --backend anthropic

Writes results/claims.jsonl, results/interventions.jsonl,
results/extractor_agreement.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cos import agent, claims as cl, interventions as iv
from cos.data import load
from cos.llm import get_llm
from cos.oracle import FingerprintOracle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(RES, "runs.jsonl"))
    ap.add_argument("--extractor", default="rule",
                    choices=["rule", "llm", "both"])
    ap.add_argument("--backend", default="mock", choices=["mock", "anthropic"])
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="extraction is a cheap task; a small model is fine")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load(seed=args.seed)
    orc = FingerprintOracle.load(os.path.join(RES, "oracle.pkl"))
    logs = agent.read_logs(args.runs)
    print(f"{len(logs)} runs loaded")

    rule = cl.RuleExtractor(data.descriptors)
    rule_claims = cl.extract_from_logs(logs, rule)
    print(f"rule extractor: {len(rule_claims)} claims")

    llm_claims: list[cl.Claim] = []
    if args.extractor in ("llm", "both"):
        llm = get_llm(args.backend, model=args.model, seed=args.seed)
        llm_ex = cl.LLMExtractor(llm, data.descriptors)
        llm_claims = cl.extract_from_logs(logs, llm_ex)
        print(f"llm extractor:  {len(llm_claims)} claims")
        agree = cl.agreement(rule_claims, llm_claims)
        print("agreement:", json.dumps(agree))
        with open(os.path.join(RES, "extractor_agreement.json"), "w") as fh:
            json.dump(agree, fh, indent=2)

    use = {"rule": rule_claims, "llm": llm_claims,
           "both": rule_claims + llm_claims}[args.extractor]
    # deduplicate on (molecule, group, descriptor, round)
    seen, uniq = set(), []
    for c in use:
        k = (c.molecule, c.group, c.descriptor, c.target_id, c.arm, c.round_index)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    print(f"{len(uniq)} unique claims to test")

    with open(os.path.join(RES, "claims.jsonl"), "w") as fh:
        for c in uniq:
            fh.write(json.dumps(c.to_json()) + "\n")

    results = iv.run_all(uniq, orc, data, seed=args.seed)
    with open(os.path.join(RES, "interventions.jsonl"), "w") as fh:
        for r in results:
            fh.write(json.dumps(r.to_json()) + "\n")

    grounded = [r for r in results if r.grounded]
    tested = [r for r in grounded if r.delta is not None]
    anchored = [r for r in tested if r.anchored]
    print(f"\ngrounded {len(grounded)}/{len(results)}  tested {len(tested)}  "
          f"human-anchored {len(anchored)}")
    print("wrote results/claims.jsonl, results/interventions.jsonl")


if __name__ == "__main__":
    main()
