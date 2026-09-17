"""Step 7: final results. Regenerates every table and figure from the logs
already on disk. No API calls, no new agent runs.

    python scripts\\07_final.py

Fixes two problems in the earlier analysis:

1. Objective. The agent optimises the MEAN predicted probability over target
   descriptors, which is separable: a molecule can score 0.6 on aldehydic and
   0.2 on sulfurous and post a respectable 0.4 without ever being both. On
   unrealisable targets that flatters the agent badly. Everything here is
   reported on both mean and MIN, and min is the headline.

2. Faithfulness null and anchoring. Sign agreement requires the descriptor to
   move, so ties count as failures and the null is not 0.5. This computes the
   null empirically by permutation: the same claims attached to different
   molecules. The human-anchored test is reported three ways (changed right /
   changed wrong / did not change) instead of collapsing non-events into
   failures.

Inputs (all already produced by steps 01-06):
    results/oracle.pkl
    results/runs_strata.jsonl
    results/strata.json
    results/contamination.jsonl
Outputs:
    results/final_tables.json
    results/fig1_strata.png   fig2_rounds.png   fig3_contamination.png
    results/fig4_faithfulness.png
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cos import claims as cl, interventions as iv
from cos.analysis import bootstrap_ci, paired_delta
from cos.data import load as load_data
from cos.oracle import FingerprintOracle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

ARM_ORDER = ["full", "no_reasoning", "scrambled", "no_feedback", "random_mutation"]
ARM_LABEL = {"full": "full agent", "no_reasoning": "no reasoning",
             "scrambled": "scrambled reasoning", "no_feedback": "no feedback",
             "random_mutation": "random search"}
STRATUM_ORDER = ["recognised", "unrecognised", "unrealisable"]
# qualitative Set2: unordered categories need distinct hues, not one ramp
_SET2 = plt.get_cmap("Set2").colors
COLORS = {arm: _SET2[i] for i, arm in enumerate(ARM_ORDER)}
COLORS["random_mutation"] = _SET2[7]  # grey, it is the baseline

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})


def jload(path):
    with open(path) as fh:
        return json.load(fh)


def jlload(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


# --------------------------------------------------------------------------
def score_run(log, orc, how="min"):
    """Best score achieved in a run under a given objective."""
    best = 0.0
    for rd in log["rounds"]:
        if not rd.get("valid") or not rd.get("proposal"):
            continue
        p = orc.predict(rd["proposal"])
        if not p.valid:
            continue
        v = [p.probs.get(d, 0.0) for d in log["target"]]
        s = float(np.min(v)) if how == "min" else float(np.mean(v))
        best = max(best, s)
    return best


def trajectory(log, orc, how="min"):
    """Running best after each round, for the rounds figure."""
    out, best = [], 0.0
    for rd in log["rounds"]:
        if rd.get("valid") and rd.get("proposal"):
            p = orc.predict(rd["proposal"])
            if p.valid:
                v = [p.probs.get(d, 0.0) for d in log["target"]]
                best = max(best, float(np.min(v)) if how == "min"
                           else float(np.mean(v)))
        out.append(best)
    return out


def build_scores(logs, orc):
    sc = {"min": defaultdict(dict), "mean": defaultdict(dict)}
    traj = defaultdict(list)
    for log in logs:
        for how in ("min", "mean"):
            sc[how][log["arm"]][log["target_id"]] = score_run(log, orc, how)
        traj[log["arm"]].append(trajectory(log, orc, "min"))
    return sc, traj


# --------------------------------------------------------------------------
def table_strata(sc, strata):
    lab = {t["id"]: k for k, v in strata.items() for t in v}
    out = {}
    for how in ("min", "mean"):
        block = {}
        for stratum in STRATUM_ORDER:
            ids = [t["id"] for t in strata[stratum]]
            row = {}
            for arm in ARM_ORDER:
                vals = [sc[how][arm][i] for i in ids if i in sc[how][arm]]
                if not vals:
                    continue
                m, lo, hi = bootstrap_ci(vals)
                row[arm] = {"n": len(vals), "mean": m, "ci": [lo, hi]}
            block[stratum] = row
        # recognised vs unrealisable, full arm, expressed against each floor
        r = block["recognised"].get("full", {}).get("mean")
        x = block["unrealisable"].get("full", {}).get("mean")
        fr = block["recognised"].get("random_mutation", {}).get("mean")
        fx = block["unrealisable"].get("random_mutation", {}).get("mean")
        block["_summary"] = {
            "full_recognised": r, "full_unrealisable": x,
            "floor_recognised": fr, "floor_unrealisable": fx,
            "ratio_recognised_to_unrealisable": (r / x) if x else None,
            "full_over_floor_recognised": (r / fr) if fr else None,
            "full_over_floor_unrealisable": (x / fx) if fx else None,
        }
        out[how] = block
    out["_note"] = (
        "The unrecognised stratum has a much higher random-search floor than "
        "the other two, so its targets are intrinsically easier and it is not "
        "comparable. Headline contrast is recognised vs unrealisable."
    )
    return out


def table_diagnostics(logs, orc, sim_threshold: float = 0.4) -> dict:
    """Proposal quality and proxy-exploitation diagnostics, over the same runs
    as everything else (Methods 3.7)."""
    per_arm = defaultdict(lambda: {"valid": [], "plaus": [], "sa": []})
    score, sim, std = [], [], []
    for log in logs:
        arm = log["arm"]
        for rd in log["rounds"]:
            ok = bool(rd.get("valid"))
            per_arm[arm]["valid"].append(float(ok))
            if not ok or not rd.get("proposal"):
                continue
            p = orc.predict(rd["proposal"])
            if not p.valid:
                continue
            v = [p.probs.get(d, 0.0) for d in log["target"]]
            score.append(float(np.min(v)))
            sim.append(p.nn_similarity)
            std.append(max(p.ensemble_std.values()) if p.ensemble_std else 0.0)
            per_arm[arm]["plaus"].append(float(p.plausible))
            if p.sa_score is not None:
                per_arm[arm]["sa"].append(p.sa_score)

    score, sim, std = np.array(score), np.array(sim), np.array(std)
    far = sim < sim_threshold
    return {
        "n_proposals": int(len(score)),
        "per_arm": {a: {"smiles_validity": float(np.mean(v["valid"])),
                        "odorant_plausible_rate": float(np.mean(v["plaus"])) if v["plaus"] else None,
                        "mean_sa_score": float(np.mean(v["sa"])) if v["sa"] else None}
                    for a, v in per_arm.items()},
        "corr_score_vs_train_similarity": float(np.corrcoef(score, sim)[0, 1]),
        "corr_score_vs_ensemble_std": float(np.corrcoef(score, std)[0, 1]),
        "mean_similarity_to_train": float(sim.mean()),
        "fraction_beyond_similarity_0.4": float(far.mean()),
        "mean_score_within_0.4": float(score[~far].mean()) if (~far).any() else None,
        "mean_score_beyond_0.4": float(score[far].mean()) if far.any() else None,
    }


def table_contrasts(sc):
    out = {}
    for how in ("min", "mean"):
        out[how] = {f"full_minus_{o}": paired_delta(sc[how]["full"], sc[how][o])
                    for o in ARM_ORDER if o != "full"}
    return out


# --------------------------------------------------------------------------
def table_faithfulness(logs, orc, data, seed=0):
    extractor = cl.RuleExtractor(data.descriptors)
    C = cl.extract_from_logs(logs, extractor)
    real = iv.run_all(C, orc, data, seed=seed)

    rng = np.random.default_rng(seed)
    mols = sorted({c.molecule for c in C})
    perm = []
    for c in C:
        m = c.molecule
        while m == c.molecule and len(mols) > 1:
            m = str(rng.choice(mols))
        perm.append(cl.Claim(c.target_id, c.arm, c.round_index, m, c.group,
                             c.descriptor, c.direction, c.sentence, "perm"))
    null = iv.run_all(perm, orc, data, seed=seed + 1)

    def block(rs):
        grounded = [r for r in rs if r.grounded]
        tested = [r for r in grounded if r.delta is not None]
        anch = [r for r in tested if r.anchored]
        moved = [r for r in anch if r.anchor_after != r.anchor_before]
        right = [r for r in moved if r.anchor_sign_agrees]
        sg, lo, hi = bootstrap_ci([float(r.sign_agrees) for r in tested])
        placebo = [r for r in tested if r.specific_vs_placebo is not None]
        return {
            "n_claims": len(rs),
            "grounding_rate": len(grounded) / len(rs) if rs else None,
            "n_tested": len(tested),
            "sign_agreement": {"mean": sg, "ci": [lo, hi]},
            "beats_placebo_rate": float(np.mean(
                [r.specific_vs_placebo for r in placebo])) if placebo else None,
            "anchored": {
                "n": len(anch),
                "label_changed": len(moved),
                "changed_correct_direction": len(right),
                "changed_wrong_direction": len(moved) - len(right),
                "label_unchanged": len(anch) - len(moved),
                "accuracy_given_change": (len(right) / len(moved)) if moved else None,
            },
        }

    out = {"real": block(real), "permuted_null": block(null)}
    for k in ("grounding_rate",):
        out[f"delta_{k}"] = out["real"][k] - out["permuted_null"][k]
    out["delta_sign_agreement"] = (out["real"]["sign_agreement"]["mean"]
                                   - out["permuted_null"]["sign_agreement"]["mean"])
    out["_note"] = (
        "The permutation null attaches each claim to a different proposed "
        "molecule. It is the correct baseline: sign agreement requires the "
        "descriptor to move, so ties count as failures and chance is not 0.5. "
        "The anchored test is reported three ways because a human label that "
        "does not change is a non-event, not a failed prediction."
    )
    return out, real, null


def table_contamination(path, floor=0.134):
    recs = jlload(path)
    ins = [r for r in recs if r.get("in_dataset")]
    named = [r for r in ins if r.get("named") and
             str(r["named"]).lower() not in ("null", "none", "")]
    unnamed = [r for r in ins if r not in named]
    f = lambda rs: float(np.mean([r["f1"] for r in rs])) if rs else None
    return {
        "n": len(ins),
        "frequency_floor_f1": floor,
        "mean_f1": f(ins),
        "named_rate": len(named) / len(ins) if ins else None,
        "mean_f1_named": f(named),
        "mean_f1_unnamed": f(unnamed),
        "high_recall_fraction": float(np.mean([r["recall"] > 0.5 for r in ins])),
        "_note": ("frequency_floor_f1 is the score from always guessing the five "
                  "commonest descriptors, computed on the same test scaffolds."),
    }


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_strata(tab, path):
    block = tab["min"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.16
    xs = np.arange(len(STRATUM_ORDER))
    for k, arm in enumerate(ARM_ORDER):
        means, errs = [], [[], []]
        for stratum in STRATUM_ORDER:
            row = block[stratum].get(arm)
            m = row["mean"] if row else 0.0
            ci = row["ci"] if row else [0, 0]
            means.append(m)
            errs[0].append(max(0, m - ci[0]))
            errs[1].append(max(0, ci[1] - m))
        ax.bar(xs + (k - 2) * width, means, width, yerr=errs, capsize=2,
               label=ARM_LABEL[arm], color=COLORS[arm],
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["recognised\n(model knows them)",
                        "unrecognised\n(model does not)",
                        "unrealisable\n(no molecule exists)"])
    ax.set_ylabel("best target match (min over descriptors)")
    ax.legend(frameon=False, ncol=5, fontsize=7.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0,
              columnspacing=1.2, handlelength=1.2)
    fig.suptitle("Every LLM arm performs alike; only random search differs",
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path)
    plt.close(fig)


def fig_rounds(traj, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for arm in ARM_ORDER:
        runs = [t for t in traj[arm] if t]
        if not runs:
            continue
        n = max(len(t) for t in runs)
        padded = np.array([t + [t[-1]] * (n - len(t)) for t in runs])
        m = padded.mean(axis=0)
        se = padded.std(axis=0) / np.sqrt(len(padded))
        x = np.arange(1, n + 1)
        ax.plot(x, m, marker="o", ms=3, label=ARM_LABEL[arm], color=COLORS[arm])
        ax.fill_between(x, m - se, m + se, color=COLORS[arm], alpha=0.15)
    ax.set_xlabel("round")
    ax.set_ylabel("running best (min objective)")
    ax.set_title("Iteration buys little after the first proposal")
    ax.set_xticks(range(1, 5))
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_contamination(tab, path):
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    labels = ["frequency\nfloor", "all test\nmolecules", "could name\nit", "could not\nname it"]
    vals = [tab["frequency_floor_f1"], tab["mean_f1"],
            tab["mean_f1_named"], tab["mean_f1_unnamed"]]
    cols = [_SET2[7], _SET2[0], _SET2[1], _SET2[2]]
    ax.bar(labels, vals, color=cols, edgecolor="white")
    ax.axhline(tab["frequency_floor_f1"], ls="--", lw=0.8, color="#666666")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("descriptor F1, no tool")
    ax.set_title("The model recalls its own test set")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_faithfulness(tab, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for ax, key, title in (
            (axes[0], "grounding_rate", "Named group is present"),
            (axes[1], "sign_agreement", "Predicted effect direction")):
        if key == "sign_agreement":
            r = tab["real"][key]["mean"]
            n = tab["permuted_null"][key]["mean"]
            rc = tab["real"][key]["ci"]
            nc = tab["permuted_null"][key]["ci"]
            errs = [[r - rc[0], n - nc[0]], [rc[1] - r, nc[1] - n]]
        else:
            r, n = tab["real"][key], tab["permuted_null"][key]
            errs = None
        ax.bar(["stated\nreasoning", "permuted\nnull"], [r, n],
               yerr=errs, capsize=3, color=[_SET2[0], _SET2[7]],
               edgecolor="white")
        tops = ([rc[1], nc[1]] if key == "sign_agreement" else [r, n])
        for i, (v, t) in enumerate(zip((r, n), tops)):
            ax.text(i, t + 0.035, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("rate")
    fig.suptitle("Reasoning is weakly grounded, not random", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    data = load_data()
    orc = FingerprintOracle.load(os.path.join(RES, "oracle.pkl"))
    logs = jlload(os.path.join(RES, "runs_strata.jsonl"))
    strata = jload(os.path.join(RES, "strata.json"))

    print(f"{len(logs)} runs, {sum(len(v) for v in strata.values())} targets")
    sc, traj = build_scores(logs, orc)

    tabs = {
        "oracle": jload(os.path.join(RES, "oracle_eval.json")),
        "matched_pair_controls": jload(os.path.join(RES, "matched_pairs.json")),
        "strata": table_strata(sc, strata),
        "contrasts": table_contrasts(sc),
        "contamination": table_contamination(
            os.path.join(RES, "contamination.jsonl")),
        "diagnostics": table_diagnostics(logs, orc),
    }
    faith, real, null = table_faithfulness(logs, orc, data)
    tabs["faithfulness"] = faith

    with open(os.path.join(RES, "final_tables.json"), "w") as fh:
        json.dump(tabs, fh, indent=2, default=float)

    fig_strata(tabs["strata"], os.path.join(RES, "fig1_strata.png"))
    fig_rounds(traj, os.path.join(RES, "fig2_rounds.png"))
    fig_contamination(tabs["contamination"], os.path.join(RES, "fig3_contamination.png"))
    fig_faithfulness(faith, os.path.join(RES, "fig4_faithfulness.png"))

    # ---- printed summary -------------------------------------------------
    b = tabs["strata"]["min"]
    print("\n" + "=" * 68)
    print("1. PERFORMANCE BY STRATUM  (min objective)")
    for stratum in STRATUM_ORDER:
        print(f"\n  {stratum}")
        for arm in ARM_ORDER:
            row = b[stratum].get(arm)
            if row:
                print(f"    {ARM_LABEL[arm]:22s} n={row['n']:2d}  {row['mean']:.3f} "
                      f"[{row['ci'][0]:.3f},{row['ci'][1]:.3f}]")
    s = b["_summary"]
    print(f"\n  full agent: recognised {s['full_recognised']:.3f} vs "
          f"unrealisable {s['full_unrealisable']:.3f} "
          f"({s['ratio_recognised_to_unrealisable']:.1f}x)")
    print(f"  above random search: {s['full_over_floor_recognised']:.1f}x and "
          f"{s['full_over_floor_unrealisable']:.1f}x")

    print("\n2. ARM CONTRASTS  (paired, min objective, all targets)")
    for k, v in tabs["contrasts"]["min"].items():
        if v:
            print(f"    {k:28s} {v['mean_delta']:+.3f} "
                  f"[{v['ci_low']:+.3f},{v['ci_high']:+.3f}]  p={v['p_two_sided']:.3f}")

    d = tabs["diagnostics"]
    print("\n2b. PROPOSAL DIAGNOSTICS")
    print(f"    n={d['n_proposals']}  "
          f"corr(score,similarity)={d['corr_score_vs_train_similarity']:+.3f}  "
          f"corr(score,ens.std)={d['corr_score_vs_ensemble_std']:+.3f}")
    print(f"    mean similarity {d['mean_similarity_to_train']:.3f}  "
          f"beyond 0.4: {d['fraction_beyond_similarity_0.4']:.3f}")
    for a, v in d["per_arm"].items():
        print(f"    {a:16s} valid={v['smiles_validity']:.3f} "
              f"plausible={v['odorant_plausible_rate']} "
              f"SA={v['mean_sa_score']:.2f}" if v['mean_sa_score'] else f"    {a}")

    c = tabs["contamination"]
    print("\n3. CONTAMINATION")
    print(f"    floor {c['frequency_floor_f1']:.3f} | overall {c['mean_f1']:.3f} | "
          f"named {c['mean_f1_named']:.3f} | unnamed {c['mean_f1_unnamed']:.3f} | "
          f"names {c['named_rate']:.0%}")

    f = tabs["faithfulness"]
    print("\n4. FAITHFULNESS  (real vs permutation null)")
    print(f"    grounding      {f['real']['grounding_rate']:.3f} vs "
          f"{f['permuted_null']['grounding_rate']:.3f}")
    print(f"    sign agreement {f['real']['sign_agreement']['mean']:.3f} vs "
          f"{f['permuted_null']['sign_agreement']['mean']:.3f}")
    a = f["real"]["anchored"]
    print(f"    human-anchored n={a['n']}: changed {a['label_changed']} "
          f"({a['changed_correct_direction']} right, "
          f"{a['changed_wrong_direction']} wrong), "
          f"unchanged {a['label_unchanged']}")
    if a["accuracy_given_change"] is not None:
        print(f"    accuracy given the label moved: {a['accuracy_given_change']:.3f} "
              f"(null {f['permuted_null']['anchored']['accuracy_given_change']})")
    print("=" * 68)
    print("wrote results/final_tables.json and fig1..fig4")


if __name__ == "__main__":
    main()
