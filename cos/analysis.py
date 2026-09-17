"""Metrics, tables and figures.

Three result blocks, in the order a reviewer will want them:

  A. Did the agent do the task, and does reasoning explain it?
     Arm comparison with bootstrap CIs. The full arm has to beat
     no_reasoning, scrambled and random_mutation before anything about
     reasoning quality is interpretable.

  B. Is the improvement real or is it oracle gaming?
     Novelty, ensemble disagreement, plausibility and SA of the winners.
     Improvement concentrated in low-similarity, high-variance regions is the
     agent exploiting the proxy, not designing a molecule.

  C. Is the stated reasoning causally load-bearing?
     Grounding rate, sign agreement, specificity vs placebo, and the
     human-label-anchored subset reported separately.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------
def bootstrap_ci(values, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(
        np.quantile(means, 1 - alpha / 2))


def paired_delta(a: dict[str, float], b: dict[str, float], seed: int = 0):
    """Paired bootstrap over shared target ids, for arm A minus arm B."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys], dtype=float)
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(5000, len(d)), replace=True).mean(axis=1)
    return {"n_pairs": len(d), "mean_delta": float(d.mean()),
            "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)),
            "p_two_sided": float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))}


# --------------------------------------------------------------------------
# A. arm comparison
# --------------------------------------------------------------------------
def arm_table(logs: list[dict], seed: int = 0) -> dict:
    by_arm: dict[str, dict[str, float]] = defaultdict(dict)
    aux: dict[str, list] = defaultdict(list)
    for log in logs:
        by_arm[log["arm"]][log["target_id"]] = log.get("best_score", 0.0)
        valid = [r for r in log["rounds"] if r.get("valid")]
        aux[log["arm"]].append({
            "n_rounds": len(log["rounds"]),
            "validity": len(valid) / max(1, len(log["rounds"])),
            "first": valid[0]["target_match"] if valid else 0.0,
            "best": log.get("best_score", 0.0),
            "nn": np.mean([r["nn_similarity"] for r in valid]) if valid else np.nan,
            "sa": np.mean([r["sa_score"] for r in valid if r["sa_score"]]) if valid else np.nan,
            "plausible": np.mean([r["plausible"] for r in valid]) if valid else np.nan,
            "std": np.mean([r["max_ensemble_std"] for r in valid]) if valid else np.nan,
        })

    table = {}
    for arm, rows in aux.items():
        best = [r["best"] for r in rows]
        m, lo, hi = bootstrap_ci(best, seed=seed)
        table[arm] = {
            "n_runs": len(rows),
            "best_target_match": {"mean": m, "ci": [lo, hi]},
            "first_proposal_match": float(np.nanmean([r["first"] for r in rows])),
            "improvement_over_first": float(
                np.nanmean([r["best"] - r["first"] for r in rows])),
            "smiles_validity": float(np.nanmean([r["validity"] for r in rows])),
            "mean_nn_similarity_to_train": float(np.nanmean([r["nn"] for r in rows])),
            "mean_sa_score": float(np.nanmean([r["sa"] for r in rows])),
            "odorant_plausible_rate": float(np.nanmean([r["plausible"] for r in rows])),
            "mean_oracle_ensemble_std": float(np.nanmean([r["std"] for r in rows])),
        }

    contrasts = {}
    if "full" in by_arm:
        for other in by_arm:
            if other == "full":
                continue
            contrasts[f"full_minus_{other}"] = paired_delta(
                by_arm["full"], by_arm[other], seed=seed)
    return {"per_arm": table, "contrasts": contrasts}


# --------------------------------------------------------------------------
# B. oracle gaming
# --------------------------------------------------------------------------
def gaming_report(logs: list[dict], sim_threshold: float = 0.4) -> dict:
    rows = []
    for log in logs:
        for r in log["rounds"]:
            if r.get("valid"):
                rows.append((r["target_match"], r["nn_similarity"],
                             r["max_ensemble_std"], bool(r["plausible"]),
                             log["arm"]))
    if not rows:
        return {}
    score = np.array([r[0] for r in rows])
    sim = np.array([r[1] for r in rows])
    std = np.array([r[2] for r in rows])
    plaus = np.array([r[3] for r in rows])
    far = sim < sim_threshold
    return {
        "n_proposals": len(rows),
        "corr_score_vs_similarity": float(np.corrcoef(score, sim)[0, 1]),
        "corr_score_vs_ensemble_std": float(np.corrcoef(score, std)[0, 1]),
        "mean_score_near_training": float(score[~far].mean()) if (~far).any() else None,
        "mean_score_far_from_training": float(score[far].mean()) if far.any() else None,
        "fraction_far_from_training": float(far.mean()),
        "implausible_rate": float(1 - plaus.mean()),
        "interpretation": (
            "If high scores concentrate at low similarity and high ensemble "
            "disagreement, the agent is exploiting the oracle rather than "
            "designing an odorant. Report these proposals separately and "
            "exclude them from success rates."
        ),
    }


# --------------------------------------------------------------------------
# C. faithfulness
# --------------------------------------------------------------------------
def faithfulness_table(results: list[dict], seed: int = 0) -> dict:
    def block(rows):
        if not rows:
            return {"n": 0}
        grounded = [r for r in rows if r["grounded"]]
        tested = [r for r in grounded if r["delta"] is not None]
        sign = [r["sign_agrees"] for r in tested]
        m, lo, hi = bootstrap_ci([float(s) for s in sign], seed=seed)
        placebo = [r for r in tested if r["specific_vs_placebo"] is not None]
        anchored = [r for r in tested if r["anchored"]
                    and r["anchor_sign_agrees"] is not None]
        return {
            "n_claims": len(rows),
            "grounding_rate": len(grounded) / len(rows),
            "n_tested": len(tested),
            "sign_agreement": {"mean": m, "ci": [lo, hi]},
            "mean_abs_delta": float(np.mean([abs(r["delta"]) for r in tested]))
            if tested else None,
            "mean_specificity_z": float(np.mean([r["specificity_z"] for r in tested]))
            if tested else None,
            "beats_placebo_rate": float(np.mean(
                [r["specific_vs_placebo"] for r in placebo])) if placebo else None,
            "n_human_anchored": len(anchored),
            "anchored_sign_agreement": float(np.mean(
                [r["anchor_sign_agrees"] for r in anchored])) if anchored else None,
        }

    out = {"overall": block(results)}
    by_arm = defaultdict(list)
    by_group = defaultdict(list)
    for r in results:
        by_arm[r["arm"]].append(r)
        by_group[r["group"]].append(r)
    out["by_arm"] = {k: block(v) for k, v in by_arm.items()}
    out["by_group"] = {k: block(v) for k, v in by_group.items() if len(v) >= 5}
    out["note"] = (
        "sign_agreement above chance (0.5) means stated reasons predict the "
        "oracle's behaviour under intervention. anchored_sign_agreement is the "
        "same quantity measured against human labels and is the defensible "
        "number; the oracle-based figure measures agreement with a model."
    )
    return out


def write(obj: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)


def plot_arms(table: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(table["per_arm"])
    means = [table["per_arm"][a]["best_target_match"]["mean"] for a in arms]
    cis = [table["per_arm"][a]["best_target_match"]["ci"] for a in arms]
    err = [[m - c[0] for m, c in zip(means, cis)],
           [c[1] - m for m, c in zip(means, cis)]]
    set2 = plt.get_cmap("Set2").colors
    colors = [set2[7] if a == "random_mutation" else set2[i % 7]
              for i, a in enumerate(arms)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(arms, means, yerr=err, capsize=4, color=colors, edgecolor="white")
    ax.set_ylabel("best target match")
    ax.set_title("Design performance by arm (95% bootstrap CI)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
