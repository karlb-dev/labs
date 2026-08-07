"""Phase 3 release audit: align interval labels and randomization objects.

This is a new methods-tier closeout over immutable Phase 3 parquets. It does
not replace the frozen decisions.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd

from jspace_part2.lib import sha256_file

from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)
from ..stats import (exact_signflip_test, family_cluster_bootstrap_ci,
                     leave_one_family_out, monte_carlo_pvalue,
                     signflip_confidence_set, wild_cluster_percentile_t_ci,
                     within_fact_composition, within_fact_model_diff)

EVIDENCE_ID = "p3-inference-audit-v1"
TIER = "methods"
SLUGS = ("olmo31-think", "olmo31-instruct", "qwen36-27b")
THRESHOLDS = (-0.5, -1.0, -1.5, -2.0)
N_RANDOMIZATIONS = 100_000
SEED = 4242


def load_effects(side: str) -> tuple[pd.DataFrame, list[str]]:
    suffix = "" if side == "confirmatory" else "_replication"
    rows, paths = [], []
    for slug in SLUGS:
        path = (metrics_dir(slug) / f"p3_grid{suffix}"
                / f"p3_grid{suffix}_{slug}.parquet")
        df = pd.read_parquet(path)
        df["model"] = slug
        df["J_eff"] = df["lp_meanJ_span_safe"] - df["lp_baseline"]
        df["C_eff"] = df["lp_ss_matched"] - df["lp_baseline"]
        df["specific"] = df["J_eff"] - df["C_eff"]
        rows.append(df)
        paths.append(str(path))
    return pd.concat(rows, ignore_index=True), paths


def family_weighted_randomization(
        values: np.ndarray, families: np.ndarray, *,
        draws: int = N_RANDOMIZATIONS, seed: int = SEED,
        alternative: str = "greater") -> dict:
    """Item sign flips with the preregistered equal-family statistic."""
    d = np.asarray(values, dtype=float)
    fam = pd.Categorical(families)
    codes = fam.codes
    m = len(fam.categories)
    counts = np.bincount(codes, minlength=m).astype(float)
    weights = 1.0 / (m * counts[codes])
    weighted = d * weights
    observed = float(weighted.sum())
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=float)
    for start in range(0, draws, 5000):
        n = min(5000, draws - start)
        signs = rng.choice((-1, 1), size=(n, len(d))).astype(np.int8)
        null[start:start + n] = signs @ weighted
    return {
        "estimate": observed,
        "p_plus_one": monte_carlo_pvalue(
            null, observed, alternative=alternative),
        "alternative": alternative,
        "n_randomizations": int(draws),
        "n_items": int(len(d)),
        "n_families": int(m),
        "null_distribution_sha256": hashlib.sha256(
            np.asarray(null, dtype="<f8").tobytes()).hexdigest(),
    }


def effect_bootstrap(values: np.ndarray, families: np.ndarray, *,
                     draws: int = 20_000, seed: int = SEED) -> dict:
    """Equal-family, family-resampling percentile interval for effect size."""
    frame = pd.DataFrame({"value": values, "family": families})
    means = frame.groupby("family", sort=True)["value"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(means), size=(draws, len(means)))
    boot = means[picks].mean(axis=1)
    return {
        "estimate": float(means.mean()),
        "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "method": "family-resampling-percentile-effect-size",
        "n_bootstrap": int(draws),
        "n_families": int(len(means)),
        "distribution_sha256": hashlib.sha256(
            np.asarray(boot, dtype="<f8").tobytes()).hexdigest(),
    }


def p3p1_block(effects: pd.DataFrame, side: str) -> tuple[dict, pd.DataFrame]:
    comp = within_fact_composition(effects, value_col="specific")
    diff = within_fact_model_diff(
        comp, model_a="qwen36-27b",
        model_b=["olmo31-think", "olmo31-instruct"])
    family = diff.groupby("canonical_family", sort=True)["diff"].mean()
    vals = family.to_numpy()
    exact = exact_signflip_test(vals)
    inverted = signflip_confidence_set(vals)
    pct_t = wild_cluster_percentile_t_ci(
        diff.rename(columns={"diff": "d"}), "d")
    estimate = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    relation = effects[["fact_id", "relation_group"]].drop_duplicates(
        "fact_id")
    by_relation = diff.merge(relation, on="fact_id", how="left")
    lofo = leave_one_family_out(
        diff.rename(columns={"diff": "d"}), "d")
    family_rows = family.rename("family_mean").reset_index()
    family_rows.insert(0, "side", side)
    block = {
        "estimate_family_weighted": estimate,
        "exact_randomization": exact,
        "randomization_compatible_confidence_set": inverted,
        "wild_cluster_percentile_t_interval": pct_t,
        "normal_small_cluster_approximation": {
            "estimate": estimate,
            "se": se,
            "ci95": [estimate - 1.96 * se, estimate + 1.96 * se],
            "method": "normal-1.96SE-small-cluster-approximation",
        },
        "sensitivity_item_weighted": float(diff["diff"].mean()),
        "sensitivity_relation_group_weighted": float(
            by_relation.groupby("relation_group")["diff"].mean().mean()),
        "median_of_family_means": float(family.median()),
        "leave_one_family_out": lofo.to_dict("records"),
        "frozen_decision": (
            "unresolved at alpha=.05 under the declared family sign-flip "
            "test; audit does not change the frozen decision"
        ),
    }
    return block, family_rows


def p3p2_block(effects: pd.DataFrame) -> dict:
    q = effects[effects["model"] == "qwen36-27b"].copy()
    out = {}
    for threshold in THRESHOLDS:
        hd = (
            (q["J_eff"].to_numpy() < threshold).astype(float)
            - (q["C_eff"].to_numpy() < threshold).astype(float)
        )
        randomization = family_weighted_randomization(
            hd, q["canonical_family"].to_numpy())
        out[str(threshold)] = {
            **randomization,
            "effect_size_interval": effect_bootstrap(
                hd, q["canonical_family"].to_numpy()),
            "tail_rate_J": float((q["J_eff"] < threshold).mean()),
            "tail_rate_control": float((q["C_eff"] < threshold).mean()),
        }
    return out


def p3p3_block(effects: pd.DataFrame) -> dict | None:
    q = effects[
        (effects["model"] == "qwen36-27b")
        & effects.get("lp_true_bridge", pd.Series(
            index=effects.index, dtype=float)).notna()
    ].copy()
    if q.empty:
        return None
    d = (q["lp_true_bridge"] - q["lp_distractor_bridge"]).to_numpy()
    return {
        **family_weighted_randomization(
            d, q["canonical_family"].to_numpy()),
        "effect_size_interval": effect_bootstrap(
            d, q["canonical_family"].to_numpy()),
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, previous = {}, 0.0
    for i, key in enumerate(order):
        value = min(max(
            pvalues[key] * (len(order) - i), previous), 1.0)
        adjusted[key] = float(value)
        previous = value
    return adjusted


def main() -> None:
    require_clean_tree("--allow-dirty" in sys.argv)
    payload, family_tables, input_paths = {}, [], []
    for side in ("confirmatory", "replication"):
        effects, paths = load_effects(side)
        input_paths.extend(paths)
        p1, families = p3p1_block(effects, side)
        p2 = p3p2_block(effects)
        p3 = p3p3_block(effects)
        pvals = {
            "P3-P1": p1["exact_randomization"]["p"],
            "P3-P2": p2["-1.0"]["p_plus_one"],
        }
        if p3 is not None:
            pvals["P3-P3"] = p3["p_plus_one"]
        payload[side] = {
            "P3-P1": p1,
            "P3-P2_all_items_threshold_curve": p2,
            "P3-P3": p3,
            "holm_with_plus_one_mc": holm_adjust(pvals),
        }
        family_tables.append(families)
    payload["interpretation"] = {
        "primary_decision_ruler": "exact family sign-flip randomization",
        "intervals": (
            "Randomization-compatible, wild-cluster percentile-t, normal "
            "small-cluster approximation, and effect-size percentile "
            "intervals are named separately."
        ),
        "immutability": (
            "Frozen outcome parquets and rejection decisions were not changed."
        ),
    }
    out_dir = metrics_dir("cross_model") / "release_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "p3_inference_audit.json"
    family_out = out_dir / "p3_inference_audit_family_values.parquet"
    pd.concat(family_tables, ignore_index=True).to_parquet(
        family_out, index=False)
    cmd = "python -m jspace_phase3.experiments.p3_inference_audit"
    write_result3(payload, out, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd, seed=SEED,
        inputs={p: sha256_file(p) for p in input_paths}))
    register(
        EVIDENCE_ID, tier=TIER, command=cmd,
        what=(
            "Phase 3 inference closeout: exact P3-P1 sign flips, inverted "
            "randomization confidence set, correctly named percentile-t and "
            "normal intervals, plus-one P3-P2/P3-P3 Monte Carlo p-values"
        ),
        outputs=[out, family_out],
        inputs={p: sha256_file(p) for p in input_paths},
    )
    print(json.dumps({
        side: {
            "P3-P1": payload[side]["P3-P1"]["exact_randomization"],
            "P3-P1_randomization_ci": payload[side]["P3-P1"][
                "randomization_compatible_confidence_set"],
            "P3-P1_percentile_t": payload[side]["P3-P1"][
                "wild_cluster_percentile_t_interval"],
            "P3-P2_at_-1": payload[side][
                "P3-P2_all_items_threshold_curve"]["-1.0"],
            "P3-P3": payload[side]["P3-P3"],
        }
        for side in ("confirmatory", "replication")
    }, indent=1))


if __name__ == "__main__":
    main()

