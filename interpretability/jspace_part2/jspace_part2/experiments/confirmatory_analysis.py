# N6/N8 — THE LOCKED CONFIRMATORY ANALYSIS (prereg §6), run from raw
# per-item parquets only after the final cell has banked.
#
# Primary Holm family (exactly two tests, prereg §6.1):
#   P-HP1  Think-vs-Instruct model-by-task interaction contrast on the
#          continuous logsumexp-alias delta, meanJ_protected vs baseline,
#          cross_model_intersection cohort, family-weighted.
#   P-HP3  Qwen paired tail-rate test at the frozen -1.0-nat threshold,
#          meanJ_protected vs matched_control, PROTECTED-ANSWER stratum
#          (clean first-token rank <= protect_k), family-clustered.
# Everything else is estimation with CIs.
#
# Estimand discipline (prereg §6.3): family-weighted (mean of family
# means) is PRIMARY; item-weighted reported as sensitivity. Clustering on
# canonical_family primary, relation_group sensitivity. Mixed model
# attempted first (statsmodels MixedLM); on convergence failure the
# declared fallback is the family-clustered paired bootstrap (4000 draws,
# seed 4242).
#
# Usage: python -m jspace_part2.experiments.confirmatory_analysis \
#          [--slugs olmo31-think,olmo31-instruct,qwen36-27b] [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result_v2)

RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")
PARTITION = RUN / "metrics" / "cross_model" / "partition_manifest.json"
THRESH = -1.0
SENS_THRESH = (-0.5, -1.5, -2.0)
N_BOOT, SEED = 4000, 4242
PROTECT_K = 10


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def load_deltas(slug: str) -> pd.DataFrame:
    pq = RUN / "metrics" / slug / "n6_grid" / f"n6_per_item_{slug}.parquet"
    df = pd.read_parquet(pq)
    base = df[df.condition == "baseline"].set_index("item_id")
    rows = []
    for cond in df.condition.unique():
        if cond == "baseline":
            continue
        sub = df[df.condition == cond].set_index("item_id")
        common = sub.index.intersection(base.index)
        for iid in common:
            b, s = base.loc[iid], sub.loc[iid]
            rows.append({
                "model": slug, "condition": cond, "item_id": iid,
                "task": b["task"], "family": b["canonical_family"],
                "relation_group": b["relation_group"],
                "cohorts_json": b.get("cohorts_json"),
                "delta": float(s["lp_logsumexp"] - b["lp_logsumexp"]),
                "delta_max": float(s["lp_max"] - b["lp_max"]),
                "delta_canonical": float(s["lp_canonical"] - b["lp_canonical"]),
                "baseline_lp": float(b["lp_logsumexp"]),
                "clean_first_rank_min": b.get("clean_first_rank_min"),
            })
    return pd.DataFrame(rows)


def fam_weighted_mean(df: pd.DataFrame, col="delta") -> float:
    return float(df.groupby("family")[col].mean().mean())


def item_weighted_mean(df: pd.DataFrame, col="delta") -> float:
    return float(df[col].mean())


def fam_boot(df: pd.DataFrame, stat_fn, n=N_BOOT, seed=SEED,
             strata_col="task") -> np.ndarray:
    """Family-clustered bootstrap, resampling families independently
    within each stratum (task), preserving within-item pairing because
    stat_fn consumes whole rows."""
    rng = np.random.default_rng(seed)
    out = []
    strata = {s: sorted(df[df[strata_col] == s]["family"].unique())
              for s in df[strata_col].unique()}
    grouped = {(s, f): g for (s, f), g in df.groupby([strata_col, "family"])}
    for _ in range(n):
        parts = []
        for s, fams in strata.items():
            pick = rng.choice(fams, size=len(fams), replace=True)
            parts.extend(grouped[(s, f)] for f in pick)
        out.append(stat_fn(pd.concat(parts)))
    return np.array(out)


def ci(boots):
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def p_two_sided(boots, observed):
    # percentile-bootstrap sign test around 0
    frac = float(np.mean(boots <= 0)) if observed > 0 else \
        float(np.mean(boots >= 0))
    return round(min(1.0, 2 * max(frac, 1.0 / len(boots))), 5)


def p_one_sided_le0(boots):
    return round(max(float(np.mean(boots <= 0)), 1.0 / len(boots)), 5)


def in_intersection(row) -> bool:
    try:
        c = json.loads(row["cohorts_json"])
        return bool(c and c.get("cross_model_intersection"))
    except Exception:
        return False


def hp1_contrast(df_pair: pd.DataFrame) -> float:
    """(twohop - onehop) on Think minus the same on Instruct,
    family-weighted, meanJ_protected deltas."""
    v = {}
    for m in ("olmo31-think", "olmo31-instruct"):
        sub = df_pair[df_pair.model == m]
        v[m] = (fam_weighted_mean(sub[sub.task == "twohop"])
                - fam_weighted_mean(sub[sub.task == "onehop"]))
    return v["olmo31-think"] - v["olmo31-instruct"]


def tail_stat(df: pd.DataFrame, thresh=THRESH) -> float:
    """Family-weighted mean of paired (hit_J - hit_C)."""
    d = df.copy()
    d["pd"] = (d.delta_J < thresh).astype(float) - \
        (d.delta_C < thresh).astype(float)
    return float(d.groupby("family")["pd"].mean().mean())


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    slugs = arg("--slugs",
                "olmo31-think,olmo31-instruct,qwen36-27b").split(",")
    dfs = pd.concat([load_deltas(s) for s in slugs], ignore_index=True)
    dfs = dfs[dfs.task.isin(["twohop", "onehop", "prose"])]
    core = dfs[dfs.task != "prose"]

    results = {"slugs": slugs, "n_delta_rows": int(len(dfs))}

    # ---------------- descriptive estimates: model x task x condition ----
    est = {}
    for (m, c, t), g in core.groupby(["model", "condition", "task"]):
        boots = fam_boot(g, fam_weighted_mean, strata_col="task")
        est[f"{m}/{c}/{t}"] = {
            "n_items": int(len(g)), "n_families": int(g.family.nunique()),
            "fam_weighted_mean": round(fam_weighted_mean(g), 4),
            "item_weighted_mean": round(item_weighted_mean(g), 4),
            "ci95_fam": ci(boots)}
    results["estimates"] = est

    # ---------------- P-HP1: interaction contrast ------------------------
    pair = core[(core.model.isin(["olmo31-think", "olmo31-instruct"]))
                & (core.condition == "meanJ_protected")]
    pair_int = pair[pair.apply(in_intersection, axis=1)]
    use = pair_int if len(pair_int) >= 30 else pair
    obs_hp1 = hp1_contrast(use)

    def hp1_boot_stat(d):
        return hp1_contrast(d)

    # resample families within (model, task) strata jointly: families are
    # shared across models (same items), so resample family labels once
    # per task stratum and apply to both models — preserves the pairing.
    rng = np.random.default_rng(SEED)
    fams_by_task = {t: sorted(use[use.task == t].family.unique())
                    for t in ("twohop", "onehop")}
    grouped = {(t, f): g for (t, f), g in use.groupby(["task", "family"])}
    hp1_boots = []
    for _ in range(N_BOOT):
        parts = []
        for t, fams in fams_by_task.items():
            for f in rng.choice(fams, size=len(fams), replace=True):
                parts.append(grouped[(t, f)])
        hp1_boots.append(hp1_contrast(pd.concat(parts)))
    hp1_boots = np.array(hp1_boots)
    p_hp1 = p_two_sided(hp1_boots, obs_hp1)

    mixed = None
    try:
        import statsmodels.formula.api as smf
        d = use.copy()
        d["is_think"] = (d.model == "olmo31-think").astype(int)
        d["is_twohop"] = (d.task == "twohop").astype(int)
        mm = smf.mixedlm("delta ~ is_think * is_twohop", d,
                         groups=d["family"]).fit(reml=True)
        mixed = {"interaction_coef": round(float(
            mm.params.get("is_think:is_twohop", np.nan)), 4),
            "interaction_p": round(float(
                mm.pvalues.get("is_think:is_twohop", np.nan)), 5),
            "converged": bool(mm.converged)}
    except Exception as e:
        mixed = {"error": str(e)[:200]}
    results["P_HP1"] = {
        "population": ("cross_model_intersection"
                       if use is pair_int else
                       "ALL partition items (intersection cohort <30 — "
                       "fallback, disclosed)"),
        "observed_contrast_nats": round(obs_hp1, 4),
        "ci95": ci(hp1_boots), "p_bootstrap": p_hp1,
        "mixed_model": mixed,
        "n_items": int(len(use)), "n_families": int(use.family.nunique())}

    # ---------------- P-HP3: Qwen paired tail-rate test ------------------
    def tail_frame(slug, stratified=True):
        q = core[core.model == slug]
        j = q[q.condition == "meanJ_protected"].set_index("item_id")
        c = q[q.condition == "matched_control"].set_index("item_id")
        common = j.index.intersection(c.index)
        d = pd.DataFrame({
            "delta_J": j.loc[common, "delta"],
            "delta_C": c.loc[common, "delta"],
            "family": j.loc[common, "family"],
            "task": j.loc[common, "task"],
            "rank": j.loc[common, "clean_first_rank_min"]})
        if stratified:
            d = d[d["rank"].notna() & (d["rank"] <= PROTECT_K)]
        return d

    hp3 = {}
    for slug in slugs:
        strat = tail_frame(slug, stratified=True)
        alli = tail_frame(slug, stratified=False)
        if len(strat) < 5:
            hp3[slug] = {"n_stratified": int(len(strat)),
                         "note": "stratum too small"}
            continue
        obs = tail_stat(strat)
        boots = fam_boot(strat, tail_stat, strata_col="task")
        entry = {
            "n_stratified": int(len(strat)),
            "n_families": int(strat.family.nunique()),
            "rate_diff_fam_weighted": round(obs, 4), "ci95": ci(boots),
            "p_one_sided": p_one_sided_le0(boots),
            "all_items_sensitivity": {
                "n": int(len(alli)),
                "rate_diff": round(tail_stat(alli), 4)},
            "threshold_sensitivity": {
                str(t): round(tail_stat(strat, t), 4) for t in SENS_THRESH}}
        hp3[slug] = entry
    results["HP3_tail"] = hp3
    results["P_HP3_qwen"] = hp3.get("qwen36-27b")

    # ---------------- Holm over the two primary tests --------------------
    ps = {"P_HP1": results["P_HP1"]["p_bootstrap"]}
    if results.get("P_HP3_qwen") and "p_one_sided" in results["P_HP3_qwen"]:
        ps["P_HP3"] = results["P_HP3_qwen"]["p_one_sided"]
    order = sorted(ps, key=lambda k_: ps[k_])
    holm, sig = {}, True
    for i, k_ in enumerate(order):
        adj = min(1.0, ps[k_] * (len(ps) - i))
        sig = sig and adj < 0.05
        holm[k_] = {"p_raw": ps[k_], "p_holm": round(adj, 5),
                    "reject_at_05": bool(sig and adj < 0.05)}
    results["holm_family_A"] = holm

    # ---------------- HP2: accessibility estimate (per model) ------------
    hp2 = {}
    for slug in slugs:
        one = core[(core.model == slug) & (core.task == "onehop")
                   & (core.condition == "meanJ_protected")].copy()
        if len(one) < 10:
            continue
        terc = one.baseline_lp.quantile([1 / 3, 2 / 3]).values
        one["stratum"] = np.where(one.baseline_lp <= terc[0], "hard",
                                  np.where(one.baseline_lp <= terc[1],
                                           "medium", "easy"))
        hp2[slug] = {
            s: {"n": int(len(g)),
                "fam_weighted_delta": round(fam_weighted_mean(g), 4)}
            for s, g in one.groupby("stratum")}
        r = np.corrcoef(one.baseline_lp, one.delta)[0, 1]
        hp2[slug]["pearson_baseline_vs_delta"] = round(float(r), 4)
    results["HP2_accessibility"] = hp2

    # ---------------- hurdle secondary -----------------------------------
    hurdle = {}
    for (m, t), g in core[core.condition == "meanJ_protected"]\
            .groupby(["model", "task"]):
        tail = g[g.delta < THRESH]
        hurdle[f"{m}/{t}"] = {
            "p_tail": round(float(len(tail)) / len(g), 4),
            "mean_magnitude_given_tail": (round(float(tail.delta.mean()), 4)
                                          if len(tail) else None)}
    results["hurdle_secondary"] = hurdle

    # ---------------- prose guard ----------------------------------------
    prose = dfs[dfs.task == "prose"]
    results["prose_guard_nll_increase"] = {
        f"{m}/{c}": round(-float(g.delta.mean()), 4)
        for (m, c), g in prose.groupby(["model", "condition"])}

    # ---------------- sensitivity: relation_group clustering -------------
    sens_rel = {}
    for (m, t), g in pair[pair.condition == "meanJ_protected"]\
            .groupby(["model", "task"]):
        g2 = g.copy()
        g2["family"] = g2["relation_group"].fillna(g2["family"])
        sens_rel[f"{m}/{t}"] = round(fam_weighted_mean(g2), 4)
    results["sensitivity_relation_group"] = sens_rel

    out = RUN / "metrics" / "cross_model" / "confirmatory_analysis.json"
    prov = Provenance(
        evidence_id="n6-confirmatory-analysis-v1", tier="confirmatory",
        command=("python -m jspace_part2.experiments.confirmatory_analysis "
                 f"--slugs {','.join(slugs)}"),
        inputs={f"parquet_{s}": sha256_file(
            RUN / "metrics" / s / "n6_grid" / f"n6_per_item_{s}.parquet")
            for s in slugs} | {"partition": sha256_file(PARTITION)},
        model={"note": "analysis step over banked parquets"}, seed=SEED)
    write_result_v2(results, out, prov)
    registry_append({
        "evidence_id": "n6-confirmatory-analysis-v1", "tier": "confirmatory",
        "what": (f"LOCKED confirmatory analysis: P-HP1 contrast "
                 f"{results['P_HP1']['observed_contrast_nats']} nats "
                 f"CI {results['P_HP1']['ci95']} p={p_hp1}; P-HP3 Qwen "
                 f"{json.dumps(results.get('P_HP3_qwen'))[:160]}; Holm "
                 f"{json.dumps(holm)}"),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(out), "sha256": sha256_file(out)}]})
    print(json.dumps({k: results[k] for k in
                      ("P_HP1", "P_HP3_qwen", "holm_family_A")}, indent=1))


if __name__ == "__main__":
    main()
