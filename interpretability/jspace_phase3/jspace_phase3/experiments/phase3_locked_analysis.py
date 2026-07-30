# Block B — the LOCKED Phase 3 primary analysis (§14, prereg §1).
# Runs ONCE, from the three banked p3_grid parquets, after the last cell.
#
#   P3-P1  family sign-flip (>=100k) on the Qwen-minus-OLMo-pair-mean
#          within-fact composition difference (span-safe specific chain)
#   P3-P2  within-item J/control label-exchange tail test on Qwen
#          (span-safe vs its matched control, frozen -1.0-nat threshold)
#   P3-P3  within-item true/distractor bridge-protection exchange on the
#          preregistered model
#   Holm over the three. Heavy-tail reporting per §14.5; named
#   estimation targets (Think-vs-Instruct thick contrast, Bank S
#   composed-minus-direct per model) with wild-cluster CIs; item- and
#   relation-group-weighted sensitivities.
#
# Usage: python -m jspace_phase3.experiments.phase3_locked_analysis \
#            [--p3p3-model <slug>] [--eid p3-locked-analysis-v1]
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)
from ..stats import (family_cluster_bootstrap_ci, family_signflip_test,
                     leave_one_family_out, within_fact_composition,
                     within_fact_model_diff, within_item_exchange_mean,
                     within_item_label_exchange_tail,
                     wild_cluster_bootstrap_t)

TIER = "phase3-confirmatory"
SLUGS = ["olmo31-think", "olmo31-instruct", "qwen36-27b"]
THRESH = -1.0


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def load_effects() -> pd.DataFrame:
    rows = []
    for slug in SLUGS:
        df = pd.read_parquet(metrics_dir(slug) / "p3_grid" /
                             f"p3_grid_{slug}.parquet")
        df["model"] = slug
        df["J_eff"] = df.lp_meanJ_span_safe - df.lp_baseline
        df["C_eff"] = df.lp_ss_matched - df.lp_baseline
        df["specific"] = df.J_eff - df.C_eff
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def tail_block(d: pd.Series, fam: pd.Series) -> dict:
    """§14.5 heavy-tail reporting for one delta distribution."""
    fm = d.groupby(fam).mean()
    return {
        "mean": round(float(d.mean()), 4),
        "median": round(float(d.median()), 4),
        "family_weighted_mean": round(float(fm.mean()), 4),
        "tail_rate": round(float((d < THRESH).mean()), 4),
        "tail_conditional_mean": round(float(
            d[d < THRESH].mean()), 4) if (d < THRESH).any() else None,
        "threshold_curve": {str(t): round(float((d < t).mean()), 4)
                            for t in (-0.5, -1.0, -1.5, -2.0, -3.0)},
        "top_influential_families": {
            k: round(float(v), 3) for k, v in
            fm.sort_values().head(5).items()},
    }


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    p3p3_model = arg("--p3p3-model")
    eid = arg("--eid", "p3-locked-analysis-v1")
    eff = load_effects()

    # ---- P3-P1: within-fact composition, Qwen minus OLMo-pair mean
    comp = within_fact_composition(
        eff.rename(columns={"specific": "specific"}), value_col="specific")
    diff = within_fact_model_diff(
        comp, model_a="qwen36-27b",
        model_b=["olmo31-think", "olmo31-instruct"])
    fam_vals = diff.groupby("canonical_family")["diff"].mean().to_numpy()
    p1 = family_signflip_test(fam_vals, draws=100_000)
    p1_ci = wild_cluster_bootstrap_t(
        diff.rename(columns={"diff": "d"}), "d")

    # ---- P3-P2: Qwen span-safe vs matched tail
    q = eff[eff.model == "qwen36-27b"].copy()
    q["delta_J"], q["delta_C"] = q.J_eff, q.C_eff
    p2 = within_item_label_exchange_tail(q, draws=100_000,
                                         threshold=THRESH)

    # ---- P3-P3: bridge rescue exchange on the preregistered model
    p3 = None
    if p3p3_model:
        m = eff[(eff.model == p3p3_model)
                & eff.get("lp_true_bridge", pd.Series(dtype=float))
                .notna()].copy()
        if len(m):
            p3 = within_item_exchange_mean(
                m, a_col="lp_true_bridge", b_col="lp_distractor_bridge",
                draws=100_000, alternative="greater")

    # ---- Holm over the family
    ps = {"P3-P1": p1["p"], "P3-P2": p2["p"]}
    if p3:
        ps["P3-P3"] = p3["p"]
    order = sorted(ps, key=lambda k: ps[k])
    holm, prev = {}, 0.0
    for i, k in enumerate(order):
        adj = min(max(ps[k] * (len(ps) - i), prev), 1.0)
        holm[k] = round(adj, 6)
        prev = adj

    # ---- estimation targets + sensitivities
    ti = within_fact_model_diff(comp, model_a="olmo31-think",
                                model_b="olmo31-instruct")
    ti_ci = family_cluster_bootstrap_ci(
        ti.rename(columns={"diff": "d"}), "d")
    banks_comp = {}
    for slug in SLUGS:
        sub = eff[(eff.model == slug) & (eff.bank == "S")]
        if len(sub):
            c = within_fact_composition(sub, value_col="specific")
            banks_comp[slug] = family_cluster_bootstrap_ci(
                c.rename(columns={"composition_penalty": "d"}), "d")
    tails = {slug: tail_block(
        eff[eff.model == slug].specific,
        eff[eff.model == slug].canonical_family) for slug in SLUGS}
    lofo_df = leave_one_family_out(
        diff.rename(columns={"diff": "d"}), "d")
    lofo = {"min": lofo_df.head(3).to_dict("records"),
            "max": lofo_df.tail(3).to_dict("records")}
    item_weighted_p1 = round(float(diff["diff"].mean()), 4)
    rg = eff[["fact_id", "relation_group"]].drop_duplicates("fact_id")
    dz = diff.merge(rg, on="fact_id", how="left")
    rg_weighted_p1 = round(float(
        dz.groupby("relation_group")["diff"].mean().mean()), 4)

    payload = {
        "P3-P1": {"estimate_family_weighted": round(
            float(pd.Series(fam_vals).mean()), 4),
            "ci_wild_cluster": p1_ci, "p": p1["p"], "n_families":
            int(len(fam_vals)),
            "sensitivity_item_weighted": item_weighted_p1,
            "sensitivity_relation_group_weighted": rg_weighted_p1,
            "lofo": lofo},
        "P3-P2": p2,
        "P3-P3": p3,
        "holm": holm,
        "estimation_targets": {
            "think_vs_instruct_thick": ti_ci,
            "bank_s_composition_by_model": banks_comp},
        "heavy_tail_by_model": tails,
        "threshold_nats": THRESH,
    }
    cmd = ("python -m jspace_phase3.experiments.phase3_locked_analysis"
           + (f" --p3p3-model {p3p3_model}" if p3p3_model else ""))
    out = metrics_dir("cross_model") / "phase3_locked_analysis.json"
    write_result3(payload, out, Provenance3(
        evidence_id=eid, tier=TIER, command=cmd, seed=4242))
    register(eid, tier=TIER, command=cmd,
             what=(f"LOCKED Phase 3 primary analysis: P3-P1 p={p1['p']:.2g} "
                   f"holm={holm.get('P3-P1')}, P3-P2 p={p2['p']:.2g} "
                   f"holm={holm.get('P3-P2')}"
                   + (f", P3-P3 p={p3['p']:.2g}" if p3 else "")),
             outputs=[out])
    print(json.dumps({k: payload[k] for k in ("P3-P1", "P3-P2", "P3-P3",
                                              "holm")}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
