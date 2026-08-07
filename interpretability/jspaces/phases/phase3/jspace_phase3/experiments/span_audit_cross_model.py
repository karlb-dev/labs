# Figure p3f03: the §4.1b decomposition across all three primaries.
#
# Panel A  arm ladders, one facet per model (arms are hues, models are
#          facets per the figures3 convention), family dots +
#          family-clustered 95% CIs
# Panel B  tail rates at the frozen -1.0-nat threshold, model groups
# Panel C  the decomposition plane: leakage-dose effect
#          (prot-energy-matched) vs content residual (span-safe), one
#          labeled point per model with CI whiskers on both axes
# Panel D  leak dose per model: share of removed energy inside the
#          protected span under label protection (answer-direction norm
#          loss annotated per model)
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ..figures3 import PAL3, SHORT_MODEL, apply_style, save_fig
from ..paths3 import figures_dir, metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)
from ..stats import family_cluster_bootstrap_ci
from .span_audit_figure import ARMS

EVIDENCE_ID = "p3-span-audit-cross-model-v1"
TIER = "phase3-development"

SLUGS = ["olmo31-think", "olmo31-instruct", "qwen36-27b"]
TAIL_ARMS = ["lp_meanJ_label_protected", "lp_meanJ_span_safe",
             "lp_prot_energy_matched", "lp_instant_rank_energy_matched"]


def load_model(slug: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = metrics_dir(slug) / "span_audit"
    df = pd.read_parquet(d / f"span_audit_items_{slug}.parquet")
    ov = pd.read_parquet(d / f"span_audit_overlap_{slug}.parquet")
    for col, _, _ in ARMS:
        df[f"delta_{col}"] = df[col] - df.lp_baseline
    df["delta_ss_control"] = (df.lp_instant_rank_energy_matched_vs_span_safe
                              - df.lp_baseline)
    return df, ov


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    data = {slug: load_model(slug) for slug in SLUGS}

    stats: dict[str, dict] = {}
    for slug, (df, ov) in data.items():
        s: dict = {}
        for col, _, _ in ARMS:
            ci = family_cluster_bootstrap_ci(df, f"delta_{col}")
            s[col] = ci | {
                "tail_rate": round(float((df[f"delta_{col}"] < -1.0).mean()), 4)}
        s["span_safe_own_control"] = family_cluster_bootstrap_ci(
            df, "delta_ss_control") | {
            "tail_rate": round(float((df.delta_ss_control < -1.0).mean()), 4)}
        s["specific_tail_span_safe"] = round(
            s["lp_meanJ_span_safe"]["tail_rate"]
            - s["span_safe_own_control"]["tail_rate"], 4)
        s["per_item_corr_label_vs_span_safe"] = round(float(np.corrcoef(
            df.delta_lp_meanJ_label_protected,
            df.delta_lp_meanJ_span_safe)[0, 1]), 4)
        lab = ov[ov.arm == "meanJ_label_protected"]
        s["geometry_label_arm"] = {
            "removed_energy_in_prot_frac": round(
                float(lab.removed_energy_in_prot_frac.mean()), 4),
            "answer_dir_survival": round(
                float(lab.answer_dir_survival.mean()), 4),
            "projector_overlap": round(
                float(lab.projector_overlap.mean()), 4)}
        s["n_items"] = int(len(df))
        s["n_families"] = int(df.canonical_family.nunique())
        stats[slug] = s

    apply_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12.8, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1],
                          hspace=0.52, wspace=0.34)

    # ---- A: arm ladders, one facet per model
    fam_means = [data[s][0].groupby("canonical_family")[f"delta_{c}"].mean()
                 for s in SLUGS for c, _, _ in ARMS]
    ylo = min(m.min() for m in fam_means) - 0.4
    yhi = max(0.75, max(m.max() for m in fam_means) + 0.25)
    rng = np.random.default_rng(4242)
    for j, slug in enumerate(SLUGS):
        df, _ = data[slug]
        ax = fig.add_subplot(gs[0, j])
        for i, (col, _, color) in enumerate(ARMS):
            fam = df.groupby("canonical_family")[f"delta_{col}"].mean()
            ax.plot(rng.normal(i, 0.055, len(fam)), fam, ".", ms=4,
                    color=color, alpha=0.45)
            ci = stats[slug][col]
            ax.errorbar(i, ci["estimate"],
                        yerr=[[ci["estimate"] - ci["ci95"][0]],
                              [ci["ci95"][1] - ci["estimate"]]],
                        fmt="o", ms=6, color=color, lw=1.8, capsize=3.5,
                        zorder=3)
        # selective direct labels: the two decomposition arms only
        for i, col in ((0, "lp_meanJ_label_protected"),
                       (1, "lp_meanJ_span_safe"),
                       (2, "lp_prot_energy_matched")):
            ci = stats[slug][col]
            ax.annotate(f"{ci['estimate']:+.2f}", (i, ci["estimate"]),
                        textcoords="offset points", xytext=(9, -1),
                        fontsize=8, color=PAL3["ink"], va="center")
        ax.axhline(0, color=PAL3["muted"], lw=1)
        ax.axhline(-1, color=PAL3["grid"], lw=1)
        ax.set_xticks(range(len(ARMS)),
                      ["label", "span-\nsafe", "prot-\nenergy",
                       "overlap", "rank+\nenergy", "persist."],
                      fontsize=7.5)
        ax.set_ylim(ylo, yhi)
        if j == 0:
            ax.set_ylabel("mean paired Δ answer logprob (nats)")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f"A{j + 1} · {SHORT_MODEL[slug]}", loc="left")
        ax.grid(True, axis="y", alpha=0.6)

    # ---- B: tail rates, grouped by model
    ax = fig.add_subplot(gs[1, 0])
    h = 0.19
    arm_colors = {c: col for c, _, col in ARMS}
    for k, col in enumerate(TAIL_ARMS):
        vals = [stats[s][col]["tail_rate"] for s in SLUGS]
        ax.barh(np.arange(len(SLUGS)) + (k - 1.5) * h, vals, height=h * 0.92,
                color=arm_colors[col], edgecolor=PAL3["surface"],
                linewidth=1.2)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.0%}", (v, i + (k - 1.5) * h),
                        textcoords="offset points", xytext=(4, 0),
                        va="center", fontsize=7, color=PAL3["ink"])
    ax.set_yticks(range(len(SLUGS)),
                  [SHORT_MODEL[s].replace(" ", "\n", 1) for s in SLUGS],
                  fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("share of items losing > 1 nat")
    ax.set_title("B · Tail rate by arm", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=arm_colors[c])
               for c in TAIL_ARMS]
    ax.legend(handles, ["label", "span-safe", "prot-energy", "rank+energy"],
              fontsize=7, loc="lower right")
    ax.grid(True, axis="x", alpha=0.6)

    # ---- C: the decomposition plane
    ax = fig.add_subplot(gs[1, 1])
    for slug in SLUGS:
        cx = stats[slug]["lp_prot_energy_matched"]
        cy = stats[slug]["lp_meanJ_span_safe"]
        ax.errorbar(cx["estimate"], cy["estimate"],
                    xerr=[[cx["estimate"] - cx["ci95"][0]],
                          [cx["ci95"][1] - cx["estimate"]]],
                    yerr=[[cy["estimate"] - cy["ci95"][0]],
                          [cy["ci95"][1] - cy["estimate"]]],
                    fmt="o", ms=7, color=PAL3["ink"], lw=1.4, capsize=3)
        ax.annotate(SHORT_MODEL[slug], (cx["estimate"], cy["estimate"]),
                    textcoords="offset points", xytext=(7, 6), fontsize=8,
                    color=PAL3["ink"])
    ax.axhline(0, color=PAL3["grid"], lw=1)
    ax.axvline(0, color=PAL3["grid"], lw=1)
    ax.plot([-1.0, 0.4], [-1.0, 0.4], "--", color=PAL3["muted"], lw=1,
            label="equal components")
    ax.set_xlabel("leakage-dose effect: prot-energy-matched (nats)")
    ax.set_ylabel("content residual: span-safe (nats)")
    ax.set_title("C · Decomposition plane", loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.6)

    # ---- D: leak dose per model (label arm)
    ax = fig.add_subplot(gs[1, 2])
    vals = [stats[s]["geometry_label_arm"]["removed_energy_in_prot_frac"]
            for s in SLUGS]
    surv = [stats[s]["geometry_label_arm"]["answer_dir_survival"]
            for s in SLUGS]
    ax.bar(range(len(SLUGS)), vals, width=0.56, color=PAL3["J"],
           edgecolor=PAL3["surface"], linewidth=1.2)
    for i, (v, sv) in enumerate(zip(vals, surv)):
        ax.annotate(f"{v:.0%}", (i, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8.5,
                    color=PAL3["ink"])
        ax.annotate(f"answer dir\n−{1 - sv:.0%} norm", (i, 0.012),
                    ha="center", va="bottom", fontsize=7,
                    color=PAL3["surface"] if v > 0.12 else PAL3["ink"])
    ax.set_xticks(range(len(SLUGS)),
                  [SHORT_MODEL[s].replace(" ", "\n", 1) for s in SLUGS],
                  fontsize=8)
    ax.set_ylabel("removed energy inside protected span")
    ax.set_ylim(0, max(vals) * 1.3 + 0.02)
    ax.set_title("D · The leak under label protection", loc="left")
    ax.grid(True, axis="y", alpha=0.6)

    fig.suptitle("§4.1b Leakage vs content across the three primaries — "
                 "60 frozen Phase 2 items × 18 families each, own/published "
                 "lenses", fontsize=11, y=0.985)
    figs = save_fig(fig, figures_dir(), "p3f03_span_cross_model", TIER)

    payload = {"models": stats, "tail_threshold_nats": -1.0}
    cmd = "python -m jspace_phase3.experiments.span_audit_cross_model"
    out = metrics_dir("cross_model") / "span_audit_cross_model_stats.json"
    write_result3(payload, out, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd, seed=4242))
    register(EVIDENCE_ID, tier=TIER, command=cmd,
             what=("figure p3f03 + family-clustered cross-model arm "
                   "statistics for the §4.1b span audits (Think, Instruct, "
                   "Qwen): leakage-dose vs content decomposition"),
             outputs=[out, *figs])
    print(json.dumps({s: {k: v for k, v in st.items()
                          if not isinstance(v, dict)} | {
        "label": st["lp_meanJ_label_protected"]["estimate"],
        "span_safe": st["lp_meanJ_span_safe"]["estimate"],
        "prot_energy": st["lp_prot_energy_matched"]["estimate"]}
        for s, st in stats.items()}, indent=1))
    print(f"figure: {figs[0]}")


if __name__ == "__main__":
    main()
