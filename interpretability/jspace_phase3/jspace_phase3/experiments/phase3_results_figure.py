# Figure p3f05 — the Phase 3 primary results (Block B, locked).
#
# Panel A  the preregistered family: P3-P1 / P3-P2 / P3-P3 estimates
#          with CIs and Holm-adjusted p-values
# Panel B  within-fact composition penalty (span-safe specific chain)
#          per model — family dots + family-clustered CIs
# Panel C  Bank F vs Bank S composition per model — the §4.3
#          workspace-word panel (working memory vs parametric recall)
# Panel D  §14.5 threshold curves per model (specific effect)
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ..figures3 import PAL3, SHORT_MODEL, apply_style, save_fig
from ..paths3 import figures_dir, metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)
from ..stats import (family_cluster_bootstrap_ci, within_fact_composition)

EVIDENCE_ID = "p3-results-figure-v2"
SUPERSEDES = "p3-results-figure-v1"  # v1: panel-A annotation/title collision
TIER = "phase3-confirmatory"
SLUGS = ["olmo31-think", "olmo31-instruct", "qwen36-27b"]


def load_effects():
    rows = []
    for slug in SLUGS:
        df = pd.read_parquet(metrics_dir(slug) / "p3_grid" /
                             f"p3_grid_{slug}.parquet")
        df["model"] = slug
        df["specific"] = (df.lp_meanJ_span_safe - df.lp_baseline) \
            - (df.lp_ss_matched - df.lp_baseline)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    locked = json.loads((metrics_dir("cross_model") /
                         "phase3_locked_analysis.json").read_text())["payload"]
    eff = load_effects()
    comp = within_fact_composition(eff, value_col="specific")

    apply_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13.0, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05],
                          hspace=0.55, wspace=0.36)

    # ---- A: the primary family
    ax = fig.add_subplot(gs[0, 0])
    p1 = locked["P3-P1"]
    ci = p1["ci_wild_cluster"]
    lo = ci["estimate"] - 1.96 * ci["se"]
    hi = ci["estimate"] + 1.96 * ci["se"]
    entries = [
        ("P3-P1\nthick contrast\n(nats)", p1["estimate_family_weighted"],
         (lo, hi), locked["holm"]["P3-P1"], PAL3["J"]),
        ("P3-P2\nspan-safe tail\n(rate)", locked["P3-P2"]["estimate"],
         None, locked["holm"]["P3-P2"], PAL3["span_safe"]),
        ("P3-P3\nbridge rescue\n(nats)", locked["P3-P3"]["estimate"],
         None, locked["holm"]["P3-P3"], PAL3["overlap_matched"]),
    ]
    for i, (lab, est, ci_, ph, color) in enumerate(entries):
        if ci_:
            ax.plot([ci_[0], ci_[1]], [i, i], color=color, lw=2)
        ax.plot([est], [i], "o", ms=9, color=color)
        ax.annotate(f"{est:+.3f}   p₍holm₎={max(ph, 1e-5):.3g}",
                    (est, i), textcoords="offset points",
                    xytext=(0, -18 if i == 0 else 12),
                    ha="center", fontsize=8.5, color=PAL3["ink"])
    ax.axvline(0, color=PAL3["muted"], lw=1)
    ax.set_yticks(range(3), [e[0] for e in entries], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("A · The preregistered family", loc="left")
    ax.grid(True, axis="x", alpha=0.6)

    # ---- B: within-fact composition penalty per model
    ax = fig.add_subplot(gs[0, 1:])
    rng = np.random.default_rng(4242)
    stats_b = {}
    for j, slug in enumerate(SLUGS):
        sub = comp[comp.model == slug]
        fam = sub.groupby("canonical_family").composition_penalty.mean()
        ax.plot(rng.normal(j, 0.06, len(fam)), fam, ".", ms=5,
                color=PAL3["J"], alpha=0.45)
        ci = family_cluster_bootstrap_ci(
            sub.rename(columns={"composition_penalty": "d"}), "d")
        stats_b[slug] = ci
        ax.errorbar(j, ci["estimate"],
                    yerr=[[ci["estimate"] - ci["ci95"][0]],
                          [ci["ci95"][1] - ci["estimate"]]],
                    fmt="o", ms=8, color=PAL3["ink"], lw=2, capsize=4,
                    zorder=3)
        ax.annotate(f"{ci['estimate']:+.2f}", (j, ci["estimate"]),
                    textcoords="offset points", xytext=(12, 0),
                    fontsize=9, color=PAL3["ink"], va="center")
    ax.axhline(0, color=PAL3["muted"], lw=1)
    ax.set_xticks(range(3), [SHORT_MODEL[s] for s in SLUGS], fontsize=9)
    ax.set_ylabel("within-fact composition penalty\n(span-safe specific, nats)")
    ax.set_title("B · Composed-vs-direct specific damage — dots are "
                 "families", loc="left")
    ax.grid(True, axis="y", alpha=0.6)

    # ---- C: Bank F vs Bank S per model
    ax = fig.add_subplot(gs[1, 0:2])
    w = 0.35
    stats_c = {}
    for j, slug in enumerate(SLUGS):
        for b, (bank, color) in enumerate((("F", PAL3["J"]),
                                           ("S", PAL3["span_safe"]))):
            sub = comp[(comp.model == slug)
                       & comp.fact_id.str.startswith("s_" if bank == "S"
                                                     else "")]
            if bank == "F":
                sub = comp[(comp.model == slug)
                           & ~comp.fact_id.str.startswith("s_")]
            if not len(sub):
                continue
            ci = family_cluster_bootstrap_ci(
                sub.rename(columns={"composition_penalty": "d"}), "d")
            stats_c[f"{slug}_{bank}"] = ci
            x = j + (b - 0.5) * w
            ax.bar(x, ci["estimate"], width=w * 0.9, color=color,
                   edgecolor=PAL3["surface"], linewidth=1)
            ax.errorbar(x, ci["estimate"],
                        yerr=[[ci["estimate"] - ci["ci95"][0]],
                              [ci["ci95"][1] - ci["estimate"]]],
                        fmt="none", ecolor=PAL3["ink"], lw=1.3, capsize=3)
    ax.axhline(0, color=PAL3["muted"], lw=1)
    ax.set_xticks(range(3), [SHORT_MODEL[s] for s in SLUGS], fontsize=9)
    ax.set_ylabel("composition penalty (nats)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PAL3["J"]),
               plt.Rectangle((0, 0), 1, 1, color=PAL3["span_safe"])]
    ax.legend(handles, ["Bank F (parametric facts)",
                        "Bank S (in-context / working memory)"],
              fontsize=8, loc="lower left")
    ax.set_title("C · The workspace-word panel: parametric recall vs "
                 "working memory (§4.3)", loc="left")
    ax.grid(True, axis="y", alpha=0.6)

    # ---- D: threshold curves
    ax = fig.add_subplot(gs[1, 2])
    markers = ["o", "s", "^"]
    for j, slug in enumerate(SLUGS):
        tc = locked["heavy_tail_by_model"][slug]["threshold_curve"]
        xs = sorted(float(t) for t in tc)
        ys = [tc[str(t)] for t in xs]
        ax.plot(xs, ys, marker=markers[j], ms=5, lw=1.6,
                color=PAL3["ink"], alpha=0.45 + 0.25 * j,
                label=SHORT_MODEL[slug])
    ax.set_xlabel("threshold (nats)")
    ax.set_ylabel("share of items below")
    ax.set_title("D · Specific-effect threshold curves", loc="left")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.6)

    fig.suptitle("Phase 3 Block B — locked primary results "
                 "(span-safe arm, frozen partition, Holm family)",
                 fontsize=11, y=0.985)
    figs = save_fig(fig, figures_dir(), "p3f05_primary_results", TIER)

    payload = {"panel_b": stats_b, "panel_c": stats_c,
               "locked_ref": "p3-locked-analysis-v1"}
    cmd = "python -m jspace_phase3.experiments.phase3_results_figure"
    out = metrics_dir("cross_model") / "p3f05_stats.json"
    write_result3(payload, out, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd, seed=4242))
    register(EVIDENCE_ID, tier=TIER, command=cmd, supersedes=SUPERSEDES,
             what="figure p3f05: locked Phase 3 primary results panels",
             outputs=[out, *figs])
    print(json.dumps(payload, indent=1)[:800])
    print(f"figure: {figs[0]}")


if __name__ == "__main__":
    main()
