#!/usr/bin/env python3
"""A3: Qwen fit ladder as a multi-object convergence study.

Reads tables/qwen_ladder_progression.csv (reconstructed, verified) and
renders the multilevel-convergence figure + synthesis. The x-axis shows
the three registered comparison boundaries; no convergence rate is
fitted or implied.

Outputs: figures/qwen_multilevel_convergence.{png,pdf},
tables/qwen_multilevel_convergence.csv, reports/QWEN_INSTRUMENT_SYNTHESIS.md
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

A = Path("/content/labs/interpretability/jspace_paper/analysis")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
BOUNDS = ["A120->A250", "A250->A500", "A500->A1000"]
XL = ["120→250", "250→500", "500→1000"]


def series(df, metric):
    s = df[df.metric == metric].set_index("fit_boundary").reindex(BOUNDS)
    return s.value.astype(float).values, s["pass"].values


def main():
    df = pd.read_csv(A / "tables/qwen_ladder_progression.csv")
    df[df.metric_class.isin(
        ["structural", "functional_sparse_geometry", "functional_causal_bridge",
         "functional_aggregate"])].to_csv(
        A / "tables/qwen_multilevel_convergence.csv", index=False)

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.size": 9.5, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 1.0,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlecolor": INK, "axes.titlesize": 10, "legend.frameon": False,
        "text.color": INK})

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.5))
    x = [0, 1, 2]

    def plot(ax, metric, color, label, marker_by_pass=True, dy=0):
        v, p = series(df, metric)
        ax.plot(x, v, color=color, lw=2, zorder=3)
        for xi, (vi, pi) in enumerate(zip(v, p)):
            ok = (pi is True) or (pi == True) or (str(pi) == "True")
            ax.plot(xi, vi, marker="o" if ok or not marker_by_pass else "X",
                    ms=9 if ok else 11, color=color, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.4)
        ax.annotate(label, (x[-1], v[-1]), xytext=(6, dy),
                    textcoords="offset points", fontsize=8.6, color=color,
                    va="center")

    # (a) structural task rows
    ax = axes[0]
    plot(ax, "task_direction_cosine_q50_conservative", BLUE, "q50", dy=7)
    plot(ax, "task_direction_cosine_q05_conservative", ORANGE, "q05", dy=-7)
    ax.axhline(0.95, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.axhline(0.90, color=MUTED, lw=1, ls=(0, (2, 3)))
    ax.text(0.02, 0.951, "gate q50 ≥ 0.95", fontsize=7.4, color=MUTED,
            va="bottom")
    ax.text(0.02, 0.901, "gate q05 ≥ 0.90", fontsize=7.4, color=MUTED,
            va="bottom")
    ax.set_ylim(0.885, 1.004)
    ax.set_title("Averaged operator:\ntask-row cosine (PASS ×3)")

    # (b) sparse geometry
    ax = axes[1]
    plot(ax, "selected_id_jaccard_median", BLUE, "ID Jaccard")
    plot(ax, "normalized_projector_overlap_median", ORANGE, "projector")
    ax.axhline(0.75, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.axhline(0.85, color=MUTED, lw=1, ls=(0, (2, 3)))
    ax.text(0.02, 0.752, "floor Jaccard ≥ 0.75", fontsize=7.4, color=MUTED,
            va="bottom")
    ax.text(0.02, 0.852, "floor overlap ≥ 0.85", fontsize=7.4, color=MUTED,
            va="bottom")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Sparse selection:\nID / projector (FAIL ×3)")

    # (c) bridge endpoints
    ax = axes[2]
    plot(ax, "bridge_rescue_difference_nats", BLUE, "rescue Δ")
    plot(ax, "bridge_preference_difference", ORANGE, "preference Δ")
    ax.axhspan(-0.25, 0.25, color=GRID, alpha=0.55, zorder=1)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.text(0.02, 0.215, "gate |Δ| ≤ 0.25 & same sign", fontsize=7.4,
            color=MUTED, va="top")
    ax.set_ylim(-0.42, 0.68)
    ax.set_title("Causal endpoints:\nrescue FAILs, preference PASSes")

    # (d) aggregates as fraction of gate
    ax = axes[3]
    agg = {
        "occupancy_difference_max_abs": ("occupancy", 1.0),
        "centered_excess_difference_max_abs_pp": ("centered excess", 1.0),
        "span_safe_specific_equal_family_mean_difference_nats":
            ("span-safe specific", 0.15),
        "tail_rate_difference": ("tail rate", 0.05),
        "g4_flip_rate_difference": ("G4 flip", 0.10),
    }
    colors = [BLUE, ORANGE, AQUA, "#4a3aa7", "#e87ba4"]
    label_dy = {"occupancy": -9, "centered excess": 4, "G4 flip": -4}
    for (metric, (label, gate)), c in zip(agg.items(), colors):
        v, _ = series(df, metric)
        frac = [abs(vi) / gate for vi in v]
        ax.plot(x, frac, color=c, lw=1.8, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax.annotate(label, (x[-1], frac[-1]),
                    xytext=(6, label_dy.get(label, 0)),
                    textcoords="offset points", fontsize=7.6, color=c,
                    va="center")
    ax.axhline(1.0, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.02, 1.02, "gate = 1.0", fontsize=7.4, color=RED, va="bottom")
    ax.set_ylim(0, 1.12)
    ax.set_title("Aggregate endpoints:\n|Δ| as fraction of gate (PASS ×15)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(XL, fontsize=8.4)
        ax.set_xlim(-0.25, 2.85)
        ax.set_xlabel("nested fit comparison (draw A)", fontsize=8.2)
    fig.suptitle("Qwen fit ladder: one operator, four levels of object — "
                 "convergence splits by level (no rate fitted)",
                 fontsize=11.5, y=1.04, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(A / f"figures/qwen_multilevel_convergence.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("figure + table written")


if __name__ == "__main__":
    main()
