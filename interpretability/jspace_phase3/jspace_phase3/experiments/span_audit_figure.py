# Figure p3f02: the §4.1b protected-span decomposition.
#
# Panel A  arm ladder — mean paired delta per arm with family-clustered
#          CIs and family dots (heavy-tail reporting rule §14.5)
# Panel B  tail rate per arm at the frozen -1.0-nat threshold
# Panel C  per-item label vs span-safe deltas (the arms hit DIFFERENT
#          items — the decomposition is not a rescaling)
# Panel D  geometry by layer: projector overlap and answer-direction
#          survival, label vs span-safe (span-safe is the positive
#          control: exactly 0 by construction)
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..figures3 import PAL3, apply_style, save_fig
from ..paths3 import figures_dir, metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)
from ..stats import family_cluster_bootstrap_ci

EVIDENCE_ID = "p3-span-audit-figure-olmo31-think-v1"
TIER = "phase3-development"

ARMS = [
    ("lp_meanJ_label_protected", "mean-J\nlabel-protected", PAL3["J"]),
    ("lp_meanJ_span_safe", "mean-J\nspan-safe", PAL3["span_safe"]),
    ("lp_prot_energy_matched", "control:\nprot-energy\nmatched",
     PAL3["overlap_matched"]),
    ("lp_overlap_matched", "control:\noverlap\nmatched", PAL3["persistent"]),
    ("lp_instant_rank_energy_matched", "control:\nrank+energy\nmatched",
     PAL3["matched"]),
    ("lp_persistent_matched", "control:\npersistent", PAL3["diag"]),
]


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    slug = "olmo31-think"
    d = metrics_dir(slug) / "span_audit"
    df = pd.read_parquet(d / f"span_audit_items_{slug}.parquet")
    ov = pd.read_parquet(d / f"span_audit_overlap_{slug}.parquet")

    for col, _, _ in ARMS:
        df[f"delta_{col}"] = df[col] - df.lp_baseline
    df["delta_ss_control"] = (df.lp_instant_rank_energy_matched_vs_span_safe
                              - df.lp_baseline)

    stats = {}
    for col, label, _ in ARMS:
        ci = family_cluster_bootstrap_ci(df, f"delta_{col}")
        stats[col] = ci | {
            "tail_rate": round(float((df[f"delta_{col}"] < -1.0).mean()), 4)}
    stats["span_safe_own_control"] = family_cluster_bootstrap_ci(
        df, "delta_ss_control")

    apply_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12.4, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1],
                          hspace=0.55, wspace=0.32)

    # ---- A: arm ladder
    ax = fig.add_subplot(gs[0, :2])
    rng = np.random.default_rng(4242)
    for i, (col, label, color) in enumerate(ARMS):
        fam = df.groupby("canonical_family")[f"delta_{col}"].mean()
        ax.plot(rng.normal(i, 0.055, len(fam)), fam, ".", ms=4.5,
                color=color, alpha=0.5)
        ci = stats[col]
        ax.errorbar(i, ci["estimate"],
                    yerr=[[ci["estimate"] - ci["ci95"][0]],
                          [ci["ci95"][1] - ci["estimate"]]],
                    fmt="o", ms=7, color=color, lw=2, capsize=4,
                    zorder=3)
        ax.annotate(f"{ci['estimate']:+.2f}", (i, ci["estimate"]),
                    textcoords="offset points", xytext=(11, 0),
                    fontsize=8.5, color=PAL3["ink"], va="center")
    ax.axhline(0, color=PAL3["muted"], lw=1)
    ax.set_xticks(range(len(ARMS)), [a[1] for a in ARMS], fontsize=8)
    ax.set_ylabel("mean paired Δ answer logprob (nats)")
    ax.set_title("A · Arm ladder — dots are canonical families, bars are "
                 "family-clustered 95% CIs", loc="left")
    ax.grid(True, axis="y", alpha=0.6)

    # ---- B: tail rates
    ax = fig.add_subplot(gs[0, 2])
    rates = [stats[c]["tail_rate"] for c, _, _ in ARMS]
    ax.barh(range(len(ARMS)), rates,
            color=[a[2] for a in ARMS], height=0.62)
    for i, r in enumerate(rates):
        ax.annotate(f"{r:.0%}", (r, i), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=8.5,
                    color=PAL3["ink"])
    # labels on the RIGHT: on the left they collide with panel A's
    # value annotations
    ax.yaxis.tick_right()
    ax.set_yticks(range(len(ARMS)),
                  [a[1].replace("\n", " ") for a in ARMS], fontsize=7.5)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.invert_yaxis()
    ax.set_xlim(0, max(rates) * 1.55)
    ax.set_xlabel("share of items losing > 1 nat")
    ax.set_title("B · Tail rate", loc="left")
    ax.grid(True, axis="x", alpha=0.6)

    # ---- C: per-item label vs span-safe
    ax = fig.add_subplot(gs[1, 0])
    x = df["delta_lp_meanJ_label_protected"]
    y = df["delta_lp_meanJ_span_safe"]
    ax.scatter(x, y, s=22, color=PAL3["span_safe"], alpha=0.8,
               edgecolors=PAL3["surface"], linewidths=0.6)
    lim = [min(x.min(), y.min()) - 0.4, 0.6]
    ax.plot(lim, lim, "--", color=PAL3["muted"], lw=1, label="y = x")
    ax.axhline(-1, color=PAL3["grid"], lw=1)
    ax.axvline(-1, color=PAL3["grid"], lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Δ label-protected (nats)")
    ax.set_ylabel("Δ span-safe (nats)")
    r = float(np.corrcoef(x, y)[0, 1])
    ax.set_title(f"C · Same items? r = {r:.2f}", loc="left")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.6)

    # ---- D: geometry by layer
    ax = fig.add_subplot(gs[1, 1])
    g = ov.groupby(["arm", "layer"])[
        ["projector_overlap", "removed_energy_in_prot_frac"]].mean()
    layers = sorted(ov.layer.unique())
    w = 0.34
    for k, (arm, color, lab) in enumerate((
            ("meanJ_label_protected", PAL3["J"], "label-protected"),
            ("meanJ_span_safe", PAL3["span_safe"], "span-safe"))):
        vals = [g.loc[(arm, l), "projector_overlap"] for l in layers]
        ax.bar(np.arange(len(layers)) + (k - 0.5) * w, vals, width=w,
               color=color, label=lab)
    ax.set_xticks(range(len(layers)), [f"L{l}" for l in layers])
    ax.set_ylabel("trace($P_J P_{prot}$)")
    ax.set_title("D · Selected-span overlap with the\nprotected span",
                 loc="left")
    ax.legend(fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.6)

    ax = fig.add_subplot(gs[1, 2])
    for k, (arm, color, lab) in enumerate((
            ("meanJ_label_protected", PAL3["J"], "label-protected"),
            ("meanJ_span_safe", PAL3["span_safe"], "span-safe"))):
        vals = [ov[(ov.arm == arm) & (ov.layer == l)]
                .answer_dir_survival.mean() for l in layers]
        ax.bar(np.arange(len(layers)) + (k - 0.5) * w, vals, width=w,
               color=color, label=lab)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (i + (k - 0.5) * w, v),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7, color=PAL3["ink"])
    ax.axhline(1.0, color=PAL3["muted"], lw=1)
    ax.set_ylim(0, 1.14)
    ax.set_xticks(range(len(layers)), [f"L{l}" for l in layers])
    ax.set_ylabel("‖(I−P) d$_{answer}$‖ / ‖d$_{answer}$‖")
    ax.set_title("E · Answer-direction survival", loc="left")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, axis="y", alpha=0.6)

    fig.suptitle("§4.1b Label protection is not span protection — "
                 "OLMo 3.1 Think, 60 frozen Phase 2 items, 18 families",
                 fontsize=11, y=0.985)
    figs = save_fig(fig, figures_dir(), "p3f02_span_decomposition", TIER)

    payload = {"arm_stats": stats,
               "per_item_corr_label_vs_span_safe": round(r, 4),
               "geometry_by_layer": json.loads(
                   g.round(6).reset_index().to_json(orient="records")),
               "n_items": int(len(df)),
               "n_families": int(df.canonical_family.nunique())}
    cmd = "python -m jspace_phase3.experiments.span_audit_figure"
    out = metrics_dir("cross_model") / "span_audit_figure_stats.json"
    write_result3(payload, out, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd, seed=4242))
    register(EVIDENCE_ID, tier=TIER, command=cmd,
             what=("figure p3f02 + family-clustered arm statistics for the "
                   "§4.1b span audit on olmo31-think"),
             outputs=[out, *figs])
    print(json.dumps(payload["arm_stats"], indent=1))
    print(f"figure: {figs[0]}")


if __name__ == "__main__":
    main()
