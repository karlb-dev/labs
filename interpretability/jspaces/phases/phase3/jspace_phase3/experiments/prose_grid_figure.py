# Figure p3f04: Workstream C — the exact control on prose, cross-model.
#
# Panel A  NLL-per-token delta by guard domain, one facet per model;
#          arms are hues (label, span-safe, exact matched, prot-energy)
# Panel B  the §2.5 verdict: label-arm prose cost vs the exact matched
#          control's, per (model, domain) — Phase 2 never ran this pair
# Panel C  top-1 agreement with clean, by arm and model
# Panel D  grammar-pair preference rate by arm and model
# Panel E  §7.4 selectivity: standardized task effect (span-audit items)
#          minus standardized prose effect, per model, both components
#          shown (the index never replaces them)
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ..figures3 import PAL3, SHORT_MODEL, apply_style, save_fig
from ..paths3 import figures_dir, metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)

EVIDENCE_ID = "p3-prose-grid-figure-v1"
TIER = "phase3-development"
SLUGS = ["olmo31-think", "olmo31-instruct", "qwen36-27b"]
ARMS = [("meanJ_label_protected", "label", PAL3["J"]),
        ("meanJ_span_safe", "span-safe", PAL3["span_safe"]),
        ("instant_rank_energy_matched", "exact matched", PAL3["matched"]),
        ("prot_energy_matched", "prot-energy", PAL3["overlap_matched"])]
DIAG = [("mechanics_random", "mechanics"),
        ("logit_label_protected", "logit")]


def load(slug):
    d = metrics_dir(slug) / "prose_grid"
    df = pd.read_parquet(d / f"prose_grid_items_{slug}.parquet")
    gr = pd.read_parquet(d / f"prose_grid_grammar_{slug}.parquet")
    for a, _, _ in ARMS + [(x, None, None) for x, _ in DIAG]:
        df[f"delta_{a}"] = df[f"{a}__nll"] - df[f"{a}__nll_clean"]
    return df, gr


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    data = {s: load(s) for s in SLUGS}
    apply_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13.2, 8.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1],
                          hspace=0.5, wspace=0.35)

    domains = list(data[SLUGS[0]][0].domain.unique())
    stats: dict = {}

    # ---- A: per-domain NLL deltas, facets by model
    for j, slug in enumerate(SLUGS):
        df, _ = data[slug]
        ax = fig.add_subplot(gs[0, j])
        for k, (arm, lab, color) in enumerate(ARMS):
            g = df.groupby("domain")[f"delta_{arm}"].mean()
            vals = [g.get(d, np.nan) for d in domains]
            ax.plot(vals, np.arange(len(domains)) + (k - 1.5) * 0.16,
                    "o", ms=5, color=color, label=lab)
        ax.axvline(0, color=PAL3["muted"], lw=1)
        ax.set_yticks(range(len(domains)),
                      [d.replace("_", " ") for d in domains], fontsize=8)
        if j:
            ax.tick_params(labelleft=False)
        ax.invert_yaxis()
        ax.set_xlabel("Δ NLL per token (nats)")
        ax.set_title(f"A{j + 1} · {SHORT_MODEL[slug]}", loc="left")
        if j == 2:
            ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, axis="x", alpha=0.6)

    # ---- B: label vs exact matched, per (model, domain)
    ax = fig.add_subplot(gs[1, 0])
    markers = ["o", "s", "^"]
    for j, slug in enumerate(SLUGS):
        df, _ = data[slug]
        g = df.groupby("domain")[["delta_meanJ_label_protected",
                                  "delta_instant_rank_energy_matched"]].mean()
        ax.plot(g.iloc[:, 0], g.iloc[:, 1], markers[j], ms=6,
                color=PAL3["ink"], alpha=0.75, label=SHORT_MODEL[slug],
                markerfacecolor="none" if j else PAL3["ink"])
    lim = ax.get_xlim()
    ax.plot(lim, lim, "--", color=PAL3["muted"], lw=1, label="equal cost")
    ax.axhline(0, color=PAL3["grid"], lw=1)
    ax.set_xlabel("label-protected J prose cost (Δ NLL)")
    ax.set_ylabel("exact matched control (Δ NLL)")
    ax.set_title("B · The control Phase 2 never ran on prose", loc="left")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.6)

    # ---- C: top-1 agreement
    ax = fig.add_subplot(gs[1, 1])
    w = 0.19
    for k, (arm, lab, color) in enumerate(ARMS):
        vals = [data[s][0][f"{arm}__top1_agreement"].mean() for s in SLUGS]
        ax.bar(np.arange(3) + (k - 1.5) * w, vals, width=w * 0.92,
               color=color, edgecolor=PAL3["surface"], linewidth=1)
    ax.set_xticks(range(3), [SHORT_MODEL[s].split()[0] + "\n" +
                             " ".join(SHORT_MODEL[s].split()[1:])
                             for s in SLUGS], fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("top-1 agreement with clean")
    ax.set_title("C · Distributional damage", loc="left")
    ax.grid(True, axis="y", alpha=0.6)

    # ---- D: grammar preference
    ax = fig.add_subplot(gs[1, 2])
    for k, (arm, lab, color) in enumerate(ARMS):
        vals = [data[s][1][f"{arm}__prefers_good"].mean() for s in SLUGS]
        ax.bar(np.arange(3) + (k - 1.5) * w, vals, width=w * 0.92,
               color=color, edgecolor=PAL3["surface"], linewidth=1)
    base = [data[s][1]["baseline__prefers_good"].mean() for s in SLUGS]
    for j, b in enumerate(base):
        ax.plot([j - 0.42, j + 0.42], [b, b], color=PAL3["ink"], lw=1.4)
        ax.annotate("clean", (j + 0.30, b), fontsize=6.5,
                    color=PAL3["ink"], va="bottom")
    ax.set_xticks(range(3), [SHORT_MODEL[s].split()[0] for s in SLUGS],
                  fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("grammatical preferred")
    ax.set_title("D · Grammar minimal pairs", loc="left")
    ax.grid(True, axis="y", alpha=0.6)

    # ---- stats payload (incl. §7.4 selectivity components)
    for slug in SLUGS:
        df, gr = data[slug]
        s: dict = {}
        for arm, _, _ in ARMS + [(x, None, None) for x, _ in DIAG]:
            s[arm] = {
                "nll_delta": round(float(df[f"delta_{arm}"].mean()), 4),
                "kl": round(float(df[f"{arm}__kl_from_clean"].mean()), 4),
                "top1": round(float(df[f"{arm}__top1_agreement"].mean()), 4),
                "grammar_pref": round(float(
                    gr[f"{arm}__prefers_good"].mean()), 4)}
        # selectivity: standardized task effect minus standardized prose
        # effect (label arm), components carried per §7.4
        sa = pd.read_parquet(metrics_dir(slug) / "span_audit" /
                             f"span_audit_items_{slug}.parquet")
        task = sa.lp_meanJ_label_protected - sa.lp_baseline
        prose = df.delta_meanJ_label_protected
        s["selectivity"] = {
            "task_std_effect": round(float(task.mean() / task.std()), 4),
            "prose_std_effect": round(float(-prose.mean() / prose.std()), 4),
            "index": round(float(task.mean() / task.std()
                                 - (-prose.mean() / prose.std())), 4)}
        s["baseline_grammar_pref"] = round(float(
            gr["baseline__prefers_good"].mean()), 4)
        stats[slug] = s

    fig.suptitle("Workstream C · The exact dose control on prose — "
                 "guard battery v2 (104 items, 8 domains), three primaries",
                 fontsize=11, y=0.985)
    figs = save_fig(fig, figures_dir(), "p3f04_prose_exact_control", TIER)

    payload = {"models": stats, "domains": domains}
    cmd = "python -m jspace_phase3.experiments.prose_grid_figure"
    out = metrics_dir("cross_model") / "prose_grid_figure_stats.json"
    write_result3(payload, out, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd, seed=4242))
    register(EVIDENCE_ID, tier=TIER, command=cmd,
             what=("figure p3f04 + cross-model Workstream C statistics: "
                   "prose exact-control grid over the v2 guard battery"),
             outputs=[out, *figs])
    print(json.dumps(payload["models"], indent=1)[:2500])
    print(f"figure: {figs[0]}")


if __name__ == "__main__":
    main()
