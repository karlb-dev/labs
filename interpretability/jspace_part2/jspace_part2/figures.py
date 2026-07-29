# VM8 repair-block figures, inside the package (nextsteps_2_2 §7-N1.1:
# executable code is authoritative here; the mirror is generated-only).
#
# Every figure is a pure function of REGISTERED metrics and carries a
# PILOT watermark (§11 / PI addendum §4.7) so no chart can circulate
# looking confirmatory. A figure whose inputs are missing skips quietly,
# so a reporting refresh never blocks a phase boundary.
#
# Palette continues the campaign's fixed entity->hue map; model identity
# is carried by position or texture, never by repainting an entity hue.
#
# Usage: python -m jspace_part2.figures
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")
FIGDIR = RUN / "figures"
M = RUN / "metrics"

PAL = {"J": "#2a78d6", "random": "#eb6834", "nonJ": "#1baf7a",
       "frozen_logit": "#8a5cf5", "ink": "#1f2430", "logit": "#5b6472",
       "muted": "#8a90a0", "grid": "#e4e6eb", "legacy": "#b9bec9",
       "warn": "#d1495b"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": PAL["muted"], "axes.labelcolor": PAL["ink"],
    "text.color": PAL["ink"], "xtick.color": PAL["ink"],
    "ytick.color": PAL["ink"], "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": PAL["grid"],
    "grid.linewidth": 0.8, "font.size": 9.5,
})

MODELS = ["olmo3-base", "olmo3-think", "olmo31-instruct", "qwen36-27b"]
SHORT = {"olmo3-base": "base", "olmo3-think": "Think",
         "olmo31-instruct": "Instruct", "qwen36-27b": "Qwen"}


def _load(p: Path):
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("payload", d)


def watermark(fig, text="PILOT"):
    fig.text(0.5, 0.5, text, fontsize=64, color="#000000", alpha=0.055,
             ha="center", va="center", rotation=28, weight="bold", zorder=0)


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    watermark(fig)
    fig.savefig(FIGDIR / name, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


# ------------------------------------------------------------------ f10
def f10_family_correction():
    """What the audited family map did to the pilot's uncertainty."""
    d = _load(M / "cross_model" / "n2_corrected_pilot.json")
    if not d:
        return
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 3.9),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    # Panel A — the twohop dynJ_protected interval, legacy vs corrected
    ys, labels = [], []
    for i, m in enumerate(MODELS):
        k = f"{m}/dynJ_protected/twohop"
        if k not in d["paired_ci_corrected"]:
            continue
        c, l = d["paired_ci_corrected"][k], d["paired_ci_legacy"][k]
        y = len(ys)
        ax.plot([l["ci_low"], l["ci_high"]], [y + 0.18] * 2, lw=5,
                color=PAL["legacy"], solid_capstyle="butt")
        ax.plot(l["estimate"], y + 0.18, "o", ms=5, color=PAL["legacy"])
        ax.plot([c["ci_low"], c["ci_high"]], [y - 0.18] * 2, lw=5,
                color=PAL["J"], solid_capstyle="butt")
        ax.plot(c["estimate"], y - 0.18, "o", ms=5, color=PAL["J"])
        ax.text(c["ci_low"] - 0.06, y - 0.18, f"{c['n_clusters']}f",
                ha="right", va="center", fontsize=7.5, color=PAL["J"])
        ax.text(l["ci_low"] - 0.06, y + 0.18, f"{l['n_clusters']}f",
                ha="right", va="center", fontsize=7.5, color=PAL["muted"])
        ys.append(y)
        labels.append(SHORT[m])
    ax.axvline(0, color=PAL["ink"], lw=0.9, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("paired protected dyn-J delta, two-hop (nats)")
    ax.set_title("A · corrected clusters widen the intervals",
                 loc="left", fontsize=10)
    ax.text(0.015, 0.03, "grey = defective prefix field   ·   "
            "blue = audited canonical families\n"
            "nf = number of independent clusters",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
            color=PAL["muted"])
    lo_ = min(v["ci_low"] for k, v in d["paired_ci_corrected"].items()
              if k.endswith("dynJ_protected/twohop"))
    ax.set_xlim(lo_ - 0.34, 0.24)
    ax.set_ylim(-0.75, len(ys) - 0.25)
    ax.grid(axis="x", alpha=0.6)

    # Panel B — ICC: the input G6 was calibrated on
    xs = np.arange(len(MODELS))
    old = [d["icc_legacy"][f"{m}/twohop"]["icc"] for m in MODELS]
    new = [d["icc_corrected"][f"{m}/twohop"]["icc"] for m in MODELS]
    bx.bar(xs - 0.19, old, 0.36, color=PAL["legacy"], label="prefix field",
           edgecolor="white")
    bx.bar(xs + 0.19, new, 0.36, color=PAL["J"], label="audited",
           edgecolor="white")
    for x, v in zip(xs - 0.19, old):
        bx.text(x, v + 0.012, f"{v:.2f}".lstrip("0"), ha="center", fontsize=7.5,
                color=PAL["muted"])
    for x, v in zip(xs + 0.19, new):
        bx.text(x, v + 0.012, f"{v:.2f}".lstrip("0"), ha="center", fontsize=7.5,
                color=PAL["J"])
    bx.set_xticks(xs)
    bx.set_xticklabels([SHORT[m] for m in MODELS])
    bx.set_ylabel("intraclass correlation")
    bx.set_ylim(0, max(max(old), max(new)) * 1.32)
    bx.set_title("B · ICC was never uniform", loc="left", fontsize=10)
    bx.legend(frameon=False, fontsize=8, loc="upper left", ncols=2)
    bx.grid(axis="y", alpha=0.6)
    fig.suptitle("The clustering unit was wrong, and it was load-bearing",
                 x=0.005, ha="left", fontsize=11.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "p2f10_family_correction.png")


# ------------------------------------------------------------------ f11
def f11_feasibility():
    """Minimum detectable effect vs the effect we actually expect."""
    d = _load(M / "cross_model" / "g6_mde.json")
    if not d:
        return
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 3.9))
    at60 = d["verdict"]["at_60_families"]

    for panel, key_mde, key_obs, unit, title in (
            (ax, "mde_rate_90", "pilot_gap", "rate difference",
             "A · tail-rate endpoint"),
            (bx, "mde_nats_90", "pilot_mean", "nats",
             "B · mean endpoint")):
        xs = np.arange(len(MODELS))
        mde = [at60[m][key_mde] or np.nan for m in MODELS]
        obs = [abs(at60[m][key_obs]) for m in MODELS]
        panel.bar(xs - 0.19, mde, 0.36, color=PAL["muted"],
                  label="smallest detectable at 90% power", edgecolor="white")
        panel.bar(xs + 0.19, obs, 0.36, color=PAL["J"],
                  label="effect the pilot suggests", edgecolor="white")
        for x, (m_, o_) in enumerate(zip(mde, obs)):
            if o_ >= m_:                       # testable
                panel.text(x + 0.19, o_ + max(obs) * 0.03, "testable",
                           ha="center", fontsize=7.5, color=PAL["nonJ"],
                           weight="bold")
            else:
                panel.text(x + 0.19, o_ + max(obs) * 0.03,
                           f"{m_ / o_:.1f}x short", ha="center", fontsize=7.5,
                           color=PAL["warn"])
        panel.set_xticks(xs)
        panel.set_xticklabels([SHORT[m] for m in MODELS])
        panel.set_ylabel(unit)
        panel.set_title(title, loc="left", fontsize=10)
        panel.set_ylim(0, max(max(mde), max(obs)) * 1.24)
        panel.grid(axis="y", alpha=0.6)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8.5, ncols=2,
               loc="upper left", bbox_to_anchor=(0.005, 0.935))
    fig.suptitle("At 60 canonical families, only two cells can carry a "
                 "binary test", x=0.005, ha="left", fontsize=11.5,
                 weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    _save(fig, "p2f11_feasibility.png")


# ------------------------------------------------------------------ f12
def f12_h7_ceiling():
    """Context-J does not rescue direction; the ceiling explains the gap."""
    ctx = _load(M / "olmo3-think" / "h7_context_j.json")
    ceil = _load(M / "olmo3-think" / "h7_ceiling.json")
    syn = _load(M / "olmo3-think" / "h7_synthesis.json")
    if not (ctx and ceil):
        return
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.2, 3.8))
    layers = sorted(int(L) for L in ctx["by_layer"])

    # A — estimator comparison
    mj = [ctx["by_layer"][str(L)]["campaign_meanJ"]["cos"] for L in layers]
    pj = [ctx["by_layer"][str(L)]["position_J_loo"]["cos"] for L in layers]
    xs = np.arange(len(layers))
    ax.bar(xs - 0.19, mj, 0.36, color=PAL["J"], label="campaign mean-J",
           edgecolor="white")
    ax.bar(xs + 0.19, pj, 0.36, color=PAL["frozen_logit"],
           label="position-conditioned (LOO)", edgecolor="white")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("response cosine")
    ax.set_title("A · conditioning does not help direction",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.6)

    # B — cosine vs measured linearity, per cell
    cells = ceil["cells"]
    for L, col in zip(layers, [PAL["J"], PAL["frozen_logit"], PAL["nonJ"],
                              PAL["logit"]]):
        sub = [c for c in cells if c["layer"] == L]
        bx.scatter([c["linearity_ratio"] for c in sub],
                   [c["cos_meanJ"] for c in sub], s=14, alpha=0.75,
                   color=col, label=f"L{L}", edgecolors="none")
    r = ceil["correlations"]["cos_meanJ"]["within_layer"]
    bx.set_xlabel("measured local linearity  r(2d)/r(d)   [2.0 = linear]")
    bx.set_ylabel("response cosine")
    bx.set_title("B · the ceiling, not the estimator", loc="left", fontsize=10)
    bx.text(0.02, 0.96, "within-layer Pearson r\n" +
            "  ".join(f"L{k} {v['r']:.2f}" for k, v in r.items()),
            transform=bx.transAxes, va="top", fontsize=7.5, color=PAL["ink"])
    bx.legend(frameon=False, fontsize=8, loc="lower right", ncols=4,
              handletextpad=0.3, columnspacing=0.9)
    bx.grid(alpha=0.6)

    # C — faithfulness where the ablation actually acts
    if syn:
        kinds = ["jrow", "random", "logit"]
        names = {"jrow": "selected J rows\n(where ablation acts)",
                 "random": "random directions", "logit": "unembedding rows"}
        cols = {"jrow": PAL["J"], "random": PAL["random"],
                "logit": PAL["logit"]}
        for k in kinds:
            ys = [syn["c_where_used"]["by_layer_kind"][str(L)][k]["cos_meanJ"]
                  for L in layers]
            cx.plot(range(len(layers)), ys, "o-", color=cols[k], lw=2, ms=5,
                    label=names[k])
        cx.set_xticks(range(len(layers)))
        cx.set_xticklabels([f"L{L}" for L in layers])
        cx.set_ylabel("response cosine")
        cx.set_title("C · most faithful where it is used", loc="left",
                     fontsize=10)
        cx.legend(frameon=False, fontsize=7.5, loc="upper left")
        cx.grid(alpha=0.6)
    fig.suptitle("D2 resolved: the averaged Jacobian's in-band gap is a "
                 "linearity ceiling, not an estimation failure",
                 x=0.005, ha="left", fontsize=11.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "p2f12_h7_ceiling.png")


# ------------------------------------------------------------------ f13
def f13_capacity_corrected():
    """The capacity estimand, before and after the repair."""
    d = _load(M / "olmo3-think" / "r2_occupancy" / "r2_occupancy_v2.json")
    if not d:
        return
    layers = sorted(int(L) for L in d["per_layer"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.8, 3.8))
    xs = np.arange(len(layers))
    raw = [d["per_layer"][str(L)]["raw_reconstruction_excess"] * 100
           for L in layers]
    cen = [d["per_layer"][str(L)]["centered_variance_explained_excess"] * 100
           for L in layers]
    lo = [d["per_layer"][str(L)]["centered_excess_ci"]["low"] * 100
          for L in layers]
    hi = [d["per_layer"][str(L)]["centered_excess_ci"]["high"] * 100
          for L in layers]
    ax.bar(xs - 0.19, raw, 0.36, color=PAL["legacy"],
           label="raw energy share (v1 label)", edgecolor="white")
    ax.bar(xs + 0.19, cen, 0.36, color=PAL["J"],
           label="centered $R^2$ (confirmatory)", edgecolor="white",
           yerr=[np.array(cen) - np.array(lo), np.array(hi) - np.array(cen)],
           ecolor=PAL["ink"], capsize=3)
    ax.axhspan(6, 10, color=PAL["random"], alpha=0.13, zorder=0)
    ax.text(len(layers) - 0.5, 8, "reported Claude band 6-10%", ha="right",
            va="center", fontsize=8, color=PAL["random"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("excess variance (%)")
    ax.set_ylim(0, 11)
    ax.set_title("A · the estimand was mislabelled — and is larger",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.6)

    med = [d["per_layer"][str(L)]["occ_median"] for L in layers]
    q25 = [d["per_layer"][str(L)]["occ_q25"] for L in layers]
    q75 = [d["per_layer"][str(L)]["occ_q75"] for L in layers]
    bx.errorbar(xs, med, yerr=[np.array(med) - np.array(q25),
                               np.array(q75) - np.array(med)],
                fmt="o", ms=7, color=PAL["J"], capsize=4, lw=2)
    bx.axhspan(10, 25, color=PAL["random"], alpha=0.13, zorder=0)
    bx.text(len(layers) - 0.5, 17, "reported Claude 10-25", ha="right",
            va="center", fontsize=8, color=PAL["random"])
    bx.set_xticks(xs)
    bx.set_xticklabels([f"L{L}" for L in layers])
    bx.set_ylabel("occupancy (median, IQR)")
    bx.set_ylim(0, 27)
    bx.set_title("B · occupancy is unchanged by the repair", loc="left",
                 fontsize=10)
    bx.grid(axis="y", alpha=0.6)
    fig.suptitle("Corrected capacity on OLMo-3-32B-Think", x=0.005,
                 ha="left", fontsize=11.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "p2f13_capacity_corrected.png")


# ------------------------------------------------------------------ f14
def f14_bank_readiness():
    """The item bank against the D5 target, and what the audit removed."""
    d = _load(M / "cross_model" / "g5_item_manifest.json")
    if not d:
        return
    fam = d["family_summary"]
    sizes = sorted((v["n_in_window"] for v in fam.values()), reverse=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 3.8),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    ax.bar(range(len(sizes)), sizes, color=PAL["J"], width=1.0,
           edgecolor="none")
    ax.axhline(3, color=PAL["warn"], lw=1.4, ls="--")
    n_ok = sum(1 for s in sizes if s >= 3)
    ax.axvline(n_ok - 0.5, color=PAL["ink"], lw=1.2, ls=":")
    ax.text(n_ok + 1, max(sizes) * 0.82,
            f"{n_ok} families with >=3\ncapable items\n(D5 target 60)",
            fontsize=8.5, color=PAL["ink"])
    ax.text(len(sizes) * 0.55, 3.5, "eligibility floor: 3 items",
            fontsize=8, color=PAL["warn"])
    ax.set_xlabel("canonical family (sorted)")
    ax.set_ylabel("items in the difficulty window")
    ax.set_title("A · bank depth per family", loc="left", fontsize=10)
    ax.grid(axis="y", alpha=0.6)

    g = d["gate"]
    bars = [("items", g["D_capability"]["n_items"]),
            ("greedy\ncapable", g["D_capability"]["n_greedy_capable"]),
            ("in difficulty\nwindow", g["D_capability"]["n_in_difficulty_window"]),
            ("excluded\n(leakage)", g["H_exclusions"]["n_excluded"])]
    cols = [PAL["muted"], PAL["logit"], PAL["J"], PAL["warn"]]
    bx.bar([b[0] for b in bars], [b[1] for b in bars], color=cols,
           edgecolor="white")
    for i, (_, v) in enumerate(bars):
        bx.text(i, v + 14, str(v), ha="center", fontsize=8.5)
    bx.set_ylabel("items")
    bx.set_title("B · G5 gate outcome", loc="left", fontsize=10)
    bx.grid(axis="y", alpha=0.6)
    fig.suptitle("Stage-3 bank after the G5 audit", x=0.005, ha="left",
                 fontsize=11.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "p2f14_bank_readiness.png")


# ------------------------------------------------------------------ f15
def f15_capacity_crossmodel():
    """Corrected centered-R2 capacity across the confirmatory models."""
    slugs = [("olmo31-think", "3.1-Think"), ("olmo31-instruct", "3.1-Instruct"),
             ("qwen36-27b", "Qwen3.6")]
    loaded = [(lab, _load(M / s / "r2_occupancy" / "r2_occupancy_v2.json"))
              for s, lab in slugs]
    loaded = [(lab, d) for lab, d in loaded if d]
    if not loaded:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    layers = sorted({int(L) for _, d in loaded for L in d["per_layer"]})
    w = 0.8 / len(loaded)
    for i, (lab, d) in enumerate(loaded):
        xs, cen, lo, hi, occ = [], [], [], [], []
        for j, L in enumerate(layers):
            e = d["per_layer"].get(str(L))
            if not e:
                continue
            xs.append(j + (i - (len(loaded) - 1) / 2) * w)
            cen.append(e["centered_variance_explained_excess"] * 100)
            lo.append(e["centered_excess_ci"]["low"] * 100)
            hi.append(e["centered_excess_ci"]["high"] * 100)
            occ.append(e.get("occ_median"))
        col = [PAL["J"], PAL["nonJ"], PAL["random"]][i % 3]
        ax.bar(xs, cen, w * 0.92, color=col, label=lab, edgecolor="white",
               yerr=[np.array(cen) - np.array(lo),
                     np.array(hi) - np.array(cen)],
               ecolor=PAL["ink"], capsize=2.5)
        for x, c, o in zip(xs, cen, occ):
            if o is not None:
                ax.text(x, c + 0.25, f"occ {o:g}", ha="center", fontsize=6.5,
                        color=PAL["muted"], rotation=90)
    ax.axhspan(6, 10, color=PAL["warn"], alpha=0.10, zorder=0)
    ax.text(len(layers) - 0.5, 8, "reported Claude band", ha="right",
            fontsize=8, color=PAL["warn"])
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("centered $R^2$ excess (%)")
    ax.set_title("Corrected capacity estimand across confirmatory models",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "p2f15_capacity_crossmodel.png")


# ------------------------------------------------------------------ f16
def f16_cohorts():
    """Capability-cohort composition after closing freeze condition 1."""
    d = _load(M / "cross_model" / "g5_item_manifest_v4.json")
    if not d or not d.get("cohort_counts"):
        return
    cc = d["cohort_counts"]
    labels = ["anchor-capable\n(3.1 pair)", "cross-model\nintersection",
              "3.1-Think", "3.1-Instruct", "Qwen3.6"]
    vals = [cc.get("lineage_anchor", 0), cc.get("cross_model_intersection", 0),
            cc.get("model_specific:olmo31-think", 0),
            cc.get("model_specific:olmo31-instruct", 0),
            cc.get("model_specific:qwen36-27b", 0)]
    cols = [PAL["J"], PAL["nonJ"], PAL["muted"], PAL["muted"], PAL["muted"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    xs = np.arange(len(vals))
    ax.bar(xs, vals, 0.62, color=cols, edgecolor="white")
    for x, v in zip(xs, vals):
        ax.text(x, v + 6, str(v), ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("items (non-excluded)")
    n3 = d.get("cross_model_intersection_families_ge3")
    ax.set_title(f"Capability cohorts closed — predicate: greedy generation "
                 f"on frozen aliases ({n3} intersection families ≥3 items)",
                 loc="left", fontsize=9.5)
    _save(fig, "p2f16_cohorts.png")


# ------------------------------------------------------------------ f17
def f17_confirmatory_primary():
    """THE confirmatory result: per model x task deltas for the primary
    arm vs the geometry-matched control, plus the two Holm tests."""
    d = _load(M / "cross_model" / "confirmatory_analysis.json")
    if not d:
        return
    est = d["estimates"]
    slugs = [("olmo31-think", "3.1-Think"), ("olmo31-instruct", "3.1-Instruct"),
             ("qwen36-27b", "Qwen3.6")]
    conds = [("meanJ_protected", PAL["J"], "protected mean-J"),
             ("matched_control", PAL["nonJ"], "matched control"),
             ("dynR_mechanics_control", PAL["legacy"], "mechanics random")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    for ax, task in zip(axes, ("twohop", "onehop")):
        xs = np.arange(len(slugs))
        for i, (cond, col, lab) in enumerate(conds):
            ys, lo, hi = [], [], []
            for s, _ in slugs:
                e = est.get(f"{s}/{cond}/{task}")
                ys.append(e["fam_weighted_mean"] if e else np.nan)
                lo.append(e["ci95_fam"][0] if e else np.nan)
                hi.append(e["ci95_fam"][1] if e else np.nan)
            x = xs + (i - 1) * 0.22
            ax.errorbar(x, ys, yerr=[np.array(ys) - np.array(lo),
                                     np.array(hi) - np.array(ys)],
                        fmt="o", color=col, capsize=3, ms=5,
                        label=lab if task == "twohop" else None)
        ax.axhline(0, color=PAL["muted"], lw=0.8)
        ax.axhline(-1.0, color=PAL["warn"], lw=0.8, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([l for _, l in slugs])
        ax.set_title(task, loc="left", fontsize=10)
        ax.grid(axis="y")
    axes[0].set_ylabel("Δ answer-seq logprob (nats, fam-weighted)")
    hp1 = d.get("P_HP1", {})
    holm = d.get("holm_family_A", {})
    fig.suptitle(
        f"CONFIRMATORY · HP1 contrast {hp1.get('observed_contrast_nats')} "
        f"nats CI {hp1.get('ci95')} · Holm "
        f"{ {k: v.get('p_holm') for k, v in holm.items()} }",
        fontsize=9, y=1.02)
    fig.legend(frameon=False, fontsize=8, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.06))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    watermark(fig, "CONFIRMATORY")
    fig.savefig(FIGDIR / "p2f17_confirmatory_primary.png", dpi=190,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote p2f17_confirmatory_primary.png")


FIGS = [f10_family_correction, f11_feasibility, f12_h7_ceiling,
        f13_capacity_corrected, f14_bank_readiness,
        f15_capacity_crossmodel, f16_cohorts, f17_confirmatory_primary]


def main():
    for f in FIGS:
        try:
            f()
        except Exception as e:      # a broken figure must never block a boundary
            print(f"FIGURE ERROR {f.__name__}: {e!r}")


if __name__ == "__main__":
    main()
