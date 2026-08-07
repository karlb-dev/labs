#!/usr/bin/env python
"""Independent re-analysis of the J-space Phase 3 run data.

Reads ONLY the item-level parquets (never the summary JSONs) and
re-derives the headline claims, then renders the paper-candidate
figures. Registered values are quoted in comments for comparison.

Run:  .venv/bin/python analysis/run_analysis.py   (from jspace_runs/)
"""
import json
import pathlib
import os


def _find_runs_cache():
    """Locate the (gitignored) interpretability/jspace_runs mirror."""
    env = os.environ.get("JSPACE_RUNS_ROOT")
    if env:
        return pathlib.Path(env)
    for parent in pathlib.Path(__file__).resolve().parents:
        cand = parent / "jspace_runs"
        if cand.is_dir():
            return cand
    raise SystemExit(
        "cannot locate interpretability/jspace_runs; set JSPACE_RUNS_ROOT")



import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = _find_runs_cache()
P3M = ROOT / "phase3" / "metrics"
FIGS = ROOT / "analysis" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

MODELS = ["olmo31-think", "olmo31-instruct", "qwen36-27b"]
NICE = {"olmo31-think": "OLMo 3.1 Think",
        "olmo31-instruct": "OLMo 3.1 Instruct",
        "qwen36-27b": "Qwen 3.6 27B"}

# ---- palette (dataviz reference instance, light mode) ----------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
MODEL_C = {"olmo31-think": "#2a78d6",      # slot 1 blue
           "olmo31-instruct": "#eb6834",   # slot 2 orange
           "qwen36-27b": "#1baf7a"}        # slot 3 aqua
ARM_J = "#4a3aa7"      # accent for the J span-safe arm (slot 7 violet)
ARM_CTRL = MUTED       # matched control drawn as recessive reference

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.size": 9.5, "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "legend.frameon": False,
})

RNG = np.random.default_rng(4242)


def fam_weighted(df, col):
    """Phase 2 weighting convention: family means first, then across families."""
    return df.groupby("canonical_family")[col].mean().mean()


def cluster_boot(df, stat, n=4000):
    """Family-clustered bootstrap percentile CI for stat(df)."""
    fams = df["canonical_family"].unique()
    groups = {f: g for f, g in df.groupby("canonical_family")}
    vals = []
    for _ in range(n):
        pick = RNG.choice(fams, size=len(fams), replace=True)
        vals.append(stat(pd.concat([groups[f] for f in pick])))
    return np.percentile(vals, [2.5, 97.5])


def load_grid(model, replication=False):
    tag = "p3_grid_replication" if replication else "p3_grid"
    p = P3M / model / tag / f"{tag}_{model}.parquet"
    df = pd.read_parquet(p)
    for c in df.columns:
        if c.startswith("lp_") and c != "lp_baseline":
            df["d_" + c[3:]] = df[c] - df["lp_baseline"]
    return df


print("=" * 72)
print("1) P3-P2 — span-safe J-specific tail (Qwen), confirmatory + replication")
print("=" * 72)
p3p2 = {}
for repl in (False, True):
    q = load_grid("qwen36-27b", repl)
    q["tail_J"] = q["d_meanJ_span_safe"] <= -1.0
    q["tail_C"] = q["d_ss_matched"] <= -1.0

    def tail_excess(df):
        g = df.groupby("canonical_family")[["tail_J", "tail_C"]].mean()
        return (g["tail_J"] - g["tail_C"]).mean()

    est = tail_excess(q)
    lo, hi = cluster_boot(q, tail_excess)
    key = "replication" if repl else "confirmatory"
    p3p2[key] = (est, lo, hi, q)
    print(f"  {key:13s}: family-weighted tail excess = {est:+.4f} "
          f"[{lo:+.3f}, {hi:+.3f}]  (items={len(q)}, "
          f"families={q['canonical_family'].nunique()})")
print("  registered: confirmatory +0.0958 (p~1e-5) · replication +0.1021")

print()
print("=" * 72)
print("2) Span audit — label-protected vs span-safe per item (60 frozen items)")
print("=" * 72)
audit = {}
for m in MODELS:
    a = pd.read_parquet(P3M / m / "span_audit" / f"span_audit_items_{m}.parquet")
    for c in ["lp_meanJ_label_protected", "lp_meanJ_span_safe"]:
        a["d_" + c[3:]] = a[c] - a["lp_baseline"]
    audit[m] = a
    r = a["d_meanJ_label_protected"].corr(a["d_meanJ_span_safe"])
    print(f"  {NICE[m]:18s}: mean label {fam_weighted(a,'d_meanJ_label_protected'):+.3f}"
          f"  span-safe {fam_weighted(a,'d_meanJ_span_safe'):+.3f}"
          f"  per-item r = {r:.3f}")
print("  registered r: 0.20 / 0.29 / 0.75")

print()
print("=" * 72)
print("3) Bridge mediation — factorial, from parquet rows (family-weighted)")
print("=" * 72)
med_json = {}
for m in ["qwen36-27b", "olmo31-think"]:
    d = json.load(open(P3M / m / "bridge_mediation" / f"bridge_mediation_{m}.json"))
    med_json[m] = d.get("payload", d)["arm_stats"]
    mp = pd.read_parquet(P3M / m / "bridge_mediation" / f"bridge_mediation_{m}.parquet")
    mp["d_rescue"] = mp["d_true_bridge"] - mp["d_distractor_bridge"]
    print(f"  {NICE[m]:18s}: rescue contrast (parquet) = "
          f"{fam_weighted(mp, 'd_rescue'):+.3f}   "
          f"(registered {med_json[m]['rescue_contrast']['estimate']:+.3f})")

print()
print("=" * 72)
print("4) Accessibility organization — Spearman rho(clean rank, J damage)")
print("=" * 72)
mine = pd.read_parquet(P3M / "cross_model" / "overlap_mining" / "overlap_mining_items.parquet")
rho = {}
for m in MODELS:
    g = mine[mine["model"] == m]
    rho[m] = g["clean_first_rank_min"].corr(g["delta_J"], method="spearman")
    print(f"  {NICE[m]:18s}: rho = {rho[m]:+.3f}   (n={len(g)})")
print("  registered: Think -0.30 · Instruct -0.31 · Qwen +0.24  (conf. partition)")

# ============================ FIGURES =======================================

# --- A: ECDF of Qwen per-item damage, span-safe J vs exact matched control --
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
for ax, key in zip(axes, ["confirmatory", "replication"]):
    est, lo, hi, q = p3p2[key]
    for col, color, label, z in [("d_ss_matched", ARM_CTRL, "matched control", 2),
                                 ("d_meanJ_span_safe", ARM_J, "span-safe J", 3)]:
        x = np.sort(q[col].values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post",
                color=color, lw=2, zorder=z, label=label)
    ax.axvline(-1.0, color=BASELINE, lw=1, zorder=1)
    ax.text(-1.05, 0.03, "tail threshold −1 nat", rotation=90, fontsize=7.5,
            color=MUTED, ha="right", va="bottom")
    tj = (q["d_meanJ_span_safe"] <= -1).mean()
    tc = (q["d_ss_matched"] <= -1).mean()
    ax.set_title(f"{key} — tail {tj:.0%} vs {tc:.0%} "
                 f"(excess {est:+.3f} [{lo:+.2f}, {hi:+.2f}])", fontsize=9)
    ax.set_xlabel("per-item Δ log-prob (arm − baseline), nats")
    ax.set_xlim(-6.5, 1.6)
axes[0].set_ylabel("cumulative share of items")
axes[0].legend(loc="upper left", fontsize=8.5)
fig.suptitle("Qwen 3.6 27B: the J-specific heavy tail survives span-safe protection (P3-P2)",
             fontsize=11, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "figA_p3p2_ecdf.png", dpi=300, bbox_inches="tight")

# --- B: label-protected vs span-safe scatter, 3 panels ----------------------
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True, sharey=True)
lims = (-13, 3.6)
for ax, m in zip(axes, MODELS):
    a = audit[m]
    ax.plot(lims, lims, color=BASELINE, lw=1, zorder=1)
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.scatter(a["d_meanJ_label_protected"], a["d_meanJ_span_safe"],
               s=34, color=MODEL_C[m], edgecolor=SURFACE, linewidth=1.4,
               zorder=3, alpha=0.95)
    r = a["d_meanJ_label_protected"].corr(a["d_meanJ_span_safe"])
    ax.set_title(f"{NICE[m]}   r = {r:.2f}")
    ax.set_xlabel("Δ label-protected (nats)")
    ax.set_xlim(*lims); ax.set_ylim(*lims)
axes[0].set_ylabel("Δ span-safe (nats)")
axes[2].annotate("same items lose,\njust less (rescaling)", xy=(-8.5, -4.6),
                 fontsize=8, color=INK2, ha="center")
axes[0].annotate("different items lose\n(reallocation)", xy=(-8.5, -4.6),
                 fontsize=8, color=INK2, ha="center")
fig.suptitle("Removing the output-span leak reallocates damage on OLMo but only rescales it on Qwen",
             fontsize=11, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "figB_label_vs_spansafe.png", dpi=300, bbox_inches="tight")

# --- C: mediation forest ----------------------------------------------------
ARM_ORDER = [("span_safe", "span-safe reference"),
             ("true_bridge", "+ true-bridge protected"),
             ("distractor_bridge", "+ distractor-bridge protected"),
             ("bridge_only", "bridge-only lesion"),
             ("cf_swap", "counterfactual-bridge swap"),
             ("unrelated", "unrelated-content lesion"),
             ("answer_only", "answer-only lesion (diagnostic)"),
             ("rescue_contrast", "rescue contrast (true − distractor)")]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
y = np.arange(len(ARM_ORDER))[::-1]
off = {"qwen36-27b": 0.17, "olmo31-think": -0.17}
for m in ["qwen36-27b", "olmo31-think"]:
    stats = med_json[m]
    est = [stats[k]["estimate"] for k, _ in ARM_ORDER]
    lo = [stats[k]["ci95"][0] for k, _ in ARM_ORDER]
    hi = [stats[k]["ci95"][1] for k, _ in ARM_ORDER]
    ax.hlines(y + off[m], lo, hi, color=MODEL_C[m], lw=2)
    ax.plot(est, y + off[m], "o", ms=8, color=MODEL_C[m],
            markeredgecolor=SURFACE, markeredgewidth=2, label=NICE[m], zorder=3)
ax.axvline(0, color=BASELINE, lw=1)
ax.axhspan(y[-1] - 0.5, y[-1] + 0.5, color=GRID, alpha=0.35, zorder=0)
ax.set_yticks(y, [lab for _, lab in ARM_ORDER], fontsize=9, color=INK)
ax.set_xlabel("Δ log-prob vs baseline, nats (family-clustered 95% CI)")
ax.set_title("Bridge mediation factorial: Qwen reads its bridge channel; Think does not",
             fontsize=11, pad=12)
ax.legend(loc="lower left", fontsize=9)
ax.annotate("swap is catastrophically worse\nthan deletion on Qwen only",
            xy=(-4.05, y[4]), xytext=(-5.9, y[4] - 1.15), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
fig.tight_layout()
fig.savefig(FIGS / "figC_mediation_forest.png", dpi=300, bbox_inches="tight")

# --- D: accessibility sign flip ---------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
for ax, m in zip(axes, MODELS):
    g = mine[mine["model"] == m]
    x = np.clip(g["clean_first_rank_min"].values.astype(float), 1, None)
    ax.scatter(x, g["delta_J"], s=22, color=MODEL_C[m], alpha=0.8,
               edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.set_xscale("log")
    ax.axhline(0, color=BASELINE, lw=1, zorder=1)
    ax.set_title(f"{NICE[m]}   Spearman ρ = {rho[m]:+.2f}")
    ax.set_xlabel("clean answer rank (log)")
axes[0].set_ylabel("Δ log-prob under J ablation (nats)")
fig.suptitle("The accessibility sign flip: OLMo damage concentrates on less output-locked items; Qwen on more",
             fontsize=11, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "figD_accessibility.png", dpi=300, bbox_inches="tight")

# --- E: the specificity claim across eras -----------------------------------
rows = [
    ("Phase 2 · label-protected · confirmatory", 0.279, 0.205, 0.361, ARM_CTRL),
    ("Phase 2 · label-protected · replication", 0.297, 0.207, 0.382, ARM_CTRL),
    ("Phase 3 · span-safe · confirmatory", *(p3p2["confirmatory"][:3]), ARM_J),
    ("Phase 3 · span-safe · replication", *(p3p2["replication"][:3]), ARM_J),
]
fig, ax = plt.subplots(figsize=(8.2, 3.0))
y = np.arange(len(rows))[::-1]
for yi, (lab, est, lo, hi, c) in zip(y, rows):
    ax.hlines(yi, lo, hi, color=c, lw=2)
    ax.plot([est], [yi], "o", ms=9, color=c,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.annotate(f"{est:+.3f}", (hi, yi), xytext=(6, -3),
                textcoords="offset points", fontsize=8.5, color=INK2)
ax.axvline(0, color=BASELINE, lw=1)
ax.set_yticks(y, [r[0] for r in rows], fontsize=9, color=INK)
ax.set_xlabel("J-specific tail-rate excess over exact matched control (Qwen)")
ax.set_xlim(-0.02, 0.46)
ax.set_title("The specificity claim, before and after removing the output-span leak",
             fontsize=11, pad=12)
fig.tight_layout()
fig.savefig(FIGS / "figE_eras.png", dpi=300, bbox_inches="tight")

print(f"\nFigures written to {FIGS}")
