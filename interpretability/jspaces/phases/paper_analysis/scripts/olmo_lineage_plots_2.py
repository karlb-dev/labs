#!/usr/bin/env python
"""OLMo lineage synthesis figures for the olmo_lineage notes document.

Reads ONLY registered Phase 4 artifacts:
  - trajectory_table CSV from p4-lineage-trajectory-analysis-olmo-dev-v1
  - the four G5 bank parquets (p4-g5-bank-*-dev-v*)

Outputs (regenerable any time):
  jspace_paper/figures/olmoL1_emergence_trajectory.png
  jspace_paper/figures/olmoL2_capability_trajectory.png

Run:  cd interpretability/jspace_runs && .venv/bin/python analysis/olmo_lineage_plots.py
"""
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



import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

RUNS = _find_runs_cache()
P4M = RUNS / "phase4" / "metrics"
OUT = pathlib.Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TRAJ = (P4M / "olmo-lineage-trajectory" / "trajectory_analysis" /
        "p4-lineage-trajectory-analysis-olmo-dev-v1" /
        "trajectory_table_olmo-lineage-trajectory.csv")
G5 = {
    "olmo3-base": P4M / "olmo3-base/g5_bank/p4-g5-bank-olmo3-base-dev-v1/g5_bank_olmo3-base.parquet",
    "olmo3-think": P4M / "olmo3-think/g5_bank/p4-g5-bank-olmo3-think-dev-v2/g5_bank_olmo3-think.parquet",
    "olmo31-think": P4M / "olmo31-think/g5_bank/p4-g5-bank-olmo31-think-dev-v1/g5_bank_olmo31-think.parquet",
    "olmo31-instruct": P4M / "olmo31-instruct/g5_bank/p4-g5-bank-olmo31-instruct-dev-v1/g5_bank_olmo31-instruct.parquet",
}

# dataviz reference palette (light mode)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
C_DIRECT, C_COMPOSED, C_CONTRAST = "#2a78d6", "#eb6834", "#4a3aa7"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.size": 9.5, "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "axes.titlesize": 10, "legend.frameon": False,
})

ORDER = ["olmo3-base", "olmo3-think", "olmo31-think", "olmo31-instruct"]
LABEL = {"olmo3-base": "Base", "olmo3-think": "3.0 Think",
         "olmo31-think": "3.1 Think", "olmo31-instruct": "3.1 Instruct\n(sibling)"}
XPOS = {"olmo3-base": 0, "olmo3-think": 1, "olmo31-think": 2,
        "olmo31-instruct": 3.35}                       # sibling detached

# ---------- L1: the emergence trajectory (registered trajectory table) -----
t = pd.read_csv(TRAJ)
PANELS = [("S:direct", "Bank S · direct specificity", C_DIRECT),
          ("S:composed", "Bank S · composed specificity", C_COMPOSED),
          ("S:composition", "Bank S · composed − direct", C_CONTRAST)]
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharex=True)
for ax, (mk, title, color) in zip(axes, PANELS):
    for frame, ls, mfc, dx in [("own", "-", color, -0.06),
                               ("common", "--", SURFACE, +0.06)]:
        rows = (t[(t.metric_key == mk) & (t.frame == frame)]
                .set_index("checkpoint_key").loc[ORDER].reset_index())
        x = [XPOS[c] + dx for c in rows.checkpoint_key]
        ax.errorbar(
            x, rows.estimate,
            yerr=[rows.estimate - rows.ci95_low, rows.ci95_high - rows.estimate],
            fmt="none", ecolor=color, elinewidth=1.6, capsize=0, alpha=0.85)
        thread = rows.checkpoint_key != "olmo31-instruct"
        ax.plot([x[i] for i in range(3)], rows.estimate[:3], ls, color=color,
                lw=1.6, alpha=0.7, zorder=2)
        for i in range(len(rows)):
            marker = "s" if rows.checkpoint_key[i] == "olmo31-instruct" else "o"
            ax.plot(x[i], rows.estimate[i], marker, ms=8, color=color,
                    markerfacecolor=mfc, markeredgecolor=color,
                    markeredgewidth=1.6, zorder=3,
                    label=f"{frame} lens" if i == 0 else None)
    ax.axhline(0, color=BASELINE, lw=1, zorder=1)
    ax.set_title(title)
    ax.set_xticks([XPOS[c] for c in ORDER], [LABEL[c] for c in ORDER],
                  fontsize=8.5, color=INK)
axes[0].set_ylabel("J-specific effect (nats)")
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("Bank-S J-specificity is created on the base→Think path and absent "
             "in the Instruct sibling (own vs common lens frames)",
             fontsize=10.5, color=INK, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "olmoL1_emergence_trajectory.png", dpi=300, bbox_inches="tight")

# ---------- L2: G5 capability trajectory (from the four G5 parquets) -------
cap = []
for ck, path in G5.items():
    g = pd.read_parquet(path)
    for (bank, variant), grp in g[g.variant.isin(["direct", "composed"])] \
            .groupby(["bank", "variant"]):
        cap.append({"checkpoint": ck, "bank": bank, "variant": variant,
                    "rate": grp.capable_generation.mean(), "n": len(grp)})
cap = pd.DataFrame(cap)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
for ax, bank in zip(axes, ["F", "S"]):
    for variant, color in [("direct", C_DIRECT), ("composed", C_COMPOSED)]:
        rows = (cap[(cap.bank == bank) & (cap.variant == variant)]
                .set_index("checkpoint").loc[ORDER].reset_index())
        x = [XPOS[c] for c in rows.checkpoint]
        ax.plot(x[:3], rows.rate[:3], "-", color=color, lw=1.6, alpha=0.7)
        for i in range(len(rows)):
            marker = "s" if rows.checkpoint[i] == "olmo31-instruct" else "o"
            ax.plot(x[i], rows.rate[i], marker, ms=8, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3,
                    label=variant if i == 0 else None)
    ax.set_title(f"Bank {bank} · G5 capable rate")
    ax.set_xticks([XPOS[c] for c in ORDER], [LABEL[c] for c in ORDER],
                  fontsize=8.5, color=INK)
    ax.set_ylim(0.0, 1.0)
axes[0].set_ylabel("capable generation rate")
axes[0].legend(loc="upper left", fontsize=8.5)
fig.suptitle("Capability across the lineage (prospective prefix-disjoint G5 scoring)",
             fontsize=10.5, color=INK, y=1.04)
fig.tight_layout()
fig.savefig(OUT / "olmoL2_capability_trajectory.png", dpi=300, bbox_inches="tight")

print("wrote", OUT / "olmoL1_emergence_trajectory.png")
print("wrote", OUT / "olmoL2_capability_trajectory.png")
print(cap.pivot_table(index="checkpoint", columns=["bank", "variant"],
                      values="rate").loc[ORDER].round(4))
