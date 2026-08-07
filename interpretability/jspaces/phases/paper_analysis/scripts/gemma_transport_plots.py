#!/usr/bin/env python
"""Gemma transport-gate figure for the Gemma nonlinear-transport handout.

Primary source (run cache, if present):
  interpretability/jspace_runs/gemma_transport_20260802/metrics/
    gemma_target/gm-jvp-gemma-stage1-v1/gemma_stage1_rows.parquet
    olmo_control/gm-jvp-olmo-calibration-v1/olmo_calibration_rows.parquet

On first run with the cache present, a small snapshot CSV is written next to
this script (gemma_transport_snapshot.csv); later runs fall back to the
snapshot so the figure regenerates without the 700 MB cache.

Output: interpretability/jspaces/phases/paper_analysis/figures/gmT1_tangent_ladder.png
Run:    cd interpretability/jspaces/phases/paper_analysis && ../jspace_runs/.venv/bin/python scripts/gemma_transport_plots.py
"""
import os
import pathlib

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
PAPER = HERE.parent


def _find_runs_cache() -> pathlib.Path | None:
    env = os.environ.get("JSPACE_RUNS_ROOT")
    if env:
        return pathlib.Path(env)
    for parent in HERE.parents:
        cand = parent / "jspace_runs"
        if cand.is_dir():
            return cand
    return None


_runs = _find_runs_cache()
RUNS = (_runs / "gemma_transport_20260802" / "metrics") if _runs else PAPER / "_no_runs_cache"
SNAP = HERE / "gemma_transport_snapshot.csv"
OUT = PAPER / "figures"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
# ordinal one-hue ramps (light -> dark = shallow -> deep layers)
BLUE_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#184f95", "#0d366b"]
RED_RAMP = ["#f2a09f", "#ec807f", "#e34948", "#c03434", "#8f2726"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.size": 9.5, "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "axes.titlesize": 10, "legend.frameon": False,
})

CEILING = 0.1          # frozen central-tangent relative-error ceiling
DECLARED_EPS = 0.1     # declared primary dose


def build_snapshot() -> pd.DataFrame:
    frames = []
    for model, path in [
        ("Gemma 4 31B", RUNS / "gemma_target/gm-jvp-gemma-stage1-v1/gemma_stage1_rows.parquet"),
        ("OLMo-3 32B Think (control)", RUNS / "olmo_control/gm-jvp-olmo-calibration-v1/olmo_calibration_rows.parquet"),
    ]:
        df = pd.read_parquet(path)
        ok = df[df.faithful_delivery & (df.response_snr >= 12)]
        agg = (ok.groupby(["source_layer", "desired_relative_epsilon"])
               .agg(central_err=("central_tangent_relative_error", "median"),
                    tangent_cos=("tangent_cosine", "median"),
                    n=("tangent_cosine", "size"))
               .reset_index())
        agg.insert(0, "model", model)
        frames.append(agg)
    snap = pd.concat(frames, ignore_index=True)
    snap.to_csv(SNAP, index=False)
    return snap


if RUNS.exists():
    snap = build_snapshot()
else:
    snap = pd.read_csv(SNAP)

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True, sharex=True)
for ax, model, ramp in [
        (axes[0], "OLMo-3 32B Think (control)", BLUE_RAMP),
        (axes[1], "Gemma 4 31B", RED_RAMP)]:
    sub = snap[snap.model == model]
    layers = sorted(sub.source_layer.unique())
    ends = []
    for i, layer in enumerate(layers):
        rows = sub[sub.source_layer == layer].sort_values("desired_relative_epsilon")
        color = ramp[min(i, len(ramp) - 1)]
        ax.plot(rows.desired_relative_epsilon, rows.central_err, "-o",
                color=color, lw=1.6, ms=5.5, markeredgecolor=SURFACE,
                markeredgewidth=1.0, label=f"L{layer}")
        last = rows.iloc[-1]
        ends.append((layer, last.desired_relative_epsilon, last.central_err))
    # end-labels with collision spreading (leader offsets when log-close)
    import math
    ends.sort(key=lambda e: e[2])
    prev_log, stack = None, 0
    for layer, x, y in ends:
        ly = math.log10(y)
        stack = stack + 1 if prev_log is not None and ly - prev_log < 0.14 else 0
        prev_log = ly
        ax.annotate(f"L{layer}", (x, y), xytext=(6, -3 + 9 * stack),
                    textcoords="offset points", fontsize=7.5, color=INK2)
    ax.axhline(CEILING, color=BASELINE, lw=1.2)
    ax.axvline(DECLARED_EPS, color=GRID, lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(model)
    ax.set_xlabel("relative perturbation ε (log)")
axes[0].set_ylabel("central tangent relative error (log)")
axes[0].annotate("frozen pass ceiling 0.1", (0.011, CEILING), xytext=(0, 5),
                 textcoords="offset points", fontsize=7.5, color=MUTED)
axes[0].annotate("declared dose", (DECLARED_EPS, 6), rotation=90, fontsize=7.5,
                 color=MUTED, ha="right", va="top")
fig.suptitle("The exact-JVP transport gate: OLMo's tangents improve with depth and pass;"
             " Gemma's diverge with dose and depth — all rows shown are delivery/SNR-clean",
             fontsize=10, color=INK, y=1.04)
fig.tight_layout()
fig.savefig(OUT / "gmT1_tangent_ladder.png", dpi=300, bbox_inches="tight")
print("wrote", OUT / "gmT1_tangent_ladder.png", "| snapshot:", SNAP.name)
