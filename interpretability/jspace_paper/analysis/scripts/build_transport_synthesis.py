#!/usr/bin/env python3
"""A5: transport-applicability matrix and map.

Combines the Gemma Stage-1 per-layer table (reconstructed byte-identical)
with the OLMo H6 registered joint row table (read-only Drive source;
verified by the OLMo reconstruction). Retrospective plot of registered
quantities only. No cell may say linear/nonlinear without map, prompt,
layer, dose, and gate — every figure element names its gate.

Outputs: tables/transport_applicability.csv,
figures/transport_applicability_map.{png,pdf},
reports/TRANSPORT_APPLICABILITY_SYNTHESIS.md
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A = Path("/content/labs/interpretability/jspace_paper/analysis")
H6 = ("/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_2_20260803"
      "/metrics/transport-validation-joint/ol2-transport-validation-joint-v1"
      "/transport_joint_rows.parquet")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, VIOLET, RED, GOOD = "#2a78d6", "#eb6834", "#4a3aa7", "#e34948", "#0ca30c"

CEILING = 0.07870368901355948
STUDY1_ALLSLOT = 0.002458111383020878
STUDY2_Q99 = 0.026234563004519824
OLMO_CONTROL_TRE = 0.11100  # late-anchor median, gm-jvp-olmo-positive-control-v1


def main():
    g = pd.read_csv(A / "tables/gemma_stage1_layer_table.csv")
    h6 = pd.read_parquet(H6)

    # ---- tidy applicability table
    rows = []
    for layer in ["L22", "L30", "L37", "L44", "L52"]:
        sub = g[g.layer == layer].set_index("metric").value
        rows.append(dict(
            model="Gemma-4-31B", checkpoint="gemma4-31b", layer=layer,
            epsilon=0.10, gate="declared-dose tangent gate",
            n_pass=int(float(sub["declared_dose_n_pass"])),
            n_rows=int(float(sub["declared_dose_n_evaluable"])),
            metric="bootstrap_tre",
            value=float(sub["declared_dose_bootstrap_tre_estimate"]),
            ci_low=float(sub["declared_dose_bootstrap_tre_ci95_low"]),
            ci_high=float(sub["declared_dose_bootstrap_tre_ci95_high"]),
            route="local_tangent_mismatch",
            source_evidence_id="gm-jvp-gemma-stage1-v1;gm2-stage1-relicense-v1"))
    cells = (h6.groupby(["joint_model_key", "source_layer",
                         "desired_relative_epsilon"])
             .agg(n_rows=("transport_row_passed", "size"),
                  n_pass=("transport_row_passed", "sum"),
                  n_meas=("measurement_eligible", "sum"),
                  n_dec=("decision_eligible", "sum"))
             .reset_index())
    for _, r in cells.iterrows():
        rows.append(dict(
            model="OLMo-3 Base" if r.joint_model_key == "base"
            else "OLMo-3.1 Think",
            checkpoint=r.joint_model_key, layer=f"L{int(r.source_layer)}",
            epsilon=float(r.desired_relative_epsilon),
            gate="H6 cell floor 0.90", n_pass=int(r.n_pass),
            n_rows=int(r.n_rows), metric="cell_pass_fraction",
            value=r.n_pass / r.n_rows, ci_low=None, ci_high=None,
            route=("pass" if r.n_pass / r.n_rows >= 0.90 else "fail"),
            source_evidence_id="ol2-transport-validation-joint-v1"))
    tidy = pd.DataFrame(rows)
    tidy.to_csv(A / "tables/transport_applicability.csv", index=False)

    # ---- figure
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.size": 9.5, "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlecolor": INK,
        "axes.titlesize": 9.8, "legend.frameon": False, "text.color": INK})
    fig = plt.figure(figsize=(12.8, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.30)

    # (a) Gemma per-layer mismatch scale
    ax = fig.add_subplot(gs[0])
    gl = tidy[tidy.model == "Gemma-4-31B"]
    xs = np.arange(len(gl))
    ax.errorbar(xs, gl.value,
                yerr=[gl.value - gl.ci_low, gl.ci_high - gl.value],
                fmt="o", color=BLUE, ms=7, capsize=3, lw=1.6, zorder=4)
    for ref, lab, c in [(OLMO_CONTROL_TRE, "OLMo control (late anchor) 0.111", GOOD),
                        (CEILING, "frozen backend ceiling 0.0787", VIOLET),
                        (STUDY2_Q99, "backend q99 0.0262", MUTED),
                        (STUDY1_ALLSLOT, "historical all-slot 0.00246", MUTED)]:
        ax.axhline(ref, color=c, lw=1.1, ls=(0, (4, 3)))
        ax.text(len(gl) - 0.45, ref * 1.12, lab, fontsize=6.8, color=c,
                ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(gl.layer, fontsize=8.6)
    ax.set_ylabel("declared-dose tangent relative error (log)", fontsize=8.2)
    ax.set_title("Gemma ε=0.10 mismatch by depth\n(0/12 pass per layer; "
                 "12–52× above backend scales)")

    # (b)(c) OLMo H6 cell maps
    eps = sorted(h6.desired_relative_epsilon.unique())
    layers = sorted(h6.source_layer.unique())
    for gi, (key, name) in enumerate(
            [("base", "OLMo-3 Base"), ("think", "OLMo-3.1 Think")], start=1):
        ax = fig.add_subplot(gs[gi])
        sub = cells[cells.joint_model_key.str.contains(key)]
        for yi, layer in enumerate(layers):
            for xi, e in enumerate(eps):
                c = sub[(sub.source_layer == layer) &
                        (sub.desired_relative_epsilon == e)]
                if not len(c):
                    continue
                r = c.iloc[0]
                frac = r.n_pass / r.n_rows
                licensed = frac >= 0.90
                face = (GOOD if licensed else
                        "#f0efec" if r.n_dec == 0 else "#cde2fb")
                ax.add_patch(mpl.patches.FancyBboxPatch(
                    (xi + 0.06, yi + 0.08), 0.88, 0.84,
                    boxstyle="round,pad=0,rounding_size=0.05",
                    facecolor=face,
                    edgecolor=GOOD if licensed else GRID,
                    linewidth=1.6 if licensed else 1.0))
                ink = "#ffffff" if licensed else (
                    MUTED if r.n_dec == 0 else INK2)
                ax.text(xi + 0.5, yi + 0.5, f"{int(r.n_pass)}/{int(r.n_rows)}",
                        ha="center", va="center", fontsize=7.4, color=ink,
                        fontweight="bold" if licensed else "normal")
        ax.set_xticks([x + 0.5 for x in range(len(eps))])
        ax.set_xticklabels([f"{e:g}" for e in eps], fontsize=7.2)
        ax.set_yticks([y + 0.5 for y in range(len(layers))])
        ax.set_yticklabels([f"L{int(l)}" for l in layers], fontsize=8.2)
        ax.set_xlim(0, len(eps))
        ax.set_ylim(0, len(layers))
        ax.set_xlabel("relative ε", fontsize=8.2)
        ax.set_title(f"{name}: rows passing / cell\n(licensed iff ≥ 0.90 "
                     "of 12; green = licensed)")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)

    handles = [
        mpl.patches.Patch(facecolor=GOOD, label="licensed cell (≥0.90)"),
        mpl.patches.Patch(facecolor="#cde2fb", edgecolor=GRID,
                          label="decision-eligible rows present, fails floor"),
        mpl.patches.Patch(facecolor="#f0efec", edgecolor=GRID,
                          label="no decision-eligible rows (below measurability)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.8, bbox_to_anchor=(0.66, -0.04))
    fig.suptitle("Transport applicability — exact-JVP finite-dose gates by "
                 "model, layer, and dose (frozen source→final-residual map)",
                 fontsize=11.3, y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(A / f"figures/transport_applicability_map.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"{len(tidy)} applicability rows; figure written")


if __name__ == "__main__":
    main()
