"""Registered figures (PNG+PDF, tier-watermarked, regenerated from tables).

Palette: fixed semantic assignment, validated with the dataviz six-check
script on the light surface (2026-08-07: PASS; CVD worst pair 7.6 in the
6-8 band — legal with the secondary encodings used here: direct scenario
labels on categorical axes, distinct markers, 2px white mark edges; the
pink/orange contrast WARN is relieved by visible labels on every figure).
Color follows the entity (family/channel), never rank. One axis per chart.
"""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import artifacts  # noqa: E402

PAL = {
    "AR": "#0072B2",        # arbitrary revealed choice
    "PC": "#009E73",        # positive control
    "NC": "#CC79A7",        # null control
    "RO": "#7E57C2",        # report-only channel
    "nuisance": "#E69F00",  # surface/nuisance endpoints
    "alarm": "#D55E00",     # threshold violations
    "ink": "#1F2430",
    "muted": "#667085",
    "grid": "#E4E7EC",
}
MARKER = {"AR": "o", "PC": "s", "NC": "D", "RO": "^", "nuisance": "v"}


def _style_ax(ax: Any, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PAL["muted"])
    ax.tick_params(colors=PAL["muted"], labelsize=8)
    ax.grid(True, axis="x", color=PAL["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=PAL["ink"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=PAL["ink"])
    if title:
        ax.set_title(title, fontsize=10, color=PAL["ink"], loc="left",
                     fontweight="bold", pad=10)


def save_fig(fig: Any, out_dir: pathlib.Path, name: str, tier: str,
             source_rows: list[dict[str, Any]] | None = None,
             run_label: str = "") -> list[pathlib.Path]:
    """jspaces convention: every figure carries its tier watermark, lands
    as PNG (handout) + PDF (paper), and registers its source rows."""
    fig.text(0.995, 0.005, tier.upper(), ha="right", va="bottom",
             fontsize=7, color=PAL["muted"], alpha=0.9)
    if run_label:
        fig.text(0.005, 0.005, run_label, ha="left", va="bottom",
                 fontsize=6, color=PAL["muted"], alpha=0.9)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_out = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
        paths_out.append(p)
    plt.close(fig)
    if source_rows is not None:
        src = out_dir.parent / "tables" / "figure_sources" / f"{name}.csv"
        artifacts.write_csv(src, source_rows)
        paths_out.append(src)
    return paths_out


def fig_scenario_forest(effects: pd.DataFrame, out_dir: pathlib.Path, *,
                        tier: str, run_label: str, nc_p95: float | None,
                        sesoi: float = 0.10,
                        name: str = "f01_scenario_effect_forest") -> None:
    """Dot+interval forest of signed effects toward pole_1, by family."""
    d = effects.sort_values(["family", "effect"],
                            ascending=[True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(d) + 1.6))
    ax.axvspan(-sesoi, sesoi, color=PAL["grid"], alpha=0.5, lw=0,
               label=f"SESOI ±{sesoi}")
    ax.axvline(0, color=PAL["muted"], lw=1)
    if nc_p95 is not None and np.isfinite(nc_p95):
        ax.axvline(nc_p95, color=PAL["NC"], lw=1.2, ls="--")
        ax.axvline(-nc_p95, color=PAL["NC"], lw=1.2, ls="--",
                   label=f"NC floor ±{nc_p95:.3f}")
    for i, row in d.iterrows():
        fam = row["family"]
        if np.isfinite(row.get("ci90_lo", np.nan)):
            ax.plot([row["ci90_lo"], row["ci90_hi"]], [i, i],
                    color=PAL[fam], lw=2, alpha=0.55, solid_capstyle="round")
        ax.plot(row["effect"], i, MARKER[fam], color=PAL[fam], ms=7,
                mec="white", mew=1.2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["scenario_id"], fontsize=8, color=PAL["ink"])
    ax.set_ylim(-0.7, len(d) - 0.3)
    _style_ax(ax, xlabel="signed effect toward pole_1  (p − 0.5, strict choice)",
              title="Scenario content effects with 90% hierarchical bootstrap CIs")
    handles = [plt.Line2D([], [], color=PAL[f], marker=MARKER[f], ls="",
                          mec="white", label=f)
               for f in ("AR", "PC", "NC") if (d["family"] == f).any()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    save_fig(fig, out_dir, name, tier,
             source_rows=d.to_dict("records"), run_label=run_label)


def fig_pc_pipeline(pc_rows: list[dict[str, Any]], out_dir: pathlib.Path, *,
                    tier: str, run_label: str,
                    name: str = "f02_positive_control_pipeline") -> None:
    d = pd.DataFrame(pc_rows)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    y = np.arange(len(d))
    ax.barh(y, d["expected_rate"], height=0.62, color=PAL["PC"],
            edgecolor="white", linewidth=2)
    ax.axvline(0.75, color=PAL["alarm"], lw=1.2, ls="--")
    ax.text(0.752, len(d) - 0.42, "per-scenario gate 0.75", fontsize=7,
            color=PAL["alarm"], va="top")
    ax.axvline(0.5, color=PAL["muted"], lw=1)
    for i, row in d.iterrows():
        ax.text(min(float(row["expected_rate"]) + 0.012, 0.99), i,
                f"{row['expected_rate']:.2f}", fontsize=8,
                va="center", color=PAL["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(d["scenario_id"], fontsize=8, color=PAL["ink"])
    ax.set_xlim(0, 1.02)
    _style_ax(ax, xlabel="expected-content choice rate (valid rows)",
              title="Positive-control pipeline: expected content wins")
    save_fig(fig, out_dir, name, tier, source_rows=pc_rows,
             run_label=run_label)


def fig_content_vs_nuisance(rows: list[dict[str, Any]], out_dir: pathlib.Path,
                            *, tier: str, run_label: str,
                            name: str = "f03_content_vs_position_asymmetry") -> None:
    d = pd.DataFrame(rows)
    d = d.sort_values("abs_content", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(d) + 1.6))
    y = np.arange(len(d))
    ax.plot(d["abs_content"], y, MARKER["AR"], color=PAL["AR"], ms=7,
            mec="white", mew=1.2, ls="", label="|content effect|")
    ax.plot(d["abs_position"], y, MARKER["nuisance"], color=PAL["nuisance"],
            ms=6, mec="white", mew=1.2, ls="", label="|position effect|")
    ax.plot(d["abs_code"], y, "P", color=PAL["RO"], ms=6, mec="white",
            mew=1.2, ls="", label="|code effect|")
    ax.axvline(0.10, color=PAL["alarm"], lw=1.1, ls="--")
    ax.text(0.102, -0.55, "nuisance warning 0.10", fontsize=7,
            color=PAL["alarm"], va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(d["scenario_id"], fontsize=8, color=PAL["ink"])
    _style_ax(ax, xlabel="absolute effect (p − 0.5 scale)",
              title="Content asymmetry vs surface nuisances, per scenario")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save_fig(fig, out_dir, name, tier, source_rows=rows, run_label=run_label)


def fig_frame_effects(rows: list[dict[str, Any]], out_dir: pathlib.Path, *,
                      tier: str, run_label: str,
                      name: str = "f04_consequence_frame_effects") -> None:
    d = pd.DataFrame(rows).sort_values("frame_effect_enacted_minus_hyp")
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(d) + 1.6))
    y = np.arange(len(d))
    for i, row in enumerate(d.itertuples()):
        ax.plot([row.p_pole1_hypothetical, row.p_pole1_enacted], [i, i],
                color=PAL["muted"], lw=1.2, alpha=0.7, zorder=1)
    ax.plot(d["p_pole1_hypothetical"], y, "o", color=PAL["nuisance"], ms=6,
            mec="white", mew=1.1, ls="", label="hypothetical frame")
    ax.plot(d["p_pole1_enacted"], y, "o", color=PAL["AR"], ms=7,
            mec="white", mew=1.1, ls="", label="enacted frame")
    ax.axvline(0.5, color=PAL["muted"], lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["scenario_id"], fontsize=8, color=PAL["ink"])
    _style_ax(ax, xlabel="p(choose pole_1), valid rows",
              title="In-context consequence-frame effect (enacted vs hypothetical)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save_fig(fig, out_dir, name, tier, source_rows=rows, run_label=run_label)


def fig_stated_revealed(rows: list[dict[str, Any]], out_dir: pathlib.Path, *,
                        tier: str, run_label: str,
                        name: str = "f05_stated_vs_revealed") -> None:
    d = pd.DataFrame(rows)
    d = d[d["ar_frame"] == "enacted"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=PAL["grid"], lw=1.4, zorder=1)
    fam_colors = d["family"].map(PAL)
    for fam in ("AR", "PC"):
        dd = d[d["family"] == fam]
        ax.plot(dd["ar_pole1_rate"], dd["ro_pole1_rate"], MARKER[fam],
                color=PAL[fam], ms=8, mec="white", mew=1.2, ls="", label=fam)
    for _, row in d.iterrows():
        ax.annotate(row["scenario_id"].replace("ar_", "").replace("pc_", ""),
                    (row["ar_pole1_rate"], row["ro_pole1_rate"]),
                    fontsize=6, color=PAL["muted"],
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    _style_ax(ax, xlabel="revealed (AR enacted) p(pole_1)",
              ylabel="report-only (RO) p(pole_1)",
              title="Stated vs revealed: matched-cell content rates")
    ax.grid(True, axis="y", color=PAL["grid"], linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save_fig(fig, out_dir, name, tier, source_rows=rows, run_label=run_label)


def fig_nc_null_distribution(nc_draw_abs: dict[str, np.ndarray],
                             ar_effects: list[dict[str, Any]],
                             out_dir: pathlib.Path, *, tier: str,
                             run_label: str,
                             name: str = "f06_nc_null_distribution") -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    allnc = np.concatenate(list(nc_draw_abs.values()))
    xs = np.sort(allnc)
    ax.plot(xs, np.linspace(0, 1, len(xs)), color=PAL["NC"], lw=2,
            label="NC bootstrap |effect| (ECDF)")
    p95 = float(np.nanquantile(allnc, 0.95))
    ax.axvline(p95, color=PAL["NC"], ls="--", lw=1.2)
    ax.text(p95 + 0.003, 0.06, f"NC p95 = {p95:.3f}", fontsize=8,
            color=PAL["NC"])
    for row in ar_effects:
        ax.axvline(abs(row["effect"]), color=PAL["AR"], lw=1.0, alpha=0.65)
    ax.plot([], [], color=PAL["AR"], lw=1.0, label="AR scenario |effect|")
    ax.set_ylim(0, 1.02)
    _style_ax(ax, xlabel="|effect| (p − 0.5 scale)", ylabel="ECDF",
              title="Empirical false-positive floor from identical-option NC scenarios")
    ax.grid(True, axis="y", color=PAL["grid"], linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save_fig(fig, out_dir, name, tier,
             source_rows=[{"scenario_id": k, "abs_p95": float(np.nanquantile(v, 0.95))}
                          for k, v in nc_draw_abs.items()],
             run_label=run_label)


def fig_margin_vs_strict(rows: list[dict[str, Any]], out_dir: pathlib.Path, *,
                         tier: str, run_label: str,
                         name: str = "f07_margin_vs_strict_endpoint") -> None:
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.axhline(0, color=PAL["muted"], lw=1)
    ax.axvline(0, color=PAL["muted"], lw=1)
    for fam in ("AR", "PC", "NC"):
        dd = d[d["family"] == fam]
        if len(dd):
            ax.plot(dd["strict_effect"], dd["margin_effect"], MARKER[fam],
                    color=PAL[fam], ms=8, mec="white", mew=1.2, ls="",
                    label=fam)
    _style_ax(ax, xlabel="strict-generation effect (p − 0.5)",
              ylabel="exact-target margin effect (p(margin>0) − 0.5)",
              title="Primary vs secondary endpoint agreement")
    ax.grid(True, axis="y", color=PAL["grid"], linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save_fig(fig, out_dir, name, tier, source_rows=rows, run_label=run_label)


def fig_validity(rows: list[dict[str, Any]], out_dir: pathlib.Path, *,
                 tier: str, run_label: str,
                 name: str = "f08_parse_validity") -> None:
    d = pd.DataFrame(rows).sort_values("valid_rate").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 0.30 * len(d) + 1.5))
    y = np.arange(len(d))
    colors = [PAL[f] for f in d["family"]]
    ax.barh(y, d["valid_rate"], height=0.6, color=colors, edgecolor="white",
            linewidth=2)
    ax.axvline(0.98, color=PAL["alarm"], lw=1.1, ls="--")
    ax.text(0.9795, -0.55, "PC gate 0.98", fontsize=7, color=PAL["alarm"],
            va="top", ha="right")
    for i, row in d.iterrows():
        ax.text(min(float(row["valid_rate"]) + 0.005, 1.0), i,
                f"{row['valid_rate']:.3f}", fontsize=7, va="center",
                color=PAL["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(d["scenario_id"] + " · " + d["channel"], fontsize=7,
                       color=PAL["ink"])
    ax.set_xlim(0, 1.05)
    _style_ax(ax, xlabel="strict valid-parse rate",
              title="Instrument health: strict parse validity by scenario × channel")
    save_fig(fig, out_dir, name, tier, source_rows=rows, run_label=run_label)
