"""Registered figures (plan §81; Phase 1 conventions: Agg backend, color
follows the entity never rank, direct labels, tier watermark, PNG+PDF,
source CSV per figure so every figure regenerates from registered
tables)."""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .artifacts import ensure_dir, write_csv  # noqa: E402

PAL = {
    "ARB": "#0072B2", "MECH": "#009E73", "PC": "#7E57C2", "NC": "#CC79A7",
    "RO": "#E69F00", "CANON": "#56B4E9", "SURF": "#8C8C8C",
    "alarm": "#D55E00", "ink": "#1F2430", "muted": "#667085",
    "grid": "#E4E7EC",
}


def _style(ax, *, xlabel="", ylabel="", title=""):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PAL["muted"])
    ax.tick_params(colors=PAL["muted"], labelsize=8)
    ax.set_xlabel(xlabel, fontsize=9, color=PAL["ink"])
    ax.set_ylabel(ylabel, fontsize=9, color=PAL["ink"])
    if title:
        ax.set_title(title, fontsize=10, color=PAL["ink"], loc="left",
                     fontweight="bold")
    ax.grid(True, axis="x", color=PAL["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def save_fig(fig, out_dir: pathlib.Path, name: str, *, tier: str,
             source_rows: list[dict] | None = None,
             run_label: str = "") -> list[pathlib.Path]:
    out_dir = ensure_dir(pathlib.Path(out_dir))
    fig.text(0.99, 0.01, tier.upper(), ha="right", va="bottom",
             fontsize=7, color=PAL["muted"])
    if run_label:
        fig.text(0.01, 0.01, run_label, ha="left", va="bottom",
                 fontsize=7, color=PAL["muted"])
    paths_out = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
        paths_out.append(p)
    plt.close(fig)
    if source_rows:
        src = ensure_dir(out_dir.parent / "tables" / "figure_sources")
        write_csv(src / f"{name}.csv", source_rows)
    return paths_out


def fig_surface_decomposition(coefs: list[dict], out_dir, *, tier,
                              run_label="",
                              name="f01_surface_policy_decomposition"):
    rows = [c for c in coefs if np.isfinite(c.get("effect", np.nan))]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.2))
    labels = [f"{c['format_id']} · {c['endpoint']}" for c in rows]
    y = np.arange(len(rows))[::-1]
    for yi, c in zip(y, rows):
        color = PAL["SURF"] if c["format_id"] == "F-P1" else PAL["ARB"]
        ax.plot([c["ci_lo"], c["ci_hi"]], [yi, yi], color=color, lw=2,
                alpha=0.6)
        ax.plot(c["effect"], yi, "o", color=color, ms=6)
        ax.annotate(f"{c['effect']:+.3f}", (c["effect"], yi),
                    textcoords="offset points", xytext=(6, 5), fontsize=7,
                    color=PAL["ink"])
    ax.axvline(0, color=PAL["muted"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    _style(ax, xlabel="effect on emitted code (rate − 0.5)",
           title="B-SURF surface-policy decomposition (null twins)")
    return save_fig(fig, out_dir, name, tier=tier, source_rows=rows,
                    run_label=run_label)


def fig_forest(rows: list[dict], out_dir, *, value, lo, hi, label_key,
               name, title, xlabel, floor: float | None = None,
               tier="frozen_behavioral", run_label="", color=PAL["ARB"],
               pass_key: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        c = color if not pass_key or r.get(pass_key) else PAL["muted"]
        ax.plot([r[lo], r[hi]], [yi, yi], color=c, lw=2, alpha=0.65)
        ax.plot(r[value], yi, "o", color=c, ms=6)
    if floor is not None and np.isfinite(floor):
        ax.axvline(floor, color=PAL["alarm"], lw=1, ls="--")
        ax.axvline(-floor, color=PAL["alarm"], lw=1, ls="--")
    ax.axvline(0, color=PAL["muted"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[label_key] for r in rows], fontsize=8)
    _style(ax, xlabel=xlabel, title=title)
    return save_fig(fig, out_dir, name, tier=tier, source_rows=rows,
                    run_label=run_label)


def fig_margin_vs_choice(margin_rows, choice_rows, out_dir, *, tier,
                         run_label="", name="f05_margin_vs_choice"):
    m = {r["scenario_id"]: r for r in margin_rows}
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for c in choice_rows:
        r = m.get(c["scenario_id"])
        if not r:
            continue
        ax.plot(r["estimate"], c["estimate"], "o", color=PAL["ARB"], ms=6)
        ax.annotate(c["scenario_id"].replace("arb_", ""),
                    (r["estimate"], c["estimate"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7,
                    color=PAL["muted"])
    ax.axhline(0, color=PAL["muted"], lw=0.8)
    ax.axvline(0, color=PAL["muted"], lw=0.8)
    _style(ax, xlabel="full-target semantic margin (nats, A−B)",
           ylabel="strict semantic choice effect (rate − 0.5)",
           title="Margin vs enacted choice (B-ARB3)")
    return save_fig(fig, out_dir, name, tier=tier,
                    source_rows=[{**m.get(c["scenario_id"], {}),
                                  "choice": c["estimate"]}
                                 for c in choice_rows],
                    run_label=run_label)


def fig_context_curves(results_rows: list[dict], out_dir, *, tier,
                       run_label="", name="f06_context_value_curves"):
    import pandas as pd

    df = pd.DataFrame([r for r in results_rows
                       if r.get("bank") == "B-MECH"
                       and not r.get("codebook_reserved")])
    if not len(df):
        return []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for scn, g in df.groupby("scenario_id"):
        by_s = g.groupby("context_strength")["margin_full_a_minus_b"].mean()
        axes[0].plot(by_s.index, by_s.values, "-o", ms=4, lw=1.5,
                     label=scn.replace("mech_", ""))
        valid = g[g["parse_status"] == "valid"].copy()
        valid["chose_a"] = valid["parsed_sem"] == "a"
        by_sc = valid.groupby("context_strength")["chose_a"].mean()
        axes[1].plot(by_sc.index, by_sc.values, "-o", ms=4, lw=1.5)
    axes[0].axhline(0, color=PAL["muted"], lw=0.8)
    axes[1].axhline(0.5, color=PAL["muted"], lw=0.8)
    _style(axes[0], xlabel="context strength (+ favors A)",
           ylabel="semantic margin (nats)", title="Margin curve")
    _style(axes[1], xlabel="context strength (+ favors A)",
           ylabel="P(choose A | valid)", title="Strict-choice curve")
    axes[0].legend(fontsize=7, frameon=False)
    src = [{"scenario_id": scn, "strength": int(s),
            "margin": float(m)}
           for scn, g in df.groupby("scenario_id")
           for s, m in g.groupby("context_strength")
           ["margin_full_a_minus_b"].mean().items()]
    return save_fig(fig, out_dir, name, tier=tier, source_rows=src,
                    run_label=run_label)


def fig_mech_controls(mech: dict, out_dir, *, tier, run_label="",
                      name="f11_mechanism_patch_and_dose_controls"):
    h = mech.get("holdout", {})
    bars = [("patch A−B", h.get("m1_patch_contrast_mean")),
            ("addition ±d", h.get("m2_addition_pm_mean")),
            ("removal", h.get("m3_removal_mean")),
            ("heldout codebook", h.get("m4_reserved_pm_mean")),
            ("final token", h.get("final_token_pm_mean")),
            ("wrong site", h.get("wrong_site_pm_mean"))]
    ctl = h.get("controls_pm_mean", {})
    for k, v in ctl.items():
        bars.append((k, v))
    bars.append(("random max|·|", h.get("randoms_pm_max_abs")))
    bars = [(k, v) for k, v in bars if v is not None and np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(bars) + 1.2))
    y = np.arange(len(bars))[::-1]
    for yi, (k, v) in zip(y, bars):
        primary = k in ("patch A−B", "addition ±d", "heldout codebook")
        ax.barh(yi, v, color=PAL["MECH"] if primary else PAL["muted"],
                height=0.62)
        ax.annotate(f"{v:+.2f}", (v, yi), textcoords="offset points",
                    xytext=(4, -3), fontsize=7, color=PAL["ink"])
    ax.axvline(0, color=PAL["muted"], lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _ in bars], fontsize=8)
    _style(ax, xlabel="margin movement (nats)",
           title=f"Mechanism assays vs controls — {mech.get('scenario_id')}")
    return save_fig(fig, out_dir, name, tier=tier,
                    source_rows=[{"assay": k, "value": v} for k, v in bars],
                    run_label=run_label)


def fig_result_ladder(statuses: dict[str, str], out_dir, *, tier,
                      run_label="", name="f14_result_ladder"):
    order = ["ENACTED_CHOICE", "SEMANTIC_MARGIN", "CONTEXTUAL_VALUE",
             "CHOICE_REPORT_COUPLED", "BEHAVIOR_SPECIFIC", "MARGIN_HANDLE",
             "DIRECT_OUTPUT", "CLEAN_NULL", "SURFACE_POLICY_ONLY",
             "INSTRUMENT_FAILURE"]
    counts = {}
    for s in statuses.values():
        counts[s] = counts.get(s, 0) + 1
    rows = [(k, counts.get(k, 0)) for k in order if counts.get(k, 0)]
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, (k, n) in zip(y, rows):
        ax.barh(yi, n, color=PAL["ARB"], height=0.6)
        ax.annotate(str(n), (n, yi), textcoords="offset points",
                    xytext=(4, -3), fontsize=8, color=PAL["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _ in rows], fontsize=8)
    _style(ax, xlabel="scenarios", title="Result ladder (plan §4 taxonomy)")
    return save_fig(fig, out_dir, name, tier=tier,
                    source_rows=[{"status": k, "n": n} for k, n in rows],
                    run_label=run_label)


def fig_cross_model_heatmap(matrix: dict[str, dict[str, float]], out_dir,
                            *, tier, run_label="",
                            name="f09_cross_model_semantic_heatmap"):
    models = list(matrix)
    scenarios = sorted({s for m in matrix.values() for s in m})
    data = np.array([[matrix[m].get(s, np.nan) for s in scenarios]
                     for m in models])
    fig, ax = plt.subplots(figsize=(0.6 * len(scenarios) + 2.5,
                                    0.55 * len(models) + 1.6))
    lim = np.nanmax(np.abs(data)) or 1.0
    im = ax.imshow(data, cmap="RdBu_r", vmin=-lim, vmax=lim,
                   aspect="auto")
    for i in range(len(models)):
        for j in range(len(scenarios)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:+.2f}", ha="center",
                        va="center", fontsize=7,
                        color=PAL["ink"])
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace("arb_", "") for s in scenarios],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title("Neutral semantic margins across models (nats, A−B)",
                 fontsize=10, loc="left", fontweight="bold",
                 color=PAL["ink"])
    fig.colorbar(im, ax=ax, shrink=0.7)
    src = [{"model": m, "scenario": s, "margin": matrix[m].get(s)}
           for m in models for s in scenarios]
    return save_fig(fig, out_dir, name, tier=tier, source_rows=src,
                    run_label=run_label)
