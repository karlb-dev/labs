#!/usr/bin/env python
"""Regenerate the synthesis figures used by ``olmo_lineage.tex``.

The script reads only registered Phase 2/4 artifacts from the unpacked
``interpretability/jspace_runs`` mirror. It does not refit a lens or rerun a
model. Point estimates and intervals in the arm-decomposition and capacity
figures are copied from the registered analysis payloads rather than
re-estimated here.

Example:
    interpretability/jspace_runs/.venv/bin/python \
        interpretability/jspace_paper/scripts/olmo_lineage_plots.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT = Path(__file__).resolve()
INTERPRETABILITY = SCRIPT.parents[2]
DEFAULT_RUNS = INTERPRETABILITY / "jspace_runs"
DEFAULT_OUT = SCRIPT.parents[1] / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DIRECT = "#2a78d6"
COMPOSED = "#eb6834"
CONTRAST = "#4a3aa7"
CONTROL = "#777570"
INSTRUCT = "#16856b"

ORDER = ["olmo3-base", "olmo3-think", "olmo31-think", "olmo31-instruct"]
LABEL = {
    "olmo3-base": "Base",
    "olmo3-think": "3.0 Think",
    "olmo31-think": "3.1 Think",
    "olmo31-instruct": "3.1 Instruct\n(sibling)",
}
XPOS = {
    "olmo3-base": 0.0,
    "olmo3-think": 1.0,
    "olmo31-think": 2.0,
    "olmo31-instruct": 3.35,
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "legend.frameon": False,
        }
    )


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing registered inputs:\n  {joined}")


def save(fig: mpl.figure.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "jspace_paper/scripts/olmo_lineage_plots.py"},
    )
    plt.close(fig)
    print(f"wrote {path}")


def plot_emergence(trajectory_path: Path, out_dir: Path) -> None:
    trajectory = pd.read_csv(trajectory_path)
    panels = [
        ("S:direct", "Bank S / direct specificity", DIRECT),
        ("S:composed", "Bank S / composed specificity", COMPOSED),
        ("S:composition", "Bank S / composed - direct", CONTRAST),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharex=True)
    for ax, (metric_key, title, color) in zip(axes, panels):
        for frame, linestyle, face, dx in [
            ("own", "-", color, -0.06),
            ("common", "--", SURFACE, 0.06),
        ]:
            rows = (
                trajectory[
                    (trajectory.metric_key == metric_key)
                    & (trajectory.frame == frame)
                ]
                .set_index("checkpoint_key")
                .loc[ORDER]
                .reset_index()
            )
            x = [XPOS[checkpoint] + dx for checkpoint in rows.checkpoint_key]
            ax.errorbar(
                x,
                rows.estimate,
                yerr=[
                    rows.estimate - rows.ci95_low,
                    rows.ci95_high - rows.estimate,
                ],
                fmt="none",
                ecolor=color,
                elinewidth=1.6,
                capsize=0,
                alpha=0.85,
            )
            ax.plot(x[:3], rows.estimate[:3], linestyle, color=color, lw=1.6, alpha=0.7)
            for index, checkpoint in enumerate(rows.checkpoint_key):
                marker = "s" if checkpoint == "olmo31-instruct" else "o"
                ax.plot(
                    x[index],
                    rows.estimate[index],
                    marker,
                    ms=8,
                    color=color,
                    markerfacecolor=face,
                    markeredgecolor=color,
                    markeredgewidth=1.6,
                    zorder=3,
                    label=f"{frame} lens" if index == 0 else None,
                )
        ax.axhline(0, color=BASELINE, lw=1, zorder=1)
        ax.set_title(title)
        ax.set_xticks(
            [XPOS[checkpoint] for checkpoint in ORDER],
            [LABEL[checkpoint] for checkpoint in ORDER],
            fontsize=8.5,
            color=INK,
        )
    axes[0].set_ylabel("J-specific effect (nats)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Bank-S J-specificity localizes to the base-to-Think interval "
        "and is absent at the Instruct endpoint",
        fontsize=10.5,
        color=INK,
        y=1.03,
    )
    fig.tight_layout()
    save(fig, out_dir / "olmoL1_emergence_trajectory.png")


def plot_capability(g5_paths: dict[str, Path], out_dir: Path) -> None:
    records = []
    for checkpoint, path in g5_paths.items():
        rows = pd.read_parquet(path)
        rows = rows[rows.variant.isin(["direct", "composed"])]
        for (bank, variant), group in rows.groupby(["bank", "variant"]):
            records.append(
                {
                    "checkpoint": checkpoint,
                    "bank": bank,
                    "variant": variant,
                    "rate": group.capable_generation.mean(),
                    "n": len(group),
                }
            )
    capability = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
    for ax, bank in zip(axes, ["F", "S"]):
        for variant, color in [("direct", DIRECT), ("composed", COMPOSED)]:
            rows = (
                capability[
                    (capability.bank == bank) & (capability.variant == variant)
                ]
                .set_index("checkpoint")
                .loc[ORDER]
                .reset_index()
            )
            x = [XPOS[checkpoint] for checkpoint in rows.checkpoint]
            ax.plot(x[:3], rows.rate[:3], "-", color=color, lw=1.6, alpha=0.7)
            for index, checkpoint in enumerate(rows.checkpoint):
                marker = "s" if checkpoint == "olmo31-instruct" else "o"
                ax.plot(
                    x[index],
                    rows.rate[index],
                    marker,
                    ms=8,
                    color=color,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.6,
                    zorder=3,
                    label=variant if index == 0 else None,
                )
        ax.set_title(f"Bank {bank} / G5 capable rate")
        ax.set_xticks(
            [XPOS[checkpoint] for checkpoint in ORDER],
            [LABEL[checkpoint] for checkpoint in ORDER],
            fontsize=8.5,
            color=INK,
        )
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("capable generation rate")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(
        "Capability across the lineage (prospective prefix-disjoint G5 scoring)",
        fontsize=10.5,
        color=INK,
        y=1.04,
    )
    fig.tight_layout()
    save(fig, out_dir / "olmoL2_capability_trajectory.png")
    print(
        capability.pivot_table(
            index="checkpoint", columns=["bank", "variant"], values="rate"
        )
        .loc[ORDER]
        .round(4)
    )


def load_analysis_payload(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)["payload"]


def plot_arm_decomposition(
    analysis_paths: dict[str, Path], out_dir: Path
) -> None:
    payloads = {
        checkpoint: load_analysis_payload(path)
        for checkpoint, path in analysis_paths.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    for ax, variant in zip(axes, ["direct", "composed"]):
        for effect_key, label, color, face, dx in [
            ("J_effect", "span-safe J arm", CONTRAST, CONTRAST, -0.06),
            ("control_effect", "matched control", CONTROL, SURFACE, 0.06),
        ]:
            estimates = []
            lows = []
            highs = []
            for checkpoint in ORDER:
                effect = payloads[checkpoint]["effects_by_bank_variant"][
                    f"S:{variant}"
                ]["family_weighted"][effect_key]
                estimates.append(effect["estimate"])
                lows.append(effect["ci95"][0])
                highs.append(effect["ci95"][1])
            x = [XPOS[checkpoint] + dx for checkpoint in ORDER]
            ax.errorbar(
                x,
                estimates,
                yerr=[
                    [estimate - low for estimate, low in zip(estimates, lows)],
                    [high - estimate for estimate, high in zip(estimates, highs)],
                ],
                fmt="none",
                ecolor=color,
                elinewidth=1.6,
                capsize=0,
                alpha=0.9,
            )
            ax.plot(x[:3], estimates[:3], "-", color=color, lw=1.6, alpha=0.7)
            for index, checkpoint in enumerate(ORDER):
                marker = "s" if checkpoint == "olmo31-instruct" else "o"
                ax.plot(
                    x[index],
                    estimates[index],
                    marker,
                    ms=8,
                    color=color,
                    markerfacecolor=face,
                    markeredgecolor=color,
                    markeredgewidth=1.6,
                    zorder=3,
                    label=label if index == 0 else None,
                )
        ax.axhline(0, color=BASELINE, lw=1)
        ax.set_title(f"Bank S / {variant}")
        ax.set_xticks(
            [XPOS[checkpoint] for checkpoint in ORDER],
            [LABEL[checkpoint] for checkpoint in ORDER],
            fontsize=8.5,
            color=INK,
        )
    axes[0].set_ylabel("Delta answer log-probability (nats)")
    axes[0].legend(loc="lower left", fontsize=8.2)
    fig.suptitle(
        "The Think-path change is in the lens-selected arm, not the matched dose",
        fontsize=10.5,
        color=INK,
        y=1.04,
    )
    fig.tight_layout()
    save(fig, out_dir / "olmoL3_bank_s_arm_decomposition.png")


def plot_capacity(r2_paths: dict[str, Path], out_dir: Path) -> None:
    labels = {
        "olmo3-think": "3.0 Think",
        "olmo31-think": "3.1 Think",
        "olmo31-instruct": "3.1 Instruct",
    }
    styles = {
        "olmo3-think": (DIRECT, "o"),
        "olmo31-think": (COMPOSED, "o"),
        "olmo31-instruct": (INSTRUCT, "s"),
    }
    layers = [24, 32, 40]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for checkpoint in ["olmo3-think", "olmo31-think", "olmo31-instruct"]:
        with r2_paths[checkpoint].open() as handle:
            per_layer = json.load(handle)["payload"]["per_layer"]
        estimates = [
            100.0 * per_layer[str(layer)]["centered_variance_explained_excess"]
            for layer in layers
        ]
        lows = [
            100.0 * per_layer[str(layer)]["centered_excess_ci"]["low"]
            for layer in layers
        ]
        highs = [
            100.0 * per_layer[str(layer)]["centered_excess_ci"]["high"]
            for layer in layers
        ]
        color, marker = styles[checkpoint]
        ax.errorbar(
            layers,
            estimates,
            yerr=[
                [estimate - low for estimate, low in zip(estimates, lows)],
                [high - estimate for estimate, high in zip(estimates, highs)],
            ],
            color=color,
            marker=marker,
            ms=7,
            lw=1.8,
            elinewidth=1.4,
            capsize=3,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=labels[checkpoint],
        )
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylabel("centered R-squared excess (%)")
    ax.set_ylim(0.25, 1.45)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.text(
        0.99,
        0.05,
        "Occupancy median = 2 in every cell\nBase checkpoint not measured",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=INK2,
    )
    ax.set_title(
        "Measured J-space capacity is nearly unchanged across OLMo post-training",
        pad=10,
    )
    fig.tight_layout()
    save(fig, out_dir / "olmoL4_capacity_posttraining.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    runs = args.runs_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    p4_metrics = runs / "phase4" / "metrics"
    trajectory_path = (
        p4_metrics
        / "olmo-lineage-trajectory"
        / "trajectory_analysis"
        / "p4-lineage-trajectory-analysis-olmo-dev-v1"
        / "trajectory_table_olmo-lineage-trajectory.csv"
    )
    g5_paths = {
        "olmo3-base": p4_metrics
        / "olmo3-base/g5_bank/p4-g5-bank-olmo3-base-dev-v1/g5_bank_olmo3-base.parquet",
        "olmo3-think": p4_metrics
        / "olmo3-think/g5_bank/p4-g5-bank-olmo3-think-dev-v2/g5_bank_olmo3-think.parquet",
        "olmo31-think": p4_metrics
        / "olmo31-think/g5_bank/p4-g5-bank-olmo31-think-dev-v1/g5_bank_olmo31-think.parquet",
        "olmo31-instruct": p4_metrics
        / "olmo31-instruct/g5_bank/p4-g5-bank-olmo31-instruct-dev-v1/g5_bank_olmo31-instruct.parquet",
    }
    analysis_paths = {
        "olmo3-base": p4_metrics
        / "olmo3-base/lineage_analysis/p4-lineage-analysis-olmo3-base-dev-v1/lineage_analysis_olmo3-base.json",
        "olmo3-think": p4_metrics
        / "olmo3-think/lineage_analysis/p4-lineage-analysis-olmo3-think-dev-v1/lineage_analysis_olmo3-think.json",
        "olmo31-think": p4_metrics
        / "olmo31-think/lineage_analysis/p4-lineage-analysis-olmo31-think-dev-v2/lineage_analysis_olmo31-think.json",
        "olmo31-instruct": p4_metrics
        / "olmo31-instruct/lineage_analysis/p4-lineage-analysis-olmo31-instruct-dev-v1/lineage_analysis_olmo31-instruct.json",
    }
    r2_paths = {
        checkpoint: runs
        / "part2"
        / "metrics"
        / checkpoint
        / "r2_occupancy"
        / "r2_occupancy_v2.json"
        for checkpoint in ["olmo3-think", "olmo31-think", "olmo31-instruct"]
    }
    require_files(
        [trajectory_path]
        + list(g5_paths.values())
        + list(analysis_paths.values())
        + list(r2_paths.values())
    )

    configure_matplotlib()
    plot_emergence(trajectory_path, out_dir)
    plot_capability(g5_paths, out_dir)
    plot_arm_decomposition(analysis_paths, out_dir)
    plot_capacity(r2_paths, out_dir)


if __name__ == "__main__":
    main()
