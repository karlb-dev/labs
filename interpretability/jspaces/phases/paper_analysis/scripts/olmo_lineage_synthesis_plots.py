#!/usr/bin/env python3
"""Regenerate synthesis figures for the OLMo J-space lineage note.

The primary input is the registered Phase 4 trajectory CSV written by
``p4_lineage_trajectory_analysis``:

  <run-root>/metrics/olmo-lineage-trajectory/trajectory_analysis/
    p4-lineage-trajectory-analysis-olmo-dev-v1/
      trajectory_table_olmo-lineage-trajectory.csv

Example
-------
python olmo_lineage_synthesis_plots.py \
  --run-root /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731 \
  --out-dir interpretability/jspaces/phases/paper_analysis/figures

A direct ``--trajectory-csv`` path may be supplied instead. The script never
re-estimates registered intervals; it only reads the registered table and
replots the stored point estimates and confidence limits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

EVIDENCE_ID = "p4-lineage-trajectory-analysis-olmo-dev-v1"
SLUG = "olmo-lineage-trajectory"
EXPECTED_CHECKPOINTS = [
    "olmo3-base",
    "olmo3-think",
    "olmo31-think",
    "olmo31-instruct",
]
DISPLAY = {
    "olmo3-base": "Base",
    "olmo3-think": "3.0 Think",
    "olmo31-think": "3.1 Think",
    "olmo31-instruct": "3.1 Instruct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Phase 4 run root containing metrics/ and figures/.",
    )
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        help="Direct path to trajectory_table_olmo-lineage-trajectory.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory for PNG/PDF outputs (default: current directory).",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def resolve_trajectory_csv(args: argparse.Namespace) -> Path:
    if args.trajectory_csv is not None:
        path = args.trajectory_csv
    elif args.run_root is not None:
        path = (
            args.run_root
            / "metrics"
            / SLUG
            / "trajectory_analysis"
            / EVIDENCE_ID
            / f"trajectory_table_{SLUG}.csv"
        )
    else:
        raise SystemExit("Pass either --run-root or --trajectory-csv.")
    if not path.is_file():
        raise SystemExit(f"Trajectory CSV not found: {path}")
    return path


def load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "checkpoint_key",
        "checkpoint_label",
        "checkpoint_index",
        "lineage_role",
        "frame",
        "metric_key",
        "estimate",
        "ci95_low",
        "ci95_high",
    }
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Trajectory CSV lacks columns: {sorted(missing)}")
    if set(frame["frame"].unique()) != {"own", "common"}:
        raise SystemExit("Expected exactly own/common frames.")
    found = list(
        frame[["checkpoint_key", "checkpoint_index"]]
        .drop_duplicates()
        .sort_values("checkpoint_index")["checkpoint_key"]
    )
    if found != EXPECTED_CHECKPOINTS:
        raise SystemExit(
            "Unexpected checkpoint order: " + ", ".join(found)
        )
    for metric in ("S:direct", "S:composition"):
        count = len(frame[frame.metric_key == metric])
        if count != 8:
            raise SystemExit(f"Expected 8 rows for {metric}, found {count}")
    return frame


def one_value(frame: pd.DataFrame, checkpoint: str, coord: str, metric: str) -> pd.Series:
    subset = frame[
        (frame.checkpoint_key == checkpoint)
        & (frame.frame == coord)
        & (frame.metric_key == metric)
    ]
    if len(subset) != 1:
        raise RuntimeError(
            f"Expected one {checkpoint}/{coord}/{metric} row, found {len(subset)}"
        )
    return subset.iloc[0]


def plot_state_space(frame: pd.DataFrame, out_dir: Path, dpi: int) -> tuple[Path, Path]:
    """Plot direct causal dependence against the composition contrast.

    The figure is intentionally a *state-space* view rather than another
    checkpoint-index plot. Moving left means stronger content-specific damage
    on direct Bank-S prompts; moving up means composed prompts are relatively
    less dependent than direct prompts. Own/common lens coordinates are shown
    together to make coordinate-frame uncertainty visible.
    """
    fig, ax = plt.subplots(figsize=(9.4, 6.8))

    # Matplotlib's default cycle provides the colors; no custom palette is
    # required to reproduce the figure.
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_for = {
        checkpoint: cycle[index % len(cycle)]
        for index, checkpoint in enumerate(EXPECTED_CHECKPOINTS)
    }

    # Draw the observed Think path separately in each coordinate frame.
    think_path = EXPECTED_CHECKPOINTS[:3]
    for coord, linestyle, label in (
        ("own", "-", "own-lens path"),
        ("common", "--", "frozen base-lens path"),
    ):
        xs = [float(one_value(frame, c, coord, "S:direct").estimate) for c in think_path]
        ys = [float(one_value(frame, c, coord, "S:composition").estimate) for c in think_path]
        ax.plot(xs, ys, linestyle=linestyle, linewidth=1.6, alpha=0.7, label=label)

    # Plot every checkpoint/frame with two-dimensional uncertainty.
    for checkpoint in EXPECTED_CHECKPOINTS:
        own = one_value(frame, checkpoint, "own", "S:direct")
        own_y = one_value(frame, checkpoint, "own", "S:composition")
        common = one_value(frame, checkpoint, "common", "S:direct")
        common_y = one_value(frame, checkpoint, "common", "S:composition")
        color = color_for[checkpoint]

        # Small connector = coordinate-frame displacement at fixed weights.
        ax.plot(
            [own.estimate, common.estimate],
            [own_y.estimate, common_y.estimate],
            linewidth=1.0,
            alpha=0.45,
            color=color,
        )

        marker = "s" if checkpoint == "olmo31-instruct" else "o"
        for coord, xrow, yrow, filled in (
            ("own", own, own_y, True),
            ("common", common, common_y, False),
        ):
            xerr = [[xrow.estimate - xrow.ci95_low], [xrow.ci95_high - xrow.estimate]]
            yerr = [[yrow.estimate - yrow.ci95_low], [yrow.ci95_high - yrow.estimate]]
            ax.errorbar(
                float(xrow.estimate),
                float(yrow.estimate),
                xerr=xerr,
                yerr=yerr,
                marker=marker,
                markersize=8.5,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.6,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.5,
                linestyle="none",
                zorder=3,
            )

        dx, dy = {
            "olmo3-base": (0.006, -0.012),
            "olmo3-think": (-0.015, 0.013),
            "olmo31-think": (-0.004, 0.019),
            "olmo31-instruct": (0.007, 0.010),
        }[checkpoint]
        ax.annotate(
            DISPLAY[checkpoint],
            (float(own.estimate), float(own_y.estimate)),
            xytext=(float(own.estimate) + dx, float(own_y.estimate) + dy),
            fontsize=10,
            weight="bold" if checkpoint in {"olmo31-think", "olmo31-instruct"} else "normal",
            arrowprops={"arrowstyle": "-", "linewidth": 0.7, "alpha": 0.55},
        )

    ax.axvline(0, linewidth=0.9, alpha=0.55)
    ax.axhline(0, linewidth=0.9, alpha=0.55)
    ax.set_xlabel("Bank-S direct J-specific effect (nats; more negative = greater dependence)")
    ax.set_ylabel("Bank-S composed minus direct specificity (nats)")
    ax.set_title("Post-training moves OLMo through a causal-use state space")
    ax.text(
        0.01,
        0.01,
        "Filled markers: own lens. Hollow markers: frozen base lens.\n"
        "Lines connect only the observed Think path; the Instruct square is a sibling endpoint.",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Known-bank Phase 4 development estimates, not randomized training or confirmatory evidence",
        fontsize=9,
        y=0.955,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "olmoL5_training_state_space.png"
    pdf = out_dir / "olmoL5_training_state_space.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    source = resolve_trajectory_csv(args)
    frame = load_table(source)
    png, pdf = plot_state_space(frame, args.out_dir, args.dpi)
    print(f"source: {source}")
    print(f"png:    {png}")
    print(f"pdf:    {pdf}")


if __name__ == "__main__":
    main()
