# Phase 3 figure conventions.
#
# Fixed entity->hue palette (conduct rule: color follows the entity,
# never rank; assigned in fixed order, never cycled). The 5-arm
# categorical set was machine-validated (dataviz six checks, light
# surface #fcfcfb): lightness band PASS, chroma PASS, CVD worst adjacent
# dE 12.0 protan / 5.7 tritan with mandatory secondary encoding
# (direct labels + facets), normal-vision floor 17.3 PASS, contrast PASS.
#
# Arms (the Phase 3 entities):
#   meanJ_label_protected  #2a78d6   (inherits Phase 2 J blue)
#   meanJ_span_safe        #0f9d8f   (NEW teal — the span-safe scalpel)
#   matched_control        #eb6834   (inherits control orange; exact
#                                     instant rank+energy match)
#   overlap_matched        #8a5cf5   (violet)
#   persistent_matched     #c23b6f   (rose)
# Diagnostics (never load-bearing color; direct-labeled): mechanics
# random / logit arms in muted slate.
#
# Models are FACETS wherever arms are hues; a chart never uses both
# dimensions as color.
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAL3 = {
    "J": "#2a78d6",
    "span_safe": "#0f9d8f",
    "matched": "#eb6834",
    "overlap_matched": "#8a5cf5",
    "persistent": "#c23b6f",
    "diag": "#5b6472",
    "ink": "#1f2430", "muted": "#8a90a0", "grid": "#e4e6eb",
    "tail": "#c23b6f",          # tail-membership accent (status-like)
    "surface": "#fcfcfb",
}

ARM_LABELS = {
    "meanJ_protected": "mean-J (label-protected)",
    "meanJ_label_protected": "mean-J (label-protected)",
    "meanJ_span_safe": "mean-J (span-safe)",
    "matched_control": "matched control (rank+energy)",
    "overlap_matched_control": "overlap-matched control",
    "persistent_matched_control": "persistent matched control",
    "dynR_mechanics_control": "mechanics random",
    "logit_protected": "logit (label-protected)",
    "logit_span_safe": "logit (span-safe)",
    "baseline": "baseline",
}

SHORT_MODEL = {"olmo31-think": "OLMo 3.1 Think",
               "olmo31-instruct": "OLMo 3.1 Instruct",
               "qwen36-27b": "Qwen 3.6 27B",
               "olmo3-think": "OLMo 3.0 Think (pilot)",
               "olmo3-base": "OLMo 3 base"}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": PAL3["surface"],
        "axes.facecolor": PAL3["surface"],
        "axes.edgecolor": PAL3["muted"], "axes.labelcolor": PAL3["ink"],
        "text.color": PAL3["ink"], "xtick.color": PAL3["ink"],
        "ytick.color": PAL3["ink"], "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": PAL3["grid"],
        "grid.linewidth": 0.8, "font.size": 9.5,
        "axes.titlesize": 10, "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save_fig(fig, out_dir: Path, name: str, tier: str) -> list[Path]:
    """Every Phase 3 figure carries its tier watermark and is written as
    PNG (report/handout) + PDF (paper)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.text(0.995, 0.005, tier.upper(), ha="right", va="bottom",
             fontsize=7, color=PAL3["muted"], alpha=0.9)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths
