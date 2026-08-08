"""Report figures. Every figure regenerates from committed/registered JSON
without loading a model (plan §12/§17.4).

Palette: validated reference categorical slots (J-lens blue #2a78d6,
logit-lens orange #eb6834, aqua #1baf7a third); sequential = single-hue
blues; text in ink tokens, thin marks, one axis per chart.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 150,
})

EVAL_ORDER = ["lens-eval-multihop", "lens-eval-multilingual",
              "lens-eval-poetry", "lens-eval-typo", "lens-eval-order-ops",
              "lens-eval-association"]


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def fig_eval_passk(lane_dir: Path, out: Path, *, lane: str,
                   prefix: str = "eval_") -> None:
    """Grouped dot-range chart: pass@1/5/20, J-lens vs logit lens per set."""
    fig, axes = plt.subplots(1, 6, figsize=(11, 2.9), sharey=True)
    for ax, set_name in zip(axes, EVAL_ORDER):
        data = _load(lane_dir / f"{prefix}{set_name}.json")
        ks = [1, 5, 20]
        x = np.arange(3)
        j = [data["aggregate_jlens"]["token_valid"][f"pass@{k}"] for k in ks]
        l = [data["aggregate_logit"]["token_valid"][f"pass@{k}"] for k in ks]
        ax.plot(x - 0.08, j, "o-", color=BLUE, lw=2, ms=5, label="Jacobian lens")
        ax.plot(x + 0.08, l, "o-", color=ORANGE, lw=2, ms=5, label="logit lens")
        ax.set_xticks(x, [f"@{k}" for k in ks])
        ax.set_ylim(0, 1.0)
        ax.set_title(set_name.replace("lens-eval-", ""), fontsize=9)
        att = data["aggregate_jlens"]["attrition"]
        ax.text(0.5, -0.32, f"n={data['n_items']}"
                + (f" (−{att['gated_intermediates']} tok-gated)"
                   if att["gated_intermediates"] else ""),
                transform=ax.transAxes, ha="center", fontsize=7, color=INK2)
    axes[0].set_ylabel("pass@k (token-valid)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(f"Six released lens evaluations — {lane}, paper-grid "
                 "(24 source layers), min-over-layers rank", y=1.04)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_layer_profiles(lane_dir: Path, out: Path, *, lane: str) -> None:
    """Median rank of intermediates per layer: where in depth each lens
    recovers content (log rank; lower is better)."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4), sharex=True)
    for ax, set_name in zip(axes.flat, EVAL_ORDER):
        data = _load(lane_dir / f"eval_{set_name}.json")
        layers = data["source_layers"]
        j_ranks = {layer: [] for layer in layers}
        l_ranks = {layer: [] for layer in layers}
        for row in data["rows"]:
            for inter in row["intermediates"]:
                if not inter["tokenization_valid"]:
                    continue
                for layer in layers:
                    j_ranks[layer].append(inter["jlens_rank_per_layer"][str(layer)]
                                          if str(layer) in inter["jlens_rank_per_layer"]
                                          else inter["jlens_rank_per_layer"][layer])
                    l_ranks[layer].append(inter["logit_rank_per_layer"][str(layer)]
                                          if str(layer) in inter["logit_rank_per_layer"]
                                          else inter["logit_rank_per_layer"][layer])
        j_med = [np.median(j_ranks[layer]) for layer in layers]
        l_med = [np.median(l_ranks[layer]) for layer in layers]
        ax.plot(layers, j_med, color=BLUE, lw=2, label="Jacobian lens")
        ax.plot(layers, l_med, color=ORANGE, lw=2, label="logit lens")
        ax.set_yscale("log")
        ax.axvspan(24, 58, color=BLUE, alpha=0.06, lw=0)
        ax.set_title(set_name.replace("lens-eval-", ""), fontsize=9)
        ax.invert_yaxis()
    for ax in axes[1]:
        ax.set_xlabel("layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("median rank (log)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Per-layer median intermediate rank — {lane} "
                 "(shaded: paper workspace band 24–58)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_verbal_report(lane_dir: Path, out: Path, *, lane: str) -> None:
    data = _load(lane_dir / f"verbal_report_{lane}.json")
    cats = data["categories"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    names = [c["category"] for c in cats]
    top5 = [(c["top5_rate"] or 0) for c in cats]
    top1 = [(c["top1_rate"] or 0) for c in cats]
    y = np.arange(len(names))
    ax1.barh(y, top5, height=0.62, color=BLUE, alpha=0.45, label="top-5")
    ax1.barh(y, top1, height=0.62, color=BLUE, label="top-1")
    ax1.set_yticks(y, names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("swap success rate (executed trials)")
    ax1.set_xlim(0, 1)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax1.set_title("verbal report: swapped-in candidate reaches rank 1 / top-5")
    before, after = [], []
    for c in cats:
        for t in c["trials"]:
            if t.get("state") == "EXECUTED":
                before.append(t["rank_before"])
                after.append(t["rank_after"])
    lim = max(max(before), max(after)) * 1.5
    ax2.scatter(before, after, s=14, color=BLUE, alpha=0.6, edgecolors="none")
    ax2.plot([1, lim], [1, lim], color=INK2, lw=1, ls="--")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(0.8, lim); ax2.set_ylim(0.8, lim)
    ax2.set_xlabel("candidate rank before swap (log)")
    ax2.set_ylabel("rank after (log)")
    below = sum(1 for b, a in zip(before, after) if a < b)
    ax2.set_title(f"rank shift per trial (n={len(before)}; "
                  f"{below} improved, {len(before)-below} not)")
    fig.suptitle(f"Verbal report — {lane}, α=1, paper band", y=1.03)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_flexgen(lane_dir: Path, out: Path, *, lane: str) -> None:
    data = _load(lane_dir / f"flexible_generalization_{lane}.json")
    cats = data["categories"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, c in zip(axes, cats):
        funcs = sorted({cell["function"] for cell in c["cells"]})
        args = None
        for cell in c["cells"]:
            pass
        args = sorted({cell["source_arg"] for cell in c["cells"]})
        grid = np.full((len(funcs), len(args) * (len(args) - 1)), np.nan)
        pair_labels = []
        pairs = [(s, t) for s in args for t in args if s != t]
        for j, (s, t) in enumerate(pairs):
            pair_labels.append(f"{s[:3]}→{t[:3]}")
            for i, f in enumerate(funcs):
                for cell in c["cells"]:
                    if (cell["function"] == f and cell["source_arg"] == s
                            and cell["target_arg"] == t):
                        a1 = cell.get("alpha1", {})
                        if a1.get("state") == "EXECUTED":
                            grid[i, j] = 1.0 if a1["top1"] else 0.0
        cmap = matplotlib.colors.ListedColormap(["#dce8f8", BLUE])
        ax.imshow(np.nan_to_num(grid, nan=-1), cmap=cmap, vmin=0, vmax=1,
                  aspect="auto")
        for i in range(len(funcs)):
            for j in range(len(pairs)):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "·", ha="center", va="center", color=INK2)
        ax.set_yticks(range(len(funcs)), funcs, fontsize=7)
        ax.set_xticks(range(len(pairs)), pair_labels, fontsize=5.5, rotation=90)
        ax.set_title(f"{c['category']}  (cap.α1={c['capable_top1_alpha1'] if c['capable_top1_alpha1'] is not None else '—'})",
                     fontsize=9)
        ax.grid(False)
    fig.suptitle(f"Flexible generalization — {lane}: function × ordered arg "
                 "swap, filled = target answer reached rank 1 (α=1; · = gated)",
                 y=1.06)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_probe_swap(lane_dir: Path, out: Path, *, lane: str) -> None:
    data = _load(lane_dir / f"probe_swap_{lane}.json")
    rows = [r for r in data["rows"] if r.get("state") == "EXECUTED"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4),
                                   gridspec_kw={"width_ratios": [1, 1.2]})
    before = [r["swap_answer_rank_before"] for r in rows]
    after = [r["swap_answer_rank_after"] for r in rows]
    colors = [BLUE if r["baseline_correct"] else INK2 for r in rows]
    lim = max(max(before), max(after)) * 1.5
    ax1.scatter(before, after, s=16, c=colors, alpha=0.65, edgecolors="none")
    ax1.plot([1, lim], [1, lim], color=INK2, lw=1, ls="--")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("swap_answer rank before (log)")
    ax1.set_ylabel("rank after (log)")
    ax1.set_title("blue = baseline-capable; gray = diagnostic-only")
    groups = {"multihop": [], "other": []}
    for r in rows:
        key = "multihop" if r["category"] == "multihop" else "other"
        groups[key].append(r)
    labels, values, ns = [], [], []
    for key in ("multihop", "other"):
        capable = [r for r in groups[key] if r["baseline_correct"]]
        rate = (sum(1 for r in capable if r["top1_success"]) / len(capable)
                if capable else 0)
        labels.append(f"{key}\n(n={len(capable)} capable)")
        values.append(rate)
    overall = data["capable_top1"] or 0
    labels.append(f"all\n(n={data['n_baseline_capable']})")
    values.append(overall)
    ax2.bar(range(3), values, width=0.55, color=[BLUE, AQUA, INK2])
    for i, v in enumerate(values):
        ax2.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    ax2.set_xticks(range(3), labels, fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("top-1 swap success (capable)")
    ax2.set_title(f"paper context: 60% (Claude n=90) — descriptive only")
    fig.suptitle(f"Probe-swap raw J-lens token arm — {lane}, α=1 "
                 f"(prompt-exact, representation-adapted)", y=1.03)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_slice(lane_dir: Path, out: Path, *, lane: str, slug: str) -> None:
    """Layer × position J-lens top-1 token grid for one example, with the
    tracked-word min-rank profile beside it."""
    data = _load(lane_dir / f"slice_examples_{lane}.json")
    example = next(e for e in data["examples"] if e["slug"] == slug)
    grid = example["grid"]
    layers = [g["layer"] for g in grid]
    positions = example["positions"]
    tokens = example["position_tokens"]
    n_show = min(len(positions), 12)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    ax1.set_xlim(-0.5, n_show - 0.5)
    ax1.set_ylim(-0.5, len(layers) - 0.5)
    tracked_words = list(example["tracked_token_ids"])
    palette = {tracked_words[i]: [BLUE, ORANGE, AQUA, MAGENTA][i % 4]
               for i in range(len(tracked_words))}
    for yi, g in enumerate(grid):
        for xi, cell in enumerate(g["cells"][-n_show:]):
            word = cell["top1"].strip()
            color = INK2
            weight = "normal"
            for target, c in palette.items():
                if word.lower() == target.lower():
                    color = c
                    weight = "bold"
            ax1.text(xi, yi, word[:9], ha="center", va="center", fontsize=6,
                     color=color, fontweight=weight)
    ax1.set_yticks(range(len(layers)), layers, fontsize=6)
    ax1.set_xticks(range(n_show),
                   [t.replace("\n", "⏎")[:8] for t in tokens[-n_show:]],
                   fontsize=6, rotation=45, ha="right")
    ax1.set_ylabel("layer")
    ax1.set_title(f"J-lens top-1 token per (layer, position) — "
                  f"“{example['prompt'][:52]}…”" if len(example["prompt"]) > 52
                  else f"J-lens top-1 token per (layer, position) — "
                       f"“{example['prompt']}”", fontsize=9)
    ax1.grid(False)
    for word in tracked_words:
        profile = []
        for g in grid:
            ranks = [c["tracked_ranks"][word] for c in g["cells"]
                     if c["tracked_ranks"][word] is not None]
            profile.append(min(ranks) if ranks else np.nan)
        ax2.plot(layers, profile, lw=2, color=palette[word], label=word)
    ax2.set_yscale("log")
    ax2.invert_yaxis()
    ax2.axvspan(24, 58, color=BLUE, alpha=0.06, lw=0)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("min rank over shown positions (log)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("tracked-word rank profile", fontsize=9)
    fig.suptitle(f"{example['note']} — {lane}", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def build_all(lane_dir: Path, figures_dir: Path, *, lane: str) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    built = []
    jobs = [
        (fig_eval_passk, f"or1f01_evals_{lane}.pdf", {}),
        (fig_layer_profiles, f"or1f02_layer_profiles_{lane}.pdf", {}),
        (fig_verbal_report, f"or1f03_verbal_report_{lane}.pdf", {}),
        (fig_flexgen, f"or1f04_flexgen_{lane}.pdf", {}),
        (fig_probe_swap, f"or1f05_probe_swap_{lane}.pdf", {}),
        (fig_slice, f"or1f06_slice_boot_{lane}.pdf", {"slug": "boot-currency"}),
        (fig_slice, f"or1f07_slice_amazon_{lane}.pdf", {"slug": "amazon-language"}),
    ]
    for func, name, kwargs in jobs:
        target = figures_dir / name
        try:
            func(lane_dir, target, lane=lane, **kwargs)
            built.append(target)
        except FileNotFoundError as missing:
            print(f"skip {name}: missing input {missing}")
    return built


if __name__ == "__main__":
    import sys

    from .paths import DRIVE_ROOT, REPORTS

    lane = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    outputs = build_all(DRIVE_ROOT / f"{lane}_lane", REPORTS / "figures",
                        lane=lane)
    for path in outputs:
        print("built", path)
