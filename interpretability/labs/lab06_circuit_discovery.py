"""Lab 6: Circuit discovery and validation, the manual way.

This lab composes the previous instrumentation labs (plus the residual-stream
and self-check habits from Lab 1) into a small, earned circuit claim:
- Lab 2: direct-logit attribution (cheap screen)
- Lab 3: attention motifs (previous-token, induction, sink patterns)
- Lab 5: intervention discipline (mean-ablation instead of zero-ablation)

The deliverable is a circuit card with three earned numbers and an explicitly
scoped mechanism sketch (heads-only routing subgraph):

* faithfulness: with every non-circuit head mean-ablated, how much of the
  original behavior remains?
* completeness: with the circuit heads mean-ablated, how much behavior remains?
* minimality: how much faithfulness is lost if each kept node is removed?

The target behavior is induction completion on fixed-length repeating patterns.
The circuit claim is intentionally heads-only: it is a routing subgraph. MLPs
are ranked and reported as supporting infrastructure, but they are not part of
the faithfulness complement. This is the manual baseline that Lab 9 will
confront with an automated attribution graph. Keep the card.

Evidence level: CAUSAL at heads-only circuit scope, on a stated prompt
population and a stated off-distribution (dataset-mean ablation).
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from typing import Any, Iterable, Sequence

import interp_bench as bench
from labs.lab02_direct_logit_attribution import compute_direct_logit_attribution
from labs.lab03_attention_routing import (
    head_attribution_scores,
    induction_score,
    prev_token_score,
)

LAB_ID = "L06"

# Greedy pruning keeps removing heads while the remaining circuit stays above
# this ratio of the base logit-difference metric.
FAITHFULNESS_FLOOR = 0.70

# Screening is deliberately broader than a minimal demo. A too-thin screen can
# make the pruning stage fail before the students learn anything about circuits.
SCREEN_TOP_ATTRIBUTION_MIN = 20
SCREEN_TOP_INDUCTION_MIN = 8
SCREEN_TOP_PREV_MIN = 8
SCREEN_TOP_ATTRIBUTION_FRAC = 0.035
SCREEN_TOP_MOTIF_FRAC = 0.012
N_MLP_CANDIDATES = 6

MOTIF_STRONG_THRESHOLD = 0.35
FIRST_TOKEN_SINK_THRESHOLD = 0.45
EDGE_MIN_SOURCE_EFFECT = 1e-6
# Olmo 7B's strongest ordered interaction in the validation run is small but
# real enough to teach redundancy. Report it as weak at 2%, strong at 5%.
EDGE_MIN_ROUTED_FRACTION = 0.02
EDGE_STRONG_ROUTED_FRACTION = 0.05


@dataclasses.dataclass(frozen=True)
class CircuitPrompt:
    example_id: str
    family: str          # discovery or heldout
    prompt: str
    target: str
    distractor: str


# All prompts are validated at runtime. The strings are chosen so the default
# course models tokenize the prompt into exactly 8 tokens and both answer
# strings into one token. Target = induction continuation. Distractor = cycle
# restart, the most plausible wrong continuation.
ALL_PROMPTS: tuple[CircuitPrompt, ...] = (
    CircuitPrompt("d_colors", "discovery", "red blue green red blue green red blue", " green", " red"),
    CircuitPrompt("d_animals", "discovery", "dog cat bird dog cat bird dog cat", " bird", " dog"),
    CircuitPrompt("d_letters", "discovery", "B F Q B F Q B F", " Q", " B"),
    CircuitPrompt("d_moon", "discovery", "moon star moon star moon star moon star", " moon", " star"),
    CircuitPrompt("d_sun", "discovery", "sun rain sun rain sun rain sun rain", " sun", " rain"),
    CircuitPrompt("d_numbers", "discovery", "seven three nine seven three nine seven three", " nine", " seven"),
    CircuitPrompt("d_fruit", "discovery", "apple pear banana apple pear banana apple pear", " banana", " apple"),
    CircuitPrompt("d_shapes", "discovery", "circle square triangle circle square triangle circle square", " triangle", " circle"),
    CircuitPrompt("d_weather", "discovery", "rain snow wind rain snow wind rain snow", " wind", " rain"),
    # Seasons in a scrambled (non-calendar) order so the continuation cannot come
    # from the seasonal-sequence prior, only from in-context copying.
    CircuitPrompt("d_seasons", "discovery", "spring autumn summer spring autumn summer spring autumn", " summer", " spring"),
    CircuitPrompt("h_metals", "heldout", "gold silver gold silver gold silver gold silver", " gold", " silver"),
    CircuitPrompt("h_compass", "heldout", "north south east north south east north south", " east", " north"),
    CircuitPrompt("h_beasts", "heldout", "wolf bear fox wolf bear fox wolf bear", " fox", " wolf"),
    CircuitPrompt("h_matter", "heldout", "glass stone iron glass stone iron glass stone", " iron", " glass"),
    CircuitPrompt("h_tools", "heldout", "hammer saw drill hammer saw drill hammer saw", " drill", " hammer"),
    CircuitPrompt("h_vehicles", "heldout", "car bus train car bus train car bus", " train", " car"),
    CircuitPrompt("h_foods", "heldout", "wine fish bread wine fish bread wine fish", " bread", " wine"),
)


@dataclasses.dataclass
class TaskExample:
    prompt: CircuitPrompt
    target_id: int
    distractor_id: int
    prompt_len: int
    base_diff: float = 0.0


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def node_name(kind: str, layer: int, head: int | None = None) -> str:
    return f"L{layer}H{head}" if kind == "head" else f"MLP{layer}"


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def safe_ratio(num: float, denom: float) -> float | None:
    if abs(denom) < 1e-9:
        return None
    return float(num) / float(denom)


def rank_map(scores: dict[Any, float], *, reverse: bool = True, key_abs: bool = False) -> dict[Any, int]:
    def sort_key(item: tuple[Any, float]) -> float:
        return abs(item[1]) if key_abs else item[1]

    return {
        key: i + 1
        for i, (key, _score) in enumerate(sorted(scores.items(), key=sort_key, reverse=reverse))
    }


def screen_budgets(n_layers: int, n_heads: int) -> tuple[int, int, int]:
    """Broad but finite screen budgets so students see the disagreement between
    cheap screens (Lab 2 attribution + Lab 3 motifs) and actual causal effect
    (the central pedagogical payload of this lab). Too narrow a screen can make
    the whole exercise look like the model only needed three heads.
    """
    total = n_layers * n_heads
    top_attr = min(total, max(SCREEN_TOP_ATTRIBUTION_MIN, math.ceil(total * SCREEN_TOP_ATTRIBUTION_FRAC)))
    top_ind = min(total, max(SCREEN_TOP_INDUCTION_MIN, math.ceil(total * SCREEN_TOP_MOTIF_FRAC)))
    top_prev = min(total, max(SCREEN_TOP_PREV_MIN, math.ceil(total * SCREEN_TOP_MOTIF_FRAC)))
    return top_attr, top_ind, top_prev


def first_token_score(pattern: Any) -> float:
    """Mean attention mass assigned to token 0, excluding the trivial first row."""
    if pattern.shape[-1] <= 1:
        return 0.0
    return float(pattern[1:, 0].mean())


def mean_logit_diff(bundle: bench.ModelBundle, examples: Sequence[TaskExample]) -> float:
    diffs = []
    for ex in examples:
        logits = bench.run_with_residual_cache(bundle, ex.prompt.prompt).final_logits_last
        diffs.append(float(logits[ex.target_id] - logits[ex.distractor_id]))
    return statistics.fmean(diffs)


def describe_head_list(heads: Sequence[tuple[int, int]]) -> str:
    return ", ".join(node_name("head", *h) for h in heads) or "none"


def edge_strength_label(raw_fraction: float | None) -> str:
    if raw_fraction is None or raw_fraction <= 0:
        return "none"
    if raw_fraction < EDGE_MIN_ROUTED_FRACTION:
        return "below_threshold"
    if raw_fraction < EDGE_STRONG_ROUTED_FRACTION:
        return "weak"
    return "strong"


# ---------------------------------------------------------------------------
# Dataset validation and baseline
# ---------------------------------------------------------------------------


def build_dataset(
    ctx: bench.RunContext, bundle: bench.ModelBundle, max_examples: int
) -> tuple[list[TaskExample], list[TaskExample], int, int]:
    """Validate tokenization, compute baseline logit gaps, and gate discovery prompts."""
    tokenizer = bundle.tokenizer
    all_prompts = list(ALL_PROMPTS)
    if max_examples > 0:
        discovery_subset = [p for p in all_prompts if p.family == "discovery"][:max_examples]
        all_prompts = discovery_subset + [p for p in all_prompts if p.family == "heldout"]

    rows: list[dict[str, Any]] = []
    discovery: list[TaskExample] = []
    heldout: list[TaskExample] = []
    valid_lengths: set[int] = set()

    for cp in all_prompts:
        target_ids = tokenizer.encode(cp.target, add_special_tokens=False)
        distractor_ids = tokenizer.encode(cp.distractor, add_special_tokens=False)
        prompt_ids = tokenizer.encode(cp.prompt, add_special_tokens=False)

        problems: list[str] = []
        if len(target_ids) != 1:
            problems.append(f"target has {len(target_ids)} tokens")
        if len(distractor_ids) != 1:
            problems.append(f"distractor has {len(distractor_ids)} tokens")
        if len(prompt_ids) != 8:
            problems.append(f"prompt has {len(prompt_ids)} tokens, expected 8")

        row: dict[str, Any] = {
            "example_id": cp.example_id,
            "family": cp.family,
            "prompt": cp.prompt,
            "prompt_tokens": " ".join(bench.visible_token(tokenizer.decode([i])) for i in prompt_ids),
            "n_prompt_tokens": len(prompt_ids),
            "target": bench.visible_token(cp.target),
            "distractor": bench.visible_token(cp.distractor),
            "target_id": target_ids[0] if len(target_ids) == 1 else "",
            "distractor_id": distractor_ids[0] if len(distractor_ids) == 1 else "",
            "tokenization_ok": not problems,
            "problems": "; ".join(problems),
        }

        if problems:
            rows.append(row)
            continue

        valid_lengths.add(len(prompt_ids))
        ex = TaskExample(cp, target_ids[0], distractor_ids[0], len(prompt_ids))
        logits = bench.run_with_residual_cache(bundle, cp.prompt).final_logits_last
        ex.base_diff = float(logits[ex.target_id] - logits[ex.distractor_id])
        row.update({
            "baseline_logit_diff": round(ex.base_diff, 4),
            "baseline_pass": ex.base_diff > 0,
        })
        rows.append(row)

        if cp.family == "discovery":
            if ex.base_diff > 0:
                discovery.append(ex)
            else:
                print(f"[lab6] dropping {cp.example_id}: base diff {ex.base_diff:+.2f} <= 0")
        else:
            heldout.append(ex)

    report = ctx.path("diagnostics", "tokenization_and_baseline.csv")
    bench.write_csv(report, rows)
    ctx.register_artifact(
        report,
        "diagnostic",
        "Tokenization contract and baseline logit gap for every Lab 6 prompt.",
    )

    if len(valid_lengths) != 1:
        raise RuntimeError(
            "Dataset contract violated: valid prompts have differing token lengths. "
            "See diagnostics/tokenization_and_baseline.csv."
        )
    if len(discovery) < 3:
        raise RuntimeError(
            f"Only {len(discovery)} discovery prompts pass the baseline gate. "
            "The model does not do this task reliably enough to trace a circuit."
        )

    dropped = sum(
        1
        for row in rows
        if row["family"] == "discovery" and (not row.get("tokenization_ok") or not row.get("baseline_pass"))
    )
    return discovery, heldout, dropped, next(iter(valid_lengths))


# ---------------------------------------------------------------------------
# Metric under intervention
# ---------------------------------------------------------------------------


def metric_under_ablation(
    bundle: bench.ModelBundle,
    examples: Sequence[TaskExample],
    head_anatomy: bench.HeadAnatomy,
    comp_anatomy: bench.ComponentAnatomy,
    heads: Sequence[tuple[int, int]],
    mlps: Sequence[int],
    head_means: Any,
    mlp_means: Any,
) -> float:
    """Mean logit(target) minus logit(distractor) with a node set mean-ablated.

    Dataset-mean ablation (not zero-ablation) keeps the intervention closer to
    the data manifold while still removing the prompt-specific computation of
    the ablated heads. This is the "off switch" that defines the circuit for
    this particular off-distribution. A different mean (or zero) defines a
    different circuit.
    """
    diffs: list[float] = []
    for ex in examples:
        logits = bench.run_with_node_set_ablation(
            bundle,
            ex.prompt.prompt,
            head_anatomy,
            comp_anatomy,
            heads=heads,
            mlps=mlps,
            head_means=head_means,
            mlp_means=mlp_means,
        )
        diffs.append(float(logits[ex.target_id] - logits[ex.distractor_id]))
    return statistics.fmean(diffs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_screen_vs_causal(ctx: bench.RunContext, cand_rows: Sequence[dict[str, Any]]) -> None:
    """Show the central lesson: cheap screens and causal effects disagree."""
    import matplotlib.pyplot as plt

    heads = [r for r in cand_rows if r["kind"] == "head"]
    if not heads:
        return

    fig, (ax_rank, ax_attr) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for ax in (ax_rank, ax_attr):
        ax.grid(True, alpha=0.3)

    colors = {
        "induction": "tab:red",
        "previous_token": "tab:blue",
        "first_token_sink": "tab:green",
        "other": "tab:gray",
    }
    markers = {
        "attribution": "o",
        "induction": "^",
        "prev_token": "s",
        "attribution+induction": "P",
        "attribution+prev_token": "X",
        "induction+prev_token": "D",
        "attribution+induction+prev_token": "*",
    }

    for r in heads:
        color = colors.get(r.get("motif_label", "other"), "tab:gray")
        marker = markers.get(r.get("screen_reason", "attribution"), "o")
        ax_rank.scatter(r["cheap_rank"], r["causal_drop"], s=55, color=color, marker=marker, alpha=0.85)
        ax_attr.scatter(abs(r["mean_attr"]), r["causal_drop"], s=55, color=color, marker=marker, alpha=0.85)

    # Annotate the causal top few and the cheap top few. Labeling every point
    # turns this plot into alphabet soup on a 7B model.
    label_nodes: set[str] = set()
    label_nodes.update(r["node"] for r in sorted(heads, key=lambda x: -x["causal_drop"])[:8])
    label_nodes.update(r["node"] for r in sorted(heads, key=lambda x: x["cheap_rank"])[:6])
    for r in heads:
        if r["node"] in label_nodes:
            ax_rank.annotate(r["node"], (r["cheap_rank"], r["causal_drop"]),
                             textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax_attr.annotate(r["node"], (abs(r["mean_attr"]), r["causal_drop"]),
                             textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax_rank.axhline(0, color="black", linewidth=0.7)
    ax_rank.set_xlabel("cheap screen rank, lower is better")
    ax_rank.set_ylabel("single-head causal drop")
    ax_rank.set_title("Screen rank vs causal effect")
    # Lower ranks are better and therefore appear on the left.

    ax_attr.axhline(0, color="black", linewidth=0.7)
    ax_attr.set_xlabel("absolute direct-logit attribution score")
    ax_attr.set_ylabel("single-head causal drop")
    ax_attr.set_title("Attribution score vs causal effect")

    for label, color in colors.items():
        if any(r.get("motif_label") == label for r in heads):
            ax_attr.scatter([], [], color=color, label=label)
    ax_attr.legend(fontsize=8, loc="best")

    fig.suptitle("Cheap screening is a hypothesis generator, not a circuit claim")
    bench.save_figure(
        ctx,
        fig,
        "screen_vs_causal.png",
        "Cheap screening rank and attribution score against single-head mean-ablation effect.",
    )


def plot_prune_trajectory(
    ctx: bench.RunContext,
    trajectory: Sequence[dict[str, Any]],
    *,
    floor: float,
) -> None:
    if not trajectory:
        return
    fig, ax = bench.new_figure(figsize=(8.5, 5.2))
    xs = [t["n_nodes"] for t in trajectory]
    ys = [t["faithfulness"] for t in trajectory]
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="tab:green")
    for t in trajectory:
        if t.get("removed"):
            ax.annotate(
                f"-{t['removed']}",
                (t["n_nodes"], t["faithfulness"]),
                textcoords="offset points",
                xytext=(2, 8),
                fontsize=7,
                rotation=30,
            )
    ax.axhline(floor, color="tab:red", linewidth=1.0, linestyle="--", label=f"floor = {floor:.2f}")
    ax.axhline(1.0, color="black", linewidth=0.7, alpha=0.45)
    ax.set_xlabel("circuit size, heads kept")
    ax.set_ylabel("faithfulness, complement mean-ablated")
    ax.set_title("Greedy pruning: what the behavior costs, node by node")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "prune_trajectory.png", "Faithfulness at each greedy pruning step.")


def plot_circuit_graph(
    ctx: bench.RunContext,
    circuit: Sequence[tuple[int, int]],
    head_labels: dict[tuple[int, int], str],
    mlp_support: Sequence[dict[str, Any]],
    edge: dict[str, Any] | None,
    n_layers: int,
    n_heads: int,
) -> None:
    fig, ax = bench.new_figure(figsize=(10.0, 6.0))
    colors = {
        "induction": "tab:red",
        "previous_token": "tab:blue",
        "first_token_sink": "tab:green",
        "other": "tab:gray",
    }
    for layer, head in circuit:
        label = head_labels.get((layer, head), "other")
        ax.scatter(layer, head, s=170, color=colors.get(label, "tab:gray"), zorder=3, edgecolors="black")
        ax.annotate(f"L{layer}H{head}\n{label}", (layer, head), textcoords="offset points", xytext=(6, 6), fontsize=7)

    shown_mlps = list(mlp_support[:4])
    for i, r in enumerate(shown_mlps):
        y = n_heads + 1 + 0.65 * (i % 3)
        ax.scatter(r["layer"], y, s=120, marker="s", color="tab:orange", zorder=3, edgecolors="black")
        ax.annotate(
            f"MLP{r['layer']}\ndrop {r['causal_drop']:+.2f}",
            (r["layer"], y),
            textcoords="offset points",
            xytext=(4, 6 + 4 * (i % 2)),
            fontsize=7,
        )
    if len(mlp_support) > len(shown_mlps):
        ax.text(
            0.99,
            0.98,
            f"+{len(mlp_support) - len(shown_mlps)} support MLPs listed in circuit_card.md",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="tab:orange",
        )

    if edge is not None and edge.get("claimed"):
        l1, h1 = edge["from"]
        l2, h2 = edge["to"]
        if (l1, h1) not in circuit:
            ax.scatter(l1, h1, s=170, facecolors="none", edgecolors="tab:blue", linewidths=1.6, zorder=3)
            ax.annotate(
                f"L{l1}H{h1}\nscreened, pruned",
                (l1, h1),
                textcoords="offset points",
                xytext=(6, -16),
                fontsize=7,
                color="tab:blue",
            )
        ax.annotate(
            "",
            xy=(l2, h2),
            xytext=(l1, h1),
            arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.8, "shrinkA": 12, "shrinkB": 12},
        )
        mid_x, mid_y = (l1 + l2) / 2, (h1 + h2) / 2
        ax.annotate(
            f"interaction {edge['raw_interaction_fraction']:.0%}",
            (mid_x, mid_y),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=8,
            color="black",
        )

    ax.set_xlim(-1, n_layers)
    ax.set_ylim(-1.5, n_heads + 4.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("head index; squares above are supporting MLPs")
    ax.set_title("Validated heads-only routing circuit")
    bench.save_figure(ctx, fig, "circuit_graph.png", "Circuit heads, motif labels, supporting MLPs, and claimed edge if any.")


def plot_fcm(ctx: bench.RunContext, fcm: dict[str, Any], *, floor: float) -> None:
    fig, ax = bench.new_figure(figsize=(8.0, 5.2))
    xs: list[float] = []
    heights: list[float] = []
    labels: list[str] = []
    colors: list[str] = []
    alphas: list[float] = []
    for i, metric in enumerate(("faithfulness", "completeness_effect")):
        for j, family in enumerate(("discovery", "heldout")):
            if family not in fcm:
                continue
            xs.append(i * 2.7 + j)
            heights.append(fcm[family][metric])
            labels.append(f"{metric.replace('_', ' ')}\n{family}")
            colors.append("tab:green" if metric == "faithfulness" else "tab:blue")
            alphas.append(1.0 if family == "discovery" else 0.55)
    bars = ax.bar(xs, heights, color=colors)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.axhline(floor, color="tab:red", linewidth=1.0, linestyle="--", label="faithfulness floor")
    ax.axhline(1.0, color="black", linewidth=0.7, alpha=0.35)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("fraction of base behavior")
    ax.set_title("Circuit scorecard: preservation and destruction of behavior")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "circuit_scorecard.png", "Faithfulness and completeness effect on discovery and held-out families.")


def plot_prompt_faithfulness(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = bench.new_figure(figsize=(9.0, 5.0))
    ordered = sorted(rows, key=lambda r: (r["faithfulness"] if r["faithfulness"] is not None else -999))
    xs = list(range(len(ordered)))
    ys = [r["faithfulness"] if r["faithfulness"] is not None else 0.0 for r in ordered]
    colors = ["tab:green" if r["family"] == "discovery" else "tab:purple" for r in ordered]
    bars = ax.bar(xs, ys, color=colors, alpha=0.85)
    ax.axhline(FAITHFULNESS_FLOOR, color="tab:red", linestyle="--", linewidth=1.0, label="floor")
    ax.axhline(1.0, color="black", linewidth=0.7, alpha=0.35)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["example_id"] for r in ordered], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("per-prompt faithfulness")
    ax.set_title("Failure cases: where the circuit least preserves the behavior")
    for bar, row in zip(bars, ordered):
        if row["faithfulness"] is not None:
            ax.annotate(f"{row['faithfulness']:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=7, rotation=90)
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "per_prompt_faithfulness.png", "Per-prompt faithfulness sorted from weakest to strongest.")


def plot_edge_interactions(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = bench.new_figure(figsize=(8.5, 5.0))
    top = sorted(rows, key=lambda r: r.get("interaction", -999), reverse=True)[:8]
    xs = list(range(len(top)))
    ys = [r["interaction"] for r in top]
    bars = ax.bar(xs, ys, color="tab:blue", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["edge"] for r in top], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("interaction = effect(prev) - effect(prev | induction ablated)")
    ax.set_title("Ordered previous-token -> induction interaction checks")
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    bench.save_figure(ctx, fig, "edge_interactions.png", "Ablation-interaction evidence for the one edge claim.")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    n_layers = bundle.anatomy.n_layers

    discovery, heldout, dropped, seq_len = build_dataset(ctx, bundle, args.max_examples)
    base_metric = statistics.fmean(ex.base_diff for ex in discovery)
    print(
        f"[lab6] discovery: {len(discovery)} prompts, dropped {dropped}; "
        f"held-out: {len(heldout)}; seq {seq_len}; base metric {base_metric:+.3f}"
    )

    # Instrument verification. Lab 6 stacks several hook systems, so the
    # microscope checks are not ceremony: they are load-bearing guards.
    probe = discovery[0].prompt.prompt
    bench.run_hook_parity_check(ctx, bundle, probe)
    comp_anatomy = bench.resolve_component_anatomy(ctx, bundle, probe, rel_tolerance=args.dla_tolerance)
    head_anatomy = bench.resolve_head_anatomy(ctx, bundle)
    first_att = bench.run_with_attention_cache(bundle, probe)
    bench.run_lens_self_check(ctx, bundle, first_att.capture)
    first_comp = bench.run_with_component_cache(bundle, probe, comp_anatomy, all_positions=False)
    bench.run_decomposition_check(ctx, bundle, first_comp, rel_tolerance=args.dla_tolerance)
    bench.run_head_decomposition_check(ctx, bundle, head_anatomy, first_att, rel_tolerance=args.dla_tolerance)
    n_heads = head_anatomy.n_heads

    # ----- captures and dataset means ---------------------------------------
    att_caps: dict[str, Any] = {}
    comp_caps: dict[str, Any] = {}
    for ex in discovery:
        att_caps[ex.prompt.example_id] = bench.run_with_attention_cache(bundle, ex.prompt.prompt, all_positions=True)
        comp_caps[ex.prompt.example_id] = bench.run_with_component_cache(
            bundle, ex.prompt.prompt, comp_anatomy, all_positions=True
        )

    head_means = torch.stack([att_caps[ex.prompt.example_id].o_in_last for ex in discovery]).mean(dim=0)
    mlp_means = torch.stack([comp_caps[ex.prompt.example_id].mlp_contrib for ex in discovery]).mean(dim=0)
    print(
        f"[lab6] dataset means from discovery prompts: heads {tuple(head_means.shape)}, "
        f"MLPs {tuple(mlp_means.shape)}"
    )
    manifest_path = ctx.path("diagnostics", "ablation_manifest.json")
    bench.write_json(
        manifest_path,
        {
            "off_distribution": "dataset mean over discovery prompts",
            "prompt_length_tokens": seq_len,
            "discovery_examples": [ex.prompt.example_id for ex in discovery],
            "head_means_shape": list(head_means.shape),
            "mlp_means_shape": list(mlp_means.shape),
            "scope": "attention heads are circuit nodes; MLPs are ranked as support only",
        },
    )
    ctx.register_artifact(manifest_path, "diagnostic", "Definition of the mean-ablation off distribution.")

    # ----- screening ----------------------------------------------------------
    head_attr: dict[tuple[int, int], list[float]] = {}
    head_induct: dict[tuple[int, int], list[float]] = {}
    head_prev: dict[tuple[int, int], list[float]] = {}
    head_first: dict[tuple[int, int], list[float]] = {}
    mlp_attr: dict[int, list[float]] = {}

    for ex in discovery:
        att = att_caps[ex.prompt.example_id]
        att_final = bench.AttentionCapture(
            capture=att.capture,
            attentions=att.attentions,
            o_in_last=att.o_in_last[:, -1],
            attn_out_last=att.attn_out_last[:, -1],
        )
        attr = head_attribution_scores(bundle, comp_anatomy, head_anatomy, att_final, ex.target_id, ex.distractor_id)

        comp = comp_caps[ex.prompt.example_id]
        comp_final = bench.ComponentCapture(
            capture=comp.capture,
            attn_contrib=comp.attn_contrib[:, -1],
            mlp_contrib=comp.mlp_contrib[:, -1],
        )
        dla = compute_direct_logit_attribution(bundle, comp_final, ex.target_id, ex.distractor_id)

        for layer in range(n_layers):
            mlp_attr.setdefault(layer, []).append(float(dla["mlp_scores"][layer]))
            for head in range(n_heads):
                key = (layer, head)
                pattern = att.attentions[layer, head]
                head_attr.setdefault(key, []).append(float(attr["scores"][layer][head]))
                head_prev.setdefault(key, []).append(prev_token_score(pattern))
                head_first.setdefault(key, []).append(first_token_score(pattern))
                ind = induction_score(pattern, att.capture.input_ids)
                head_induct.setdefault(key, []).append(0.0 if ind is None else float(ind))

    mean_attr = {k: statistics.fmean(v) for k, v in head_attr.items()}
    mean_induct = {k: statistics.fmean(v) for k, v in head_induct.items()}
    mean_prev = {k: statistics.fmean(v) for k, v in head_prev.items()}
    mean_first = {k: statistics.fmean(v) for k, v in head_first.items()}
    mean_mlp = {k: statistics.fmean(v) for k, v in mlp_attr.items()}

    attr_rank = rank_map(mean_attr, key_abs=True)
    induct_rank = rank_map(mean_induct)
    prev_rank = rank_map(mean_prev)
    top_attr, top_ind, top_prev = screen_budgets(n_layers, n_heads)

    screen_reasons: dict[tuple[int, int], set[str]] = {}
    for key in sorted(mean_attr, key=lambda k: -abs(mean_attr[k]))[:top_attr]:
        screen_reasons.setdefault(key, set()).add("attribution")
    for key in sorted(mean_induct, key=lambda k: -mean_induct[k])[:top_ind]:
        screen_reasons.setdefault(key, set()).add("induction")
    for key in sorted(mean_prev, key=lambda k: -mean_prev[k])[:top_prev]:
        screen_reasons.setdefault(key, set()).add("prev_token")

    def motif_label(key: tuple[int, int]) -> str:
        if mean_induct.get(key, 0.0) >= MOTIF_STRONG_THRESHOLD:
            return "induction"
        if mean_prev.get(key, 0.0) >= MOTIF_STRONG_THRESHOLD:
            return "previous_token"
        if mean_first.get(key, 0.0) >= FIRST_TOKEN_SINK_THRESHOLD:
            return "first_token_sink"
        return "other"

    head_labels = {key: motif_label(key) for key in screen_reasons}
    mlp_candidates = sorted(mean_mlp, key=lambda layer: -abs(mean_mlp[layer]))[:N_MLP_CANDIDATES]
    print(
        f"[lab6] screened {len(screen_reasons)} candidate heads "
        f"(attr {top_attr}, induction {top_ind}, previous-token {top_prev}) "
        f"+ {len(mlp_candidates)} candidate MLPs"
    )

    # ----- causal ranking ------------------------------------------------------
    cand_rows: list[dict[str, Any]] = []
    head_causal_drop: dict[tuple[int, int], float] = {}
    head_single_metric: dict[tuple[int, int], float] = {}

    for key in sorted(screen_reasons, key=lambda k: min(attr_rank[k], induct_rank[k], prev_rank[k])):
        layer, head = key
        ablated = metric_under_ablation(
            bundle,
            discovery,
            head_anatomy,
            comp_anatomy,
            heads=[key],
            mlps=[],
            head_means=head_means,
            mlp_means=mlp_means,
        )
        drop = base_metric - ablated
        head_causal_drop[key] = drop
        head_single_metric[key] = ablated
        reason = "+".join(sorted(screen_reasons[key]))
        cheap_rank = min(attr_rank[key], induct_rank[key], prev_rank[key])
        cand_rows.append(
            {
                "node": node_name("head", layer, head),
                "kind": "head",
                "layer": layer,
                "head": head,
                "screen_reason": reason,
                "cheap_rank": cheap_rank,
                "mean_attr": round(mean_attr[key], 5),
                "abs_attr_rank": attr_rank[key],
                "induction_score": round(mean_induct[key], 5),
                "induction_rank": induct_rank[key],
                "prev_token_score": round(mean_prev[key], 5),
                "prev_token_rank": prev_rank[key],
                "first_token_score": round(mean_first[key], 5),
                "motif_label": head_labels[key],
                "single_ablated_metric": round(ablated, 5),
                "causal_drop": round(drop, 5),
            }
        )

    for layer in mlp_candidates:
        ablated = metric_under_ablation(
            bundle,
            discovery,
            head_anatomy,
            comp_anatomy,
            heads=[],
            mlps=[layer],
            head_means=head_means,
            mlp_means=mlp_means,
        )
        cand_rows.append(
            {
                "node": node_name("mlp", layer),
                "kind": "mlp",
                "layer": layer,
                "head": "",
                "screen_reason": "attribution",
                "cheap_rank": "",
                "mean_attr": round(mean_mlp[layer], 5),
                "abs_attr_rank": "",
                "induction_score": "",
                "induction_rank": "",
                "prev_token_score": "",
                "prev_token_rank": "",
                "first_token_score": "",
                "motif_label": "support_mlp",
                "single_ablated_metric": round(ablated, 5),
                "causal_drop": round(base_metric - ablated, 5),
            }
        )

    # Add causal ranks for screened heads.
    head_rows = [r for r in cand_rows if r["kind"] == "head"]
    causal_rank_by_node = {
        r["node"]: i + 1 for i, r in enumerate(sorted(head_rows, key=lambda row: -row["causal_drop"]))
    }
    for row in cand_rows:
        row["causal_rank"] = causal_rank_by_node.get(row["node"], "")
        if row["kind"] == "head" and isinstance(row["cheap_rank"], int) and row["causal_rank"]:
            row["rank_gap_cheap_minus_causal"] = int(row["cheap_rank"]) - int(row["causal_rank"])
        else:
            row["rank_gap_cheap_minus_causal"] = ""

    cand_rows_sorted = sorted(cand_rows, key=lambda r: (r["kind"] != "head", -float(r["causal_drop"])))
    cand_path = ctx.path("tables", "candidate_components.csv")
    bench.write_csv_with_context(ctx, cand_path, cand_rows_sorted)
    ctx.register_artifact(cand_path, "table", "Screened candidates with cheap scores, motif labels, and causal drops.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, cand_rows_sorted)
    ctx.register_artifact(results_path, "results", "Alias of candidate_components.csv for the standard run contract.")

    # ----- greedy pruning -------------------------------------------------------
    all_heads = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]

    def faithfulness_of(circuit_heads: Sequence[tuple[int, int]], examples: Sequence[TaskExample], base: float) -> float:
        circuit_set = set(circuit_heads)
        complement = [head for head in all_heads if head not in circuit_set]
        metric = metric_under_ablation(
            bundle,
            examples,
            head_anatomy,
            comp_anatomy,
            heads=complement,
            mlps=[],
            head_means=head_means,
            mlp_means=mlp_means,
        )
        ratio = safe_ratio(metric, base)
        if ratio is None:
            raise RuntimeError("Cannot compute faithfulness ratio because the base metric is zero.")
        return ratio

    circuit = [key for key, drop in head_causal_drop.items() if drop > 0]
    if not circuit:
        raise RuntimeError(
            "No screened head has a positive causal drop; cannot assemble a circuit. "
            "See tables/candidate_components.csv."
        )
    circuit.sort(key=lambda key: -head_causal_drop[key])
    current_faith = faithfulness_of(circuit, discovery, base_metric)
    trajectory: list[dict[str, Any]] = [
        {
            "step": 0,
            "n_nodes": len(circuit),
            "faithfulness": round(current_faith, 5),
            "removed": "",
            "rule": "start with every positive-causal screened head",
        }
    ]
    print(f"[lab6] starting circuit: {len(circuit)} heads, faithfulness {current_faith:.3f}")

    stop_reason = "one head remains"
    while len(circuit) > 1:
        options: list[tuple[float, tuple[int, int]]] = []
        for key in circuit:
            reduced = [h for h in circuit if h != key]
            options.append((faithfulness_of(reduced, discovery, base_metric), key))
        best_faith, best_key = max(options, key=lambda item: item[0])

        if current_faith < FAITHFULNESS_FLOOR:
            if best_faith <= current_faith + 1e-9:
                stop_reason = "candidate set is below the faithfulness floor and no removal improves it"
                break
        elif best_faith < FAITHFULNESS_FLOOR:
            stop_reason = "next removal would cross the faithfulness floor"
            break

        circuit = [h for h in circuit if h != best_key]
        current_faith = best_faith
        trajectory.append(
            {
                "step": len(trajectory),
                "n_nodes": len(circuit),
                "faithfulness": round(best_faith, 5),
                "removed": node_name("head", *best_key),
                "rule": "removed least costly head",
            }
        )
        print(
            f"[lab6] pruned {node_name('head', *best_key)} -> {len(circuit)} heads, "
            f"faithfulness {best_faith:.3f}"
        )
    else:
        stop_reason = "one head remains"

    traj_path = ctx.path("tables", "prune_trajectory.csv")
    bench.write_csv_with_context(ctx, traj_path, trajectory)
    ctx.register_artifact(traj_path, "table", "Faithfulness at each greedy pruning step.")
    meets_floor = current_faith >= FAITHFULNESS_FLOOR

    # ----- F / C / M evaluation ---------------------------------------------------
    heldout_positive = [ex for ex in heldout if ex.base_diff > 0]
    heldout_base = statistics.fmean(ex.base_diff for ex in heldout_positive) if heldout_positive else None
    fcm: dict[str, Any] = {
        "faithfulness_floor": FAITHFULNESS_FLOOR,
        "meets_faithfulness_floor": meets_floor,
        "prune_stop_reason": stop_reason,
    }

    def evaluate_family(name: str, examples: Sequence[TaskExample], base: float | None) -> None:
        if not examples or base is None or base <= 0:
            return
        faith = faithfulness_of(circuit, examples, base)
        circuit_ablated = metric_under_ablation(
            bundle,
            examples,
            head_anatomy,
            comp_anatomy,
            heads=circuit,
            mlps=[],
            head_means=head_means,
            mlp_means=mlp_means,
        )
        completeness_ratio = safe_ratio(circuit_ablated, base)
        if completeness_ratio is None:
            return
        fcm[name] = {
            "n_prompts": len(examples),
            "base_metric": round(base, 5),
            "faithfulness": round(faith, 5),
            "circuit_ablated_metric": round(circuit_ablated, 5),
            "completeness_ratio": round(completeness_ratio, 5),
            "completeness_effect": round(1.0 - completeness_ratio, 5),
        }

    evaluate_family("discovery", discovery, base_metric)
    evaluate_family("heldout", heldout_positive, heldout_base)

    minimality_rows: list[dict[str, Any]] = []
    discovery_faith = fcm["discovery"]["faithfulness"]
    for key in circuit:
        reduced = [h for h in circuit if h != key]
        f_without = faithfulness_of(reduced, discovery, base_metric)
        marginal = discovery_faith - f_without
        minimality_rows.append(
            {
                "node": node_name("head", *key),
                "layer": key[0],
                "head": key[1],
                "motif_label": head_labels.get(key, "other"),
                "single_head_causal_drop": round(head_causal_drop.get(key, float("nan")), 5),
                "faithfulness_without": round(f_without, 5),
                "marginal_value": round(marginal, 5),
                "minimality_passes_positive_marginal": marginal > 0,
            }
        )
    minimality_rows.sort(key=lambda row: row["marginal_value"])
    fcm["minimality_worst_marginal"] = min((r["marginal_value"] for r in minimality_rows), default=None)
    fcm["minimality_all_positive"] = all(r["minimality_passes_positive_marginal"] for r in minimality_rows)

    fcm_path = ctx.path("faithfulness_completeness_minimality.json")
    bench.write_json(
        fcm_path,
        {
            "circuit": [node_name("head", *key) for key in circuit],
            "circuit_tuples": [list(key) for key in circuit],
            **fcm,
            "minimality": minimality_rows,
            "interpretation": {
                "faithfulness": "complement of circuit heads mean-ablated; higher is more sufficient",
                "completeness_ratio": "circuit heads mean-ablated; lower means the circuit was more necessary",
                "minimality": "loss in faithfulness when each kept head is removed",
            },
        },
    )
    ctx.register_artifact(fcm_path, "metrics", "Faithfulness, completeness, and minimality for the final circuit.")
    min_path = ctx.path("tables", "pruned_circuit.csv")
    bench.write_csv_with_context(ctx, min_path, minimality_rows)
    ctx.register_artifact(min_path, "table", "Every kept node with its marginal faithfulness value.")
    print(
        f"[lab6] final circuit {[node_name('head', *h) for h in circuit]}: "
        + ", ".join(
            f"{family} F={values['faithfulness']:.2f} C-ratio={values['completeness_ratio']:.2f}"
            for family, values in fcm.items()
            if isinstance(values, dict) and "faithfulness" in values
        )
    )

    # ----- the one edge claim: ordered ablation interaction -------------------
    induction_heads = [key for key in circuit if head_labels.get(key) == "induction"]
    prev_heads = [
        key
        for key in screen_reasons
        if head_labels.get(key) == "previous_token" and head_causal_drop.get(key, 0.0) > EDGE_MIN_SOURCE_EFFECT
    ]
    edge_rows: list[dict[str, Any]] = []
    for h_prev in prev_heads:
        for h_ind in induction_heads:
            if h_prev == h_ind or h_prev[0] >= h_ind[0]:
                continue
            m_i = head_single_metric.get(h_ind)
            if m_i is None:
                m_i = metric_under_ablation(
                    bundle,
                    discovery,
                    head_anatomy,
                    comp_anatomy,
                    heads=[h_ind],
                    mlps=[],
                    head_means=head_means,
                    mlp_means=mlp_means,
                )
            m_p = head_single_metric[h_prev]
            m_ip = metric_under_ablation(
                bundle,
                discovery,
                head_anatomy,
                comp_anatomy,
                heads=[h_ind, h_prev],
                mlps=[],
                head_means=head_means,
                mlp_means=mlp_means,
            )
            effect_p = base_metric - m_p
            effect_p_given_i = m_i - m_ip
            interaction = effect_p - effect_p_given_i
            raw_frac = safe_ratio(interaction, effect_p)
            strength = edge_strength_label(raw_frac)
            edge_rows.append(
                {
                    "edge": f"{node_name('head', *h_prev)} -> {node_name('head', *h_ind)}",
                    "from_layer": h_prev[0],
                    "from_head": h_prev[1],
                    "to_layer": h_ind[0],
                    "to_head": h_ind[1],
                    "effect_prev_alone": round(effect_p, 5),
                    "effect_prev_given_induction_ablated": round(effect_p_given_i, 5),
                    "interaction": round(interaction, 5),
                    "raw_interaction_fraction": round_or_none(raw_frac, 5),
                    "edge_strength": strength,
                    "claimable_fraction": raw_frac is not None and EDGE_MIN_ROUTED_FRACTION <= raw_frac <= 1.0,
                }
            )

    edge_csv_path = ctx.path("tables", "edge_interactions.csv")
    bench.write_csv_with_context(ctx, edge_csv_path, sorted(edge_rows, key=lambda r: r["interaction"], reverse=True))
    ctx.register_artifact(edge_csv_path, "table", "All ordered previous-token to induction ablation-interaction checks.")

    edge: dict[str, Any] | None = None
    if edge_rows:
        best_edge = max(edge_rows, key=lambda r: r["interaction"])
        h_prev = (best_edge["from_layer"], best_edge["from_head"])
        h_ind = (best_edge["to_layer"], best_edge["to_head"])
        raw_fraction = best_edge["raw_interaction_fraction"]
        claimed = bool(best_edge["claimable_fraction"] and best_edge["interaction"] > 0)
        if claimed:
            reason = f"ordered pair has positive interaction above the reporting threshold; strength={best_edge['edge_strength']}"
        elif raw_fraction is not None and raw_fraction > 1.0:
            reason = "interaction is positive but larger than the source effect, so it is not a literal routed fraction"
        elif best_edge["interaction"] <= 0:
            reason = "best ordered pair has no positive interaction"
        else:
            reason = "best ordered pair is below the routed-fraction threshold"
        edge = {
            "claimed": claimed,
            "from": h_prev,
            "to": h_ind,
            "edge": best_edge["edge"],
            "effect_prev_alone": best_edge["effect_prev_alone"],
            "effect_prev_given_induction_ablated": best_edge["effect_prev_given_induction_ablated"],
            "interaction": best_edge["interaction"],
            "raw_interaction_fraction": raw_fraction,
            "strength": best_edge["edge_strength"],
            "reason": reason,
        }
    else:
        edge = {
            "claimed": False,
            "edge": None,
            "reason": (
                "No ordered previous-token head before an induction head survived the causal and motif checks. "
                "The lab therefore makes no edge claim."
            ),
        }

    edge_path = ctx.path("tables", "edge_claim.json")
    bench.write_json(
        edge_path,
        {
            **edge,
            "thresholds": {
                "min_source_effect": EDGE_MIN_SOURCE_EFFECT,
                "min_routed_fraction": EDGE_MIN_ROUTED_FRACTION,
                "strong_routed_fraction": EDGE_STRONG_ROUTED_FRACTION,
                "requires_source_layer_lt_target_layer": True,
            },
            "explanation": (
                "Ablation interaction asks whether a previous-token head's effect shrinks "
                "when the induction head is already ablated. This licenses only an "
                "interaction-granularity edge. Path patching would be needed to localize "
                "the route to keys, values, or another subpath."
            ),
        },
    )
    ctx.register_artifact(edge_path, "metrics", "The one edge claim, or the reason no edge was claimed.")
    if edge.get("claimed"):
        print(
            f"[lab6] {edge['strength']} edge claimed: "
            f"{edge['edge']} ({edge['raw_interaction_fraction']:.0%} interaction fraction)"
        )
    else:
        print(f"[lab6] no edge claimed: {edge['reason']}")

    # ----- per-prompt failures -------------------------------------------------
    per_prompt_rows: list[dict[str, Any]] = []
    circuit_set = set(circuit)
    complement = [head for head in all_heads if head not in circuit_set]
    for ex in discovery + heldout:
        if ex.base_diff <= 0:
            per_prompt_rows.append(
                {
                    "example_id": ex.prompt.example_id,
                    "family": ex.prompt.family,
                    "prompt": ex.prompt.prompt,
                    "base_diff": round(ex.base_diff, 5),
                    "circuit_diff": "",
                    "faithfulness": None,
                    "note": "base model did not prefer the target; ratio undefined",
                }
            )
            continue
        logits = bench.run_with_node_set_ablation(
            bundle,
            ex.prompt.prompt,
            head_anatomy,
            comp_anatomy,
            heads=complement,
            mlps=[],
            head_means=head_means,
            mlp_means=mlp_means,
        )
        circuit_diff = float(logits[ex.target_id] - logits[ex.distractor_id])
        per_prompt_rows.append(
            {
                "example_id": ex.prompt.example_id,
                "family": ex.prompt.family,
                "prompt": ex.prompt.prompt,
                "base_diff": round(ex.base_diff, 5),
                "circuit_diff": round(circuit_diff, 5),
                "faithfulness": round(circuit_diff / ex.base_diff, 5),
                "note": "",
            }
        )
    per_prompt_path = ctx.path("tables", "per_prompt_faithfulness.csv")
    bench.write_csv_with_context(ctx, per_prompt_path, per_prompt_rows)
    ctx.register_artifact(per_prompt_path, "table", "Per-prompt faithfulness and the two weakest failure cases.")
    failures = sorted(
        (row for row in per_prompt_rows if row["faithfulness"] is not None),
        key=lambda row: row["faithfulness"],
    )[:2]

    # ----- plots ---------------------------------------------------------------
    if not args.no_plots:
        plot_screen_vs_causal(ctx, cand_rows_sorted)
        plot_prune_trajectory(ctx, trajectory, floor=FAITHFULNESS_FLOOR)
        mlp_support = [r for r in cand_rows_sorted if r["kind"] == "mlp" and float(r["causal_drop"]) > 0]
        plot_circuit_graph(ctx, circuit, head_labels, mlp_support, edge, n_layers, n_heads)
        if "discovery" in fcm:
            plot_fcm(ctx, fcm, floor=FAITHFULNESS_FLOOR)
        plot_prompt_faithfulness(ctx, per_prompt_rows)
        plot_edge_interactions(ctx, edge_rows)

    # ----- circuit card --------------------------------------------------------
    supporting_mlps = sorted(
        (r for r in cand_rows_sorted if r["kind"] == "mlp" and float(r["causal_drop"]) > 0),
        key=lambda row: -float(row["causal_drop"]),
    )
    card: list[str] = [
        "# Circuit card: induction completion",
        "",
        f"- **Model:** `{bundle.anatomy.model_id}` | run `{ctx.run_dir.name}`",
        "- **Task:** predict the induction continuation of fixed-length repeating patterns.",
        "- **Metric:** mean logit(target) minus logit(distractor).",
        f"- **Base metric:** {base_metric:+.3f} on {len(discovery)} discovery prompts.",
        f"- **Dataset:** {len(discovery)} discovery prompts, {len(heldout)} held-out prompts "
        f"({len(heldout_positive)} baseline-positive for F/C/M), {dropped} discovery prompts dropped.",
        f"- **Circuit scope:** heads-only routing graph. MLPs stay intact for faithfulness and completeness.",
        f"- **Ablation off distribution:** dataset mean over discovery prompts, sequence length {seq_len}.",
        f"- **Validated heads:** {describe_head_list(circuit)}.",
        f"- **Faithfulness verdict:** {'passes' if meets_floor else 'does not pass'} the floor of {FAITHFULNESS_FLOOR:.2f}.",
        "",
        "## Candidate screen",
        "",
        f"Screened {len(screen_reasons)} heads using attribution top {top_attr}, induction top {top_ind}, "
        f"and previous-token top {top_prev}. Screening proposes suspects; mean-ablation decides which suspects matter.",
        "",
        "## Validated nodes",
        "",
        "| node | motif | single-head causal drop | marginal value |",
        "|---|---|---:|---:|",
    ]
    marginal_by_node = {row["node"]: row for row in minimality_rows}
    for key in circuit:
        node = node_name("head", *key)
        marg = marginal_by_node.get(node, {})
        card.append(
            f"| {node} | {head_labels.get(key, 'other')} | "
            f"{head_causal_drop.get(key, float('nan')):+.3f} | {marg.get('marginal_value', '')} |"
        )
    card += [
        "",
        "**Supporting MLPs, not circuit nodes:** "
        + (
            ", ".join(f"MLP{r['layer']} (drop {float(r['causal_drop']):+.2f})" for r in supporting_mlps)
            if supporting_mlps
            else "none with positive causal drop among screened MLPs"
        ),
        "",
        "## Faithfulness, completeness, minimality",
        "",
        "| family | n | base metric | faithfulness | completeness ratio | completeness effect |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in ("discovery", "heldout"):
        if family in fcm:
            v = fcm[family]
            card.append(
                f"| {family} | {v['n_prompts']} | {v['base_metric']} | {v['faithfulness']} | "
                f"{v['completeness_ratio']} | {v['completeness_effect']} |"
            )
    card += [
        "",
        f"Minimality worst marginal value: `{fcm['minimality_worst_marginal']}`. "
        "A negative value means at least one kept head hurt faithfulness under this pruning rule.",
    ]
    if any(isinstance(v, dict) and v.get("faithfulness", 0) > 1.0 for v in fcm.values()):
        card += [
            "",
            "Faithfulness above 1.0 is reported rather than clipped. It means the mean-ablated complement "
            "was suppressing the target metric on that prompt family.",
        ]
    card += [
        "",
        "## Edge claim",
        "",
    ]
    if edge and edge.get("claimed"):
        card.append(
            f"Claimed {edge['strength']} edge `{edge['edge']}`: raw interaction fraction "
            f"{edge['raw_interaction_fraction']:.0%} "
            f"(reporting threshold {EDGE_MIN_ROUTED_FRACTION:.0%}; strong threshold {EDGE_STRONG_ROUTED_FRACTION:.0%}). "
            "This is an ablation-interaction edge, not path patching."
        )
    else:
        card.append(f"No edge claimed. Reason: {edge['reason'] if edge else 'no edge diagnostic produced'}")
    card += [
        "",
        "## Failure cases the circuit least explains",
        "",
    ]
    if failures:
        for row in failures:
            card.append(
                f"- `{row['example_id']}` ({row['family']}): faithfulness {row['faithfulness']} "
                f"with base diff {row['base_diff']}."
            )
    else:
        card.append("- No ratio-defined failure cases were available.")
    card += [
        "",
        "## Scope and filler terms, MDC honesty section",
        "",
        "- Population: fixed-length 8-token repeating patterns from the listed vocabularies.",
        "- Circuit nodes: attention heads only. MLP layers are supporting infrastructure, not part of this routing graph.",
        "- Intervention: dataset-mean ablation at all positions. A different off distribution defines a different circuit.",
        "- Edge evidence: ablation interaction. Claims about keys, values, or exact subpaths are filler terms unless path patching is added.",
        "- Natural-text induction, longer cycles, and alternate tokenizations are outside this card's scope.",
        "",
    ]
    card_path = ctx.path("circuit_card.md")
    bench.write_text(card_path, "\n".join(card))
    ctx.register_artifact(card_path, "summary", "The Lab 6 circuit card deliverable.")

    # ----- metrics, claims, summary ------------------------------------------
    metrics = {
        "base_metric": base_metric,
        "n_discovery": len(discovery),
        "n_heldout": len(heldout),
        "n_heldout_positive": len(heldout_positive),
        "circuit": [node_name("head", *key) for key in circuit],
        "circuit_tuples": [list(key) for key in circuit],
        "screen_budgets": {"attribution": top_attr, "induction": top_ind, "prev_token": top_prev},
        "fcm": {key: value for key, value in fcm.items() if isinstance(value, dict)},
        "meets_faithfulness_floor": meets_floor,
        "minimality_worst_marginal": fcm["minimality_worst_marginal"],
        "minimality_all_positive": fcm["minimality_all_positive"],
        "prune_stop_reason": stop_reason,
        "edge": edge,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 6 metrics.")

    run_name = ctx.run_dir.name
    claims: list[dict[str, str]] = []
    if meets_floor:
        claim_text = (
            f"A {len(circuit)}-head routing circuit ({describe_head_list(circuit)}) in {bundle.anatomy.model_id} "
            f"is faithful at {fcm['discovery']['faithfulness']:.2f} of base behavior on {len(discovery)} "
            "8-token induction prompts when every non-circuit head is dataset-mean ablated. "
            f"Ablating the circuit leaves {fcm['discovery']['completeness_ratio']:.2f} of base. "
            "MLPs are left intact, so this is a heads-only routing claim."
        )
    else:
        claim_text = (
            f"The screened Lab 6 head set in {bundle.anatomy.model_id} did not meet the faithfulness floor: "
            f"the final {len(circuit)}-head circuit ({describe_head_list(circuit)}) preserved "
            f"{fcm['discovery']['faithfulness']:.2f} of base behavior, below the {FAITHFULNESS_FLOOR:.2f} floor. "
            "This is causal evidence about the screened components, but not a successful faithful-circuit claim."
        )
    claims.append(
        {
            "id": f"{LAB_ID}-C1",
            "tag": "CAUSAL",
            "text": claim_text,
            "artifact": f"runs/{run_name}/faithfulness_completeness_minimality.json",
            "falsifier": (
                "Zero ablation, resample ablation, longer prompts, or natural-text induction collapses the result. "
                "That would show the circuit was specific to this off distribution or prompt family."
            ),
        }
    )

    if "heldout" in fcm:
        heldout_note = (
            " Held-out faithfulness above 1.0 means the mean-ablated complement was suppressing the metric, "
            "not that the circuit is better than the full model."
            if fcm["heldout"]["faithfulness"] > 1.0
            else ""
        )
        claims.append(
            {
                "id": f"{LAB_ID}-C2",
                "tag": "CAUSAL",
                "text": (
                    f"The heads-only routing circuit transfers from discovery to {len(heldout_positive)} held-out "
                    f"vocabulary prompts with faithfulness {fcm['heldout']['faithfulness']:.2f} versus "
                    f"{fcm['discovery']['faithfulness']:.2f} on discovery. The claim is induction-pattern transfer, "
                    "not natural-language generality."
                    + heldout_note
                ),
                "artifact": f"runs/{run_name}/plots/circuit_scorecard.png",
                "falsifier": "Held-out families, longer cycles, or a paraphrased natural-text induction set lose the faithfulness effect.",
            }
        )

    if edge and edge.get("claimed"):
        claims.append(
            {
                "id": f"{LAB_ID}-C3",
                "tag": "CAUSAL",
                "text": (
                    f"Ordered edge claim at interaction granularity: {edge['edge']} has raw interaction fraction "
                    f"{edge['raw_interaction_fraction']:.0%} ({edge['strength']}). The previous-token head's effect is "
                    f"{edge['effect_prev_alone']:+.2f} alone versus {edge['effect_prev_given_induction_ablated']:+.2f} "
                    "when the induction head is already ablated."
                ),
                "artifact": f"runs/{run_name}/tables/edge_claim.json",
                "falsifier": "Path patching localizes the effect to a different route, or a redundant induction head absorbs the interaction.",
            }
        )

    bench.write_ledger_suggestions(ctx, LAB_ID, claims)

    lines: list[str] = [
        "# Lab 6 run summary: circuit discovery, the manual way",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({n_layers} blocks x {n_heads} heads)",
        f"- task: induction completion, {len(discovery)} discovery + {len(heldout)} held-out prompts "
        f"({len(heldout_positive)} baseline-positive for F/C/M)",
        "- evidence level: `CAUSAL` at heads-only circuit scope",
        "- intervention: dataset-mean ablation at all positions",
        "- self-checks: hook parity, lens, component decomposition, head decomposition",
        "",
        "## 1-4. Behavior, measurement, intervention, headline",
        "",
        f"- base metric {base_metric:+.3f}; circuit of {len(circuit)} heads ({describe_head_list(circuit)})",
        f"- pruning stop: {stop_reason}",
        f"- faithfulness floor: {FAITHFULNESS_FLOOR:.2f}; verdict: {'pass' if meets_floor else 'not passed'}",
    ]
    for family in ("discovery", "heldout"):
        if family in fcm:
            values = fcm[family]
            lines.append(
                f"- {family}: faithfulness {values['faithfulness']}, "
                f"completeness ratio {values['completeness_ratio']}, "
                f"completeness effect {values['completeness_effect']}"
            )
    lines += [
        f"- minimality: worst marginal value {fcm['minimality_worst_marginal']}",
        f"- edge: {edge['edge'] + ' (' + edge['strength'] + ')' if edge and edge.get('claimed') else 'none claimed'}",
        "",
        "## 5. Claims",
        "",
    ]
    for claim in claims:
        lines.append(f"- `{claim['id']}` {claim['tag']}: {claim['text']}")
        lines.append(f"  - falsifier: {claim['falsifier']}")
    lines += [
        "",
        "## 6. The reading order",
        "",
        "1. `circuit_card.md` - the deliverable; everything else is evidence for it.",
        "2. `plots/circuit_graph.png` - validated heads, support MLPs, and any claimed edge.",
        "3. `plots/prune_trajectory.png` - what each node costs during greedy pruning.",
        "4. `plots/screen_vs_causal.png` - the central lesson: cheap screening (attribution or motif) is a hypothesis generator, not a circuit claim. The off-diagonal points are the payload.",
        "5. `plots/circuit_scorecard.png` - discovery vs held-out faithfulness and completeness (held-out is often *higher* faithfulness; that is the generalization test).",
        "6. `tables/per_prompt_faithfulness.csv` and `plots/per_prompt_faithfulness.png` - the specific prompts the circuit least explains (the anti-cherry-pick evidence; the lowest bars go into the card).",
        "7. `plots/edge_interactions.png` and `tables/edge_interactions.csv` - the ordered interaction checks (weak vs strong, layer-order respected, not path patching).",
        "",
        "## 7. Caveats students must carry forward",
        "",
        "- The circuit is a heads-only routing graph; MLPs are support, not nodes in the claim. This is the manual baseline Lab 9 will confront with an automated feature graph.",
        "- Mean-ablation defines the off state (dataset mean, fixed length). Changing the off state (zero, different mean, longer prompts) defines a different circuit. The ablation_manifest.json records the exact choice.",
        "- The edge test is an ablation interaction (previous-token effect shrinks when the induction head is already ablated), not path patching. Claims about keys/values or exact subpaths are filler terms.",
        "- Keep this card. Lab 9 will compare this manual graph with an attribution graph so you can see what each method buys and what each quietly assumes.",
        "",
    ]
    summary_path = ctx.path("run_summary.md")
    bench.write_text(summary_path, "\n".join(lines))
    ctx.register_artifact(summary_path, "summary", "The seven standard lab-summary questions answered.")
    print(f"[lab6] wrote circuit_card.md, run_summary.md, and {len(claims)} drafted ledger claims")
