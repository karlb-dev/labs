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
# Plotting and evidence-table helpers
# ---------------------------------------------------------------------------

MOTIF_COLOR_FALLBACKS = {
    "induction": "#E69F00",
    "previous_token": "#0072B2",
    "first_token_sink": "#7E57C2",
    "diffuse": "#999999",
    "other": "#555555",
    "support_mlp": "#CC79A7",
}

MOTIF_MARKER_FALLBACKS = {
    "induction": "*",
    "previous_token": "o",
    "first_token_sink": "s",
    "diffuse": ".",
    "other": "x",
    "support_mlp": "D",
}

FAMILY_COLOR_FALLBACKS = {
    "discovery": "#009E73",
    "heldout": "#0072B2",
    "undefined": "#888888",
}


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _motif_color(label: str) -> str:
    func = getattr(bench, "plot_motif_color", None)
    if callable(func):
        return func(label, MOTIF_COLOR_FALLBACKS.get(label, "#555555"))
    return MOTIF_COLOR_FALLBACKS.get(str(label), "#555555")


def _motif_marker(label: str) -> str:
    func = getattr(bench, "plot_motif_marker", None)
    marker = func(label, MOTIF_MARKER_FALLBACKS.get(label, "o")) if callable(func) else MOTIF_MARKER_FALLBACKS.get(str(label), "o")
    # Filled markers keep dense scatterplots readable and avoid Matplotlib edgecolor warnings.
    return "o" if marker in {"x", ".", ","} else marker


def _family_color(family: str) -> str:
    func = getattr(bench, "plot_category_color", None)
    if callable(func):
        return func(family, FAMILY_COLOR_FALLBACKS.get(family, "#555555"))
    return FAMILY_COLOR_FALLBACKS.get(str(family), "#555555")


def _component_color(component: str) -> str:
    func = getattr(bench, "plot_component_color", None)
    if callable(func):
        return func(component, "#555555")
    return {"head": "#0072B2", "mlp": "#E69F00", "attn": "#0072B2"}.get(str(component), "#555555")


def _lighten(color: str, amount: float = 0.55) -> str:
    func = getattr(bench, "lighten_color", None)
    if callable(func):
        return func(color, amount)
    try:
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        return mcolors.to_hex(tuple(c + (1.0 - c) * amount for c in rgb))
    except Exception:
        return color


def _panel_label(ax: Any, label: str) -> None:
    func = getattr(bench, "add_panel_label", None)
    if callable(func):
        func(ax, label)
    else:
        ax.text(
            -0.08,
            1.04,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def _zero_line(ax: Any, axis: str = "y", **kwargs: Any) -> None:
    func = getattr(bench, "add_zero_line", None)
    if callable(func):
        func(ax, axis=axis, **kwargs)
    else:
        if axis == "y":
            ax.axhline(0, color=kwargs.get("color", "black"), linewidth=kwargs.get("linewidth", 0.8), alpha=kwargs.get("alpha", 0.7))
        else:
            ax.axvline(0, color=kwargs.get("color", "black"), linewidth=kwargs.get("linewidth", 0.8), alpha=kwargs.get("alpha", 0.7))


def _style(ax: Any, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    func = getattr(bench, "style_ax", None)
    if callable(func):
        func(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    else:
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)


def plot_reading_guide_rows() -> list[dict[str, str]]:
    """A small map from artifact to the concept it teaches."""
    return [
        {
            "artifact": "plots/circuit_discovery_dashboard.png",
            "question": "What is the overall manual circuit claim?",
            "read_for": "F/C/M, pruning, candidate quality, and prompt-level failures in one place.",
        },
        {
            "artifact": "plots/candidate_evidence_matrix.png",
            "question": "Which heads survived the evidence ladder?",
            "read_for": "cheap screen signals, motif labels, causal drop, and final membership side by side.",
        },
        {
            "artifact": "plots/causal_motif_atlas.png",
            "question": "Where are causally useful heads in layer/head space?",
            "read_for": "screened heads, signed causal effects, motif labels, and final circuit membership.",
        },
        {
            "artifact": "plots/screen_vs_causal.png",
            "question": "Where did cheap screening lie?",
            "read_for": "heads with high attribution or motif scores but near-zero or negative causal drop.",
        },
        {
            "artifact": "plots/minimality_ledger.png",
            "question": "Did every final head earn its rent?",
            "read_for": "marginal faithfulness loss when each kept head is removed.",
        },
        {
            "artifact": "plots/prompt_failure_scatter.png",
            "question": "Where does the circuit fail or over-recover?",
            "read_for": "base behavior strength versus circuit-only faithfulness for each prompt.",
        },
        {
            "artifact": "plots/edge_interaction_map.png",
            "question": "Was an ordered edge earned?",
            "read_for": "previous-token to induction interaction size, fraction, and layer ordering.",
        },
    ]


def build_circuit_evidence_matrix(
    cand_rows: Sequence[dict[str, Any]],
    minimality_rows: Sequence[dict[str, Any]],
    circuit: Sequence[tuple[int, int]],
    edge: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Join cheap screen, motif, ablation, pruning, and edge status into one table."""
    circuit_set = {node_name("head", *key) for key in circuit}
    marginal = {row["node"]: row for row in minimality_rows}
    edge_roles: dict[str, str] = {}
    if edge and edge.get("claimed"):
        edge_roles[node_name("head", *edge["from"])] = "edge_source"
        edge_roles[node_name("head", *edge["to"])] = "edge_target"

    out: list[dict[str, Any]] = []
    for row in cand_rows:
        node = str(row.get("node", ""))
        if row.get("kind") != "head":
            continue
        cheap_rank = _to_int(row.get("cheap_rank"), 10**9)
        causal_rank = _to_int(row.get("causal_rank"), 10**9)
        causal_drop = _to_float(row.get("causal_drop"), 0.0)
        tags: list[str] = []
        if node in circuit_set:
            tags.append("final_circuit")
        if causal_drop > 0:
            tags.append("positive_causal_drop")
        elif causal_drop < 0:
            tags.append("negative_causal_drop")
        motif = str(row.get("motif_label", "other"))
        if motif not in ("", "other"):
            tags.append(f"motif:{motif}")
        if abs(cheap_rank - causal_rank) >= 8 and causal_rank < 10**9:
            tags.append("screen_causal_disagreement")
        if edge_roles.get(node):
            tags.append(edge_roles[node])
        mrow = marginal.get(node, {})
        out.append(
            {
                "node": node,
                "layer": row.get("layer", ""),
                "head": row.get("head", ""),
                "motif_label": motif,
                "screen_reason": row.get("screen_reason", ""),
                "cheap_rank": row.get("cheap_rank", ""),
                "causal_rank": row.get("causal_rank", ""),
                "rank_gap_cheap_minus_causal": row.get("rank_gap_cheap_minus_causal", ""),
                "mean_attr": row.get("mean_attr", ""),
                "abs_attr": round(abs(_to_float(row.get("mean_attr"), 0.0)), 6),
                "induction_score": row.get("induction_score", ""),
                "prev_token_score": row.get("prev_token_score", ""),
                "first_token_score": row.get("first_token_score", ""),
                "single_ablated_metric": row.get("single_ablated_metric", ""),
                "causal_drop": row.get("causal_drop", ""),
                "in_final_circuit": node in circuit_set,
                "marginal_value": mrow.get("marginal_value", ""),
                "minimality_passes_positive_marginal": mrow.get("minimality_passes_positive_marginal", ""),
                "edge_role": edge_roles.get(node, ""),
                "evidence_tags": ";".join(tags),
            }
        )
    return sorted(out, key=lambda r: (not bool(r["in_final_circuit"]), -_to_float(r.get("causal_drop"), 0.0), _to_int(r.get("cheap_rank"), 10**9)))


def build_prompt_failure_modes(rows: Sequence[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        faith = row.get("faithfulness")
        faith_float = _to_float(faith, float("nan"))
        base = _to_float(row.get("base_diff"), float("nan"))
        circuit_diff = _to_float(row.get("circuit_diff"), float("nan"))
        if not _is_finite(faith_float):
            mode = "undefined_base_not_target_preferring"
            gap_to_floor = ""
            over_recovery = ""
        elif faith_float < floor:
            mode = "under_floor"
            gap_to_floor = round(faith_float - floor, 5)
            over_recovery = round(max(0.0, faith_float - 1.0), 5)
        elif faith_float > 1.05:
            mode = "over_recovery"
            gap_to_floor = round(faith_float - floor, 5)
            over_recovery = round(faith_float - 1.0, 5)
        else:
            mode = "preserved"
            gap_to_floor = round(faith_float - floor, 5)
            over_recovery = round(max(0.0, faith_float - 1.0), 5)
        out.append(
            {
                **row,
                "faithfulness": round(faith_float, 5) if _is_finite(faith_float) else None,
                "base_diff": round(base, 5) if _is_finite(base) else "",
                "circuit_diff": round(circuit_diff, 5) if _is_finite(circuit_diff) else "",
                "failure_mode": mode,
                "gap_to_floor": gap_to_floor,
                "over_recovery_above_full_model": over_recovery,
            }
        )
    return out


def plot_screen_vs_causal(ctx: bench.RunContext, cand_rows: Sequence[dict[str, Any]]) -> None:
    """Show the central lesson: cheap screens and causal effects disagree."""
    import matplotlib.pyplot as plt

    heads = [r for r in cand_rows if r.get("kind") == "head"]
    if not heads:
        return

    fig, (ax_rank, ax_attr) = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    for ax in (ax_rank, ax_attr):
        _style(ax)
        _zero_line(ax)

    for r in heads:
        motif = str(r.get("motif_label", "other"))
        color = _motif_color(motif)
        marker = _motif_marker(motif)
        drop = _to_float(r.get("causal_drop"), 0.0)
        cheap = _to_float(r.get("cheap_rank"), 0.0)
        attr = abs(_to_float(r.get("mean_attr"), 0.0))
        alpha = 0.92 if drop > 0 else 0.45
        size = 78 if drop > 0 else 44
        ax_rank.scatter(cheap, drop, s=size, color=color, marker=marker, alpha=alpha, edgecolors="black", linewidths=0.35)
        ax_attr.scatter(attr, drop, s=size, color=color, marker=marker, alpha=alpha, edgecolors="black", linewidths=0.35)

    label_nodes: set[str] = set()
    label_nodes.update(str(r["node"]) for r in sorted(heads, key=lambda x: -_to_float(x.get("causal_drop"), 0.0))[:8])
    label_nodes.update(str(r["node"]) for r in sorted(heads, key=lambda x: _to_float(x.get("cheap_rank"), 10**9))[:6])
    label_nodes.update(str(r["node"]) for r in sorted(heads, key=lambda x: abs(_to_float(x.get("rank_gap_cheap_minus_causal"), 0.0)), reverse=True)[:4])
    for r in heads:
        if str(r.get("node")) in label_nodes:
            ax_rank.annotate(str(r["node"]), (_to_float(r.get("cheap_rank"), 0.0), _to_float(r.get("causal_drop"), 0.0)), textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax_attr.annotate(str(r["node"]), (abs(_to_float(r.get("mean_attr"), 0.0)), _to_float(r.get("causal_drop"), 0.0)), textcoords="offset points", xytext=(4, 4), fontsize=7)

    _style(ax_rank, title="Cheap screen rank vs causal effect", xlabel="cheap screen rank, lower is better", ylabel="single-head causal drop")
    _style(ax_attr, title="Frozen attribution magnitude vs causal effect", xlabel="|direct-logit attribution|", ylabel="single-head causal drop")
    for motif in ("previous_token", "induction", "first_token_sink", "other"):
        if any(str(r.get("motif_label", "other")) == motif for r in heads):
            ax_attr.scatter([], [], color=_motif_color(motif), marker=_motif_marker(motif), label=motif)
    ax_attr.legend(fontsize=8, loc="best")
    fig.suptitle("Cheap screening is a hypothesis generator, not a circuit claim")
    bench.save_figure(ctx, fig, "screen_vs_causal.png", "Cheap screen rank and attribution magnitude against single-head mean-ablation effect.")


def plot_prune_trajectory(ctx: bench.RunContext, trajectory: Sequence[dict[str, Any]], *, floor: float) -> None:
    if not trajectory:
        return
    fig, ax = bench.new_figure(figsize=(8.8, 5.3))
    xs = [t["n_nodes"] for t in trajectory]
    ys = [_to_float(t["faithfulness"], 0.0) for t in trajectory]
    ax.plot(xs, ys, marker="o", linewidth=2.3, color=_family_color("discovery"))
    ax.fill_between(xs, [floor for _ in xs], ys, where=[y >= floor for y in ys], alpha=0.10, color=_family_color("discovery"), interpolate=True)
    ax.fill_between(xs, ys, [floor for _ in xs], where=[y < floor for y in ys], alpha=0.10, color="#D55E00", interpolate=True)
    for t in trajectory:
        if t.get("removed"):
            ax.annotate(f"-{t['removed']}", (t["n_nodes"], _to_float(t["faithfulness"], 0.0)), textcoords="offset points", xytext=(2, 8), fontsize=7, rotation=30)
    ax.axhline(floor, color="#D55E00", linewidth=1.1, linestyle="--", label=f"floor = {floor:.2f}")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.45, label="full-model baseline")
    ax.invert_xaxis()
    _style(ax, title="Greedy pruning: what the behavior costs, node by node", xlabel="circuit size, heads kept", ylabel="faithfulness, complement mean-ablated")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "prune_trajectory.png", "Faithfulness at each greedy pruning step, with floor and full-model reference.")


def plot_circuit_graph(
    ctx: bench.RunContext,
    circuit: Sequence[tuple[int, int]],
    head_labels: dict[tuple[int, int], str],
    mlp_support: Sequence[dict[str, Any]],
    edge: dict[str, Any] | None,
    n_layers: int,
    n_heads: int,
) -> None:
    fig, ax = bench.new_figure(figsize=(10.8, 6.2))
    for layer, head in circuit:
        label = head_labels.get((layer, head), "other")
        ax.scatter(layer, head, s=210, color=_motif_color(label), marker=_motif_marker(label), zorder=3, edgecolors="black", linewidths=0.8)
        ax.annotate(f"L{layer}H{head}\n{label}", (layer, head), textcoords="offset points", xytext=(6, 6), fontsize=7)

    shown_mlps = list(mlp_support[:5])
    for i, r in enumerate(shown_mlps):
        y = n_heads + 1 + 0.70 * (i % 3)
        ax.scatter(_to_float(r.get("layer"), 0.0), y, s=145, marker="s", color=_component_color("mlp"), zorder=3, edgecolors="black", linewidths=0.8)
        ax.annotate(f"MLP{r['layer']}\ndrop {_to_float(r.get('causal_drop'), 0.0):+.2f}", (_to_float(r.get("layer"), 0.0), y), textcoords="offset points", xytext=(4, 6 + 4 * (i % 2)), fontsize=7)
    if len(mlp_support) > len(shown_mlps):
        ax.text(0.99, 0.98, f"+{len(mlp_support) - len(shown_mlps)} support MLPs in card", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=_component_color("mlp"))

    if edge is not None and edge.get("claimed"):
        l1, h1 = edge["from"]
        l2, h2 = edge["to"]
        if (l1, h1) not in circuit:
            ax.scatter(l1, h1, s=200, facecolors="none", edgecolors=_motif_color("previous_token"), linewidths=1.8, zorder=3)
            ax.annotate(f"L{l1}H{h1}\nscreened, pruned", (l1, h1), textcoords="offset points", xytext=(6, -18), fontsize=7, color=_motif_color("previous_token"))
        ax.annotate("", xy=(l2, h2), xytext=(l1, h1), arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 2.0, "shrinkA": 13, "shrinkB": 13})
        mid_x, mid_y = (l1 + l2) / 2, (h1 + h2) / 2
        ax.annotate(f"{edge['strength']} interaction\n{edge['raw_interaction_fraction']:.0%}", (mid_x, mid_y), textcoords="offset points", xytext=(0, 12), fontsize=8, ha="center")

    for motif in ("previous_token", "induction", "first_token_sink", "other"):
        if any(head_labels.get(key, "other") == motif for key in circuit):
            ax.scatter([], [], color=_motif_color(motif), marker=_motif_marker(motif), label=motif, edgecolors="black", linewidths=0.6)
    if shown_mlps:
        ax.scatter([], [], color=_component_color("mlp"), marker="s", label="support MLP", edgecolors="black", linewidths=0.6)
    ax.set_xlim(-1, n_layers)
    ax.set_ylim(-1.5, n_heads + 4.8)
    _style(ax, title="Validated heads-only routing circuit", xlabel="layer", ylabel="head index; squares above are supporting MLPs")
    ax.legend(fontsize=8, loc="upper left", ncols=2)
    bench.save_figure(ctx, fig, "circuit_graph.png", "Circuit heads, motif labels, support MLPs, and the claimed edge if any.")


def plot_fcm(ctx: bench.RunContext, fcm: dict[str, Any], *, floor: float) -> None:
    fig, ax = bench.new_figure(figsize=(8.2, 5.2))
    groups: list[tuple[str, str, float]] = []
    for family in ("discovery", "heldout"):
        if family in fcm:
            groups.append((family, "faithfulness", _to_float(fcm[family].get("faithfulness"), 0.0)))
            groups.append((family, "completeness_effect", _to_float(fcm[family].get("completeness_effect"), 0.0)))
    xs = list(range(len(groups)))
    colors = [_family_color(fam) if metric == "faithfulness" else _lighten(_family_color(fam), 0.35) for fam, metric, _ in groups]
    bars = ax.bar(xs, [v for _, _, v in groups], color=colors, alpha=0.92)
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.axhline(floor, color="#D55E00", linewidth=1.1, linestyle="--", label="faithfulness floor")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{metric.replace('_', ' ')}\n{family}" for family, metric, _ in groups], fontsize=8)
    _style(ax, title="Circuit scorecard: preservation and destruction of behavior", ylabel="fraction of base behavior")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "circuit_scorecard.png", "Faithfulness and completeness effect on discovery and held-out families.")


def plot_prompt_faithfulness(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = bench.new_figure(figsize=(9.6, 5.2))
    ordered = sorted(rows, key=lambda r: (_to_float(r.get("faithfulness"), -999.0)))
    xs = list(range(len(ordered)))
    ys = [_to_float(r.get("faithfulness"), 0.0) for r in ordered]
    colors = [_family_color(str(r.get("family", "undefined"))) for r in ordered]
    bars = ax.bar(xs, ys, color=colors, alpha=0.88)
    ax.axhline(FAITHFULNESS_FLOOR, color="#D55E00", linestyle="--", linewidth=1.1, label="floor")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.35, label="full model")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r.get("example_id", "")) for r in ordered], rotation=35, ha="right", fontsize=8)
    for bar, row in zip(bars, ordered):
        faith = _to_float(row.get("faithfulness"), float("nan"))
        if math.isfinite(faith):
            ax.annotate(f"{faith:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=7, rotation=90)
    _style(ax, title="Failure cases and over-recovery: per-prompt circuit faithfulness", ylabel="per-prompt faithfulness")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "per_prompt_faithfulness.png", "Per-prompt faithfulness sorted from weakest to strongest.")


def plot_edge_interactions(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = bench.new_figure(figsize=(8.8, 5.0))
    top = sorted(rows, key=lambda r: _to_float(r.get("interaction"), -999.0), reverse=True)[:10]
    xs = list(range(len(top)))
    ys = [_to_float(r.get("interaction"), 0.0) for r in top]
    colors = [
        _motif_color("induction") if str(r.get("edge_strength")) in {"strong", "weak"} else _lighten(_motif_color("induction"), 0.55)
        for r in top
    ]
    bars = ax.bar(xs, ys, color=colors, alpha=0.9)
    _zero_line(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r.get("edge", "")) for r in top], rotation=35, ha="right", fontsize=8)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    _style(ax, title="Ordered previous-token to induction interaction checks", ylabel="interaction = effect(prev) - effect(prev | induction ablated)")
    bench.save_figure(ctx, fig, "edge_interactions.png", "Ablation-interaction evidence for the one edge claim.")


def plot_minimality_ledger(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = bench.new_figure(figsize=(8.4, max(3.8, 0.35 * len(rows) + 1.8)))
    ordered = sorted(rows, key=lambda r: _to_float(r.get("marginal_value"), 0.0))
    ys = list(range(len(ordered)))
    vals = [_to_float(r.get("marginal_value"), 0.0) for r in ordered]
    labels = [str(r.get("node", "")) for r in ordered]
    colors = [_motif_color(str(r.get("motif_label", "other"))) for r in ordered]
    ax.hlines(ys, 0, vals, color=colors, alpha=0.72, linewidth=2.8)
    ax.scatter(vals, ys, color=colors, s=85, edgecolors="black", linewidths=0.45, zorder=3)
    _zero_line(ax, axis="x")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    for y, v in zip(ys, vals):
        ax.annotate(f"{v:+.3f}", (v, y), textcoords="offset points", xytext=(5 if v >= 0 else -5, 0), ha="left" if v >= 0 else "right", va="center", fontsize=8)
    _style(ax, title="Minimality ledger: marginal faithfulness of each kept head", xlabel="faithfulness lost when this head is removed", ylabel="final circuit head")
    bench.save_figure(ctx, fig, "minimality_ledger.png", "Marginal value of each kept circuit head under the pruning rule.")


def plot_prompt_failure_scatter(ctx: bench.RunContext, rows: Sequence[dict[str, Any]], *, floor: float) -> None:
    valid = [r for r in rows if _is_finite(r.get("faithfulness")) and _is_finite(r.get("base_diff"))]
    if not valid:
        return
    fig, ax = bench.new_figure(figsize=(8.6, 5.5))
    for family in sorted({str(r.get("family", "")) for r in valid}):
        fam_rows = [r for r in valid if str(r.get("family", "")) == family]
        ax.scatter(
            [_to_float(r.get("base_diff"), 0.0) for r in fam_rows],
            [_to_float(r.get("faithfulness"), 0.0) for r in fam_rows],
            s=[54 + 22 * max(0.0, min(2.0, _to_float(r.get("over_recovery_above_full_model"), 0.0))) for r in fam_rows],
            color=_family_color(family),
            alpha=0.85,
            label=family,
            edgecolors="black",
            linewidths=0.35,
        )
    ax.axhline(floor, color="#D55E00", linestyle="--", linewidth=1.1, label="faithfulness floor")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.4, label="full-model baseline")
    label_rows = sorted(valid, key=lambda r: _to_float(r.get("faithfulness"), 0.0))[:3] + sorted(valid, key=lambda r: _to_float(r.get("faithfulness"), 0.0), reverse=True)[:2]
    seen: set[str] = set()
    for r in label_rows:
        node = str(r.get("example_id", ""))
        if node in seen:
            continue
        seen.add(node)
        ax.annotate(node, (_to_float(r.get("base_diff"), 0.0), _to_float(r.get("faithfulness"), 0.0)), textcoords="offset points", xytext=(5, 5), fontsize=8)
    _style(ax, title="Prompt-level audit: weak behavior, failures, and over-recovery", xlabel="full-model base logit diff", ylabel="circuit faithfulness")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "prompt_failure_scatter.png", "Per-prompt base behavior against circuit-only faithfulness.")


def plot_candidate_evidence_matrix(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    heads = [r for r in rows if r.get("node")]
    if not heads:
        return
    # Keep the final circuit and the strongest disagreements visible without making a postage stamp atlas.
    chosen = sorted(
        heads,
        key=lambda r: (
            not bool(r.get("in_final_circuit")),
            -abs(_to_float(r.get("rank_gap_cheap_minus_causal"), 0.0)),
            -_to_float(r.get("causal_drop"), 0.0),
        ),
    )[:28]
    columns = [
        ("|attr|", "abs_attr", False),
        ("induction", "induction_score", False),
        ("prev-token", "prev_token_score", False),
        ("sink", "first_token_score", False),
        ("causal drop", "causal_drop", True),
        ("marginal", "marginal_value", True),
    ]
    raw_cols: list[list[float]] = []
    for _, key, signed in columns:
        vals = [_to_float(r.get(key), 0.0) for r in chosen]
        if signed:
            scale = max([abs(v) for v in vals] + [1e-9])
            raw_cols.append([v / scale for v in vals])
        else:
            vmin, vmax = min(vals), max(vals)
            denom = max(vmax - vmin, 1e-9)
            raw_cols.append([((v - vmin) / denom) for v in vals])
    data = [[raw_cols[j][i] for j in range(len(columns))] for i in range(len(chosen))]

    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(data, dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, max(4.2, 0.36 * len(chosen) + 1.7)), constrained_layout=True)
    im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[0] for c in columns], rotation=30, ha="right")
    labels = [f"{'★ ' if r.get('in_final_circuit') else ''}{r['node']} · {r.get('motif_label', 'other')}" for r in chosen]
    ax.set_yticks(range(len(chosen)))
    ax.set_yticklabels(labels, fontsize=7.5)
    for i, r in enumerate(chosen):
        for j, (_label, key, _signed) in enumerate(columns):
            v = _to_float(r.get(key), float("nan"))
            txt = "" if not math.isfinite(v) else (f"{v:+.2f}" if key in {"causal_drop", "marginal_value"} else f"{v:.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.8, color="black")
    for i, r in enumerate(chosen):
        if r.get("edge_role"):
            ax.text(len(columns) - 0.1, i, str(r["edge_role"]).replace("edge_", ""), ha="left", va="center", fontsize=7, color="black")
    ax.set_title("Candidate evidence matrix: screen signals, causal tests, and final membership")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("column-normalized evidence score")
    bench.save_figure(ctx, fig, "candidate_evidence_matrix.png", "Candidate heads with cheap scores, motif labels, causal drops, final membership, and marginality.")


def plot_causal_motif_atlas(ctx: bench.RunContext, rows: Sequence[dict[str, Any]], n_layers: int, n_heads: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors

    if not rows:
        return
    grid = np.full((n_heads, n_layers), np.nan, dtype=float)
    in_circuit: list[tuple[int, int, str]] = []
    for r in rows:
        layer = _to_int(r.get("layer"), -1)
        head = _to_int(r.get("head"), -1)
        if 0 <= layer < n_layers and 0 <= head < n_heads:
            grid[head, layer] = _to_float(r.get("causal_drop"), 0.0)
            if r.get("in_final_circuit"):
                in_circuit.append((layer, head, str(r.get("motif_label", "other"))))
    if np.all(np.isnan(grid)):
        return
    vmax = float(np.nanmax(np.abs(grid))) if np.any(~np.isnan(grid)) else 1.0
    vmax = max(vmax, 1e-6)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    for layer, head, motif in in_circuit:
        ax.scatter(layer, head, marker=_motif_marker(motif), s=95, facecolors="none", edgecolors="black", linewidths=1.6)
        ax.annotate(f"L{layer}H{head}", (layer, head), textcoords="offset points", xytext=(3, 3), fontsize=6.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("head")
    ax.set_title("Causal motif atlas: screened heads on the layer-head grid")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("single-head causal drop")
    bench.save_figure(ctx, fig, "causal_motif_atlas.png", "Layer-head atlas of screened-head causal drops with final circuit heads outlined.")


def plot_edge_interaction_map(ctx: bench.RunContext, rows: Sequence[dict[str, Any]]) -> None:
    valid = [r for r in rows if _is_finite(r.get("interaction"))]
    if not valid:
        return
    fig, ax = bench.new_figure(figsize=(8.3, 5.5))
    for r in valid:
        frac = _to_float(r.get("raw_interaction_fraction"), 0.0)
        inter = _to_float(r.get("interaction"), 0.0)
        strength = str(r.get("edge_strength", "none"))
        color = _motif_color("induction") if strength in {"weak", "strong"} else _lighten(_motif_color("previous_token"), 0.35)
        ax.scatter(
            _to_float(r.get("from_layer"), 0.0),
            _to_float(r.get("to_layer"), 0.0),
            s=80 + 420 * max(0.0, min(0.12, frac)) / 0.12,
            color=color,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
        )
        if inter >= sorted([_to_float(x.get("interaction"), 0.0) for x in valid], reverse=True)[min(4, len(valid) - 1)]:
            ax.annotate(str(r.get("edge", "")), (_to_float(r.get("from_layer"), 0.0), _to_float(r.get("to_layer"), 0.0)), textcoords="offset points", xytext=(4, 4), fontsize=7)
    lo = min([_to_float(r.get("from_layer"), 0.0) for r in valid] + [_to_float(r.get("to_layer"), 0.0) for r in valid]) - 1
    hi = max([_to_float(r.get("from_layer"), 0.0) for r in valid] + [_to_float(r.get("to_layer"), 0.0) for r in valid]) + 1
    ax.plot([lo, hi], [lo, hi], color="black", alpha=0.25, linewidth=0.9, label="same layer")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _style(ax, title="Edge interaction map: ordered previous-token to induction pairs", xlabel="source layer", ylabel="target layer")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "edge_interaction_map.png", "Ordered edge checks by source and target layer, sized by routed-fraction proxy.")


def plot_circuit_discovery_dashboard(
    ctx: bench.RunContext,
    fcm: dict[str, Any],
    trajectory: Sequence[dict[str, Any]],
    cand_rows: Sequence[dict[str, Any]],
    prompt_rows: Sequence[dict[str, Any]],
    *,
    floor: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()

    # A. F/C/M scorecard.
    groups: list[tuple[str, str, float]] = []
    for family in ("discovery", "heldout"):
        if family in fcm:
            groups.append((family, "faithfulness", _to_float(fcm[family].get("faithfulness"), 0.0)))
            groups.append((family, "completeness effect", _to_float(fcm[family].get("completeness_effect"), 0.0)))
    xs = list(range(len(groups)))
    ax0.bar(xs, [v for _, _, v in groups], color=[_family_color(f) if m == "faithfulness" else _lighten(_family_color(f), 0.35) for f, m, _ in groups])
    ax0.axhline(floor, color="#D55E00", linestyle="--", linewidth=1.0)
    ax0.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
    ax0.set_xticks(xs)
    ax0.set_xticklabels([f"{m}\n{f}" for f, m, _ in groups], fontsize=8)
    _style(ax0, title="F/C scorecard", ylabel="fraction of base behavior")
    _panel_label(ax0, "A")

    # B. Pruning trajectory.
    if trajectory:
        x = [t["n_nodes"] for t in trajectory]
        y = [_to_float(t["faithfulness"], 0.0) for t in trajectory]
        ax1.plot(x, y, marker="o", color=_family_color("discovery"), linewidth=2.0)
        ax1.axhline(floor, color="#D55E00", linestyle="--", linewidth=1.0)
        ax1.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
        ax1.invert_xaxis()
    _style(ax1, title="Greedy pruning path", xlabel="heads kept", ylabel="faithfulness")
    _panel_label(ax1, "B")

    # C. Candidate quality.
    heads = [r for r in cand_rows if r.get("kind") == "head"]
    for r in heads:
        motif = str(r.get("motif_label", "other"))
        ax2.scatter(
            _to_float(r.get("cheap_rank"), 0.0),
            _to_float(r.get("causal_drop"), 0.0),
            color=_motif_color(motif),
            marker=_motif_marker(motif),
            s=54,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.35,
        )
    _zero_line(ax2)
    _style(ax2, title="Cheap screen vs causal test", xlabel="cheap rank, lower is better", ylabel="causal drop")
    _panel_label(ax2, "C")

    # D. Prompt audit.
    valid = [r for r in prompt_rows if _is_finite(r.get("faithfulness"))]
    ordered = sorted(valid, key=lambda r: _to_float(r.get("faithfulness"), 0.0))
    xs2 = list(range(len(ordered)))
    ax3.bar(xs2, [_to_float(r.get("faithfulness"), 0.0) for r in ordered], color=[_family_color(str(r.get("family", ""))) for r in ordered], alpha=0.85)
    ax3.axhline(floor, color="#D55E00", linestyle="--", linewidth=1.0)
    ax3.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
    ax3.set_xticks(xs2)
    ax3.set_xticklabels([str(r.get("example_id", "")) for r in ordered], rotation=35, ha="right", fontsize=7)
    _style(ax3, title="Prompt-level failures and over-recovery", ylabel="faithfulness")
    _panel_label(ax3, "D")

    fig.suptitle("Manual circuit discovery: from suspects to a scoped heads-only claim", fontsize=14)
    bench.save_figure(ctx, fig, "circuit_discovery_dashboard.png", "One-screen summary of Lab 6 F/C/M, pruning, screening, and prompt-level audit.")


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
    prompt_failure_rows = build_prompt_failure_modes(per_prompt_rows, floor=FAITHFULNESS_FLOOR)
    prompt_failure_path = ctx.path("tables", "prompt_failure_modes.csv")
    bench.write_csv_with_context(ctx, prompt_failure_path, prompt_failure_rows)
    ctx.register_artifact(prompt_failure_path, "table", "Prompt-level failure modes, including over-recovery and under-floor cases.")

    failures = sorted(
        (row for row in prompt_failure_rows if row["faithfulness"] is not None),
        key=lambda row: row["faithfulness"],
    )[:2]

    evidence_rows = build_circuit_evidence_matrix(cand_rows_sorted, minimality_rows, circuit, edge)
    evidence_path = ctx.path("tables", "circuit_evidence_matrix.csv")
    bench.write_csv_with_context(ctx, evidence_path, evidence_rows)
    ctx.register_artifact(evidence_path, "table", "Joined evidence for every screened head: screen, motif, causal, minimality, and edge status.")

    guide_path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, guide_path, plot_reading_guide_rows())
    ctx.register_artifact(guide_path, "table", "Short guide mapping each Lab 6 visualization to the concept it teaches.")

    # ----- plots ---------------------------------------------------------------
    if not args.no_plots:
        plot_circuit_discovery_dashboard(ctx, fcm, trajectory, cand_rows_sorted, prompt_failure_rows, floor=FAITHFULNESS_FLOOR)
        plot_screen_vs_causal(ctx, cand_rows_sorted)
        plot_prune_trajectory(ctx, trajectory, floor=FAITHFULNESS_FLOOR)
        mlp_support = [r for r in cand_rows_sorted if r["kind"] == "mlp" and float(r["causal_drop"]) > 0]
        plot_circuit_graph(ctx, circuit, head_labels, mlp_support, edge, n_layers, n_heads)
        if "discovery" in fcm:
            plot_fcm(ctx, fcm, floor=FAITHFULNESS_FLOOR)
        plot_prompt_faithfulness(ctx, prompt_failure_rows)
        plot_prompt_failure_scatter(ctx, prompt_failure_rows, floor=FAITHFULNESS_FLOOR)
        plot_minimality_ledger(ctx, minimality_rows)
        plot_candidate_evidence_matrix(ctx, evidence_rows)
        plot_causal_motif_atlas(ctx, evidence_rows, n_layers, n_heads)
        plot_edge_interactions(ctx, edge_rows)
        plot_edge_interaction_map(ctx, edge_rows)

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
        f"and previous-token top {top_prev}. Screening proposes suspects; mean-ablation decides which suspects matter. ",
        "See `tables/circuit_evidence_matrix.csv` and `plots/candidate_evidence_matrix.png` for the joined evidence ladder.",
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
        "2. `plots/circuit_discovery_dashboard.png` - F/C/M, pruning, screening, and prompt failures on one page.",
        "3. `tables/circuit_evidence_matrix.csv` and `plots/candidate_evidence_matrix.png` - the joined evidence ladder for every screened head.",
        "4. `plots/circuit_graph.png` - validated heads, support MLPs, and any claimed edge.",
        "5. `plots/prune_trajectory.png` and `plots/minimality_ledger.png` - what pruning costs and whether each final node earns its rent.",
        "6. `plots/screen_vs_causal.png` and `plots/causal_motif_atlas.png` - where cheap screening, motif labels, and causal effects agree or disagree.",
        "7. `tables/prompt_failure_modes.csv`, `plots/per_prompt_faithfulness.png`, and `plots/prompt_failure_scatter.png` - the specific prompts the circuit least explains or over-recovers.",
        "8. `plots/edge_interactions.png`, `plots/edge_interaction_map.png`, and `tables/edge_interactions.csv` - ordered interaction checks, weak versus strong, layer-order respected, not path patching.",
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
