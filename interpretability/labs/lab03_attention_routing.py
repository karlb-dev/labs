"""Lab 3: Attention — routing, induction, and what heads actually do.

Two ideas, deliberately separated:

1. **Routing is not contribution.** Attention weights say which positions a
   head reads; the head's *output* (its slice of the out-projection input,
   mapped through W_O) says what it writes. Both are measured for every head,
   and the lab's scatter plots are built on the tension between them.

2. **Heads compose.** Induction (Olsson et al.) is a two-head circuit: a
   previous-token head writes "what preceded me" at EARLIER positions; the
   induction head reads it from the final position. The lab makes this
   measurable with two ablation scopes — final-position-only (the direct path
   the attribution counted) vs all-positions (including upstream writes). A
   previous-token head with ~zero direct effect and a large all-position
   effect is composition caught in the act.

The bench owns capture (`run_with_attention_cache`), verified per-head
decomposition (`run_head_decomposition_check`), and scoped head ablation.
The lab owns motif scores, labels, attribution math (frozen-norm, continuing
Lab 2's convention — here the per-block post-norm composes with the final
norm on post-norm architectures), and the experiment design.

Evidence levels: OBS (patterns/motifs), ATTR (head attribution), CAUSAL
(scoped ablations, with the scope stated in every claim).
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import random
import statistics
from typing import Any

import interp_bench as bench

LAB_ID = "L03"

CATEGORIES = ("synthetic", "cycle", "natural", "control")

# Motif thresholds. Deliberately simple and visible; students are told the
# rule and invited to break it. Sink gets a HIGHER bar and LOWER priority:
# parking attention on position 0 is the model's resting pattern, so most
# heads carry large sink mass as background. A head that idles 65% on the
# sink and puts the rest on induction targets whenever they exist is an
# induction head — an argmax over raw scores would mislabel half the
# induction circuitry as sinks (we measured exactly that on gpt2 before
# fixing this rule; the head table keeps all raw scores so you can re-derive
# the mistake).
MOTIF_SCORE_BAR = 0.35
SINK_SCORE_BAR = 0.5
DIFFUSE_ENTROPY_FRAC = 0.85  # mean entropy / max possible entropy


@dataclasses.dataclass(frozen=True)
class PatternPrompt:
    example_id: str
    category: str
    prompt: str
    target: str
    distractor: str
    note: str = ""


# Synthetic vocabularies avoid alphabet/number-line priors: "A B C A B" is a
# broken microscope because the alphabetic continuation equals the induction
# answer. "B F Q" has no such prior. Each synthetic prompt repeats its cycle
# so induction has multiple query positions to act on.
ALL_EXAMPLES: tuple[PatternPrompt, ...] = (
    PatternPrompt("synth_letters", "synthetic", "B F Q B F Q B F", " Q", " B",
                  "Non-alphabetical letters: no sequence prior to confound induction."),
    PatternPrompt("synth_colors", "synthetic", "red blue green red blue green red blue", " green", " red"),
    PatternPrompt("synth_animals", "synthetic", "dog cat bird dog cat bird dog cat", " bird", " dog"),
    PatternPrompt("synth_numbers", "synthetic", "seven three nine seven three nine seven three", " nine", " seven",
                  "Number words in a non-arithmetic order."),
    PatternPrompt("cycle_moon", "cycle", "moon star moon star moon star moon", " star", " moon",
                  "Period-2 cycle: previous-token and induction predictions coincide less cleanly."),
    PatternPrompt("cycle_sun", "cycle", "sun rain sun rain sun rain sun", " rain", " sun"),
    PatternPrompt("nat_lab", "natural", "Marcus went to the lab. Olivia went to the", " lab", " store",
                  "Natural-text induction: repeated phrase, new subject."),
    PatternPrompt("nat_door", "natural", "The wizard opened the ancient door. Behind the ancient", " door", " wall"),
    PatternPrompt("nat_apples", "natural", "She bought fresh apples at the market. Everyone loved the fresh",
                  " apples", " fruit"),
    PatternPrompt("ctrl_fox", "control", "The quick brown fox jumps over the lazy", " dog", " cat",
                  "No token repeats: induction-pattern scores should be ~undefined/low here."),
    PatternPrompt("ctrl_paris", "control", "The Eiffel Tower is in", " Paris", " Rome",
                  "Factual recall, no repetition: a head strongly 'inducting' here is a label failure."),
)

SMALL_SET_IDS = ("synth_colors", "cycle_moon", "nat_lab", "ctrl_fox")
MEDIUM_SET_IDS = SMALL_SET_IDS + ("synth_letters", "synth_animals", "nat_door", "ctrl_paris")


def validate_prompt_schema(examples: list[PatternPrompt]) -> None:
    seen: set[str] = set()
    for ex in examples:
        if not ex.example_id or not ex.prompt:
            raise ValueError(f"Example {ex!r} is missing an id or prompt.")
        if ex.example_id in seen:
            raise ValueError(f"Duplicate example_id {ex.example_id!r}.")
        seen.add(ex.example_id)
        if ex.category not in CATEGORIES:
            raise ValueError(f"Example {ex.example_id!r}: unknown category {ex.category!r}.")
        if not ex.target or not ex.distractor:
            raise ValueError(f"Example {ex.example_id!r} needs a target and a distractor.")


def interleave_by_category(examples: list[PatternPrompt]) -> list[PatternPrompt]:
    queues: dict[str, list[PatternPrompt]] = {cat: [] for cat in CATEGORIES}
    for ex in examples:
        queues.setdefault(ex.category, []).append(ex)
    out: list[PatternPrompt] = []
    while any(queues.values()):
        for cat in queues:
            if queues[cat]:
                out.append(queues[cat].pop(0))
    return out


def load_custom_prompt_set(path: pathlib.Path) -> list[PatternPrompt]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"Could not read prompt set {str(path)!r}: {exc}. "
            "--prompt-set must be one of small | medium | full, or a path to a prompts .json file."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse prompt JSON at {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("Custom prompt file must be a JSON list of objects.")
    allowed = {f.name for f in dataclasses.fields(PatternPrompt)}
    out: list[PatternPrompt] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt item {i} is not an object: {item!r}")
        extra = set(item) - allowed
        if extra:
            raise ValueError(f"Prompt item {i} has unknown keys: {sorted(extra)}")
        out.append(PatternPrompt(**item))
    return out


def build_prompt_set(args: Any) -> list[PatternPrompt]:
    if args.prompt_set == "full":
        examples = list(ALL_EXAMPLES)
    elif args.prompt_set == "medium":
        examples = [ex for ex in ALL_EXAMPLES if ex.example_id in MEDIUM_SET_IDS]
    elif args.prompt_set == "small":
        examples = [ex for ex in ALL_EXAMPLES if ex.example_id in SMALL_SET_IDS]
    else:
        examples = load_custom_prompt_set(pathlib.Path(args.prompt_set))
    validate_prompt_schema(examples)
    examples = interleave_by_category(examples)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    return examples


# ---------------------------------------------------------------------------
# Motif scores
# ---------------------------------------------------------------------------
#
# All scores are means of attention mass over the query positions where the
# motif is DEFINED, so a score of 0.8 reads as "80% of this head's attention
# goes where the motif says, where the motif says anything at all".


def prev_token_score(pattern: Any) -> float:
    """Mean attention to position q-1, over q >= 1. pattern: [seq, seq]."""
    seq = pattern.shape[0]
    if seq < 2:
        return 0.0
    return float(statistics.fmean(float(pattern[q, q - 1]) for q in range(1, seq)))


def first_token_score(pattern: Any) -> float:
    """Mean attention to position 0 over q >= 1 (the attention-sink resting
    pattern; lands on BOS when one exists, plain first token here)."""
    seq = pattern.shape[0]
    if seq < 2:
        return 0.0
    return float(statistics.fmean(float(pattern[q, 0]) for q in range(1, seq)))


def induction_targets(input_ids: list[int]) -> dict[int, list[int]]:
    """For each query position q, the positions j+1 such that token[j] ==
    token[q] for a previous occurrence j < q. Empty dict entries omitted."""
    targets: dict[int, list[int]] = {}
    for q, tok in enumerate(input_ids):
        hits = [j + 1 for j in range(q) if input_ids[j] == tok and j + 1 <= q]
        if hits:
            targets[q] = hits
    return targets


def induction_score(pattern: Any, input_ids: list[int]) -> float | None:
    """Mean attention mass on induction targets, over queries where the motif
    is defined (the current token occurred before). None when never defined —
    control prompts should land here, and that None is itself data."""
    targets = induction_targets(input_ids)
    if not targets:
        return None
    masses = []
    for q, hits in targets.items():
        masses.append(float(sum(float(pattern[q, j]) for j in hits)))
    return float(statistics.fmean(masses))


def attention_entropy_bits(pattern: Any) -> tuple[float, float]:
    """(mean entropy in bits, mean entropy as a fraction of its causal max).

    Row q has q+1 valid keys, so the max entropy at q is log2(q+1); the
    fraction makes early rows comparable with late ones.
    """
    seq = pattern.shape[0]
    ents, fracs = [], []
    for q in range(1, seq):
        row = pattern[q, : q + 1]
        ent = 0.0
        for v in row.tolist():
            if v > 0:
                ent -= v * math.log2(v)
        ents.append(ent)
        fracs.append(ent / math.log2(q + 1))
    if not ents:
        return 0.0, 0.0
    return float(statistics.fmean(ents)), float(statistics.fmean(fracs))


def label_head(prev: float, induct: float | None, first: float, entropy_frac: float) -> str:
    """The transparent labeling rule (stated in the handout, meant to be
    argued with): content motifs above the bar win, larger one first; the
    sink label is a fallback for heads doing nothing else; high relative
    entropy with no motif is 'diffuse'; the rest are 'other'."""
    content = {"previous_token": prev, "induction": induct if induct is not None else 0.0}
    best = max(content, key=lambda k: content[k])
    if content[best] >= MOTIF_SCORE_BAR:
        return best
    if first >= SINK_SCORE_BAR:
        return "first_token_sink"
    if entropy_frac >= DIFFUSE_ENTROPY_FRAC:
        return "diffuse"
    return "other"


# ---------------------------------------------------------------------------
# Head attribution (frozen-norm, composing Lab 2's convention)
# ---------------------------------------------------------------------------


def head_attribution_scores(
    bundle: bench.ModelBundle,
    comp_anatomy: bench.ComponentAnatomy,
    head_anatomy: bench.HeadAnatomy,
    att: bench.AttentionCapture,
    target_id: int,
    distractor_id: int,
) -> dict[str, Any]:
    """Score every head's final-position write against the answer direction.

    Two frozen linearizations compose here, and the lab says so out loud:

    1. The FINAL norm is frozen at the actual final stream (exactly Lab 2's
       `w` vector — recomputed here for transparency).
    2. On post-norm architectures (Olmo-3), each block's attention output
       passes through a per-block norm BEFORE joining the stream. That norm's
       scale is frozen at the block's actual attention output, making head
       contributions linear within the block too. On GPT-2 this step is the
       identity.

    Each freeze is exact for the whole it was frozen on, approximate for the
    parts. The per-block sum of head scores (+ shared bias terms) therefore
    matches Lab 2's per-block attention score, which the run cross-checks.
    """
    import torch

    x_final = att.capture.streams[-1, -1]
    d_model = x_final.shape[0]
    w_u = bundle.lm_head.weight
    direction = (w_u[target_id].detach() - w_u[distractor_id].detach()).to("cpu", torch.float32)

    norm = bundle.final_norm
    gain = norm.weight.detach().to("cpu", torch.float32)
    is_rms = "rms" in type(norm).__name__.lower()
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))
    if is_rms:
        s_final = 1.0 / float(torch.sqrt(x_final.pow(2).mean() + eps))
        w = s_final * gain * direction
    else:
        var = float(x_final.var(unbiased=False))
        s_final = 1.0 / float((var + eps) ** 0.5)
        v = s_final * gain * direction
        w = v - v.mean() * torch.ones(d_model)

    n_layers = bundle.anatomy.n_layers
    n_heads = head_anatomy.n_heads
    scores = [[0.0] * n_heads for _ in range(n_layers)]
    for layer in range(n_layers):
        # Per-block post-norm linearization (identity on GPT-2).
        if comp_anatomy.attn_source == "post_norm":
            post = getattr(bundle.blocks[layer], comp_anatomy.attn_hook_path)
            p_gain = post.weight.detach().to("cpu", torch.float32)
            p_eps = float(getattr(post, "variance_epsilon", getattr(post, "eps", 1e-5)))
            a_full = att.attn_out_last[layer]
            p_scale = 1.0 / float(torch.sqrt(a_full.pow(2).mean() + p_eps))
            block_map = lambda c: p_scale * p_gain * c  # noqa: E731
        else:
            block_map = lambda c: c  # noqa: E731
        for head in range(n_heads):
            contrib = bench.head_contribution(bundle, head_anatomy, layer, head, att.o_in_last[layer])
            scores[layer][head] = float(block_map(contrib) @ w)
    return {"scores": scores, "direction_vector": w, "frozen_final_scale": s_final}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_attention_heatmap_panel(
    ctx: bench.RunContext,
    att: bench.AttentionCapture,
    heads: list[tuple[int, int, str]],
    example_id: str,
) -> None:
    """Token-labeled attention heatmaps for the showcase motif heads."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not heads:
        return
    labels = [bench.visible_token(t) for t in att.capture.tokens_text]
    n = len(heads)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, (layer, head, title) in zip(axes, heads):
        pattern = np.array(att.attentions[layer, head])
        im = ax.imshow(pattern, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"L{layer}H{head}\n{title}", fontsize=9)
        ax.set_xlabel("key (reads from)")
        ax.set_ylabel("query (reads at)")
    fig.colorbar(im, ax=axes, fraction=0.025, label="attention weight")
    fig.suptitle(f"Motif heads on {example_id!r} (rows sum to 1; lower triangle only — causal)")
    bench.save_figure(ctx, fig, f"attention_heads_{bench.sanitize_tag(example_id)}.png",
                      "Token-labeled attention patterns for the showcase motif heads.")


def plot_motif_maps(ctx: bench.RunContext, head_rows: list[dict[str, Any]], n_layers: int, n_heads: int) -> None:
    """Three layer-by-head grids: previous-token, induction, first-token-sink."""
    import matplotlib.pyplot as plt
    import numpy as np

    grids = {
        "previous_token": np.zeros((n_layers, n_heads)),
        "induction": np.zeros((n_layers, n_heads)),
        "first_token_sink": np.zeros((n_layers, n_heads)),
    }
    for r in head_rows:
        grids["previous_token"][r["layer"], r["head"]] = r["prev_token_score"]
        grids["induction"][r["layer"], r["head"]] = r["induction_score"] if r["induction_score"] != "" else 0.0
        grids["first_token_sink"][r["layer"], r["head"]] = r["first_token_score"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for ax, (name, grid) in zip(axes, grids.items()):
        im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("head")
    axes[0].set_ylabel("layer")
    fig.colorbar(im, ax=axes, fraction=0.02, label="motif score (mean attention mass)")
    fig.suptitle("Motif map: where the named attention patterns live (mean over repeat-bearing prompts)")
    bench.save_figure(ctx, fig, "motif_maps.png",
                      "Layer-by-head grids of previous-token, induction, and sink scores.")


def plot_head_attribution_zoom(
    ctx: bench.RunContext, head_rows: list[dict[str, Any]], n_layers: int
) -> None:
    """Lab 2 said which LAYERS matter; resolve those bars into heads."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = bench.new_figure(figsize=(10.0, 5.5))
    by_layer: dict[int, list[float]] = {l: [] for l in range(n_layers)}
    for r in head_rows:
        by_layer[r["layer"]].append(r["mean_target_attribution"])
    totals = [sum(v) for v in by_layer.values()]
    ax.bar(range(n_layers), totals, color="lightsteelblue", label="layer total (all heads)")
    top = sorted(head_rows, key=lambda r: abs(r["mean_target_attribution"]), reverse=True)[:8]
    for r in top:
        ax.scatter(r["layer"], r["mean_target_attribution"], s=46, zorder=3,
                   color="tab:red" if r["pattern_label"] == "induction" else "tab:orange")
        ax.annotate(f"L{r['layer']}H{r['head']} ({r['pattern_label']})",
                    (r["layer"], r["mean_target_attribution"]),
                    textcoords="offset points", xytext=(4, 5), fontsize=7, rotation=20)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean attribution to answer direction")
    ax.set_title("Lab 2's attention bars, resolved into heads (red = induction-labeled)")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "head_attribution_by_layer.png",
                      "Per-layer attention attribution resolved into individual heads.")


def plot_direct_vs_indirect(ctx: bench.RunContext, ablation_rows: list[dict[str, Any]]) -> None:
    """The composition reveal: direct-path effect vs all-position effect."""
    import matplotlib.pyplot as plt

    pairs: dict[tuple, dict[str, Any]] = {}
    for r in ablation_rows:
        key = (r["example_id"], r["layer"], r["head"])
        pairs.setdefault(key, {"label": r["pattern_label"]})[r["scope"]] = r["causal_effect"]
    xs, ys, labels = [], [], []
    for key, d in pairs.items():
        if "final_pos" in d and "all_pos" in d:
            xs.append(d["final_pos"])
            ys.append(d["all_pos"])
            labels.append(d["label"])
    if not xs:
        return
    fig, ax = bench.new_figure(figsize=(7.5, 6.5))
    colors = {"induction": "tab:red", "previous_token": "tab:blue",
              "first_token_sink": "tab:green", "diffuse": "tab:gray", "other": "tab:olive"}
    for lab in sorted(set(labels)):
        sel = [(x, y) for x, y, l in zip(xs, ys, labels) if l == lab]
        ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=44, alpha=0.8,
                   color=colors.get(lab, "tab:gray"), label=f"{lab} (n={len(sel)})")
    lim = max(max(abs(v) for v in xs), max(abs(v) for v in ys)) * 1.15 or 1.0
    ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=0.8, linestyle="--", label="direct = total")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("direct effect (head zeroed at final position only)")
    ax.set_ylabel("total effect (head zeroed at all positions)")
    ax.set_title("Composition made visible: points far above the diagonal\nact through later layers, not the direct path")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "direct_vs_indirect_effect.png",
                      "Per-head direct-path vs all-position ablation effect; the gap is composition.")


ATTRIBUTION_NOISE_FLOOR = 0.1  # bf16 logit-diff noise is ~0.06; below this, ranks are noise


def attribution_ablation_rhos(ablation_rows: list[dict[str, Any]]) -> tuple[float | None, float | None, int]:
    """(rho over all pairs, rho over above-noise pairs, n above noise).

    Most motif/control candidates have near-zero attribution AND near-zero
    effect; in bf16 both numbers sit below the measurement noise floor, so
    their relative ranks are coin flips that dilute the pooled correlation.
    Reporting both numbers — instead of quietly picking the prettier one —
    is the honest version of this comparison.
    """
    from labs.lab02_direct_logit_attribution import spearman_rho

    rows = [r for r in ablation_rows if r["scope"] == "final_pos"]
    if not rows:
        return None, None, 0
    xs = [r["attribution_score"] for r in rows]
    ys = [r["causal_effect"] for r in rows]
    signal = [(x, y) for x, y in zip(xs, ys) if abs(x) >= ATTRIBUTION_NOISE_FLOOR]
    rho_signal = spearman_rho([p[0] for p in signal], [p[1] for p in signal]) if len(signal) >= 3 else None
    return spearman_rho(xs, ys), rho_signal, len(signal)


def plot_attribution_vs_ablation(ctx: bench.RunContext, ablation_rows: list[dict[str, Any]]) -> float | None:
    rows = [r for r in ablation_rows if r["scope"] == "final_pos"]
    if not rows:
        return None
    xs = [r["attribution_score"] for r in rows]
    ys = [r["causal_effect"] for r in rows]
    rho, rho_signal, n_signal = attribution_ablation_rhos(ablation_rows)
    fig, ax = bench.new_figure(figsize=(7.5, 6.0))
    ax.scatter(xs, ys, s=40, alpha=0.8, color="tab:purple")
    ax.axvspan(-ATTRIBUTION_NOISE_FLOOR, ATTRIBUTION_NOISE_FLOOR, color="gray", alpha=0.12,
               label=f"|attribution| < {ATTRIBUTION_NOISE_FLOOR} (noise floor)")
    lim = max(max(abs(v) for v in xs), max(abs(v) for v in ys)) * 1.15 or 1.0
    ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("head attribution on this example (frozen-norm logit diff)")
    ax.set_ylabel("direct-path ablation effect")
    title = "Head attribution vs direct-path ablation"
    if rho is not None:
        title += f"\nrho = {rho:.3f} (all n={len(xs)})"
        if rho_signal is not None:
            title += f"; rho = {rho_signal:.3f} above noise floor (n={n_signal})"
    ax.set_title(title)
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "head_attribution_vs_ablation.png",
                      "Does per-head attribution predict direct-path ablation? Both rhos reported.")
    return rho


# ---------------------------------------------------------------------------
# Claims and summary
# ---------------------------------------------------------------------------


def draft_claims(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    head_table: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    natural_confirmations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    run_name = ctx.run_dir.name
    claims: list[dict[str, str]] = []

    inducts = [r for r in head_table if r["pattern_label"] == "induction"]
    if inducts:
        top = max(inducts, key=lambda r: r["induction_score"] if r["induction_score"] != "" else 0.0)
        claims.append(
            {
                "id": f"{LAB_ID}-C1",
                "tag": "OBS",
                "text": (
                    f"{len(inducts)} of {len(head_table)} heads in {bundle.anatomy.model_id} were labeled "
                    f"induction by the stated rule (motif score >= {MOTIF_SCORE_BAR}); the strongest, "
                    f"L{top['layer']}H{top['head']}, put {top['induction_score']:.2f} of its attention mass "
                    "on induction targets in repeat-bearing prompts."
                ),
                "artifact": f"runs/{run_name}/tables/head_table.csv",
                "falsifier": (
                    "On a fresh repeated-pattern family (different vocabulary), the same heads do not "
                    "score above the bar — the label was prompt-set-specific."
                ),
            }
        )
    if natural_confirmations:
        ok = [r for r in natural_confirmations if r["confirmed"]]
        claims.append(
            {
                "id": f"{LAB_ID}-C2",
                "tag": "OBS",
                "text": (
                    f"{len(ok)}/{len(natural_confirmations)} synthetic-labeled induction heads kept an "
                    "induction score >= half their synthetic score on natural repeated-phrase prompts — "
                    "the motif is not an artifact of the toy patterns."
                ),
                "artifact": f"runs/{run_name}/tables/natural_confirmation.csv",
                "falsifier": "Longer natural documents with distractor repeats break the correspondence.",
            }
        )
    pairs: dict[tuple, dict[str, float]] = {}
    for r in ablation_rows:
        key = (r["example_id"], r["layer"], r["head"], r["pattern_label"])
        pairs.setdefault(key, {})[r["scope"]] = r["causal_effect"]
    comp_cases = [
        (k, v) for k, v in pairs.items()
        if k[3] == "previous_token" and "final_pos" in v and "all_pos" in v
        and abs(v["final_pos"]) < 0.25 * abs(v["all_pos"]) and abs(v["all_pos"]) > 0.2
    ]
    if comp_cases:
        (ex, layer, head, _), v = max(comp_cases, key=lambda kv: abs(kv[1]["all_pos"]))
        claims.append(
            {
                "id": f"{LAB_ID}-C3",
                "tag": "CAUSAL",
                "text": (
                    f"Previous-token head L{layer}H{head} shows composition: zeroing it at the final position "
                    f"changes the answer logit gap by {v['final_pos']:+.2f}, but zeroing it at ALL positions "
                    f"changes it by {v['all_pos']:+.2f} (on {ex}). Its causal role runs through what it writes "
                    "at earlier positions for later heads to read, not through its own direct output."
                ),
                "artifact": f"runs/{run_name}/tables/head_ablation_results.csv",
                "falsifier": (
                    "Patching (Lab 5) fails to localize the indirect path to the candidate induction head's "
                    "keys/queries — the all-position effect was diffuse, not composition."
                ),
            }
        )
    return claims


def render_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    head_table: list[dict[str, Any]],
    label_counts: dict[str, int],
    ablation_rows: list[dict[str, Any]],
    rho: float | None,
    natural_confirmations: list[dict[str, Any]],
    dropped: int,
    n_examples: int,
    claims: list[dict[str, str]],
) -> str:
    a = bundle.anatomy
    args = ctx.args
    lines = [
        "# Lab 3 run summary: attention routing and head motifs",
        "",
        "## Run identity",
        "",
        f"- model: `{a.model_id}` ({a.n_layers} blocks x {head_table[-1]['head'] + 1} heads)",
        f"- dtype: `{args.dtype}` | attn implementation: `{args.attn_implementation}` (patterns require eager)",
        f"- examples: {n_examples} kept, {dropped} dropped at the single-token gate",
        "- evidence levels: OBS (motifs), ATTR (head attribution), CAUSAL (scoped ablations)",
        "- self-checks: hook parity, lens self-check, component anatomy, head decomposition",
        "",
        "## 1. What behavior was studied?",
        "",
        "Next-token copying under repetition (synthetic cycles, natural repeated phrases) and",
        "non-repeating controls, with single-token target/distractor pairs.",
        "",
        "## 2. What internal object was measured?",
        "",
        "Every head's attention pattern (routing) AND its final-position write scored against",
        "the answer direction (contribution) — the lab's premise is that these are different",
        "facts, and the artifacts keep them separate.",
        "",
        "## 3. What intervention or control was used?",
        "",
        "Scoped head ablation: final-position-only (direct path, commensurable with attribution)",
        "vs all-positions (includes upstream writes that later layers read). Control prompts",
        "without repetition keep the induction label honest; random and low-score heads anchor",
        "the ablation comparison.",
        "",
        "## 4. Headline numbers",
        "",
        f"- head labels: " + ", ".join(f"{k}: {v}" for k, v in sorted(label_counts.items())),
    ]
    top_attr = sorted(head_table, key=lambda r: abs(r["mean_target_attribution"]), reverse=True)[:5]
    lines.append("- top heads by |attribution|: " + ", ".join(
        f"L{r['layer']}H{r['head']} ({r['pattern_label']}, {r['mean_target_attribution']:+.2f})" for r in top_attr
    ))
    if rho is not None:
        lines.append(f"- Spearman rho (attribution vs direct-path ablation): {rho:.3f}")
    if natural_confirmations:
        ok = sum(1 for r in natural_confirmations if r["confirmed"])
        lines.append(f"- induction heads confirmed on natural text: {ok}/{len(natural_confirmations)}")
    lines += [
        "",
        "## 5. What claim is supported, and at what evidence level?",
        "",
    ]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    lines += [
        "",
        "## 6. The reading order",
        "",
        "1. `plots/motif_maps.png` — where the named patterns live.",
        "2. `plots/attention_heads_*.png` — what those patterns look like on real tokens.",
        "3. `plots/head_attribution_by_layer.png` — Lab 2's attention bars resolved into heads.",
        "4. `plots/direct_vs_indirect_effect.png` — composition: the lab's payload.",
        "5. `tables/head_table.csv` — every head, every score, one row.",
        "",
        "## 7. Caveats students must carry forward",
        "",
        "- A motif label is a description of a pattern on THIS prompt distribution, not a job title.",
        "- High attention is not high contribution (compare the motif maps with the attribution zoom).",
        "- The all-position ablation effect bundles every indirect path; Lab 5's patching",
        "  separates them.",
        "- The sink heads' large attention mass with near-zero attribution is the canonical",
        "  'heatmap astrology' trap — write it down once and never fall for it again.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    examples = build_prompt_set(args)
    print(f"[lab3] prompt set: {len(examples)} examples")

    # Single-token gate (same contract as Lab 2).
    tokenizer = bundle.tokenizer
    kept: list[tuple[PatternPrompt, int, int]] = []
    gate_rows = []
    for ex in examples:
        t_ids = tokenizer.encode(ex.target, add_special_tokens=False)
        d_ids = tokenizer.encode(ex.distractor, add_special_tokens=False)
        ok = len(t_ids) == 1 and len(d_ids) == 1
        gate_rows.append(
            {
                "example_id": ex.example_id, "category": ex.category, "kept": ok,
                "target": bench.visible_token(ex.target), "target_n_tokens": len(t_ids),
                "distractor": bench.visible_token(ex.distractor), "distractor_n_tokens": len(d_ids),
                "note": ex.note,
            }
        )
        if ok:
            kept.append((ex, t_ids[0], d_ids[0]))
        else:
            print(f"[lab3] dropping {ex.example_id}: multi-token answer")
    gate_path = ctx.path("diagnostics", "answer_tokenization.csv")
    bench.write_csv(gate_path, gate_rows)
    ctx.register_artifact(gate_path, "diagnostic", "Single-token gate for targets/distractors.")
    if not kept:
        raise RuntimeError("Every example was dropped at the single-token gate.")
    dropped = len(examples) - len(kept)
    print(f"[lab3] running {len(kept)} examples ({dropped} dropped)")

    # Instrument verification before any science.
    first_prompt = kept[0][0].prompt
    bench.run_hook_parity_check(ctx, bundle, first_prompt)
    comp_anatomy = bench.resolve_component_anatomy(ctx, bundle, first_prompt, rel_tolerance=args.dla_tolerance)
    head_anatomy = bench.resolve_head_anatomy(ctx, bundle)
    first_att = bench.run_with_attention_cache(bundle, first_prompt)
    bench.run_lens_self_check(ctx, bundle, first_att.capture)
    bench.run_head_decomposition_check(ctx, bundle, head_anatomy, first_att, rel_tolerance=args.dla_tolerance)

    n_layers = bundle.anatomy.n_layers
    n_heads = head_anatomy.n_heads

    # Per-example capture and per-head measurement.
    captures: dict[str, bench.AttentionCapture] = {}
    per_head_acc: dict[tuple[int, int], dict[str, list]] = {
        (l, h): {"prev": [], "induct": [], "first": [], "ent": [], "ent_frac": [], "attr": []}
        for l in range(n_layers) for h in range(n_heads)
    }
    natural_induct: dict[tuple[int, int], list[float]] = {}
    synth_induct: dict[tuple[int, int], list[float]] = {}

    for i, (ex, t_id, d_id) in enumerate(kept):
        att = first_att if ex.prompt == first_prompt else bench.run_with_attention_cache(bundle, ex.prompt)
        captures[ex.example_id] = att
        attr = head_attribution_scores(bundle, comp_anatomy, head_anatomy, att, t_id, d_id)
        model_diff = float(att.capture.final_logits_last[t_id] - att.capture.final_logits_last[d_id])
        for l in range(n_layers):
            for h in range(n_heads):
                acc = per_head_acc[(l, h)]
                pattern = att.attentions[l, h]
                acc["prev"].append(prev_token_score(pattern))
                acc["first"].append(first_token_score(pattern))
                ind = induction_score(pattern, att.capture.input_ids)
                if ind is not None:
                    acc["induct"].append(ind)
                    bucket = natural_induct if ex.category == "natural" else (
                        synth_induct if ex.category in ("synthetic", "cycle") else None
                    )
                    if bucket is not None:
                        bucket.setdefault((l, h), []).append(ind)
                ent, ent_frac = attention_entropy_bits(pattern)
                acc["ent"].append(ent)
                acc["ent_frac"].append(ent_frac)
                acc["attr"].append(attr["scores"][l][h])
        traj = bench.compute_lens_trajectory(bundle, att.capture, target_id=t_id, distractor_id=d_id, topk=args.topk)
        bench.dump_example_state(ctx, bundle, ex.example_id, att.capture, traj, target=ex.target, distractor=ex.distractor)
        print(f"[lab3] [{i + 1}/{len(kept)}] {ex.example_id} logit_diff={model_diff:+.3f}")

    # The head table: one row per head, every score.
    head_table: list[dict[str, Any]] = []
    for l in range(n_layers):
        for h in range(n_heads):
            acc = per_head_acc[(l, h)]
            prev = statistics.fmean(acc["prev"])
            first = statistics.fmean(acc["first"])
            induct = statistics.fmean(acc["induct"]) if acc["induct"] else None
            ent_frac = statistics.fmean(acc["ent_frac"])
            head_table.append(
                {
                    "layer": l,
                    "head": h,
                    "prev_token_score": round(prev, 4),
                    "induction_score": round(induct, 4) if induct is not None else "",
                    "first_token_score": round(first, 4),
                    "mean_entropy_bits": round(statistics.fmean(acc["ent"]), 4),
                    "mean_entropy_frac": round(ent_frac, 4),
                    "mean_target_attribution": round(statistics.fmean(acc["attr"]), 4),
                    "pattern_label": label_head(prev, induct, first, ent_frac),
                }
            )
    head_path = ctx.path("tables", "head_table.csv")
    bench.write_csv(head_path, head_table)
    ctx.register_artifact(head_path, "table", "Every head: motif scores, entropy, label, attribution.")
    label_counts: dict[str, int] = {}
    for r in head_table:
        label_counts[r["pattern_label"]] = label_counts.get(r["pattern_label"], 0) + 1
    print("[lab3] head labels: " + ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items())))

    # Natural-text confirmation of synthetic-labeled induction heads.
    natural_confirmations = []
    for (l, h), synth_scores in sorted(synth_induct.items(), key=lambda kv: -statistics.fmean(kv[1]))[:8]:
        s_mean = statistics.fmean(synth_scores)
        if s_mean < MOTIF_SCORE_BAR:
            continue
        nat = natural_induct.get((l, h), [])
        n_mean = statistics.fmean(nat) if nat else 0.0
        natural_confirmations.append(
            {
                "layer": l, "head": h,
                "synthetic_induction_score": round(s_mean, 4),
                "natural_induction_score": round(n_mean, 4),
                "confirmed": bool(nat) and n_mean >= 0.5 * s_mean,
            }
        )
    if natural_confirmations:
        nat_path = ctx.path("tables", "natural_confirmation.csv")
        bench.write_csv(nat_path, natural_confirmations)
        ctx.register_artifact(nat_path, "table", "Do synthetic-labeled induction heads still induct on natural text?")

    # Scoped ablations: top heads per motif + by attribution + controls.
    ablation_rows: list[dict[str, Any]] = []
    if args.ablate_top > 0:
        by_label = lambda lab: [r for r in head_table if r["pattern_label"] == lab]  # noqa: E731
        candidates: list[dict[str, Any]] = []
        candidates += sorted(by_label("induction"), key=lambda r: -(r["induction_score"] or 0))[: args.ablate_top]
        candidates += sorted(by_label("previous_token"), key=lambda r: -r["prev_token_score"])[:2]
        candidates += sorted(by_label("first_token_sink"), key=lambda r: -r["first_token_score"])[:1]
        chosen_keys = {(r["layer"], r["head"]) for r in candidates}
        by_attr = sorted(head_table, key=lambda r: -abs(r["mean_target_attribution"]))
        candidates += [r for r in by_attr if (r["layer"], r["head"]) not in chosen_keys][:2]
        chosen_keys = {(r["layer"], r["head"]) for r in candidates}
        rng = random.Random(args.seed)
        pool = [r for r in head_table if (r["layer"], r["head"]) not in chosen_keys]
        candidates += rng.sample(pool, k=min(2, len(pool)))  # random controls
        candidates += sorted(pool, key=lambda r: abs(r["mean_target_attribution"]))[:1]  # low-attr control

        kept_index = {ex.example_id: i for i, (ex, _, _) in enumerate(kept)}
        abl_examples = [(ex, t, d) for ex, t, d in kept if ex.category in ("synthetic", "cycle", "natural")]
        abl_examples = abl_examples[:4] if len(abl_examples) > 4 else abl_examples
        total = len(candidates) * 2 * len(abl_examples)
        print(f"[lab3] ablating {len(candidates)} heads x 2 scopes x {len(abl_examples)} prompts = {total} forwards")
        for r in candidates:
            for ex, t_id, d_id in abl_examples:
                base = captures[ex.example_id].capture.final_logits_last
                base_diff = float(base[t_id] - base[d_id])
                # Attribution on THIS example, not the cross-prompt mean —
                # the ablation effect is per-example, so the pairing must be.
                example_attr = per_head_acc[(r["layer"], r["head"])]["attr"][kept_index[ex.example_id]]
                for scope in ("final_pos", "all_pos"):
                    logits = bench.run_with_head_ablation(
                        bundle, ex.prompt, head_anatomy, r["layer"], r["head"], scope=scope
                    )
                    abl_diff = float(logits[t_id] - logits[d_id])
                    ablation_rows.append(
                        {
                            "example_id": ex.example_id,
                            "category": ex.category,
                            "layer": r["layer"],
                            "head": r["head"],
                            "pattern_label": r["pattern_label"],
                            "scope": scope,
                            "attribution_score": round(example_attr, 4),
                            "mean_attribution_score": r["mean_target_attribution"],
                            "base_logit_diff": round(base_diff, 4),
                            "ablated_logit_diff": round(abl_diff, 4),
                            "causal_effect": round(base_diff - abl_diff, 4),
                        }
                    )
        abl_path = ctx.path("tables", "head_ablation_results.csv")
        bench.write_csv(abl_path, ablation_rows)
        ctx.register_artifact(abl_path, "table", "Scoped head ablations: direct-path vs all-position effects.")

    # Plots.
    rho = None
    if not args.no_plots:
        plot_motif_maps(ctx, head_table, n_layers, n_heads)
        showcase_id = args.showcase or next(
            (ex.example_id for ex, _, _ in kept if ex.category == "synthetic"), kept[0][0].example_id
        )
        if showcase_id in captures:
            best = {
                lab: max(by, key=lambda r: r["mean_target_attribution" if lab == "induction" else
                                            ("prev_token_score" if lab == "previous_token" else "first_token_score")])
                for lab, by in (
                    (lab, [r for r in head_table if r["pattern_label"] == lab])
                    for lab in ("induction", "previous_token", "first_token_sink")
                )
                if by
            }
            heads = [(r["layer"], r["head"], lab.replace("_", " ")) for lab, r in best.items()]
            plot_attention_heatmap_panel(ctx, captures[showcase_id], heads, showcase_id)
        else:
            print(f"[lab3] WARNING: --showcase {args.showcase!r} did not match any kept example id.")
        plot_head_attribution_zoom(ctx, head_table, n_layers)
        plot_direct_vs_indirect(ctx, ablation_rows)
        rho = plot_attribution_vs_ablation(ctx, ablation_rows)

    rho_all, rho_signal, n_signal = attribution_ablation_rhos(ablation_rows)
    metrics = {
        "n_examples": len(kept),
        "n_dropped": dropped,
        "label_counts": label_counts,
        "spearman_attribution_vs_direct_ablation": rho_all,
        "spearman_above_noise_floor": rho_signal,
        "n_pairs_above_noise_floor": n_signal,
        "attribution_noise_floor": ATTRIBUTION_NOISE_FLOOR,
        "n_ablations": len(ablation_rows),
        "natural_confirmations": natural_confirmations,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 3 metrics.")

    claims = draft_claims(ctx, bundle, head_table, ablation_rows, natural_confirmations)
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    summary = render_summary(
        ctx, bundle, head_table, label_counts, ablation_rows, rho,
        natural_confirmations, dropped, len(kept), claims,
    )
    summary_path = ctx.path("run_summary.md")
    bench.write_text(summary_path, summary)
    ctx.register_artifact(summary_path, "summary", "The seven standard questions answered with this run's numbers.")
    print(f"[lab3] wrote run_summary.md and {len(claims)} drafted ledger claims")
