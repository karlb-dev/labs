"""Lab 2: Direct logit attribution and component accounting.

The experiment: decompose a final logit difference into the contributions of
the embedding stream and every attention/MLP block, scored against an answer
direction ``unembed[target] - unembed[distractor]``.

The pedagogical core lives in :func:`compute_direct_logit_attribution`,
deliberately written out in this file rather than hidden in the bench: the
course runs raw HF weights (no LayerNorm folding), so the final norm must be
*linearized* before component scores mean anything, and students should see
exactly where that approximation enters (it is exact for the sum, frozen for
the parts).

The bench owns the instrument: verified contribution hook points
(``resolve_component_anatomy``), the capture (``run_with_component_cache``),
the decomposition self-check (``run_decomposition_check``), and direct-path
ablation (``run_with_component_ablation``). The lab owns the question.

Evidence levels: attribution (ATTR) for the ledger itself; the ablation
extension produces narrow CAUSAL evidence scoped to the direct path.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import random
import statistics
from typing import Any

import interp_bench as bench

LAB_ID = "L02"

CATEGORIES = ("fact", "relation", "grammar", "conflict")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
#
# Every example is a single-token target/distractor pair, verified on both
# course tokenizers (gpt2 BPE and Olmo-3) at authoring time -- except
# `plural_mouse`, which is included BECAUSE " mouses" is multi-token on both:
# the tokenization gate should be seen working, not just described.
#
# Categories:
#   fact      -- stored knowledge; distractor is a plausible same-type confuser
#   relation  -- antonym completions; distractor is a near-synonym or same-axis foil
#   grammar   -- morphology; distractor is the unmarked/wrong form
#   conflict  -- in-context override of a stored fact; target is the IN-CONTEXT
#                answer, so components pushing the stored fact score NEGATIVE


@dataclasses.dataclass(frozen=True)
class AnswerPairExample:
    example_id: str
    category: str
    prompt: str
    target: str
    distractor: str
    note: str = ""


ALL_EXAMPLES: tuple[AnswerPairExample, ...] = (
    AnswerPairExample("cap_germany", "fact", "The capital of Germany is", " Berlin", " Paris",
                      "The canonical DLA example; Paris is a high-prior city confuser."),
    AnswerPairExample("cap_france", "fact", "The capital of France is", " Paris", " London"),
    AnswerPairExample("eiffel", "fact", "The Eiffel Tower is in", " Paris", " Rome"),
    AnswerPairExample("water_oxygen", "fact", "Water is made of hydrogen and", " oxygen", " carbon"),
    AnswerPairExample("jupiter", "fact", "The largest planet in the Solar System is", " Jupiter", " Saturn"),
    AnswerPairExample("opp_hot", "relation", "The opposite of hot is", " cold", " warm",
                      "Distractor is on the same temperature axis but not the antonym."),
    AnswerPairExample("opp_up", "relation", "The opposite of up is", " down", " left"),
    AnswerPairExample("opp_big", "relation", "The opposite of big is", " small", " large",
                      "Distractor is a near-synonym of the subject, not of the answer."),
    AnswerPairExample("opp_day", "relation", "The opposite of day is", " night", " morning"),
    AnswerPairExample("plural_dog", "grammar", "The plural of dog is", " dogs", " dog"),
    AnswerPairExample("past_run", "grammar", "The past tense of run is", " ran", " runs"),
    AnswerPairExample("past_eat", "grammar", "The past tense of eat is", " ate", " eats"),
    AnswerPairExample("plural_mouse", "grammar", "The plural of mouse is", " mice", " mouses",
                      "Deliberately kept: ' mouses' is multi-token on both course "
                      "tokenizers, so this example exercises the drop path."),
    AnswerPairExample("conflict_germany", "conflict",
                      "In this story, the capital of Germany is Paris. The capital of Germany is",
                      " Paris", " Berlin",
                      "Target is the in-context answer; the stored fact is the distractor."),
    AnswerPairExample("conflict_hot", "conflict",
                      "On opposite day, the opposite of hot is warm. The opposite of hot is",
                      " warm", " cold"),
    AnswerPairExample("conflict_sky", "conflict",
                      "In the novel, the sky is green. In the novel, the sky is",
                      " green", " blue"),
)

SMALL_SET_IDS = ("cap_germany", "opp_hot", "past_run", "conflict_germany")
MEDIUM_SET_IDS = SMALL_SET_IDS + ("cap_france", "opp_big", "plural_dog", "conflict_sky")


def validate_prompt_schema(examples: list[AnswerPairExample]) -> None:
    seen: set[str] = set()
    for ex in examples:
        if not ex.example_id or not ex.prompt:
            raise ValueError(f"Example {ex!r} is missing an id or prompt.")
        if ex.example_id in seen:
            raise ValueError(f"Duplicate example_id {ex.example_id!r}.")
        seen.add(ex.example_id)
        if ex.category not in CATEGORIES:
            raise ValueError(
                f"Example {ex.example_id!r} has unknown category {ex.category!r}; "
                f"expected one of {CATEGORIES}."
            )
        if not ex.target or not ex.distractor:
            raise ValueError(
                f"Example {ex.example_id!r} needs both a target and a distractor: "
                "DLA scores an answer DIRECTION, which is a difference of two tokens."
            )
        if ex.prompt.endswith(" "):
            raise ValueError(
                f"Example {ex.example_id!r}: prompt ends with a space, but targets "
                "carry their own leading space (tokenizer convention)."
            )


def interleave_by_category(examples: list[AnswerPairExample]) -> list[AnswerPairExample]:
    """Round-robin so a --max-examples cap still covers every category."""
    queues: dict[str, list[AnswerPairExample]] = {cat: [] for cat in CATEGORIES}
    for ex in examples:
        queues.setdefault(ex.category, []).append(ex)
    out: list[AnswerPairExample] = []
    while any(queues.values()):
        for cat in queues:
            if queues[cat]:
                out.append(queues[cat].pop(0))
    return out


def load_custom_prompt_set(path: pathlib.Path) -> list[AnswerPairExample]:
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
    allowed = {f.name for f in dataclasses.fields(AnswerPairExample)}
    examples: list[AnswerPairExample] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt item {i} is not an object: {item!r}")
        extra = set(item) - allowed
        if extra:
            raise ValueError(f"Prompt item {i} has unknown keys: {sorted(extra)}")
        examples.append(AnswerPairExample(**item))
    return examples


def build_prompt_set(args: Any) -> list[AnswerPairExample]:
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
# Tokenization gate
# ---------------------------------------------------------------------------


def single_token_id(tokenizer: Any, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def tokenize_and_filter(
    ctx: bench.RunContext, bundle: bench.ModelBundle, examples: list[AnswerPairExample]
) -> tuple[list[tuple[AnswerPairExample, int, int]], int]:
    """Keep only examples whose target AND distractor are single tokens."""
    tokenizer = bundle.tokenizer
    rows: list[dict[str, Any]] = []
    kept: list[tuple[AnswerPairExample, int, int]] = []
    for ex in examples:
        t_ids = tokenizer.encode(ex.target, add_special_tokens=False)
        d_ids = tokenizer.encode(ex.distractor, add_special_tokens=False)
        ok = len(t_ids) == 1 and len(d_ids) == 1
        reason = ""
        if len(t_ids) != 1:
            reason = f"target {ex.target!r} tokenizes to {len(t_ids)} tokens"
        elif len(d_ids) != 1:
            reason = f"distractor {ex.distractor!r} tokenizes to {len(d_ids)} tokens"
        rows.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "kept": ok,
                "drop_reason": reason,
                "target": bench.visible_token(ex.target),
                "target_n_tokens": len(t_ids),
                "target_id": t_ids[0] if len(t_ids) == 1 else "",
                "target_pieces": "|".join(tokenizer.convert_ids_to_tokens(t_ids)),
                "distractor": bench.visible_token(ex.distractor),
                "distractor_n_tokens": len(d_ids),
                "distractor_id": d_ids[0] if len(d_ids) == 1 else "",
                "distractor_pieces": "|".join(tokenizer.convert_ids_to_tokens(d_ids)),
                "n_prompt_tokens": len(tokenizer.encode(ex.prompt, add_special_tokens=False)),
                "note": ex.note,
            }
        )
        if ok:
            kept.append((ex, t_ids[0], d_ids[0]))
        else:
            print(f"[lab2] dropping {ex.example_id}: {reason}")
    path = ctx.path("diagnostics", "answer_tokenization.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "diagnostic", "Single-token gate for every target/distractor, with drop reasons.")
    return kept, len(examples) - len(kept)


# ---------------------------------------------------------------------------
# The core: frozen-norm direct logit attribution
# ---------------------------------------------------------------------------


def compute_direct_logit_attribution(
    bundle: bench.ModelBundle,
    comp: bench.ComponentCapture,
    target_id: int,
    distractor_id: int,
) -> dict[str, Any]:
    """Score every component against the answer direction, frozen-norm style.

    The model computes, at the final position:

        logits = lm_head( final_norm( x ) )           x = final pre-norm stream
        x = embed + sum_k attn_k + sum_k mlp_k        (verified by the bench)

    We want per-component logit-difference scores that ADD UP to the real
    logit difference. The obstacle is final_norm: its scale factor depends on
    the WHOLE stream, so it is not linear in the components. The standard
    move (this course's pinned convention) is to FREEZE the data-dependent
    statistics at their actual values from this forward pass, which makes the
    map linear -- exactly correct for the full stream x, an approximation
    when read per-component:

      RMSNorm:    y = (x / rms(x)) * g
                  freeze s = 1/rms(x)  ->  y = (s*g) ∘ x      (linear)
      LayerNorm:  y = ((x - mean(x)) / std(x)) * g + b
                  freeze s = 1/std(x)  ->  mean() is already linear;
                  bias b is a constant shared term, not any component's credit

    With answer direction d = unembed[target] - unembed[distractor], the
    score of component c is a single dot product c @ w, where w folds the
    frozen scale, the norm gain, and (for LayerNorm) the mean-subtraction:

      RMSNorm:    w = s * (g ∘ d)
      LayerNorm:  w = v - mean(v) * ones,  v = s * (g ∘ d)
                  (because (c - mean(c)*ones) @ v == c @ (v - mean(v)*ones))

    Constant terms that belong to no component (LayerNorm bias through d,
    lm_head bias difference) are reported separately so the ledger still sums
    to the model's logit difference instead of silently leaking.
    """
    import torch

    x_final = comp.capture.streams[-1, -1]                      # [d_model] fp32 cpu
    d_model = x_final.shape[0]

    # Answer direction in fp32 on CPU, regardless of model dtype/device.
    w_u = bundle.lm_head.weight
    direction = (w_u[target_id].detach() - w_u[distractor_id].detach()).to("cpu", torch.float32)

    norm = bundle.final_norm
    norm_class = type(norm).__name__
    gain = norm.weight.detach().to("cpu", torch.float32)
    is_rms = "rms" in norm_class.lower()
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))

    constant = 0.0
    if is_rms:
        frozen_scale = 1.0 / float(torch.sqrt(x_final.pow(2).mean() + eps))
        w = frozen_scale * gain * direction
    else:
        # LayerNorm uses the biased variance of the full stream.
        var = float(x_final.var(unbiased=False))
        frozen_scale = 1.0 / float((var + eps) ** 0.5)
        v = frozen_scale * gain * direction
        w = v - v.mean() * torch.ones(d_model)
        norm_bias = getattr(norm, "bias", None)
        if norm_bias is not None:
            constant += float(norm_bias.detach().to("cpu", torch.float32) @ direction)
    lm_bias = getattr(bundle.lm_head, "bias", None)
    if lm_bias is not None:
        lm_bias = lm_bias.detach().to("cpu", torch.float32)
        constant += float(lm_bias[target_id] - lm_bias[distractor_id])

    # The ledger: one score per component, plus the shared constant.
    embed_score = float(comp.capture.streams[0, -1] @ w)
    attn_scores = [float(comp.attn_contrib[k] @ w) for k in range(comp.attn_contrib.shape[0])]
    mlp_scores = [float(comp.mlp_contrib[k] @ w) for k in range(comp.mlp_contrib.shape[0])]

    # Bookkeeping checks. ledger_total vs frozen_logit_diff is the linearity
    # bookkeeping (exact up to the bench-verified decomposition residual);
    # frozen vs model is the fp32-reimplementation-vs-compute-dtype gap and
    # is reported, not enforced.
    ledger_total = embed_score + sum(attn_scores) + sum(mlp_scores) + constant
    frozen_logit_diff = float(x_final @ w) + constant
    model_logit_diff = float(
        comp.capture.final_logits_last[target_id] - comp.capture.final_logits_last[distractor_id]
    )

    return {
        "norm_class": norm_class,
        "norm_kind": "rmsnorm" if is_rms else "layernorm",
        "frozen_scale": frozen_scale,
        "constant": constant,
        "embed_score": embed_score,
        "attn_scores": attn_scores,
        "mlp_scores": mlp_scores,
        "ledger_total": ledger_total,
        "frozen_logit_diff": frozen_logit_diff,
        "model_logit_diff": model_logit_diff,
        "ledger_vs_frozen_abs_err": abs(ledger_total - frozen_logit_diff),
        "frozen_vs_model_abs_err": abs(frozen_logit_diff - model_logit_diff),
        "scoring_vector": w,  # consumed by the ablation comparison, not serialized
    }


def cumulative_curve(dla: dict[str, Any]) -> list[float]:
    """Cumulative ledger by depth: [embed, +block0, +block1, ...] + constant."""
    total = dla["embed_score"] + dla["constant"]
    curve = [total]
    for a, m in zip(dla["attn_scores"], dla["mlp_scores"]):
        total += a + m
        curve.append(total)
    return curve


def component_rows(example_id: str, category: str, dla: dict[str, Any]) -> list[dict[str, Any]]:
    """Long-form ledger rows for one example."""
    denom = abs(dla["frozen_logit_diff"]) or 1.0
    rows = [
        {
            "example_id": example_id,
            "category": category,
            "component": "embed",
            "layer": "",
            "score": round(dla["embed_score"], 5),
            "frac_of_logit_diff": round(dla["embed_score"] / denom, 4),
        }
    ]
    for kind_key, kind in (("attn_scores", "attn"), ("mlp_scores", "mlp")):
        for layer, score in enumerate(dla[kind_key]):
            rows.append(
                {
                    "example_id": example_id,
                    "category": category,
                    "component": kind,
                    "layer": layer,
                    "score": round(score, 5),
                    "frac_of_logit_diff": round(score / denom, 4),
                }
            )
    return rows


def top_components(dla: dict[str, Any], n: int) -> list[tuple[str, int, float]]:
    """The n components with the largest |score|, as (kind, layer, score)."""
    scored = [("attn", k, s) for k, s in enumerate(dla["attn_scores"])]
    scored += [("mlp", k, s) for k, s in enumerate(dla["mlp_scores"])]
    return sorted(scored, key=lambda t: abs(t[2]), reverse=True)[:n]


# ---------------------------------------------------------------------------
# Extension: attribution vs direct-path ablation
# ---------------------------------------------------------------------------


def run_ablation_comparison(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    comp_anatomy: bench.ComponentAnatomy,
    example: AnswerPairExample,
    target_id: int,
    distractor_id: int,
    dla: dict[str, Any],
    n_top: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Zero-ablate top-|attribution| components plus matched controls.

    Direct-path ablation (final position only) is deliberately commensurable
    with the DLA score: it removes exactly the contribution the ledger
    counted. ``causal_effect`` = base logit diff - ablated logit diff, so
    positive = the component was pushing toward the target.
    """
    ranked = sorted(
        [("attn", k, s) for k, s in enumerate(dla["attn_scores"])]
        + [("mlp", k, s) for k, s in enumerate(dla["mlp_scores"])],
        key=lambda t: abs(t[2]),
        reverse=True,
    )
    chosen: list[tuple[str, int, float, str]] = [(k, l, s, "top") for k, l, s in ranked[:n_top]]
    rest = ranked[n_top:]
    if rest:
        k, l, s = rest[rng.randrange(len(rest))]
        chosen.append((k, l, s, "random_control"))
        k, l, s = ranked[-1]
        if (k, l) != (chosen[-1][0], chosen[-1][1]):
            chosen.append((k, l, s, "low_attribution_control"))

    rows = []
    for kind, layer, score, why in chosen:
        logits = bench.run_with_component_ablation(bundle, example.prompt, comp_anatomy, kind, layer)
        ablated_diff = float(logits[target_id] - logits[distractor_id])
        rows.append(
            {
                "example_id": example.example_id,
                "category": example.category,
                "component": kind,
                "layer": layer,
                "selection": why,
                "attribution_score": round(score, 5),
                "base_logit_diff": round(dla["model_logit_diff"], 5),
                "ablated_logit_diff": round(ablated_diff, 5),
                "causal_effect": round(dla["model_logit_diff"] - ablated_diff, 5),
            }
        )
    return rows


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation, midrank ties, no scipy dependency."""
    if len(xs) < 3:
        return None

    def midranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            mid = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = mid
            i = j + 1
        return ranks

    rx, ry = midranks(xs), midranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def category_color(category: str) -> str:
    return bench.CATEGORY_COLORS.get(category, "tab:gray") if hasattr(bench, "CATEGORY_COLORS") else "tab:gray"


def plot_contribution_by_layer(
    ctx: bench.RunContext, per_example: list[dict[str, Any]], n_layers: int
) -> None:
    """2x2 panel, one per category: mean attn/mlp contribution per layer."""
    import numpy as np

    import matplotlib.pyplot as plt

    cats = [c for c in CATEGORIES if any(r["category"] == c for r in per_example)]
    if not cats:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    for ax, cat in zip(axes.flat, cats):
        rows = [r for r in per_example if r["category"] == cat]
        attn = np.mean([r["dla"]["attn_scores"] for r in rows], axis=0)
        mlp = np.mean([r["dla"]["mlp_scores"] for r in rows], axis=0)
        x = np.arange(n_layers)
        ax.bar(x - 0.2, attn, width=0.4, label="attn", color="tab:blue")
        ax.bar(x + 0.2, mlp, width=0.4, label="mlp", color="tab:orange")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{cat} (n={len(rows)})")
        ax.set_xlabel("layer")
        ax.set_ylabel("logit-diff contribution")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(cats):]:
        ax.set_visible(False)
    fig.suptitle("Mean component contribution to the answer direction, by layer")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "contribution_by_layer.png",
                      "Per-category mean attention/MLP contribution per layer.")


def plot_cumulative(ctx: bench.RunContext, per_example: list[dict[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.0, 5.5))
    plotted = False
    for cat in CATEGORIES:
        rows = [r for r in per_example if r["category"] == cat]
        if not rows:
            continue
        plotted = True
        color = category_color(cat)
        for r in rows:
            ax.plot(range(len(r["curve"])), r["curve"], color=color, alpha=0.3, linewidth=0.8)
        depth = min(len(r["curve"]) for r in rows)
        mean = [statistics.fmean(r["curve"][d] for r in rows) for d in range(depth)]
        ax.plot(range(depth), mean, color=color, linewidth=2.5, label=f"{cat} (n={len(rows)})")
    if not plotted:
        bench.close_figure(fig)
        return
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("depth (0 = embeddings + constants, k = after block k-1)")
    ax.set_ylabel("cumulative logit difference (frozen norm)")
    ax.set_title("Cumulative DLA ledger: where the logit difference is assembled")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "cumulative_logit_diff.png",
                      "Cumulative component ledger per example with category means.")


def plot_attribution_vs_ablation(ctx: bench.RunContext, ablation_rows: list[dict[str, Any]]) -> float | None:
    if not ablation_rows:
        return None
    xs = [r["attribution_score"] for r in ablation_rows]
    ys = [r["causal_effect"] for r in ablation_rows]
    rho = spearman_rho(xs, ys)
    fig, ax = bench.new_figure(figsize=(7.5, 6.5))
    markers = {"top": "o", "random_control": "s", "low_attribution_control": "^"}
    for why, marker in markers.items():
        sel = [r for r in ablation_rows if r["selection"] == why]
        if not sel:
            continue
        ax.scatter(
            [r["attribution_score"] for r in sel],
            [r["causal_effect"] for r in sel],
            marker=marker,
            s=42,
            alpha=0.8,
            label=f"{why} (n={len(sel)})",
        )
    lim = max(max(abs(v) for v in xs), max(abs(v) for v in ys)) * 1.1 or 1.0
    ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=0.8, linestyle="--", label="attribution = effect")
    ax.set_xlabel("attribution score (frozen-norm logit diff)")
    ax.set_ylabel("direct-path ablation effect (logit diff change)")
    title = "Attribution vs causal effect"
    if rho is not None:
        title += f" (Spearman rho = {rho:.3f}, n = {len(xs)})"
    ax.set_title(title)
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "attribution_vs_ablation.png",
                      "Does the ledger predict what direct-path ablation does?")
    return rho


def plot_dla_vs_lens(
    ctx: bench.RunContext,
    example: AnswerPairExample,
    curve: list[float],
    traj: bench.LensTrajectory,
) -> None:
    """Showcase: cumulative frozen-norm ledger vs the moving-basis logit lens."""
    if traj.logit_target is None or traj.logit_distractor is None:
        return
    lens_diff = [t - d for t, d in zip(traj.logit_target, traj.logit_distractor)]
    fig, ax = bench.new_figure(figsize=(9.0, 5.5))
    ax.plot(range(len(curve)), curve, linewidth=2.5, label="cumulative DLA (final norm frozen)")
    ax.plot(range(len(lens_diff)), lens_diff, linewidth=2.0, linestyle="--",
            label="logit lens (norm recomputed per depth)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("depth")
    ax.set_ylabel("logit(target) - logit(distractor)")
    ax.set_title(f"Two readouts of the same stream: {example.example_id}")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, f"dla_vs_lens_{bench.sanitize_tag(example.example_id)}.png",
                      "Frozen-norm cumulative ledger vs per-depth logit lens for the showcase example.")


# ---------------------------------------------------------------------------
# Summary, claims
# ---------------------------------------------------------------------------


def aggregate_by_category(per_example: list[dict[str, Any]], n_layers: int) -> list[dict[str, Any]]:
    out = []
    for cat in CATEGORIES:
        rows = [r for r in per_example if r["category"] == cat]
        if not rows:
            continue
        attn_total = statistics.fmean(sum(r["dla"]["attn_scores"]) for r in rows)
        mlp_total = statistics.fmean(sum(r["dla"]["mlp_scores"]) for r in rows)
        tops = [top_components(r["dla"], 1)[0] for r in rows]
        out.append(
            {
                "category": cat,
                "n_examples": len(rows),
                "mean_model_logit_diff": round(statistics.fmean(r["dla"]["model_logit_diff"] for r in rows), 4),
                "mean_embed_score": round(statistics.fmean(r["dla"]["embed_score"] for r in rows), 4),
                "mean_attn_total": round(attn_total, 4),
                "mean_mlp_total": round(mlp_total, 4),
                "mean_top_component_share": round(
                    statistics.fmean(
                        abs(t[2]) / max(abs(r["dla"]["frozen_logit_diff"]), 1e-9)
                        for t, r in zip(tops, rows)
                    ),
                    4,
                ),
                "top_component_modal_kind": statistics.mode(t[0] for t in tops),
                "median_top_component_layer": statistics.median(t[1] for t in tops),
                "mean_min_component_score": round(
                    statistics.fmean(min(r["dla"]["attn_scores"] + r["dla"]["mlp_scores"]) for r in rows), 4
                ),
                "mean_frozen_vs_model_abs_err": round(
                    statistics.fmean(r["dla"]["frozen_vs_model_abs_err"] for r in rows), 4
                ),
            }
        )
    return out


def draft_claims(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    cat_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    rho: float | None,
) -> list[dict[str, str]]:
    run_name = ctx.run_dir.name
    L = bundle.anatomy.n_layers
    by_cat = {r["category"]: r for r in cat_rows}
    claims: list[dict[str, str]] = []

    fact = by_cat.get("fact")
    conflict = by_cat.get("conflict")
    if fact:
        claims.append(
            {
                "id": f"{LAB_ID}-C1",
                "tag": "ATTR",
                "text": (
                    f"On {fact['n_examples']} factual prompts, {bundle.anatomy.model_id}'s answer logit "
                    f"difference was assembled mainly by {fact['top_component_modal_kind']} components "
                    f"(modal top component), median top-component layer {fact['median_top_component_layer']}/{L}; "
                    f"mean totals: attn {fact['mean_attn_total']}, mlp {fact['mean_mlp_total']}, "
                    f"embeddings {fact['mean_embed_score']}."
                ),
                "artifact": f"runs/{run_name}/tables/layer_component_summary.csv",
                "falsifier": (
                    "Re-scoring with a tuned lens / non-frozen normalization, or a different "
                    "distractor of the same type, moves the dominant component kind or depth materially."
                ),
            }
        )
    if conflict and fact:
        claims.append(
            {
                "id": f"{LAB_ID}-C2",
                "tag": "ATTR",
                "text": (
                    f"Conflict prompts (in-context override) showed components actively pushing the stored "
                    f"fact against the in-context answer: mean most-negative component score "
                    f"{conflict['mean_min_component_score']} vs {fact['mean_min_component_score']} on facts. "
                    "The ledger records opposing entries, not just a smaller total."
                ),
                "artifact": f"runs/{run_name}/tables/component_contributions.csv",
                "falsifier": (
                    "Length-matched non-conflict controls show equally negative minimum component scores, "
                    "i.e. the negative entries are a prompt-length artifact rather than a conflict signature."
                ),
            }
        )
    if ablation_rows and rho is not None:
        top_rows = [r for r in ablation_rows if r["selection"] == "top"]
        agree = sum(1 for r in top_rows if (r["attribution_score"] > 0) == (r["causal_effect"] > 0))
        claims.append(
            {
                "id": f"{LAB_ID}-C3",
                "tag": "CAUSAL",
                "text": (
                    f"Direct-path zero-ablation of {len(ablation_rows)} components (final position only) "
                    f"tracked attribution with Spearman rho = {rho:.3f}; {agree}/{len(top_rows)} top-attributed "
                    "components had a causal effect of the same sign. Scope: this tests only the direct path "
                    "the ledger counts, not indirect effects through later attention."
                ),
                "artifact": f"runs/{run_name}/tables/ablation_results.csv",
                "falsifier": (
                    "Full-sequence or mean-ablation produces a materially different ranking, showing the "
                    "direct-path restriction was doing the work."
                ),
            }
        )
    return claims


def render_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    per_example: list[dict[str, Any]],
    cat_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    rho: float | None,
    dropped: int,
    claims: list[dict[str, str]],
) -> str:
    args = ctx.args
    a = bundle.anatomy
    worst_ledger = max((r["dla"]["ledger_vs_frozen_abs_err"] for r in per_example), default=0.0)
    worst_model_gap = max((r["dla"]["frozen_vs_model_abs_err"] for r in per_example), default=0.0)
    lines = [
        "# Lab 2 run summary: direct logit attribution",
        "",
        "## Run identity",
        "",
        f"- model: `{a.model_id}` ({a.n_layers} blocks, d_model {a.d_model})",
        f"- dtype: `{args.dtype}` | quantization: `{args.quantization}` | ablate-top: {args.ablate_top}",
        f"- examples: {len(per_example)} kept, {dropped} dropped at the single-token gate",
        "- evidence level: `ATTR` (ledger) + narrow `CAUSAL` (direct-path ablation extension)",
        "- self-checks: hook parity, lens self-check, component anatomy probe, decomposition check",
        "",
        "## 1. What behavior was studied?",
        "",
        "Next-token answer preference between a target and a matched distractor on four prompt",
        "families (facts, relations, grammar, in-context conflict).",
        "",
        "## 2. What internal object was measured?",
        "",
        "The exact tensor each component (embeddings, every attention block, every MLP block)",
        "adds to the final position's residual stream, scored against the answer direction",
        "`unembed[target] - unembed[distractor]` under the frozen-final-norm linearization.",
        f"Hook points were verified, not assumed: see `diagnostics/component_anatomy.json`",
        f"(this model: attn={a.architecture} resolved at runtime).",
        "",
        "## 3. What intervention or control was used?",
        "",
        f"Direct-path zero-ablation of the top-{args.ablate_top} attributed components per example,",
        "plus one random and one low-attribution control component, at the final position only —",
        "deliberately commensurable with what the ledger counts.",
        "",
        "## 4. Headline numbers",
        "",
    ]
    lines.append(
        "| category | n | mean logit diff | embed | attn total | mlp total | top kind | top layer (median) | min comp |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for r in cat_rows:
        lines.append(
            f"| {r['category']} | {r['n_examples']} | {r['mean_model_logit_diff']} | {r['mean_embed_score']} | "
            f"{r['mean_attn_total']} | {r['mean_mlp_total']} | {r['top_component_modal_kind']} | "
            f"{r['median_top_component_layer']} | {r['mean_min_component_score']} |"
        )
    lines += [
        "",
        "## 5. Does the ledger balance?",
        "",
        f"- worst |ledger total - frozen logit diff| across examples: {worst_ledger:.5f}",
        f"  (pure bookkeeping; bounded by the bench's decomposition check)",
        f"- worst |frozen fp32 logit diff - model {args.dtype} logit diff|: {worst_model_gap:.5f}",
        "  (the fp32-reimplementation gap; reported, not enforced)",
        "",
        "## 6. Attribution vs causation",
        "",
    ]
    if ablation_rows:
        lines.append(
            f"{len(ablation_rows)} direct-path ablations. Spearman rho(attribution, causal effect) = "
            f"{'n/a' if rho is None else f'{rho:.3f}'}. See `plots/attribution_vs_ablation.png` and "
            "`tables/ablation_results.csv` — the off-diagonal points are the lab's payload: "
            "arithmetically correct ledger entries whose causal weight differs."
        )
    else:
        lines.append("Ablation comparison skipped (`--ablate-top 0`).")
    lines += [
        "",
        "## 7. What claim is supported, and what would kill it?",
        "",
        "Drafted claims (edit before committing to the ledger):",
        "",
    ]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    lines += [
        "",
        "### Caveats students must carry forward",
        "",
        "- Frozen-norm scores are exact only in aggregate; per-component credit assumes the",
        "  norm scale would not change without that component (it would).",
        "- Direct-path ablation does not measure indirect effects through later layers;",
        "  Lab 5 (patching) owns that question.",
        "- Negative entries in the conflict family are attribution, not yet a located",
        "  'fact recall' mechanism.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    args = ctx.args
    examples = build_prompt_set(args)
    print(f"[lab2] prompt set: {len(examples)} examples before tokenization gate")

    kept, dropped = tokenize_and_filter(ctx, bundle, examples)
    if not kept:
        raise RuntimeError("Every example was dropped at the single-token gate; nothing to run.")
    print(f"[lab2] running {len(kept)} examples ({dropped} dropped)")

    manifest = [
        {
            "example_id": ex.example_id,
            "category": ex.category,
            "prompt": ex.prompt,
            "target": bench.visible_token(ex.target),
            "target_id": t_id,
            "distractor": bench.visible_token(ex.distractor),
            "distractor_id": d_id,
            "note": ex.note,
        }
        for ex, t_id, d_id in kept
    ]
    manifest_path = ctx.path("tables", "prompt_manifest.csv")
    bench.write_csv(manifest_path, manifest)
    ctx.register_artifact(manifest_path, "table", "Every kept example with ids and token ids.")

    # Instrument verification, in dependency order, before any science.
    first_prompt = kept[0][0].prompt
    bench.run_hook_parity_check(ctx, bundle, first_prompt)
    comp_anatomy = bench.resolve_component_anatomy(
        ctx, bundle, first_prompt, rel_tolerance=args.dla_tolerance
    )
    first_comp = bench.run_with_component_cache(bundle, first_prompt, comp_anatomy)
    bench.run_lens_self_check(ctx, bundle, first_comp.capture)
    bench.run_decomposition_check(ctx, bundle, first_comp, rel_tolerance=args.dla_tolerance)

    per_example: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    showcase: tuple[AnswerPairExample, list[float], bench.LensTrajectory] | None = None

    for i, (ex, t_id, d_id) in enumerate(kept):
        comp = first_comp if ex.prompt == first_prompt else bench.run_with_component_cache(
            bundle, ex.prompt, comp_anatomy
        )
        dla = compute_direct_logit_attribution(bundle, comp, t_id, d_id)
        curve = cumulative_curve(dla)
        contributions.extend(component_rows(ex.example_id, ex.category, dla))

        traj = bench.compute_lens_trajectory(
            bundle, comp.capture, target_id=t_id, distractor_id=d_id, topk=args.topk
        )
        bench.dump_example_state(
            ctx, bundle, ex.example_id, comp.capture, traj, target=ex.target, distractor=ex.distractor
        )

        if args.ablate_top > 0:
            rng = random.Random(args.seed * 100003 + i)
            ablation_rows.extend(
                run_ablation_comparison(ctx, bundle, comp_anatomy, ex, t_id, d_id, dla, args.ablate_top, rng)
            )

        if showcase is None and (
            ex.example_id == args.showcase or (args.showcase is None and ex.category == "conflict")
        ):
            showcase = (ex, curve, traj)

        top = top_components(dla, 1)[0]
        print(
            f"[lab2] [{i + 1}/{len(kept)}] {ex.example_id} logit_diff={dla['model_logit_diff']:+.3f} "
            f"top={top[0]}@{top[1]} ({top[2]:+.3f}) embed={dla['embed_score']:+.3f}"
        )
        per_example.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "dla": dla,
                "curve": curve,
                "example": ex,
            }
        )

    # Tables.
    contrib_path = ctx.path("tables", "component_contributions.csv")
    bench.write_csv(contrib_path, contributions)
    ctx.register_artifact(contrib_path, "table", "Long-form ledger: every component's score for every example.")

    example_rows = []
    for r in per_example:
        dla = r["dla"]
        tops = top_components(dla, 3)
        example_rows.append(
            {
                "example_id": r["example_id"],
                "category": r["category"],
                "model_logit_diff": round(dla["model_logit_diff"], 4),
                "frozen_logit_diff": round(dla["frozen_logit_diff"], 4),
                "constant": round(dla["constant"], 4),
                "embed_score": round(dla["embed_score"], 4),
                "attn_total": round(sum(dla["attn_scores"]), 4),
                "mlp_total": round(sum(dla["mlp_scores"]), 4),
                "top1": f"{tops[0][0]}@{tops[0][1]}:{tops[0][2]:+.3f}",
                "top2": f"{tops[1][0]}@{tops[1][1]}:{tops[1][2]:+.3f}" if len(tops) > 1 else "",
                "top3": f"{tops[2][0]}@{tops[2][1]}:{tops[2][2]:+.3f}" if len(tops) > 2 else "",
                "ledger_vs_frozen_abs_err": round(dla["ledger_vs_frozen_abs_err"], 6),
                "frozen_vs_model_abs_err": round(dla["frozen_vs_model_abs_err"], 6),
            }
        )
    ex_path = ctx.path("tables", "example_summary.csv")
    bench.write_csv(ex_path, example_rows)
    ctx.register_artifact(ex_path, "table", "Per-example ledger totals, top components, and balance checks.")

    n_layers = bundle.anatomy.n_layers
    layer_rows = []
    for cat in CATEGORIES:
        rows = [r for r in per_example if r["category"] == cat]
        if not rows:
            continue
        for layer in range(n_layers):
            layer_rows.append(
                {
                    "category": cat,
                    "layer": layer,
                    "mean_attn_score": round(statistics.fmean(r["dla"]["attn_scores"][layer] for r in rows), 5),
                    "mean_mlp_score": round(statistics.fmean(r["dla"]["mlp_scores"][layer] for r in rows), 5),
                    "n_examples": len(rows),
                }
            )
    layer_path = ctx.path("tables", "layer_component_summary.csv")
    bench.write_csv(layer_path, layer_rows)
    ctx.register_artifact(layer_path, "table", "Per-category mean attn/MLP contribution by layer.")

    if ablation_rows:
        abl_path = ctx.path("tables", "ablation_results.csv")
        bench.write_csv(abl_path, ablation_rows)
        ctx.register_artifact(abl_path, "table", "Attribution vs direct-path ablation effect for every ablated component.")

    cat_rows = aggregate_by_category(per_example, n_layers)
    cat_path = ctx.path("tables", "category_summary.csv")
    bench.write_csv(cat_path, cat_rows)
    ctx.register_artifact(cat_path, "table", "Category-level headline numbers.")

    # Plots.
    rho: float | None = None
    if not args.no_plots:
        plot_contribution_by_layer(ctx, per_example, n_layers)
        plot_cumulative(ctx, per_example)
        rho = plot_attribution_vs_ablation(ctx, ablation_rows)
        if showcase is not None:
            plot_dla_vs_lens(ctx, showcase[0], showcase[1], showcase[2])
        elif args.showcase is not None:
            print(
                f"[lab2] WARNING: --showcase {args.showcase!r} did not match any kept example id; "
                "no DLA-vs-lens plot was produced."
            )
    elif ablation_rows:
        rho = spearman_rho(
            [r["attribution_score"] for r in ablation_rows],
            [r["causal_effect"] for r in ablation_rows],
        )

    metrics = {
        "n_examples": len(per_example),
        "n_dropped": dropped,
        "spearman_attribution_vs_ablation": rho,
        "categories": {r["category"]: r for r in cat_rows},
        "worst_ledger_vs_frozen_abs_err": max(
            (r["dla"]["ledger_vs_frozen_abs_err"] for r in per_example), default=0.0
        ),
        "worst_frozen_vs_model_abs_err": max(
            (r["dla"]["frozen_vs_model_abs_err"] for r in per_example), default=0.0
        ),
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 2 metrics.")

    claims = draft_claims(ctx, bundle, cat_rows, ablation_rows, rho)
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    summary = render_summary(ctx, bundle, per_example, cat_rows, ablation_rows, rho, dropped, claims)
    summary_path = ctx.path("run_summary.md")
    bench.write_text(summary_path, summary)
    ctx.register_artifact(summary_path, "summary", "The seven standard questions answered with this run's numbers.")
    print(f"[lab2] wrote run_summary.md and {len(claims)} drafted ledger claims")
