"""Lab 2: Direct logit attribution and component accounting.

The experiment: decompose a final logit difference into the contributions of
the embedding stream and every attention/MLP block, scored against an answer
direction ``unembed[target] - unembed[distractor]``.

The pedagogical core lives in :func:`compute_direct_logit_attribution`,
deliberately written out in this file rather than hidden in the bench: the
course runs raw HF weights (no LayerNorm folding), so the final norm must be
*frozen* (data-dependent statistics taken from the actual forward pass) before
component scores mean anything. Students should see exactly where the
approximation enters: it is exact for the aggregate ledger, an approximation
when read per-component. This is the same "instrument first" discipline as Lab 1.

The bench owns the instrument: verified contribution hook points
(``resolve_component_anatomy`` via per-block residual-delta probe, not name
heuristics), the capture (``run_with_component_cache``), the decomposition
self-check (``run_decomposition_check``), and final-position component ablation
(``run_with_component_ablation``). The lab owns the question and the
interpretation of mismatches.

Evidence levels: attribution (ATTR) for the ledger itself; the ablation
extension produces narrow CAUSAL evidence scoped to final-position writes.
Mismatches between frozen attribution and live effect are the central teaching
material ("the ledger is arithmetically correct yet can still mislead about
responsibility").
"""

from __future__ import annotations

import csv
import dataclasses
import json
import pathlib
import random
import statistics
from collections import Counter
from typing import Any

import interp_bench as bench

LAB_ID = "L02"

CATEGORIES = ("fact", "relation", "grammar", "conflict")
SIGN_EPS = 1e-8
SMALL_DENOM_EPS = 1e-6
TOP_COMPONENTS_PER_EXAMPLE = 8


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
    # Stronger conflict to pop "in-context vs memorized" attribution vs effect
    AnswerPairExample("conflict_france_berlin", "conflict",
                      "According to the map in the book, the capital of France is Berlin. The capital of France is",
                      " Berlin", " Paris",
                      "In-context fact directly contradicts stored knowledge; good for seeing opposing component signs."),
    # More facts for robustness
    AnswerPairExample("cap_australia", "fact", "The capital of Australia is", " Canberra", " Sydney"),
    AnswerPairExample("cap_canada", "fact", "The capital of Canada is", " Ottawa", " Toronto"),
    AnswerPairExample("fact_moon", "fact", "The largest planet in the Solar System is", " Jupiter", " Saturn"),
    AnswerPairExample("fact_heart", "fact", "The organ that pumps blood is the", " heart", " liver"),
    # More relations/grammar
    AnswerPairExample("opp_north", "relation", "The opposite of north is", " south", " east"),
    AnswerPairExample("opp_light", "relation", "The opposite of light is", " dark", " heavy"),
    AnswerPairExample("past_go", "grammar", "The past tense of go is", " went", " goes"),
    AnswerPairExample("plural_child", "grammar", "The plural of child is", " children", " kids"),
    # More conflict
    AnswerPairExample("conflict_day", "conflict",
                      "In the story the day after Monday is Friday. The day after Monday is",
                      " Friday", " Tuesday"),
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
    """Load a custom Lab 2 prompt set from JSON or CSV.

    JSON remains the canonical format, but CSV is convenient for students who
    are building paired prompt families in a spreadsheet. Only the dataclass
    fields are accepted: a stray column is usually a typo in the microscope's
    coordinate system, not harmless decoration.
    """
    allowed = {f.name for f in dataclasses.fields(AnswerPairExample)}
    examples: list[AnswerPairExample] = []
    if not path.exists():
        raise ValueError(
            f"Could not read prompt set {str(path)!r}. --prompt-set must be one of "
            "small | medium | full, or a path to a .json/.csv prompt file."
        )

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                item = {k: (v if v is not None else "") for k, v in row.items() if k}
                extra = set(item) - allowed
                if extra:
                    raise ValueError(f"Prompt CSV row {i} has unknown columns: {sorted(extra)}")
                examples.append(AnswerPairExample(**{k: item.get(k, "") for k in allowed}))
        return examples

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read prompt set {str(path)!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse prompt JSON at {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("Custom prompt file must be a JSON list of objects.")
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
        "answer_direction_norm": float(direction.norm()),
        "scoring_vector_norm": float(w.norm()),
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


def component_score_mass(dla: dict[str, Any], *, include_embed_and_constant: bool = False) -> float:
    """Total absolute writer mass, used for bounded concentration metrics."""
    mass = sum(abs(s) for s in dla["attn_scores"] + dla["mlp_scores"])
    if include_embed_and_constant:
        mass += abs(float(dla["embed_score"])) + abs(float(dla["constant"]))
    return mass


def signed_token(score: float) -> str:
    """Human-readable direction label for a component score."""
    if score > SIGN_EPS:
        return "toward_target"
    if score < -SIGN_EPS:
        return "toward_distractor"
    return "neutral"


def safe_signed_fraction(score: float, denom: float) -> float | str:
    """Signed fraction of a net total, blank when the denominator is tiny.

    This avoids the classic DLA footgun: a conflict prompt can have nearly
    zero net logit difference because large positive and negative ledger rows
    cancel. Dividing by that signed net makes harmless rows look galactic.
    """
    if abs(denom) < SMALL_DENOM_EPS:
        return ""
    return round(score / denom, 4)


def component_rows(example_id: str, category: str, dla: dict[str, Any]) -> list[dict[str, Any]]:
    """Long-form ledger rows for one example.

    The legacy ``frac_of_logit_diff`` field is retained for compatibility, but
    the safer teaching quantity is ``abs_mass_share``: how much of the total
    absolute ledger mass this row accounts for. Net fractions are a cancellation
    trap on conflict prompts.
    """
    frozen = float(dla["frozen_logit_diff"])
    denom_for_legacy = abs(frozen) or 1.0
    gross_mass = component_score_mass(dla, include_embed_and_constant=True) or 1.0

    def row(component: str, layer: int | str, score: float, stream_depth_after: int | str = "") -> dict[str, Any]:
        return {
            "example_id": example_id,
            "category": category,
            "component": component,
            "layer": layer,
            "stream_depth_after": stream_depth_after,
            "score": round(score, 5),
            "abs_score": round(abs(score), 5),
            "pushes": signed_token(score),
            "frac_of_logit_diff": round(score / denom_for_legacy, 4),
            "signed_fraction_of_frozen_diff": safe_signed_fraction(score, frozen),
            "abs_mass_share": round(abs(score) / gross_mass, 4),
        }

    rows = [
        row("embed", "", float(dla["embed_score"]), 0),
        row("constant", "", float(dla["constant"]), "all"),
    ]
    for kind_key, kind in (("attn_scores", "attn"), ("mlp_scores", "mlp")):
        for layer, score in enumerate(dla[kind_key]):
            rows.append(row(kind, layer, float(score), layer + 1))
    return rows


def top_components(dla: dict[str, Any], n: int) -> list[tuple[str, int, float]]:
    """The n components with the largest |score|, as (kind, layer, score)."""
    scored = [("attn", k, s) for k, s in enumerate(dla["attn_scores"])]
    scored += [("mlp", k, s) for k, s in enumerate(dla["mlp_scores"])]
    return sorted(scored, key=lambda t: abs(t[2]), reverse=True)[:n]


def top_component_rows(
    example_id: str,
    category: str,
    dla: dict[str, Any],
    n: int = TOP_COMPONENTS_PER_EXAMPLE,
) -> list[dict[str, Any]]:
    """Top-|attribution| components with bounded shares and signs."""
    mass = component_score_mass(dla) or 1.0
    rows = []
    for rank, (kind, layer, score) in enumerate(top_components(dla, n), start=1):
        rows.append(
            {
                "example_id": example_id,
                "category": category,
                "rank": rank,
                "component": kind,
                "layer": layer,
                "stream_depth_after": layer + 1,
                "score": round(score, 5),
                "abs_score": round(abs(score), 5),
                "pushes": signed_token(float(score)),
                "abs_writer_mass_share": round(abs(float(score)) / mass, 4),
            }
        )
    return rows


def block_ledger_rows(
    example_id: str,
    category: str,
    dla: dict[str, Any],
    curve: list[float],
) -> list[dict[str, Any]]:
    """One row per block: attention write, MLP write, block total, cumulative total."""
    rows = []
    for layer, (attn, mlp) in enumerate(zip(dla["attn_scores"], dla["mlp_scores"])):
        total = float(attn + mlp)
        rows.append(
            {
                "example_id": example_id,
                "category": category,
                "layer": layer,
                "stream_depth_after": layer + 1,
                "attn_score": round(float(attn), 5),
                "mlp_score": round(float(mlp), 5),
                "block_total": round(total, 5),
                "block_abs_mass": round(abs(float(attn)) + abs(float(mlp)), 5),
                "dominant_subcomponent": "attn" if abs(float(attn)) >= abs(float(mlp)) else "mlp",
                "block_pushes": signed_token(total),
                "cumulative_after_block": round(float(curve[layer + 1]), 5),
            }
        )
    return rows


def block_reconstruction_rows(
    example_id: str,
    category: str,
    comp: bench.ComponentCapture,
) -> list[dict[str, Any]]:
    """Per-example proof that captured attn+MLP writes rebuild residual deltas."""
    rows = []
    for layer in range(comp.attn_contrib.shape[0]):
        delta = comp.capture.streams[layer + 1, -1] - comp.capture.streams[layer, -1]
        recon = comp.attn_contrib[layer] + comp.mlp_contrib[layer]
        abs_err = float((recon - delta).norm())
        delta_norm = max(float(delta.norm()), 1e-9)
        rows.append(
            {
                "example_id": example_id,
                "category": category,
                "layer": layer,
                "stream_depth_before": layer,
                "stream_depth_after": layer + 1,
                "delta_norm": round(delta_norm, 6),
                "captured_sum_norm": round(float(recon.norm()), 6),
                "abs_err": round(abs_err, 8),
                "rel_err": round(abs_err / delta_norm, 8),
            }
        )
    return rows


def answer_behavior_row(
    bundle: bench.ModelBundle,
    example: AnswerPairExample,
    target_id: int,
    distractor_id: int,
    logits: Any,
) -> dict[str, Any]:
    """Behavioral readout for the target/distractor pair before attribution."""
    import torch

    logits = logits.detach().to("cpu", torch.float32)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    target_logit = float(logits[target_id])
    distractor_logit = float(logits[distractor_id])
    target_rank = int((logits > logits[target_id]).sum().item()) + 1
    distractor_rank = int((logits > logits[distractor_id]).sum().item()) + 1
    top_logit, top_id = torch.max(logits, dim=0)
    top_id_int = int(top_id.item())
    return {
        "example_id": example.example_id,
        "category": example.category,
        "target": bench.visible_token(example.target),
        "target_id": target_id,
        "distractor": bench.visible_token(example.distractor),
        "distractor_id": distractor_id,
        "target_logit": round(target_logit, 5),
        "distractor_logit": round(distractor_logit, 5),
        "model_logit_diff": round(target_logit - distractor_logit, 5),
        "target_logprob": round(float(log_probs[target_id]), 5),
        "distractor_logprob": round(float(log_probs[distractor_id]), 5),
        "target_prob": round(float(probs[target_id]), 8),
        "distractor_prob": round(float(probs[distractor_id]), 8),
        "target_rank": target_rank,
        "distractor_rank": distractor_rank,
        "prefers_target_over_distractor": target_logit > distractor_logit,
        "top_token_id": top_id_int,
        "top_token": bench.visible_token(bundle.tokenizer.decode([top_id_int])),
        "top_logit": round(float(top_logit), 5),
    }


def balance_row(example_id: str, category: str, dla: dict[str, Any]) -> dict[str, Any]:
    """Per-example ledger balance and final-norm metadata."""
    gross = component_score_mass(dla, include_embed_and_constant=True)
    frozen = float(dla["frozen_logit_diff"])
    cancellation = abs(frozen) / gross if gross else 0.0
    return {
        "example_id": example_id,
        "category": category,
        "norm_class": dla["norm_class"],
        "norm_kind": dla["norm_kind"],
        "frozen_scale": round(float(dla["frozen_scale"]), 8),
        "answer_direction_norm": round(float(dla["answer_direction_norm"]), 6),
        "scoring_vector_norm": round(float(dla["scoring_vector_norm"]), 6),
        "gross_ledger_mass": round(gross, 6),
        "net_to_gross_ratio": round(cancellation, 6),
        "ledger_total": round(float(dla["ledger_total"]), 6),
        "frozen_logit_diff": round(frozen, 6),
        "model_logit_diff": round(float(dla["model_logit_diff"]), 6),
        "ledger_vs_frozen_abs_err": round(float(dla["ledger_vs_frozen_abs_err"]), 8),
        "frozen_vs_model_abs_err": round(float(dla["frozen_vs_model_abs_err"]), 8),
    }


# ---------------------------------------------------------------------------
# Extension: attribution vs final-position ablation
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
    """Zero-ablate top-|attribution| components plus matched random and low-attribution controls.

    The intervention is final-position scoped only (earlier-token writes left
    intact). This makes it a narrow causal test of roughly the same *direct*
    channel the ledger scored at the prediction position. It is deliberately
    *not* the same as subtracting the frozen-norm attribution row: later layers
    at the final position still run on the modified stream, and the final norm
    is live. The recorded ``effect_minus_attribution`` (and same-sign rate) is
    therefore the first concrete look at the "ledger balances but may still
    mislead about responsibility" lesson. Full indirect effects through earlier
    positions are left to Lab 5 (patching) and Lab 3 (head ablation scope).
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
        effect = float(dla["model_logit_diff"] - ablated_diff)
        mismatch = effect - float(score)
        rows.append(
            {
                "example_id": example.example_id,
                "category": example.category,
                "component": kind,
                "layer": layer,
                "stream_depth_after": layer + 1,
                "selection": why,
                "ablation_scope": "final_position_live_downstream",
                "attribution_score": round(float(score), 5),
                "base_logit_diff": round(float(dla["model_logit_diff"]), 5),
                "ablated_logit_diff": round(ablated_diff, 5),
                "causal_effect": round(effect, 5),
                "effect_minus_attribution": round(mismatch, 5),
                "same_sign": signed_token(float(score)) == signed_token(effect),
                "attribution_pushes": signed_token(float(score)),
                "effect_pushes": signed_token(effect),
                "abs_mismatch": round(abs(mismatch), 5),
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



def summarize_ablation_by_selection(ablation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the final-position ablation extension by selection type."""
    out = []
    for selection in ("top", "random_control", "low_attribution_control"):
        rows = [r for r in ablation_rows if r["selection"] == selection]
        if not rows:
            continue
        out.append(
            {
                "selection": selection,
                "n": len(rows),
                "mean_abs_attribution": round(statistics.fmean(abs(float(r["attribution_score"])) for r in rows), 5),
                "mean_abs_causal_effect": round(statistics.fmean(abs(float(r["causal_effect"])) for r in rows), 5),
                "mean_effect_minus_attribution": round(
                    statistics.fmean(float(r["effect_minus_attribution"]) for r in rows), 5
                ),
                "median_abs_mismatch": round(statistics.median(float(r["abs_mismatch"]) for r in rows), 5),
                "same_sign_rate": round(statistics.fmean(1.0 if r["same_sign"] else 0.0 for r in rows), 4),
            }
        )
    return out

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


def plot_signed_component_heatmap(
    ctx: bench.RunContext,
    per_example: list[dict[str, Any]],
    n_layers: int,
) -> None:
    """Example x layer heatmap of total direct contribution per block."""
    if not per_example:
        return
    import numpy as np

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    rows = sorted(
        per_example,
        key=lambda r: (
            CATEGORIES.index(r["category"]) if r["category"] in CATEGORIES else len(CATEGORIES),
            r["example_id"],
        ),
    )
    data = np.array(
        [
            [
                float(r["dla"]["attn_scores"][layer] + r["dla"]["mlp_scores"][layer])
                for layer in range(n_layers)
            ]
            for r in rows
        ],
        dtype=float,
    )
    lim = float(np.nanpercentile(np.abs(data), 95)) if data.size else 1.0
    lim = lim or 1.0
    fig_height = max(5.5, min(10.0, 0.35 * len(rows) + 1.8))
    fig, ax = bench.new_figure(figsize=(10.5, fig_height))
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", norm=norm)
    ax.set_xticks(range(0, n_layers, max(1, n_layers // 8)))
    ax.set_xlabel("layer")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['category'][:3]}:{r['example_id']}" for r in rows], fontsize=8)
    ax.set_title("Signed direct contribution by example and layer (attn + MLP)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("logit-diff contribution")

    previous = rows[0]["category"]
    for i, row in enumerate(rows[1:], start=1):
        if row["category"] != previous:
            ax.axhline(i - 0.5, color="black", linewidth=0.7)
            previous = row["category"]
    fig.tight_layout()
    bench.save_figure(
        ctx,
        fig,
        "signed_component_heatmap.png",
        "Per-example heatmap of signed block contribution to the answer direction.",
    )


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



def plot_category_ledger_composition(ctx: bench.RunContext, per_example: list[dict[str, Any]]) -> None:
    """Gross vs net ledger composition by category.

    This plot makes cancellation visible: a category can have a small mean net
    logit difference while large positive and negative components fight under
    the hood.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    cats = [cat for cat in CATEGORIES if any(r["category"] == cat for r in per_example)]
    if not cats:
        return
    fields = ["positive_mass", "negative_mass_abs", "net_abs"]
    data = []
    for cat in cats:
        rows = [r for r in per_example if r["category"] == cat]
        pos_vals, neg_vals, net_vals = [], [], []
        for r in rows:
            vals = [float(r["dla"]["embed_score"]), float(r["dla"]["constant"])]
            vals += [float(v) for v in r["dla"]["attn_scores"] + r["dla"]["mlp_scores"]]
            pos_vals.append(sum(v for v in vals if v > 0))
            neg_vals.append(abs(sum(v for v in vals if v < 0)))
            net_vals.append(abs(float(r["dla"]["frozen_logit_diff"])))
        data.append([statistics.fmean(pos_vals), statistics.fmean(neg_vals), statistics.fmean(net_vals)])

    x = np.arange(len(cats))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for j, field in enumerate(fields):
        ax.bar(x + (j - 1) * width, [row[j] for row in data], width=width, label=field)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("mean absolute logit-diff units")
    ax.set_title("Gross ledger mass vs net answer preference")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    bench.save_figure(
        ctx,
        fig,
        "category_ledger_composition.png",
        "Positive mass, negative mass, and net answer preference by category.",
    )


def plot_balance_errors(ctx: bench.RunContext, balance_rows: list[dict[str, Any]]) -> None:
    if not balance_rows:
        return
    fig, ax = bench.new_figure(figsize=(9.0, 5.4))
    xs = list(range(len(balance_rows)))
    ledger_err = [float(r["ledger_vs_frozen_abs_err"]) for r in balance_rows]
    model_err = [float(r["frozen_vs_model_abs_err"]) for r in balance_rows]
    ax.scatter(xs, ledger_err, marker="o", label="ledger vs frozen")
    ax.scatter(xs, model_err, marker="s", label="frozen fp32 vs model dtype")
    ax.set_yscale("symlog", linthresh=1e-7)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r["example_id"]) for r in balance_rows], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("absolute error (symlog)")
    ax.set_title("Ledger balance checks by example")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "ledger_balance_errors.png", "Per-example ledger balance and dtype-gap errors.")


def plot_top_component_lollipop(ctx: bench.RunContext, top_rows: list[dict[str, Any]]) -> None:
    """Top component per example, sorted by signed score."""
    if not top_rows:
        return
    rows = [r for r in top_rows if int(r["rank"]) == 1]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r["score"]))
    fig_height = max(5.0, min(10.5, 0.36 * len(rows) + 1.5))
    fig, ax = bench.new_figure(figsize=(9.2, fig_height))
    y = list(range(len(rows)))
    for yi, row in zip(y, rows):
        score = float(row["score"])
        ax.plot([0, score], [yi, yi], color=category_color(str(row["category"])), alpha=0.6, linewidth=2.0)
        ax.scatter(score, yi, s=58, color=category_color(str(row["category"])), edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['example_id']} {r['component']}@{r['layer']}" for r in rows],
        fontsize=8,
    )
    ax.set_xlabel("top component attribution score")
    ax.set_title("Largest component per example: sign and size")
    bench.save_figure(ctx, fig, "top_component_by_example.png", "Top-|attribution| component for every example.")


def plot_showcase_waterfall(
    ctx: bench.RunContext,
    example: AnswerPairExample,
    dla: dict[str, Any],
    n: int = 12,
) -> None:
    """A compact waterfall for one example's largest ledger entries."""
    components: list[tuple[str, float]] = [("embed", float(dla["embed_score"])), ("constant", float(dla["constant"]))]
    components += [(f"A{layer}", float(score)) for layer, score in enumerate(dla["attn_scores"])]
    components += [(f"M{layer}", float(score)) for layer, score in enumerate(dla["mlp_scores"])]
    chosen = sorted(components, key=lambda kv: abs(kv[1]), reverse=True)[:n]
    chosen = sorted(chosen, key=lambda kv: kv[1])
    fig, ax = bench.new_figure(figsize=(8.8, max(5.0, 0.38 * len(chosen) + 1.5)))
    ys = list(range(len(chosen)))
    for y, (name, score) in zip(ys, chosen):
        ax.plot([0, score], [y, y], color="tab:blue" if score >= 0 else "tab:red", alpha=0.65, linewidth=2.4)
        ax.scatter(score, y, color="tab:blue" if score >= 0 else "tab:red", edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([name for name, _ in chosen], fontsize=8)
    ax.set_xlabel("logit-diff contribution")
    ax.set_title(f"Largest ledger entries for {example.example_id}")
    bench.save_figure(
        ctx,
        fig,
        f"ledger_waterfall_{bench.sanitize_tag(example.example_id)}.png",
        "Largest positive and negative ledger entries for the showcase example.",
    )


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
    ax.set_ylabel("final-position ablation effect (logit diff change)")
    title = "Attribution vs causal effect"
    if rho is not None:
        title += f" (Spearman rho = {rho:.3f}, n = {len(xs)})"
    ax.set_title(title)
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "attribution_vs_ablation.png",
                      "Does the ledger predict what final-position ablation does?")
    return rho


def plot_ablation_mismatches(ctx: bench.RunContext, ablation_rows: list[dict[str, Any]]) -> None:
    """Label the largest attribution-vs-ablation disagreements."""
    if not ablation_rows:
        return
    rows = sorted(
        ablation_rows,
        key=lambda r: abs(float(r["causal_effect"]) - float(r["attribution_score"])),
        reverse=True,
    )[: min(12, len(ablation_rows))]
    fig_height = max(5.0, 0.45 * len(rows) + 1.5)
    fig, ax = bench.new_figure(figsize=(9.5, fig_height))
    y_positions = list(range(len(rows)))
    for y, row in zip(y_positions, reversed(rows)):
        attr = float(row["attribution_score"])
        effect = float(row["causal_effect"])
        color = category_color(str(row["category"]))
        ax.plot([attr, effect], [y, y], color=color, alpha=0.55, linewidth=2.0)
        ax.scatter(attr, y, marker="o", color=color, edgecolor="black", linewidth=0.5, s=52)
        ax.scatter(effect, y, marker="s", color=color, edgecolor="black", linewidth=0.5, s=52)
    labels = [
        f"{r['example_id']} {r['component']}@{r['layer']} ({r['selection']})"
        for r in reversed(rows)
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("logit-diff units: circle = attribution, square = ablation effect")
    ax.set_title("Largest attribution-vs-ablation mismatches")
    ax.grid(True, axis="x", alpha=0.3)
    bench.save_figure(
        ctx,
        fig,
        "ablation_mismatch_examples.png",
        "Largest final-position attribution vs ablation-effect disagreements, labeled by component.",
    )


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
        attn_abs_mass = statistics.fmean(sum(abs(float(s)) for s in r["dla"]["attn_scores"]) for r in rows)
        mlp_abs_mass = statistics.fmean(sum(abs(float(s)) for s in r["dla"]["mlp_scores"]) for r in rows)
        tops = [top_components(r["dla"], 1)[0] for r in rows]
        top_abs_scores = [abs(t[2]) for t in tops]
        top_mass_shares = []
        cancellation = []
        positive_mass = []
        negative_mass = []
        for top, row in zip(tops, rows):
            mass = component_score_mass(row["dla"])
            gross = component_score_mass(row["dla"], include_embed_and_constant=True)
            vals = [float(row["dla"]["embed_score"]), float(row["dla"]["constant"])]
            vals += [float(v) for v in row["dla"]["attn_scores"] + row["dla"]["mlp_scores"]]
            top_mass_shares.append(abs(top[2]) / mass if mass else 0.0)
            cancellation.append(abs(float(row["dla"]["frozen_logit_diff"])) / gross if gross else 0.0)
            positive_mass.append(sum(v for v in vals if v > 0))
            negative_mass.append(abs(sum(v for v in vals if v < 0)))
        top_kind_counts = Counter(t[0] for t in tops)
        modal_kind = top_kind_counts.most_common(1)[0][0]
        out.append(
            {
                "category": cat,
                "n_examples": len(rows),
                "mean_model_logit_diff": round(statistics.fmean(r["dla"]["model_logit_diff"] for r in rows), 4),
                "mean_frozen_logit_diff": round(statistics.fmean(r["dla"]["frozen_logit_diff"] for r in rows), 4),
                "mean_embed_score": round(statistics.fmean(r["dla"]["embed_score"] for r in rows), 4),
                "mean_constant": round(statistics.fmean(r["dla"]["constant"] for r in rows), 4),
                "mean_attn_total": round(attn_total, 4),
                "mean_mlp_total": round(mlp_total, 4),
                "mean_attn_abs_mass": round(attn_abs_mass, 4),
                "mean_mlp_abs_mass": round(mlp_abs_mass, 4),
                "mean_positive_mass": round(statistics.fmean(positive_mass), 4),
                "mean_negative_mass_abs": round(statistics.fmean(negative_mass), 4),
                "mean_net_to_gross_ratio": round(statistics.fmean(cancellation), 4),
                "mean_top_component_abs_score": round(statistics.fmean(top_abs_scores), 4),
                "mean_top_component_mass_share": round(statistics.fmean(top_mass_shares), 4),
                "top_component_modal_kind": modal_kind,
                "top_component_kind_counts": ";".join(f"{k}:{v}" for k, v in sorted(top_kind_counts.items())),
                "median_top_component_layer": statistics.median(t[1] for t in tops),
                "mean_min_component_score": round(
                    statistics.fmean(min(r["dla"]["attn_scores"] + r["dla"]["mlp_scores"]) for r in rows), 4
                ),
                "mean_frozen_vs_model_abs_err": round(
                    statistics.fmean(r["dla"]["frozen_vs_model_abs_err"] for r in rows), 6
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
                "artifact": f"runs/{run_name}/tables/category_summary.csv",
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
                    f"Final-position zero-ablation of {len(ablation_rows)} components tracked attribution "
                    f"with Spearman rho = {rho:.3f}; {agree}/{len(top_rows)} top-attributed components "
                    "had a causal effect of the same sign. Scope: earlier-token writes are left intact, "
                    "but later final-position layers do run on the modified stream, so mismatches are evidence "
                    "that the ledger is not a causal map."
                ),
                "artifact": f"runs/{run_name}/tables/ablation_results.csv",
                "falsifier": (
                    "Full-sequence or mean-ablation produces a materially different ranking, or the controls "
                    "match the top components, showing the final-position test was not isolating real contributors."
                ),
            }
        )
    return claims


def render_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    comp_anatomy: bench.ComponentAnatomy,
    per_example: list[dict[str, Any]],
    cat_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    ablation_summary_rows: list[dict[str, Any]],
    rho: float | None,
    dropped: int,
    claims: list[dict[str, str]],
) -> str:
    args = ctx.args
    a = bundle.anatomy
    worst_ledger = max((r["dla"]["ledger_vs_frozen_abs_err"] for r in per_example), default=0.0)
    worst_model_gap = max((r["dla"]["frozen_vs_model_abs_err"] for r in per_example), default=0.0)
    strongest = None
    if per_example:
        strongest = max(
            per_example,
            key=lambda r: abs(top_components(r["dla"], 1)[0][2]) if top_components(r["dla"], 1) else 0.0,
        )
    lines = [
        "# Lab 2 run summary: direct logit attribution",
        "",
        "## Run identity",
        "",
        f"- model: `{a.model_id}` ({a.n_layers} blocks, d_model {a.d_model})",
        f"- dtype: `{args.dtype}` | quantization: `{args.quantization}` | ablate-top: {args.ablate_top}",
        f"- examples: {len(per_example)} kept, {dropped} dropped at the single-token gate",
        "- evidence level: `ATTR` for the ledger + narrow `CAUSAL` for final-position ablation",
        "- self-checks: hook parity, lens self-check, component anatomy probe, decomposition check",
        "",
        "## 1. What behavior was studied?",
        "",
        "Next-token answer preference between a target and a matched distractor on four prompt",
        "families: facts, relations, grammar, and in-context conflict.",
        "",
        "## 2. What internal object was measured?",
        "",
        "The tensor each component adds to the final position's residual stream:",
        "embeddings, every attention-block contribution, and every MLP-block contribution.",
        "Each row is scored against `unembed[target] - unembed[distractor]` under the",
        "frozen-final-norm linearization.",
        f"Hook points were verified, not assumed: see `diagnostics/component_anatomy.json`",
        (
            f"(this model: attn={comp_anatomy.attn_source}, mlp={comp_anatomy.mlp_source}; "
            f"max block reconstruction rel err={comp_anatomy.max_block_recon_rel_err:.3g})."
        ),
        "",
        "## 3. What intervention or control was used?",
        "",
        f"Final-position zero-ablation of the top-{args.ablate_top} attributed components per example,",
        "plus one random and one low-attribution control component. This is deliberately narrower",
        "than Lab 5 patching: earlier-token writes are left intact, but later final-position layers",
        "do run on the changed stream. So ablation disagreement is a feature, not a bug.",
        "",
        "## 4. Headline numbers",
        "",
    ]
    lines.append(
        "| category | n | mean logit diff | embed | attn total | mlp total | pos mass | neg mass | net/gross | top kind | top layer | top abs |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for r in cat_rows:
        lines.append(
            f"| {r['category']} | {r['n_examples']} | {r['mean_model_logit_diff']} | {r['mean_embed_score']} | "
            f"{r['mean_attn_total']} | {r['mean_mlp_total']} | {r['mean_positive_mass']} | "
            f"{r['mean_negative_mass_abs']} | {r['mean_net_to_gross_ratio']} | "
            f"{r['top_component_modal_kind']} | {r['median_top_component_layer']} | "
            f"{r['mean_top_component_abs_score']} |"
        )
    if strongest is not None:
        kind, layer, score = top_components(strongest["dla"], 1)[0]
        lines += [
            "",
            (
                f"Strongest single ledger row in this run: `{strongest['example_id']}` "
                f"{kind}@{layer} with score {score:+.4f}. See `tables/top_components.csv`."
            ),
        ]
    lines += [
        "",
        "## 5. Does the ledger balance?",
        "",
        f"- worst |ledger total - frozen logit diff| across examples: {worst_ledger:.6f}",
        "  (bookkeeping error; bounded by the component decomposition check)",
        f"- worst |frozen fp32 logit diff - model {args.dtype} logit diff|: {worst_model_gap:.6f}",
        "  (fp32 reimplementation versus the actual model dtype; reported, not enforced)",
        "- see `tables/dla_balance.csv`, `diagnostics/block_reconstruction_by_example.csv`,",
        "  and `plots/ledger_balance_errors.png` for the audit trail.",
        "",
        "## 6. Attribution vs causation",
        "",
    ]
    if ablation_rows:
        same_sign = statistics.fmean(1.0 if r["same_sign"] else 0.0 for r in ablation_rows)
        top_rows = [r for r in ablation_rows if r["selection"] == "top"]
        sign_mismatches = sum(1 for r in top_rows if not r["same_sign"])
        lines.append(
            f"{len(ablation_rows)} final-position ablations. Spearman rho(attribution, causal effect) = "
            f"{'n/a' if rho is None else f'{rho:.3f}'}, with same-sign rate {same_sign:.2f}. "
            f"Of the top-attributed components, {sign_mismatches}/{len(top_rows)} had the opposite sign in the live ablation. "
            "See `tables/ablation_results.csv`, `tables/ablation_summary_by_selection.csv`, "
            "`plots/attribution_vs_ablation.png`, and `plots/ablation_mismatch_examples.png`. "
            "The important column is `effect_minus_attribution`: it measures how much the live "
            "intervention departed from the frozen ledger row. Mismatches are the pedagogical payload."
        )
        if ablation_summary_rows:
            lines += ["", "Selection summary:", "", "| selection | n | mean abs attr | mean abs effect | same-sign | median abs mismatch |"]
            lines.append("|---|---:|---:|---:|---:|---:|")
            for r in ablation_summary_rows:
                lines.append(
                    f"| {r['selection']} | {r['n']} | {r['mean_abs_attribution']} | "
                    f"{r['mean_abs_causal_effect']} | {r['same_sign_rate']} | {r['median_abs_mismatch']} |"
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
        "### What is not supported",
        "",
        "- A large DLA row does not prove that the component stores the fact or rule.",
        "- A balanced ledger does not prove causal responsibility; it proves arithmetic accounting",
        "  under a chosen linearization.",
        "- Final-position ablation does not measure full-sequence indirect effects. Lab 5 owns that.",
        "",
        "### Caveats students must carry forward",
        "",
        "- Frozen-norm scores are exact only in aggregate; per-component credit assumes the",
        "  norm scale would not change without that component.",
        "- Conflict prompts can have low net/gross ratios: positive and negative rows are fighting,",
        "  so signed fractions of the net logit difference can be unstable.",
        "- Component hook points are architecture-specific. The anatomy probe is part of the result.",
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
    bench.write_csv_with_context(ctx, manifest_path, manifest)
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
    behavior_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    block_rows_all: list[dict[str, Any]] = []
    top_rows_all: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    showcase: tuple[AnswerPairExample, dict[str, Any], list[float], bench.LensTrajectory] | None = None

    for i, (ex, t_id, d_id) in enumerate(kept):
        comp = first_comp if ex.prompt == first_prompt else bench.run_with_component_cache(
            bundle, ex.prompt, comp_anatomy
        )
        dla = compute_direct_logit_attribution(bundle, comp, t_id, d_id)
        curve = cumulative_curve(dla)
        contributions.extend(component_rows(ex.example_id, ex.category, dla))
        behavior_rows.append(answer_behavior_row(bundle, ex, t_id, d_id, comp.capture.final_logits_last))
        balance_rows.append(balance_row(ex.example_id, ex.category, dla))
        block_rows_all.extend(block_ledger_rows(ex.example_id, ex.category, dla, curve))
        top_rows_all.extend(top_component_rows(ex.example_id, ex.category, dla))
        reconstruction_rows.extend(block_reconstruction_rows(ex.example_id, ex.category, comp))

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
            showcase = (ex, dla, curve, traj)

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
                "behavior": behavior_rows[-1],
            }
        )

    # Tables.
    contrib_path = ctx.path("tables", "component_contributions.csv")
    bench.write_csv_with_context(ctx, contrib_path, contributions)
    ctx.register_artifact(contrib_path, "table", "Long-form ledger: every component's score for every example.")

    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, contributions)
    ctx.register_artifact(results_path, "results", "Alias of component_contributions.csv for the standard run contract.")

    behavior_path = ctx.path("tables", "baseline_behavior.csv")
    bench.write_csv_with_context(ctx, behavior_path, behavior_rows)
    ctx.register_artifact(behavior_path, "table", "Target/distractor logits, ranks, probabilities, and top token before attribution.")

    balance_path = ctx.path("tables", "dla_balance.csv")
    bench.write_csv_with_context(ctx, balance_path, balance_rows)
    ctx.register_artifact(balance_path, "table", "Per-example frozen-norm metadata and ledger balance checks.")

    top_path = ctx.path("tables", "top_components.csv")
    bench.write_csv_with_context(ctx, top_path, top_rows_all)
    ctx.register_artifact(top_path, "table", "Top-|attribution| components per example with signs and bounded mass shares.")

    block_path = ctx.path("tables", "block_ledger.csv")
    bench.write_csv_with_context(ctx, block_path, block_rows_all)
    ctx.register_artifact(block_path, "table", "One row per block: attention, MLP, block total, and cumulative ledger value.")

    recon_path = ctx.path("diagnostics", "block_reconstruction_by_example.csv")
    bench.write_csv(recon_path, reconstruction_rows)
    ctx.register_artifact(recon_path, "diagnostic", "Per-example proof that captured attn+MLP writes reconstruct block residual deltas.")

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
                "gross_ledger_mass": round(component_score_mass(dla, include_embed_and_constant=True), 4),
                "net_to_gross_ratio": round(
                    abs(float(dla["frozen_logit_diff"])) / (component_score_mass(dla, include_embed_and_constant=True) or 1.0),
                    4,
                ),
                "top1": f"{tops[0][0]}@{tops[0][1]}:{tops[0][2]:+.3f}",
                "top2": f"{tops[1][0]}@{tops[1][1]}:{tops[1][2]:+.3f}" if len(tops) > 1 else "",
                "top3": f"{tops[2][0]}@{tops[2][1]}:{tops[2][2]:+.3f}" if len(tops) > 2 else "",
                "ledger_vs_frozen_abs_err": round(dla["ledger_vs_frozen_abs_err"], 6),
                "frozen_vs_model_abs_err": round(dla["frozen_vs_model_abs_err"], 6),
            }
        )
    ex_path = ctx.path("tables", "example_summary.csv")
    bench.write_csv_with_context(ctx, ex_path, example_rows)
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
                    "mean_block_total": round(
                        statistics.fmean(
                            r["dla"]["attn_scores"][layer] + r["dla"]["mlp_scores"][layer] for r in rows
                        ),
                        5,
                    ),
                    "mean_abs_block_mass": round(
                        statistics.fmean(
                            abs(float(r["dla"]["attn_scores"][layer])) + abs(float(r["dla"]["mlp_scores"][layer]))
                            for r in rows
                        ),
                        5,
                    ),
                    "n_examples": len(rows),
                }
            )
    layer_path = ctx.path("tables", "layer_component_summary.csv")
    bench.write_csv_with_context(ctx, layer_path, layer_rows)
    ctx.register_artifact(layer_path, "table", "Per-category mean attn/MLP contribution by layer.")

    ablation_summary_rows = summarize_ablation_by_selection(ablation_rows)
    if ablation_rows:
        abl_path = ctx.path("tables", "ablation_results.csv")
        bench.write_csv_with_context(ctx, abl_path, ablation_rows)
        ctx.register_artifact(abl_path, "table", "Attribution vs final-position ablation effect for every ablated component.")
        abl_summary_path = ctx.path("tables", "ablation_summary_by_selection.csv")
        bench.write_csv_with_context(ctx, abl_summary_path, ablation_summary_rows)
        ctx.register_artifact(abl_summary_path, "table", "Ablation extension aggregated by top/random/low-attribution selection.")

    cat_rows = aggregate_by_category(per_example, n_layers)
    cat_path = ctx.path("tables", "category_summary.csv")
    bench.write_csv_with_context(ctx, cat_path, cat_rows)
    ctx.register_artifact(cat_path, "table", "Category-level headline numbers.")

    # Plots.
    rho: float | None = None
    if not args.no_plots:
        plot_contribution_by_layer(ctx, per_example, n_layers)
        plot_signed_component_heatmap(ctx, per_example, n_layers)
        plot_cumulative(ctx, per_example)
        plot_category_ledger_composition(ctx, per_example)
        plot_balance_errors(ctx, balance_rows)
        plot_top_component_lollipop(ctx, top_rows_all)
        rho = plot_attribution_vs_ablation(ctx, ablation_rows)
        plot_ablation_mismatches(ctx, ablation_rows)
        if showcase is not None:
            plot_dla_vs_lens(ctx, showcase[0], showcase[2], showcase[3])
            plot_showcase_waterfall(ctx, showcase[0], showcase[1])
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
        "n_prefers_target_over_distractor": sum(1 for r in behavior_rows if r["prefers_target_over_distractor"]),
        "spearman_attribution_vs_ablation": rho,
        "ablation_summary_by_selection": ablation_summary_rows,
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
    summary = render_summary(
        ctx,
        bundle,
        comp_anatomy,
        per_example,
        cat_rows,
        ablation_rows,
        ablation_summary_rows,
        rho,
        dropped,
        claims,
    )
    summary_path = ctx.path("run_summary.md")
    bench.write_text(summary_path, summary)
    ctx.register_artifact(summary_path, "summary", "The seven standard questions answered with this run's numbers.")
    print(f"[lab2] wrote run_summary.md and {len(claims)} drafted ledger claims")
