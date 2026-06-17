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
import math
import pathlib
import random
import re
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
    # Optional metadata used by the richer Lab 2 plots.  Built-in examples
    # infer these fields automatically, while custom CSV/JSON prompt sets can
    # supply them directly.  Accepted CSV aliases include relation/family/task
    # for relation_family and contrast/difficulty/distractor_type for
    # contrast_type.
    relation_family: str = ""
    contrast_type: str = ""
    source: str = "built_in"


PROMPT_FIELD_ALIASES = {
    "relation": "relation_family",
    "family": "relation_family",
    "task": "relation_family",
    "relation_type": "relation_family",
    "semantic_family": "relation_family",
    "difficulty": "contrast_type",
    "contrast": "contrast_type",
    "pair_type": "contrast_type",
    "comparison_type": "contrast_type",
    "distractor_type": "contrast_type",
}


def _tag_value(text: str, *keys: str) -> str:
    """Extract simple key=value tags from notes, forgiving separators."""
    for key in keys:
        m = re.search(rf"(?:^|[;|,\s]){re.escape(key)}\s*=\s*([^;|,]+)", text or "", flags=re.I)
        if m:
            return sanitize_label(m.group(1))
    return ""


def sanitize_label(value: str) -> str:
    """Stable, CSV/plot-safe metadata label."""
    value = str(value or "").strip().lower()
    value = value.replace("/", "_").replace(" ", "_").replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def normalize_prompt_record(item: dict[str, Any], where: str) -> AnswerPairExample:
    """Accept a custom prompt row while preserving a strict schema.

    The lab is intentionally fussy about prompt fields, because an extra column
    often means a misspelled target/distractor.  A few common metadata aliases
    are normalized rather than rejected so relation-family CSVs can be dropped
    in without spreadsheet acrobatics.
    """
    allowed = {f.name for f in dataclasses.fields(AnswerPairExample)}
    cleaned = {str(k): (v if v is not None else "") for k, v in item.items() if k is not None}
    for src, dst in PROMPT_FIELD_ALIASES.items():
        if src in cleaned and dst not in cleaned:
            cleaned[dst] = cleaned[src]
    ignored_aliases = set(PROMPT_FIELD_ALIASES)
    extra = set(cleaned) - allowed - ignored_aliases
    if extra:
        raise ValueError(f"Prompt {where} has unknown fields: {sorted(extra)}")
    row = {field: cleaned.get(field, "") for field in allowed}
    if not row.get("source"):
        row["source"] = "custom"
    return AnswerPairExample(**row)


def infer_relation_family(ex: AnswerPairExample) -> str:
    explicit = sanitize_label(ex.relation_family) or _tag_value(ex.note, "relation", "family", "task")
    if explicit:
        return explicit
    eid = ex.example_id.lower()
    prompt = ex.prompt.lower()
    if ex.category == "conflict" or eid.startswith("conflict"):
        if "capital" in prompt:
            return "conflict_capital"
        if "opposite" in prompt:
            return "conflict_antonym"
        return "context_override"
    if eid.startswith("cap_") or "capital of" in prompt or eid in {"eiffel", "water_oxygen", "jupiter", "fact_moon", "fact_heart"}:
        return "factual_recall"
    if eid.startswith("opp_") or "opposite of" in prompt:
        return "antonym"
    if eid.startswith("plural_") or "plural of" in prompt:
        return "morphology_plural"
    if eid.startswith("past_") or "past tense" in prompt:
        return "morphology_past"
    return sanitize_label(ex.category) or "unknown"


def infer_contrast_type(ex: AnswerPairExample) -> str:
    explicit = sanitize_label(ex.contrast_type) or _tag_value(
        ex.note, "contrast", "difficulty", "pair_type", "distractor_type"
    )
    if explicit:
        return explicit
    eid = ex.example_id.lower()
    if eid.endswith("_hard") or "same_class" in eid or "sameclass" in eid:
        return "same_class_hard"
    if eid.endswith("_easy") or "cross_class" in eid or "crossclass" in eid:
        return "cross_class_easy"
    if ex.category == "conflict":
        return "context_vs_prior"
    if ex.category == "grammar":
        return "inflected_vs_wrong_form"
    if ex.category == "relation":
        return "answer_vs_axis_foil"
    return "target_vs_type_foil"


def example_meta(ex: AnswerPairExample) -> dict[str, str]:
    return {
        "relation_family": infer_relation_family(ex),
        "contrast_type": infer_contrast_type(ex),
        "source": sanitize_label(ex.source) or "built_in",
    }


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
    examples: list[AnswerPairExample] = []
    if not path.exists():
        raise ValueError(
            f"Could not read prompt set {str(path)!r}. --prompt-set must be one of "
            "small | medium | full, or a path to a .json/.csv prompt file."
        )

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                examples.append(normalize_prompt_record(row, f"CSV row {i}"))
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
        examples.append(normalize_prompt_record(item, f"JSON item {i}"))
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
                **example_meta(ex),
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


def component_rows(
    example_id: str,
    category: str,
    dla: dict[str, Any],
    relation_family: str = "",
    contrast_type: str = "",
    source: str = "",
) -> list[dict[str, Any]]:
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
            "relation_family": relation_family,
            "contrast_type": contrast_type,
            "source": source,
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
    relation_family: str = "",
    contrast_type: str = "",
    source: str = "",
) -> list[dict[str, Any]]:
    """Top-|attribution| components with bounded shares and signs."""
    mass = component_score_mass(dla) or 1.0
    rows = []
    for rank, (kind, layer, score) in enumerate(top_components(dla, n), start=1):
        rows.append(
            {
                "example_id": example_id,
                "category": category,
                "relation_family": relation_family,
                "contrast_type": contrast_type,
                "source": source,
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
    relation_family: str = "",
    contrast_type: str = "",
    source: str = "",
) -> list[dict[str, Any]]:
    """One row per block: attention write, MLP write, block total, cumulative total."""
    rows = []
    for layer, (attn, mlp) in enumerate(zip(dla["attn_scores"], dla["mlp_scores"])):
        total = float(attn + mlp)
        rows.append(
            {
                "example_id": example_id,
                "category": category,
                "relation_family": relation_family,
                "contrast_type": contrast_type,
                "source": source,
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
        **example_meta(example),
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
# Higher-level Lab 2 summaries for richer visualizations
# ---------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Small dependency-free percentile with linear interpolation."""
    if not values:
        return float("nan")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def median_iqr(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    return (percentile(values, 0.5), percentile(values, 0.25), percentile(values, 0.75))


def finite_floats(values: list[Any]) -> list[float]:
    """Convert a nested-ish numeric list to finite floats, skipping NaN/inf."""
    out: list[float] = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def abs_percentile(values: list[Any], q: float, default: float = 1.0) -> float:
    vals = [abs(v) for v in finite_floats(values)]
    if not vals:
        return default
    return percentile(vals, q) or default


def layer_phase(layer: int, n_layers: int) -> str:
    """Coarse phase name for a 0-indexed transformer block."""
    if n_layers <= 0:
        return "unknown"
    frac_after = (layer + 1) / n_layers
    if frac_after <= 1 / 3:
        return "early"
    if frac_after <= 2 / 3:
        return "middle"
    if frac_after < 0.9:
        return "late"
    return "final_tail"


def _phase_order_key(phase: str) -> int:
    return {"embed_const": 0, "early": 1, "middle": 2, "late": 3, "final_tail": 4}.get(phase, 99)


def _component_values_by_phase(dla: dict[str, Any], n_layers: int) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = {("embed_const", "all"): [float(dla["embed_score"]), float(dla["constant"])]}
    for layer, score in enumerate(dla["attn_scores"]):
        phase = layer_phase(layer, n_layers)
        out.setdefault((phase, "attn"), []).append(float(score))
        out.setdefault((phase, "all"), []).append(float(score))
    for layer, score in enumerate(dla["mlp_scores"]):
        phase = layer_phase(layer, n_layers)
        out.setdefault((phase, "mlp"), []).append(float(score))
        out.setdefault((phase, "all"), []).append(float(score))
    return out


def summarize_values(values: list[float]) -> dict[str, float]:
    pos = sum(v for v in values if v > 0)
    neg_abs = abs(sum(v for v in values if v < 0))
    net = sum(values)
    gross = pos + neg_abs
    return {
        "net_score": net,
        "positive_mass": pos,
        "negative_mass_abs": neg_abs,
        "gross_mass": gross,
        "net_to_gross_ratio": abs(net) / gross if gross else 0.0,
        "n_components": len(values),
    }


def phase_ledger_summary(per_example: list[dict[str, Any]], n_layers: int) -> list[dict[str, Any]]:
    """Category x phase x component-kind summary, including gross cancellation."""
    records: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for r in per_example:
        for (phase, component_kind), values in _component_values_by_phase(r["dla"], n_layers).items():
            s = summarize_values(values)
            key = (str(r["category"]), phase, component_kind)
            groups.setdefault(key, []).append(s)
    for (category, phase, component_kind), vals in sorted(groups.items(), key=lambda kv: (CATEGORIES.index(kv[0][0]) if kv[0][0] in CATEGORIES else 99, _phase_order_key(kv[0][1]), kv[0][2])):
        records.append(
            {
                "category": category,
                "phase": phase,
                "component_kind": component_kind,
                "n_examples": len(vals),
                "mean_net_score": round(statistics.fmean(v["net_score"] for v in vals), 5),
                "median_net_score": round(statistics.median(v["net_score"] for v in vals), 5),
                "mean_positive_mass": round(statistics.fmean(v["positive_mass"] for v in vals), 5),
                "mean_negative_mass_abs": round(statistics.fmean(v["negative_mass_abs"] for v in vals), 5),
                "mean_gross_mass": round(statistics.fmean(v["gross_mass"] for v in vals), 5),
                "mean_net_to_gross_ratio": round(statistics.fmean(v["net_to_gross_ratio"] for v in vals), 5),
                "mean_n_components": round(statistics.fmean(v["n_components"] for v in vals), 2),
            }
        )
    return records


def relation_family_summary(per_example: list[dict[str, Any]], n_layers: int) -> list[dict[str, Any]]:
    """Summarize built-in and custom relation families on the same ledger axes."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in per_example:
        key = (str(r.get("category", "")), str(r.get("relation_family", "unknown")), str(r.get("contrast_type", "")))
        groups.setdefault(key, []).append(r)
    rows: list[dict[str, Any]] = []
    for (category, family, contrast), items in sorted(groups.items()):
        top = [top_components(r["dla"], 1)[0] for r in items]
        vals = []
        for r in items:
            vals.extend([float(r["dla"]["embed_score"]), float(r["dla"]["constant"])])
            vals.extend(float(v) for v in r["dla"]["attn_scores"] + r["dla"]["mlp_scores"])
        s = summarize_values(vals)
        tops_by_phase = Counter(layer_phase(t[1], n_layers) for t in top)
        top_kind_counts = Counter(t[0] for t in top)
        rows.append(
            {
                "category": category,
                "relation_family": family,
                "contrast_type": contrast,
                "n_examples": len(items),
                "mean_model_logit_diff": round(statistics.fmean(r["dla"]["model_logit_diff"] for r in items), 5),
                "mean_frozen_logit_diff": round(statistics.fmean(r["dla"]["frozen_logit_diff"] for r in items), 5),
                "mean_embed_score": round(statistics.fmean(r["dla"]["embed_score"] for r in items), 5),
                "mean_attn_total": round(statistics.fmean(sum(r["dla"]["attn_scores"]) for r in items), 5),
                "mean_mlp_total": round(statistics.fmean(sum(r["dla"]["mlp_scores"]) for r in items), 5),
                "gross_ledger_mass": round(s["gross_mass"] / max(1, len(items)), 5),
                "net_to_gross_ratio": round(s["net_to_gross_ratio"], 5),
                "top_component_modal_kind": top_kind_counts.most_common(1)[0][0] if top_kind_counts else "",
                "top_component_kind_counts": ";".join(f"{k}:{v}" for k, v in sorted(top_kind_counts.items())),
                "median_top_component_layer": statistics.median(t[1] for t in top) if top else "",
                "top_component_modal_phase": tops_by_phase.most_common(1)[0][0] if tops_by_phase else "",
                "top_component_phase_counts": ";".join(f"{k}:{v}" for k, v in sorted(tops_by_phase.items(), key=lambda kv: _phase_order_key(kv[0]))),
            }
        )
    return rows


def ablation_mismatch_summary(ablation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate attribution-vs-live-effect mismatch by category/component/selection."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in ablation_rows:
        key = (str(r["category"]), str(r["component"]), str(r["selection"]))
        groups.setdefault(key, []).append(r)
    out: list[dict[str, Any]] = []
    for (category, component, selection), rows in sorted(groups.items()):
        out.append(
            {
                "category": category,
                "component": component,
                "selection": selection,
                "n": len(rows),
                "mean_attribution": round(statistics.fmean(float(r["attribution_score"]) for r in rows), 5),
                "mean_causal_effect": round(statistics.fmean(float(r["causal_effect"]) for r in rows), 5),
                "mean_effect_minus_attribution": round(statistics.fmean(float(r["effect_minus_attribution"]) for r in rows), 5),
                "median_abs_mismatch": round(statistics.median(float(r["abs_mismatch"]) for r in rows), 5),
                "same_sign_rate": round(statistics.fmean(1.0 if r["same_sign"] else 0.0 for r in rows), 4),
            }
        )
    return out


def plot_reading_guide_rows(has_ablation: bool) -> list[dict[str, str]]:
    rows = [
        {"artifact": "plots/dla_dashboard.png", "concept": "one-screen overview", "question": "Where is the answer margin assembled, and is the ledger internally contested?"},
        {"artifact": "plots/cumulative_logit_diff.png", "concept": "assembly over depth", "question": "When does the frozen ledger turn toward the target or distractor?"},
        {"artifact": "plots/contribution_by_layer.png", "concept": "writer timing", "question": "Which layers and writer types do the signed pushing?"},
        {"artifact": "plots/ledger_phase_atlas.png", "concept": "coarse depth phases", "question": "Do early/middle/late/final-tail blocks play different roles by category?"},
        {"artifact": "plots/category_ledger_composition.png", "concept": "cancellation", "question": "How much positive and negative mass is hiding behind the net logit difference?"},
        {"artifact": "plots/answer_margin_vs_cancellation.png", "concept": "confidence vs internal fight", "question": "Can a confident-looking answer still be assembled by opposed components?"},
        {"artifact": "plots/component_type_balance_scatter.png", "concept": "attention vs MLP balance", "question": "Do examples lean attention-heavy, MLP-heavy, or both?"},
        {"artifact": "plots/relation_family_ledger_matrix.png", "concept": "dataset joins", "question": "Do relation families or hard/easy distractor contrasts recruit different phases?"},
        {"artifact": "plots/top_component_by_example.png", "concept": "largest row audit", "question": "Which single component dominates each example, and in what direction?"},
        {"artifact": "plots/ledger_waterfall_<example>.png", "concept": "showcase ledger", "question": "How do the largest entries cumulatively assemble one example's margin?"},
        {"artifact": "plots/dla_vs_lens_<example>.png", "concept": "readout convention contrast", "question": "Why does frozen DLA differ from the moving-basis logit lens?"},
    ]
    if has_ablation:
        rows.extend([
            {"artifact": "plots/attribution_vs_ablation.png", "concept": "attribution-to-causal calibration", "question": "Does a large DLA row predict a large live final-position effect?"},
            {"artifact": "plots/ablation_mismatch_examples.png", "concept": "mismatch audit", "question": "Which selected components are most ledger-like versus most downstream-amplified?"},
            {"artifact": "plots/ablation_mismatch_by_layer.png", "concept": "mismatch localization", "question": "Where over depth do frozen attribution and live effect disagree?"},
        ])
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def category_color(category: str) -> str:
    if hasattr(bench, "plot_category_color"):
        return bench.plot_category_color(category)
    return getattr(bench, "CATEGORY_COLORS", {}).get(category, "#555555")


def category_marker(category: str) -> str:
    if hasattr(bench, "plot_category_marker"):
        return bench.plot_category_marker(category)
    return {"fact": "o", "relation": "^", "grammar": "s", "conflict": "X"}.get(category, "o")


def component_color(kind: str) -> str:
    palette = getattr(bench, "COMPONENT_COLORS", {})
    return palette.get(kind, {"attn": "#0072B2", "mlp": "#E69F00", "embed": "#555555", "constant": "#999999", "all": "#222222"}.get(kind, "#555555"))


def selection_marker(selection: str) -> str:
    return {"top": "o", "random_control": "s", "low_attribution_control": "^"}.get(selection, "o")


def add_phase_guides(ax: Any, n_layers: int) -> None:
    if hasattr(bench, "add_depth_phase_guides"):
        bench.add_depth_phase_guides(ax, n_layers)
        return
    for frac in (1 / 3, 2 / 3):
        ax.axvline(n_layers * frac, color="#777777", linestyle=":", linewidth=0.7, alpha=0.25)
    ax.axvline(n_layers, color="#444444", linestyle=":", linewidth=0.9, alpha=0.5)


def label_panel(ax: Any, label: str) -> None:
    if hasattr(bench, "add_panel_label"):
        bench.add_panel_label(ax, label)
    else:
        ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right")


def _array_iqr(rows: list[list[float]], n: int) -> tuple[list[float], list[float], list[float]]:
    med, lo, hi = [], [], []
    for i in range(n):
        vals = [float(r[i]) for r in rows if i < len(r)]
        m, q1, q3 = median_iqr(vals)
        med.append(m); lo.append(q1); hi.append(q3)
    return med, lo, hi


def flatten_axes(axes: Any) -> list[Any]:
    """Return a plain list from a Matplotlib axes object/list/ndarray."""
    if isinstance(axes, (list, tuple)):
        out: list[Any] = []
        for item in axes:
            out.extend(flatten_axes(item))
        return out
    if hasattr(axes, "flat"):
        return list(axes.flat)
    return [axes]


def plot_contribution_by_layer(ctx: bench.RunContext, per_example: list[dict[str, Any]], n_layers: int) -> None:
    """Category panels with median/IQR attention and MLP contribution by layer."""
    if not per_example:
        return
    import matplotlib.pyplot as plt

    cats = [c for c in CATEGORIES if any(r["category"] == c for r in per_example)]
    if not cats:
        return
    ncols = 2
    nrows = math.ceil(len(cats) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.2, 4.0 * nrows), sharex=True)
    axes_list = flatten_axes(axes)
    x = list(range(n_layers))
    for idx, (ax, cat) in enumerate(zip(axes_list, cats)):
        rows = [r for r in per_example if r["category"] == cat]
        for kind, style in (("attn", "-"), ("mlp", "--")):
            arr = [r["dla"][f"{kind}_scores"] for r in rows]
            med, lo, hi = _array_iqr(arr, n_layers)
            color = component_color(kind)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, med, linestyle=style, color=color, linewidth=2.2, label=f"{kind} median")
            peak = int(max(range(n_layers), key=lambda j: abs(med[j]))) if med else 0
            if med and math.isfinite(med[peak]):
                ax.scatter([peak], [med[peak]], color=color, edgecolor="white", linewidth=0.7, zorder=4, s=34)
        ax.axhline(0, color="#222222", linewidth=0.8)
        add_phase_guides(ax, n_layers)
        ax.set_title(f"{cat} (n={len(rows)})")
        ax.set_ylabel("logit-diff contribution")
        ax.legend(loc="upper left", fontsize=8)
        label_panel(ax, chr(ord("A") + idx))
    for ax in axes_list[len(cats):]:
        ax.set_visible(False)
    for ax in axes_list[:len(cats)]:
        ax.set_xlabel("layer")
    fig.suptitle("Component writers by layer: median line, IQR ribbon")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "contribution_by_layer.png", "Median/IQR attention and MLP contribution per category and layer.")


def plot_signed_component_heatmap(ctx: bench.RunContext, per_example: list[dict[str, Any]], n_layers: int) -> None:
    """Example x layer heatmap of total direct contribution per block."""
    if not per_example:
        return
    import matplotlib.colors as mcolors

    rows = sorted(
        per_example,
        key=lambda r: (
            CATEGORIES.index(r["category"]) if r["category"] in CATEGORIES else len(CATEGORIES),
            float(r["behavior"].get("model_logit_diff", 0.0)),
            r["example_id"],
        ),
    )
    data = [[float(r["dla"]["attn_scores"][layer] + r["dla"]["mlp_scores"][layer]) for layer in range(n_layers)] for r in rows]
    lim = abs_percentile([value for row in data for value in row], 0.96)
    fig_height = max(5.5, min(12.0, 0.34 * len(rows) + 2.0))
    fig, ax = bench.new_figure(figsize=(11.2, fig_height))
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", norm=norm)
    tick_step = max(1, n_layers // 8)
    ax.set_xticks(range(0, n_layers, tick_step))
    ax.set_xlabel("layer")
    ax.set_yticks(range(len(rows)))
    labels = [f"{r['category'][:3]}:{r.get('relation_family','')[:8]}:{r['example_id']}" for r in rows]
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_title("Signed block contribution atlas (attention + MLP)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("logit-diff contribution")
    previous = rows[0]["category"]
    for i, row in enumerate(rows[1:], start=1):
        if row["category"] != previous:
            ax.axhline(i - 0.5, color="#111111", linewidth=0.8)
            previous = row["category"]
    fig.tight_layout()
    bench.save_figure(ctx, fig, "signed_component_heatmap.png", "Example-by-layer heatmap of signed block contribution to the answer direction.")


def plot_cumulative(ctx: bench.RunContext, per_example: list[dict[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(10.2, 5.8))
    plotted = False
    max_depth = 0
    for cat in CATEGORIES:
        rows = [r for r in per_example if r["category"] == cat]
        if not rows:
            continue
        plotted = True
        color = category_color(cat)
        depth = min(len(r["curve"]) for r in rows)
        max_depth = max(max_depth, depth - 1)
        xs = list(range(depth))
        for r in rows:
            ax.plot(xs, r["curve"][:depth], color=color, alpha=0.18, linewidth=0.9)
        med, lo, hi = _array_iqr([r["curve"][:depth] for r in rows], depth)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.10, linewidth=0)
        ax.plot(xs, med, color=color, linewidth=2.7, label=f"{cat} median (n={len(rows)})")
        if hasattr(bench, "label_line_end"):
            bench.label_line_end(ax, xs, med, cat, color=color)
    if not plotted:
        bench.close_figure(fig)
        return
    ax.axhline(0, color="#222222", linewidth=0.8)
    add_phase_guides(ax, max_depth)
    ax.set_xlabel("depth (0 = embeddings + constants, k = after block k-1)")
    ax.set_ylabel("cumulative logit difference (frozen norm)")
    ax.set_title("Cumulative DLA ledger: median assembly path with per-example traces")
    ax.legend(fontsize=8, loc="best")
    bench.save_figure(ctx, fig, "cumulative_logit_diff.png", "Cumulative component ledger per example with category medians and IQR ribbons.")


def plot_category_ledger_composition(ctx: bench.RunContext, per_example: list[dict[str, Any]]) -> None:
    """Diverging gross mass bars: positive writes vs negative writes."""
    cats = [cat for cat in CATEGORIES if any(r["category"] == cat for r in per_example)]
    if not cats:
        return
    fig, ax = bench.new_figure(figsize=(9.4, max(4.2, 0.65 * len(cats) + 2.2)))
    y = list(range(len(cats)))
    for yi, cat in zip(y, cats):
        rows = [r for r in per_example if r["category"] == cat]
        pos_vals, neg_vals, net_vals = [], [], []
        for r in rows:
            vals = [float(r["dla"]["embed_score"]), float(r["dla"]["constant"])] + [float(v) for v in r["dla"]["attn_scores"] + r["dla"]["mlp_scores"]]
            pos_vals.append(sum(v for v in vals if v > 0))
            neg_vals.append(abs(sum(v for v in vals if v < 0)))
            net_vals.append(sum(vals))
        pos = statistics.fmean(pos_vals)
        neg = statistics.fmean(neg_vals)
        net = statistics.fmean(net_vals)
        color = category_color(cat)
        ax.barh(yi, pos, left=0, height=0.45, color=color, alpha=0.70, label="positive mass" if yi == 0 else None)
        ax.barh(yi, -neg, left=0, height=0.45, color=color, alpha=0.25, label="negative mass" if yi == 0 else None)
        ax.scatter([net], [yi], marker="D", color="#222222", s=40, zorder=5, label="net" if yi == 0 else None)
        ax.text(pos, yi + 0.27, f"+{pos:.1f}", ha="right", va="bottom", fontsize=7, color="#222222")
        ax.text(-neg, yi + 0.27, f"-{neg:.1f}", ha="left", va="bottom", fontsize=7, color="#222222")
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.set_xlabel("mean logit-diff mass (positive right, negative left)")
    ax.set_title("Gross ledger mass vs net answer preference")
    ax.legend(loc="best", fontsize=8)
    bench.save_figure(ctx, fig, "category_ledger_composition.png", "Diverging positive and negative component mass by category, with net marker.")


def plot_balance_errors(ctx: bench.RunContext, balance_rows: list[dict[str, Any]]) -> None:
    if not balance_rows:
        return
    rows = sorted(balance_rows, key=lambda r: max(float(r["ledger_vs_frozen_abs_err"]), float(r["frozen_vs_model_abs_err"])))
    fig, ax = bench.new_figure(figsize=(10.0, 5.6))
    xs = list(range(len(rows)))
    ledger_err = [float(r["ledger_vs_frozen_abs_err"]) for r in rows]
    model_err = [float(r["frozen_vs_model_abs_err"]) for r in rows]
    ax.scatter(xs, ledger_err, marker="o", label="ledger vs frozen", s=42)
    ax.scatter(xs, model_err, marker="s", label="frozen fp32 vs model dtype", s=42)
    ax.set_yscale("symlog", linthresh=1e-7)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r["example_id"]) for r in rows], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("absolute error (symlog)")
    ax.set_title("Ledger balance checks by example")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "ledger_balance_errors.png", "Per-example ledger balance and dtype-gap errors.")


def plot_top_component_lollipop(ctx: bench.RunContext, top_rows: list[dict[str, Any]]) -> None:
    if not top_rows:
        return
    rows = [r for r in top_rows if int(r["rank"]) == 1]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (str(r["category"]), float(r["score"])))
    fig_height = max(5.5, min(12.5, 0.38 * len(rows) + 1.8))
    fig, ax = bench.new_figure(figsize=(9.8, fig_height))
    y = list(range(len(rows)))
    for yi, row in zip(y, rows):
        score = float(row["score"])
        color = category_color(str(row["category"]))
        marker = "o" if row["component"] == "attn" else "s"
        ax.plot([0, score], [yi, yi], color=color, alpha=0.55, linewidth=2.2)
        ax.scatter(score, yi, s=55 + 180 * float(row.get("abs_writer_mass_share", 0)), marker=marker, color=color, edgecolor="black", linewidth=0.5, zorder=4)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['example_id']}  {r['component']}@{r['layer']}  ({float(r.get('abs_writer_mass_share',0))*100:.0f}%)" for r in rows], fontsize=8)
    ax.set_xlabel("top component attribution score (shape: circle=attn, square=MLP)")
    ax.set_title("Largest component per example: sign, size, and share of writer mass")
    bench.save_figure(ctx, fig, "top_component_by_example.png", "Top-|attribution| component for every example, sized by share of writer mass.")


def plot_showcase_waterfall(ctx: bench.RunContext, example: AnswerPairExample, dla: dict[str, Any], n: int = 14) -> None:
    """Waterfall in computation order, with an other bucket for the long tail."""
    components: list[tuple[int, str, float]] = [(0, "embed", float(dla["embed_score"])), (1, "constant", float(dla["constant"]))]
    order = 2
    for layer, (a, m) in enumerate(zip(dla["attn_scores"], dla["mlp_scores"])):
        components.append((order, f"A{layer}", float(a))); order += 1
        components.append((order, f"M{layer}", float(m))); order += 1
    top_names = {name for _, name, _ in sorted(components, key=lambda kv: abs(kv[2]), reverse=True)[:n]}
    chosen = [(o, name, score) for o, name, score in components if name in top_names]
    other = sum(score for _, name, score in components if name not in top_names)
    if abs(other) > 1e-9:
        chosen.append((10_000, "other", other))
    chosen = sorted(chosen, key=lambda kv: kv[0])
    fig, ax = bench.new_figure(figsize=(9.8, max(5.4, 0.39 * len(chosen) + 2.0)))
    running = 0.0
    ys, labels = [], []
    for i, (_, name, score) in enumerate(chosen):
        y = len(chosen) - i - 1
        x0, x1 = running, running + score
        left, width = min(x0, x1), abs(score)
        color = "#0072B2" if score >= 0 else "#D55E00"
        ax.barh(y, width, left=left, height=0.55, color=color, alpha=0.72)
        ax.plot([x0, x0], [y - 0.26, y + 0.26], color="#333333", linewidth=0.6, alpha=0.5)
        ax.text(x1, y, f"{score:+.2f}", va="center", ha="left" if score >= 0 else "right", fontsize=7.5)
        running = x1
        ys.append(y); labels.append(name)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.axvline(float(dla["frozen_logit_diff"]), color="#222222", linestyle="--", linewidth=1.0, label="frozen total")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("cumulative frozen-norm logit-diff ledger")
    ax.set_title(f"Waterfall ledger for {example.example_id}: target {bench.visible_token(example.target)} vs distractor {bench.visible_token(example.distractor)}")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, f"ledger_waterfall_{bench.sanitize_tag(example.example_id)}.png", "Computation-order waterfall of the largest ledger entries plus other mass.")


def plot_attribution_vs_ablation(ctx: bench.RunContext, ablation_rows: list[dict[str, Any]]) -> float | None:
    if not ablation_rows:
        return None
    xs = [float(r["attribution_score"]) for r in ablation_rows]
    ys = [float(r["causal_effect"]) for r in ablation_rows]
    rho = spearman_rho(xs, ys)
    fig, ax = bench.new_figure(figsize=(8.0, 6.6))
    for row in ablation_rows:
        edge = "black" if row["same_sign"] else "#D55E00"
        ax.scatter(float(row["attribution_score"]), float(row["causal_effect"]), marker=selection_marker(str(row["selection"])), s=52, alpha=0.78, color=category_color(str(row["category"])), edgecolor=edge, linewidth=0.7)
    lim = max(max(abs(v) for v in xs), max(abs(v) for v in ys)) * 1.12 or 1.0
    ax.plot([-lim, lim], [-lim, lim], color="#555555", linewidth=0.9, linestyle="--", label="attribution = effect")
    ax.axhline(0, color="#222222", linewidth=0.6)
    ax.axvline(0, color="#222222", linewidth=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("attribution score (frozen-norm logit diff)")
    ax.set_ylabel("final-position ablation effect (logit diff change)")
    title = "Attribution vs causal effect"
    if rho is not None:
        title += f" (Spearman rho = {rho:.3f}, n = {len(xs)})"
    ax.set_title(title)
    # Minimal custom legend by selection.
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color="#666666", marker=selection_marker(k), linestyle="None", label=k, markersize=6) for k in ("top", "random_control", "low_attribution_control") if any(r["selection"] == k for r in ablation_rows)]
    handles.append(mlines.Line2D([], [], color="#555555", linestyle="--", label="identity"))
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    bench.save_figure(ctx, fig, "attribution_vs_ablation.png", "Ledger score versus live final-position ablation effect, colored by category and shaped by selection.")
    return rho


def plot_ablation_mismatches(ctx: bench.RunContext, ablation_rows: list[dict[str, Any]]) -> None:
    if not ablation_rows:
        return
    rows = sorted(ablation_rows, key=lambda r: abs(float(r["causal_effect"]) - float(r["attribution_score"])), reverse=True)[: min(14, len(ablation_rows))]
    fig_height = max(5.2, 0.46 * len(rows) + 1.8)
    fig, ax = bench.new_figure(figsize=(10.0, fig_height))
    y_positions = list(range(len(rows)))
    for y, row in zip(y_positions, reversed(rows)):
        attr = float(row["attribution_score"])
        effect = float(row["causal_effect"])
        color = category_color(str(row["category"]))
        ax.plot([attr, effect], [y, y], color=color, alpha=0.55, linewidth=2.3)
        ax.scatter(attr, y, marker="o", color=color, edgecolor="black", linewidth=0.5, s=54)
        ax.scatter(effect, y, marker="s", color=color, edgecolor="black", linewidth=0.5, s=54)
        ax.text(effect, y + 0.22, f"Δ={effect-attr:+.2f}", fontsize=7, ha="left" if effect >= attr else "right", color="#333333")
    labels = [f"{r['example_id']} {r['component']}@{r['layer']} ({r['selection']})" for r in reversed(rows)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("logit-diff units: circle = attribution, square = ablation effect")
    ax.set_title("Largest attribution-vs-ablation mismatches")
    ax.grid(True, axis="x", alpha=0.25)
    bench.save_figure(ctx, fig, "ablation_mismatch_examples.png", "Largest final-position attribution versus ablation-effect disagreements.")


def plot_ablation_mismatch_by_layer(ctx: bench.RunContext, ablation_rows: list[dict[str, Any]]) -> None:
    if not ablation_rows:
        return
    fig, ax = bench.new_figure(figsize=(9.6, 5.4))
    for row in ablation_rows:
        layer = int(row["layer"])
        mismatch = float(row["effect_minus_attribution"])
        size = 35 + 22 * min(8.0, abs(float(row["attribution_score"])))
        ax.scatter(layer, mismatch, s=size, marker="o" if row["component"] == "attn" else "s", color=category_color(str(row["category"])), alpha=0.75, edgecolor="black", linewidth=0.4)
    max_layer = max(int(r["layer"]) for r in ablation_rows)
    add_phase_guides(ax, max_layer + 1)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("component layer (shape: circle=attn, square=MLP)")
    ax.set_ylabel("effect_minus_attribution")
    ax.set_title("Where live ablation departs from frozen attribution")
    bench.save_figure(ctx, fig, "ablation_mismatch_by_layer.png", "Mismatch between live final-position ablation effect and frozen DLA score by layer.")


def plot_dla_vs_lens(ctx: bench.RunContext, example: AnswerPairExample, curve: list[float], traj: bench.LensTrajectory) -> None:
    if traj.logit_target is None or traj.logit_distractor is None:
        return
    lens_diff = [float(t - d) for t, d in zip(traj.logit_target, traj.logit_distractor)]
    fig, ax = bench.new_figure(figsize=(9.8, 5.8))
    xs_curve = list(range(len(curve)))
    xs_lens = list(range(len(lens_diff)))
    ax.plot(xs_curve, curve, linewidth=2.8, label="cumulative DLA (final norm frozen)", color="#0072B2")
    ax.plot(xs_lens, lens_diff, linewidth=2.1, linestyle="--", label="logit lens (norm recomputed per depth)", color="#D55E00")
    overlap = min(len(curve), len(lens_diff))
    if overlap:
        ax.fill_between(range(overlap), curve[:overlap], lens_diff[:overlap], color="#777777", alpha=0.10, label="readout-convention gap")
    ax.axhline(0, color="#222222", linewidth=0.8)
    add_phase_guides(ax, len(curve) - 1)
    ax.set_xlabel("depth")
    ax.set_ylabel("logit(target) - logit(distractor)")
    ax.set_title(f"Two readouts of the same stream: {example.example_id}")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, f"dla_vs_lens_{bench.sanitize_tag(example.example_id)}.png", "Frozen-norm cumulative ledger versus per-depth logit lens for the showcase example.")


def plot_phase_ledger_atlas(ctx: bench.RunContext, phase_rows: list[dict[str, Any]]) -> None:
    if not phase_rows:
        return
    import matplotlib.colors as mcolors
    cats = [c for c in CATEGORIES if any(r["category"] == c and r["component_kind"] == "all" for r in phase_rows)]
    phases = ["embed_const", "early", "middle", "late", "final_tail"]
    data = [[float("nan") for _ in phases] for _ in cats]
    gross = [[float("nan") for _ in phases] for _ in cats]
    lookup = {(r["category"], r["phase"]): r for r in phase_rows if r["component_kind"] == "all"}
    for i, cat in enumerate(cats):
        for j, phase in enumerate(phases):
            row = lookup.get((cat, phase))
            if row:
                data[i][j] = float(row["mean_net_score"])
                gross[i][j] = float(row["mean_gross_mass"])
    lim = abs_percentile([value for row in data for value in row], 0.95)
    fig, ax = bench.new_figure(figsize=(9.8, max(4.8, 0.58 * len(cats) + 2.0)))
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    im = ax.imshow(data, cmap="coolwarm", norm=norm, aspect="auto")
    ax.set_xticks(range(len(phases))); ax.set_xticklabels([p.replace("_", "\n") for p in phases])
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    for i in range(len(cats)):
        for j in range(len(phases)):
            if math.isfinite(data[i][j]):
                ax.text(j, i, f"{data[i][j]:+.1f}\nΣ|·|={gross[i][j]:.1f}", ha="center", va="center", fontsize=7, color="#111111")
    ax.set_title("Ledger phase atlas: net score colored, gross mass printed")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("mean net logit-diff score")
    bench.save_figure(ctx, fig, "ledger_phase_atlas.png", "Category-by-depth-phase ledger atlas: color is net score, text is gross mass.")


def plot_component_type_balance(ctx: bench.RunContext, per_example: list[dict[str, Any]]) -> None:
    if not per_example:
        return
    fig, ax = bench.new_figure(figsize=(8.4, 6.6))
    for r in per_example:
        attn = sum(float(v) for v in r["dla"]["attn_scores"])
        mlp = sum(float(v) for v in r["dla"]["mlp_scores"])
        size = 45 + 12 * min(component_score_mass(r["dla"]), 10.0)
        ax.scatter(attn, mlp, color=category_color(str(r["category"])), marker=category_marker(str(r["category"])), s=size, edgecolor="black", linewidth=0.5, alpha=0.78)
    vals = [sum(float(v) for v in r["dla"]["attn_scores"]) for r in per_example] + [sum(float(v) for v in r["dla"]["mlp_scores"]) for r in per_example]
    lim = (max(abs(v) for v in vals) * 1.15 if vals else 1.0) or 1.0
    ax.axhline(0, color="#222222", linewidth=0.8); ax.axvline(0, color="#222222", linewidth=0.8)
    ax.plot([-lim, lim], [-lim, lim], color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("total attention contribution")
    ax.set_ylabel("total MLP contribution")
    ax.set_title("Attention-vs-MLP ledger balance per example")
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=category_color(c), marker=category_marker(c), linestyle="None", label=c, markersize=7) for c in CATEGORIES if any(r["category"] == c for r in per_example)]
    ax.legend(handles=handles, fontsize=8, loc="best")
    bench.save_figure(ctx, fig, "component_type_balance_scatter.png", "Per-example attention-total versus MLP-total contribution, sized by gross writer mass.")


def plot_answer_margin_vs_cancellation(ctx: bench.RunContext, per_example: list[dict[str, Any]]) -> None:
    if not per_example:
        return
    fig, ax = bench.new_figure(figsize=(8.8, 6.2))
    for r in per_example:
        gross = component_score_mass(r["dla"], include_embed_and_constant=True)
        net_ratio = abs(float(r["dla"]["frozen_logit_diff"])) / gross if gross else 0.0
        margin = float(r["dla"]["model_logit_diff"])
        ax.scatter(margin, net_ratio, s=45 + 16 * min(gross, 12.0), color=category_color(str(r["category"])), marker=category_marker(str(r["category"])), edgecolor="black", linewidth=0.5, alpha=0.78)
        if net_ratio < 0.25 or abs(margin) < 0.5:
            ax.annotate(str(r["example_id"]), (margin, net_ratio), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("model target-vs-distractor logit diff")
    ax.set_ylabel("net/gross ledger ratio (lower = more cancellation)")
    ax.set_title("Confidence is not the same as an uncontested ledger")
    bench.save_figure(ctx, fig, "answer_margin_vs_cancellation.png", "Final answer margin versus ledger cancellation ratio, sized by gross ledger mass.")


def plot_relation_family_matrix(ctx: bench.RunContext, relation_rows: list[dict[str, Any]], phase_rows: list[dict[str, Any]]) -> None:
    if not relation_rows:
        return
    import matplotlib.colors as mcolors
    families = sorted({f"{r['category']}:{r['relation_family']}:{r['contrast_type']}" for r in relation_rows})
    if len(families) <= 1:
        return
    # For relation-family rows we only have final aggregates; recompute family phase rows from phase_rows is category-only.
    # The compact matrix therefore uses mean totals by component class, which works for both built-in and custom sets.
    cols = ["embed", "attn", "mlp", "gross", "net/gross"]
    data = [[0.0 for _ in cols] for _ in families]
    lookup = {f"{r['category']}:{r['relation_family']}:{r['contrast_type']}": r for r in relation_rows}
    for i, family in enumerate(families):
        r = lookup[family]
        data[i][0] = float(r["mean_embed_score"])
        data[i][1] = float(r["mean_attn_total"])
        data[i][2] = float(r["mean_mlp_total"])
        data[i][3] = float(r["gross_ledger_mass"])
        data[i][4] = float(r["net_to_gross_ratio"])
    lim = abs_percentile([row[j] for row in data for j in range(3)], 0.95)
    fig, ax = bench.new_figure(figsize=(10.6, max(5.0, 0.44 * len(families) + 2.0)))
    # Put gross/net columns on an unsigned scale by normalizing them into color range but printing values.
    display = [[row[j] if j < 3 else 0.0 for j in range(len(cols))] for row in data]
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    im = ax.imshow(display, cmap="coolwarm", norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(families))); ax.set_yticklabels(families, fontsize=7.5)
    for i in range(len(families)):
        for j in range(len(cols)):
            ax.text(j, i, f"{data[i][j]:+.2f}" if j < 3 else f"{data[i][j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Relation-family ledger matrix (custom hard/easy sets land here)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("signed mean contribution for embed/attn/MLP")
    bench.save_figure(ctx, fig, "relation_family_ledger_matrix.png", "Relation-family matrix of mean embed, attention, MLP, gross mass, and cancellation ratio.")


def plot_dla_dashboard(ctx: bench.RunContext, per_example: list[dict[str, Any]], phase_rows: list[dict[str, Any]], ablation_rows: list[dict[str, Any]], rho: float | None) -> None:
    """A compact four-panel overview for teaching and quick run triage."""
    if not per_example:
        return
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    ax = axes[0, 0]
    for cat in CATEGORIES:
        rows = [r for r in per_example if r["category"] == cat]
        if not rows:
            continue
        depth = min(len(r["curve"]) for r in rows)
        xs = list(range(depth))
        med, lo, hi = _array_iqr([r["curve"][:depth] for r in rows], depth)
        color = category_color(cat)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.10, linewidth=0)
        ax.plot(xs, med, color=color, linewidth=2.3, label=cat)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_title("A. cumulative ledger")
    ax.set_xlabel("depth"); ax.set_ylabel("logit diff")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for r in per_example:
        gross = component_score_mass(r["dla"], include_embed_and_constant=True)
        net_ratio = abs(float(r["dla"]["frozen_logit_diff"])) / gross if gross else 0.0
        ax.scatter(float(r["dla"]["model_logit_diff"]), net_ratio, s=34 + 13 * min(gross, 10.0), color=category_color(str(r["category"])), marker=category_marker(str(r["category"])), alpha=0.75, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("B. margin vs cancellation")
    ax.set_xlabel("model logit diff"); ax.set_ylabel("net/gross")

    ax = axes[1, 0]
    cats = [c for c in CATEGORIES if any(pr["category"] == c and pr["component_kind"] == "all" for pr in phase_rows)]
    phases = ["embed_const", "early", "middle", "late", "final_tail"]
    data = [[float("nan") for _ in phases] for _ in cats]
    lookup = {(r["category"], r["phase"]): r for r in phase_rows if r["component_kind"] == "all"}
    for i, cat in enumerate(cats):
        for j, phase in enumerate(phases):
            row = lookup.get((cat, phase))
            if row:
                data[i][j] = float(row["mean_net_score"])
    lim = abs_percentile([value for row in data for value in row], 0.95)
    im = ax.imshow(data, cmap="coolwarm", norm=mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
    ax.set_xticks(range(len(phases))); ax.set_xticklabels([p.replace("_", "\n") for p in phases], fontsize=8)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    ax.set_title("C. phase net score")

    ax = axes[1, 1]
    if ablation_rows:
        for row in ablation_rows:
            ax.scatter(float(row["attribution_score"]), float(row["causal_effect"]), marker=selection_marker(str(row["selection"])), s=38, color=category_color(str(row["category"])), alpha=0.75, edgecolor="black", linewidth=0.4)
        xs = [float(r["attribution_score"]) for r in ablation_rows]
        ys = [float(r["causal_effect"]) for r in ablation_rows]
        lim = (max(max(abs(v) for v in xs), max(abs(v) for v in ys)) * 1.1) or 1.0
        ax.plot([-lim, lim], [-lim, lim], color="#555555", linestyle="--", linewidth=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        title = "D. attribution vs ablation"
        if rho is not None:
            title += f" (ρ={rho:.2f})"
        ax.set_title(title)
        ax.set_xlabel("attribution"); ax.set_ylabel("live effect")
    else:
        vals = [(sum(float(v) for v in r["dla"]["attn_scores"]), sum(float(v) for v in r["dla"]["mlp_scores"]), r) for r in per_example]
        for attn, mlp, r in vals:
            ax.scatter(attn, mlp, color=category_color(str(r["category"])), marker=category_marker(str(r["category"])), s=48, edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="#222222", linewidth=0.8); ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_title("D. attention vs MLP totals")
        ax.set_xlabel("attention total"); ax.set_ylabel("MLP total")
    fig.suptitle("Direct Logit Attribution dashboard")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "dla_dashboard.png", "One-screen Lab 2 dashboard: assembly, cancellation, phase net score, and ablation calibration.")


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
    relation_rows: list[dict[str, Any]] | None = None,
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
    relation_rows = relation_rows or []
    relation_scoped = [r for r in relation_rows if int(r.get("n_examples", 0)) >= 2]
    if len(relation_scoped) >= 2:
        strongest_family = max(relation_scoped, key=lambda r: abs(float(r.get("mean_frozen_logit_diff", 0.0))))
        claims.append(
            {
                "id": f"{LAB_ID}-C4",
                "tag": "ATTR",
                "text": (
                    "Relation-family metadata changed the unit of comparison from one prompt at a time to "
                    f"family-level ledgers: the largest mean frozen logit difference among multi-example families was "
                    f"{strongest_family['category']}:{strongest_family['relation_family']}:{strongest_family['contrast_type']} "
                    f"with mean frozen diff {strongest_family['mean_frozen_logit_diff']} and modal top phase "
                    f"{strongest_family['top_component_modal_phase']}."
                ),
                "artifact": f"runs/{run_name}/tables/relation_family_summary.csv",
                "falsifier": (
                    "A held-out relation-family file with the same hard/easy contrast changes the modal top phase or "
                    "removes the family-level difference, showing the apparent relation pattern was prompt-specific."
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
    phase_rows: list[dict[str, Any]] | None = None,
    relation_rows: list[dict[str, Any]] | None = None,
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
    relation_rows = relation_rows or []
    if relation_rows:
        multi = [r for r in relation_rows if int(r.get("n_examples", 0)) >= 2]
        if multi:
            lines += [
                "",
                "Relation-family view:",
                "",
                "| family | n | mean frozen diff | attn total | mlp total | modal top phase |",
                "|---|---:|---:|---:|---:|---|",
            ]
            for r in sorted(multi, key=lambda x: (x["category"], x["relation_family"], x["contrast_type"]))[:12]:
                lines.append(
                    f"| {r['category']}:{r['relation_family']}:{r['contrast_type']} | {r['n_examples']} | "
                    f"{r['mean_frozen_logit_diff']} | {r['mean_attn_total']} | {r['mean_mlp_total']} | "
                    f"{r['top_component_modal_phase']} |"
                )
    lines += [
        "",
        "Plot reading guide: start with `plots/dla_dashboard.png`, then use `tables/plot_reading_guide.csv` as the map from visual to concept.",
        "The new phase and relation-family tables are `tables/phase_ledger_summary.csv` and `tables/relation_family_summary.csv`.",
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
            **example_meta(ex),
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
        meta = example_meta(ex)
        contributions.extend(component_rows(ex.example_id, ex.category, dla, **meta))
        behavior_rows.append(answer_behavior_row(bundle, ex, t_id, d_id, comp.capture.final_logits_last))
        balance_rows.append(balance_row(ex.example_id, ex.category, dla))
        block_rows_all.extend(block_ledger_rows(ex.example_id, ex.category, dla, curve, **meta))
        top_rows_all.extend(top_component_rows(ex.example_id, ex.category, dla, **meta))
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
                **example_meta(ex),
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
                "relation_family": r.get("relation_family", ""),
                "contrast_type": r.get("contrast_type", ""),
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

    phase_rows = phase_ledger_summary(per_example, n_layers)
    phase_path = ctx.path("tables", "phase_ledger_summary.csv")
    bench.write_csv_with_context(ctx, phase_path, phase_rows)
    ctx.register_artifact(phase_path, "table", "Category x depth-phase ledger summaries with net, gross, positive, and negative mass.")

    relation_rows = relation_family_summary(per_example, n_layers)
    relation_path = ctx.path("tables", "relation_family_summary.csv")
    bench.write_csv_with_context(ctx, relation_path, relation_rows)
    ctx.register_artifact(relation_path, "table", "Relation-family / contrast-type summary for built-in and custom prompt datasets.")

    mismatch_summary_rows = ablation_mismatch_summary(ablation_rows)
    if mismatch_summary_rows:
        mismatch_summary_path = ctx.path("tables", "ablation_mismatch_summary.csv")
        bench.write_csv_with_context(ctx, mismatch_summary_path, mismatch_summary_rows)
        ctx.register_artifact(mismatch_summary_path, "table", "Attribution-vs-ablation mismatch aggregated by category, component type, and selection.")

    plot_guide_rows = plot_reading_guide_rows(bool(ablation_rows))
    plot_guide_path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv(plot_guide_path, plot_guide_rows)
    ctx.register_artifact(plot_guide_path, "table", "Map from Lab 2 plots to the concept each plot teaches.")

    # Plots.
    rho: float | None = None
    if not args.no_plots:
        plot_contribution_by_layer(ctx, per_example, n_layers)
        plot_signed_component_heatmap(ctx, per_example, n_layers)
        plot_cumulative(ctx, per_example)
        plot_category_ledger_composition(ctx, per_example)
        plot_phase_ledger_atlas(ctx, phase_rows)
        plot_component_type_balance(ctx, per_example)
        plot_answer_margin_vs_cancellation(ctx, per_example)
        plot_relation_family_matrix(ctx, relation_rows, phase_rows)
        plot_balance_errors(ctx, balance_rows)
        plot_top_component_lollipop(ctx, top_rows_all)
        rho = plot_attribution_vs_ablation(ctx, ablation_rows)
        plot_ablation_mismatches(ctx, ablation_rows)
        plot_ablation_mismatch_by_layer(ctx, ablation_rows)
        plot_dla_dashboard(ctx, per_example, phase_rows, ablation_rows, rho)
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
        "relation_families": {f"{r['category']}:{r['relation_family']}:{r['contrast_type']}": r for r in relation_rows},
        "phase_summary_rows": len(phase_rows),
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

    claims = draft_claims(ctx, bundle, cat_rows, ablation_rows, rho, relation_rows)
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
        phase_rows=phase_rows,
        relation_rows=relation_rows,
    )
    summary_path = ctx.path("run_summary.md")
    bench.write_text(summary_path, summary)
    ctx.register_artifact(summary_path, "summary", "The seven standard questions answered with this run's numbers.")
    print(f"[lab2] wrote run_summary.md and {len(claims)} drafted ledger claims")
