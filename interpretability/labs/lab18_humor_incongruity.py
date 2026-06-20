"""Lab 18: Humor as incongruity, resolution, and cheap correlates.

This lab turns the overlarge word "humor" into a smaller object the course
can actually test: setup-dependent incongruity that resolves into a
joke-shaped ending. The lab asks whether a frozen instruct model exposes a
residual-stream handle for that object, and whether activation addition moves
joke-shaped generations more than surprise, silliness, positivity, or a random
control.

Evidence labels:
  * OBS for setup entropy, teacher-forced surprisal, and attention-to-setup;
  * DECODE for held-out joke-vs-control probe selectivity with null controls;
  * CAUSAL, narrowly, for activation-addition effects that survive cheap
    correlate controls and hand-label review.

The lab does not claim that the model experiences funniness. Its required
artifact is the audit that decides whether the handle is closer to joke
structure, joke register, raw surprise, silliness, positivity, or nothing clean.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import re
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L18"
DATA_FILE = "humor_incongruity_pairs.csv"
CONDITIONS = ("joke", "literal", "surprise", "silly", "positive")
CONTROL_CONDITIONS = ("literal", "surprise", "silly", "positive")
SEMANTIC_DIRECTIONS = ("joke_structure", "surprise", "silly", "positive")

PROMPT_SET_FAMILY_CAPS = {"small": 2, "medium": 4, "full": 0}
REPORT_SPLIT = "test"
N_NULL_REPS = 5
MAX_NEW_TOKENS = 42
ENGINE_MAX_CONCURRENT = 16
MAX_STEERING_ITEMS = 10
MAX_ATTENTION_ITEMS = 10
STEERING_DOSES = (0.25, 0.50, 0.75)

SYSTEM_PROMPT = (
    "You are a careful assistant. Analyze short text without adding personal "
    "experience claims. Keep responses concise."
)

GENERIC_JOKE_MARKERS = (
    "because|turns out|said|only|needed|wanted|pun|joke|joking|wordplay|"
    "punchline|setup|actually|couldn't|wouldn't|forecast|signal|key|date|"
    "cells|rolls|verse|breakpoints|deduction|plot twist"
)
GENERIC_SILLY_MARKERS = "tiny|soup|dance|triangle|hat|socks|spoon|glitter|midnight|banana|llama"
GENERIC_SURPRISE_MARKERS = "suddenly|unexpected|instead|future|hidden|weather|ticket|door|map|alarm"
GENERIC_POSITIVE_MARKERS = "good|great|friendly|happy|smiled|calm|helpful|relieved|hopeful|comforting|kind"


@dataclasses.dataclass
class HumorItem:
    item_id: str
    family: str
    setup: str
    joke_completion: str
    literal_completion: str
    surprise_completion: str
    silly_completion: str
    positive_completion: str
    setup_anchor: str = ""
    resolution_keyword: str = ""
    joke_markers: str = ""
    silly_markers: str = ""
    surprise_markers: str = ""
    positive_markers: str = ""
    note: str = ""


@dataclasses.dataclass
class RenderedAttention:
    prompt: str
    input_ids: list[int]
    tokens_text: list[str]
    attentions: Any


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def rounded(x: Any, ndigits: int = 4) -> Any:
    try:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return round(float(x), ndigits)
    except Exception:
        pass
    return x


def none_if_nan(x: Any, ndigits: int = 4) -> Any:
    try:
        val = float(x)
    except Exception:
        return x
    if not math.isfinite(val):
        return None
    return round(val, ndigits)


def safe_fmean(vals: Sequence[float], default: float = float("nan")) -> float:
    finite: list[float] = []
    for value in vals:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            finite.append(f)
    return float(statistics.fmean(finite)) if finite else default


def safe_stdev(vals: Sequence[float], default: float = float("nan")) -> float:
    finite: list[float] = []
    for value in vals:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            finite.append(f)
    if len(finite) < 2:
        return default
    return float(statistics.stdev(finite))


def auc_from_scores(pos: Sequence[float], neg: Sequence[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def maybe_unit(v: Any) -> Any | None:
    norm = v.norm()
    try:
        norm_val = float(norm)
    except Exception:
        return None
    if not math.isfinite(norm_val) or norm_val < 1e-8:
        return None
    return v / norm


def unit(v: Any) -> Any:
    out = maybe_unit(v)
    if out is None:
        raise RuntimeError("Direction norm was zero or non-finite.")
    return out


def random_unit(d_model: int, seed: int) -> Any:
    import torch

    gen = torch.Generator().manual_seed(int(seed))
    return unit(torch.randn(d_model, generator=gen))


def cosine(a: Any, b: Any) -> float:
    denom = (a.norm() * b.norm()).clamp_min(1e-9)
    return float((a @ b) / denom)


def as_path(text: str) -> pathlib.Path:
    p = pathlib.Path(text).expanduser()
    if p.is_absolute():
        return p
    return bench.COURSE_ROOT / p


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------


FALLBACK_ROWS: tuple[dict[str, str], ...] = (
    {
        "item_id": "fallback_library_01",
        "family": "wordplay",
        "setup": "The librarian warned the dictionary it was getting too thick.",
        "joke_completion": "It said it just had too many words to define itself.",
        "literal_completion": "It had been printed on unusually heavy paper.",
        "surprise_completion": "A weather balloon landed on the checkout desk.",
        "silly_completion": "A tiny spoon danced beside it wearing socks.",
        "positive_completion": "The librarian smiled and put it on a stronger shelf.",
        "setup_anchor": "dictionary",
        "resolution_keyword": "words",
        "joke_markers": "words|define|dictionary",
        "silly_markers": "spoon|socks|danced",
        "surprise_markers": "weather balloon|landed",
        "positive_markers": "smiled|stronger shelf",
        "note": "Tier A smoke fallback, not science data.",
    },
    {
        "item_id": "fallback_library_02",
        "family": "wordplay",
        "setup": "The calendar quit the band right before rehearsal.",
        "joke_completion": "It said its days were already numbered.",
        "literal_completion": "It had a conflicting appointment that evening.",
        "surprise_completion": "The drummer was secretly a vending machine.",
        "silly_completion": "Twelve tiny hats began arguing with a triangle.",
        "positive_completion": "Everyone calmly rescheduled and felt relieved.",
        "setup_anchor": "calendar",
        "resolution_keyword": "numbered",
        "joke_markers": "days|numbered|calendar",
        "silly_markers": "hats|triangle",
        "surprise_markers": "vending machine|secretly",
        "positive_markers": "calmly|relieved",
        "note": "Tier A smoke fallback, not science data.",
    },
    {
        "item_id": "fallback_workshop_01",
        "family": "reversal",
        "setup": "The carpenter brought a ruler to the poetry reading.",
        "joke_completion": "He wanted to measure the meter.",
        "literal_completion": "He had forgotten to leave it at the workshop.",
        "surprise_completion": "The stage lights turned into a map of Mars.",
        "silly_completion": "A glittery spoon recited the alphabet backwards.",
        "positive_completion": "The poets welcomed him and shared the microphone.",
        "setup_anchor": "ruler",
        "resolution_keyword": "meter",
        "joke_markers": "measure|meter|poetry",
        "silly_markers": "glittery spoon|alphabet",
        "surprise_markers": "Mars|stage lights",
        "positive_markers": "welcomed|shared",
        "note": "Tier A smoke fallback, not science data.",
    },
    {
        "item_id": "fallback_workshop_02",
        "family": "reversal",
        "setup": "The detective interviewed the broken pencil.",
        "joke_completion": "It had a good point once, but now it was pointless.",
        "literal_completion": "It had snapped during a note-taking session.",
        "surprise_completion": "A hidden door opened under the carpet.",
        "silly_completion": "Three bananas held a midnight committee meeting.",
        "positive_completion": "The detective kindly replaced it with a new one.",
        "setup_anchor": "pencil",
        "resolution_keyword": "pointless",
        "joke_markers": "point|pointless|pencil",
        "silly_markers": "bananas|midnight",
        "surprise_markers": "hidden door|opened",
        "positive_markers": "kindly|new one",
        "note": "Tier A smoke fallback, not science data.",
    },
    {
        "item_id": "fallback_cafe_01",
        "family": "script_violation",
        "setup": "The espresso machine asked for a raise after the morning rush.",
        "joke_completion": "It said it had been under a lot of pressure.",
        "literal_completion": "It had produced many drinks for the cafe.",
        "surprise_completion": "A quiet comet appeared in the milk pitcher.",
        "silly_completion": "Tiny socks marched across the counter singing.",
        "positive_completion": "The barista laughed kindly and cleaned it carefully.",
        "setup_anchor": "espresso machine",
        "resolution_keyword": "pressure",
        "joke_markers": "pressure|espresso|under",
        "silly_markers": "tiny socks|singing",
        "surprise_markers": "comet|milk pitcher",
        "positive_markers": "kindly|cleaned",
        "note": "Tier A smoke fallback, not science data.",
    },
    {
        "item_id": "fallback_cafe_02",
        "family": "script_violation",
        "setup": "The traffic light started taking improv classes.",
        "joke_completion": "It wanted to stop being so predictable.",
        "literal_completion": "It was installed near the theater school.",
        "surprise_completion": "A submarine surfaced in the crosswalk.",
        "silly_completion": "A llama in a cape bowed to a soup can.",
        "positive_completion": "Drivers stayed calm and the city improved safety signs.",
        "setup_anchor": "traffic light",
        "resolution_keyword": "stop",
        "joke_markers": "stop|predictable|traffic light",
        "silly_markers": "llama|cape|soup",
        "surprise_markers": "submarine|crosswalk",
        "positive_markers": "calm|safety",
        "note": "Tier A smoke fallback, not science data.",
    },
)


def item_from_mapping(row: Mapping[str, Any]) -> HumorItem:
    values: dict[str, str] = {}
    for field in dataclasses.fields(HumorItem):
        values[field.name] = str(row.get(field.name, "") or "").strip()
    return HumorItem(**values)


def load_rows_from_path(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Custom Lab 18 prompt file does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("items") or payload.get("rows") or []
        if not isinstance(payload, list):
            raise ValueError("Custom Lab 18 JSON must be a list or an object with an items list.")
        return [dict(row) for row in payload]
    if path.suffix.lower() in {".csv", ".tsv"}:
        dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
        with path.open(newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f, dialect=dialect)]
    raise ValueError("Custom Lab 18 prompt set must be .csv, .tsv, or .json.")


def validate_items(items: Sequence[HumorItem]) -> tuple[list[HumorItem], list[dict[str, Any]]]:
    seen: set[str] = set()
    valid: list[HumorItem] = []
    report: list[dict[str, Any]] = []
    for item in items:
        reasons: list[str] = []
        if not item.item_id:
            reasons.append("missing_item_id")
        if item.item_id in seen:
            reasons.append("duplicate_item_id")
        if not item.family:
            reasons.append("missing_family")
        if not item.setup:
            reasons.append("missing_setup")
        for condition in CONDITIONS:
            if not completion_for(item, condition):
                reasons.append(f"missing_{condition}_completion")
        for condition in CONDITIONS:
            if completion_for(item, condition).strip() == item.setup.strip():
                reasons.append(f"{condition}_completion_equals_setup")
        ok = not reasons
        if ok:
            valid.append(item)
            seen.add(item.item_id)
        report.append({
            "item_id": item.item_id,
            "family": item.family,
            "valid": ok,
            "drop_reason": ";".join(reasons),
            "setup_chars": len(item.setup),
            "joke_chars": len(item.joke_completion),
            "literal_chars": len(item.literal_completion),
            "surprise_chars": len(item.surprise_completion),
            "silly_chars": len(item.silly_completion),
            "positive_chars": len(item.positive_completion),
            "has_setup_anchor": bool(item.setup_anchor),
            "has_resolution_keyword": bool(item.resolution_keyword),
        })
    return valid, report


def load_items(args: Any) -> tuple[list[HumorItem], dict[str, Any], list[dict[str, Any]]]:
    prompt_set = str(getattr(args, "prompt_set", "small"))
    corpus_path_arg = str(getattr(args, "corpus_path", "") or "").strip()
    source_path: pathlib.Path | None = None
    source_kind = "frozen_csv"
    used_fallback = False

    if corpus_path_arg:
        source_path = as_path(corpus_path_arg)
        raw_rows = load_rows_from_path(source_path)
        source_kind = "custom_file"
    elif prompt_set in PROMPT_SET_FAMILY_CAPS:
        frozen_path = bench.COURSE_ROOT / "data" / DATA_FILE
        if frozen_path.exists():
            source_path = frozen_path
            raw_rows = load_rows_from_path(frozen_path)
        else:
            if prompt_set == "full" and str(getattr(args, "tier", "")) != "a":
                raise RuntimeError(f"Frozen dataset missing for science run: {frozen_path}")
            raw_rows = list(FALLBACK_ROWS)
            source_kind = "built_in_smoke_fallback"
            used_fallback = True
    else:
        source_path = as_path(prompt_set)
        raw_rows = load_rows_from_path(source_path)
        source_kind = "custom_prompt_set"

    raw_items = [item_from_mapping(row) for row in raw_rows]
    valid_items, validation_rows = validate_items(raw_items)
    if len(valid_items) < 2:
        raise RuntimeError("Lab 18 needs at least two valid items after validation.")

    if prompt_set in PROMPT_SET_FAMILY_CAPS:
        cap = PROMPT_SET_FAMILY_CAPS[prompt_set]
    else:
        cap = 0
    max_examples = int(getattr(args, "max_examples", -1) or -1)
    if max_examples > 0:
        cap = max_examples

    by_family: dict[str, list[HumorItem]] = defaultdict(list)
    for item in valid_items:
        by_family[item.family].append(item)

    selected: list[HumorItem] = []
    for family, rows in sorted(by_family.items()):
        ranked = sorted(rows, key=lambda r: stable_hash_int(f"{family}:{r.item_id}"))
        selected.extend(ranked[:cap] if cap > 0 else ranked)
    selected = sorted(selected, key=lambda r: (r.family, r.item_id))

    small_family_counts = {
        family: sum(1 for row in selected if row.family == family)
        for family in sorted(by_family)
        if sum(1 for row in selected if row.family == family) < 2
    }
    if small_family_counts and prompt_set == "full" and not used_fallback:
        raise RuntimeError(
            "Lab 18 science runs need at least two rows per family for split hygiene; "
            f"small families: {small_family_counts}"
        )

    data_hash = None
    if source_path is not None and source_path.exists():
        data_hash = bench.sha256_file(source_path)
    info = {
        "data_file": str(source_path) if source_path is not None else "built_in_fallback_rows",
        "data_sha256": data_hash,
        "source_kind": source_kind,
        "used_smoke_fallback": used_fallback,
        "fallback_warning": (
            "Built-in rows prove plumbing only. They are not frozen science data."
            if used_fallback else ""
        ),
        "prompt_set": prompt_set,
        "family_cap": cap,
        "n_raw_rows": len(raw_rows),
        "n_valid_rows": len(valid_items),
        "n_rows": len(selected),
        "families": sorted({row.family for row in selected}),
        "counts_by_family": {
            family: sum(1 for row in selected if row.family == family)
            for family in sorted({row.family for row in selected})
        },
        "conditions": list(CONDITIONS),
        "required_schema": [field.name for field in dataclasses.fields(HumorItem)],
        "selection_rule": "deterministic per-family cap by stable hash; full keeps all valid rows",
        "science_status": "smoke_only" if used_fallback else "frozen_or_custom_data",
    }
    return selected, info, validation_rows


# ---------------------------------------------------------------------------
# Prompt rendering and exact-chat instrumentation
# ---------------------------------------------------------------------------


def render_chat(bundle: bench.ModelBundle, user_message: str, *, system: str | None = SYSTEM_PROMPT) -> str:
    return bench.apply_chat_template(
        bundle,
        user_message,
        system=system,
        add_generation_prompt=True,
    )


def completion_for(item: HumorItem, condition: str) -> str:
    key = f"{condition}_completion"
    return getattr(item, key)


def contrast_message(item: HumorItem, condition: str) -> str:
    return (
        "Read this setup and ending as a compact text-analysis example.\n"
        f"Setup: {item.setup}\n"
        f"Ending: {completion_for(item, condition)}\n"
        "Reply with exactly one word: noted."
    )


def setup_only_message(item: HumorItem) -> str:
    return (
        "Read this setup before any ending is supplied.\n"
        f"Setup: {item.setup}\n"
        "Reply with exactly one word: noted."
    )


def generation_message(item: HumorItem) -> str:
    return (
        "Write one short, original ending for this setup. Keep it under 18 words.\n"
        f"Setup: {item.setup}\n"
        "Ending:"
    )


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_exact_chat_hook_parity(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    rendered_prompt: str,
) -> Any:
    """Hook parity for already-rendered chat prompts.

    The bench's generic hook parity uses tokenizer defaults. Chat labs must
    tokenize rendered prompts with add_special_tokens=False, because generation
    and activation capture operate on exactly that rendered string.
    """
    block_outputs: dict[int, Any] = {}

    def make_hook(idx: int):
        def hook(module: Any, hook_args: tuple, output: Any) -> None:
            out = output[0] if isinstance(output, tuple) else output
            block_outputs[idx] = bench.tensor_cpu_float(out)

        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(bundle.blocks)]
    try:
        capture = bench.run_with_residual_cache(bundle, rendered_prompt, add_special_tokens=False)
    finally:
        for handle in handles:
            handle.remove()

    rows: list[dict[str, Any]] = []
    max_diff = 0.0
    compared = 0
    missing: list[int] = []
    for layer in range(bundle.anatomy.n_layers):
        if layer not in block_outputs:
            missing.append(layer)
            continue
        hook_out = block_outputs[layer][0]
        expected = capture.streams[layer + 1]
        diff = (hook_out - expected).abs()
        layer_max = float(diff.max())
        max_diff = max(max_diff, layer_max)
        compared += 1
        rows.append({
            "layer": layer,
            "stream_depth_compared": layer + 1,
            "max_abs_diff": layer_max,
            "mean_abs_diff": float(diff.mean()),
            "ok_at_tolerance": layer_max <= float(getattr(ctx.args, "hook_tolerance", 0.0)),
        })
    by_layer_path = ctx.path("diagnostics", "exact_chat_hook_parity_by_layer.csv")
    bench.write_csv_with_context(ctx, by_layer_path, rows)
    ctx.register_artifact(by_layer_path, "diagnostic", "Exact rendered-chat hook parity by layer.")

    ok = (not missing) and compared == bundle.anatomy.n_layers and max_diff <= float(getattr(ctx.args, "hook_tolerance", 0.0))
    result = {
        "prompt_hash": prompt_hash(rendered_prompt),
        "add_special_tokens": False,
        "blocks_compared": compared,
        "n_layers": bundle.anatomy.n_layers,
        "missing_layers": missing,
        "max_abs_diff": max_diff,
        "tolerance": float(getattr(ctx.args, "hook_tolerance", 0.0)),
        "ok": bool(ok),
        "allow_hook_mismatch": bool(getattr(ctx.args, "allow_hook_mismatch", False)),
        "why_local_check_exists": (
            "Lab 18 measures chat-templated prompts exactly as generation sees them. "
            "The rendered string is tokenized with add_special_tokens=False."
        ),
    }
    path = ctx.path("diagnostics", "exact_chat_hook_parity.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Summary of exact rendered-chat hook parity.")
    print(f"[lab18] exact chat hook parity: {'OK' if ok else 'MISMATCH'} (max |diff|={max_diff:g})")
    if not ok and not bool(getattr(ctx.args, "allow_hook_mismatch", False)):
        raise RuntimeError("Exact chat hook parity failed; see diagnostics/exact_chat_hook_parity*." )
    return capture


# ---------------------------------------------------------------------------
# Splits and feature capture
# ---------------------------------------------------------------------------


def split_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "train" if bool(value) else REPORT_SPLIT


def make_split(items: Sequence[HumorItem], seed: int) -> dict[str, str]:
    split: dict[str, str] = {}
    by_family: dict[str, list[HumorItem]] = defaultdict(list)
    for item in items:
        by_family[item.family].append(item)
    for family, rows in by_family.items():
        ranked = sorted(rows, key=lambda r: stable_hash_int(f"{seed}:{family}:{r.item_id}"))
        if len(ranked) == 1:
            split[ranked[0].item_id] = "train"
            continue
        if len(ranked) == 2:
            split[ranked[0].item_id] = "train"
            split[ranked[1].item_id] = REPORT_SPLIT
            continue
        if len(ranked) <= 5:
            n_train = len(ranked) - 2
            n_dev = 1
        else:
            n_train = max(1, int(round(0.60 * len(ranked))))
            n_dev = max(1, int(round(0.20 * len(ranked))))
            if n_train + n_dev >= len(ranked):
                n_dev = 1
                n_train = len(ranked) - 2
        for idx, row in enumerate(ranked):
            if idx < n_train:
                split[row.item_id] = "train"
            elif idx < n_train + n_dev:
                split[row.item_id] = "dev"
            else:
                split[row.item_id] = REPORT_SPLIT
    return split


def split_rows(items: Sequence[HumorItem], split: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    by_family: dict[str, list[HumorItem]] = defaultdict(list)
    for item in items:
        by_family[item.family].append(item)
    for item in items:
        rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "split": split_name(split[item.item_id]),
            "family_n_rows": len(by_family[item.family]),
            "setup_excerpt": item.setup[:120],
            "note": item.note,
        })
    return rows


def split_balance_rows(items: Sequence[HumorItem], split: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for family in sorted({item.family for item in items}):
        fam = [item for item in items if item.family == family]
        rows.append({
            "family": family,
            "n_rows": len(fam),
            "n_train": sum(1 for item in fam if split_name(split[item.item_id]) == "train"),
            "n_dev": sum(1 for item in fam if split_name(split[item.item_id]) == "dev"),
            "n_test": sum(1 for item in fam if split_name(split[item.item_id]) == REPORT_SPLIT),
            "has_dev": any(split_name(split[item.item_id]) == "dev" for item in fam),
            "has_test": any(split_name(split[item.item_id]) == REPORT_SPLIT for item in fam),
        })
    rows.append({
        "family": "ALL",
        "n_rows": len(items),
        "n_train": sum(1 for item in items if split_name(split[item.item_id]) == "train"),
        "n_dev": sum(1 for item in items if split_name(split[item.item_id]) == "dev"),
        "n_test": sum(1 for item in items if split_name(split[item.item_id]) == REPORT_SPLIT),
        "has_dev": any(split_name(split[item.item_id]) == "dev" for item in items),
        "has_test": any(split_name(split[item.item_id]) == REPORT_SPLIT for item in items),
    })
    return rows


def cache_features(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    items: Sequence[HumorItem],
) -> tuple[Any, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    import torch

    rows = []
    stacked = []
    features: dict[str, dict[str, Any]] = {}
    phase_features: dict[str, dict[str, Any]] = {}
    report_every = max(1, len(items) // 4)
    for i, item in enumerate(items):
        features[item.item_id] = {}
        for condition in CONDITIONS:
            user_message = contrast_message(item, condition)
            prompt = render_chat(bundle, user_message)
            cap = bench.run_with_residual_cache(bundle, prompt, add_special_tokens=False)
            ending = completion_for(item, condition)
            ending_idxs, ending_method = token_span_indices(bundle, prompt, ending)
            resolution_idxs, resolution_method = token_span_indices(bundle, prompt, item.resolution_keyword or ending)
            query_idxs = resolution_idxs or ending_idxs
            if query_idxs:
                read_idx = max(query_idxs)
                read_site = "resolution_token" if resolution_idxs else "ending_last_token"
            else:
                read_idx = len(cap.input_ids) - 1
                read_site = "final_prompt_token_fallback"
            streams = cap.streams[:, read_idx, :]
            features[item.item_id][condition] = streams
            stacked.append(streams)
            rows.append({
                "item_id": item.item_id,
                "family": item.family,
                "condition": condition,
                "prompt_hash": prompt_hash(prompt),
                "prompt_tokens": len(cap.input_ids),
                "read_site": read_site,
                "read_token_index": read_idx,
                "read_token_text": cap.tokens_text[read_idx] if 0 <= read_idx < len(cap.tokens_text) else "",
                "ending_span_method": ending_method,
                "resolution_span_method": resolution_method,
                "n_ending_tokens": len(ending_idxs),
                "n_resolution_tokens": len(resolution_idxs),
                "final_token_index": len(cap.input_ids) - 1,
                "final_token_text": cap.tokens_text[-1] if cap.tokens_text else "",
                "last_8_tokens": "|".join(cap.tokens_text[-8:]),
                "user_message_excerpt": user_message[:180].replace("\n", " "),
            })
        setup_prompt = render_chat(bundle, setup_only_message(item))
        setup_cap = bench.run_with_residual_cache(bundle, setup_prompt, add_special_tokens=False)
        setup_idxs, setup_method = token_span_indices(bundle, setup_prompt, item.setup)
        setup_idx = max(setup_idxs) if setup_idxs else len(setup_cap.input_ids) - 1
        setup_streams = setup_cap.streams[:, setup_idx, :]
        phase_features[item.item_id] = {
            "setup": setup_streams,
            "joke": features[item.item_id]["joke"],
        }
        stacked.append(setup_streams)
        rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "condition": "setup_only",
            "prompt_hash": prompt_hash(setup_prompt),
            "prompt_tokens": len(setup_cap.input_ids),
            "read_site": "setup_last_token" if setup_idxs else "final_prompt_token_fallback",
            "read_token_index": setup_idx,
            "read_token_text": setup_cap.tokens_text[setup_idx] if 0 <= setup_idx < len(setup_cap.tokens_text) else "",
            "setup_span_method": setup_method,
            "n_setup_tokens": len(setup_idxs),
            "final_token_index": len(setup_cap.input_ids) - 1,
            "final_token_text": setup_cap.tokens_text[-1] if setup_cap.tokens_text else "",
            "last_8_tokens": "|".join(setup_cap.tokens_text[-8:]),
            "user_message_excerpt": setup_only_message(item)[:180].replace("\n", " "),
        })
        if (i + 1) % report_every == 0:
            print(f"[lab18] cached humor/control features for {i + 1}/{len(items)} rows")

    path = ctx.path("diagnostics", "prompt_render_audit.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Rendered chat prompt hashes, token counts, and extraction-token sites.")
    return torch.stack(stacked), features, phase_features


# ---------------------------------------------------------------------------
# Directions and probe sweeps
# ---------------------------------------------------------------------------


def train_rows(items: Sequence[HumorItem], split: Mapping[str, Any]) -> list[HumorItem]:
    return [item for item in items if split_name(split[item.item_id]) == "train"]


def rows_for_split(items: Sequence[HumorItem], split: Mapping[str, Any], name: str) -> list[HumorItem]:
    return [item for item in items if split_name(split[item.item_id]) == name]


def dev_rows(items: Sequence[HumorItem], split: Mapping[str, Any]) -> list[HumorItem]:
    return rows_for_split(items, split, "dev")


def eval_rows(items: Sequence[HumorItem], split: Mapping[str, Any]) -> list[HumorItem]:
    rows = rows_for_split(items, split, REPORT_SPLIT)
    if rows:
        return rows
    rows = dev_rows(items, split)
    if rows:
        return rows
    rows = [item for item in items if split_name(split[item.item_id]) != "train"]
    return rows if rows else list(items)


def diff_vector(
    item: HumorItem,
    features: Mapping[str, Mapping[str, Any]],
    depth: int,
    name: str,
) -> Any:
    import torch

    if name == "joke_structure":
        control_mean = torch.stack([features[item.item_id][c][depth] for c in CONTROL_CONDITIONS]).mean(dim=0)
        return features[item.item_id]["joke"][depth] - control_mean
    if name == "surprise":
        return features[item.item_id]["surprise"][depth] - features[item.item_id]["literal"][depth]
    if name == "silly":
        return features[item.item_id]["silly"][depth] - features[item.item_id]["literal"][depth]
    if name == "positive":
        return features[item.item_id]["positive"][depth] - features[item.item_id]["literal"][depth]
    raise ValueError(name)


def fit_direction(
    rows: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    depth: int,
    name: str,
    *,
    sign_seed: int | None = None,
) -> Any | None:
    import torch

    diffs = []
    for item in rows:
        diff = diff_vector(item, features, depth, name)
        if sign_seed is not None and stable_hash_int(f"{sign_seed}:{name}:{item.item_id}") % 2:
            diff = -diff
        diffs.append(diff)
    if not diffs:
        return None
    return maybe_unit(torch.stack(diffs).mean(dim=0))


def projection_scores(
    rows: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    direction: Any,
    depth: int,
) -> tuple[list[float], list[float]]:
    pos = [float(features[item.item_id]["joke"][depth] @ direction) for item in rows]
    neg = [
        float(features[item.item_id][condition][depth] @ direction)
        for item in rows
        for condition in CONTROL_CONDITIONS
    ]
    return pos, neg


def orient_direction_on_rows(
    direction: Any,
    rows: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    depth: int,
) -> Any:
    pos, neg = projection_scores(rows, features, direction, depth)
    if pos and neg and safe_fmean(pos) < safe_fmean(neg):
        return -direction
    return direction


def evaluate_direction(
    rows: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    direction: Any,
    depth: int,
) -> dict[str, Any]:
    pos, neg = projection_scores(rows, features, direction, depth)
    auc = auc_from_scores(pos, neg)
    return {
        "auc": auc,
        "mean_joke_projection": safe_fmean(pos),
        "mean_control_projection": safe_fmean(neg),
        "projection_gap": safe_fmean(pos) - safe_fmean(neg),
        "n_eval_jokes": len(pos),
        "n_eval_controls": len(neg),
    }


def train_selection_auc(
    train_items: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    depth: int,
    d_model: int,
    seed: int,
    kind: str,
    rep: int = 0,
) -> tuple[dict[str, Any], str]:
    """Train-only depth-selection score.

    With enough training rows, each training item is scored by a direction fit
    without that item. Tiny smoke runs fall back to in-sample selection and
    record that mode explicitly.
    """
    if len(train_items) >= 3 and kind in {"real", "shuffled_sign"}:
        pos: list[float] = []
        neg: list[float] = []
        for heldout in train_items:
            fit_rows_ = [item for item in train_items if item.item_id != heldout.item_id]
            sign_seed = seed + 100_000 * (rep + 1) + depth if kind == "shuffled_sign" else None
            direction = fit_direction(fit_rows_, features, depth, "joke_structure", sign_seed=sign_seed)
            if direction is None:
                continue
            direction = orient_direction_on_rows(direction, fit_rows_, features, depth)
            p, n = projection_scores([heldout], features, direction, depth)
            pos.extend(p)
            neg.extend(n)
        stats = {
            "auc": auc_from_scores(pos, neg),
            "mean_joke_projection": safe_fmean(pos),
            "mean_control_projection": safe_fmean(neg),
            "projection_gap": safe_fmean(pos) - safe_fmean(neg),
            "n_eval_jokes": len(pos),
            "n_eval_controls": len(neg),
        }
        return stats, "leave_one_train_item_out"

    if kind == "real":
        direction = fit_direction(train_items, features, depth, "joke_structure")
    elif kind == "shuffled_sign":
        direction = fit_direction(train_items, features, depth, "joke_structure", sign_seed=seed + rep * 997 + depth)
    elif kind == "random_oriented":
        direction = random_unit(d_model, seed + 9901 * (rep + 1) + depth)
    else:
        raise ValueError(kind)
    if direction is None:
        return {"auc": float("nan"), "mean_joke_projection": float("nan"), "mean_control_projection": float("nan"), "projection_gap": float("nan"), "n_eval_jokes": 0, "n_eval_controls": 0}, "no_direction"
    direction = orient_direction_on_rows(direction, train_items, features, depth)
    return evaluate_direction(train_items, features, direction, depth), "in_sample_small_n"


def summarize_null_stats(stats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "auc": safe_fmean([float(s["auc"]) for s in stats]),
        "auc_sd": safe_stdev([float(s["auc"]) for s in stats]),
        "mean_joke_projection": safe_fmean([float(s["mean_joke_projection"]) for s in stats]),
        "mean_control_projection": safe_fmean([float(s["mean_control_projection"]) for s in stats]),
        "projection_gap": safe_fmean([float(s["projection_gap"]) for s in stats]),
        "n_eval_jokes": stats[0].get("n_eval_jokes", 0) if stats else 0,
        "n_eval_controls": stats[0].get("n_eval_controls", 0) if stats else 0,
    }


def probe_row(depth: int, split_scored: str, kind: str, rep: Any, stats: Mapping[str, Any]) -> dict[str, Any]:
    auc = float(stats.get("auc", float("nan")))
    return {
        "probe": "joke_structure_vs_literal_surprise_silly_positive",
        "depth": depth,
        "split_scored": split_scored,
        "direction_kind": kind,
        "control_rep": rep,
        "auc": rounded(auc),
        "auc_sd": rounded(stats.get("auc_sd", float("nan"))),
        "selectivity_vs_chance": rounded(auc - 0.5),
        "mean_joke_projection": rounded(stats.get("mean_joke_projection", float("nan"))),
        "mean_control_projection": rounded(stats.get("mean_control_projection", float("nan"))),
        "projection_gap": rounded(stats.get("projection_gap", float("nan"))),
        "n_eval_jokes": stats.get("n_eval_jokes", 0),
        "n_eval_controls": stats.get("n_eval_controls", 0),
    }


def run_probe_sweep(
    items: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, Any],
    seed: int,
    d_model: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    n_depths = next(iter(next(iter(features.values())).values())).shape[0]
    fit_rows_ = train_rows(items, split)
    split_score_rows = {"dev": dev_rows(items, split), REPORT_SPLIT: eval_rows(items, split)}
    report: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for depth in range(1, n_depths):
        real_sel, mode = train_selection_auc(fit_rows_, features, depth, d_model, seed, "real")
        shuffled_sel_stats = [
            train_selection_auc(fit_rows_, features, depth, d_model, seed, "shuffled_sign", rep=i)[0]
            for i in range(N_NULL_REPS)
        ]
        random_sel_stats = [
            train_selection_auc(fit_rows_, features, depth, d_model, seed, "random_oriented", rep=i)[0]
            for i in range(N_NULL_REPS)
        ]
        shuffled_sel = summarize_null_stats(shuffled_sel_stats)
        random_sel = summarize_null_stats(random_sel_stats)
        selection_score = float(real_sel["auc"]) - max(0.5, float(shuffled_sel["auc"]), float(random_sel["auc"]))
        selection_rows.append({
            "depth": depth,
            "probe_split": "train_loo",
            "selection_mode": mode,
            "train_real_auc": rounded(real_sel["auc"]),
            "train_shuffled_mean_auc": rounded(shuffled_sel["auc"]),
            "train_random_mean_auc": rounded(random_sel["auc"]),
            "train_control_adjusted_score": rounded(selection_score),
            "mean_real_auc": rounded(real_sel["auc"]),
            "mean_shuffled_auc": rounded(shuffled_sel["auc"]),
            "mean_random_auc": rounded(random_sel["auc"]),
            "control_adjusted_score": rounded(selection_score),
            "n_train_items": len(fit_rows_),
            "n_eval_items": sum(len(rows) for rows in split_score_rows.values()),
            "note": "Train leave-one-out audit; dev selects depth when available and test is report-only.",
        })

        real = fit_direction(fit_rows_, features, depth, "joke_structure")
        if real is not None:
            real = orient_direction_on_rows(real, fit_rows_, features, depth)
            for split_key, score_rows in split_score_rows.items():
                if score_rows:
                    report.append(probe_row(depth, split_key, "real", "", evaluate_direction(score_rows, features, real, depth)))

        split_selection_stats: dict[str, dict[str, Any]] = {}
        for split_key, score_rows in split_score_rows.items():
            if not score_rows:
                continue
            shuffled_eval_stats = []
            for rep in range(N_NULL_REPS):
                direction = fit_direction(fit_rows_, features, depth, "joke_structure", sign_seed=seed + rep * 9973 + depth)
                if direction is None:
                    continue
                direction = orient_direction_on_rows(direction, fit_rows_, features, depth)
                stats = evaluate_direction(score_rows, features, direction, depth)
                shuffled_eval_stats.append(stats)
                report.append(probe_row(depth, split_key, "shuffled_sign", rep, stats))
            shuffled_eval = summarize_null_stats(shuffled_eval_stats) if shuffled_eval_stats else {}
            if shuffled_eval_stats:
                report.append(probe_row(depth, split_key, "shuffled_sign_mean", "mean", shuffled_eval))

            random_eval_stats = []
            for rep in range(N_NULL_REPS):
                direction = random_unit(d_model, seed + rep * 7919 + depth)
                direction = orient_direction_on_rows(direction, fit_rows_, features, depth)
                stats = evaluate_direction(score_rows, features, direction, depth)
                random_eval_stats.append(stats)
                report.append(probe_row(depth, split_key, "random_oriented", rep, stats))
            random_eval = summarize_null_stats(random_eval_stats)
            report.append(probe_row(depth, split_key, "random_oriented_mean", "mean", random_eval))

            real_auc = evaluate_direction(score_rows, features, real, depth)["auc"] if real is not None else float("nan")
            shuf_auc = float(shuffled_eval.get("auc", float("nan")))
            rand_auc = float(random_eval.get("auc", float("nan")))
            control_score = real_auc - max(0.5, shuf_auc if math.isfinite(shuf_auc) else 0.5, rand_auc if math.isfinite(rand_auc) else 0.5)
            split_selection_stats[split_key] = {
                "real_auc": real_auc,
                "shuffled_auc": shuf_auc,
                "random_auc": rand_auc,
                "score": control_score,
                "n": len(score_rows),
            }
            selection_rows.append({
                "depth": depth,
                "probe_split": split_key,
                "selection_mode": "train_fit_scored_on_" + split_key,
                "train_real_auc": rounded(real_sel["auc"]),
                "train_shuffled_mean_auc": rounded(shuffled_sel["auc"]),
                "train_random_mean_auc": rounded(random_sel["auc"]),
                "train_control_adjusted_score": rounded(selection_score),
                "mean_real_auc": rounded(real_auc),
                "mean_shuffled_auc": rounded(shuf_auc),
                "mean_random_auc": rounded(rand_auc),
                "control_adjusted_score": rounded(control_score),
                "n_train_items": len(fit_rows_),
                "n_eval_items": len(score_rows),
                "note": "Dev rows select depth; test rows are report-only.",
            })

    def depth_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
        score = float(row.get("control_adjusted_score") or -999.0)
        auc = float(row.get("mean_real_auc") or -999.0)
        return (score, auc, int(row["depth"]))

    selection_candidates = [row for row in selection_rows if row.get("probe_split") == "dev" and isinstance(row.get("control_adjusted_score"), (int, float))]
    if not selection_candidates:
        selection_candidates = [row for row in selection_rows if row.get("probe_split") == "train_loo"]
    best_depth = int(max(selection_candidates, key=depth_key)["depth"])
    for row in selection_rows:
        row["selected_depth"] = int(row["depth"]) == best_depth
    return report, selection_rows, best_depth


def run_phase_probe(
    items: Sequence[HumorItem],
    phase_features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, Any],
    seed: int,
    d_model: int,
    depth: int,
) -> list[dict[str, Any]]:
    import torch

    fit_rows_ = train_rows(items, split)
    score_rows = eval_rows(items, split)
    diffs = [phase_features[item.item_id]["joke"][depth] - phase_features[item.item_id]["setup"][depth] for item in fit_rows_]
    if not diffs:
        return []
    rows = []
    real = maybe_unit(torch.stack(diffs).mean(dim=0))
    if real is not None:
        pos = [float(phase_features[item.item_id]["joke"][depth] @ real) for item in score_rows]
        neg = [float(phase_features[item.item_id]["setup"][depth] @ real) for item in score_rows]
        rows.append({
            "probe": "punchline_phase_full_joke_vs_setup_only",
            "depth": depth,
            "direction_kind": "real",
            "control_rep": "",
            "auc": rounded(auc_from_scores(pos, neg)),
            "selectivity_vs_chance": rounded(auc_from_scores(pos, neg) - 0.5),
            "mean_full_joke_projection": rounded(safe_fmean(pos)),
            "mean_setup_projection": rounded(safe_fmean(neg)),
            "projection_gap": rounded(safe_fmean(pos) - safe_fmean(neg)),
            "n_eval_pairs": len(score_rows),
        })

    null_rows: dict[str, list[dict[str, Any]]] = {"shuffled_sign": [], "random_oriented": []}
    for rep in range(N_NULL_REPS):
        shuffled_diffs = [
            (-diff if stable_hash_int(f"{seed}:phase:{rep}:{item.item_id}") % 2 else diff)
            for item, diff in zip(fit_rows_, diffs)
        ]
        shuffled = maybe_unit(torch.stack(shuffled_diffs).mean(dim=0))
        random = random_unit(d_model, seed + 4049 + rep * 101)
        for kind, direction in (("shuffled_sign", shuffled), ("random_oriented", random)):
            if direction is None:
                continue
            pos = [float(phase_features[item.item_id]["joke"][depth] @ direction) for item in score_rows]
            neg = [float(phase_features[item.item_id]["setup"][depth] @ direction) for item in score_rows]
            auc = auc_from_scores(pos, neg)
            out = {
                "probe": "punchline_phase_full_joke_vs_setup_only",
                "depth": depth,
                "direction_kind": kind,
                "control_rep": rep,
                "auc": rounded(auc),
                "selectivity_vs_chance": rounded(auc - 0.5),
                "mean_full_joke_projection": rounded(safe_fmean(pos)),
                "mean_setup_projection": rounded(safe_fmean(neg)),
                "projection_gap": rounded(safe_fmean(pos) - safe_fmean(neg)),
                "n_eval_pairs": len(score_rows),
            }
            rows.append(out)
            null_rows[kind].append(out)
    for kind, sub in null_rows.items():
        if not sub:
            continue
        aucs = [float(row["auc"]) for row in sub]
        rows.append({
            "probe": "punchline_phase_full_joke_vs_setup_only",
            "depth": depth,
            "direction_kind": f"{kind}_mean",
            "control_rep": "mean",
            "auc": rounded(safe_fmean(aucs)),
            "auc_sd": rounded(safe_stdev(aucs)),
            "selectivity_vs_chance": rounded(safe_fmean(aucs) - 0.5),
            "mean_full_joke_projection": rounded(safe_fmean([float(row["mean_full_joke_projection"]) for row in sub])),
            "mean_setup_projection": rounded(safe_fmean([float(row["mean_setup_projection"]) for row in sub])),
            "projection_gap": rounded(safe_fmean([float(row["projection_gap"]) for row in sub])),
            "n_eval_pairs": len(score_rows),
        })
    return rows


def family_heldout_probe_rows(
    items: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    depth: int,
    seed: int,
    d_model: int,
) -> list[dict[str, Any]]:
    """Train the joke-structure direction with one family held out.

    This tests whether the handle is a reusable joke-structure direction or a
    family-local lexicon trick. It is descriptive and small-n sensitive, but it
    prevents the ledger from quietly treating a within-family split as a fresh
    family result.
    """
    rows: list[dict[str, Any]] = []
    families = sorted({item.family for item in items})
    for heldout_family in families:
        fit_rows_ = [item for item in items if item.family != heldout_family]
        score_rows = [item for item in items if item.family == heldout_family]
        if len(fit_rows_) < 2 or not score_rows:
            continue
        real = fit_direction(fit_rows_, features, depth, "joke_structure")
        if real is not None:
            real = orient_direction_on_rows(real, fit_rows_, features, depth)
            stats = evaluate_direction(score_rows, features, real, depth)
            rows.append({
                "heldout_family": heldout_family,
                "depth": depth,
                "direction_kind": "real",
                "control_rep": "",
                "auc": rounded(stats["auc"]),
                "auc_sd": "",
                "projection_gap": rounded(stats["projection_gap"]),
                "n_eval_jokes": stats["n_eval_jokes"],
                "n_eval_controls": stats["n_eval_controls"],
                "train_families": ";".join(f for f in families if f != heldout_family),
            })
        shuffled_stats = []
        random_stats = []
        for rep in range(N_NULL_REPS):
            shuffled = fit_direction(fit_rows_, features, depth, "joke_structure", sign_seed=seed + 1237 * rep + stable_hash_int(heldout_family) % 10000)
            if shuffled is not None:
                shuffled = orient_direction_on_rows(shuffled, fit_rows_, features, depth)
                shuffled_stats.append(evaluate_direction(score_rows, features, shuffled, depth))
            random = random_unit(d_model, seed + 7919 * rep + stable_hash_int("random:" + heldout_family) % 10000)
            random = orient_direction_on_rows(random, fit_rows_, features, depth)
            random_stats.append(evaluate_direction(score_rows, features, random, depth))
        for kind, stats_list in (("shuffled_sign_mean", shuffled_stats), ("random_oriented_mean", random_stats)):
            if not stats_list:
                continue
            stats = summarize_null_stats(stats_list)
            rows.append({
                "heldout_family": heldout_family,
                "depth": depth,
                "direction_kind": kind,
                "control_rep": "mean",
                "auc": rounded(stats["auc"]),
                "auc_sd": rounded(stats["auc_sd"]),
                "projection_gap": rounded(stats["projection_gap"]),
                "n_eval_jokes": stats["n_eval_jokes"],
                "n_eval_controls": stats["n_eval_controls"],
                "train_families": ";".join(f for f in families if f != heldout_family),
            })
    return rows


# ---------------------------------------------------------------------------
# Surprisal and entropy
# ---------------------------------------------------------------------------


def target_surprisal_bits(
    bundle: bench.ModelBundle,
    prompt: str,
    target: str,
    *,
    resolution_keyword: str = "",
) -> dict[str, Any]:
    import torch

    tokenizer = bundle.tokenizer
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_text = target if target.startswith((" ", "\n", "\t")) else " " + target
    try:
        target_enc = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        target_enc = tokenizer(target_text, add_special_tokens=False)
        target_enc["offset_mapping"] = []
    target_ids = target_enc["input_ids"]
    offsets = target_enc.get("offset_mapping") or []
    if not prompt_ids or not target_ids:
        return {
            "target_tokens": len(target_ids),
            "mean_surprisal_bits": float("nan"),
            "total_surprisal_bits": float("nan"),
            "resolution_keyword_surprisal_bits": float("nan"),
            "resolution_keyword_tokens": 0,
        }
    ids = prompt_ids + target_ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=bundle.input_device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    log_probs = torch.log_softmax(out.logits[0].float(), dim=-1)
    losses = []
    start = len(prompt_ids)
    for i, tok_id in enumerate(target_ids):
        pos = start + i
        if pos == 0:
            continue
        losses.append(float(-log_probs[pos - 1, tok_id] / math.log(2.0)))

    keyword_losses: list[float] = []
    kw = resolution_keyword.strip().lower()
    if kw and offsets:
        target_low = target_text.lower()
        k0 = target_low.find(kw)
        if k0 >= 0:
            k1 = k0 + len(kw)
            for i, (a, b) in enumerate(offsets):
                if i < len(losses) and b > k0 and a < k1:
                    keyword_losses.append(losses[i])

    return {
        "target_tokens": len(target_ids),
        "mean_surprisal_bits": safe_fmean(losses),
        "total_surprisal_bits": sum(losses) if losses else float("nan"),
        "resolution_keyword_surprisal_bits": safe_fmean(keyword_losses),
        "resolution_keyword_tokens": len(keyword_losses),
        "tokenization_note": "target text encoded separately and appended to rendered chat prompt",
    }


def next_token_entropy_bits(bundle: bench.ModelBundle, prompt: str) -> float:
    import torch

    logits = bench.next_token_logits(bundle, prompt)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log2(probs.clamp_min(1e-45))
    return float(-(probs * log_probs).sum())


def run_surprisal_measurements(bundle: bench.ModelBundle, items: Sequence[HumorItem]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        setup_prompt = render_chat(
            bundle,
            "Read this setup and prepare for a short ending.\n"
            f"Setup: {item.setup}\n"
            "Ending:",
        )
        entropy = next_token_entropy_bits(bundle, setup_prompt)
        target_prompt = render_chat(
            bundle,
            "Complete this setup with the supplied ending.\n"
            f"Setup: {item.setup}\n"
            "Ending:",
        )
        for condition in CONDITIONS:
            stats = target_surprisal_bits(
                bundle,
                target_prompt,
                completion_for(item, condition),
                resolution_keyword=item.resolution_keyword if condition == "joke" else "",
            )
            rows.append({
                "item_id": item.item_id,
                "family": item.family,
                "condition": condition,
                "setup_next_token_entropy_bits": rounded(entropy),
                "target_tokens": stats["target_tokens"],
                "mean_surprisal_bits": rounded(stats["mean_surprisal_bits"]),
                "total_surprisal_bits": rounded(stats["total_surprisal_bits"]),
                "resolution_keyword": item.resolution_keyword if condition == "joke" else "",
                "resolution_keyword_surprisal_bits": rounded(stats["resolution_keyword_surprisal_bits"]),
                "resolution_keyword_tokens": stats["resolution_keyword_tokens"],
                "tokenization_note": stats["tokenization_note"],
            })
    return rows


def summarise_surprisal(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for condition in CONDITIONS:
        sub = [row for row in rows if row.get("condition") == condition]
        mean_vals = [float(row["mean_surprisal_bits"]) for row in sub if isinstance(row.get("mean_surprisal_bits"), (int, float))]
        total_vals = [float(row["total_surprisal_bits"]) for row in sub if isinstance(row.get("total_surprisal_bits"), (int, float))]
        entropy_vals = [float(row["setup_next_token_entropy_bits"]) for row in sub if isinstance(row.get("setup_next_token_entropy_bits"), (int, float))]
        out.append({
            "condition": condition,
            "n": len(sub),
            "mean_token_surprisal_bits": rounded(safe_fmean(mean_vals)),
            "sd_token_surprisal_bits": rounded(safe_stdev(mean_vals)),
            "mean_total_surprisal_bits": rounded(safe_fmean(total_vals)),
            "mean_setup_entropy_bits": rounded(safe_fmean(entropy_vals)),
        })
    by_cond = {
        row["condition"]: float(row["mean_token_surprisal_bits"])
        for row in out
        if isinstance(row.get("mean_token_surprisal_bits"), (int, float))
    }
    for row in out:
        row["minus_literal_mean_token_surprisal"] = rounded(by_cond.get(str(row["condition"]), float("nan")) - by_cond.get("literal", float("nan")))
        row["minus_joke_mean_token_surprisal"] = rounded(by_cond.get(str(row["condition"]), float("nan")) - by_cond.get("joke", float("nan")))
    return out


# ---------------------------------------------------------------------------
# Marker scoring and steering
# ---------------------------------------------------------------------------


def keyword_patterns(spec: str) -> list[str]:
    return [p.strip().lower() for p in str(spec).split("|") if p.strip()]


def marker_count(text: str, spec: str) -> int:
    low = text.lower()
    count = 0
    for pat in keyword_patterns(spec):
        count += len(re.findall(rf"(?<![a-z0-9]){re.escape(pat)}(?![a-z0-9])", low))
    return count


def word_stats(text: str) -> dict[str, Any]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if not words:
        return {"n_words": 0, "distinct_ratio": 0.0, "repetition_rate": 0.0}
    distinct = len(set(words)) / len(words)
    return {"n_words": len(words), "distinct_ratio": distinct, "repetition_rate": 1.0 - distinct}


def score_generation(item: HumorItem, text: str) -> dict[str, Any]:
    joke_specific = marker_count(text, item.joke_markers)
    joke_generic = marker_count(text, GENERIC_JOKE_MARKERS)
    silly = marker_count(text, item.silly_markers) + marker_count(text, GENERIC_SILLY_MARKERS)
    surprise = marker_count(text, item.surprise_markers) + marker_count(text, GENERIC_SURPRISE_MARKERS)
    positive = marker_count(text, item.positive_markers) + marker_count(text, GENERIC_POSITIVE_MARKERS)
    joke = joke_specific + joke_generic
    stats = word_stats(text)
    low = text.lower()
    return {
        "joke_marker_count": joke,
        "joke_specific_marker_count": joke_specific,
        "joke_generic_marker_count": joke_generic,
        "silly_marker_count": silly,
        "surprise_marker_count": surprise,
        "positive_marker_count": positive,
        "joke_vs_cheap_margin": joke - max(silly, surprise, positive),
        "cheap_marker_total": silly + surprise + positive,
        "contains_setup_anchor": int(bool(item.setup_anchor and item.setup_anchor.lower() in low)),
        "contains_resolution_keyword": int(bool(item.resolution_keyword and item.resolution_keyword.lower() in low)),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "n_words": stats["n_words"],
        "distinct_ratio": rounded(stats["distinct_ratio"]),
        "repetition_rate": rounded(stats["repetition_rate"]),
        "hand_label_funny_0_1": "",
        "hand_label_joke_structure_0_1": "",
        "hand_label_silly_0_1": "",
        "hand_label_surprising_0_1": "",
        "hand_label_positive_0_1": "",
        "hand_label_notes": "",
    }


def selected_eval_rows(items: Sequence[HumorItem], split: Mapping[str, Any]) -> list[HumorItem]:
    candidates = eval_rows(items, split)
    by_family: dict[str, list[HumorItem]] = defaultdict(list)
    for item in candidates:
        by_family[item.family].append(item)
    for family in by_family:
        by_family[family] = sorted(by_family[family], key=lambda r: stable_hash_int(f"steer:{r.family}:{r.item_id}"))
    picked: list[HumorItem] = []
    while len(picked) < min(MAX_STEERING_ITEMS, len(candidates)):
        moved = False
        for family in sorted(by_family):
            if by_family[family] and len(picked) < MAX_STEERING_ITEMS:
                picked.append(by_family[family].pop(0))
                moved = True
        if not moved:
            break
    return picked


def run_steering(
    bundle: bench.ModelBundle,
    items: Sequence[HumorItem],
    directions: Mapping[str, Any],
    shuffled_direction: Any,
    depth: int,
    d_model: int,
    seed: int,
    ref_norm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    injection_layer = max(0, depth - 1)
    prompts = [render_chat(bundle, generation_message(item)) for item in items]
    baseline_outs = bench.generate_continuous(
        bundle,
        prompts,
        MAX_NEW_TOKENS,
        max_concurrent=ENGINE_MAX_CONCURRENT,
        progress_label="lab18 steering baseline",
    )
    rows: list[dict[str, Any]] = []
    for item, text in zip(items, baseline_outs):
        rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "steering_condition": "baseline",
            "direction_name": "none",
            "dose_fraction": 0.0,
            "injection_layer": "",
            "stream_depth_source": depth,
            "steering_scale": 0.0,
            "generation": text,
            **score_generation(item, text),
        })

    random = random_unit(d_model, seed + 8803)
    conditions = [
        ("joke_structure_direction", "joke_structure", directions["joke_structure"], 1.0),
        ("opposite_joke_structure_direction", "joke_structure", directions["joke_structure"], -1.0),
        ("surprise_direction", "surprise", directions["surprise"], 1.0),
        ("silly_direction", "silly", directions["silly"], 1.0),
        ("positive_direction", "positive", directions["positive"], 1.0),
        ("shuffled_joke_direction", "shuffled_joke_structure", shuffled_direction, 1.0),
        ("random_direction", "random", random, 1.0),
    ]
    for condition, direction_name, vec, sign in conditions:
        for dose in STEERING_DOSES:
            abs_scale = sign * float(dose) * ref_norm
            outs = bench.generate_continuous(
                bundle,
                prompts,
                MAX_NEW_TOKENS,
                max_concurrent=ENGINE_MAX_CONCURRENT,
                progress_label=f"lab18 steering {condition} dose={dose}",
                steer=(injection_layer, vec, abs_scale),
            )
            for item, text in zip(items, outs):
                rows.append({
                    "item_id": item.item_id,
                    "family": item.family,
                    "steering_condition": condition,
                    "direction_name": direction_name,
                    "dose_fraction": dose,
                    "injection_layer": injection_layer,
                    "stream_depth_source": depth,
                    "steering_scale": rounded(abs_scale),
                    "generation": text,
                    **score_generation(item, text),
                })

    baseline = [row for row in rows if row["steering_condition"] == "baseline"]
    metric_keys = (
        "joke_marker_count",
        "joke_specific_marker_count",
        "joke_generic_marker_count",
        "silly_marker_count",
        "surprise_marker_count",
        "positive_marker_count",
        "joke_vs_cheap_margin",
        "cheap_marker_total",
        "contains_setup_anchor",
        "contains_resolution_keyword",
        "exclamation_count",
        "n_words",
        "distinct_ratio",
        "repetition_rate",
    )
    base_metrics = {
        key: safe_fmean([float(row[key]) for row in baseline])
        for key in metric_keys
    }
    effect_rows: list[dict[str, Any]] = []
    groups = sorted({(row["steering_condition"], row["dose_fraction"]) for row in rows})
    for condition, dose in groups:
        sub = [row for row in rows if row["steering_condition"] == condition and row["dose_fraction"] == dose]
        out: dict[str, Any] = {"steering_condition": condition, "dose_fraction": dose, "n": len(sub)}
        for key in metric_keys:
            mean_val = safe_fmean([float(row[key]) for row in sub])
            out[f"mean_{key}"] = rounded(mean_val)
            out[f"{key}_delta_vs_baseline"] = rounded(mean_val - base_metrics[key])
        effect_rows.append(out)

    by_condition_dose = {
        (row["steering_condition"], row["dose_fraction"]): float(row.get("joke_vs_cheap_margin_delta_vs_baseline", float("nan")))
        for row in effect_rows
    }
    cheap_names = ("surprise_direction", "silly_direction", "positive_direction", "random_direction", "shuffled_joke_direction")
    for row in effect_rows:
        dose = row["dose_fraction"]
        best_cheap = safe_fmean([], default=float("nan"))
        cheap_vals = [by_condition_dose.get((name, dose), float("nan")) for name in cheap_names]
        cheap_vals = [v for v in cheap_vals if math.isfinite(float(v))]
        if cheap_vals:
            best_cheap = max(cheap_vals)
        own = float(row.get("joke_vs_cheap_margin_delta_vs_baseline", float("nan")))
        row["joke_margin_delta_minus_best_cheap_same_dose"] = rounded(own - best_cheap)
    return rows, effect_rows


def write_generation_labeling_guide(ctx: bench.RunContext) -> None:
    lines = [
        "# Lab 18 Generation Labeling Guide",
        "",
        "The marker counts in `humor_steering_generations.csv` are a triage tool, not a judge.",
        "Fill the blank hand-label columns before making a claim about funniness.",
        "",
        "Suggested binary labels:",
        "",
        "- `hand_label_funny_0_1`: a human reader would plausibly call the ending funny.",
        "- `hand_label_joke_structure_0_1`: the ending has setup-dependent incongruity plus resolution, even if it is not funny.",
        "- `hand_label_silly_0_1`: the ending is random or whimsical without resolving the setup.",
        "- `hand_label_surprising_0_1`: the ending violates expectation, whether or not it resolves.",
        "- `hand_label_positive_0_1`: the ending is mainly warm, happy, reassuring, or kind.",
        "",
        "A strong Lab 18 steering result should increase joke structure without merely increasing silliness, raw surprise, exclamation marks, or positive tone.",
        "",
    ]
    path = ctx.path("tables", "generation_labeling_guide.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "guide", "Hand-labeling guide for Lab 18 steering generations.")


# ---------------------------------------------------------------------------
# Attention-to-setup diagnostics
# ---------------------------------------------------------------------------


def run_rendered_attention(bundle: bench.ModelBundle, prompt: str) -> RenderedAttention:
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)
    with torch.no_grad():
        out = bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
    if not out.attentions or any(a is None for a in out.attentions):
        raise RuntimeError(
            "The model returned no attention patterns. Lab 18 needs eager attention; "
            "set needs_eager in the registry or pass --attn-implementation eager."
        )
    ids = input_ids[0].detach().cpu().tolist()
    attentions = torch.stack([bench.tensor_cpu_float(a[0]) for a in out.attentions])
    return RenderedAttention(
        prompt=prompt,
        input_ids=ids,
        tokens_text=[tokenizer.decode([i]) for i in ids],
        attentions=attentions,
    )


def token_span_indices(bundle: bench.ModelBundle, text: str, substring: str) -> tuple[list[int], str]:
    if not substring:
        return [], "empty_substring"
    start = text.find(substring)
    if start < 0:
        return [], "substring_not_found"
    end = start + len(substring)
    tokenizer = bundle.tokenizer
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc.get("offset_mapping") or []
        ids = enc.get("input_ids") or []
        if len(offsets) == len(ids):
            idxs = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
            if idxs:
                return idxs, "offset_mapping"
    except Exception:
        pass

    full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    sub_variants = [substring, " " + substring, substring.strip()]
    for variant in sub_variants:
        sub_ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not sub_ids:
            continue
        for i in range(0, len(full_ids) - len(sub_ids) + 1):
            if full_ids[i:i + len(sub_ids)] == sub_ids:
                return list(range(i, i + len(sub_ids))), "token_subsequence"
    return [], "span_lookup_failed"


def attention_to_setup_rows(bundle: bench.ModelBundle, items: Sequence[HumorItem]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []
    for item in items[:MAX_ATTENTION_ITEMS]:
        for condition in CONDITIONS:
            message = contrast_message(item, condition)
            prompt = render_chat(bundle, message)
            att = run_rendered_attention(bundle, prompt)
            setup_idxs, setup_method = token_span_indices(bundle, prompt, item.setup)
            anchor_idxs, anchor_method = token_span_indices(bundle, prompt, item.setup_anchor or item.setup)
            ending = completion_for(item, condition)
            ending_idxs, ending_method = token_span_indices(bundle, prompt, ending)
            resolution = item.resolution_keyword or ending
            resolution_idxs, resolution_method = token_span_indices(bundle, prompt, resolution)
            query_idxs = resolution_idxs or ending_idxs
            span_rows.append({
                "item_id": item.item_id,
                "family": item.family,
                "condition": condition,
                "n_prompt_tokens": len(att.input_ids),
                "n_setup_tokens": len(setup_idxs),
                "n_anchor_tokens": len(anchor_idxs),
                "n_ending_tokens": len(ending_idxs),
                "n_resolution_tokens": len(resolution_idxs),
                "setup_method": setup_method,
                "anchor_method": anchor_method,
                "ending_method": ending_method,
                "resolution_method": resolution_method,
                "query_token_index": max(query_idxs) if query_idxs else "",
                "query_token_text": att.tokens_text[max(query_idxs)] if query_idxs and max(query_idxs) < len(att.tokens_text) else "",
            })
            if not setup_idxs or not query_idxs:
                rows.append({
                    "item_id": item.item_id,
                    "family": item.family,
                    "condition": condition,
                    "layer": "",
                    "mean_attention_to_setup": "",
                    "max_head_attention_to_setup": "",
                    "mean_attention_to_anchor": "",
                    "max_head_attention_to_anchor": "",
                    "n_setup_tokens": len(setup_idxs),
                    "n_anchor_tokens": len(anchor_idxs),
                    "query_token_index": "",
                    "query_token_text": "",
                    "span_method": f"setup={setup_method};anchor={anchor_method};ending={ending_method};resolution={resolution_method}",
                    "note": "span lookup failed; inspect diagnostics/attention_span_audit.csv",
                })
                continue
            query_idx = max(query_idxs)
            token_text = att.tokens_text[query_idx] if query_idx < len(att.tokens_text) else ""
            for layer in range(att.attentions.shape[0]):
                head_setup = att.attentions[layer, :, query_idx, setup_idxs].sum(dim=-1)
                if anchor_idxs:
                    head_anchor = att.attentions[layer, :, query_idx, anchor_idxs].sum(dim=-1)
                    mean_anchor = float(head_anchor.mean())
                    max_anchor = float(head_anchor.max())
                else:
                    mean_anchor = float("nan")
                    max_anchor = float("nan")
                rows.append({
                    "item_id": item.item_id,
                    "family": item.family,
                    "condition": condition,
                    "layer": layer,
                    "mean_attention_to_setup": rounded(float(head_setup.mean())),
                    "max_head_attention_to_setup": rounded(float(head_setup.max())),
                    "mean_attention_to_anchor": rounded(mean_anchor),
                    "max_head_attention_to_anchor": rounded(max_anchor),
                    "n_setup_tokens": len(setup_idxs),
                    "n_anchor_tokens": len(anchor_idxs),
                    "query_token_index": query_idx,
                    "query_token_text": token_text,
                    "span_method": f"setup={setup_method};anchor={anchor_method};ending={ending_method};resolution={resolution_method}",
                    "note": "attention from resolution/final ending token back to setup span",
                })
    return rows, span_rows


def attention_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    layers = sorted({int(row["layer"]) for row in rows if isinstance(row.get("layer"), int)})
    for condition in CONDITIONS:
        for layer in layers:
            sub = [row for row in rows if row.get("condition") == condition and row.get("layer") == layer]
            vals = [float(row["mean_attention_to_setup"]) for row in sub if isinstance(row.get("mean_attention_to_setup"), (int, float))]
            anchor_vals = [float(row["mean_attention_to_anchor"]) for row in sub if isinstance(row.get("mean_attention_to_anchor"), (int, float))]
            out.append({
                "condition": condition,
                "layer": layer,
                "n": len(vals),
                "mean_attention_to_setup": rounded(safe_fmean(vals)),
                "mean_attention_to_anchor": rounded(safe_fmean(anchor_vals)),
            })
    literal_by_layer = {
        int(row["layer"]): float(row["mean_attention_to_setup"])
        for row in out
        if row.get("condition") == "literal" and isinstance(row.get("mean_attention_to_setup"), (int, float))
    }
    for row in out:
        if isinstance(row.get("mean_attention_to_setup"), (int, float)):
            row["minus_literal_attention_to_setup"] = rounded(float(row["mean_attention_to_setup"]) - literal_by_layer.get(int(row["layer"]), float("nan")))
        else:
            row["minus_literal_attention_to_setup"] = ""
    return out


# ---------------------------------------------------------------------------
# Direction audits and projections
# ---------------------------------------------------------------------------


def direction_cosine_rows(directions: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = sorted(directions)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({
                "direction_a": a,
                "direction_b": b,
                "cosine": rounded(cosine(directions[a], directions[b])),
                "abs_cosine": rounded(abs(cosine(directions[a], directions[b]))),
            })
    return rows


def projection_audit_rows(
    items: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    directions: Mapping[str, Any],
    depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for item in items:
        for condition in CONDITIONS:
            for direction_name, direction in directions.items():
                rows.append({
                    "item_id": item.item_id,
                    "family": item.family,
                    "condition": condition,
                    "direction_name": direction_name,
                    "depth": depth,
                    "projection": rounded(float(features[item.item_id][condition][depth] @ direction)),
                })
    summary = []
    for condition in CONDITIONS:
        for direction_name in sorted(directions):
            vals = [
                float(row["projection"]) for row in rows
                if row["condition"] == condition and row["direction_name"] == direction_name
                and isinstance(row.get("projection"), (int, float))
            ]
            summary.append({
                "condition": condition,
                "direction_name": direction_name,
                "depth": depth,
                "mean_projection": rounded(safe_fmean(vals)),
                "sd_projection": rounded(safe_stdev(vals)),
                "n": len(vals),
            })
    literal_means = {
        row["direction_name"]: float(row["mean_projection"])
        for row in summary
        if row["condition"] == "literal" and isinstance(row.get("mean_projection"), (int, float))
    }
    for row in summary:
        if isinstance(row.get("mean_projection"), (int, float)):
            row["minus_literal_mean_projection"] = rounded(float(row["mean_projection"]) - literal_means.get(row["direction_name"], float("nan")))
    return rows, summary


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def metric_at(rows: Sequence[Mapping[str, Any]], kind: str, depth: int, key: str = "auc", split_scored: str = REPORT_SPLIT) -> float:
    vals = [
        float(row[key]) for row in rows
        if row.get("direction_kind") == kind
        and row.get("split_scored") == split_scored
        and int(row.get("depth", -1)) == int(depth)
        and isinstance(row.get(key), (int, float))
    ]
    return safe_fmean(vals)


def effect_delta(rows: Sequence[Mapping[str, Any]], condition: str, key: str, dose: float | None = None) -> float:
    vals = []
    for row in rows:
        if row.get("steering_condition") != condition:
            continue
        if dose is not None and abs(float(row.get("dose_fraction", -999.0)) - dose) > 1e-9:
            continue
        if isinstance(row.get(key), (int, float)):
            vals.append(float(row[key]))
    return safe_fmean(vals)


def max_abs_cosine_with_controls(metrics: Mapping[str, Any]) -> float:
    vals = []
    for key in ("joke_surprise_cosine", "joke_silly_cosine", "joke_positive_cosine"):
        try:
            val = abs(float(metrics.get(key)))
        except Exception:
            continue
        if math.isfinite(val):
            vals.append(val)
    return max(vals) if vals else float("nan")


def verdict_from_metrics(metrics: Mapping[str, Any]) -> str:
    try:
        selectivity = float(metrics.get("real_selectivity_vs_best_null") or float("nan"))
    except Exception:
        selectivity = float("nan")
    try:
        cosmax = max_abs_cosine_with_controls(metrics)
    except Exception:
        cosmax = float("nan")
    try:
        steer_gap = float(metrics.get("joke_steering_specificity_gap") or float("nan"))
    except Exception:
        steer_gap = float("nan")
    if math.isfinite(selectivity) and selectivity > 0.10 and math.isfinite(cosmax) and cosmax < 0.75:
        if math.isfinite(steer_gap) and steer_gap > 0.20:
            return "validated_joke_structure_handle_with_cautious_steering"
        return "decodable_joke_structure_handle_not_causally_separated_by_this_run"
    if math.isfinite(selectivity) and selectivity > 0.05:
        return "weak_or_confounded_joke_register_handle"
    return "not_validated_by_controls"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_surprisal(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.8, 5.1))
    xs = list(range(len(CONDITIONS)))
    means = [
        safe_fmean([
            float(row["mean_surprisal_bits"]) for row in rows
            if row.get("condition") == condition and isinstance(row.get("mean_surprisal_bits"), (int, float))
        ])
        for condition in CONDITIONS
    ]
    ax.bar(xs, means)
    ax.set_xticks(xs)
    ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
    bench.style_ax(ax, title="Ending surprisal by condition", ylabel="mean target-token surprisal (bits)", legend=False)
    bench.save_figure(ctx, fig, "humor_surprisal_trajectories.png", "Mean target-token surprisal for joke and control endings.")


def plot_probe(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]], best_depth: int) -> None:
    fig, ax = bench.new_figure(figsize=(9.0, 5.2))
    kinds = ("real", "shuffled_sign_mean", "random_oriented_mean")
    for kind in kinds:
        sub = [row for row in rows if row.get("direction_kind") == kind and row.get("split_scored") == REPORT_SPLIT]
        if not sub:
            continue
        xs = [int(row["depth"]) for row in sub]
        ys = [float(row["auc"]) for row in sub]
        ax.plot(xs, ys, marker="o", label=kind)
    ax.axhline(0.5, linestyle=":", linewidth=1.1)
    ax.axvline(best_depth, linestyle="--", linewidth=1.0, label=f"selected depth {best_depth}")
    bench.style_ax(ax, title="Joke-vs-control probe by stream depth", xlabel="residual stream depth", ylabel="held-out AUC")
    bench.save_figure(ctx, fig, "joke_probe_by_layer.png", "Held-out joke-vs-control probe AUC with null-control means.")


def plot_steering(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.0, 5.2))
    conditions = [
        "joke_structure_direction",
        "surprise_direction",
        "silly_direction",
        "positive_direction",
        "shuffled_joke_direction",
        "random_direction",
    ]
    for condition in conditions:
        sub = [row for row in rows if row.get("steering_condition") == condition]
        if not sub:
            continue
        xs = [float(row["dose_fraction"]) for row in sub]
        ys = [float(row["joke_vs_cheap_margin_delta_vs_baseline"]) for row in sub]
        ax.plot(xs, ys, marker="o", label=condition.replace("_direction", ""))
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    bench.style_ax(ax, title="Activation-addition dose response", xlabel="dose fraction of residual norm", ylabel="joke-vs-cheap marker margin delta", legend_loc="best")
    bench.save_figure(ctx, fig, "humor_steering_dose_response.png", "Dose response for joke-structure steering versus cheap-correlate directions.")


def plot_cosines(ctx: bench.RunContext, directions: Mapping[str, Any]) -> None:
    fig, ax = bench.new_figure(figsize=(6.2, 5.5))
    names = list(SEMANTIC_DIRECTIONS)
    mat = [[cosine(directions[a], directions[b]) for b in names] for a in names]
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    bench.style_ax(ax, title="Direction cosine audit", legend=False)
    bench.save_figure(ctx, fig, "humor_direction_cosines.png", "Cosines among joke-structure, surprise, silly, and positive directions.")


def plot_attention(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.0, 5.2))
    for condition in CONDITIONS:
        sub = [row for row in rows if row.get("condition") == condition]
        if not sub:
            continue
        xs = [int(row["layer"]) for row in sub if isinstance(row.get("layer"), int)]
        ys = [float(row["mean_attention_to_setup"]) for row in sub if isinstance(row.get("mean_attention_to_setup"), (int, float))]
        if xs and ys:
            ax.plot(xs, ys, marker="o", label=condition)
    bench.style_ax(ax, title="Attention from resolution token back to setup", xlabel="attention layer", ylabel="mean attention mass to setup")
    bench.save_figure(ctx, fig, "attention_to_setup.png", "Attention-to-setup summary by condition and layer.")


def plot_projection_summary(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.0, 5.2))
    joke_rows = [row for row in rows if row.get("direction_name") == "joke_structure"]
    xs = list(range(len(CONDITIONS)))
    ys = []
    for condition in CONDITIONS:
        vals = [float(row["mean_projection"]) for row in joke_rows if row.get("condition") == condition and isinstance(row.get("mean_projection"), (int, float))]
        ys.append(safe_fmean(vals))
    ax.bar(xs, ys)
    ax.set_xticks(xs)
    ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
    bench.style_ax(ax, title="Selected direction projections by condition", ylabel="mean projection on joke-structure direction", legend=False)
    bench.save_figure(ctx, fig, "joke_projection_by_condition.png", "Condition means along the selected joke-structure direction.")



# ---------------------------------------------------------------------------
# Visualization upgrade: synthesis tables and audit-first plots
# ---------------------------------------------------------------------------

HUMOR_VIS_CONDITIONS = ("joke", "literal", "surprise", "silly", "positive")
HUMOR_STEERING_ORDER = (
    "baseline",
    "joke_structure_direction",
    "opposite_joke_structure_direction",
    "surprise_direction",
    "silly_direction",
    "positive_direction",
    "shuffled_joke_direction",
    "random_direction",
)
HUMOR_EVIDENCE_COLUMNS = (
    "heldout_probe",
    "control_gap",
    "family_transfer",
    "cheap_cosine_risk",
    "setup_dependence",
    "steering_specificity",
    "label_readiness",
)


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _clip01(value: Any) -> float:
    v = _num(value, 0.0)
    if not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def _mean_key(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return safe_fmean([_num(row.get(key)) for row in rows])


def _humor_color(name: str, default: str = "#555555") -> str:
    fn = getattr(bench, "plot_humor_color", None)
    if callable(fn):
        return fn(str(name), default=default)
    palette = {
        "joke": "#d62728",
        "joke_structure": "#d62728",
        "joke_structure_direction": "#d62728",
        "literal": "#7f7f7f",
        "surprise": "#9467bd",
        "surprise_direction": "#9467bd",
        "silly": "#2ca02c",
        "silly_direction": "#2ca02c",
        "positive": "#ff7f0e",
        "positive_direction": "#ff7f0e",
        "shuffled": "#bcbd22",
        "shuffled_joke_direction": "#bcbd22",
        "random": "#8c564b",
        "random_direction": "#8c564b",
        "opposite_joke_structure_direction": "#1f77b4",
        "real": "#d62728",
        "control": "#7f7f7f",
        "cheap": "#9467bd",
        "label": "#17becf",
    }
    return palette.get(str(name), default)


def _humor_marker(name: str, default: str = "o") -> str:
    fn = getattr(bench, "plot_humor_marker", None)
    if callable(fn):
        return fn(str(name), default=default)
    markers = {
        "joke": "o",
        "literal": "s",
        "surprise": "^",
        "silly": "D",
        "positive": "P",
        "real": "o",
        "shuffled": "x",
        "random": "+",
    }
    return markers.get(str(name), default)


def _empty_panel(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#666666")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)


def _kind_auc_by_depth(rows: Sequence[Mapping[str, Any]], kind: str, split_scored: str = REPORT_SPLIT) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in rows:
        if row.get("direction_kind") != kind:
            continue
        if row.get("split_scored") != split_scored:
            continue
        depth = row.get("depth")
        try:
            d = int(depth)
        except Exception:
            continue
        auc = _num(row.get("auc"))
        if math.isfinite(auc):
            out[d] = auc
    return out


def make_family_generalization_summary(family_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({str(row.get("heldout_family")) for row in family_rows if row.get("heldout_family") not in (None, "")})
    out: list[dict[str, Any]] = []
    for family in families:
        sub = [row for row in family_rows if str(row.get("heldout_family")) == family]
        real = _mean_key([row for row in sub if row.get("direction_kind") == "real"], "auc")
        shuffled = _mean_key([row for row in sub if row.get("direction_kind") in {"shuffled_sign_mean", "shuffled_sign"}], "auc")
        random = _mean_key([row for row in sub if row.get("direction_kind") in {"random_oriented_mean", "random_oriented"}], "auc")
        best_control = max([v for v in (0.5, shuffled, random) if math.isfinite(v)] or [0.5])
        out.append({
            "heldout_family": family,
            "real_auc": rounded(real),
            "shuffled_auc": rounded(shuffled),
            "random_auc": rounded(random),
            "best_control_auc": rounded(best_control),
            "control_gap": rounded(real - best_control),
            "n_probe_rows": len(sub),
            "n_eval_jokes": safe_fmean([
                _num(row.get("n_eval_jokes"))
                for row in sub
                if row.get("direction_kind") == "real"
            ]),
            "status": "passes" if math.isfinite(real - best_control) and real - best_control >= 0.08 else "weak_or_failed",
        })
    return out


def make_steering_operating_points(effect_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in effect_rows:
        condition = str(row.get("steering_condition", ""))
        dose = _num(row.get("dose_fraction"), 0.0)
        joke_delta = _num(row.get("joke_vs_cheap_margin_delta_vs_baseline"), 0.0)
        cheap_delta = _num(row.get("cheap_marker_total_delta_vs_baseline"), 0.0)
        surprise_delta = _num(row.get("surprise_marker_count_delta_vs_baseline"), 0.0)
        silly_delta = _num(row.get("silly_marker_count_delta_vs_baseline"), 0.0)
        positive_delta = _num(row.get("positive_marker_count_delta_vs_baseline"), 0.0)
        repetition_delta = _num(row.get("repetition_rate_delta_vs_baseline"), 0.0)
        distinct_delta = _num(row.get("distinct_ratio_delta_vs_baseline"), 0.0)
        specificity = _num(row.get("joke_margin_delta_minus_best_cheap_same_dose"), float("nan"))
        if not math.isfinite(specificity) and condition == "joke_structure_direction":
            competitors = [
                _num(other.get("joke_vs_cheap_margin_delta_vs_baseline"))
                for other in effect_rows
                if _num(other.get("dose_fraction"), -999) == dose and str(other.get("steering_condition")) != condition
            ]
            competitors = [v for v in competitors if math.isfinite(v)]
            specificity = joke_delta - max(competitors) if competitors else float("nan")
        side_cost = max(0.0, cheap_delta) + max(0.0, repetition_delta) + max(0.0, -distinct_delta)
        out.append({
            "steering_condition": condition,
            "direction_family": condition.replace("_direction", ""),
            "dose_fraction": rounded(dose),
            "joke_margin_delta": rounded(joke_delta),
            "specificity_gap_vs_best_control": rounded(specificity),
            "cheap_marker_delta": rounded(cheap_delta),
            "surprise_marker_delta": rounded(surprise_delta),
            "silly_marker_delta": rounded(silly_delta),
            "positive_marker_delta": rounded(positive_delta),
            "repetition_delta": rounded(repetition_delta),
            "distinct_ratio_delta": rounded(distinct_delta),
            "side_effect_cost_proxy": rounded(side_cost),
            "claimable_before_hand_labels": bool(condition == "joke_structure_direction" and math.isfinite(specificity) and specificity > 0.15 and side_cost < max(1.0, abs(joke_delta) + 0.5)),
        })
    return out


def make_cheap_explanation_audit(
    metrics: Mapping[str, Any],
    surprisal_summary: Sequence[Mapping[str, Any]],
    projection_summary: Sequence[Mapping[str, Any]],
    attention_summary: Sequence[Mapping[str, Any]],
    operating_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_surprisal = {str(row.get("condition")): _num(row.get("mean_token_surprisal_bits")) for row in surprisal_summary}
    by_proj: dict[str, float] = {}
    for row in projection_summary:
        if row.get("direction_name") == "joke_structure":
            by_proj[str(row.get("condition"))] = _num(row.get("mean_projection"))
    joke_proj = by_proj.get("joke", float("nan"))
    max_control_proj = max([v for k, v in by_proj.items() if k != "joke" and math.isfinite(v)] or [float("nan")])
    attn_joke = safe_fmean([_num(row.get("mean_attention_to_setup")) for row in attention_summary if row.get("condition") == "joke"])
    attn_literal = safe_fmean([_num(row.get("mean_attention_to_setup")) for row in attention_summary if row.get("condition") == "literal"])
    max_dose = max([_num(row.get("dose_fraction"), 0.0) for row in operating_rows] or [0.0])
    joke_op = [row for row in operating_rows if row.get("steering_condition") == "joke_structure_direction" and _num(row.get("dose_fraction"), -1) == max_dose]
    op = joke_op[0] if joke_op else {}
    rows = [
        {
            "cheap_explanation": "raw_surprise",
            "audit_signal": "joke_surprisal_minus_literal and joke-surprise cosine",
            "risk_score": rounded(max(0.0, abs(_num(metrics.get("joke_surprise_cosine"))) * 0.7 + (1.0 if _num(metrics.get("joke_surprisal_minus_literal")) > 0.5 else 0.0) * 0.3)),
            "main_value": metrics.get("joke_surprisal_minus_literal"),
            "control_value": metrics.get("joke_surprise_cosine"),
            "interpretation": "high risk means the handle may be surprise without resolution",
        },
        {
            "cheap_explanation": "silliness",
            "audit_signal": "joke-silly cosine and silly steering",
            "risk_score": rounded(abs(_num(metrics.get("joke_silly_cosine")))),
            "main_value": metrics.get("joke_silly_cosine"),
            "control_value": _num(op.get("silly_marker_delta")),
            "interpretation": "high risk means whimsical weirdness may explain the result",
        },
        {
            "cheap_explanation": "positive_tone",
            "audit_signal": "joke-positive cosine and positive markers",
            "risk_score": rounded(abs(_num(metrics.get("joke_positive_cosine")))),
            "main_value": metrics.get("joke_positive_cosine"),
            "control_value": _num(op.get("positive_marker_delta")),
            "interpretation": "high risk means friendly tone may explain the handle",
        },
        {
            "cheap_explanation": "generic_joke_register",
            "audit_signal": "generic marker movement before hand labels",
            "risk_score": rounded(_clip01(max(0.0, _num(op.get("cheap_marker_delta"))) / max(1.0, abs(_num(op.get("joke_margin_delta"))) + 1e-6))),
            "main_value": _num(op.get("joke_margin_delta")),
            "control_value": _num(op.get("cheap_marker_delta")),
            "interpretation": "hand labels decide whether marker movement is real joke structure",
        },
        {
            "cheap_explanation": "no_setup_dependence",
            "audit_signal": "joke projection and attention advantage over controls",
            "risk_score": rounded(1.0 - _clip01(max(0.0, joke_proj - max_control_proj) + max(0.0, attn_joke - attn_literal))),
            "main_value": rounded(joke_proj - max_control_proj),
            "control_value": rounded(attn_joke - attn_literal),
            "interpretation": "high risk means the ending may not use the setup",
        },
        {
            "cheap_explanation": "layer_or_probe_shopping",
            "audit_signal": "test selectivity over best null after dev depth selection",
            "risk_score": rounded(1.0 - _clip01(_num(metrics.get("real_selectivity_vs_best_null")) / 0.15)),
            "main_value": metrics.get("real_selectivity_vs_best_null"),
            "control_value": max(_num(metrics.get("shuffled_auc_best_depth")), _num(metrics.get("random_auc_best_depth"))),
            "interpretation": "high risk means depth/control choices may carry the result",
        },
    ]
    return rows


def make_item_evidence_summary(
    items: Sequence[HumorItem],
    split: Mapping[str, Any],
    projection_rows: Sequence[Mapping[str, Any]],
    surprisal_rows: Sequence[Mapping[str, Any]],
    steering_generation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    proj: dict[tuple[str, str], float] = {}
    for row in projection_rows:
        if row.get("direction_name") == "joke_structure":
            proj[(str(row.get("item_id")), str(row.get("condition")))] = _num(row.get("projection"))
    surp: dict[tuple[str, str], float] = {}
    for row in surprisal_rows:
        surp[(str(row.get("item_id")), str(row.get("condition")))] = _num(row.get("mean_surprisal_bits"))
    by_item_gen: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in steering_generation_rows:
        by_item_gen[str(row.get("item_id"))].append(row)
    out: list[dict[str, Any]] = []
    for item in items:
        iid = item.item_id
        control_projs = [proj.get((iid, c), float("nan")) for c in CONTROL_CONDITIONS]
        control_projs = [v for v in control_projs if math.isfinite(v)]
        joke_proj = proj.get((iid, "joke"), float("nan"))
        max_control_proj = max(control_projs) if control_projs else float("nan")
        joke_surp = surp.get((iid, "joke"), float("nan"))
        literal_surp = surp.get((iid, "literal"), float("nan"))
        gens = by_item_gen.get(iid, [])
        baseline = [g for g in gens if g.get("steering_condition") == "baseline"]
        steered = [g for g in gens if g.get("steering_condition") == "joke_structure_direction"]
        max_dose = max([_num(g.get("dose_fraction"), 0.0) for g in steered] or [0.0])
        top_steered = [g for g in steered if _num(g.get("dose_fraction"), -1) == max_dose]
        out.append({
            "item_id": iid,
            "family": item.family,
            "split": split_name(split.get(iid)),
            "projection_joke": rounded(joke_proj),
            "projection_max_control": rounded(max_control_proj),
            "projection_specificity_gap": rounded(joke_proj - max_control_proj),
            "joke_surprisal_bits": rounded(joke_surp),
            "literal_surprisal_bits": rounded(literal_surp),
            "joke_minus_literal_surprisal": rounded(joke_surp - literal_surp),
            "baseline_joke_margin": rounded(_mean_key(baseline, "joke_vs_cheap_margin")),
            "maxdose_joke_margin": rounded(_mean_key(top_steered, "joke_vs_cheap_margin")),
            "maxdose_setup_anchor_rate": rounded(_mean_key(top_steered, "contains_setup_anchor")),
            "maxdose_resolution_keyword_rate": rounded(_mean_key(top_steered, "contains_resolution_keyword")),
            "setup_excerpt": item.setup[:100],
        })
    return out


def make_humor_evidence_matrix(metrics: Mapping[str, Any], family_summary: Sequence[Mapping[str, Any]], cheap_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    family_gap = safe_fmean([_num(row.get("control_gap")) for row in family_summary])
    cheap_risk = max([_num(row.get("risk_score"), 0.0) for row in cheap_rows] or [0.0])
    setup_signal = max(0.0, _num(metrics.get("joke_minus_literal_attention_to_setup"), 0.0))
    rows = [
        {
            "evidence_object": "heldout_probe",
            "evidence_rung": "DECODE",
            "headline_metric": metrics.get("real_auc_best_depth"),
            "control_metric": max(_num(metrics.get("shuffled_auc_best_depth")), _num(metrics.get("random_auc_best_depth"))),
            "score_0_to_1": rounded(_clip01((_num(metrics.get("real_auc_best_depth")) - 0.5) / 0.35)),
            "claim_boundary": "joke-vs-control handle, not funniness",
        },
        {
            "evidence_object": "control_gap",
            "evidence_rung": "DECODE",
            "headline_metric": metrics.get("real_selectivity_vs_best_null"),
            "control_metric": 0.0,
            "score_0_to_1": rounded(_clip01(_num(metrics.get("real_selectivity_vs_best_null")) / 0.15)),
            "claim_boundary": "must beat shuffled and random controls",
        },
        {
            "evidence_object": "family_transfer",
            "evidence_rung": "DECODE",
            "headline_metric": rounded(family_gap),
            "control_metric": 0.0,
            "score_0_to_1": rounded(_clip01(family_gap / 0.15)),
            "claim_boundary": "separates reusable handle from family-local lexicon",
        },
        {
            "evidence_object": "cheap_cosine_risk",
            "evidence_rung": "AUDIT",
            "headline_metric": rounded(cheap_risk),
            "control_metric": 0.0,
            "score_0_to_1": rounded(1.0 - _clip01(cheap_risk)),
            "claim_boundary": "risk from surprise, silliness, positivity, and register",
        },
        {
            "evidence_object": "setup_dependence",
            "evidence_rung": "OBS",
            "headline_metric": metrics.get("joke_minus_literal_attention_to_setup"),
            "control_metric": 0.0,
            "score_0_to_1": rounded(_clip01(setup_signal / 0.08)),
            "claim_boundary": "routing/projection support, not mechanism proof",
        },
        {
            "evidence_object": "steering_specificity",
            "evidence_rung": "CAUSAL",
            "headline_metric": metrics.get("joke_steering_specificity_gap"),
            "control_metric": 0.0,
            "score_0_to_1": rounded(_clip01(_num(metrics.get("joke_steering_specificity_gap")) / 0.30)),
            "claim_boundary": "only marker-level until hand labels are filled",
        },
        {
            "evidence_object": "label_readiness",
            "evidence_rung": "HUMAN-LABEL",
            "headline_metric": 0.0,
            "control_metric": 1.0,
            "score_0_to_1": 0.0,
            "claim_boundary": "required before saying generation became funnier",
        },
    ]
    for row in rows:
        score = _num(row.get("score_0_to_1"), 0.0)
        row["status"] = "strong" if score >= 0.75 else ("mixed" if score >= 0.45 else "weak")
    return rows


def write_visual_synthesis_tables(
    ctx: bench.RunContext,
    items: Sequence[HumorItem],
    split: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    phase_rows: Sequence[Mapping[str, Any]],
    family_heldout_rows: Sequence[Mapping[str, Any]],
    surprisal_rows: Sequence[Mapping[str, Any]],
    surprisal_summary: Sequence[Mapping[str, Any]],
    projection_rows: Sequence[Mapping[str, Any]],
    projection_summary: Sequence[Mapping[str, Any]],
    attention_summary: Sequence[Mapping[str, Any]],
    steering_generations: Sequence[Mapping[str, Any]],
    steering_effects: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    family_summary = make_family_generalization_summary(family_heldout_rows)
    operating_rows = make_steering_operating_points(steering_effects)
    cheap_rows = make_cheap_explanation_audit(metrics, surprisal_summary, projection_summary, attention_summary, operating_rows)
    item_rows = make_item_evidence_summary(items, split, projection_rows, surprisal_rows, steering_generations)
    evidence_rows = make_humor_evidence_matrix(metrics, family_summary, cheap_rows)
    guide_rows = [
        {
            "artifact": "plots/humor_evidence_dashboard.png",
            "concept": "one-screen claim audit",
            "look_for": "probe/control gap, family transfer, steering specificity, and cheap-risk panel",
        },
        {
            "artifact": "plots/depth_control_gap_atlas.png",
            "concept": "depth selection hygiene",
            "look_for": "real curve beating shuffled/random means at the selected depth",
        },
        {
            "artifact": "plots/family_generalization_atlas.png",
            "concept": "family-local versus reusable handle",
            "look_for": "one-family-held-out gaps that stay positive across joke families",
        },
        {
            "artifact": "plots/cheap_explanation_matrix.png",
            "concept": "deflationary audit",
            "look_for": "raw surprise, silliness, positivity, generic marker, or no-setup explanations with high risk",
        },
        {
            "artifact": "plots/setup_dependence_atlas.png",
            "concept": "setup-dependent resolution evidence",
            "look_for": "joke projections and attention-to-setup higher than cheap controls",
        },
        {
            "artifact": "plots/generation_marker_atlas.png",
            "concept": "steering effects by dose and condition",
            "look_for": "joke-structure steering moving margin more than control directions",
        },
        {
            "artifact": "plots/steering_operating_frontier.png",
            "concept": "dose choice as benefit versus side-effect cost",
            "look_for": "a nonzero dose with positive control gap and low cheap-marker/repetition cost",
        },
        {
            "artifact": "plots/item_incongruity_atlas.png",
            "concept": "per-item heterogeneity",
            "look_for": "whether aggregate effects are broad or carried by one row",
        },
        {
            "artifact": "plots/humor_evidence_matrix.png",
            "concept": "claim-readiness ledger",
            "look_for": "which evidence rungs are strong, mixed, or weak",
        },
    ]
    outputs = {
        "family_generalization_summary": family_summary,
        "steering_operating_points": operating_rows,
        "cheap_explanation_audit": cheap_rows,
        "item_incongruity_summary": item_rows,
        "humor_evidence_matrix": evidence_rows,
        "plot_reading_guide": guide_rows,
    }
    for name, rows in outputs.items():
        path = ctx.path("tables", f"{name}.csv")
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", f"Lab 18 visualization upgrade synthesis table: {name.replace('_', ' ')}.")
    return outputs


def plot_depth_control_gap_atlas(ctx: bench.RunContext, probe_rows: Sequence[Mapping[str, Any]], selection_rows: Sequence[Mapping[str, Any]], best_depth: int) -> None:
    fig, ax = bench.new_figure(figsize=(9.4, 5.4))
    real = _kind_auc_by_depth(probe_rows, "real")
    shuffled = _kind_auc_by_depth(probe_rows, "shuffled_sign_mean")
    random = _kind_auc_by_depth(probe_rows, "random_oriented_mean")
    depths = sorted(set(real) | set(shuffled) | set(random))
    if not depths:
        _empty_panel(ax, "no probe rows")
    else:
        real_y = [real.get(d, float("nan")) for d in depths]
        shuf_y = [shuffled.get(d, float("nan")) for d in depths]
        rand_y = [random.get(d, float("nan")) for d in depths]
        best_y = [max([v for v in (0.5, shuffled.get(d, float("nan")), random.get(d, float("nan"))) if math.isfinite(v)] or [0.5]) for d in depths]
        gap_y = [(real.get(d, float("nan")) - best_y[i]) if math.isfinite(real.get(d, float("nan"))) else float("nan") for i, d in enumerate(depths)]
        ax.plot(depths, real_y, marker="o", label="real held-out AUC", color=_humor_color("real"))
        ax.plot(depths, best_y, marker="s", label="best null floor", color=_humor_color("control"))
        ax.plot(depths, shuf_y, linestyle=":", alpha=0.75, label="shuffled mean", color=_humor_color("shuffled"))
        ax.plot(depths, rand_y, linestyle=":", alpha=0.75, label="random mean", color=_humor_color("random"))
        ax.axhline(0.5, linestyle=":", linewidth=1.0, color="#333333")
        ax.axvline(best_depth, linestyle="--", linewidth=1.2, color=_humor_color("joke"), label=f"selected depth {best_depth}")
        for d, g in zip(depths, gap_y):
            if d == best_depth and math.isfinite(g):
                ax.annotate(f"gap {g:+.2f}", (d, real.get(d, 0.5)), xytext=(6, 8), textcoords="offset points", fontsize=8)
    bench.style_ax(ax, title="Depth selection audit: real probe must clear null rails", xlabel="stream depth", ylabel="held-out AUC")
    bench.save_figure(ctx, fig, "depth_control_gap_atlas.png", "Real joke-vs-control AUC by depth against shuffled/random null floors.")


def plot_family_generalization_atlas(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.2, 5.2))
    if not rows:
        _empty_panel(ax, "no family-heldout rows")
    else:
        labels = [str(row.get("heldout_family")) for row in rows]
        real = [_num(row.get("real_auc")) for row in rows]
        best = [_num(row.get("best_control_auc")) for row in rows]
        gap = [_num(row.get("control_gap")) for row in rows]
        xs = list(range(len(labels)))
        width = 0.36
        ax.bar([x - width / 2 for x in xs], real, width=width, label="real", color=_humor_color("joke"), alpha=0.9)
        ax.bar([x + width / 2 for x in xs], best, width=width, label="best null", color=_humor_color("control"), alpha=0.65)
        ax.axhline(0.5, linestyle=":", color="#333333", linewidth=1.0)
        for x, g in zip(xs, gap):
            if math.isfinite(g):
                ax.text(x, max(0.03, max(real[x], best[x]) + 0.02), f"{g:+.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=25, ha="right")
    bench.style_ax(ax, title="Family-heldout transfer: does the handle leave home?", ylabel="AUC", legend_loc="best")
    bench.save_figure(ctx, fig, "family_generalization_atlas.png", "One-family-heldout joke-structure AUC versus control floor.")


def plot_cheap_explanation_matrix(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.9, 5.2))
    if not rows:
        _empty_panel(ax, "no cheap explanation audit rows")
    else:
        labels = [str(row.get("cheap_explanation")) for row in rows]
        risk = [_clip01(row.get("risk_score")) for row in rows]
        ys = list(range(len(labels)))
        colors = [_humor_color("cheap") if v >= 0.5 else _humor_color("joke") for v in risk]
        ax.barh(ys, risk, color=colors, alpha=0.82)
        ax.axvline(0.5, linestyle="--", color="#555555", linewidth=1.0, label="audit concern")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        for y, v in zip(ys, risk):
            ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=8)
        ax.invert_yaxis()
    bench.style_ax(ax, title="Cheap explanations get first right of refusal", xlabel="risk score (higher = more deflationary pressure)")
    bench.save_figure(ctx, fig, "cheap_explanation_matrix.png", "Deflationary audit for surprise, silliness, positivity, register, setup dependence, and layer shopping.")


def plot_setup_dependence_atlas(
    ctx: bench.RunContext,
    projection_summary: Sequence[Mapping[str, Any]],
    attention_summary: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)
    ax = axes[0]
    vals = []
    labels = list(HUMOR_VIS_CONDITIONS)
    for condition in labels:
        sub = [row for row in projection_summary if row.get("condition") == condition and row.get("direction_name") == "joke_structure"]
        vals.append(_mean_key(sub, "mean_projection"))
    if any(math.isfinite(v) for v in vals):
        ax.bar(range(len(labels)), vals, color=[_humor_color(c) for c in labels], alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.axhline(0, color="#333333", linewidth=0.8)
        bench.style_ax(ax, title="selected direction projection", ylabel="mean projection", legend=False)
    else:
        _empty_panel(ax, "no projection summary")
    ax = axes[1]
    layers = sorted({int(row.get("layer")) for row in attention_summary if isinstance(row.get("layer"), int)})
    if layers:
        for condition in ("joke", "literal", "surprise"):
            ys = [_mean_key([row for row in attention_summary if row.get("condition") == condition and row.get("layer") == layer], "mean_attention_to_setup") for layer in layers]
            ax.plot(layers, ys, marker=_humor_marker(condition), label=condition, color=_humor_color(condition))
        bench.style_ax(ax, title="resolution-token attention to setup", xlabel="attention layer", ylabel="attention mass")
    else:
        _empty_panel(ax, "no attention summary")
    bench.save_figure(ctx, fig, "setup_dependence_atlas.png", "Projection and routing evidence for setup-dependent joke resolution.")


def plot_generation_marker_atlas(ctx: bench.RunContext, operating_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    conditions = [c for c in HUMOR_STEERING_ORDER if any(row.get("steering_condition") == c for row in operating_rows)]
    doses = sorted({_num(row.get("dose_fraction"), 0.0) for row in operating_rows})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    if not conditions or not doses:
        _empty_panel(axes[0], "no steering summary")
        _empty_panel(axes[1], "no steering summary")
    else:
        keys = ["joke_margin_delta", "cheap_marker_delta"]
        titles = ["joke-vs-cheap margin delta", "cheap-marker side movement"]
        for ax, key, title in zip(axes, keys, titles):
            mat = np.full((len(conditions), len(doses)), np.nan)
            for i, cond in enumerate(conditions):
                for j, dose in enumerate(doses):
                    sub = [row for row in operating_rows if row.get("steering_condition") == cond and abs(_num(row.get("dose_fraction")) - dose) < 1e-9]
                    if sub:
                        mat[i, j] = _num(sub[0].get(key))
            finite = mat[np.isfinite(mat)]
            vmax = max(0.1, float(np.nanmax(np.abs(finite))) if finite.size else 1.0)
            im = ax.imshow(mat, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
            ax.set_xticks(range(len(doses)))
            ax.set_xticklabels([f"{d:.2g}" for d in doses])
            ax.set_yticks(range(len(conditions)))
            ax.set_yticklabels([c.replace("_direction", "").replace("_", "\n") for c in conditions], fontsize=8)
            for i in range(len(conditions)):
                for j in range(len(doses)):
                    if math.isfinite(float(mat[i, j])):
                        ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            bench.style_ax(ax, title=title, xlabel="dose", legend=False)
    bench.save_figure(ctx, fig, "generation_marker_atlas.png", "Dose-by-condition heatmaps for joke margin and cheap-marker side movement.")


def plot_steering_operating_frontier(ctx: bench.RunContext, operating_rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.8, 5.8))
    rows = [row for row in operating_rows if row.get("steering_condition") != "baseline"]
    if not rows:
        _empty_panel(ax, "no steering operating points")
    else:
        for row in rows:
            cond = str(row.get("steering_condition"))
            x = _num(row.get("side_effect_cost_proxy"), 0.0)
            y = _num(row.get("joke_margin_delta"), 0.0)
            dose = _num(row.get("dose_fraction"), 0.0)
            ax.scatter(x, y, s=60 + 40 * dose, color=_humor_color(cond), marker="o", alpha=0.85, edgecolor="#222222", linewidth=0.4)
            if cond in {"joke_structure_direction", "surprise_direction", "silly_direction", "positive_direction"}:
                ax.annotate(f"{cond.replace('_direction','')}@{dose:.2g}", (x, y), xytext=(5, 3), textcoords="offset points", fontsize=7)
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.axvline(0, color="#333333", linewidth=0.9)
    bench.style_ax(ax, title="Steering operating frontier: benefit must outrun side effects", xlabel="side-effect cost proxy", ylabel="joke-vs-cheap margin delta", legend=False)
    bench.save_figure(ctx, fig, "steering_operating_frontier.png", "Activation-addition operating points: marker benefit versus cheap-marker and fluency costs.")


def plot_item_incongruity_atlas(ctx: bench.RunContext, item_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    rows = sorted(item_rows, key=lambda r: (_num(r.get("projection_specificity_gap"), -999), str(r.get("item_id"))))
    if len(rows) > 30:
        rows = rows[-30:]
    metrics = [
        "projection_specificity_gap",
        "joke_minus_literal_surprisal",
        "baseline_joke_margin",
        "maxdose_joke_margin",
        "maxdose_setup_anchor_rate",
        "maxdose_resolution_keyword_rate",
    ]
    fig, ax = bench.new_figure(figsize=(10.4, max(4.8, 0.28 * len(rows) + 2.0)))
    if not rows:
        _empty_panel(ax, "no per-item evidence rows")
    else:
        mat = np.array([[_num(row.get(m)) for m in metrics] for row in rows], dtype=float)
        # Robust column normalization to signed z-ish values for mixed-unit display.
        norm = np.zeros_like(mat)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            finite = col[np.isfinite(col)]
            if finite.size:
                center = np.nanmedian(finite)
                scale = np.nanpercentile(np.abs(finite - center), 75) or 1.0
                norm[:, j] = np.clip((col - center) / (scale * 2.0), -1.0, 1.0)
            else:
                norm[:, j] = np.nan
        im = ax.imshow(norm, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{row.get('family')}:{row.get('item_id')}" for row in rows], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="within-column normalized signal")
    bench.style_ax(ax, title="Per-item incongruity atlas: aggregates should not hide one row", legend=False)
    bench.save_figure(ctx, fig, "item_incongruity_atlas.png", "Per-item projection, surprisal, generation, and setup-dependence indicators.")


def plot_humor_evidence_matrix(ctx: bench.RunContext, evidence_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [str(row.get("evidence_object")) for row in evidence_rows]
    vals = np.array([_clip01(row.get("score_0_to_1")) for row in evidence_rows], dtype=float)
    fig, ax = bench.new_figure(figsize=(9.0, 4.9))
    if not labels:
        _empty_panel(ax, "no evidence matrix rows")
    else:
        im = ax.imshow(vals.reshape(1, -1), aspect="auto", vmin=0, vmax=1, cmap="YlGn")
        ax.set_yticks([0])
        ax.set_yticklabels(["claim readiness"])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([label.replace("_", "\n") for label in labels], rotation=0, fontsize=8)
        for j, (v, row) in enumerate(zip(vals, evidence_rows)):
            ax.text(j, 0, f"{v:.2f}\n{row.get('status')}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="0 = weak, 1 = strong")
    bench.style_ax(ax, title="Humor claim-readiness matrix", legend=False)
    bench.save_figure(ctx, fig, "humor_evidence_matrix.png", "Joined evidence matrix for Lab 18 claim readiness.")


def plot_humor_evidence_dashboard(
    ctx: bench.RunContext,
    metrics: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    operating_rows: Sequence[Mapping[str, Any]],
    cheap_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    best_depth: int,
) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    real = _kind_auc_by_depth(probe_rows, "real")
    shuffled = _kind_auc_by_depth(probe_rows, "shuffled_sign_mean")
    random = _kind_auc_by_depth(probe_rows, "random_oriented_mean")
    depths = sorted(set(real) | set(shuffled) | set(random))
    if depths:
        ax.plot(depths, [real.get(d, float("nan")) for d in depths], color=_humor_color("joke"), marker="o", label="real")
        ax.plot(depths, [max([v for v in (0.5, shuffled.get(d, float("nan")), random.get(d, float("nan"))) if math.isfinite(v)] or [0.5]) for d in depths], color=_humor_color("control"), marker="s", label="best null")
        ax.axhline(0.5, linestyle=":", color="#333333", linewidth=0.9)
        ax.axvline(best_depth, linestyle="--", color=_humor_color("joke"), linewidth=1.1)
        bench.style_ax(ax, title="A. DECODE depth rail", xlabel="stream depth", ylabel="held-out AUC")
    else:
        _empty_panel(ax, "no probe rows")

    ax = axes[0, 1]
    if family_rows:
        labels = [str(row.get("heldout_family")) for row in family_rows]
        gaps = [_num(row.get("control_gap"), 0.0) for row in family_rows]
        ax.barh(range(len(labels)), gaps, color=[_humor_color("joke") if g > 0 else _humor_color("control") for g in gaps], alpha=0.85)
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        bench.style_ax(ax, title="B. Family-heldout gap", xlabel="real AUC minus best null")
    else:
        _empty_panel(ax, "no family transfer rows")

    ax = axes[1, 0]
    conditions = ["joke_structure_direction", "surprise_direction", "silly_direction", "positive_direction", "shuffled_joke_direction", "random_direction"]
    for cond in conditions:
        sub = sorted([row for row in operating_rows if row.get("steering_condition") == cond], key=lambda r: _num(r.get("dose_fraction")))
        if not sub:
            continue
        ax.plot([_num(row.get("dose_fraction")) for row in sub], [_num(row.get("joke_margin_delta")) for row in sub], marker="o", label=cond.replace("_direction", ""), color=_humor_color(cond))
    ax.axhline(0, linestyle=":", color="#333333", linewidth=0.9)
    bench.style_ax(ax, title="C. CAUSAL steering over cheap controls", xlabel="dose", ylabel="joke-vs-cheap margin delta", legend_loc="best")

    ax = axes[1, 1]
    if evidence_rows:
        labels = [str(row.get("evidence_object")) for row in evidence_rows]
        vals = [_clip01(row.get("score_0_to_1")) for row in evidence_rows]
        ax.bar(range(len(labels)), vals, color=[_humor_color("joke") if v >= 0.5 else _humor_color("cheap") for v in vals], alpha=0.85)
        ax.axhline(0.5, linestyle="--", color="#555555", linewidth=0.9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([label.replace("_", "\n") for label in labels], fontsize=7)
        ax.set_ylim(0, 1.05)
        bench.style_ax(ax, title="D. Claim-readiness scorecard", ylabel="0 weak  →  1 strong", legend=False)
    else:
        _empty_panel(ax, "no evidence rows")

    fig.suptitle("Lab 18 humor-as-incongruity evidence dashboard", fontsize=15)
    bench.save_figure(ctx, fig, "humor_evidence_dashboard.png", "One-screen Lab 18 dashboard: decode rails, family transfer, steering specificity, and claim readiness.")


def plot_visual_upgrade_suite(
    ctx: bench.RunContext,
    metrics: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    projection_summary: Sequence[Mapping[str, Any]],
    attention_summary: Sequence[Mapping[str, Any]],
    visual_tables: Mapping[str, Sequence[Mapping[str, Any]]],
    best_depth: int,
) -> None:
    family_summary = list(visual_tables.get("family_generalization_summary", []))
    operating_rows = list(visual_tables.get("steering_operating_points", []))
    cheap_rows = list(visual_tables.get("cheap_explanation_audit", []))
    item_rows = list(visual_tables.get("item_incongruity_summary", []))
    evidence_rows = list(visual_tables.get("humor_evidence_matrix", []))
    plot_humor_evidence_dashboard(ctx, metrics, probe_rows, selection_rows, family_summary, operating_rows, cheap_rows, evidence_rows, best_depth)
    plot_depth_control_gap_atlas(ctx, probe_rows, selection_rows, best_depth)
    plot_family_generalization_atlas(ctx, family_summary)
    plot_cheap_explanation_matrix(ctx, cheap_rows)
    plot_setup_dependence_atlas(ctx, projection_summary, attention_summary)
    plot_generation_marker_atlas(ctx, operating_rows)
    plot_steering_operating_frontier(ctx, operating_rows)
    plot_item_incongruity_atlas(ctx, item_rows)
    plot_humor_evidence_matrix(ctx, evidence_rows)


# ---------------------------------------------------------------------------
# Written artifacts
# ---------------------------------------------------------------------------


def write_humor_card(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    verdict = metrics.get("verdict")
    lines = [
        "# Lab 18 Humor Incongruity Card",
        "",
        "## Verdict",
        "",
        f"`{verdict}`",
        "",
        "## What the instrument measured",
        "",
        "A train-split mass-mean direction contrasts joke endings against matched literal, surprising, silly, and positive endings for the same setup. The selected stream depth is chosen from dev control-adjusted scores, then reported on test items.",
        "",
        "## Headline numbers",
        "",
        f"- Model: `{metrics.get('model_id')}`",
        f"- Rows: {metrics.get('n_rows')} (test rows used for steering: {metrics.get('n_eval_rows')})",
        f"- Selected stream depth: {metrics.get('best_depth')} (injection layer: {metrics.get('injection_layer')})",
        f"- Held-out joke-vs-control AUC: {metrics.get('real_auc_best_depth')}",
        f"- Null AUCs, shuffled/random means: {metrics.get('shuffled_auc_best_depth')} / {metrics.get('random_auc_best_depth')}",
        f"- Real selectivity over best null: {metrics.get('real_selectivity_vs_best_null')}",
        f"- Family-heldout mean real AUC / control gap: {metrics.get('family_heldout_mean_real_auc')} / {metrics.get('family_heldout_mean_control_gap')}",
        f"- Humor/surprise, humor/silly, humor/positive cosines: {metrics.get('joke_surprise_cosine')} / {metrics.get('joke_silly_cosine')} / {metrics.get('joke_positive_cosine')}",
        f"- Highest-dose joke-structure steering delta: {metrics.get('joke_steering_joke_margin_delta')}",
        f"- Highest-dose surprise steering delta: {metrics.get('surprise_steering_joke_margin_delta')}",
        f"- Highest-dose random steering delta: {metrics.get('random_steering_joke_margin_delta')}",
        f"- Joke steering specificity gap over best cheap control: {metrics.get('joke_steering_specificity_gap')}",
        "",
        "## What this does not show",
        "",
        "- It does not show that the model experiences anything funny.",
        "- It does not show that attention to the setup is the mechanism.",
        "- It does not show human-rated funniness unless the hand-label columns are filled and agree.",
        "- It does not show a general humor circuit beyond this dataset and this model.",
        "",
        "## Read next",
        "",
        "1. `operationalization_audit.md`",
        "2. `tables/joke_probe_by_layer.csv` and `plots/joke_probe_by_layer.png`",
        "3. `tables/family_heldout_probe.csv` - checks whether the handle transfers across held-out joke families.",
        "4. `tables/humor_direction_audit.csv` and `plots/humor_steering_dose_response.png`",
        "4. `tables/humor_steering_generations.csv`, then hand-label the scaffold",
        "",
    ]
    path = ctx.path("humor_incongruity_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "card", "Read-first Lab 18 verdict card with scope and caveats.")


def write_operationalization_audit(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    audit_result = "passed" if metrics.get("verdict") == "validated_joke_structure_handle_with_cautious_steering" else ("failed" if metrics.get("verdict") == "not_validated_by_controls" else "mixed")
    lines = [
        "# Lab 18 Operationalization Audit",
        "",
        "```yaml",
        "headline_claim: \"The model exposes a setup-dependent joke-structure handle.\"",
        "cheap_explanation: \"Raw surprise, silliness, positivity, generic joke register, chat-template residue, or a marker rubric.\"",
        "killer_control: \"Matched surprise/silly/positive endings, shuffled/random directions, family-heldout probes, surprisal audit, and hand labels.\"",
        f"result: \"{audit_result}\"",
        "claim_allowed: \"handle if controls pass; marker/register correlation if steering labels are unfilled; no subjective-funniness claim\"",
        "```",
        "",
        "## What was measured",
        "",
        "The lab measures a joke-structure or joke-register handle: residual-stream differences between matched joke endings and literal, surprising, silly, and positive endings for the same setup.",
        "",
        "It does not measure subjective funniness, enjoyment, social uptake, or a human-like sense of humor.",
        "",
        "## Cheap-explanation ledger",
        "",
        "| Cheap explanation | Artifact pressure test | Current-run signal |",
        "|---|---|---|",
        f"| Raw surprise | `tables/humor_surprisal_summary.csv`, surprise direction cosine and steering | joke minus literal surprisal: {metrics.get('joke_surprisal_minus_literal')}; joke/surprise cosine: {metrics.get('joke_surprise_cosine')} |",
        f"| Silliness | silly-not-joke controls and silly steering | joke/silly cosine: {metrics.get('joke_silly_cosine')} |",
        f"| Positive sentiment | positive-not-joke controls and positive steering | joke/positive cosine: {metrics.get('joke_positive_cosine')} |",
        f"| Generic joke register | hand-label scaffold for generations | marker-only steering delta: {metrics.get('joke_steering_joke_margin_delta')} |",
        f"| Setup dependence | attention from resolution token back to setup | joke minus literal attention: {metrics.get('joke_minus_literal_attention_to_setup')} |",
        f"| Probe capacity or layer shopping | shuffled and random nulls, dev depth selection, test reporting | selectivity over best null: {metrics.get('real_selectivity_vs_best_null')} |",
        "",
        "## Allowed claim",
        "",
        "A Lab 18 claim is a handle claim. It may say that this model exposes a direction that separates and possibly steers joke-shaped endings under these controls. If it collapses into surprise, silliness, positivity, or generic joke markers, that is the result.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "audit", "Operationalization limits and cheap-explanation audit for Lab 18.")


def write_run_summary(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 18 Run Summary: Humor as Incongruity",
        "",
        "## 1. Behavior and object",
        "",
        "The behavior is short-ending generation after a setup. The object is a narrow joke-structure contrast, not subjective funniness.",
        "",
        "## 2. Main measurement",
        "",
        f"A train-fit direction selected stream depth {metrics.get('best_depth')} on dev and reached test AUC {metrics.get('real_auc_best_depth')} versus shuffled/random means {metrics.get('shuffled_auc_best_depth')} / {metrics.get('random_auc_best_depth')}.",
        "",
        "## 3. Controls",
        "",
        f"The best-null adjusted selectivity was {metrics.get('real_selectivity_vs_best_null')}. Family-heldout mean AUC/control gap was {metrics.get('family_heldout_mean_real_auc')} / {metrics.get('family_heldout_mean_control_gap')}. Cosines with surprise, silly, and positive controls were {metrics.get('joke_surprise_cosine')}, {metrics.get('joke_silly_cosine')}, and {metrics.get('joke_positive_cosine')}.",
        "",
        "## 4. Causal extension",
        "",
        f"At the highest steering dose, joke-structure steering changed the marker margin by {metrics.get('joke_steering_joke_margin_delta')}; the specificity gap over the best cheap control was {metrics.get('joke_steering_specificity_gap')}.",
        "",
        "## 5. Current verdict",
        "",
        f"`{metrics.get('verdict')}`",
        "",
        "## 6. What the evidence does not support",
        "",
        "No result here shows felt humor, a general humor module, or a causal attention route. Hand labels are required before treating marker movement as funniness movement.",
        "",
        "## 7. What would falsify the interpretation",
        "",
        "A shuffled/random/null control matching the probe, collinearity with surprise or sentiment, cheap-correlate steering matching the joke-structure effect, or hand labels rejecting the marker rubric.",
        "",
        "Read `humor_incongruity_card.md` and `operationalization_audit.md` before translating this run into a ledger claim.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable summary of headline Lab 18 metrics.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if not bench.supports_chat_template(bundle):
        raise RuntimeError("Lab 18 requires an instruct model with a chat template.")

    bench_note = {
        "lab_id": getattr(args, "lab", "lab18"),
        "chat_template_supported_by_model": True,
        "bench_chat_template_labs_contains_lab18": "lab18" in getattr(bench, "CHAT_TEMPLATE_LABS", frozenset()),
        "attention_implementation": getattr(args, "attn_implementation", "unknown"),
        "recommended_registry_profile": {
            "needs_eager": "true",
            "model_tier_a": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "model_tier_b": "allenai/Olmo-3-7B-Instruct",
            "max_examples_tier_a": "2 or 3 per family",
        },
        "note": "This lab renders prompts itself, so content is valid even if registry diagnostics lag behind. Attention plots require eager attention.",
    }
    note_path = ctx.path("diagnostics", "bench_integration_note.json")
    bench.write_json(note_path, bench_note)
    ctx.register_artifact(note_path, "diagnostic", "Lab 18 benchmark integration note.")

    items, data_info, validation_rows = load_items(args)
    print(f"[lab18] {data_info['n_rows']} rows; prompt_set={args.prompt_set}; source={data_info['source_kind']}")
    manifest_path = ctx.path("diagnostics", "frozen_data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 18 data source, hash, filters, and counts.")
    validation_path = ctx.path("diagnostics", "data_validation_report.csv")
    bench.write_csv_with_context(ctx, validation_path, validation_rows)
    ctx.register_artifact(validation_path, "diagnostic", "Raw-row validation and drop reasons for Lab 18.")

    first_prompt = render_chat(bundle, contrast_message(items[0], "joke"))
    first_capture = run_exact_chat_hook_parity(ctx, bundle, first_prompt)
    bench.run_lens_self_check(ctx, bundle, first_capture)

    split = make_split(items, args.seed)
    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, split_rows(items, split))
    ctx.register_artifact(split_path, "diagnostic", "Family-stratified train/dev/test split for Lab 18.")
    split_balance_path = ctx.path("diagnostics", "split_balance.csv")
    bench.write_csv_with_context(ctx, split_balance_path, split_balance_rows(items, split))
    ctx.register_artifact(split_balance_path, "diagnostic", "Per-family train/dev/test balance summary.")

    feat_tensor, features, phase_features = cache_features(ctx, bundle, items)
    row_norms = feat_tensor.norm(dim=-1)
    norm_rows = []
    for depth in range(row_norms.shape[1]):
        vals = row_norms[:, depth].tolist()
        norm_rows.append({
            "depth": depth,
            "stream_depth_convention": "streams[k] = residual after k blocks; depth 0 is embeddings",
            "mean_norm": rounded(safe_fmean(vals)),
            "sd_norm": rounded(safe_stdev(vals)),
            "min_norm": rounded(float(row_norms[:, depth].min())),
            "max_norm": rounded(float(row_norms[:, depth].max())),
        })
    norm_path = ctx.path("diagnostics", "activation_norms_by_depth.csv")
    bench.write_csv_with_context(ctx, norm_path, norm_rows)
    ctx.register_artifact(norm_path, "diagnostic", "Humor/control prompt residual norm audit.")

    probe_rows, selection_rows, best_depth = run_probe_sweep(items, features, split, args.seed, bundle.anatomy.d_model)
    phase_rows = run_phase_probe(items, phase_features, split, args.seed, bundle.anatomy.d_model, best_depth)
    probe_path = ctx.path("tables", "joke_probe_by_layer.csv")
    bench.write_csv_with_context(ctx, probe_path, probe_rows)
    ctx.register_artifact(probe_path, "table", "Joke-vs-control held-out probe sweep with shuffled/random controls.")
    selection_path = ctx.path("tables", "joke_depth_selection.csv")
    bench.write_csv_with_context(ctx, selection_path, selection_rows)
    ctx.register_artifact(selection_path, "table", "Train audit, dev selection, and test report curve for the joke-structure direction.")
    depth_json = ctx.path("diagnostics", "depth_selection.json")
    bench.write_json(depth_json, {
        "selected_depth": best_depth,
        "selection_rule": "max dev control_adjusted_score = dev real AUC minus max(0.5, shuffled mean AUC, random mean AUC); tiny smoke runs may fall back to train_loo",
        "stream_depth_convention": "streams[k] = residual after k blocks; injection layer is depth - 1",
        "n_null_reps": N_NULL_REPS,
        "report_split": REPORT_SPLIT,
        "warning": "Tiny smoke runs may lack dev rows and fall back to train_loo; see joke_depth_selection.csv selection_mode.",
    })
    ctx.register_artifact(depth_json, "diagnostic", "Depth-selection rule and selected stream depth.")
    phase_path = ctx.path("tables", "punchline_phase_probe.csv")
    bench.write_csv_with_context(ctx, phase_path, phase_rows)
    ctx.register_artifact(phase_path, "table", "Setup-only versus full-joke punchline-phase probe at the selected depth.")
    family_heldout_rows = family_heldout_probe_rows(items, features, best_depth, args.seed, bundle.anatomy.d_model)
    family_heldout_path = ctx.path("tables", "family_heldout_probe.csv")
    bench.write_csv_with_context(ctx, family_heldout_path, family_heldout_rows)
    ctx.register_artifact(family_heldout_path, "table", "One-family-held-out joke-structure probe check at the selected depth.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, probe_rows)
    ctx.register_artifact(results_path, "results", "Alias of joke_probe_by_layer.csv for the standard run contract.")
    print(f"[lab18] selected stream depth {best_depth}")

    surprisal_rows = run_surprisal_measurements(bundle, items)
    surprisal_summary = summarise_surprisal(surprisal_rows)
    surprisal_path = ctx.path("tables", "humor_surprisal_trajectories.csv")
    bench.write_csv_with_context(ctx, surprisal_path, surprisal_rows)
    ctx.register_artifact(surprisal_path, "table", "Teacher-forced target surprisal and setup entropy for joke/control endings.")
    surprisal_summary_path = ctx.path("tables", "humor_surprisal_summary.csv")
    bench.write_csv_with_context(ctx, surprisal_summary_path, surprisal_summary)
    ctx.register_artifact(surprisal_summary_path, "table", "Condition-level surprisal and entropy summary.")

    fit_rows_ = train_rows(items, split)
    directions = {
        name: fit_direction(fit_rows_, features, best_depth, name)
        for name in SEMANTIC_DIRECTIONS
    }
    if any(direction is None for direction in directions.values()):
        missing = [name for name, direction in directions.items() if direction is None]
        raise RuntimeError(f"Could not build Lab 18 directions at depth {best_depth}: {missing}")
    directions = {name: orient_direction_on_rows(direction, fit_rows_, features, best_depth) for name, direction in directions.items()}  # type: ignore[arg-type]
    shuffled_direction = fit_direction(fit_rows_, features, best_depth, "joke_structure", sign_seed=args.seed + 90917)
    if shuffled_direction is None:
        shuffled_direction = random_unit(bundle.anatomy.d_model, args.seed + 90917)
    shuffled_direction = orient_direction_on_rows(shuffled_direction, fit_rows_, features, best_depth)

    cos_rows = direction_cosine_rows(directions)  # type: ignore[arg-type]
    cos_path = ctx.path("tables", "direction_cosines.csv")
    bench.write_csv_with_context(ctx, cos_path, cos_rows)
    ctx.register_artifact(cos_path, "table", "Pairwise cosines among joke-structure, surprise, silliness, and positivity directions.")

    projection_rows, projection_summary = projection_audit_rows(eval_rows(items, split), features, directions, best_depth)  # type: ignore[arg-type]
    projection_path = ctx.path("tables", "projection_by_condition.csv")
    bench.write_csv_with_context(ctx, projection_path, projection_rows)
    ctx.register_artifact(projection_path, "table", "Per-condition projections onto joke and cheap-correlate directions.")
    projection_summary_path = ctx.path("tables", "projection_by_condition_summary.csv")
    bench.write_csv_with_context(ctx, projection_summary_path, projection_summary)
    ctx.register_artifact(projection_summary_path, "table", "Condition-level projection means at the selected depth.")

    attn_items = selected_eval_rows(items, split)
    attn_rows, span_rows = attention_to_setup_rows(bundle, attn_items)
    attn_path = ctx.path("tables", "attention_to_setup.csv")
    bench.write_csv_with_context(ctx, attn_path, attn_rows)
    ctx.register_artifact(attn_path, "table", "Attention from resolution token to setup span for joke and control prompts.")
    span_path = ctx.path("diagnostics", "attention_span_audit.csv")
    bench.write_csv_with_context(ctx, span_path, span_rows)
    ctx.register_artifact(span_path, "diagnostic", "Span lookup audit for setup, anchor, ending, and resolution tokens.")
    attn_summary = attention_summary_rows(attn_rows)
    attn_summary_path = ctx.path("tables", "attention_to_setup_summary.csv")
    bench.write_csv_with_context(ctx, attn_summary_path, attn_summary)
    ctx.register_artifact(attn_summary_path, "table", "Condition-level attention-to-setup summary by layer.")

    ref_norm = safe_fmean(row_norms[:, best_depth].tolist(), default=1.0)
    steering_items = selected_eval_rows(items, split)
    steering_generations, steering_effects = run_steering(
        bundle,
        steering_items,
        directions,  # type: ignore[arg-type]
        shuffled_direction,
        best_depth,
        bundle.anatomy.d_model,
        args.seed,
        ref_norm,
    )
    generation_path = ctx.path("tables", "humor_steering_generations.csv")
    bench.write_csv_with_context(ctx, generation_path, steering_generations)
    ctx.register_artifact(generation_path, "table", "Baseline and steered endings with marker and hand-label scaffold.")
    effects_path = ctx.path("tables", "humor_direction_audit.csv")
    bench.write_csv_with_context(ctx, effects_path, steering_effects)
    ctx.register_artifact(effects_path, "table", "Joke-structure steering effect compared with surprise, silly, positive, shuffled, and random controls.")
    write_generation_labeling_guide(ctx)

    state_common = {
        "depth": best_depth,
        "injection_layer_for_activation_addition": max(0, best_depth - 1),
        "depth_convention": "bench streams[k]: 0 = embeddings, k = residual after block k",
        "read_site": "resolution token when available; otherwise ending-last token; fallback rows logged in diagnostics/prompt_render_audit.csv",
        "model_id": bundle.anatomy.model_id,
        "d_model": bundle.anatomy.d_model,
        "n_layers": bundle.anatomy.n_layers,
        "method": "train-split mass-mean directions over matched joke/control endings; depth selected on dev and reported on test",
        "semantic_warning": "joke_structure is a handle name, not evidence of felt funniness",
    }
    save_directions = {"humor": directions["joke_structure"], **directions, "shuffled_joke_structure": shuffled_direction}  # type: ignore[dict-item]
    state_path = ctx.path("state", "humor_directions.pt")
    torch.save({**state_common, "directions": save_directions}, state_path)
    ctx.register_artifact(state_path, "tensor", "Joke-structure, surprise, silly, positive, and control directions.")
    humor_path = ctx.path("state", "humor_direction.pt")
    torch.save({**state_common, "direction": directions["joke_structure"], "alias": "joke_structure"}, humor_path)  # type: ignore[index]
    ctx.register_artifact(humor_path, "tensor", "Selected joke-structure direction, kept under legacy humor_direction name.")
    meta_path = ctx.path("state", "humor_direction_metadata.json")
    bench.write_json(meta_path, {**state_common, "directions": sorted(save_directions), "n_null_reps": N_NULL_REPS})
    ctx.register_artifact(meta_path, "state", "Human-readable metadata for Lab 18 saved directions.")

    max_dose = max(STEERING_DOSES)
    real_auc = metric_at(probe_rows, "real", best_depth)
    shuffled_auc = metric_at(probe_rows, "shuffled_sign_mean", best_depth)
    random_auc = metric_at(probe_rows, "random_oriented_mean", best_depth)
    best_null_auc = max(0.5, shuffled_auc if math.isfinite(shuffled_auc) else 0.5, random_auc if math.isfinite(random_auc) else 0.5)
    by_surprisal = {
        row["condition"]: float(row["mean_token_surprisal_bits"])
        for row in surprisal_summary
        if isinstance(row.get("mean_token_surprisal_bits"), (int, float))
    }
    humor_surprise_cos = cosine(directions["joke_structure"], directions["surprise"])  # type: ignore[index]
    humor_silly_cos = cosine(directions["joke_structure"], directions["silly"])  # type: ignore[index]
    humor_positive_cos = cosine(directions["joke_structure"], directions["positive"])  # type: ignore[index]
    joke_delta = effect_delta(steering_effects, "joke_structure_direction", "joke_vs_cheap_margin_delta_vs_baseline", max_dose)
    surprise_delta = effect_delta(steering_effects, "surprise_direction", "joke_vs_cheap_margin_delta_vs_baseline", max_dose)
    silly_delta = effect_delta(steering_effects, "silly_direction", "joke_vs_cheap_margin_delta_vs_baseline", max_dose)
    positive_delta = effect_delta(steering_effects, "positive_direction", "joke_vs_cheap_margin_delta_vs_baseline", max_dose)
    shuffled_delta = effect_delta(steering_effects, "shuffled_joke_direction", "joke_vs_cheap_margin_delta_vs_baseline", max_dose)
    random_delta = effect_delta(steering_effects, "random_direction", "joke_vs_cheap_margin_delta_vs_baseline", max_dose)
    cheap_best_delta = max([v for v in (surprise_delta, silly_delta, positive_delta, shuffled_delta, random_delta) if math.isfinite(float(v))] or [float("nan")])
    family_real_aucs = [
        float(row["auc"]) for row in family_heldout_rows
        if row.get("direction_kind") == "real" and isinstance(row.get("auc"), (int, float))
    ]
    family_control_gaps: list[float] = []
    for family in sorted({row["heldout_family"] for row in family_heldout_rows}):
        real_vals = [
            float(row["auc"]) for row in family_heldout_rows
            if row.get("heldout_family") == family and row.get("direction_kind") == "real" and isinstance(row.get("auc"), (int, float))
        ]
        control_vals = [
            float(row["auc"]) for row in family_heldout_rows
            if row.get("heldout_family") == family and row.get("direction_kind") != "real" and isinstance(row.get("auc"), (int, float))
        ]
        if real_vals and control_vals:
            family_control_gaps.append(safe_fmean(real_vals) - max(control_vals))
    attn_joke = safe_fmean([
        float(row["mean_attention_to_setup"]) for row in attn_summary
        if row.get("condition") == "joke" and isinstance(row.get("mean_attention_to_setup"), (int, float))
    ])
    attn_lit = safe_fmean([
        float(row["mean_attention_to_setup"]) for row in attn_summary
        if row.get("condition") == "literal" and isinstance(row.get("mean_attention_to_setup"), (int, float))
    ])

    metrics: dict[str, Any] = {
        "model_id": bundle.anatomy.model_id,
        "seed": int(args.seed),
        "report_split": REPORT_SPLIT,
        "n_rows": len(items),
        "n_report_rows": len(eval_rows(items, split)),
        "n_eval_rows": len(steering_items),
        "best_depth": best_depth,
        "injection_layer": max(0, best_depth - 1),
        "real_auc_best_depth": none_if_nan(real_auc),
        "shuffled_auc_best_depth": none_if_nan(shuffled_auc),
        "random_auc_best_depth": none_if_nan(random_auc),
        "real_selectivity_vs_shuffled": none_if_nan(real_auc - shuffled_auc),
        "real_selectivity_vs_best_null": none_if_nan(real_auc - best_null_auc),
        "family_heldout_mean_real_auc": none_if_nan(safe_fmean(family_real_aucs)),
        "family_heldout_mean_control_gap": none_if_nan(safe_fmean(family_control_gaps)),
        "mean_joke_surprisal_bits": none_if_nan(by_surprisal.get("joke", float("nan"))),
        "mean_literal_surprisal_bits": none_if_nan(by_surprisal.get("literal", float("nan"))),
        "mean_surprise_surprisal_bits": none_if_nan(by_surprisal.get("surprise", float("nan"))),
        "joke_surprisal_minus_literal": none_if_nan(by_surprisal.get("joke", float("nan")) - by_surprisal.get("literal", float("nan"))),
        "joke_surprise_cosine": none_if_nan(humor_surprise_cos),
        "joke_silly_cosine": none_if_nan(humor_silly_cos),
        "joke_positive_cosine": none_if_nan(humor_positive_cos),
        "humor_surprise_cosine": none_if_nan(humor_surprise_cos),
        "humor_silly_cosine": none_if_nan(humor_silly_cos),
        "humor_positive_cosine": none_if_nan(humor_positive_cos),
        "joke_steering_joke_margin_delta": none_if_nan(joke_delta),
        "surprise_steering_joke_margin_delta": none_if_nan(surprise_delta),
        "silly_steering_joke_margin_delta": none_if_nan(silly_delta),
        "positive_steering_joke_margin_delta": none_if_nan(positive_delta),
        "shuffled_steering_joke_margin_delta": none_if_nan(shuffled_delta),
        "random_steering_joke_margin_delta": none_if_nan(random_delta),
        "joke_steering_specificity_gap": none_if_nan(joke_delta - cheap_best_delta),
        "joke_minus_literal_attention_to_setup": none_if_nan(attn_joke - attn_lit),
        "max_steering_dose_fraction": max_dose,
        "steering_doses": list(STEERING_DOSES),
        "n_null_reps": N_NULL_REPS,
        "data": data_info,
    }
    metrics["verdict"] = verdict_from_metrics(metrics)

    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 18 metrics and verdict.")

    visual_tables = write_visual_synthesis_tables(
        ctx,
        items,
        split,
        probe_rows,
        selection_rows,
        phase_rows,
        family_heldout_rows,
        surprisal_rows,
        surprisal_summary,
        projection_rows,
        projection_summary,
        attn_summary,
        steering_generations,
        steering_effects,
        metrics,
    )

    if not args.no_plots:
        plot_visual_upgrade_suite(
            ctx,
            metrics,
            probe_rows,
            selection_rows,
            family_heldout_rows,
            projection_summary,
            attn_summary,
            visual_tables,
            best_depth,
        )
        plot_surprisal(ctx, surprisal_rows)
        plot_probe(ctx, probe_rows, best_depth)
        plot_steering(ctx, steering_effects)
        plot_cosines(ctx, directions)  # type: ignore[arg-type]
        plot_attention(ctx, attn_summary)
        plot_projection_summary(ctx, projection_summary)

    write_humor_card(ctx, metrics)
    write_operationalization_audit(ctx, metrics)
    write_run_summary(ctx, metrics)

    run_name = ctx.run_dir.name
    selectivity = metrics.get("real_selectivity_vs_best_null")
    if isinstance(selectivity, (int, float)) and float(selectivity) > 0.10:
        decode_text = (
            f"At stream depth {best_depth}, the joke-structure direction separates held-out "
            f"joke endings from literal/surprise/silly/positive controls with AUC "
            f"{metrics['real_auc_best_depth']} versus shuffled/random means "
            f"{metrics['shuffled_auc_best_depth']} / {metrics['random_auc_best_depth']}. "
            "This is a joke-structure handle claim, not a claim about subjective funniness."
        )
    else:
        decode_text = (
            f"In this run, Lab 18 did not validate a selective joke-structure probe: held-out AUC was "
            f"{metrics['real_auc_best_depth']} versus shuffled/random means "
            f"{metrics['shuffled_auc_best_depth']} / {metrics['random_auc_best_depth']}."
        )

    gap = metrics.get("joke_steering_specificity_gap")
    if isinstance(gap, (int, float)) and float(gap) > 0.20:
        steer_text = (
            f"At dose {max_dose}, joke-structure steering changed the joke-vs-cheap marker margin by "
            f"{metrics['joke_steering_joke_margin_delta']}, exceeding the best cheap-control direction by "
            f"{metrics['joke_steering_specificity_gap']}. Hand labels are still required before calling this funniness."
        )
    else:
        steer_text = (
            f"In this run, joke-structure steering was not cleanly separated from cheap controls: its marker-margin "
            f"delta was {metrics['joke_steering_joke_margin_delta']} and its specificity gap was "
            f"{metrics['joke_steering_specificity_gap']}. Treat this as marker movement or an inconclusive causal result."
        )

    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "DECODE",
            "text": decode_text,
            "artifact": f"runs/{run_name}/tables/joke_probe_by_layer.csv",
            "falsifier": (
                "Shuffled or random controls match the AUC, the selected depth fails on a fresh family, "
                "or direction cosines show the handle is just surprise, silliness, or positivity."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "CAUSAL",
            "text": steer_text,
            "artifact": f"runs/{run_name}/tables/humor_direction_audit.csv",
            "falsifier": (
                "Surprise/silly/positive/shuffled/random steering matches the effect, or hand labels show "
                "the marker rubric confuses joke shape with generic weirdness or sentiment."
            ),
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
