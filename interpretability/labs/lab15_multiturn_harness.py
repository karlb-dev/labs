"""Lab 15: Multi-turn instrumentation.

This is a harness-validation lab, not a model-science lab. Later advanced
labs want to say things like "persona state rose over turns" or "the model
capitulated after pressure." Those claims are counterfeit coins if the chat
turn spans, cache reads, or patch positions are off by one token.

The lab therefore treats a harmless scripted topic trace as a calibration
ritual. It validates:

  * chat-template string tokenization versus direct template tokenization;
  * message-level and content-level token spans;
  * exact residual-stream hook semantics on already-rendered chat text;
  * full-recompute versus KV-cache boundary states;
  * self-patching a boundary block output as a no-op;
  * topic traces against random and length/template null directions.

Evidence level: OBS, and only for instrumentation. The orchid trace is a demo,
not evidence of persona, belief, or stable memory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L15"

SYSTEM_PROMPT = (
    "You are a precise assistant. Keep replies short and preserve the user's "
    "topic labels exactly."
)

# KV-cache parity tolerance. Full recompute vs incremental prefill is bitwise
# identical only in fp32. In bf16/fp16 the two paths round differently and a
# single residual dimension can diverge (Olmo-3 carries outlier/sink dims)
# while the rest of the vector agrees, so the gate is the WHOLE-VECTOR relative
# L2 error ||full-cached|| / ||full||, not a per-dimension max. A structural
# cache bug (off-by-one position/window) perturbs the whole vector and blows
# this up; one bf16-divergent dim does not. CACHE_PARITY_ATOL is retained for
# the legacy bench_integration_note field.
CACHE_PARITY_ATOL = 2e-3
# Thresholds calibrated empirically on Olmo-3-7B-Instruct, 18 boundaries:
# fp32 max rel-L2 6e-6 (cosine 0.9999997); bf16 max rel-L2 0.042 (cosine 0.999).
# An off-by-one in the cache window instead gives rel-L2 ~0.3+ / cosine <0.95 at
# the affected boundary, so 0.08 separates rounding from real bugs with margin.
CACHE_PARITY_REL_L2 = 1e-3        # fp32: near-exact whole-vector agreement
CACHE_PARITY_REL_L2_LOWP = 8e-2   # bf16/fp16: a few percent vector error is rounding
PATCH_NOOP_ATOL = 3e-3
TRACE_DEPTH_FRACTION = 0.5
N_RANDOM_NULL_DIRECTIONS = 8
NULL_SLOPE_WARN_RATIO = 0.60
NULL_SLOPE_ABS_WARN = 0.20
PATCH_LAYER_FRACTIONS = (0.25, 0.50, 0.75)


@dataclasses.dataclass(frozen=True)
class ConversationSpec:
    name: str
    messages: tuple[dict[str, str], ...]
    expected_topic: str
    note: str = ""


@dataclasses.dataclass(frozen=True)
class Segment:
    """One rendered chat-template message segment.

    The message span covers every token introduced when this message is added
    to the chat template: role header, content, separators, and any end-of-turn
    markers. The content span is narrower when the tokenizer can map character
    offsets back to token spans. Later labs should say explicitly which one
    they use.
    """

    index: int
    role: str
    message_start: int
    message_end: int
    content_start: int
    content_end: int
    message_start_char: int
    message_end_char: int
    content_start_char: int | None
    content_end_char: int | None
    content: str
    content_span_found: bool
    content_span_method: str

    @property
    def start(self) -> int:
        return self.message_start

    @property
    def end(self) -> int:
        return self.message_end

    @property
    def boundary_token(self) -> int:
        return self.message_end - 1

    @property
    def content_boundary_token(self) -> int:
        return self.content_end - 1


@dataclasses.dataclass(frozen=True)
class RenderedConversation:
    spec: ConversationSpec
    rendered: str
    input_ids: list[int]
    segments: list[Segment]
    info: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class DirectionPair:
    pair_id: str
    positive_topic: str
    negative_topic: str
    positive_messages: tuple[dict[str, str], ...]
    negative_messages: tuple[dict[str, str], ...]
    note: str


# ---------------------------------------------------------------------------
# Small numerical and serialization helpers
# ---------------------------------------------------------------------------


def rounded(x: Any, ndigits: int = 6) -> Any:
    try:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return round(float(x), ndigits)
    except Exception:
        pass
    return x


def safe_fmean(vals: Sequence[float], default: float = float("nan")) -> float:
    finite = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(statistics.fmean(finite)) if finite else default


def safe_max_abs(vals: Sequence[float], default: float = float("nan")) -> float:
    finite = [abs(float(v)) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return max(finite) if finite else default


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = []
    for x, y in zip(xs, ys):
        try:
            xf = float(x)
            yf = float(y)
        except Exception:
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 2:
        return float("nan")
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = statistics.fmean(xvals)
    my = statistics.fmean(yvals)
    denom = sum((x - mx) ** 2 for x in xvals)
    if denom < 1e-12:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / denom


def unit(v: Any) -> Any:
    norm = v.norm().clamp_min(1e-9)
    if not bool(norm.isfinite()):
        raise RuntimeError("Direction norm was not finite.")
    return v / norm


def random_unit(d_model: int, seed: int) -> Any:
    import torch

    gen = torch.Generator().manual_seed(int(seed))
    return unit(torch.randn(d_model, generator=gen))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids: Sequence[int]) -> str:
    joined = ",".join(str(int(x)) for x in ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def role_turn_index(segments: Sequence[Segment], target_idx: int) -> int:
    """Return a 1-indexed turn count excluding system messages."""

    count = 0
    for seg in segments[: target_idx + 1]:
        if seg.role != "system":
            count += 1
    return count


# ---------------------------------------------------------------------------
# Demo conversations and direction prompts
# ---------------------------------------------------------------------------


def chat_messages(user: str, assistant: str, *, system: str = SYSTEM_PROMPT) -> tuple[dict[str, str], ...]:
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    )


def conversations() -> list[ConversationSpec]:
    """Length and template matched conversations used by the harness demo."""

    topic_messages = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "We are planning a small orchid greenhouse. Remember that orchids are the project focus."},
        {"role": "assistant", "content": "Noted: the greenhouse plan is centered on orchids."},
        {"role": "user", "content": "Add humidity, bark medium, and filtered light as orchid-care requirements."},
        {"role": "assistant", "content": "The orchid plan now includes humidity, airy bark medium, and filtered light."},
        {"role": "user", "content": "Add watering rhythm and airflow for the orchid benches."},
        {"role": "assistant", "content": "The orchid benches should balance watering rhythm, airflow, and spacing."},
        {"role": "user", "content": "Summarize the greenhouse plan with the orchid priorities in order."},
        {"role": "assistant", "content": "Orchid priorities: humidity, filtered light, bark medium, airflow, and careful watering."},
    )
    control_messages = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "We are planning a small archive room. Remember that folders are the project focus."},
        {"role": "assistant", "content": "Noted: the archive plan is centered on folders."},
        {"role": "user", "content": "Add labels, shelf spacing, and climate notes as folder-care requirements."},
        {"role": "assistant", "content": "The folder plan now includes labels, shelf spacing, and climate notes."},
        {"role": "user", "content": "Add review rhythm and aisle airflow for the folder shelves."},
        {"role": "assistant", "content": "The folder shelves should balance review rhythm, airflow, and spacing."},
        {"role": "user", "content": "Summarize the archive plan with the folder priorities in order."},
        {"role": "assistant", "content": "Folder priorities: labels, shelf spacing, climate notes, airflow, and careful review."},
    )
    return [
        ConversationSpec("orchid_topic", topic_messages, "orchid", "Positive topic trace conversation."),
        ConversationSpec("archive_length_control", control_messages, "folder", "Length/template matched non-orchid control."),
    ]


def direction_contrast_pairs() -> tuple[DirectionPair, ...]:
    """Matched mini-dialogues used to build topic and length-null directions."""

    return (
        DirectionPair(
            pair_id="greenhouse_vs_archive",
            positive_topic="orchid",
            negative_topic="archive",
            positive_messages=chat_messages(
                "Topic focus: orchid greenhouse humidity bark medium filtered light watering airflow.",
                "The notes are about orchid care in a greenhouse.",
            ),
            negative_messages=chat_messages(
                "Topic focus: archive room labels shelf spacing climate notes review airflow.",
                "The notes are about folder care in an archive room.",
            ),
            note="Domain words matched to the main demo conversations.",
        ),
        DirectionPair(
            pair_id="potting_vs_cataloging",
            positive_topic="orchid",
            negative_topic="archive",
            positive_messages=chat_messages(
                "Topic focus: orchid potting roots mist shade growers benches trays.",
                "The notes concern orchid potting and plant care.",
            ),
            negative_messages=chat_messages(
                "Topic focus: archive catalog boxes index tabs clerks shelves trays.",
                "The notes concern archive cataloging and folder care.",
            ),
            note="Second topic pair reduces dependence on one exact phrase.",
        ),
        DirectionPair(
            pair_id="blooms_vs_records",
            positive_topic="orchid",
            negative_topic="archive",
            positive_messages=chat_messages(
                "Topic focus: orchid blooms stems moss humidity greenhouse checklist.",
                "The checklist remains about orchids and greenhouse upkeep.",
            ),
            negative_messages=chat_messages(
                "Topic focus: archive records tabs boxes climate storage checklist.",
                "The checklist remains about records and archive upkeep.",
            ),
            note="Third topic pair gives the mean direction some breadth.",
        ),
    )


def length_null_pairs() -> tuple[DirectionPair, ...]:
    return (
        DirectionPair(
            pair_id="schedule_vs_ledger",
            positive_topic="neutral_a",
            negative_topic="neutral_b",
            positive_messages=chat_messages(
                "Topic focus: schedule notes columns rows markers entries counts.",
                "The notes are about neutral planning fields.",
            ),
            negative_messages=chat_messages(
                "Topic focus: ledger items rows columns markers entries counts.",
                "The notes are about neutral planning records.",
            ),
            note="Neutral length/template contrast.",
        ),
        DirectionPair(
            pair_id="index_vs_matrix",
            positive_topic="neutral_a",
            negative_topic="neutral_b",
            positive_messages=chat_messages(
                "Topic focus: index cards rows columns sample totals markers.",
                "The notes are about neutral index fields.",
            ),
            negative_messages=chat_messages(
                "Topic focus: matrix cells rows columns sample totals markers.",
                "The notes are about neutral matrix fields.",
            ),
            note="Second neutral contrast, matched lexical shape.",
        ),
        DirectionPair(
            pair_id="forms_vs_tables",
            positive_topic="neutral_a",
            negative_topic="neutral_b",
            positive_messages=chat_messages(
                "Topic focus: forms pages rows columns entries counts labels.",
                "The notes are about neutral form pages.",
            ),
            negative_messages=chat_messages(
                "Topic focus: tables sheets rows columns entries counts labels.",
                "The notes are about neutral table sheets.",
            ),
            note="Third neutral contrast.",
        ),
    )


# ---------------------------------------------------------------------------
# Rendering, tokenization, and segmentation
# ---------------------------------------------------------------------------


def render_messages(
    bundle: bench.ModelBundle,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    if not messages:
        return ""
    return bundle.tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
    )


def token_ids(bundle: bench.ModelBundle, rendered: str) -> list[int]:
    ids = bundle.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def find_subsequence(
    haystack: Sequence[int],
    needle: Sequence[int],
    *,
    start: int,
    end: int,
) -> tuple[int | None, int | None]:
    needle_list = [int(x) for x in needle]
    if not needle_list:
        return None, None
    limit = int(end) - len(needle_list)
    for i in range(int(start), limit + 1):
        if [int(x) for x in haystack[i : i + len(needle_list)]] == needle_list:
            return i, i + len(needle_list)
    return None, None


def direct_template_ids(
    bundle: bench.ModelBundle,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> list[int]:
    out = bundle.tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=add_generation_prompt
    )
    # transformers 5 returns a BatchEncoding (a UserDict, NOT a dict subclass)
    # from apply_chat_template(tokenize=True), so an `isinstance(out, dict)`
    # guard silently misses it and iterating yields the key 'input_ids'.
    # Pull input_ids by mapping protocol before any positional indexing.
    if hasattr(out, "keys") and "input_ids" in out:
        ids = out["input_ids"]
    else:
        ids = out
    if hasattr(ids, "tolist"):  # torch tensor / numpy array
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def token_texts(bundle: bench.ModelBundle, ids: Sequence[int]) -> list[str]:
    return [bundle.tokenizer.decode([int(tid)]) for tid in ids]


def token_pieces(bundle: bench.ModelBundle, ids: Sequence[int]) -> list[str]:
    with suppress_tokenizer_errors():
        return [str(x) for x in bundle.tokenizer.convert_ids_to_tokens(list(ids))]
    return token_texts(bundle, ids)


class suppress_tokenizer_errors:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        return exc_type is not None


def offsets_for_rendered(bundle: bench.ModelBundle, rendered: str, ids: Sequence[int]) -> list[tuple[int, int]] | None:
    try:
        enc = bundle.tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return None
    offsets = enc.get("offset_mapping") if isinstance(enc, Mapping) else None
    if offsets is None:
        return None
    if offsets and isinstance(offsets[0], list) and offsets and offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    pairs = [(int(a), int(b)) for a, b in offsets]
    if len(pairs) != len(ids):
        return None
    return pairs


def char_span_to_token_span(
    offsets: Sequence[tuple[int, int]] | None,
    char_start: int | None,
    char_end: int | None,
    *,
    fallback: tuple[int, int],
) -> tuple[int, int, str, bool]:
    if offsets is None:
        return fallback[0], fallback[1], "message_segment_fallback_no_offsets", False
    if char_start is None or char_end is None or char_end <= char_start:
        return fallback[0], fallback[1], "message_segment_fallback_no_content_chars", False
    hits = []
    for idx, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if b > char_start and a < char_end:
            hits.append(idx)
    if not hits:
        return fallback[0], fallback[1], "message_segment_fallback_no_overlap", False
    return min(hits), max(hits) + 1, "offset_mapping", True


def build_segments(bundle: bench.ModelBundle, conv: ConversationSpec) -> RenderedConversation:
    rendered = render_messages(bundle, conv.messages)
    full_ids = token_ids(bundle, rendered)
    direct_ids = direct_template_ids(bundle, conv.messages)
    offsets = offsets_for_rendered(bundle, rendered, full_ids)

    segments: list[Segment] = []
    stable_token_prefix = True
    stable_string_prefix = True
    prev_ids: list[int] = []
    prev_rendered = ""
    content_cursor = 0

    for idx, message in enumerate(conv.messages):
        partial_rendered = render_messages(bundle, conv.messages[: idx + 1])
        partial_ids = token_ids(bundle, partial_rendered)
        if partial_ids[: len(prev_ids)] != prev_ids:
            stable_token_prefix = False
        if not partial_rendered.startswith(prev_rendered):
            stable_string_prefix = False

        message_start = len(prev_ids)
        message_end = len(partial_ids)

        # Content char-span: search the FULL rendered string with a forward
        # cursor rather than the per-prefix partial render. Some chat templates
        # (e.g. Olmo-3-Instruct) close the FINAL assistant turn with a
        # different stop token than a non-final one (<|endoftext|> vs
        # <|im_end|>), so a prefix render is not a byte-prefix of the full
        # render and partial-render char offsets drift. The full render is the
        # exact string the model is scored on, so locating content there keeps
        # the content span correct on every template.
        content_text = str(message["content"])
        rel = rendered.find(content_text, content_cursor) if content_text else -1
        if rel >= 0:
            content_start_char = rel
            content_end_char = rel + len(content_text)
            content_cursor = content_end_char
        else:
            content_start_char = None
            content_end_char = None

        # Message char-span from the full render's offset mapping, so the char
        # columns stay consistent with the token boundaries above.
        if offsets is not None and message_end > message_start:
            message_start_char = offsets[message_start][0]
            message_end_char = offsets[message_end - 1][1]
        else:
            message_start_char = len(prev_rendered) if partial_rendered.startswith(prev_rendered) else 0
            message_end_char = len(partial_rendered)

        c_start, c_end, method, found = char_span_to_token_span(
            offsets,
            content_start_char,
            content_end_char,
            fallback=(message_start, message_end),
        )
        if not found and content_start_char is not None:
            # Slow tokenizers may not expose offset mappings. Exact subsequence
            # matching is a fallback only, because BPE context can change the
            # first content token. When it works, downstream labs still get a
            # true content span instead of the broad message span.
            needle_ids = token_ids(bundle, str(message["content"]))
            sub_start, sub_end = find_subsequence(
                full_ids, needle_ids, start=message_start, end=message_end
            )
            if sub_start is not None and sub_end is not None:
                c_start, c_end, method, found = sub_start, sub_end, "token_subsequence_fallback", True
        within_message = message_start <= c_start <= c_end <= message_end
        if not within_message:
            c_start, c_end, method, found = message_start, message_end, f"{method}_outside_message_fallback", False

        segments.append(
            Segment(
                index=idx,
                role=str(message["role"]),
                message_start=message_start,
                message_end=message_end,
                content_start=c_start,
                content_end=c_end,
                message_start_char=message_start_char,
                message_end_char=message_end_char,
                content_start_char=content_start_char,
                content_end_char=content_end_char,
                content=str(message["content"]),
                content_span_found=bool(found),
                content_span_method=method,
            )
        )
        prev_ids = partial_ids
        prev_rendered = partial_rendered

    info = {
        "conversation": conv.name,
        "expected_topic": conv.expected_topic,
        "rendered_token_count": len(full_ids),
        "rendered_char_count": len(rendered),
        "direct_token_count": len(direct_ids),
        "string_vs_direct_template_ids_match": full_ids == direct_ids,
        "incremental_token_prefix_stable": stable_token_prefix,
        "incremental_string_prefix_stable": stable_string_prefix,
        "final_incremental_ids_match": prev_ids == full_ids,
        "final_incremental_string_match": prev_rendered == rendered,
        "offset_mapping_available": offsets is not None,
        "rendered_sha256": sha256_text(rendered),
        "input_ids_sha256": sha256_ids(full_ids),
    }
    return RenderedConversation(conv, rendered, full_ids, segments, info)


def think_span_rows_for_rendered(bundle: bench.ModelBundle, conv: RenderedConversation) -> list[dict[str, Any]]:
    offsets = offsets_for_rendered(bundle, conv.rendered, conv.input_ids)
    rows: list[dict[str, Any]] = []
    for match_idx, match in enumerate(re.finditer(r"<think>(.*?)</think>", conv.rendered, flags=re.DOTALL | re.IGNORECASE)):
        start, end, method, found = char_span_to_token_span(
            offsets,
            match.start(1),
            match.end(1),
            fallback=(0, 0),
        )
        rows.append({
            "conversation": conv.spec.name,
            "think_span_index": match_idx,
            "char_start": match.start(1),
            "char_end": match.end(1),
            "token_start": start,
            "token_end_exclusive": end,
            "n_tokens": max(0, end - start),
            "span_found": found,
            "span_method": method,
            "excerpt": match.group(1)[:100].replace("\n", " "),
        })
    if not rows:
        rows.append({
            "conversation": conv.spec.name,
            "think_span_index": -1,
            "char_start": "",
            "char_end": "",
            "token_start": "",
            "token_end_exclusive": "",
            "n_tokens": 0,
            "span_found": False,
            "span_method": "no_think_markers_present",
            "excerpt": "",
        })
    return rows


def generation_prompt_rows_for_rendered(bundle: bench.ModelBundle, conv: RenderedConversation) -> list[dict[str, Any]]:
    """Check that add_generation_prompt=True extends user-turn prefixes cleanly.

    Later labs often stop after a user message and ask the model to generate an
    assistant turn. This check makes that boundary explicit without generating
    any text.
    """

    rows: list[dict[str, Any]] = []
    for idx, message in enumerate(conv.spec.messages):
        if message.get("role") != "user":
            continue
        prefix = conv.spec.messages[: idx + 1]
        no_prompt_rendered = render_messages(bundle, prefix, add_generation_prompt=False)
        with_prompt_rendered = render_messages(bundle, prefix, add_generation_prompt=True)
        no_prompt_ids = token_ids(bundle, no_prompt_rendered)
        with_prompt_ids = token_ids(bundle, with_prompt_rendered)
        direct_with_prompt_ids = direct_template_ids(
            bundle, prefix, add_generation_prompt=True
        )
        prefix_ok = with_prompt_ids[: len(no_prompt_ids)] == no_prompt_ids
        direct_ok = with_prompt_ids == direct_with_prompt_ids
        rows.append({
            "conversation": conv.spec.name,
            "last_user_segment_index": idx,
            "prefix_tokens_without_generation_prompt": len(no_prompt_ids),
            "tokens_with_generation_prompt": len(with_prompt_ids),
            "generation_prompt_extra_tokens": len(with_prompt_ids) - len(no_prompt_ids),
            "with_prompt_extends_prefix": prefix_ok,
            "string_vs_direct_with_prompt_ids_match": direct_ok,
            "extra_decoded_excerpt": bundle.tokenizer.decode(with_prompt_ids[len(no_prompt_ids):])[:120].replace("\n", "\\n"),
            "ok": prefix_ok and direct_ok,
        })
    return rows


def write_turn_boundary_check(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    specs: Sequence[ConversationSpec],
) -> tuple[dict[str, RenderedConversation], dict[str, Any]]:
    rendered_by_name: dict[str, RenderedConversation] = {}
    conversation_records: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    think_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    all_ok = True

    for spec in specs:
        conv = build_segments(bundle, spec)
        rendered_by_name[spec.name] = conv
        segments = conv.segments
        ids = conv.input_ids
        coverage_ok = bool(segments) and segments[0].message_start == 0 and segments[-1].message_end == len(ids)
        no_gaps = all(a.message_end == b.message_start for a, b in zip(segments, segments[1:]))
        positive_message_widths = all(seg.message_end > seg.message_start for seg in segments)
        content_inside_message = all(seg.message_start <= seg.content_start <= seg.content_end <= seg.message_end for seg in segments)
        no_assistant_leak = True
        leaks: list[str] = []
        for i, seg in enumerate(segments):
            if seg.role != "user":
                continue
            next_assistant = next((s for s in segments[i + 1:] if s.role == "assistant"), None)
            if next_assistant is None:
                continue
            decoded_content = bundle.tokenizer.decode(ids[seg.content_start: seg.content_end])
            snippet = next_assistant.content[: min(28, len(next_assistant.content))]
            if snippet and snippet in decoded_content:
                no_assistant_leak = False
                leaks.append(f"user content span {seg.index} contains assistant snippet {snippet!r}")

        conv_generation_rows = generation_prompt_rows_for_rendered(bundle, conv)
        generation_rows.extend(conv_generation_rows)
        generation_prompt_ok = all(bool(r["ok"]) for r in conv_generation_rows)

        # The trust gate is that each derived content span, decoded back from
        # the full-render token ids, reproduces the message text (whitespace
        # normalized). This validates the spans directly against the string the
        # model is scored on, instead of requiring incremental prefix renders to
        # be byte-identical -- a property legitimate templates violate when the
        # final turn's stop token differs from a non-final one (Olmo-3-Instruct).
        def _norm_ws(s: str) -> str:
            return "".join(s.split())
        content_spans_found = all(seg.content_span_found for seg in segments)
        content_spans_decode_match = all(
            _norm_ws(bundle.tokenizer.decode(ids[seg.content_start: seg.content_end])) == _norm_ws(seg.content)
            for seg in segments
        )
        content_spans_ok = content_spans_found and content_spans_decode_match

        ok = (
            coverage_ok
            and no_gaps
            and positive_message_widths
            and content_inside_message
            and no_assistant_leak
            and generation_prompt_ok
            and content_spans_ok
            and conv.info["string_vs_direct_template_ids_match"]
            and conv.info["final_incremental_ids_match"]
        )
        all_ok = all_ok and ok
        method_counts = Counter(seg.content_span_method for seg in segments)
        conversation_records.append({
            **conv.info,
            "coverage_ok": coverage_ok,
            "no_gaps": no_gaps,
            "positive_message_widths": positive_message_widths,
            "content_inside_message": content_inside_message,
            "no_assistant_text_in_user_content_spans": no_assistant_leak,
            "generation_prompt_ok": generation_prompt_ok,
            "content_spans_found": content_spans_found,
            "content_spans_decode_match": content_spans_decode_match,
            "content_spans_ok": content_spans_ok,
            "content_span_method_counts": dict(method_counts),
            "all_content_spans_offset_mapped": all(seg.content_span_found for seg in segments),
            "incremental_prefix_stable_informational": (
                "incremental_token_prefix_stable is recorded but NOT a trust "
                "gate: templates may re-render a prior turn's stop token when it "
                "is no longer final; content spans are validated directly."
            ),
            "leaks": leaks,
            "ok": ok,
        })
        preview_rows.append({
            "conversation": spec.name,
            "rendered_token_count": len(ids),
            "rendered_char_count": len(conv.rendered),
            "rendered_sha256": conv.info["rendered_sha256"],
            "rendered_prefix": conv.rendered[:500].replace("\n", "\\n"),
            "rendered_suffix": conv.rendered[-500:].replace("\n", "\\n"),
        })
        for seg in segments:
            segment_ids = ids[seg.message_start: seg.message_end]
            content_ids = ids[seg.content_start: seg.content_end]
            segment_rows.append({
                "conversation": spec.name,
                "segment_index": seg.index,
                "role": seg.role,
                "message_start": seg.message_start,
                "message_end_exclusive": seg.message_end,
                "message_boundary_token": seg.boundary_token,
                "message_n_tokens": seg.message_end - seg.message_start,
                "content_start": seg.content_start,
                "content_end_exclusive": seg.content_end,
                "content_boundary_token": seg.content_boundary_token,
                "content_n_tokens": seg.content_end - seg.content_start,
                "content_span_found": seg.content_span_found,
                "content_span_method": seg.content_span_method,
                "message_start_char": seg.message_start_char,
                "message_end_char": seg.message_end_char,
                "content_start_char": seg.content_start_char if seg.content_start_char is not None else "",
                "content_end_char": seg.content_end_char if seg.content_end_char is not None else "",
                "message_decoded_excerpt": bundle.tokenizer.decode(segment_ids)[:140].replace("\n", "\\n"),
                "content_decoded_excerpt": bundle.tokenizer.decode(content_ids)[:140].replace("\n", "\\n"),
                "content_text_excerpt": seg.content[:140].replace("\n", "\\n"),
            })
        think_rows.extend(think_span_rows_for_rendered(bundle, conv))

    result = {
        "ok": all_ok,
        "conversations": conversation_records,
        "explanation": (
            "Message token boundaries are derived from incremental chat-template prefix renders; content spans are "
            "located in the FULL rendered string and mapped with tokenizer offset mapping. The hard trust gates are: "
            "full-vs-direct template token parity, complete coverage, no gaps, no assistant content inside user "
            "content spans, every content span decodes back to its message text, and add_generation_prompt=True "
            "cleanly extends user-turn prefixes. Incremental prefix byte-identity is recorded for information but is "
            "NOT a trust gate: some templates (Olmo-3-Instruct) close the final assistant turn with a different stop "
            "token (<|endoftext|>) than a non-final one (<|im_end|>), so a prefix render is legitimately not a "
            "byte-prefix of the full render."
        ),
    }
    path = ctx.path("diagnostics", "turn_boundary_check.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Chat-template turn segmentation, content spans, and template parity checks.")

    segment_path = ctx.path("tables", "turn_segments.csv")
    bench.write_csv_with_context(ctx, segment_path, segment_rows)
    ctx.register_artifact(segment_path, "table", "Message-level and content-level token spans for each rendered conversation.")

    preview_path = ctx.path("diagnostics", "rendered_conversation_preview.csv")
    bench.write_csv_with_context(ctx, preview_path, preview_rows)
    ctx.register_artifact(preview_path, "diagnostic", "Rendered chat-template previews and hashes.")

    think_path = ctx.path("diagnostics", "think_span_report.csv")
    bench.write_csv_with_context(ctx, think_path, think_rows)
    ctx.register_artifact(think_path, "diagnostic", "Detected <think> spans if present; normally empty for this Lab 15 demo.")

    gen_path = ctx.path("diagnostics", "generation_prompt_boundary_check.csv")
    bench.write_csv_with_context(ctx, gen_path, generation_rows)
    ctx.register_artifact(gen_path, "diagnostic", "Checks that add_generation_prompt=True cleanly extends user-turn prefixes.")

    if not all_ok:
        raise RuntimeError("Turn-boundary/template parity failed; see diagnostics/turn_boundary_check.json.")
    return rendered_by_name, result


def write_generation_prompt_boundary_check(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    conversations_by_name: Mapping[str, RenderedConversation],
) -> dict[str, Any]:
    """Audit the user-ended prefixes later labs use before assistant generation."""

    rows: list[dict[str, Any]] = []
    prefix_ok = True
    direct_ok = True
    any_stub_missing = False
    scripted_prefix_all = True

    for conv in conversations_by_name.values():
        for seg in conv.segments:
            if seg.role != "user":
                continue
            prefix_messages = conv.spec.messages[: seg.index + 1]
            no_gen_rendered = render_messages(bundle, prefix_messages, add_generation_prompt=False)
            gen_rendered = render_messages(bundle, prefix_messages, add_generation_prompt=True)
            no_gen_ids = token_ids(bundle, no_gen_rendered)
            gen_ids = token_ids(bundle, gen_rendered)
            direct_gen_ids = direct_template_ids(bundle, prefix_messages, add_generation_prompt=True)
            this_prefix_ok = gen_ids[: len(no_gen_ids)] == no_gen_ids
            this_direct_ok = gen_ids == direct_gen_ids
            stub_len = len(gen_ids) - len(no_gen_ids) if this_prefix_ok else -1
            stub_missing = stub_len <= 0
            scripted_prefix = len(gen_ids) <= len(conv.input_ids) and conv.input_ids[: len(gen_ids)] == gen_ids
            prefix_ok = prefix_ok and this_prefix_ok
            direct_ok = direct_ok and this_direct_ok
            any_stub_missing = any_stub_missing or stub_missing
            scripted_prefix_all = scripted_prefix_all and scripted_prefix
            rows.append({
                "conversation": conv.spec.name,
                "source_user_segment_index": seg.index,
                "no_generation_prompt_token_count": len(no_gen_ids),
                "with_generation_prompt_token_count": len(gen_ids),
                "assistant_generation_stub_n_tokens": stub_len,
                "assistant_generation_stub_present": not stub_missing,
                "prefix_without_generation_prompt_is_prefix": this_prefix_ok,
                "string_vs_direct_template_ids_match": this_direct_ok,
                "matches_scripted_full_conversation_prefix": scripted_prefix,
                "generation_prompt_boundary_token": len(gen_ids) - 1 if gen_ids else "",
                "rendered_sha256": sha256_text(gen_rendered),
                "generation_prompt_tail": gen_rendered[-220:].replace("\n", "\\n"),
            })

    table_path = ctx.path("tables", "generation_prompt_boundaries.csv")
    bench.write_csv_with_context(ctx, table_path, rows)
    ctx.register_artifact(table_path, "table", "User-ended prefixes rendered with add_generation_prompt=True.")

    result = {
        "ok": bool(prefix_ok and direct_ok),
        "n_user_prefixes": len(rows),
        "prefix_without_generation_prompt_is_prefix_for_all": bool(prefix_ok),
        "string_vs_direct_template_ids_match_for_all": bool(direct_ok),
        "assistant_generation_stub_present_for_all": not any_stub_missing,
        "matches_scripted_full_conversation_prefix_for_all": bool(scripted_prefix_all),
        "warning": (
            "assistant_generation_stub_present_for_all can be false on some templates; this is a warning, not a hard failure. "
            "The hard gate is that add_generation_prompt=True preserves the no-generation prefix and matches direct template tokenization."
        ),
    }
    path = ctx.path("diagnostics", "generation_prompt_boundary_check.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Assistant-generation boundary check for user-ended chat prefixes.")
    if not result["ok"]:
        raise RuntimeError("Generation-prompt boundary check failed; see diagnostics/generation_prompt_boundary_check.json.")
    return result


# ---------------------------------------------------------------------------
# Exact chat prompt residual capture and self-checks
# ---------------------------------------------------------------------------


def model_accepts_kwarg(model: Any, name: str) -> bool:
    try:
        sig = inspect.signature(model.forward)
    except Exception:
        return False
    return name in sig.parameters


def run_ids(
    bundle: bench.ModelBundle,
    ids: Sequence[int],
    *,
    past_key_values: Any = None,
    use_cache: bool = False,
    total_attention_length: int | None = None,
    cache_start: int = 0,
) -> tuple[Any, Any, Any]:
    """Run already-tokenized, already-rendered chat IDs.

    Unlike bench.run_with_residual_cache, this helper never re-tokenizes and
    never adds special tokens. That is the whole point of Lab 15: the measured
    sequence is exactly the chat-template sequence whose turn spans were just
    audited.
    """

    import torch

    if not ids:
        raise ValueError("run_ids received an empty token sequence.")
    input_ids = torch.tensor([list(ids)], device=bundle.input_device)
    total_len = int(total_attention_length or len(ids))
    attention_mask = torch.ones((1, total_len), dtype=torch.long, device=bundle.input_device)
    captured: dict[str, Any] = {}

    def final_norm_pre_hook(module: Any, hook_args: tuple) -> None:
        captured["final_prenorm"] = bench.tensor_cpu_float(hook_args[0])

    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "output_hidden_states": True,
        "use_cache": use_cache,
    }
    if model_accepts_kwarg(bundle.model, "cache_position"):
        kwargs["cache_position"] = torch.arange(
            int(cache_start), int(cache_start) + len(ids), device=bundle.input_device
        )

    handle = bundle.final_norm.register_forward_pre_hook(final_norm_pre_hook)
    try:
        with torch.no_grad():
            out = bundle.model(**kwargs)
    finally:
        handle.remove()
    if "final_prenorm" not in captured:
        raise RuntimeError("Final-norm pre-hook did not fire during multi-turn capture.")
    streams = torch.stack(
        [bench.tensor_cpu_float(h[0]) for h in out.hidden_states[:-1]]
        + [captured["final_prenorm"][0]]
    )
    return streams, bench.tensor_cpu_float(out.logits[0, -1]), getattr(out, "past_key_values", None)


def make_forward_capture_for_ids(
    bundle: bench.ModelBundle,
    rendered: str,
    ids: Sequence[int],
    *,
    streams: Any | None = None,
    final_logits_last: Any | None = None,
) -> bench.ForwardCapture:
    if streams is None or final_logits_last is None:
        streams, final_logits_last, _ = run_ids(bundle, ids, use_cache=False)
    return bench.ForwardCapture(
        prompt=rendered,
        input_ids=[int(x) for x in ids],
        tokens_raw=token_pieces(bundle, ids),
        tokens_text=token_texts(bundle, ids),
        streams=streams,
        final_logits_last=final_logits_last,
    )


def write_chat_exact_hook_parity_check(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    conv: RenderedConversation,
) -> tuple[dict[str, Any], Any, Any]:
    block_outputs: dict[int, Any] = {}

    def make_hook(idx: int):
        def hook(module: Any, hook_args: tuple, output: Any) -> None:
            out = output[0] if isinstance(output, tuple) else output
            block_outputs[idx] = bench.tensor_cpu_float(out)

        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(bundle.blocks)]
    try:
        streams, final_logits, _ = run_ids(bundle, conv.input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    n_layers = bundle.anatomy.n_layers
    by_layer_rows: list[dict[str, Any]] = []
    missing_layers: list[int] = []
    max_diff = 0.0
    max_mean_diff = 0.0
    compared = 0
    for layer in range(n_layers):
        if layer not in block_outputs:
            missing_layers.append(layer)
            continue
        hook_out = block_outputs[layer][0]
        expected = streams[layer + 1]
        abs_diff = (hook_out - expected).abs()
        layer_max = float(abs_diff.max())
        layer_mean = float(abs_diff.mean())
        max_diff = max(max_diff, layer_max)
        max_mean_diff = max(max_mean_diff, layer_mean)
        compared += 1
        by_layer_rows.append({
            "conversation": conv.spec.name,
            "layer": layer,
            "stream_depth_expected": layer + 1,
            "max_abs_diff": layer_max,
            "mean_abs_diff": layer_mean,
            "hook_l2": float(hook_out.norm()),
            "expected_l2": float(expected.norm()),
            "ok_at_tolerance": layer_max <= float(getattr(ctx.args, "hook_tolerance", 0.0)),
        })

    by_layer_path = ctx.path("diagnostics", "chat_exact_hook_parity_by_layer.csv")
    bench.write_csv_with_context(ctx, by_layer_path, by_layer_rows)
    ctx.register_artifact(by_layer_path, "diagnostic", "Block-hook versus stream-depth parity on exact chat-template token IDs.")

    tolerance = float(getattr(ctx.args, "hook_tolerance", 0.0))
    ok = (not missing_layers) and compared == n_layers and max_diff <= tolerance
    result = {
        "conversation": conv.spec.name,
        "input_ids_sha256": conv.info["input_ids_sha256"],
        "blocks_compared": compared,
        "n_layers": n_layers,
        "missing_layers": missing_layers,
        "max_abs_diff": max_diff,
        "max_mean_abs_diff": max_mean_diff,
        "tolerance": tolerance,
        "ok": bool(ok),
        "allow_hook_mismatch": bool(getattr(ctx.args, "allow_hook_mismatch", False)),
        "explanation": (
            "This is the bench hook-parity check repeated on already-rendered chat IDs with add_special_tokens=False. "
            "Block k's output must equal streams[k+1]. This catches chat-template BOS drift and stream-depth off-by-one errors."
        ),
    }
    path = ctx.path("diagnostics", "chat_exact_hook_parity.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Exact-chat residual hook parity check.")
    if not ok and not bool(getattr(ctx.args, "allow_hook_mismatch", False)):
        raise RuntimeError("Exact chat hook parity failed; see diagnostics/chat_exact_hook_parity.json.")
    return result, streams, final_logits


def write_exact_lens_check(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    conv: RenderedConversation,
    streams: Any,
    logits: Any,
) -> dict[str, Any]:
    capture = make_forward_capture_for_ids(
        bundle,
        conv.rendered,
        conv.input_ids,
        streams=streams,
        final_logits_last=logits,
    )
    result = bench.run_lens_self_check(ctx, bundle, capture)
    exact_path = ctx.path("diagnostics", "chat_exact_lens_self_check.json")
    bench.write_json(exact_path, {
        **result,
        "conversation": conv.spec.name,
        "input_ids_sha256": conv.info["input_ids_sha256"],
        "note": "Alias copy of the standard lens self-check, run on exact chat-template token IDs.",
    })
    ctx.register_artifact(exact_path, "diagnostic", "Final-depth logit-lens self-check on exact chat-template token IDs.")
    return result


# ---------------------------------------------------------------------------
# KV-cache parity
# ---------------------------------------------------------------------------


def full_recompute_boundary_states(
    bundle: bench.ModelBundle,
    ids: Sequence[int],
    boundaries: Sequence[int],
) -> list[tuple[Any, Any]]:
    out: list[tuple[Any, Any]] = []
    for boundary in boundaries:
        streams, logits, _ = run_ids(bundle, ids[: boundary + 1], use_cache=False)
        out.append((streams[:, -1, :], logits))
    return out


def incremental_boundary_states(
    bundle: bench.ModelBundle,
    ids: Sequence[int],
    boundaries: Sequence[int],
) -> list[tuple[Any, Any]]:
    out: list[tuple[Any, Any]] = []
    past = None
    prev = 0
    for boundary in boundaries:
        chunk = ids[prev: boundary + 1]
        if not chunk:
            raise RuntimeError("Empty incremental chunk while computing KV-cache parity.")
        total_len = boundary + 1
        streams, logits, past = run_ids(
            bundle,
            chunk,
            past_key_values=past,
            use_cache=True,
            total_attention_length=total_len,
            cache_start=prev,
        )
        if past is None:
            raise RuntimeError("Model returned no past_key_values under use_cache=True.")
        out.append((streams[:, -1, :], logits))
        prev = boundary + 1
    return out


def write_cache_parity_check(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    conversations_by_name: Mapping[str, RenderedConversation],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    worst_hidden = 0.0
    worst_mean_hidden = 0.0
    worst_rel_hidden = 0.0
    worst_logits = 0.0
    worst_record: dict[str, Any] | None = None

    import torch
    low_precision = next(bundle.model.parameters()).dtype in (torch.bfloat16, torch.float16)
    rel_l2_thresh = CACHE_PARITY_REL_L2_LOWP if low_precision else CACHE_PARITY_REL_L2

    for conv_name, conv in conversations_by_name.items():
        boundaries = [seg.boundary_token for seg in conv.segments]
        full = full_recompute_boundary_states(bundle, conv.input_ids, boundaries)
        cached = incremental_boundary_states(bundle, conv.input_ids, boundaries)
        for seg, (full_stream, full_logits), (cached_stream, cached_logits) in zip(conv.segments, full, cached):
            hidden_diff = (full_stream - cached_stream).abs()
            logit_diff = (full_logits - cached_logits).abs()
            # The gate is the WHOLE-VECTOR relative L2 error, not a per-dim max.
            # In bf16 a single dimension can round-divide (here ~0.22 abs on a
            # ~0.09-magnitude dim) while every other dim agrees to ~1e-3; that
            # one dim is irrelevant to a turn-trace projection. A real cache bug
            # (off-by-one position/window) instead perturbs the whole vector, so
            # relative L2 catches it. Cosine is reported as a second view.
            fv = full_stream.float().flatten()
            cv = cached_stream.float().flatten()
            rel_l2 = float((fv - cv).norm() / (fv.norm() + 1e-6))
            cos = float(torch.nn.functional.cosine_similarity(fv, cv, dim=0))
            within_tol = rel_l2 <= rel_l2_thresh
            row = {
                "conversation": conv_name,
                "segment_index": seg.index,
                "role": seg.role,
                "turn_index_non_system": role_turn_index(conv.segments, seg.index),
                "boundary_token": seg.boundary_token,
                "rel_l2_hidden_diff": rel_l2,
                "cosine_full_vs_cached": cos,
                "max_abs_hidden_diff": float(hidden_diff.max()),
                "mean_abs_hidden_diff": float(hidden_diff.mean()),
                "max_abs_logit_diff": float(logit_diff.max()),
                "mean_abs_logit_diff": float(logit_diff.mean()),
                "ok_at_tolerance": within_tol,
            }
            rows.append(row)
            if worst_record is None or rel_l2 > worst_rel_hidden:
                worst_record = row
            worst_hidden = max(worst_hidden, float(hidden_diff.max()))
            worst_mean_hidden = max(worst_mean_hidden, float(hidden_diff.mean()))
            worst_rel_hidden = max(worst_rel_hidden, rel_l2)
            worst_logits = max(worst_logits, float(logit_diff.max()))

    csv_path = ctx.path("diagnostics", "cache_recompute_parity_by_boundary.csv")
    bench.write_csv_with_context(ctx, csv_path, rows)
    ctx.register_artifact(csv_path, "diagnostic", "Boundary-level KV-cache residuals versus full recompute.")

    result = {
        "n_conversations": len(conversations_by_name),
        "n_boundaries": len(rows),
        "max_abs_hidden_diff": worst_hidden,
        "max_mean_hidden_diff": worst_mean_hidden,
        "max_rel_l2_hidden_diff": worst_rel_hidden,
        "max_abs_logit_diff": worst_logits,
        "rel_l2_threshold": rel_l2_thresh,
        "dtype_low_precision": low_precision,
        "ok": worst_rel_hidden <= rel_l2_thresh,
        "worst_record": worst_record,
        "explanation": (
            "Each conversation prefix was measured two ways: full recompute of the prefix and incremental prefill "
            "with past_key_values. The residual stream at each message boundary must match the full recompute in "
            "whole-vector relative L2 error ||full-cached||/||full|| <= threshold (dtype-aware: bf16/fp16 round "
            "differently than fp32 and a single outlier dim can diverge while the rest agree). A structural cache "
            "bug perturbs the whole vector and fails this; one bf16-divergent dim does not. Logit diffs are an "
            "extra smoke check."
        ),
    }
    path = ctx.path("diagnostics", "cache_recompute_parity.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "KV-cache-aware boundary capture parity against full recompute.")
    if not result["ok"]:
        raise RuntimeError("KV-cache parity failed; see diagnostics/cache_recompute_parity.json.")
    return result


# ---------------------------------------------------------------------------
# Self-patching no-op
# ---------------------------------------------------------------------------


def logits_with_self_patch(
    bundle: bench.ModelBundle,
    ids: Sequence[int],
    *,
    layer: int,
    pos: int,
    vector: Any,
) -> Any:
    import torch

    input_ids = torch.tensor([list(ids)], device=bundle.input_device)
    attention_mask = torch.ones((1, len(ids)), dtype=torch.long, device=bundle.input_device)
    block = bundle.blocks[int(layer)]

    def patch_hook(module: Any, hook_args: tuple, output: Any) -> Any:
        out = output[0] if isinstance(output, tuple) else output
        patched = out.clone()
        patched[:, int(pos), :] = vector.to(patched.device, patched.dtype)
        if isinstance(output, tuple):
            return (patched,) + tuple(output[1:])
        return patched

    handle = block.register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return bench.tensor_cpu_float(out.logits[0, -1])


def patch_layers_for_model(bundle: bench.ModelBundle) -> list[int]:
    n = bundle.anatomy.n_layers
    if n <= 1:
        return [0]
    layers = sorted({min(n - 1, max(0, int(round(frac * (n - 1))))) for frac in PATCH_LAYER_FRACTIONS})
    if not layers:
        layers = [min(n - 1, max(0, n // 2))]
    return layers


def patch_positions_for_conversation(conv: RenderedConversation) -> list[tuple[str, int, int]]:
    user_segments = [seg for seg in conv.segments if seg.role == "user"]
    assistant_segments = [seg for seg in conv.segments if seg.role == "assistant"]
    sites: list[tuple[str, int, int]] = []
    if user_segments:
        sites.append(("last_user_message_boundary", user_segments[-1].index, user_segments[-1].boundary_token))
        sites.append(("last_user_content_boundary", user_segments[-1].index, user_segments[-1].content_boundary_token))
    if assistant_segments:
        sites.append(("final_assistant_message_boundary", assistant_segments[-1].index, assistant_segments[-1].boundary_token))
    # Deduplicate by token position while keeping names informative.
    seen: set[int] = set()
    unique: list[tuple[str, int, int]] = []
    for name, seg_idx, pos in sites:
        if pos not in seen:
            unique.append((name, seg_idx, pos))
            seen.add(pos)
    return unique


def write_patch_noop_check(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    conversations_by_name: Mapping[str, RenderedConversation],
    full_stream_cache: Mapping[str, tuple[Any, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    worst = 0.0
    worst_mean = 0.0
    worst_row: dict[str, Any] | None = None
    layers = patch_layers_for_model(bundle)

    for conv_name, conv in conversations_by_name.items():
        streams, base_logits = full_stream_cache[conv_name]
        for site_name, segment_index, pos in patch_positions_for_conversation(conv):
            for layer in layers:
                stream_depth = layer + 1
                vector = streams[stream_depth, pos]
                patched_logits = logits_with_self_patch(bundle, conv.input_ids, layer=layer, pos=pos, vector=vector)
                diff = (patched_logits - base_logits).abs()
                wrong_depth = max(0, stream_depth - 1)
                wrong_vector_l2 = float((streams[stream_depth, pos] - streams[wrong_depth, pos]).norm())
                row = {
                    "conversation": conv_name,
                    "site": site_name,
                    "segment_index": segment_index,
                    "position": pos,
                    "layer": layer,
                    "patched_stream_depth": stream_depth,
                    "max_abs_logit_diff": float(diff.max()),
                    "mean_abs_logit_diff": float(diff.mean()),
                    "ok_at_tolerance": float(diff.max()) <= PATCH_NOOP_ATOL,
                    "wrong_depth_sentinel_depth": wrong_depth,
                    "wrong_depth_vector_l2_delta": wrong_vector_l2,
                }
                rows.append(row)
                if float(diff.max()) > worst:
                    worst_row = row
                worst = max(worst, float(diff.max()))
                worst_mean = max(worst_mean, float(diff.mean()))

    csv_path = ctx.path("diagnostics", "patch_noop_sites.csv")
    bench.write_csv_with_context(ctx, csv_path, rows)
    ctx.register_artifact(csv_path, "diagnostic", "Self-patching no-op results by layer and turn-boundary site.")

    result = {
        "n_sites": len(rows),
        "layers_tested": layers,
        "max_abs_logit_diff": worst,
        "max_mean_abs_logit_diff": worst_mean,
        "atol": PATCH_NOOP_ATOL,
        "ok": worst <= PATCH_NOOP_ATOL,
        "worst_record": worst_row,
        "stream_depth_convention": "patching block layer k targets streams[k+1], the block output after layer k",
        "explanation": (
            "A decoder block output at real turn-boundary sites was replaced with the same run's cached vector. "
            "This should be an identity operation. The wrong-depth sentinel column records how different streams[k] "
            "and streams[k+1] are at the site; it is a smoke alarm for off-by-one patching, not a hard gate."
        ),
    }
    path = ctx.path("diagnostics", "patch_noop_check.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Self-patching turn-boundary block outputs is a no-op.")
    if not result["ok"]:
        raise RuntimeError("Turn-boundary patch no-op failed; see diagnostics/patch_noop_check.json.")
    return result


# ---------------------------------------------------------------------------
# Topic and null directions
# ---------------------------------------------------------------------------


def final_streams_for_messages(bundle: bench.ModelBundle, messages: Sequence[Mapping[str, str]]) -> Any:
    rendered = render_messages(bundle, messages)
    ids = token_ids(bundle, rendered)
    streams, _, _ = run_ids(bundle, ids, use_cache=False)
    return streams[:, -1, :]


def build_trace_direction_bank(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    seed: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    import torch

    topic_pairs = direction_contrast_pairs()
    null_pairs = length_null_pairs()

    topic_vectors: list[Any] = []
    null_vectors: list[Any] = []
    pair_records: list[dict[str, Any]] = []

    for pair in topic_pairs:
        pos = final_streams_for_messages(bundle, pair.positive_messages)
        neg = final_streams_for_messages(bundle, pair.negative_messages)
        diff = pos - neg
        topic_vectors.append(diff)
        pair_records.append({
            "pair_id": pair.pair_id,
            "direction_family": "topic_orchid_minus_archive",
            "positive_topic": pair.positive_topic,
            "negative_topic": pair.negative_topic,
            "note": pair.note,
            "positive_rendered_sha256": sha256_text(render_messages(bundle, pair.positive_messages)),
            "negative_rendered_sha256": sha256_text(render_messages(bundle, pair.negative_messages)),
        })
    for pair in null_pairs:
        pos = final_streams_for_messages(bundle, pair.positive_messages)
        neg = final_streams_for_messages(bundle, pair.negative_messages)
        diff = pos - neg
        null_vectors.append(diff)
        pair_records.append({
            "pair_id": pair.pair_id,
            "direction_family": "length_matched_null",
            "positive_topic": pair.positive_topic,
            "negative_topic": pair.negative_topic,
            "note": pair.note,
            "positive_rendered_sha256": sha256_text(render_messages(bundle, pair.positive_messages)),
            "negative_rendered_sha256": sha256_text(render_messages(bundle, pair.negative_messages)),
        })

    n_depths = bundle.anatomy.n_layers + 1
    bank: dict[int, dict[str, Any]] = {}
    depth_manifest_rows: list[dict[str, Any]] = []
    for depth in range(n_depths):
        topic_stack = torch.stack([v[depth] for v in topic_vectors])
        null_stack = torch.stack([v[depth] for v in null_vectors])
        topic_mean = topic_stack.mean(dim=0)
        null_mean = null_stack.mean(dim=0)
        dirs: dict[str, Any] = {
            "topic_orchid_minus_archive": unit(topic_mean),
            "length_matched_null": unit(null_mean),
        }
        for idx in range(N_RANDOM_NULL_DIRECTIONS):
            dirs[f"random_null_{idx:02d}"] = random_unit(bundle.anatomy.d_model, int(seed) + 10_000 + 101 * depth + idx)
        bank[depth] = dirs
        depth_manifest_rows.append({
            "stream_depth": depth,
            "depth_fraction": rounded(depth / max(1, n_depths - 1)),
            "topic_preunit_norm": float(topic_mean.norm()),
            "length_null_preunit_norm": float(null_mean.norm()),
            "topic_pair_mean_cosine": rounded(mean_pair_cosine(topic_stack)),
            "length_null_pair_mean_cosine": rounded(mean_pair_cosine(null_stack)),
        })

    manifest = {
        "n_depths": n_depths,
        "n_topic_pairs": len(topic_pairs),
        "n_length_null_pairs": len(null_pairs),
        "n_random_null_directions": N_RANDOM_NULL_DIRECTIONS,
        "pairs": pair_records,
        "explanation": (
            "Directions are rebuilt at each stream depth from matched mini-dialogues. The topic direction is "
            "orchid minus archive; the length null uses neutral matched contrasts; random nulls are isotropic unit vectors."
        ),
    }
    depth_path = ctx.path("diagnostics", "trace_direction_depth_manifest.csv")
    bench.write_csv_with_context(ctx, depth_path, depth_manifest_rows)
    ctx.register_artifact(depth_path, "diagnostic", "Direction norms and pair agreement by stream depth.")
    return bank, manifest


def mean_pair_cosine(stack: Any) -> float:
    import torch

    if stack.shape[0] < 2:
        return float("nan")
    units = stack / stack.norm(dim=1, keepdim=True).clamp_min(1e-9)
    vals = []
    for i in range(units.shape[0]):
        for j in range(i + 1, units.shape[0]):
            vals.append(float((units[i] * units[j]).sum()))
    return float(statistics.fmean(vals)) if vals else float("nan")


def write_trace_direction_manifest(
    ctx: bench.RunContext,
    bank: Mapping[int, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    trace_depth: int,
) -> dict[str, Any]:
    directions = bank[int(trace_depth)]
    cosine_rows: list[dict[str, Any]] = []
    names = sorted(directions)
    for a in names:
        for b in names:
            if a > b:
                continue
            cosine_rows.append({
                "stream_depth": trace_depth,
                "direction_a": a,
                "direction_b": b,
                "cosine": rounded(float(directions[a] @ directions[b])),
            })
    cosine_path = ctx.path("tables", "trace_direction_cosines.csv")
    bench.write_csv_with_context(ctx, cosine_path, cosine_rows)
    ctx.register_artifact(cosine_path, "table", "Cosines among topic, length-null, and random-null directions at the trace depth.")

    payload = {
        **dict(manifest),
        "trace_depth": trace_depth,
        "trace_depth_fraction": trace_depth / max(1, int(manifest["n_depths"]) - 1),
        "direction_names_at_trace_depth": names,
    }
    path = ctx.path("diagnostics", "trace_direction_manifest.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "How the topic and null directions were built.")
    return payload


# ---------------------------------------------------------------------------
# Projection traces
# ---------------------------------------------------------------------------


def compute_full_stream_cache(
    bundle: bench.ModelBundle,
    conversations_by_name: Mapping[str, RenderedConversation],
) -> dict[str, tuple[Any, Any]]:
    out: dict[str, tuple[Any, Any]] = {}
    for name, conv in conversations_by_name.items():
        streams, logits, _ = run_ids(bundle, conv.input_ids, use_cache=False)
        out[name] = (streams, logits)
    return out


def projection_rows(
    bundle: bench.ModelBundle,
    conversations_by_name: Mapping[str, RenderedConversation],
    full_stream_cache: Mapping[str, tuple[Any, Any]],
    directions: Mapping[str, Any],
    depth: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for conv_name, conv in conversations_by_name.items():
        streams, _ = full_stream_cache[conv_name]
        for seg in conv.segments:
            turn_idx = role_turn_index(conv.segments, seg.index)
            message_span = streams[depth, seg.message_start: seg.message_end, :]
            content_span = streams[depth, seg.content_start: seg.content_end, :]
            message_span_mean = message_span.mean(dim=0)
            content_span_mean = content_span.mean(dim=0)
            boundary = streams[depth, seg.boundary_token, :]
            content_boundary = streams[depth, seg.content_boundary_token, :]
            cumulative = streams[depth, : seg.message_end, :].mean(dim=0)
            for direction_name, direction in directions.items():
                rows.append({
                    "conversation": conv_name,
                    "segment_index": seg.index,
                    "turn_index_non_system": turn_idx,
                    "role": seg.role,
                    "message_start_token": seg.message_start,
                    "message_end_exclusive": seg.message_end,
                    "content_start_token": seg.content_start,
                    "content_end_exclusive": seg.content_end,
                    "n_message_tokens": seg.message_end - seg.message_start,
                    "n_content_tokens": seg.content_end - seg.content_start,
                    "content_span_method": seg.content_span_method,
                    "stream_depth": depth,
                    "direction": direction_name,
                    "message_span_mean_projection": rounded(float(message_span_mean @ direction)),
                    "content_span_mean_projection": rounded(float(content_span_mean @ direction)),
                    "message_boundary_projection": rounded(float(boundary @ direction)),
                    "content_boundary_projection": rounded(float(content_boundary @ direction)),
                    "cumulative_message_mean_projection": rounded(float(cumulative @ direction)),
                    "content_excerpt": seg.content[:100].replace("\n", " "),
                })
    return rows


def slope_rows_from_trace(rows: Sequence[Mapping[str, Any]], value_field: str) -> list[dict[str, Any]]:
    slope_rows: list[dict[str, Any]] = []
    conversations = sorted({str(r["conversation"]) for r in rows})
    directions = sorted({str(r["direction"]) for r in rows})
    for conv in conversations:
        for direction in directions:
            sub = [
                r for r in rows
                if r["conversation"] == conv and r["direction"] == direction and r["role"] != "system"
            ]
            xs = [float(r["turn_index_non_system"]) for r in sub]
            ys = [float(r[value_field]) for r in sub]
            slope_rows.append({
                "conversation": conv,
                "direction": direction,
                "value_field": value_field,
                "slope": rounded(slope(xs, ys)),
                "start_projection": rounded(ys[0] if ys else float("nan")),
                "end_projection": rounded(ys[-1] if ys else float("nan")),
                "delta_end_minus_start": rounded((ys[-1] - ys[0]) if len(ys) >= 2 else float("nan")),
                "n_points": len(ys),
            })
    return slope_rows


def write_null_trace_check(
    ctx: bench.RunContext,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    slope_rows = slope_rows_from_trace(rows, "cumulative_message_mean_projection")
    csv_path = ctx.path("diagnostics", "null_trace_slopes.csv")
    bench.write_csv_with_context(ctx, csv_path, slope_rows)
    ctx.register_artifact(csv_path, "diagnostic", "Projection slopes for topic and null directions.")

    finite = [
        float(r["slope"])
        for r in slope_rows
        if isinstance(r["slope"], (int, float)) and math.isfinite(float(r["slope"]))
    ]
    topic_slope = get_slope_value(slope_rows, "orchid_topic", "topic_orchid_minus_archive")
    topic_control_slope = get_slope_value(slope_rows, "archive_length_control", "topic_orchid_minus_archive")
    length_null_slope = get_slope_value(slope_rows, "orchid_topic", "length_matched_null")
    random_slopes = [
        float(r["slope"])
        for r in slope_rows
        if str(r["conversation"]) == "orchid_topic" and str(r["direction"]).startswith("random_null_")
        and isinstance(r["slope"], (int, float)) and math.isfinite(float(r["slope"]))
    ]
    topic_gap = abs(topic_slope - topic_control_slope) if math.isfinite(topic_slope) and math.isfinite(topic_control_slope) else float("nan")
    null_threshold = max(NULL_SLOPE_ABS_WARN, NULL_SLOPE_WARN_RATIO * abs(topic_slope)) if math.isfinite(topic_slope) else NULL_SLOPE_ABS_WARN
    length_null_abs = abs(length_null_slope) if math.isfinite(length_null_slope) else float("inf")
    random_max_abs = safe_max_abs(random_slopes, default=float("inf"))
    flat_enough_for_demo = bool(length_null_abs <= null_threshold and random_max_abs <= null_threshold)

    result = {
        "ok": len(finite) == len(slope_rows),
        "flat_enough_for_demo_claim": flat_enough_for_demo,
        "topic_slope": rounded(topic_slope),
        "topic_on_control_slope": rounded(topic_control_slope),
        "topic_gap_vs_control": rounded(topic_gap),
        "length_null_slope": rounded(length_null_slope),
        "random_null_mean_abs_slope": rounded(safe_fmean([abs(x) for x in random_slopes])),
        "random_null_max_abs_slope": rounded(random_max_abs),
        "null_flatness_threshold": rounded(null_threshold),
        "n_random_nulls": len(random_slopes),
        "explanation": (
            "Finite slopes are a hard instrumentation check. Null flatness is a demo-claim gate: if the length or random "
            "null slopes are large, later multi-turn science must strengthen length/template controls before claiming drift."
        ),
    }
    path = ctx.path("diagnostics", "null_trace_check.json")
    bench.write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Null projection trace sanity and flatness audit.")
    if not result["ok"]:
        raise RuntimeError("Null trace slopes were non-finite; see diagnostics/null_trace_check.json.")
    return result


def get_slope_value(slope_rows: Sequence[Mapping[str, Any]], conv: str, direction: str) -> float:
    for r in slope_rows:
        if r.get("conversation") == conv and r.get("direction") == direction:
            try:
                return float(r.get("slope"))
            except Exception:
                return float("nan")
    return float("nan")


def trace_depth_sweep_rows(
    conversations_by_name: Mapping[str, RenderedConversation],
    full_stream_cache: Mapping[str, tuple[Any, Any]],
    bank: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directions_to_report = ["topic_orchid_minus_archive", "length_matched_null", "random_null_00"]
    for depth, dirs in sorted(bank.items()):
        for conv_name, conv in conversations_by_name.items():
            streams, _ = full_stream_cache[conv_name]
            for direction_name in directions_to_report:
                direction = dirs[direction_name]
                sub_x: list[float] = []
                sub_y: list[float] = []
                for seg in conv.segments:
                    if seg.role == "system":
                        continue
                    turn_idx = role_turn_index(conv.segments, seg.index)
                    cumulative = streams[depth, : seg.message_end, :].mean(dim=0)
                    sub_x.append(float(turn_idx))
                    sub_y.append(float(cumulative @ direction))
                rows.append({
                    "stream_depth": depth,
                    "depth_fraction": rounded(depth / max(1, len(bank) - 1)),
                    "conversation": conv_name,
                    "direction": direction_name,
                    "cumulative_projection_slope": rounded(slope(sub_x, sub_y)),
                    "start_projection": rounded(sub_y[0] if sub_y else float("nan")),
                    "end_projection": rounded(sub_y[-1] if sub_y else float("nan")),
                    "delta_end_minus_start": rounded((sub_y[-1] - sub_y[0]) if len(sub_y) >= 2 else float("nan")),
                    "n_points": len(sub_y),
                })
    return rows



# ---------------------------------------------------------------------------
# Visualization upgrade: turn-indexed science starts with plumbing receipts
# ---------------------------------------------------------------------------

LAB15_ROLE_ORDER = ("system", "user", "assistant")
LAB15_DIRECTION_FAMILIES = ("topic", "topic_on_control", "length_null", "random_null")


def lab15_color(key: str, default: str = "#555555") -> str:
    helper = getattr(bench, "plot_multiturn_color", None)
    if callable(helper):
        try:
            return helper(key, default)
        except TypeError:
            return helper(key)
    palette = {
        "pass": "#009E73",
        "warn": "#E69F00",
        "fail": "#D55E00",
        "unknown": "#777777",
        "system": "#777777",
        "user": "#0072B2",
        "assistant": "#CC79A7",
        "message": "#9E9E9E",
        "content": "#56B4E9",
        "template": "#D0D0D0",
        "topic": "#009E73",
        "topic_orchid_minus_archive": "#009E73",
        "topic_on_control": "#0072B2",
        "archive_control": "#0072B2",
        "length_null": "#E69F00",
        "length_matched_null": "#E69F00",
        "random_null": "#777777",
        "cache": "#0072B2",
        "patch": "#CC79A7",
        "generation_prompt": "#7E57C2",
        "span": "#56B4E9",
        "hard_gate": "#009E73",
        "soft_gate": "#E69F00",
        "downstream": "#7E57C2",
    }
    return palette.get(str(key), default)


def lab15_marker(key: str, default: str = "o") -> str:
    helper = getattr(bench, "plot_multiturn_marker", None)
    if callable(helper):
        try:
            return helper(key, default)
        except TypeError:
            return helper(key)
    return {
        "system": "s",
        "user": "o",
        "assistant": "^",
        "topic": "o",
        "length_null": "D",
        "random_null": "x",
        "cache": "o",
        "patch": "s",
    }.get(str(key), default)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        val = float(x)
    except Exception:
        return default
    return val if math.isfinite(val) else default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "ok", "pass", "passed"}


def _read_csv_rows(path: Any) -> list[dict[str, Any]]:
    import csv
    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k, v in list(row.items()):
            if isinstance(v, str):
                s = v.strip()
                if s.lower() in {"true", "false"}:
                    row[k] = s.lower() == "true"
                else:
                    try:
                        if s and re.fullmatch(r"[-+]?\d+", s):
                            row[k] = int(s)
                        elif s and re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s):
                            row[k] = float(s)
                    except Exception:
                        pass
    return rows


def _short_check_name(name: str) -> str:
    mapping = {
        "turn_boundary_check": "turn spans",
        "generation_prompt_boundary_check": "generation boundary",
        "chat_exact_hook_parity": "exact hooks",
        "chat_exact_lens_self_check": "exact lens",
        "cache_recompute_parity": "cache parity",
        "patch_noop_check": "patch no-op",
        "null_trace_check": "null trace",
        "bench_registry": "bench registry",
    }
    return mapping.get(name, str(name).replace("_", " "))


def _status_from_bool(ok: Any, warn: bool = False) -> str:
    if not _truthy(ok):
        return "fail"
    return "warn" if warn else "pass"


def _status_score(status: str) -> float:
    return {"pass": 1.0, "warn": 0.55, "fail": 0.05, "unknown": 0.35}.get(str(status), 0.35)


def _artifact_for_check(name: str) -> str:
    return {
        "turn_boundary_check": "diagnostics/turn_boundary_check.json + tables/turn_segments.csv",
        "generation_prompt_boundary_check": "diagnostics/generation_prompt_boundary_check.json + tables/generation_prompt_boundaries.csv",
        "chat_exact_hook_parity": "diagnostics/chat_exact_hook_parity.json",
        "chat_exact_lens_self_check": "diagnostics/chat_exact_lens_self_check.json",
        "cache_recompute_parity": "diagnostics/cache_recompute_parity.json + diagnostics/cache_recompute_parity_by_boundary.csv",
        "patch_noop_check": "diagnostics/patch_noop_check.json + diagnostics/patch_noop_sites.csv",
        "null_trace_check": "diagnostics/null_trace_check.json + diagnostics/null_trace_slopes.csv",
        "bench_registry": "diagnostics/bench_integration_note.json",
    }.get(name, "")


def build_harness_evidence_rows(metrics: Mapping[str, Any], checks: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(name: str, *, gate: str, ok: Any, observed: Any = "", threshold: Any = "", warning: bool = False, why: str = "") -> None:
        status = _status_from_bool(ok, warning)
        rows.append({
            "check": name,
            "display_name": _short_check_name(name),
            "gate_type": gate,
            "status": status,
            "status_score": _status_score(status),
            "observed": rounded(observed) if isinstance(observed, (int, float)) else observed,
            "threshold": rounded(threshold) if isinstance(threshold, (int, float)) else threshold,
            "artifact": _artifact_for_check(name),
            "why_it_matters": why,
        })

    turn = checks.get("turn_boundary_check", {})
    gen = checks.get("generation_prompt_boundary_check", {})
    hook = checks.get("chat_exact_hook_parity", {})
    lens = checks.get("chat_exact_lens_self_check", {})
    cache = checks.get("cache_recompute_parity", {})
    patch = checks.get("patch_noop_check", {})
    null = checks.get("null_trace_check", {})

    span_methods = metrics.get("content_span_method_counts", {}) or {}
    fallback_count = sum(int(v) for k, v in span_methods.items() if "fallback" in str(k).lower() or "message" in str(k).lower())
    total_spans = sum(int(v) for v in span_methods.values()) if isinstance(span_methods, Mapping) else 0
    content_warning = bool(fallback_count > 0)

    add("turn_boundary_check", gate="hard", ok=turn.get("ok", False), observed=metrics.get("n_segments_total", ""), threshold="all spans covered", why="Turn-indexed states are meaningless if rendered-message spans have gaps or leaks.")
    add("content_span_mapping", gate="hard-ish", ok=turn.get("ok", False), observed=f"{total_spans - fallback_count}/{total_spans} precise", threshold="decoded content matches", warning=content_warning, why="Later content-specific claims need content spans, not role/template scaffolding.")
    add("generation_prompt_boundary_check", gate="hard", ok=gen.get("ok", False), observed=gen.get("n_user_prefixes", ""), threshold="prefix + direct ids match", why="Generation-time probes often live after an assistant header, not after raw user text.")
    add("chat_exact_hook_parity", gate="hard", ok=hook.get("ok", False), observed=hook.get("max_abs_diff", float("nan")), threshold=hook.get("tolerance", ""), why="Block hooks must equal the named residual stream on exact rendered chat IDs.")
    add("chat_exact_lens_self_check", gate="hard", ok=lens.get("ok", True), observed=lens.get("max_abs_diff", lens.get("max_abs_logit_diff", "")), threshold=lens.get("tolerance", "bench tolerance"), why="Final-depth readout must match the model's real output on the same rendered prompt.")
    add("cache_recompute_parity", gate="hard", ok=cache.get("ok", False), observed=cache.get("max_rel_l2_hidden_diff", metrics.get("cache_parity_max_abs_hidden_diff", float("nan"))), threshold=cache.get("rel_l2_threshold", "dtype-aware"), why="Cached turn traces can counterfeit drift unless they match full recompute.")
    add("patch_noop_check", gate="hard", ok=patch.get("ok", False), observed=patch.get("max_abs_logit_diff", metrics.get("patch_noop_max_abs_logit_diff", float("nan"))), threshold=patch.get("atol", metrics.get("patch_noop_atol", "")), why="Cross-turn patching only means something if self-patching is an identity operation.")
    add("null_trace_check", gate="demo", ok=null.get("ok", False), observed=null.get("random_null_max_abs_slope", metrics.get("random_null_max_abs_slope", float("nan"))), threshold=null.get("null_flatness_threshold", "flat vs topic"), warning=not _truthy(null.get("flat_enough_for_demo_claim", False)), why="A rising projection is not a semantic state if length/template/random nulls rise too.")
    add("bench_registry", gate="integration", ok=metrics.get("bench_chat_template_labs_has_lab15", True), observed="lab15 in CHAT_TEMPLATE_LABS" if metrics.get("bench_chat_template_labs_has_lab15", False) else "not registered", threshold="registry metadata consistent", warning=not metrics.get("bench_chat_template_labs_has_lab15", False), why="Bench diagnostics should agree that this run used chat-template semantics.")
    return rows


def write_visual_synthesis_tables(
    ctx: bench.RunContext,
    metrics: Mapping[str, Any],
    checks: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    evidence_rows = build_harness_evidence_rows(metrics, checks)
    evidence_path = ctx.path("tables", "harness_evidence_matrix.csv")
    bench.write_csv_with_context(ctx, evidence_path, evidence_rows)
    ctx.register_artifact(evidence_path, "table", "Joined Lab 15 self-check evidence matrix for plotting and reports.")

    turn_rows = _read_csv_rows(ctx.path("tables", "turn_segments.csv"))
    cache_rows = _read_csv_rows(ctx.path("diagnostics", "cache_recompute_parity_by_boundary.csv"))
    patch_rows = _read_csv_rows(ctx.path("diagnostics", "patch_noop_sites.csv"))
    gen_rows = _read_csv_rows(ctx.path("tables", "generation_prompt_boundaries.csv"))
    cache_by_key = {(str(r.get("conversation")), _safe_int(r.get("segment_index"))): r for r in cache_rows}
    gen_by_key = {(str(r.get("conversation")), _safe_int(r.get("source_user_segment_index", r.get("last_user_segment_index")))): r for r in gen_rows}
    patch_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for r in patch_rows:
        key = (str(r.get("conversation")), _safe_int(r.get("segment_index")))
        prev = patch_by_key.get(key)
        if prev is None or _safe_float(r.get("max_abs_logit_diff"), -1) > _safe_float(prev.get("max_abs_logit_diff"), -1):
            patch_by_key[key] = r
    boundary_rows: list[dict[str, Any]] = []
    for r in turn_rows:
        key = (str(r.get("conversation")), _safe_int(r.get("segment_index")))
        msg_n = _safe_float(r.get("message_n_tokens"), 0)
        content_n = _safe_float(r.get("content_n_tokens"), 0)
        cache_r = cache_by_key.get(key, {})
        patch_r = patch_by_key.get(key, {})
        gen_r = gen_by_key.get(key, {})
        boundary_rows.append({
            "conversation": key[0],
            "segment_index": key[1],
            "role": r.get("role"),
            "message_tokens": msg_n,
            "content_tokens": content_n,
            "template_tokens_estimate": max(0.0, msg_n - content_n),
            "content_share": rounded(content_n / msg_n if msg_n else float("nan")),
            "content_span_method": r.get("content_span_method", ""),
            "cache_rel_l2_hidden_diff": rounded(_safe_float(cache_r.get("rel_l2_hidden_diff"))),
            "cache_cosine_full_vs_cached": rounded(_safe_float(cache_r.get("cosine_full_vs_cached"))),
            "cache_ok": cache_r.get("ok_at_tolerance", ""),
            "patch_noop_max_abs_logit_diff": rounded(_safe_float(patch_r.get("max_abs_logit_diff"))),
            "patch_noop_site": patch_r.get("site", ""),
            "generation_stub_tokens": gen_r.get("assistant_generation_stub_n_tokens", gen_r.get("generation_prompt_extra_tokens", "")) if r.get("role") == "user" else "",
            "generation_boundary_ok": gen_r.get("prefix_without_generation_prompt_is_prefix", gen_r.get("with_prompt_extends_prefix", "")) if r.get("role") == "user" else "",
        })
    boundary_path = ctx.path("tables", "boundary_diagnostic_matrix.csv")
    bench.write_csv_with_context(ctx, boundary_path, boundary_rows)
    ctx.register_artifact(boundary_path, "table", "Per-segment joined span/cache/patch/generation boundary diagnostics.")

    slope_rows = _read_csv_rows(ctx.path("diagnostics", "null_trace_slopes.csv"))
    slope_summary: list[dict[str, Any]] = []
    for r in slope_rows:
        direction = str(r.get("direction"))
        conv = str(r.get("conversation"))
        s = _safe_float(r.get("slope"))
        if direction == "topic_orchid_minus_archive" and conv == "orchid_topic":
            family = "topic target"
        elif direction == "topic_orchid_minus_archive":
            family = "topic on control"
        elif direction == "length_matched_null":
            family = "length/template null"
        elif direction.startswith("random_null_"):
            family = "random null"
        else:
            family = "other"
        slope_summary.append({
            **dict(r),
            "direction_family": family,
            "abs_slope": rounded(abs(s) if math.isfinite(s) else float("nan")),
            "claim_risk": "high" if family != "topic target" and math.isfinite(s) and abs(s) >= max(NULL_SLOPE_ABS_WARN, 0.6 * abs(_safe_float(metrics.get("topic_trace_slope"), 0.0))) else "normal",
        })
    slope_path = ctx.path("tables", "trace_slope_summary.csv")
    bench.write_csv_with_context(ctx, slope_path, slope_summary)
    ctx.register_artifact(slope_path, "table", "Projection slopes with direction families and null-drift risk labels.")

    readiness_rows = [
        {
            "downstream_use": "turn-indexed projections",
            "status": "ready" if all(_truthy(checks.get(k, {}).get("ok", False)) for k in ("turn_boundary_check", "chat_exact_hook_parity", "chat_exact_lens_self_check")) else "blocked",
            "required_evidence": "template spans + exact chat hook/lens parity",
            "main_artifacts": "turn_segments.csv; chat_exact_hook_parity.json; chat_exact_lens_self_check.json",
        },
        {
            "downstream_use": "cache-efficient turn traces",
            "status": "ready" if _truthy(checks.get("cache_recompute_parity", {}).get("ok", False)) else "blocked",
            "required_evidence": "cached boundary states match full recompute",
            "main_artifacts": "cache_recompute_parity_by_boundary.csv",
        },
        {
            "downstream_use": "cross-turn activation patching",
            "status": "ready" if _truthy(checks.get("patch_noop_check", {}).get("ok", False)) else "blocked",
            "required_evidence": "self-patching at boundary sites is no-op",
            "main_artifacts": "patch_noop_sites.csv",
        },
        {
            "downstream_use": "generation-time reads",
            "status": "ready" if _truthy(checks.get("generation_prompt_boundary_check", {}).get("ok", False)) else "blocked",
            "required_evidence": "add_generation_prompt boundary extends user prefix and matches direct ids",
            "main_artifacts": "generation_prompt_boundaries.csv",
        },
        {
            "downstream_use": "semantic drift claims",
            "status": "ready" if _truthy(checks.get("null_trace_check", {}).get("flat_enough_for_demo_claim", False)) else "caution",
            "required_evidence": "topic trace beats length/template/random nulls",
            "main_artifacts": "null_trace_slopes.csv; trace_depth_sweep.csv",
        },
    ]
    readiness_path = ctx.path("tables", "downstream_readiness_card.csv")
    bench.write_csv_with_context(ctx, readiness_path, readiness_rows)
    ctx.register_artifact(readiness_path, "table", "Which later multi-turn uses are licensed or blocked by this run.")

    guide_rows = [
        {"plot": "harness_evidence_dashboard.png", "concept": "one-screen pass/warn/fail board for the instrumentation stack", "read_first": True},
        {"plot": "harness_evidence_matrix.png", "concept": "which check licenses which downstream use", "read_first": True},
        {"plot": "turn_span_map.png", "concept": "message spans versus content spans in exact rendered token space", "read_first": True},
        {"plot": "generation_boundary_audit.png", "concept": "assistant-generation prefix boundaries after user turns", "read_first": False},
        {"plot": "cache_patch_diagnostics.png", "concept": "cache parity and patch no-op errors by boundary/site", "read_first": True},
        {"plot": "demo_turn_trace.png", "concept": "topic trace with archive, length, and random null controls", "read_first": False},
        {"plot": "trace_depth_sweep.png", "concept": "depth-picking hazard: topic and null slopes vary over depth", "read_first": False},
        {"plot": "depth_selection_atlas.png", "concept": "control-adjusted depth selection and null drift in one matrix", "read_first": False},
        {"plot": "trace_slope_ledger.png", "concept": "all trace slopes, sorted by null risk", "read_first": False},
        {"plot": "downstream_readiness_card.png", "concept": "what later labs may safely inherit", "read_first": True},
    ]
    guide_path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, guide_path, guide_rows)
    ctx.register_artifact(guide_path, "table", "Plot reading guide for the upgraded Lab 15 visual suite.")

    return {
        "evidence_matrix": evidence_rows,
        "boundary_matrix": boundary_rows,
        "slope_summary": slope_summary,
        "readiness": readiness_rows,
        "plot_guide": guide_rows,
    }


def _maybe_panel_label(ax: Any, label: str) -> None:
    helper = getattr(bench, "add_panel_label", None)
    if callable(helper):
        try:
            helper(ax, label)
            return
        except Exception:
            pass
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def _finalize_axes(ax: Any, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, legend: bool = True) -> None:
    try:
        bench.style_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel, legend=legend)
    except TypeError:
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        bench.style_ax(ax)


def plot_turn_trace(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    fig, ax = bench.new_figure(figsize=(10.6, 6.0))
    # All random nulls as a gray band: the demo topic only gets a claim if it separates from the controls.
    random_dirs = sorted({str(r["direction"]) for r in rows if str(r.get("direction", "")).startswith("random_null_")})
    for direction in random_dirs:
        sub = [r for r in rows if r["conversation"] == "orchid_topic" and r["direction"] == direction and r["role"] != "system"]
        if not sub:
            continue
        ax.plot(
            [float(r["turn_index_non_system"]) for r in sub],
            [float(r["cumulative_message_mean_projection"]) for r in sub],
            color=lab15_color("random_null"), alpha=0.16, linewidth=1.0,
        )
    styles = [
        ("orchid_topic", "topic_orchid_minus_archive", lab15_color("topic"), "-", "orchid conversation · topic direction"),
        ("archive_length_control", "topic_orchid_minus_archive", lab15_color("archive_control"), "--", "archive control · topic direction"),
        ("orchid_topic", "length_matched_null", lab15_color("length_null"), ":", "orchid conversation · length/template null"),
    ]
    for conv, direction, color, linestyle, label in styles:
        sub = [r for r in rows if r["conversation"] == conv and r["direction"] == direction and r["role"] != "system"]
        if not sub:
            continue
        ax.plot(
            [float(r["turn_index_non_system"]) for r in sub],
            [float(r["cumulative_message_mean_projection"]) for r in sub],
            marker="o", color=color, linestyle=linestyle, linewidth=2.5, label=label,
        )
        # Content-boundary dots show that the same trace is available at a narrower site.
        ax.scatter(
            [float(r["turn_index_non_system"]) for r in sub],
            [float(r["content_boundary_projection"]) for r in sub],
            s=26, color=color, alpha=0.55, marker="D", zorder=3,
        )
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.text(0.01, 0.02, "lines = cumulative prefix mean; diamonds = content boundary", transform=ax.transAxes, fontsize=8, color="#555555")
    _finalize_axes(ax, "Lab 15 demo trace: topic direction must separate from null controls", "message boundary, system excluded", "projection onto direction")
    bench.save_figure(ctx, fig, "demo_turn_trace.png", "Topic and null projection traces over scripted multi-turn conversations, including random-null controls and content-boundary markers.")


def plot_trace_depth_sweep(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    fig, ax = bench.new_figure(figsize=(10.6, 5.8))
    by = {(str(r.get("conversation")), str(r.get("direction")), int(r.get("stream_depth", 0))): _safe_float(r.get("cumulative_projection_slope")) for r in rows}
    depths = sorted({int(r.get("stream_depth", 0)) for r in rows})
    topic = [by.get(("orchid_topic", "topic_orchid_minus_archive", d), float("nan")) for d in depths]
    control = [by.get(("archive_length_control", "topic_orchid_minus_archive", d), float("nan")) for d in depths]
    gap = [t - c if math.isfinite(t) and math.isfinite(c) else float("nan") for t, c in zip(topic, control)]
    length = [by.get(("orchid_topic", "length_matched_null", d), float("nan")) for d in depths]
    random = [by.get(("orchid_topic", "random_null_00", d), float("nan")) for d in depths]
    ax.plot(depths, gap, color=lab15_color("topic"), linewidth=2.7, marker="o", label="topic slope minus archive-control slope")
    ax.plot(depths, topic, color=lab15_color("topic"), linewidth=1.3, linestyle="--", alpha=0.65, label="raw topic slope on orchid")
    ax.plot(depths, control, color=lab15_color("archive_control"), linewidth=1.3, linestyle="--", alpha=0.65, label="topic slope on archive control")
    ax.plot(depths, length, color=lab15_color("length_null"), linewidth=1.8, linestyle=":", label="length/template null slope")
    ax.plot(depths, random, color=lab15_color("random_null"), linewidth=1.5, linestyle=":", label="random-null-00 slope")
    ax.axhline(0.0, color="black", linewidth=0.9)
    if gap:
        finite = [(d, g) for d, g in zip(depths, gap) if math.isfinite(g)]
        if finite:
            d_star, g_star = max(finite, key=lambda x: abs(x[1]))
            ax.axvline(d_star, color=lab15_color("topic"), linestyle="--", linewidth=1.0, alpha=0.55)
            ax.text(d_star, g_star, f" strongest gap @ {d_star}", fontsize=8, va="bottom")
    _finalize_axes(ax, "Depth sweep: depth-picking is now part of the hypothesis", "stream depth k, after k blocks", "projection slope over message boundaries")
    bench.save_figure(ctx, fig, "trace_depth_sweep.png", "Control-adjusted projection-slope depth sweep for topic and null controls.")


def plot_turn_span_map(ctx: bench.RunContext, conversations_by_name: Mapping[str, RenderedConversation]) -> None:
    fig, ax = bench.new_figure(figsize=(11.0, 5.8))
    y = 0
    yticks = []
    ylabels = []
    for conv_name, conv in conversations_by_name.items():
        for seg in conv.segments:
            role_color = lab15_color(seg.role)
            template_left = seg.message_start
            template_width = seg.message_end - seg.message_start
            content_left = seg.content_start
            content_width = seg.content_end - seg.content_start
            ax.barh(y, template_width, left=template_left, height=0.62, color=lab15_color("template"), alpha=0.8, edgecolor="none")
            ax.barh(y, content_width, left=content_left, height=0.36, color=role_color, alpha=0.88, edgecolor="black", linewidth=0.3)
            ax.axvline(seg.boundary_token, ymin=max(0, (y - 0.4) / max(1, len(conv.segments) * len(conversations_by_name))), ymax=1, color=role_color, alpha=0.10)
            yticks.append(y)
            ylabels.append(f"{conv_name} · {seg.index} · {seg.role}")
            y += 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    # Role legend by hand.
    for role in LAB15_ROLE_ORDER:
        ax.scatter([], [], color=lab15_color(role), marker="s", s=80, label=f"{role} content")
    ax.scatter([], [], color=lab15_color("template"), marker="s", s=80, label="message/template span")
    _finalize_axes(ax, "Rendered-chat token map: content spans ride inside template spans", "token index in rendered chat", "", legend=True)
    bench.save_figure(ctx, fig, "turn_span_map.png", "Token-span map distinguishing full message/template spans from narrower content spans.")


def plot_generation_boundary_audit(ctx: bench.RunContext) -> None:
    rows = _read_csv_rows(ctx.path("tables", "generation_prompt_boundaries.csv"))
    if not rows:
        return
    fig, ax = bench.new_figure(figsize=(10.6, 4.9))
    labels = [f"{r.get('conversation')}\nuser seg {r.get('source_user_segment_index', r.get('last_user_segment_index'))}" for r in rows]
    base = [_safe_float(r.get("no_generation_prompt_token_count", r.get("prefix_tokens_without_generation_prompt")), 0) for r in rows]
    extra = [_safe_float(r.get("assistant_generation_stub_n_tokens", r.get("generation_prompt_extra_tokens")), 0) for r in rows]
    x = list(range(len(rows)))
    ax.bar(x, base, color=lab15_color("span"), alpha=0.45, label="prefix without generation prompt")
    ax.bar(x, extra, bottom=base, color=lab15_color("generation_prompt"), alpha=0.85, label="assistant-generation stub")
    for i, r in enumerate(rows):
        ok = _truthy(r.get("prefix_without_generation_prompt_is_prefix", r.get("with_prompt_extends_prefix"))) and _truthy(r.get("string_vs_direct_template_ids_match", r.get("string_vs_direct_with_prompt_ids_match")))
        ax.text(i, base[i] + extra[i] + max(base + extra + [1]) * 0.015, "✓" if ok else "✗", ha="center", color=lab15_color("pass" if ok else "fail"), fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    _finalize_axes(ax, "Assistant-generation boundaries: the assistant header is a measured site", "user-ended prefix", "tokens")
    bench.save_figure(ctx, fig, "generation_boundary_audit.png", "Generation-prompt boundary token counts and parity verdicts for user-ended prefixes.")


def plot_trace_slope_ledger(ctx: bench.RunContext) -> None:
    rows = _read_csv_rows(ctx.path("tables", "trace_slope_summary.csv")) or _read_csv_rows(ctx.path("diagnostics", "null_trace_slopes.csv"))
    if not rows:
        return
    rows = sorted(rows, key=lambda r: abs(_safe_float(r.get("slope"), 0)), reverse=True)
    top = rows[:24]
    fig, ax = bench.new_figure(figsize=(10.0, max(4.8, 0.32 * len(top) + 1.0)))
    labels = [f"{r.get('conversation')} · {r.get('direction')}" for r in top]
    vals = [_safe_float(r.get("slope"), 0.0) for r in top]
    colors = []
    for r in top:
        fam = str(r.get("direction_family", ""))
        if "topic target" in fam:
            colors.append(lab15_color("topic"))
        elif "topic on control" in fam:
            colors.append(lab15_color("archive_control"))
        elif "length" in fam:
            colors.append(lab15_color("length_null"))
        else:
            colors.append(lab15_color("random_null"))
    y = list(range(len(top)))
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    _finalize_axes(ax, "Slope ledger: every pretty trace pays rent against nulls", "projection slope over turns", "")
    bench.save_figure(ctx, fig, "trace_slope_ledger.png", "All topic and null projection slopes sorted by magnitude.")


def plot_depth_selection_atlas(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import numpy as np
    fig, ax = bench.new_figure(figsize=(10.6, 4.8))
    depths = sorted({int(r.get("stream_depth", 0)) for r in rows})
    by = {(str(r.get("conversation")), str(r.get("direction")), int(r.get("stream_depth", 0))): _safe_float(r.get("cumulative_projection_slope")) for r in rows}
    metrics = [
        ("topic gap\norchid−archive", [by.get(("orchid_topic", "topic_orchid_minus_archive", d), float("nan")) - by.get(("archive_length_control", "topic_orchid_minus_archive", d), float("nan")) for d in depths]),
        ("raw topic\norchid", [by.get(("orchid_topic", "topic_orchid_minus_archive", d), float("nan")) for d in depths]),
        ("topic on\narchive", [by.get(("archive_length_control", "topic_orchid_minus_archive", d), float("nan")) for d in depths]),
        ("length null\norchid", [by.get(("orchid_topic", "length_matched_null", d), float("nan")) for d in depths]),
        ("random null 00\norchid", [by.get(("orchid_topic", "random_null_00", d), float("nan")) for d in depths]),
    ]
    mat = np.array([vals for _, vals in metrics], dtype=float)
    finite = mat[np.isfinite(mat)]
    vmax = max(0.05, float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 1.0)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[0] for m in metrics], fontsize=8)
    tick_step = max(1, len(depths)//10)
    ax.set_xticks(range(0, len(depths), tick_step))
    ax.set_xticklabels([str(d) for d in depths[::tick_step]], fontsize=8)
    ax.set_xlabel("stream depth k")
    ax.set_title("Depth-selection atlas: the selected layer must beat its controls")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="turn-slope")
    bench.save_figure(ctx, fig, "depth_selection_atlas.png", "Matrix view of control-adjusted topic slopes and null slopes over stream depth.")


def plot_cache_patch_diagnostics(ctx: bench.RunContext) -> None:
    import matplotlib.pyplot as plt
    cache_rows = _read_csv_rows(ctx.path("diagnostics", "cache_recompute_parity_by_boundary.csv"))
    patch_rows = _read_csv_rows(ctx.path("diagnostics", "patch_noop_sites.csv"))
    if not cache_rows and not patch_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    if cache_rows:
        labels = [f"{r.get('conversation')}\n{r.get('segment_index')}:{r.get('role')}" for r in cache_rows]
        vals = [_safe_float(r.get("rel_l2_hidden_diff"), 0.0) for r in cache_rows]
        colors = [lab15_color("pass" if _truthy(r.get("ok_at_tolerance")) else "fail") for r in cache_rows]
        x = list(range(len(cache_rows)))
        axes[0].bar(x, vals, color=colors, alpha=0.85)
        thresh_vals = [_safe_float(r.get("rel_l2_threshold"), float("nan")) for r in cache_rows]
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        axes[0].set_ylabel("relative L2 hidden diff")
        axes[0].set_title("KV cache vs full recompute")
    else:
        axes[0].axis("off")
    if patch_rows:
        labels = [f"{r.get('conversation')}\nL{r.get('layer')} · {r.get('site')}" for r in patch_rows]
        vals = [_safe_float(r.get("max_abs_logit_diff"), 0.0) for r in patch_rows]
        colors = [lab15_color("pass" if _truthy(r.get("ok_at_tolerance")) else "fail") for r in patch_rows]
        x = list(range(len(patch_rows)))
        axes[1].bar(x, vals, color=colors, alpha=0.85)
        axes[1].axhline(PATCH_NOOP_ATOL, color=lab15_color("fail"), linestyle="--", linewidth=1.0, label="tolerance")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=6.2)
        axes[1].set_ylabel("max |logit diff| under self-patch")
        axes[1].set_title("Self-patching no-op")
        axes[1].legend(fontsize=7)
    else:
        axes[1].axis("off")
    for i, ax in enumerate(axes):
        _maybe_panel_label(ax, chr(ord("A") + i))
        ax.grid(True, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Cache and patch diagnostics: boring means trustworthy", fontsize=13)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "cache_patch_diagnostics.png", "Boundary-level cache parity and self-patching no-op diagnostics.")


def plot_harness_evidence_dashboard(
    ctx: bench.RunContext,
    metrics: Mapping[str, Any],
    checks: Mapping[str, Mapping[str, Any]],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    evidence = list(tables.get("evidence_matrix", [])) or build_harness_evidence_rows(metrics, checks)
    boundary = list(tables.get("boundary_matrix", [])) or _read_csv_rows(ctx.path("tables", "boundary_diagnostic_matrix.csv"))
    slope_rows = list(tables.get("slope_summary", [])) or _read_csv_rows(ctx.path("tables", "trace_slope_summary.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))

    # A. Check status strip
    ax = axes[0, 0]
    names = [str(r.get("display_name")) for r in evidence]
    scores = [_safe_float(r.get("status_score"), 0.0) for r in evidence]
    colors = [lab15_color(str(r.get("status", "unknown"))) for r in evidence]
    x = np.arange(len(evidence))
    ax.bar(x, scores, color=colors, alpha=0.9)
    ax.axhline(1.0, color="#999999", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("pass/warn/fail score")
    ax.set_title("Self-check verdicts")
    for i, r in enumerate(evidence):
        ax.text(i, scores[i] + 0.03, {"pass":"✓", "warn":"!", "fail":"✗"}.get(str(r.get("status")), "?"), ha="center", fontsize=10, fontweight="bold")

    # B. Content/template load by role
    ax = axes[0, 1]
    role_stats: dict[str, list[float]] = defaultdict(list)
    for r in boundary:
        role = str(r.get("role"))
        role_stats[role].append(_safe_float(r.get("content_share")))
    roles = [r for r in LAB15_ROLE_ORDER if r in role_stats]
    vals = [safe_fmean([v for v in role_stats[r] if math.isfinite(v)], 0.0) for r in roles]
    ax.bar(range(len(roles)), vals, color=[lab15_color(r) for r in roles], alpha=0.85)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("content tokens / message tokens")
    ax.set_title("How much of a message is content?")

    # C. Cache/patch numeric gates
    ax = axes[1, 0]
    items = [
        ("cache rel-L2", _safe_float(checks.get("cache_recompute_parity", {}).get("max_rel_l2_hidden_diff")), _safe_float(checks.get("cache_recompute_parity", {}).get("rel_l2_threshold")), "cache"),
        ("patch max logit", _safe_float(checks.get("patch_noop_check", {}).get("max_abs_logit_diff")), _safe_float(checks.get("patch_noop_check", {}).get("atol")), "patch"),
        ("hook max diff", _safe_float(checks.get("chat_exact_hook_parity", {}).get("max_abs_diff")), max(_safe_float(checks.get("chat_exact_hook_parity", {}).get("tolerance"), 0.0), 1e-12), "span"),
    ]
    labels = [i[0] for i in items]
    obs = [i[1] if math.isfinite(i[1]) else 0.0 for i in items]
    thresh = [i[2] if math.isfinite(i[2]) else float("nan") for i in items]
    x = np.arange(len(items))
    ax.bar(x - 0.15, obs, width=0.3, color=[lab15_color(i[3]) for i in items], label="observed")
    ax.bar(x + 0.15, [t if math.isfinite(t) else 0 for t in thresh], width=0.3, color="#BBBBBB", label="threshold")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Numeric gates on a log-ish scale")
    ax.legend(fontsize=7)

    # D. Null slopes
    ax = axes[1, 1]
    slope_plot_rows = [r for r in slope_rows if str(r.get("conversation")) == "orchid_topic"]
    families = []
    for r in slope_plot_rows:
        d = str(r.get("direction"))
        if d == "topic_orchid_minus_archive":
            fam = "topic"
        elif d == "length_matched_null":
            fam = "length null"
        elif d.startswith("random_null_"):
            fam = "random null"
        else:
            fam = d
        families.append((fam, _safe_float(r.get("slope"))))
    # Aggregate random nulls for readability.
    topic_val = next((v for f, v in families if f == "topic"), float("nan"))
    length_val = next((v for f, v in families if f == "length null"), float("nan"))
    random_vals = [v for f, v in families if f == "random null" and math.isfinite(v)]
    labels = ["topic", "length null", "random mean |s|", "random max |s|"]
    vals = [topic_val, length_val, safe_fmean([abs(v) for v in random_vals], 0.0), safe_max_abs(random_vals, 0.0)]
    colors = [lab15_color("topic"), lab15_color("length_null"), lab15_color("random_null"), lab15_color("random_null")]
    ax.bar(range(len(vals)), vals, color=colors, alpha=0.88)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Null-trace pressure test")
    ax.set_ylabel("slope")

    for idx, ax in enumerate(axes.flat):
        _maybe_panel_label(ax, chr(ord("A") + idx))
        ax.grid(True, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Lab 15 multi-turn harness: the microscope audits itself", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    bench.save_figure(ctx, fig, "harness_evidence_dashboard.png", "One-screen dashboard for Lab 15 self-checks, span load, numeric gates, and null traces.")


def plot_harness_evidence_matrix(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt
    fig, ax = bench.new_figure(figsize=(12.5, max(5.0, 0.45 * len(rows) + 1.8)))
    ax.axis("off")
    columns = ["check", "gate", "status", "observed", "threshold", "licenses"]
    cell_text = []
    cell_colours = []
    for r in rows:
        license_text = str(r.get("why_it_matters", ""))[:62] + ("…" if len(str(r.get("why_it_matters", ""))) > 62 else "")
        row = [str(r.get("display_name")), str(r.get("gate_type")), str(r.get("status")), str(r.get("observed")), str(r.get("threshold")), license_text]
        cell_text.append(row)
        bg = lab15_color(str(r.get("status", "unknown")))
        cell_colours.append(["#FFFFFF", "#F7F7F7", bg, "#FFFFFF", "#FFFFFF", "#FFFFFF"])
    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc="left", loc="center", cellColours=cell_colours)
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EEEEEE")
        if col == 2 and row > 0:
            cell.set_text_props(color="white", weight="bold")
    ax.set_title("Harness evidence matrix: every later multi-turn claim inherits these gates", pad=14)
    bench.save_figure(ctx, fig, "harness_evidence_matrix.png", "Table plot of Lab 15 self-check gates, status, tolerances, and downstream purpose.")


def plot_downstream_readiness_card(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = bench.new_figure(figsize=(11.0, max(4.8, 0.45 * len(rows) + 1.4)))
    ax.axis("off")
    cols = ["downstream use", "status", "required evidence", "artifacts"]
    cell_text = [[str(r.get("downstream_use")), str(r.get("status")), str(r.get("required_evidence")), str(r.get("main_artifacts"))] for r in rows]
    colours = []
    for r in rows:
        status = str(r.get("status"))
        c = lab15_color("pass" if status == "ready" else "warn" if status == "caution" else "fail")
        colours.append(["#FFFFFF", c, "#FFFFFF", "#FFFFFF"])
    table = ax.table(cellText=cell_text, colLabels=cols, cellLoc="left", loc="center", cellColours=colours)
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EEEEEE")
        if col == 1 and row > 0:
            cell.set_text_props(color="white", weight="bold")
    ax.set_title("Downstream readiness: what later labs may safely inherit", pad=14)
    bench.save_figure(ctx, fig, "downstream_readiness_card.png", "Readiness card for later multi-turn projections, caching, generation reads, and patching.")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def write_bench_integration_note(ctx: bench.RunContext, bundle: bench.ModelBundle) -> dict[str, Any]:
    registry_has_lab15 = bool(hasattr(bench, "LAB_PROFILES") and "lab15" in getattr(bench, "LAB_PROFILES"))
    chat_set_has_lab15 = bool(hasattr(bench, "CHAT_TEMPLATE_LABS") and "lab15" in getattr(bench, "CHAT_TEMPLATE_LABS"))
    payload = {
        "registry_has_lab15": registry_has_lab15,
        "chat_template_labs_has_lab15": chat_set_has_lab15,
        "current_lab_arg": getattr(ctx.args, "lab", ""),
        "model_id": bundle.anatomy.model_id,
        "chat_template_present": bool(getattr(bundle.tokenizer, "chat_template", None)),
        "note": (
            "This lab file can run once the bench registry routes --lab lab15 to it. If tokenizer_info.json says "
            "chat_template_used_by_lab=false, add lab15 to CHAT_TEMPLATE_LABS in the bench so diagnostics match reality."
        ),
    }
    path = ctx.path("diagnostics", "bench_integration_note.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Optional bench registry and chat-template integration note.")
    return payload


def check_status(result: Mapping[str, Any]) -> str:
    return "PASS" if bool(result.get("ok")) else "FAIL"


def write_report(
    ctx: bench.RunContext,
    metrics: Mapping[str, Any],
    checks: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 15 Multi-Turn Harness Report",
        "",
        "This lab validates the measurement apparatus. The scripted topic trace is a demo, not a claim about model psychology.",
        "",
        "## Self-check stack",
        "",
    ]
    for name, result in checks.items():
        lines.append(f"- `{name}`: **{check_status(result)}**")
    lines += [
        "",
        "## Headline diagnostics",
        "",
        f"- Model: `{metrics.get('model_id')}`",
        f"- Trace depth: {metrics.get('trace_depth')} of {metrics.get('n_depths_minus_one')} blocks",
        f"- Exact chat hook parity max |diff|: {metrics.get('chat_hook_parity_max_abs_diff')}",
        f"- Cache parity max hidden |diff|: {metrics.get('cache_parity_max_abs_hidden_diff')}",
        f"- Cache parity max logit |diff|: {metrics.get('cache_parity_max_abs_logit_diff')}",
        f"- Patch no-op max logit |diff|: {metrics.get('patch_noop_max_abs_logit_diff')}",
        f"- Topic slope on orchid conversation: {metrics.get('topic_trace_slope')}",
        f"- Topic slope on archive control: {metrics.get('topic_on_control_slope')}",
        f"- Length-null slope: {metrics.get('length_null_slope')}",
        f"- Max random-null slope magnitude: {metrics.get('random_null_max_abs_slope')}",
        f"- Null controls flat enough for the demo claim: {metrics.get('null_controls_flat_enough_for_demo')}",
        "",
        "## How later labs should reuse this",
        "",
        "1. Render with the tokenizer chat template and cache the rendered text hash.",
        "2. Derive both message spans and content spans from prefix renders.",
        "3. Compare cached boundary states against full recompute before reading any turn trace.",
        "4. Keep a random-direction and length/template null trace beside every exciting projection.",
        "5. State whether you read message boundaries, content boundaries, content means, or cumulative prefix means.",
        "",
        "The plot is the little lantern. The self-check stack is the bridge.",
        "",
    ]
    text = "\n".join(lines)
    path = ctx.path("multiturn_harness_report.md")
    bench.write_text(path, text)
    ctx.register_artifact(path, "summary", "Multi-turn instrumentation self-check report.")

    card_path = ctx.path("multiturn_harness_card.md")
    bench.write_text(card_path, text)
    ctx.register_artifact(card_path, "summary", "Read-this-first Lab 15 harness card, aliasing the self-check report.")


def write_operationalization_audit(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 15 Operationalization Audit",
        "",
        "## What was measured",
        "",
        "The lab measures whether turn-indexed residual-stream reads are well-defined under the current chat template, tokenizer, model cache behavior, and patch hook target. It does not establish a persona, belief, memory, or self-report phenomenon.",
        "",
        "## Cheap explanations attacked here",
        "",
        "| Cheap explanation | Audit pressure | Current artifact |",
        "|---|---|---|",
        "| Template residue | string tokenization must match direct chat-template tokenization | `diagnostics/turn_boundary_check.json` |",
        "| Role-span leakage | user content spans are checked for assistant-text leakage | `tables/turn_segments.csv` |",
        "| Generation-boundary guessing | user-ended prefixes are rendered with `add_generation_prompt=True` and audited | `diagnostics/generation_prompt_boundary_check.json` |",
        "| BOS or special-token drift | exact-chat hook and lens checks run on `add_special_tokens=False` token IDs | `diagnostics/chat_exact_hook_parity.json` |",
        "| Cache artifact | incremental boundary states must match full recompute | `diagnostics/cache_recompute_parity.json` |",
        "| Patch hook drift | self-patching block output k with `streams[k+1]` must be identity | `diagnostics/patch_noop_check.json` |",
        "| Sequence-length drift | topic trace is compared with a length/template null and random directions | `diagnostics/null_trace_check.json` |",
        "",
        "## Current run status",
        "",
        f"- Turn boundaries ok: {metrics.get('turn_boundary_ok')}",
        f"- Exact chat hook parity ok: {metrics.get('chat_hook_parity_ok')}",
        f"- Exact chat lens self-check ok: {metrics.get('lens_self_check_ok')}",
        f"- Cache parity ok: {metrics.get('cache_parity_ok')}",
        f"- Patch no-op ok: {metrics.get('patch_noop_ok')}",
        f"- Null traces finite: {metrics.get('null_trace_ok')}",
        f"- Null controls flat enough for a demo drift claim: {metrics.get('null_controls_flat_enough_for_demo')}",
        "",
        "## Allowed claim",
        "",
        "The allowed claim is an instrumentation claim: for this tokenizer/model pair and these rendered conversations, turn-boundary states can be segmented, cached, recomputed, and self-patched to the reported tolerances. Later labs must earn their own model-state interpretation with their own controls.",
        "",
        "## Not licensed",
        "",
        "- The orchid trace is not a persistent topic memory claim.",
        "- A rising projection in a future persona lab is not a persona until the Lab 15 null trace and that lab's own confounds survive.",
        "- A belief-revision trace is not a belief unless the probe direction has passed the relevant bridge audit.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "audit", "Operationalization limits for multi-turn instrumentation.")


def write_run_summary(
    ctx: bench.RunContext,
    metrics: Mapping[str, Any],
    checks: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 15 run summary: multi-turn instrumentation",
        "",
        "## Run identity",
        "",
        f"- model: `{metrics.get('model_id')}`",
        f"- trace depth: {metrics.get('trace_depth')}",
        "- evidence level: `OBS`, instrumentation validation only",
        "- self-checks: turn boundaries, generation-prompt boundaries, exact-chat hook parity, exact-chat lens parity, cache parity, patch no-op, null trace audit",
        "",
        "## 1. What behavior was studied?",
        "",
        "A harmless scripted orchid planning conversation and a length/template matched archive conversation. The behavior is not the target; it is a calibration scene for turn-indexed measurements.",
        "",
        "## 2. What internal object was measured?",
        "",
        "Residual-stream vectors at message boundaries, content boundaries, content means, and cumulative prefix means, using the course convention `streams[k]` = pre-norm residual after k blocks.",
        "",
        "## 3. What intervention or control was used?",
        "",
        "The lab compares full recompute against KV-cache prefill, self-patches block outputs with their own cached boundary vectors, and traces topic projections beside random and length/template null directions.",
        "",
        "## 4. Headline numbers",
        "",
        f"- cache parity max hidden |diff|: {metrics.get('cache_parity_max_abs_hidden_diff')}",
        f"- patch no-op max logit |diff|: {metrics.get('patch_noop_max_abs_logit_diff')}",
        f"- topic slope: {metrics.get('topic_trace_slope')}",
        f"- control-conversation topic slope: {metrics.get('topic_on_control_slope')}",
        f"- length-null slope: {metrics.get('length_null_slope')}",
        f"- max random-null slope magnitude: {metrics.get('random_null_max_abs_slope')}",
        "",
        "## 5. Claims",
        "",
        "Only an instrumentation claim is drafted. Any persona, belief, roleplay, or self-report interpretation must wait for later labs.",
        "",
        "## 6. What could falsify this?",
        "",
        "Any supported chat template failing segmentation, cache parity exceeding tolerance, patch no-op failure, or null controls drifting strongly enough to explain a future turn trace.",
        "",
        "## 7. What should you inspect first?",
        "",
        "Read `multiturn_harness_report.md`, then `diagnostics/turn_boundary_check.json`, `tables/turn_segments.csv`, `diagnostics/cache_recompute_parity.json`, and `diagnostics/patch_noop_check.json` before opening the plot.",
        "",
        "## Self-check status table",
        "",
        "| check | status |",
        "|---|---|",
    ]
    for name, result in checks.items():
        lines.append(f"| `{name}` | {check_status(result)} |")
    lines.append("")
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Standard Lab 15 run summary.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    if not bench.supports_chat_template(bundle):
        raise RuntimeError(
            "Lab 15 requires an instruct/chat model with a chat template. "
            "Use the lab15 registry profile or pass an instruct model explicitly."
        )

    specs = conversations()
    conversations_by_name, turn_check = write_turn_boundary_check(ctx, bundle, specs)
    generation_check = write_generation_prompt_boundary_check(ctx, bundle, conversations_by_name)
    bench_note = write_bench_integration_note(ctx, bundle)

    # Use the main conversation for exact-chat hook/lens checks. The hook
    # semantics are architecture-wide; the cache and patch checks run on all
    # rendered conversations below.
    topic_conv = conversations_by_name["orchid_topic"]
    chat_hook_check, topic_streams, topic_logits = write_chat_exact_hook_parity_check(ctx, bundle, topic_conv)
    lens_check = write_exact_lens_check(ctx, bundle, topic_conv, topic_streams, topic_logits)

    cache_check = write_cache_parity_check(ctx, bundle, conversations_by_name)
    full_stream_cache = compute_full_stream_cache(bundle, conversations_by_name)
    patch_check = write_patch_noop_check(ctx, bundle, conversations_by_name, full_stream_cache)

    n_depths = bundle.anatomy.n_layers + 1
    trace_depth = min(n_depths - 1, max(1, int(round(TRACE_DEPTH_FRACTION * bundle.anatomy.n_layers))))
    direction_bank, direction_manifest = build_trace_direction_bank(ctx, bundle, int(ctx.args.seed))
    write_trace_direction_manifest(ctx, direction_bank, direction_manifest, trace_depth)

    rows = projection_rows(
        bundle,
        conversations_by_name,
        full_stream_cache,
        direction_bank[trace_depth],
        trace_depth,
    )
    trace_path = ctx.path("tables", "turn_projection_trace.csv")
    bench.write_csv_with_context(ctx, trace_path, rows)
    ctx.register_artifact(trace_path, "table", "Per-turn topic/random/length-null projections over scripted conversations.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, rows)
    ctx.register_artifact(results_path, "results", "Alias of turn_projection_trace.csv for the standard run contract.")

    null_check = write_null_trace_check(ctx, rows)

    depth_rows = trace_depth_sweep_rows(conversations_by_name, full_stream_cache, direction_bank)
    depth_path = ctx.path("tables", "trace_depth_sweep.csv")
    bench.write_csv_with_context(ctx, depth_path, depth_rows)
    ctx.register_artifact(depth_path, "table", "Projection-slope sweep over stream depths for topic and null directions.")

    if not ctx.args.no_plots:
        plot_turn_trace(ctx, rows)
        plot_trace_depth_sweep(ctx, depth_rows)
        plot_turn_span_map(ctx, conversations_by_name)
        plot_generation_boundary_audit(ctx)
        plot_trace_slope_ledger(ctx)
        plot_depth_selection_atlas(ctx, depth_rows)
        plot_cache_patch_diagnostics(ctx)

    slope_rows = slope_rows_from_trace(rows, "cumulative_message_mean_projection")
    topic_slope = get_slope_value(slope_rows, "orchid_topic", "topic_orchid_minus_archive")
    topic_control_slope = get_slope_value(slope_rows, "archive_length_control", "topic_orchid_minus_archive")
    length_null_slope = get_slope_value(slope_rows, "orchid_topic", "length_matched_null")
    random_slopes = [
        float(r["slope"])
        for r in slope_rows
        if r.get("conversation") == "orchid_topic"
        and str(r.get("direction", "")).startswith("random_null_")
        and isinstance(r.get("slope"), (int, float))
        and math.isfinite(float(r["slope"]))
    ]
    span_method_counts = Counter(
        seg.content_span_method
        for conv in conversations_by_name.values()
        for seg in conv.segments
    )

    metrics = {
        "model_id": bundle.anatomy.model_id,
        "n_layers": bundle.anatomy.n_layers,
        "n_depths": n_depths,
        "n_depths_minus_one": n_depths - 1,
        "trace_depth": trace_depth,
        "trace_depth_fraction": rounded(trace_depth / max(1, n_depths - 1)),
        "n_conversations": len(conversations_by_name),
        "n_segments_total": sum(len(conv.segments) for conv in conversations_by_name.values()),
        "turn_boundary_ok": bool(turn_check["ok"]),
        "generation_prompt_boundary_ok": bool(generation_check["ok"]),
        "generation_prompt_stub_present_for_all": bool(generation_check["assistant_generation_stub_present_for_all"]),
        "chat_hook_parity_ok": bool(chat_hook_check["ok"]),
        "chat_hook_parity_max_abs_diff": rounded(chat_hook_check["max_abs_diff"]),
        "lens_self_check_ok": bool(lens_check["ok"]),
        "cache_parity_ok": bool(cache_check["ok"]),
        "cache_parity_max_abs_hidden_diff": rounded(cache_check["max_abs_hidden_diff"]),
        "cache_parity_max_abs_logit_diff": rounded(cache_check["max_abs_logit_diff"]),
        "patch_noop_ok": bool(patch_check["ok"]),
        "patch_noop_max_abs_logit_diff": rounded(patch_check["max_abs_logit_diff"]),
        "null_trace_ok": bool(null_check["ok"]),
        "null_controls_flat_enough_for_demo": bool(null_check["flat_enough_for_demo_claim"]),
        "topic_trace_slope": rounded(topic_slope),
        "topic_on_control_slope": rounded(topic_control_slope),
        "topic_slope_gap_vs_control": rounded(topic_slope - topic_control_slope if math.isfinite(topic_slope) and math.isfinite(topic_control_slope) else float("nan")),
        "length_null_slope": rounded(length_null_slope),
        "random_null_mean_abs_slope": rounded(safe_fmean([abs(x) for x in random_slopes])),
        "random_null_max_abs_slope": rounded(safe_max_abs(random_slopes)),
        "cache_parity_atol": CACHE_PARITY_ATOL,
        "patch_noop_atol": PATCH_NOOP_ATOL,
        "content_span_method_counts": dict(span_method_counts),
        "bench_registry_has_lab15": bool(bench_note["registry_has_lab15"]),
        "bench_chat_template_labs_has_lab15": bool(bench_note["chat_template_labs_has_lab15"]),
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 15 instrumentation metrics.")

    checks = {
        "turn_boundary_check": turn_check,
        "generation_prompt_boundary_check": generation_check,
        "chat_exact_hook_parity": chat_hook_check,
        "chat_exact_lens_self_check": lens_check,
        "cache_recompute_parity": cache_check,
        "patch_noop_check": patch_check,
        "null_trace_check": null_check,
    }
    visual_tables = write_visual_synthesis_tables(ctx, metrics, checks)
    if not ctx.args.no_plots:
        plot_harness_evidence_dashboard(ctx, metrics, checks, visual_tables)
        plot_harness_evidence_matrix(ctx, visual_tables.get("evidence_matrix", []))
        plot_downstream_readiness_card(ctx, visual_tables.get("readiness", []))
    write_report(ctx, metrics, checks)
    write_operationalization_audit(ctx, metrics)
    write_run_summary(ctx, metrics, checks)

    run_name = ctx.run_dir.name
    null_clause = (
        "null controls were flat enough for the demo trace"
        if metrics["null_controls_flat_enough_for_demo"]
        else "null controls were mixed, so downstream drift claims need stronger length/template controls"
    )
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "OBS",
            "text": (
                f"For {bundle.anatomy.model_id}, the Lab 15 exact-chat multi-turn harness segmented rendered "
                f"chat prompts, reproduced KV-cache boundary states versus full recompute to max hidden |diff| "
                f"{metrics['cache_parity_max_abs_hidden_diff']}, and self-patched turn-boundary block outputs with "
                f"max logit |diff| {metrics['patch_noop_max_abs_logit_diff']} at tested layers. {null_clause}. "
                "This is an instrumentation claim, not a persona, belief, or memory claim."
            ),
            "artifact": f"runs/{run_name}/multiturn_harness_report.md",
            "falsifier": (
                "A supported chat template fails segment coverage or exact-token parity, cached-vs-recomputed boundary "
                "states exceed tolerance, self-patching is not a no-op, or null-direction drift explains the claimed trace."
            ),
        }
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
