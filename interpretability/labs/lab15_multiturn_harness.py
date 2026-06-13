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

CACHE_PARITY_ATOL = 2e-3
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
    ids = bundle.tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
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

    for idx, message in enumerate(conv.messages):
        partial_rendered = render_messages(bundle, conv.messages[: idx + 1])
        partial_ids = token_ids(bundle, partial_rendered)
        if partial_ids[: len(prev_ids)] != prev_ids:
            stable_token_prefix = False
        if not partial_rendered.startswith(prev_rendered):
            stable_string_prefix = False

        message_start = len(prev_ids)
        message_end = len(partial_ids)
        message_start_char = len(prev_rendered) if partial_rendered.startswith(prev_rendered) else 0
        message_end_char = len(partial_rendered)
        appended_text = partial_rendered[message_start_char:message_end_char]
        rel = appended_text.find(message["content"])
        if rel >= 0:
            content_start_char = message_start_char + rel
            content_end_char = content_start_char + len(message["content"])
        else:
            content_start_char = None
            content_end_char = None

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
        direct_with_prompt_ids = bundle.tokenizer.apply_chat_template(
            list(prefix), tokenize=True, add_generation_prompt=True
        )
        if isinstance(direct_with_prompt_ids, dict):
            direct_with_prompt_ids = direct_with_prompt_ids["input_ids"]
        if direct_with_prompt_ids and isinstance(direct_with_prompt_ids[0], list):
            direct_with_prompt_ids = direct_with_prompt_ids[0]
        direct_with_prompt_ids = [int(x) for x in direct_with_prompt_ids]
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

        ok = (
            coverage_ok
            and no_gaps
            and positive_message_widths
            and content_inside_message
            and no_assistant_leak
            and generation_prompt_ok
            and conv.info["string_vs_direct_template_ids_match"]
            and conv.info["incremental_token_prefix_stable"]
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
            "content_span_method_counts": dict(method_counts),
            "all_content_spans_offset_mapped": all(seg.content_span_found for seg in segments),
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
            "Message spans are derived by repeatedly rendering prefixes with the model's own chat template. "
            "Content spans are then mapped with tokenizer offset mapping when available. The hard checks require "
            "template token parity, stable prefix tokenization, complete coverage, no gaps, and no assistant content "
            "inside user content spans. User-turn prefixes rendered with add_generation_prompt=True must also "
            "extend the no-generation prefix cleanly."
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
    worst_logits = 0.0
    worst_record: dict[str, Any] | None = None

    for conv_name, conv in conversations_by_name.items():
        boundaries = [seg.boundary_token for seg in conv.segments]
        full = full_recompute_boundary_states(bundle, conv.input_ids, boundaries)
        cached = incremental_boundary_states(bundle, conv.input_ids, boundaries)
        for seg, (full_stream, full_logits), (cached_stream, cached_logits) in zip(conv.segments, full, cached):
            hidden_diff = (full_stream - cached_stream).abs()
            logit_diff = (full_logits - cached_logits).abs()
            row = {
                "conversation": conv_name,
                "segment_index": seg.index,
                "role": seg.role,
                "turn_index_non_system": role_turn_index(conv.segments, seg.index),
                "boundary_token": seg.boundary_token,
                "max_abs_hidden_diff": float(hidden_diff.max()),
                "mean_abs_hidden_diff": float(hidden_diff.mean()),
                "max_abs_logit_diff": float(logit_diff.max()),
                "mean_abs_logit_diff": float(logit_diff.mean()),
                "ok_at_tolerance": float(hidden_diff.max()) <= CACHE_PARITY_ATOL,
            }
            rows.append(row)
            if float(hidden_diff.max()) > worst_hidden:
                worst_record = row
            worst_hidden = max(worst_hidden, float(hidden_diff.max()))
            worst_mean_hidden = max(worst_mean_hidden, float(hidden_diff.mean()))
            worst_logits = max(worst_logits, float(logit_diff.max()))

    csv_path = ctx.path("diagnostics", "cache_recompute_parity_by_boundary.csv")
    bench.write_csv_with_context(ctx, csv_path, rows)
    ctx.register_artifact(csv_path, "diagnostic", "Boundary-level KV-cache residuals versus full recompute.")

    result = {
        "n_conversations": len(conversations_by_name),
        "n_boundaries": len(rows),
        "max_abs_hidden_diff": worst_hidden,
        "max_mean_hidden_diff": worst_mean_hidden,
        "max_abs_logit_diff": worst_logits,
        "atol_hidden": CACHE_PARITY_ATOL,
        "ok": worst_hidden <= CACHE_PARITY_ATOL,
        "worst_record": worst_record,
        "explanation": (
            "Each conversation prefix was measured two ways: full recompute of the prefix and incremental prefill "
            "with past_key_values. The residual stream at each message boundary must match or later turn traces "
            "may be cache artifacts. Logit diffs are reported as an extra smoke check; the hard gate is residual diff."
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
# Plots
# ---------------------------------------------------------------------------


def plot_turn_trace(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.5, 5.8))
    styles = {
        ("orchid_topic", "topic_orchid_minus_archive"): ("tab:green", "-", "orchid conversation, topic direction"),
        ("archive_length_control", "topic_orchid_minus_archive"): ("tab:blue", "--", "archive control, topic direction"),
        ("orchid_topic", "length_matched_null"): ("tab:orange", ":", "orchid conversation, length null"),
        ("orchid_topic", "random_null_00"): ("tab:gray", ":", "orchid conversation, random null"),
    }
    for (conv, direction), (color, linestyle, label) in styles.items():
        sub = [
            r for r in rows
            if r["conversation"] == conv and r["direction"] == direction and r["role"] != "system"
        ]
        if not sub:
            continue
        ax.plot(
            [float(r["turn_index_non_system"]) for r in sub],
            [float(r["cumulative_message_mean_projection"]) for r in sub],
            marker="o",
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    ax.set_xlabel("message boundary, system excluded")
    ax.set_ylabel("cumulative mean projection")
    ax.set_title("Lab 15 demo trace: topic direction against null controls")
    ax.legend(fontsize=8)
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "demo_turn_trace.png", "Topic and null projection traces over scripted multi-turn conversations.")


def plot_trace_depth_sweep(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.5, 5.8))
    styles = {
        ("orchid_topic", "topic_orchid_minus_archive"): ("tab:green", "-", "topic direction on orchid conversation"),
        ("archive_length_control", "topic_orchid_minus_archive"): ("tab:blue", "--", "topic direction on archive control"),
        ("orchid_topic", "length_matched_null"): ("tab:orange", ":", "length-null on orchid conversation"),
        ("orchid_topic", "random_null_00"): ("tab:gray", ":", "random-null on orchid conversation"),
    }
    for (conv, direction), (color, linestyle, label) in styles.items():
        sub = [r for r in rows if r["conversation"] == conv and r["direction"] == direction]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: int(r["stream_depth"]))
        ax.plot(
            [int(r["stream_depth"]) for r in sub],
            [float(r["cumulative_projection_slope"]) for r in sub],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.set_xlabel("stream depth k, after k blocks")
    ax.set_ylabel("projection slope over message boundaries")
    ax.set_title("Depth sweep for the demo trace and controls")
    ax.legend(fontsize=8)
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "trace_depth_sweep.png", "Projection-slope depth sweep for topic and null controls.")


def plot_turn_span_map(ctx: bench.RunContext, conversations_by_name: Mapping[str, RenderedConversation]) -> None:
    fig, ax = bench.new_figure(figsize=(9.5, 4.8))
    y = 0
    yticks = []
    ylabels = []
    for conv_name, conv in conversations_by_name.items():
        for seg in conv.segments:
            ax.barh(y, seg.message_end - seg.message_start, left=seg.message_start, height=0.62, alpha=0.35)
            ax.barh(y, seg.content_end - seg.content_start, left=seg.content_start, height=0.28, alpha=0.85)
            yticks.append(y)
            ylabels.append(f"{conv_name}:{seg.index}:{seg.role}")
            y += 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("token index in rendered chat")
    ax.set_title("Message spans, wide bars, and content spans, narrow bars")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "turn_span_map.png", "Token-span map for rendered multi-turn conversations.")


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
