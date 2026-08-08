"""Prompt rendering and position location (RENDER_AND_POSITION_CONTRACT).

Raw strings go through the upstream ``model.encode`` path verbatim; chat
renders go through the checkpoint tokenizer's template and are tokenized
with ``add_special_tokens=False`` (the template supplies its own special
tokens; per-lane BOS behavior is asserted and recorded at admission).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import torch


@dataclass
class Rendered:
    """A rendered prompt: text, token ids (1 x seq), and provenance."""

    text: str
    input_ids: torch.Tensor
    form: str  # raw | chat_generation | chat_prefill
    template_kwargs: dict

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def final_position(self) -> int:
        return self.seq_len - 1


def render_raw(model, text: str, *, max_length: int = 2048) -> Rendered:
    input_ids = model.encode(text, max_length=max_length)
    if input_ids.shape[1] >= max_length:
        raise RuntimeError(f"raw render hit max_length={max_length}: truncation")
    return Rendered(text=text, input_ids=input_ids, form="raw",
                    template_kwargs={"max_length": max_length})


def render_chat(
    model,
    messages: list[dict],
    *,
    continue_final: bool = False,
    max_length: int = 4096,
    extra_template_kwargs: dict | None = None,
) -> Rendered:
    """Render a role-dict prompt through the checkpoint chat template.

    ``continue_final=False`` appends the generation prompt (message list
    must end in a user turn); ``continue_final=True`` keeps the final
    assistant message open (prefill semantics).
    """
    tokenizer = model.tokenizer
    kwargs = dict(extra_template_kwargs or {})
    if continue_final:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True, **kwargs
        )
    else:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    encoded = tokenizer(
        text, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=max_length,
    )
    input_ids = encoded.input_ids.to(model.input_device)
    if input_ids.shape[1] >= max_length:
        raise RuntimeError(f"chat render hit max_length={max_length}: truncation")
    return Rendered(
        text=text, input_ids=input_ids,
        form="chat_prefill" if continue_final else "chat_generation",
        template_kwargs={"continue_final_message": continue_final,
                         "add_generation_prompt": not continue_final, **kwargs},
    )


# ---------------------------------------------------------------- positions

def _decode(tokenizer, ids) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def position_before_substring(
    rendered: Rendered, tokenizer, target: str, *, occurrence: str = "first"
) -> int:
    """Token position immediately preceding ``target``'s occurrence.

    Located by character offset in the rendered text, then converted to the
    token whose decoded prefix ends exactly at (or last strictly before)
    the target start; asserted by decode round-trip.
    """
    text = rendered.text
    start = text.find(target) if occurrence == "first" else text.rfind(target)
    if start < 0:
        raise ValueError(f"target {target!r} not in rendered text")
    ids = rendered.input_ids[0].tolist()
    # Walk the token boundary map.
    prefix_lengths = []
    for i in range(len(ids)):
        prefix_lengths.append(len(_decode(tokenizer, ids[: i + 1])))
    position = None
    for i, end in enumerate(prefix_lengths):
        if end <= start:
            position = i
        else:
            break
    if position is None:
        raise ValueError(f"no token ends at or before target start {start}")
    decoded_prefix = _decode(tokenizer, ids[: position + 1])
    remainder = text[len(decoded_prefix):]
    if not remainder.lstrip().startswith(target.lstrip()):
        raise AssertionError(
            f"position audit failed: after token {position} the text does "
            f"not continue with {target!r} (got {remainder[:40]!r})"
        )
    return position


def last_newline_position(rendered: Rendered, tokenizer) -> int:
    """Position of the last token whose decoded text contains a newline."""
    ids = rendered.input_ids[0].tolist()
    for i in range(len(ids) - 1, -1, -1):
        if "\n" in _decode(tokenizer, [ids[i]]):
            return i
    raise ValueError("no newline token in rendered prompt")


def find_token_span(
    rendered: Rendered, tokenizer, span_text: str, *, from_end: bool = True
) -> tuple[int, int]:
    """Inclusive token span [start, end] whose decode reproduces
    ``span_text`` (modulo surrounding whitespace), located by rendering
    boundaries — the span is matched at the character level then mapped to
    token indices, asserted by decode."""
    text = rendered.text
    start_char = text.rfind(span_text) if from_end else text.find(span_text)
    if start_char < 0:
        raise ValueError(f"span {span_text[:40]!r} not in rendered text")
    end_char = start_char + len(span_text)
    ids = rendered.input_ids[0].tolist()
    token_start = token_end = None
    running = ""
    for i in range(len(ids)):
        prev_len = len(running)
        running = _decode(tokenizer, ids[: i + 1])
        if token_start is None and len(running) > start_char:
            token_start = i
        if len(running) >= end_char and prev_len < end_char:
            token_end = i
            break
    if token_start is None or token_end is None:
        raise ValueError("span does not align to token stream")
    span_decoded = _decode(tokenizer, ids[token_start : token_end + 1])
    if unicodedata.normalize("NFC", span_text).strip() not in unicodedata.normalize(
        "NFC", span_decoded
    ):
        raise AssertionError(
            f"span audit failed: {span_decoded[:60]!r} !~ {span_text[:60]!r}"
        )
    return token_start, token_end


# ------------------------------------------------------------ target tokens

def single_token_forms(tokenizer, target: str) -> dict:
    """Candidate single-token realizations of ``target`` (contract §6).

    Returns {form: token_id} for the in-context (leading-space) and bare
    forms that (i) encode to exactly one token with add_special_tokens
    False and (ii) decode back to the target modulo ASCII-whitespace strip
    (NFC, case-sensitive).
    """
    normalized = unicodedata.normalize("NFC", target)
    forms: dict[str, int] = {}
    for variant, key in ((" " + normalized.strip(), "space"),
                         (normalized.strip(), "bare")):
        ids = tokenizer(variant, add_special_tokens=False).input_ids
        if len(ids) != 1:
            continue
        decoded = tokenizer.decode(ids)
        if unicodedata.normalize("NFC", decoded).strip() == normalized.strip():
            forms[key] = int(ids[0])
    return forms


def preferred_token(tokenizer, target: str) -> int | None:
    """The intervention/scoring token for ``target``: in-context (leading
    space) preferred, bare fallback; None => TOKENIZATION_GATED."""
    forms = single_token_forms(tokenizer, target)
    if "space" in forms:
        return forms["space"]
    if "bare" in forms:
        return forms["bare"]
    return None
