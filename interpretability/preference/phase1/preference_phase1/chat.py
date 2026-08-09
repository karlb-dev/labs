"""Chat-template rendering with the Lab 15 parity discipline.

Every prompt is rendered through ``tokenizer.apply_chat_template`` and then
tokenized with ``add_special_tokens=False``; string-render tokenization
must equal direct template tokenization, and the generation boundary is the
final rendered prompt token (addendum E5: the frozen decision position).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .canonical import sha256_text


@dataclasses.dataclass(frozen=True)
class RenderedPrompt:
    rendered: str
    input_ids: tuple[int, ...]
    rendered_sha256: str
    ids_sha256: str
    parity_ok: bool
    boundary_index: int          # index of the final prompt token (decision pos)


def _ids_of(out: Any) -> list[int]:
    # transformers 5: apply_chat_template(tokenize=True) returns a
    # BatchEncoding (UserDict) — pull input_ids via mapping protocol.
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(x) for x in out]


def render_messages(tokenizer: Any, messages: list[dict[str, str]],
                    *, add_generation_prompt: bool = True) -> RenderedPrompt:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]
    direct = _ids_of(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt))
    parity = ids == direct
    return RenderedPrompt(
        rendered=rendered,
        input_ids=tuple(ids),
        rendered_sha256=sha256_text(rendered),
        ids_sha256=sha256_text(" ".join(str(i) for i in ids)),
        parity_ok=parity,
        boundary_index=len(ids) - 1,
    )


def render_item_prompt(tokenizer: Any, item: dict[str, Any]) -> RenderedPrompt:
    messages = [
        {"role": "system", "content": item["system_prompt"]},
        {"role": "user", "content": item["user_prompt"]},
    ]
    return render_messages(tokenizer, messages, add_generation_prompt=True)


def boundary_audit_rows(tokenizer: Any, items: list[dict[str, Any]],
                        *, sample: int = 8) -> list[dict[str, Any]]:
    """Chat-template audit rows (plan §4.3): parity, boundary, hashes, and
    no-assistant-leak check (the rendered prompt must end at the assistant
    stub with zero answer tokens)."""
    rows = []
    for item in items[:sample]:
        rp = render_item_prompt(tokenizer, item)
        no_prompt = render_messages(
            tokenizer,
            [{"role": "system", "content": item["system_prompt"]},
             {"role": "user", "content": item["user_prompt"]}],
            add_generation_prompt=False,
        )
        prefix_preserved = rp.input_ids[: len(no_prompt.input_ids)] == no_prompt.input_ids
        rows.append({
            "item_id": item["item_id"],
            "parity_ok": rp.parity_ok,
            "generation_prompt_preserves_prefix": prefix_preserved,
            "boundary_index": rp.boundary_index,
            "prompt_token_count": len(rp.input_ids),
            "rendered_sha256": rp.rendered_sha256,
            "ids_sha256": rp.ids_sha256,
            "tail_tokens": "|".join(
                tokenizer.decode([i]) for i in rp.input_ids[-4:]),
        })
    return rows


def target_ids(tokenizer: Any, code: str, *, leading_space: bool = False) -> tuple[int, ...]:
    text = (" " + code) if leading_space else code
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return tuple(int(x) for x in ids)
