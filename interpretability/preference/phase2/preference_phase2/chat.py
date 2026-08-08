"""Per-model chat rendering with parity discipline + shims (plan §50).

- string-render path is authoritative for input_ids; tokenized-path
  parity is a hard gate (Phase 1 / Lab 15 discipline)
- Qwen: thinking disabled through the exact template API; a hard
  assertion rejects any think span before the decision
- Gemma: no-system-role shim (system text folded into the first user
  turn, frozen separator); render parity audited like every model
- site token maps are resolved from the fast tokenizer's offset mapping
  against the user prompt's location inside the rendered string
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .canonical import sha256_text
from .models import ModelPin

GEMMA_SYSTEM_SEP = "\n\n[Task instructions]\n"
THINK_MARKERS = ("<think>", "</think>", "<|think|>")


@dataclass(frozen=True)
class RenderedPrompt:
    rendered: str
    input_ids: tuple[int, ...]
    rendered_sha256: str
    ids_sha256: str
    parity_ok: bool
    boundary_index: int          # final prompt token (direct-output site)
    site_token_index: Mapping[str, int]


def _ids_of(out) -> list[int]:
    ids = out["input_ids"] if not hasattr(out, "tolist") else out
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def messages_for(pin: ModelPin, system_prompt: str,
                 user_prompt: str) -> list[dict[str, str]]:
    if pin.system_role_ok:
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    return [{"role": "user",
             "content": system_prompt + GEMMA_SYSTEM_SEP + user_prompt}]


def _apply_template(tokenizer, pin: ModelPin, messages, *, tokenize: bool):
    kwargs: dict[str, Any] = {"add_generation_prompt": True,
                              "tokenize": tokenize}
    if pin.disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        if "enable_thinking" in kwargs:
            # template API without the flag: hard failure for Qwen (the
            # port gate demands the exact API), pass-through otherwise
            if pin.family == "qwen":
                raise RuntimeError(
                    "qwen port: chat template does not accept "
                    "enable_thinking; STOP_P(qwen)")
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_item_prompt(tokenizer, pin: ModelPin,
                       item: Mapping[str, Any]) -> RenderedPrompt:
    messages = messages_for(pin, item["system_prompt"], item["user_prompt"])
    rendered = _apply_template(tokenizer, pin, messages, tokenize=False)
    if pin.disable_thinking:
        # Qwen's template disables thinking by pre-filling a constant
        # EMPTY think block in the generation preamble; the hard gate
        # rejects (a) any non-empty think content and (b) a think block
        # appearing before the user turn ends (anchor movement).
        import re
        blocks = re.findall(r"<think>(.*?)</think>", rendered, re.S)
        n_open = rendered.count("<think>")
        if len(blocks) != n_open or any(b.strip() for b in blocks):
            raise RuntimeError(
                "non-empty or unclosed think span before the decision "
                "(port gate hard fail)")
        first_think = rendered.find("<think>")
        if 0 <= first_think < rendered.rindex(item["user_prompt"]):
            raise RuntimeError(
                "think block precedes the user turn (anchor moved; port "
                "gate hard fail)")
    enc = tokenizer(rendered, add_special_tokens=False,
                    return_offsets_mapping=True)
    ids = _ids_of(enc)
    offsets = enc["offset_mapping"]
    if offsets and isinstance(offsets[0], list) and offsets and \
            isinstance(offsets[0][0], list):
        offsets = offsets[0]
    direct = _ids_of(_apply_template(tokenizer, pin, messages,
                                     tokenize=True))
    parity_ok = ids == direct
    user_start = rendered.rindex(item["user_prompt"])
    site_tokens: dict[str, int] = {}
    for site, char_pos in (item.get("site_char_spans") or {}).items():
        target = user_start + int(char_pos)
        tok_idx = None
        if site.endswith("_start"):
            # start-anchored: the token containing the character AT the
            # start index (first token whose span extends past it)
            for i, (a, b) in enumerate(offsets):
                if b > target:
                    tok_idx = i
                    break
        else:
            # end-anchored: the token containing the site's final
            # character (char_pos is an exclusive end)
            for i, (a, b) in enumerate(offsets):
                if a < target <= b:
                    tok_idx = i
                    break
                if a >= target:
                    tok_idx = max(0, i - 1)
                    break
        site_tokens[site] = tok_idx if tok_idx is not None else len(ids) - 1
    site_tokens["final_prompt_token" if item.get("channel") != "RO"
                else "ro_final_prompt_token"] = len(ids) - 1
    return RenderedPrompt(
        rendered=rendered, input_ids=tuple(ids),
        rendered_sha256=sha256_text(rendered),
        ids_sha256=sha256_text(" ".join(str(i) for i in ids)),
        parity_ok=parity_ok, boundary_index=len(ids) - 1,
        site_token_index=site_tokens,
    )


def target_ids(tokenizer, code: str) -> tuple[int, ...]:
    enc = tokenizer(code, add_special_tokens=False)
    return tuple(_ids_of(enc))
