"""Pinned model identities (addendum E8; HARNESS_DECISION.md).

The repo has no central pinning machinery, so the campaign pins here.
Revisions resolved against the HF Hub on 2026-08-07 and frozen; every
manifest records them. Weights hub-download to local NVMe
(paths.hf_local_cache()); model loads never stream through DriveFS.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any


@dataclasses.dataclass(frozen=True)
class ModelPin:
    key: str                 # model_tier vocabulary: a | b | c
    model_id: str
    revision: str
    dtype: str               # load dtype for primary use
    chat_template: bool = True


PINS: dict[str, ModelPin] = {
    "a": ModelPin("a", "HuggingFaceTB/SmolLM2-135M-Instruct",
                  "12fd25f77366fa6b3b4b768ec3050bf629380bac", "float32"),
    "b": ModelPin("b", "allenai/Olmo-3-7B-Instruct",
                  "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc", "bfloat16"),
    "c": ModelPin("c", "allenai/Olmo-3.1-32B-Instruct",
                  "ac0587e4a7744a551c059d8cd17ba220bc940dae", "bfloat16"),
}

PRIMARY = PINS["b"]
SMOKE = PINS["a"]


def _use_local_cache() -> None:
    from . import paths

    os.environ.setdefault("HF_HUB_CACHE", str(paths.hf_local_cache()))


def load_tokenizer(pin: ModelPin) -> Any:
    _use_local_cache()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(pin.model_id, revision=pin.revision)


def chat_template_hash(tokenizer: Any) -> str:
    from .canonical import sha256_text

    template = getattr(tokenizer, "chat_template", None) or ""
    return sha256_text(str(template))


def model_manifest(pin: ModelPin, tokenizer: Any | None = None) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "model_id": pin.model_id,
        "revision": pin.revision,
        "dtype": pin.dtype,
        "model_tier": pin.key,
    }
    if tokenizer is not None:
        manifest["tokenizer_class"] = type(tokenizer).__name__
        manifest["vocab_size"] = int(getattr(tokenizer, "vocab_size", 0) or 0)
        manifest["chat_template_sha256"] = chat_template_hash(tokenizer)
    return manifest
