"""Pinned model identities (plan §49; revisions resolved 2026-08-08 and
frozen — no silent substitution)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import paths
from .canonical import sha256_text


@dataclass(frozen=True)
class ModelPin:
    key: str
    model_id: str
    revision: str
    dtype: str
    family: str            # olmo | qwen | gemma | smoke
    system_role_ok: bool = True
    disable_thinking: bool = False
    post_softcap_logits: bool = False
    min_vram_gb: int = 80


PINS: dict[str, ModelPin] = {
    "smoke": ModelPin(
        key="smoke", model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        dtype="float32", family="smoke", min_vram_gb=0),
    "olmo32b": ModelPin(
        key="olmo32b", model_id="allenai/Olmo-3.1-32B-Instruct",
        revision="ac0587e4a7744a551c059d8cd17ba220bc940dae",
        dtype="bfloat16", family="olmo"),
    "olmo7b": ModelPin(
        key="olmo7b", model_id="allenai/Olmo-3-7B-Instruct",
        revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        dtype="bfloat16", family="olmo", min_vram_gb=24),
    "qwen": ModelPin(
        key="qwen", model_id="Qwen/Qwen3.6-27B",
        revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        dtype="bfloat16", family="qwen", disable_thinking=True),
    "gemma": ModelPin(
        key="gemma", model_id="google/gemma-4-31B-it",
        revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
        dtype="bfloat16", family="gemma", system_role_ok=False,
        post_softcap_logits=True),
}

PRIMARY = PINS["olmo32b"]
MODEL_ORDER = ("olmo32b", "olmo7b", "qwen", "gemma")


def _use_local_cache() -> None:
    os.environ.setdefault("HF_HUB_CACHE", str(paths.hf_local_cache()))


def load_tokenizer(pin: ModelPin):
    _use_local_cache()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(pin.model_id, revision=pin.revision)


def chat_template_hash(tokenizer) -> str:
    return sha256_text(str(getattr(tokenizer, "chat_template", None) or ""))


def model_manifest(pin: ModelPin, tokenizer=None) -> dict:
    out = {"model_key": pin.key, "model_id": pin.model_id,
           "revision": pin.revision, "dtype": pin.dtype,
           "family": pin.family}
    if tokenizer is not None:
        out.update({
            "tokenizer_class": type(tokenizer).__name__,
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
            "chat_template_sha256": chat_template_hash(tokenizer),
        })
    return out
