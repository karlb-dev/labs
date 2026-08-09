"""Model loading for the two lanes (never both in one process).

Loads from the pinned local snapshots only (`local_files_only`), wraps via
upstream ``jlens.from_hf``, and runs the admission shape/binding checks.
"""
from __future__ import annotations

import torch

from .paths import (
    OLMO_MODEL_ID,
    OLMO_MODEL_REVISION,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
    model_snapshot,
)

EXPECTED_LAYERS = 64
QWEN_D_MODEL = 5120
QWEN_EXPECTED_GDN_LAYERS = 48


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible. HARD STOP: no CPU fallback for model work."
        )


def _load(model_id: str, revision: str, *, dtype=torch.bfloat16):
    import transformers

    require_cuda()
    snapshot = model_snapshot(model_id, revision)
    if not snapshot.exists():
        raise RuntimeError(f"pinned snapshot missing: {snapshot}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=False,
    )
    model.eval()
    return model, tokenizer


def count_gdn_layers(hf_model) -> int:
    """Count Gated DeltaNet (fused linear-attention) blocks in Qwen 3.6."""
    count = 0
    for module in hf_model.modules():
        name = type(module).__name__.lower()
        if "gateddeltanet" in name or "gated_deltanet" in name:
            count += 1
    return count


def load_qwen():
    """Load the pinned Qwen lane; returns (lens_model, hf_model, tokenizer)."""
    import jlens

    hf_model, tokenizer = _load(QWEN_MODEL_ID, QWEN_MODEL_REVISION)
    model = jlens.from_hf(hf_model, tokenizer)
    if model.n_layers != EXPECTED_LAYERS or model.d_model != QWEN_D_MODEL:
        raise RuntimeError(
            f"Qwen shape mismatch: {model.n_layers} layers, d={model.d_model}"
        )
    gdn = count_gdn_layers(hf_model)
    if gdn != QWEN_EXPECTED_GDN_LAYERS:
        raise RuntimeError(f"expected 48 GDN blocks, found {gdn}")
    return model, hf_model, tokenizer


def load_olmo():
    """Load the pinned OLMo lane; returns (lens_model, hf_model, tokenizer)."""
    import jlens

    hf_model, tokenizer = _load(OLMO_MODEL_ID, OLMO_MODEL_REVISION)
    model = jlens.from_hf(hf_model, tokenizer)
    if model.n_layers != EXPECTED_LAYERS:
        raise RuntimeError(f"OLMo layer mismatch: {model.n_layers}")
    return model, hf_model, tokenizer


def admission_facts(model, hf_model, tokenizer) -> dict:
    """Recorded-as-fact runtime/render properties for the manifest."""
    text_config = hf_model.config.get_text_config()
    return {
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "vocab_size": int(getattr(text_config, "vocab_size", -1)),
        "logit_softcap": model._logit_softcap,
        "dtype": str(next(hf_model.parameters()).dtype),
        "attn_implementation": str(
            getattr(hf_model.config, "_attn_implementation", "unknown")
        ),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "add_bos_token_attr": getattr(tokenizer, "add_bos_token", None),
        "chat_template_present": tokenizer.chat_template is not None,
    }
