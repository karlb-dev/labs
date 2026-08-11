"""Load Muse Glimmer and wrap for jlens."""
from __future__ import annotations

from pathlib import Path

import torch

from .paths import (
    EXPECTED_D_MODEL,
    EXPECTED_N_LAYERS,
    MUSE_MODEL_ID,
    MUSE_MODEL_REVISION,
    model_snapshot,
)
from .util import log


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible. HARD STOP: no CPU fallback for model work."
        )


def _resolve_snapshot() -> Path:
    snap = model_snapshot()
    if not snap.exists():
        raise RuntimeError(
            f"pinned Muse snapshot missing: {snap}\n"
            f"Download: HF_HUB_CACHE=/content/hf_local hf download "
            f"{MUSE_MODEL_ID} --revision {MUSE_MODEL_REVISION}"
        )
    # Require config + at least one weight shard
    if not (snap / "config.json").exists():
        raise RuntimeError(f"config.json missing under {snap}")
    shards = list(snap.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors under {snap}")
    return snap


def load_hf(dtype=torch.bfloat16):
    """Load tokenizer + HF multimodal/causal model from the pinned snapshot."""
    import transformers

    require_cuda()
    snap = _resolve_snapshot()
    log(f"loading Muse from {snap}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snap), local_files_only=True, trust_remote_code=True
    )

    # Prefer the image-text class; fall back through AutoModel variants.
    hf_model = None
    errors = []
    loaders = []
    if hasattr(transformers, "AutoModelForImageTextToText"):
        loaders.append(("AutoModelForImageTextToText", transformers.AutoModelForImageTextToText))
    if hasattr(transformers, "AutoModelForCausalLM"):
        loaders.append(("AutoModelForCausalLM", transformers.AutoModelForCausalLM))
    if hasattr(transformers, "AutoModelForVision2Seq"):
        loaders.append(("AutoModelForVision2Seq", transformers.AutoModelForVision2Seq))
    loaders.append(("AutoModel", transformers.AutoModel))

    for name, cls in loaders:
        try:
            log(f"trying {name}.from_pretrained ...")
            hf_model = cls.from_pretrained(
                str(snap),
                local_files_only=True,
                torch_dtype=dtype,
                device_map={"": 0},
                trust_remote_code=True,
            )
            log(f"loaded via {name}: {type(hf_model).__name__}")
            break
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {e}")
            log(f"  {name} failed: {type(e).__name__}: {e}")

    if hf_model is None:
        raise RuntimeError("all loaders failed:\n" + "\n".join(errors))

    hf_model.eval()
    # Confirm a parameter is on CUDA
    param = next(hf_model.parameters())
    if param.device.type != "cuda":
        raise RuntimeError(f"model parameter not on CUDA: {param.device}")
    return hf_model, tokenizer


def _patch_muse_unembed(model, hf_model) -> float:
    """Muse applies ``output_multiplier`` *before* logit softcap.

    Upstream jlens only knows softcap (Gemma-style). Without the multiplier,
    pre-softcap logits are ~5× too large, softcap saturates, and top-k is
    garbage. Fold the multiplier into ``unembed`` so readout matches HF.
    """
    text_config = hf_model.config.get_text_config()
    mult = float(getattr(text_config, "output_multiplier", 1.0) or 1.0)
    model._output_multiplier = mult
    if abs(mult - 1.0) < 1e-12:
        return mult

    softcap = model._logit_softcap
    lm_head = model._lm_head
    final_norm = model._final_norm

    def unembed(residual):
        target_device = lm_head.weight.device
        target_dtype = lm_head.weight.dtype
        logits = lm_head(final_norm(residual.to(target_dtype).to(target_device)))
        logits = logits * mult
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    model.unembed = unembed  # type: ignore[method-assign]
    log(f"patched Muse unembed: output_multiplier={mult} softcap={softcap}")
    return mult


def wrap_jlens(hf_model, tokenizer, *, force_bos: bool = True):
    """Wrap with jlens.from_hf; try multimodal layouts explicitly if needed."""
    import jlens
    from jlens.hf import Layout

    try:
        model = jlens.from_hf(hf_model, tokenizer, force_bos=force_bos)
    except ValueError:
        # Multimodal wrappers often nest the LM under model.language_model
        for path in ("model.language_model", "language_model", "model"):
            try:
                model = jlens.from_hf(
                    hf_model,
                    tokenizer,
                    layout=Layout(path=path),
                    force_bos=force_bos,
                )
                log(f"jlens layout override: {path}")
                break
            except Exception as e:  # noqa: BLE001
                log(f"layout {path} failed: {e}")
        else:
            raise
    if model.n_layers != EXPECTED_N_LAYERS or model.d_model != EXPECTED_D_MODEL:
        raise RuntimeError(
            f"shape mismatch: got {model.n_layers} x {model.d_model}, "
            f"expected {EXPECTED_N_LAYERS} x {EXPECTED_D_MODEL}"
        )
    _patch_muse_unembed(model, hf_model)
    return model


def load_muse():
    """Full load: (lens_model, hf_model, tokenizer)."""
    hf_model, tokenizer = load_hf()
    model = wrap_jlens(hf_model, tokenizer)
    return model, hf_model, tokenizer


def admission_facts(model, hf_model, tokenizer) -> dict:
    text_config = hf_model.config.get_text_config()
    return {
        "model_id": MUSE_MODEL_ID,
        "revision": MUSE_MODEL_REVISION,
        "hf_class": type(hf_model).__name__,
        "jlens_class": type(model).__name__,
        "layout_path": getattr(getattr(model, "layout", None), "path", None),
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "vocab_size": int(getattr(text_config, "vocab_size", -1)),
        "logit_softcap": model._logit_softcap,
        "output_multiplier": float(
            getattr(model, "_output_multiplier", None)
            or getattr(text_config, "output_multiplier", 1.0)
            or 1.0
        ),
        "dtype": str(next(hf_model.parameters()).dtype),
        "device": str(next(hf_model.parameters()).device),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "add_bos_token_attr": getattr(tokenizer, "add_bos_token", None),
        "chat_template_present": tokenizer.chat_template is not None,
        "attn_implementation": str(
            getattr(hf_model.config, "_attn_implementation", "unknown")
        ),
    }
