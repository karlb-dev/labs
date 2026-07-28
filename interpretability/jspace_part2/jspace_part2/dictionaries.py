# Dictionary builders for J-space instruments.
#
# D_l = normalize((W_U ⊙ g) @ J_l) — one row per vocab token: the direction
# at layer l along which pushing h most raises that token's FINAL logit.
# Logit dictionary = normalize(W_U ⊙ g) (no Jacobian; layer-independent).
from __future__ import annotations

import torch


def _final_norm_module(hf_model):
    """The pre-unembed norm, wrapper-aware (Gemma4 nests under
    `model.language_model`, most others sit at `model`)."""
    for path in ("model.norm", "model.language_model.norm",
                 "model.final_layernorm", "transformer.ln_f"):
        obj = hf_model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("could not locate the final pre-unembed norm")


def effective_gain(hf_model) -> torch.Tensor:
    """The final norm's EFFECTIVE per-dimension gain, measured rather than
    assumed.

    RMSNorm implementations disagree on convention: Llama/OLMo/Qwen apply
    `x_normed * weight`, while Gemma applies `x_normed * (1 + weight)`
    (transformers PR #29402). Reading `.weight` directly is therefore
    WRONG for Gemma — and silently so, because Gemma's stored weights sit
    near 0 (the effective gain lives in the 1+). A dictionary built that
    way would be near-zero/sign-scrambled with no error raised.

    Rather than special-casing families, recover the gain by probing the
    module: for any RMSNorm, rms(ones) == 1, so norm(ones) == gain. This
    is convention-agnostic and stays correct if a future architecture
    invents another variant.
    """
    norm = _final_norm_module(hf_model)
    w = norm.weight.detach()
    ones = torch.ones(1, w.shape[0], device=w.device, dtype=torch.float32)
    with torch.no_grad():
        g = norm(ones).detach().float().reshape(-1)
    return g


def unembed_and_gain(hf_model) -> tuple[torch.Tensor, torch.Tensor]:
    W_U = hf_model.get_output_embeddings().weight.detach().float()
    return W_U, effective_gain(hf_model)


def build_j_dictionaries(hf_model, lens, layers, device="cuda",
                         dtype=torch.float16,
                         chunk: int = 65536) -> dict[int, torch.Tensor]:
    """Chunked fp16-staged build: large-vocab models (Qwen 248k rows) OOM if
    the fp32 intermediate is materialized whole (part-1 s18 lesson, relearned
    on this card at 93 GB). Transient cost ≈ chunk×d fp32 only."""
    W = hf_model.get_output_embeddings().weight.detach()      # bf16, on GPU
    g = effective_gain(hf_model).to(device)   # measured, not assumed
    V, d = W.shape
    out = {}
    for l in layers:
        J = lens.jacobians[l].to(device=device, dtype=torch.float32)
        D = torch.empty(V, d, device=device, dtype=dtype)
        for s in range(0, V, chunk):
            e = min(s + chunk, V)
            blk = (W[s:e].float().to(device) * g[None, :]) @ J
            D[s:e] = torch.nn.functional.normalize(blk, dim=1).to(dtype)
            del blk
        out[l] = D
        del J
        torch.cuda.empty_cache()
    return out


def build_logit_dictionary(hf_model, layers, device="cuda",
                           dtype=torch.float16) -> dict[int, torch.Tensor]:
    W_U, g = unembed_and_gain(hf_model)
    D = torch.nn.functional.normalize(
        (W_U.to(device) * g.to(device)[None, :]), dim=1).to(dtype)
    return {l: D for l in layers}
