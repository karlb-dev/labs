# Dictionary builders for J-space instruments.
#
# D_l = normalize((W_U ⊙ g) @ J_l) — one row per vocab token: the direction
# at layer l along which pushing h most raises that token's FINAL logit.
# Logit dictionary = normalize(W_U ⊙ g) (no Jacobian; layer-independent).
from __future__ import annotations

import torch


def unembed_and_gain(hf_model) -> tuple[torch.Tensor, torch.Tensor]:
    W_U = hf_model.get_output_embeddings().weight.detach().float()
    g = hf_model.model.norm.weight.detach().float()
    return W_U, g


def build_j_dictionaries(hf_model, lens, layers, device="cuda",
                         dtype=torch.float16,
                         chunk: int = 65536) -> dict[int, torch.Tensor]:
    """Chunked fp16-staged build: large-vocab models (Qwen 248k rows) OOM if
    the fp32 intermediate is materialized whole (part-1 s18 lesson, relearned
    on this card at 93 GB). Transient cost ≈ chunk×d fp32 only."""
    W = hf_model.get_output_embeddings().weight.detach()      # bf16, on GPU
    g = hf_model.model.norm.weight.detach().float().to(device)
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
