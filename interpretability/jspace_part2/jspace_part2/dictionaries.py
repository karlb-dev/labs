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
                         dtype=torch.float16) -> dict[int, torch.Tensor]:
    W_U, g = unembed_and_gain(hf_model)
    out = {}
    for l in layers:
        J = lens.jacobians[l].to(device=device, dtype=torch.float32)
        D = (W_U.to(device) * g.to(device)[None, :]) @ J
        out[l] = torch.nn.functional.normalize(D, dim=1).to(dtype)
        del D, J
    return out


def build_logit_dictionary(hf_model, layers, device="cuda",
                           dtype=torch.float16) -> dict[int, torch.Tensor]:
    W_U, g = unembed_and_gain(hf_model)
    D = torch.nn.functional.normalize(
        (W_U.to(device) * g.to(device)[None, :]), dim=1).to(dtype)
    return {l: D for l in layers}
