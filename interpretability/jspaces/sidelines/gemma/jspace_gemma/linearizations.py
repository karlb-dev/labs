"""Small exact linearization primitives used by later G3/G4 interventions."""
from __future__ import annotations

import torch


def rmsnorm_first_order(
    x: torch.Tensor,
    *,
    clean: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    """Evaluate the analytic first-order RMSNorm chart around ``clean``."""
    clean32 = clean.float()
    delta = x.float() - clean32
    mean_square = clean32.pow(2).mean(dim=-1, keepdim=True) + epsilon
    inverse_rms = mean_square.pow(-0.5)
    directional_mean = (clean32 * delta).mean(dim=-1, keepdim=True)
    derivative = delta * inverse_rms - clean32 * directional_mean * mean_square.pow(-1.5)
    base = clean32 * inverse_rms
    result = base + derivative
    if weight is not None:
        result = result * weight.float()
    return result.to(x.dtype)


def frozen_gate_mlp(
    x: torch.Tensor,
    *,
    clean: torch.Tensor,
    gate_proj,
    up_proj,
    down_proj,
    activation,
) -> torch.Tensor:
    clean_gate = activation(gate_proj(clean)).detach()
    return down_proj(clean_gate * up_proj(x))
