"""Dictionary construction and readout parity (contract §1).

Three objects, kept distinct (addendum §2.2):

(a) readout parity — recompute ``unembed(J_l @ h)`` independently and
    match ``lens.apply`` within fp tolerance (hard stop);
(b) intervention vectors — ``v_t = J_l^T u_t`` (row *t* of ``W_U J_l``),
    fixed and norm-free;
(c) probe form — ``<v_t, h>`` matches full readout only up to the
    per-position final-norm rescaling; conformance uses top-k order.
"""
from __future__ import annotations

import torch

PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-4


def unembedding_rows(model, token_ids: list[int]) -> torch.Tensor:
    """Rows of W_U for the given tokens: ``[n_tokens, d_model]`` fp32 CPU."""
    weight = model._lm_head.weight.detach()
    return weight[token_ids].float().cpu()


def final_norm_gain(model) -> torch.Tensor | None:
    """The final RMSNorm gain vector g, or None if the norm has no weight."""
    norm = model._final_norm
    weight = getattr(norm, "weight", None)
    return None if weight is None else weight.detach().float().cpu()


def token_vectors(
    lens, model, layer: int, token_ids: list[int], *, fold_gain: bool = False
) -> torch.Tensor:
    """Source-space intervention vectors ``v_t`` at ``layer``.

    ``v_t = J_l^T u_t`` (paper §2.1: row *t* of ``W_U J_l``), fp32 CPU,
    shape ``[n_tokens, d_model]``. ``fold_gain=True`` uses ``g ⊙ u_t``
    instead of ``u_t`` (the g-folding audit's alternative; the
    paper-literal default is unfolded). Computes on whatever device the
    lens layer lives on (see :func:`lens_to_device`).
    """
    U = unembedding_rows(model, token_ids)  # [n, d] fp32 CPU
    if fold_gain:
        gain = final_norm_gain(model)
        if gain is not None:
            U = U * gain
    J = lens.jacobians[layer].float()  # [d, d]; transported = J @ h
    return (U.to(J.device) @ J).cpu()  # rows: u_t^T J  -> [n, d]


def lens_to_device(lens, device, *, layers: list[int] | None = None) -> None:
    """Move lens Jacobians to ``device`` in place (transport then runs
    device-resident instead of re-uploading 100MB per call)."""
    for layer in layers if layers is not None else list(lens.jacobians):
        lens.jacobians[layer] = lens.jacobians[layer].to(device)


def recompute_readout(model, lens, layer: int, residual: torch.Tensor) -> torch.Tensor:
    """Independent recomputation of the lens readout at ``layer``:
    ``unembed(J_l @ h)`` including final norm and softcap, fp32 CPU."""
    J_device = lens.jacobians[layer].device
    transported = lens.transport(residual.float().to(J_device), layer)
    return model.unembed(transported).float().cpu()


def readout_parity(
    model, lens, prompt: str, *, layers: list[int], positions: list[int]
) -> dict:
    """Hard-stop parity check (contract §1a): independent recomputation vs
    ``lens.apply`` on actual activations, plus exact top-20 order match."""
    lens_logits, _, _ = lens.apply(model, prompt, layers=layers, positions=positions)
    from jlens.hooks import ActivationRecorder

    input_ids = model.encode(prompt, max_length=512)
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        activations = {
            layer: recorder.activations[layer].detach() for layer in layers
        }
    worst = 0.0
    for layer in layers:
        h = activations[layer][0][list(positions)].float().cpu()
        mine = recompute_readout(model, lens, layer, h)
        theirs = lens_logits[layer].float()
        if not torch.allclose(mine, theirs, rtol=PARITY_RTOL, atol=PARITY_ATOL):
            return {"ok": False, "layer": layer,
                    "max_abs_diff": float((mine - theirs).abs().max())}
        if not torch.equal(
            mine.topk(20, dim=-1).indices, theirs.topk(20, dim=-1).indices
        ):
            return {"ok": False, "layer": layer, "why": "top-20 order mismatch"}
        worst = max(worst, float((mine - theirs).abs().max()))
    return {"ok": True, "max_abs_diff": worst,
            "layers": layers, "positions": positions}


def g_folding_audit(
    lens, model, *, token_ids: list[int], layers: list[int]
) -> dict:
    """Contract §5: cosine between unfolded and g-folded ``v_t`` over every
    battery token x band layer. min >= 0.99 registers folding immaterial."""
    gain = final_norm_gain(model)
    if gain is None:
        return {"applicable": False, "reason": "final norm has no gain vector"}
    worst = 1.0
    worst_cell = None
    for layer in layers:
        unfolded = token_vectors(lens, model, layer, token_ids, fold_gain=False)
        folded = token_vectors(lens, model, layer, token_ids, fold_gain=True)
        cosines = torch.nn.functional.cosine_similarity(unfolded, folded, dim=1)
        layer_min = float(cosines.min())
        if layer_min < worst:
            worst = layer_min
            worst_cell = {"layer": layer,
                          "token_id": int(token_ids[int(cosines.argmin())])}
    return {"applicable": True, "min_cosine": worst, "worst_cell": worst_cell,
            "immaterial": worst >= 0.99}
