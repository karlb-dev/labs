"""Readout helpers: parity, g-fold, token vectors, depth profiles."""
from __future__ import annotations

import torch

PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-4


def unembedding_rows(model, token_ids: list[int]) -> torch.Tensor:
    """Rows of the effective unembedding (includes Muse output_multiplier)."""
    weight = model._lm_head.weight.detach()
    U = weight[token_ids].float().cpu()
    mult = float(getattr(model, "_output_multiplier", 1.0) or 1.0)
    if abs(mult - 1.0) > 1e-12:
        U = U * mult
    return U


def final_norm_gain(model) -> torch.Tensor | None:
    norm = model._final_norm
    weight = getattr(norm, "weight", None)
    return None if weight is None else weight.detach().float().cpu()


def token_vectors(lens, model, layer: int, token_ids: list[int], *, fold_gain: bool = False):
    U = unembedding_rows(model, token_ids)
    if fold_gain:
        gain = final_norm_gain(model)
        if gain is not None:
            U = U * gain
    J = lens.jacobians[layer].float()
    return (U.to(J.device) @ J).cpu()


def lens_to_device(lens, device, *, layers: list[int] | None = None) -> None:
    for layer in layers if layers is not None else list(lens.jacobians):
        lens.jacobians[layer] = lens.jacobians[layer].to(device)


def recompute_readout(model, lens, layer: int, residual: torch.Tensor) -> torch.Tensor:
    J_device = lens.jacobians[layer].device
    transported = lens.transport(residual.float().to(J_device), layer)
    return model.unembed(transported).float().cpu()


def readout_parity(model, lens, prompt: str, *, layers: list[int], positions: list[int]) -> dict:
    lens_logits, _, _ = lens.apply(model, prompt, layers=layers, positions=positions)
    from jlens.hooks import ActivationRecorder

    input_ids = model.encode(prompt, max_length=512)
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        activations = {layer: recorder.activations[layer].detach() for layer in layers}
    worst = 0.0
    for layer in layers:
        h = activations[layer][0][list(positions)].float().cpu()
        mine = recompute_readout(model, lens, layer, h)
        theirs = lens_logits[layer].float()
        if not torch.allclose(mine, theirs, rtol=PARITY_RTOL, atol=PARITY_ATOL):
            return {
                "ok": False,
                "layer": layer,
                "max_abs_diff": float((mine - theirs).abs().max()),
            }
        if not torch.equal(
            mine.topk(20, dim=-1).indices, theirs.topk(20, dim=-1).indices
        ):
            return {"ok": False, "layer": layer, "why": "top-20 order mismatch"}
        worst = max(worst, float((mine - theirs).abs().max()))
    return {"ok": True, "max_abs_diff": worst, "layers": layers, "positions": positions}


def g_folding_audit(lens, model, *, token_ids: list[int], layers: list[int]) -> dict:
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
            worst_cell = {
                "layer": layer,
                "token_id": int(token_ids[int(cosines.argmin())]),
            }
    return {
        "applicable": True,
        "min_cosine": worst,
        "worst_cell": worst_cell,
        "immaterial": worst >= 0.99,
    }


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    """1-indexed rank of token_id in a 1d logit vector."""
    return int((logits > logits[token_id]).sum().item()) + 1


def preferred_token(tokenizer, word: str) -> int | None:
    """Prefer a single-token encoding of ` word` then `word`."""
    for form in (f" {word}", word):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


def identity_lens(d_model: int, layers: list[int]):
    """Build a JacobianLens with identity matrices (pre-fit geometry probe)."""
    from jlens.lens import JacobianLens

    Js = {L: torch.eye(d_model, dtype=torch.float32) for L in layers}
    return JacobianLens(Js, n_prompts=0, d_model=d_model)
