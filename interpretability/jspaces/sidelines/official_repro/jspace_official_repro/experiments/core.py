"""Shared executor for coordinate-swap trials (Qwen and OLMo cores)."""
from __future__ import annotations

import torch

from ..interventions import HookPlan, InterventionSession
from ..readout import token_vectors
from ..rendering import Rendered
from ..scoring import forward_logits, rank_of


@torch.no_grad()
def baseline_answer(model, rendered: Rendered, *, position: int | None = None):
    """Greedy next token (+ full logits) at the scored position."""
    position = rendered.final_position if position is None else position
    logits, _ = forward_logits(model, rendered.input_ids, positions=[position])
    row = logits[0]
    token_id = int(row.argmax())
    return {
        "position": position,
        "greedy_token_id": token_id,
        "greedy_token": model.tokenizer.decode([token_id]),
        "logits": row,
    }


@torch.no_grad()
def swap_trial(
    model,
    lens,
    rendered: Rendered,
    *,
    band: list[int],
    source_token_id: int,
    target_token_id: int,
    score_token_ids: list[int],
    alpha: float = 1.0,
    scored_position: int | None = None,
    positions: list[int] | None = None,
    collect_diagnostics: bool = False,
) -> dict:
    """One coordinate-swap forward: swap source->target across ``band`` at
    ``positions`` (default: every prompt position), score
    ``score_token_ids`` at the scored position."""
    scored_position = (
        rendered.final_position if scored_position is None else scored_position
    )
    positions = positions if positions is not None else list(range(rendered.seq_len))
    vectors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in band:
        pair = token_vectors(lens, model, layer,
                             [source_token_id, target_token_id])
        vectors[layer] = (pair[0], pair[1])
    plan = HookPlan(layers=band, positions=positions)
    with InterventionSession(
        model.layers, plan, kind="swap", vectors=vectors, alpha=alpha,
        collect_diagnostics=collect_diagnostics,
    ) as session:
        logits, _ = forward_logits(model, rendered.input_ids,
                                   positions=[scored_position])
        fully_gated = len(session.gated_positions) == len(band) * len(positions)
        session.assert_fires(1)
    row = logits[0]
    ranks = {t: rank_of(row, t) for t in score_token_ids}
    best = min(ranks.values()) if ranks else None
    top1_id = int(row.argmax())
    result = {
        "alpha": alpha,
        "scored_position": scored_position,
        "n_band_layers": len(band),
        "n_positions": len(positions),
        "geometry_gated_cells": len(session.gated_positions),
        "geometry_fully_gated": fully_gated,
        "score_ranks": {str(t): r for t, r in ranks.items()},
        "best_rank": best,
        "top1_token_id": top1_id,
        "top1_token": model.tokenizer.decode([top1_id]),
        "top1_success": best == 1 if best is not None else None,
    }
    if collect_diagnostics and session.diagnostics:
        sample_key = next(iter(session.diagnostics))
        diag = session.diagnostics[sample_key]
        result["sample_diagnostics"] = {
            "layer_position": list(sample_key),
            "cosine": diag.cosine,
            "condition_number": diag.condition_number,
            "patch_norm": diag.patch_norm,
            "residual_norm": diag.residual_norm,
            "coord_reconstruction_error": diag.coord_reconstruction_error,
            "orth_complement_error": diag.orth_complement_error,
        }
    return result


def clean_rank_of(logits: torch.Tensor, token_id: int) -> int:
    return rank_of(logits, token_id)
