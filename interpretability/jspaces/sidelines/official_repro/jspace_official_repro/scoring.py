"""Forward execution, rank scoring, and raw-record writing (plan §12)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from jlens.hooks import ActivationRecorder


@torch.no_grad()
def forward_logits(
    model,
    input_ids: torch.Tensor,
    *,
    positions: list[int],
    record_layers: list[int] | None = None,
    session=None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """One forward pass; model logits at ``positions`` (+ optional recorded
    residuals). ``session`` is an entered InterventionSession (hooks fire
    during this forward). Returns (model_logits [n_pos, vocab], residuals
    {layer: [n_pos, d]})."""
    final_layer = model.n_layers - 1
    record_at = sorted(set(record_layers or []) | {final_layer})
    with ActivationRecorder(model.layers, at=record_at) as recorder:
        model.forward(input_ids)
        activations = {
            layer: recorder.activations[layer].detach() for layer in record_at
        }
    final = activations[final_layer][0][list(positions)].float()
    logits = model.unembed(final).float().cpu()
    residuals = {
        layer: activations[layer][0][list(positions)].float().cpu()
        for layer in (record_layers or [])
    }
    return logits, residuals


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    """1-indexed competition rank of ``token_id`` in a [vocab] logit row."""
    value = logits[token_id]
    return int((logits > value).sum().item()) + 1


def min_rank_over(logits: torch.Tensor, token_ids: list[int]) -> int:
    return min(rank_of(logits, t) for t in token_ids)


def lens_ranks(
    lens_logits: dict[int, torch.Tensor], token_ids: list[int], *, position_index: int = 0
) -> dict[int, int]:
    """Per-layer min-over-synonym rank at one readout position."""
    return {
        layer: min_rank_over(logits[position_index], token_ids)
        for layer, logits in lens_logits.items()
    }


def pass_at_k(per_item_ranks: list[list[int]], k: int) -> float:
    """Mean over items of the fraction of intermediates with rank <= k."""
    fractions = [
        sum(1 for rank in item if rank <= k) / len(item)
        for item in per_item_ranks
        if item
    ]
    return sum(fractions) / len(fractions) if fractions else float("nan")


class RawRecordWriter:
    """Append-only JSONL raw rows (plan §12 schema); one file per cell."""

    def __init__(self, path: str | Path, *, common: dict) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"immutable raw record exists: {self.path}")
        self.common = common
        self._handle = self.path.open("w")
        self.n_rows = 0

    def write(self, row: dict) -> None:
        stamped = {**self.common, **row,
                   "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self._handle.write(json.dumps(stamped, ensure_ascii=False) + "\n")
        self.n_rows += 1

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
