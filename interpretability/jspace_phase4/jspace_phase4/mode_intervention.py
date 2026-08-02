"""Phase-owned span-safe-J and exact-profile matched interventions.

The matched arm computes the span-safe-J rank and removed-energy profile on
the *same hidden state* and applies a stable random subspace with that exact
profile.  It therefore needs no forked or shared mutable KV cache.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import torch

from jspace_part2.lib import orthonormal_basis_from_rows
from jspace_phase3.controls import build_instant_matched_subspace

from .phase_hooks import DelimiterSpec, Phase, classify_token_phases
from .seeds import stable_seed


@dataclass
class ModeInterventionRecord:
    layer: int
    phase: str
    forward_index: int
    position: int
    arm: str
    selected_ids: list[int]
    protected_ids: list[int]
    selected_protected_overlap: int
    requested_rank: int
    delivered_rank: int
    target_energy_frac: float
    delivered_energy_frac: float
    energy_relative_error: float
    maximum_protected_cosine: float
    protected_effective_rank: int
    control_clamped: bool


@dataclass
class ModeInterventionLog:
    records: list[ModeInterventionRecord] = field(default_factory=list)
    hook_fires: dict[str, int] = field(default_factory=lambda: {
        phase.value: 0 for phase in Phase})
    wrong_phase_hook_fires: int = 0

    def rows(self) -> list[dict]:
        return [asdict(record) for record in self.records]

    def summary(self) -> dict:
        if not self.records:
            return {
                "n_positions": 0,
                "hook_fires": dict(self.hook_fires),
                "wrong_phase_hook_fires": self.wrong_phase_hook_fires,
            }
        return {
            "n_positions": len(self.records),
            "hook_fires": dict(self.hook_fires),
            "wrong_phase_hook_fires": self.wrong_phase_hook_fires,
            "requested_rank_total": sum(
                row.requested_rank for row in self.records),
            "delivered_rank_total": sum(
                row.delivered_rank for row in self.records),
            "rank_match_exact": all(
                row.requested_rank == row.delivered_rank
                for row in self.records),
            "maximum_energy_relative_error": max(
                row.energy_relative_error for row in self.records),
            "maximum_selected_protected_overlap": max(
                row.selected_protected_overlap for row in self.records),
            "maximum_protected_cosine": max(
                row.maximum_protected_cosine for row in self.records),
            "control_clamped_positions": sum(
                row.control_clamped for row in self.records),
        }


def accepted_alias_token_ids(tokenizer, aliases: Sequence[str]) -> list[int]:
    """Return the sorted union of every complete accepted-alias sequence."""
    values = set()
    for alias in aliases:
        encoded = tokenizer(alias, add_special_tokens=False)
        token_ids = encoded["input_ids"] if isinstance(
            encoded, Mapping) else encoded.input_ids
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise RuntimeError("accepted alias tokenization is batched")
            token_ids = token_ids[0]
        if not token_ids:
            raise RuntimeError(f"accepted alias tokenizes empty: {alias!r}")
        values.update(int(value) for value in token_ids)
    return sorted(values)


def combined_protection_sets(
        clean_logits: torch.Tensor, *, alias_token_ids: Sequence[int],
        top_k: int) -> torch.Tensor:
    """Clean per-position top-k plus every accepted-alias token piece."""
    if clean_logits.ndim != 2:
        raise ValueError("clean logits must be [positions, vocabulary]")
    if top_k <= 0 or top_k > clean_logits.shape[1]:
        raise ValueError("invalid clean protection top-k")
    top = clean_logits.topk(top_k, dim=-1).indices
    aliases = torch.tensor(
        sorted(set(int(value) for value in alias_token_ids)),
        device=top.device, dtype=torch.long)
    if aliases.numel() == 0:
        raise ValueError("accepted-alias protection is empty")
    if int(aliases.min()) < 0 or int(aliases.max()) >= clean_logits.shape[1]:
        raise ValueError("accepted-alias token exceeds model vocabulary")
    return torch.cat([
        top, aliases.unsqueeze(0).expand(top.shape[0], -1)], dim=1)


def prediction_phase(
        token_ids: Sequence[int], *, prompt_length: int,
        delimiters: DelimiterSpec) -> str:
    """Phase whose next token is about to be predicted after ``token_ids``."""
    parsed = classify_token_phases(
        token_ids, prompt_length=prompt_length, delimiters=delimiters)
    if parsed.start_index is not None and parsed.end_index is None:
        return Phase.REASONING.value
    return Phase.FINAL_ANSWER.value


def answer_prediction_mask(
        *, sequence_length: int, context_length: int,
        device: torch.device | str | None = None) -> torch.Tensor:
    """Positions whose hidden state predicts a supplied answer token.

    If an answer has ``a`` pieces appended to a generated context of length
    ``c``, the predictors are positions ``c-1`` through ``c+a-2``.
    """
    if not 1 <= context_length < sequence_length:
        raise ValueError("answer context must precede a nonempty answer")
    result = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    result[context_length - 1:sequence_length - 1] = True
    return result


class ExactProfileModeAblator:
    """Combined span-safe-J / exact-profile control forward hooks."""

    ALLOWED_ARMS = {"span_safe_j", "matched_control"}

    def __init__(self, layers, band: Sequence[int]):
        self.layers = layers
        self.band = [int(value) for value in band]
        self.handles = []
        self.mode: dict | None = None
        self.log = ModeInterventionLog()

    def reset(self) -> None:
        self.mode = None
        self.log = ModeInterventionLog()

    def configure(
            self, *, arm: str, dictionaries: Mapping[int, torch.Tensor],
            protection_sets: torch.Tensor, active_position_mask: torch.Tensor,
            target_phase: str, current_phase: str, forward_index: int,
            k: int, evidence_id: str, item_id: str, condition: str,
            base_seed: int, energy_relative_floor: float) -> None:
        if arm not in self.ALLOWED_ARMS:
            raise ValueError(f"unsupported mode intervention arm: {arm}")
        target = Phase(target_phase).value
        current = Phase(current_phase).value
        if current != target:
            self.log.wrong_phase_hook_fires += 1
            raise RuntimeError(
                f"intervention configured in wrong phase: target={target}, "
                f"current={current}")
        if active_position_mask.ndim != 1:
            raise ValueError("active position mask must be one-dimensional")
        self.mode = {
            "arm": arm,
            "dicts": dictionaries,
            "protect_sets": protection_sets,
            "active_position_mask": active_position_mask,
            "target_phase": target,
            "current_phase": current,
            "forward_index": int(forward_index),
            "k": int(k),
            "evidence_id": evidence_id,
            "item_id": item_id,
            "condition": condition,
            "base_seed": int(base_seed),
            "energy_relative_floor": float(energy_relative_floor),
        }

    def _hook(self, layer: int):
        def apply(_module, _inputs, output):
            if self.mode is None:
                return output
            hidden = output[0] if not torch.is_tensor(output) else output
            changed = self._apply(hidden, layer)
            return changed if torch.is_tensor(output) else (
                changed, *output[1:])
        return apply

    def __enter__(self):
        for layer in self.band:
            self.handles.append(
                self.layers[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _apply(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        mode = self.mode
        if mode is None:  # pragma: no cover - hook already guards this
            return hidden
        phase = str(mode["current_phase"])
        if phase != mode["target_phase"]:
            self.log.wrong_phase_hook_fires += 1
            raise RuntimeError("mode intervention hook crossed phase boundary")
        dictionary = mode["dicts"][layer]
        batch, positions, dimension = hidden.shape
        if batch != 1:
            raise NotImplementedError("mode intervention requires batch size 1")
        active = mode["active_position_mask"].to(
            hidden.device, dtype=torch.bool)
        protection = mode["protect_sets"].to(hidden.device, dtype=torch.long)
        if active.shape != (positions,):
            raise ValueError("active-position mask length drift")
        if protection.ndim == 1:
            protection = protection.unsqueeze(0).expand(positions, -1)
        if protection.shape[0] != positions:
            raise ValueError("per-position protection length drift")
        if int(protection.min()) < 0 or int(protection.max()) >= dictionary.shape[0]:
            raise ValueError("protected token exceeds dictionary rows")
        flat = hidden[0].float()
        scores = (flat.to(dictionary.dtype) @ dictionary.T).float()
        scores = torch.where(
            scores > 0, scores,
            torch.full_like(scores, float("-inf")))
        scores.scatter_(1, protection, float("-inf"))
        take = min(int(mode["k"]), dictionary.shape[0])
        top_scores, top_ids = scores.topk(take, dim=1)
        valid = torch.isfinite(top_scores)
        selected = dictionary[top_ids].float() * valid.unsqueeze(-1)

        protected_rows = dictionary[protection].float()
        protected_u, protected_s, _ = torch.linalg.svd(
            protected_rows.transpose(1, 2), full_matrices=False)
        protected_threshold = (
            protected_s[:, :1] * 1e-4).clamp_min(1e-7)
        protected_mask = protected_s > protected_threshold
        protected_basis = protected_u * protected_mask.unsqueeze(1)

        label_u, label_s, _ = torch.linalg.svd(
            selected.transpose(1, 2), full_matrices=False)
        label_threshold = (label_s[:, :1] * 1e-4).clamp_min(1e-7)
        selected_residual = selected - torch.einsum(
            "tkd,tdp->tkp", selected, protected_basis) @ \
            protected_basis.transpose(1, 2)
        span_u, span_s, _ = torch.linalg.svd(
            selected_residual.transpose(1, 2), full_matrices=False)
        span_mask = span_s > label_threshold
        span_basis = span_u * span_mask.unsqueeze(1)
        coefficients = torch.einsum("tdk,td->tk", span_basis, flat)
        j_removed = torch.einsum("tdk,tk->td", span_basis, coefficients)
        before = (flat * flat).sum(dim=1).clamp_min(1e-30)
        target_energy = (j_removed * j_removed).sum(dim=1) / before
        target_rank = span_mask.sum(dim=1)
        changed = flat.clone()

        for position in range(positions):
            if not bool(active[position]):
                continue
            rank = int(target_rank[position])
            target = float(target_energy[position])
            selected_ids = [
                int(value) for value in top_ids[position][
                    valid[position]].detach().cpu().tolist()]
            protected_ids = [
                int(value) for value in protection[position]
                .detach().cpu().tolist()]
            overlap_count = len(set(selected_ids) & set(protected_ids))
            protected_rank = int(protected_mask[position].sum())
            control_clamped = False
            if mode["arm"] == "span_safe_j":
                basis = span_basis[position, :, :rank]
            elif rank:
                seed = stable_seed(
                    experiment_id=str(mode["evidence_id"]),
                    item_id=str(mode["item_id"]),
                    condition=(
                        f"{mode['condition']}|forward={mode['forward_index']}"),
                    layer=layer, position=position,
                    base_seed=int(mode["base_seed"]),
                )
                basis, information = build_instant_matched_subspace(
                    flat[position], rank, target,
                    protected_rows[position], seed)
                control_clamped = bool(information["clamped"])
            else:
                basis = flat.new_zeros((dimension, 0))
            removed = (
                basis @ (basis.T @ flat[position])
                if basis.shape[1] else flat.new_zeros(dimension))
            changed[position] = flat[position] - removed
            delivered = float(removed @ removed) / float(before[position])
            floor = float(mode["energy_relative_floor"])
            relative = abs(delivered - target) / max(target, floor)
            if basis.shape[1] and protected_rank:
                maximum_cosine = float((
                    basis.T @ protected_basis[
                        position, :, :protected_rank]).abs().max())
            else:
                maximum_cosine = 0.0
            self.log.records.append(ModeInterventionRecord(
                layer=layer,
                phase=phase,
                forward_index=int(mode["forward_index"]),
                position=position,
                arm=str(mode["arm"]),
                selected_ids=selected_ids,
                protected_ids=protected_ids,
                selected_protected_overlap=overlap_count,
                requested_rank=rank,
                delivered_rank=int(basis.shape[1]),
                target_energy_frac=target,
                delivered_energy_frac=delivered,
                energy_relative_error=relative,
                maximum_protected_cosine=maximum_cosine,
                protected_effective_rank=protected_rank,
                control_clamped=control_clamped,
            ))
        self.log.hook_fires[phase] += 1
        changed = torch.where(active.unsqueeze(1), changed, flat)
        return changed.unsqueeze(0).to(hidden.dtype)
