"""Outcome-blind geometry for the one-shot Bank-B orthogonal rescue."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping, Sequence

import torch

from jspace_part2.lib import orthonormal_basis_from_rows


@dataclass(frozen=True)
class OrthogonalInterventionRecord:
    arm: str
    layer: int
    phase: str
    forward_index: int
    position: int
    requested_k: int
    selected_ids: tuple[int, ...]
    selected_rank: int
    effective_rank: int
    lost_rank: int
    removed_norm: float
    injection_direction_norm: float | None
    delivered_injection_norm: float
    injection_dose_relative_error: float
    injection_dose_absolute_error: float


@dataclass
class OrthogonalInterventionLog:
    hook_fires: dict[str, int] = field(default_factory=lambda: {
        "prefill": 0, "decode": 0})
    positions: list[OrthogonalInterventionRecord] = field(
        default_factory=list)


def _local_indices(
        identifiers: torch.Tensor, offsets: Mapping[int, int], *,
        device: torch.device) -> torch.Tensor:
    values = [int(value) for value in identifiers.detach().cpu().reshape(-1)]
    missing = [value for value in values if value not in offsets]
    if missing:
        raise ValueError(
            f"partial dictionary lacks token ids {sorted(set(missing))[:8]}")
    return torch.tensor(
        [int(offsets[value]) for value in values],
        device=device, dtype=torch.long).reshape(identifiers.shape)


class OrthogonalBridgeAblator:
    """Prompt-only bridge lesion with mechanically audited substitution.

    Unlike the historical full-vocabulary ablator, this implementation uses a
    hash-pinned partial dictionary.  Candidate selection is restricted to all
    true-bridge token rows and is deliberately independent of activation sign.
    Clean top-logit protection remains aligned per position.  Each optional
    unit injection receives exactly the norm removed by that position's
    span-safe bridge lesion.
    """

    def __init__(self, layers, band: Sequence[int], *,
                 dictionaries: Mapping[int, torch.Tensor],
                 offsets: Mapping[int, int]):
        self._layers = layers
        self.band = [int(value) for value in band]
        self.dictionaries = dictionaries
        self.offsets = {int(key): int(value) for key, value in offsets.items()}
        self._handles = []
        self.mode: dict | None = None
        self.phase = "prefill"
        self.forward_index = 0
        self.log = OrthogonalInterventionLog()

    def reset_log(self) -> None:
        self.log = OrthogonalInterventionLog()

    def _apply(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        mode = self.mode
        if mode is None:
            return hidden
        table = self.dictionaries[int(layer)]
        batch, length, width = hidden.shape
        if batch != 1:
            raise NotImplementedError("orthogonal bridge assay is batch-size 1")
        if table.ndim != 2 or table.shape[1] != width:
            raise ValueError("partial dictionary/hidden width mismatch")
        limit = int(mode.get("active_position_limit", length))
        if not 0 <= limit <= length:
            raise ValueError("active_position_limit is outside the sequence")

        restrict_ids = mode["restrict_ids"].to(table.device).long()
        if restrict_ids.ndim != 1 or restrict_ids.numel() == 0:
            raise ValueError("restrict_ids must be a nonempty vector")
        if str(mode.get("selection_sign_rule")) != (
                "all_rows_regardless_of_activation_sign"):
            raise ValueError("orthogonal bridge selection-sign rule drift")
        restrict_local = _local_indices(
            restrict_ids, self.offsets, device=table.device)
        candidate_rows = table.index_select(0, restrict_local).float()
        flat = hidden.reshape(length, width).float()
        scores = flat.to(table.dtype) @ candidate_rows.T.to(table.dtype)
        scores = scores.float()

        protect_ids = mode["protect_ids"].to(table.device).long()
        if protect_ids.ndim == 1:
            protect_ids = protect_ids.unsqueeze(0).expand(length, -1)
        if protect_ids.ndim != 2 or protect_ids.shape[0] != length:
            raise ValueError("protect_ids must align with sequence positions")
        protect_local = _local_indices(
            protect_ids, self.offsets, device=table.device)
        protected_rows = table[protect_local].float()
        blocked = (
            restrict_ids.reshape(1, -1, 1)
            == protect_ids.unsqueeze(1)).any(dim=2)
        scores = scores.masked_fill(blocked, float("-inf"))

        requested_k = int(mode["k"])
        take = min(requested_k, int(restrict_ids.numel()))
        top_values, top_indices = scores.topk(take, dim=1)
        valid = torch.isfinite(top_values)
        selected_rows = candidate_rows[top_indices] * valid.unsqueeze(-1)
        selected_ids = restrict_ids[top_indices]

        selected_u, selected_s, _ = torch.linalg.svd(
            selected_rows.transpose(1, 2), full_matrices=False)
        selected_threshold = (
            selected_s[:, :1] * 1e-4).clamp_min(1e-7)
        selected_rank_mask = selected_s > selected_threshold
        selected_rank = selected_rank_mask.sum(dim=1)

        protected_u, protected_s, _ = torch.linalg.svd(
            protected_rows.transpose(1, 2), full_matrices=False)
        protected_threshold = (
            protected_s[:, :1] * 1e-4).clamp_min(1e-7)
        protected_basis = protected_u * (
            protected_s > protected_threshold).unsqueeze(1)
        loadings = torch.einsum(
            "tkd,tdp->tkp", selected_rows, protected_basis)
        residual_rows = selected_rows - torch.einsum(
            "tkp,tdp->tkd", loadings, protected_basis)
        residual_u, residual_s, _ = torch.linalg.svd(
            residual_rows.transpose(1, 2), full_matrices=False)
        effective_mask = residual_s > selected_threshold
        effective_basis = residual_u * effective_mask.unsqueeze(1)
        effective_rank = effective_mask.sum(dim=1)

        coefficients = torch.einsum(
            "tdk,td->tk", effective_basis, flat)
        removed = torch.einsum(
            "tdk,tk->td", effective_basis, coefficients)
        lesioned = flat - removed
        removed_norm = removed.norm(dim=1)

        injection = mode.get("inject_dir")
        direction_norm = None
        delivered = torch.zeros_like(removed_norm)
        absolute_error = torch.zeros_like(removed_norm)
        relative_error = torch.zeros_like(removed_norm)
        if injection is not None:
            direction = injection[int(layer)] if isinstance(
                injection, Mapping) else injection
            direction = direction.to(flat.device).float()
            if direction.ndim != 1 or direction.numel() != width \
                    or not torch.isfinite(direction).all():
                raise ValueError("invalid bridge injection direction")
            direction_norm = float(direction.norm())
            norm_error = abs(direction_norm - 1.0)
            if norm_error > float(mode[
                    "maximum_injection_direction_norm_error"]):
                raise RuntimeError(
                    f"injection direction norm drift at L{layer}: "
                    f"{direction_norm}")
            direction = direction / max(direction_norm, 1e-30)
            addition = removed_norm.unsqueeze(1) * direction.unsqueeze(0)
            lesioned = lesioned + addition
            delivered = addition.norm(dim=1)
            absolute_error = (delivered - removed_norm).abs()
            relative_error = absolute_error / removed_norm.clamp_min(1e-30)
            bad = (
                (absolute_error > float(mode[
                    "maximum_injection_dose_absolute_error"]))
                & (relative_error > float(mode[
                    "maximum_injection_dose_relative_error"])))
            if bool(bad[:limit].any()):
                raise RuntimeError(
                    f"injection dose mismatch at L{layer}: "
                    f"max abs={float(absolute_error[:limit].max())}, "
                    f"max rel={float(relative_error[:limit].max())}")

        active = torch.arange(length, device=flat.device) < limit
        result = torch.where(active.unsqueeze(1), lesioned, flat)
        self.log.hook_fires[self.phase] += 1
        arm = str(mode["arm"])
        for position in range(limit):
            ids = tuple(int(value) for value in selected_ids[
                position][valid[position]].detach().cpu())
            self.log.positions.append(OrthogonalInterventionRecord(
                arm=arm, layer=int(layer), phase=self.phase,
                forward_index=int(self.forward_index), position=position,
                requested_k=requested_k, selected_ids=ids,
                selected_rank=int(selected_rank[position]),
                effective_rank=int(effective_rank[position]),
                lost_rank=int(selected_rank[position]
                              - effective_rank[position]),
                removed_norm=float(removed_norm[position]),
                injection_direction_norm=direction_norm,
                delivered_injection_norm=float(delivered[position]),
                injection_dose_relative_error=float(relative_error[position]),
                injection_dose_absolute_error=float(absolute_error[position]),
            ))
        return result.reshape(batch, length, width).to(hidden.dtype)

    def _hook(self, layer: int):
        def apply(_module, _inputs, output):
            if self.mode is None or self.phase not in self.mode.get(
                    "active_phases", {"prefill"}):
                return output
            hidden = output if torch.is_tensor(output) else output[0]
            changed = self._apply(hidden, int(layer))
            return changed if torch.is_tensor(output) else (
                changed, *output[1:])
        return apply

    def __enter__(self):
        for layer in self.band:
            self._handles.append(self._layers[layer].register_forward_hook(
                self._hook(layer)))
        return self

    def __exit__(self, *_error):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@torch.no_grad()
def partial_j_dictionary_rows(
        hf_model, gain: torch.Tensor, jacobian: torch.Tensor,
        token_ids: Sequence[int], *, device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16, chunk: int = 1024) -> torch.Tensor:
    """Build normalized J-dictionary rows for an explicit token-ID set."""
    ordered = [int(value) for value in token_ids]
    if len(ordered) != len(set(ordered)):
        raise ValueError("partial dictionary token IDs must be unique")
    operator = jacobian.to(device=device, dtype=torch.float32)
    width = int(operator.shape[0])
    if operator.ndim != 2 or operator.shape[1] != width:
        raise ValueError("Jacobian operator must be square")
    if not ordered:
        return torch.empty((0, width), device=device, dtype=dtype)
    weight = hf_model.get_output_embeddings().weight.detach()
    ids = torch.tensor(ordered, device=weight.device, dtype=torch.long)
    if int(ids.min()) < 0 or int(ids.max()) >= int(weight.shape[0]):
        raise ValueError("partial dictionary token ID outside vocabulary")
    output = torch.empty(
        (len(ordered), width), device=device, dtype=dtype)
    effective = gain.to(device=device, dtype=torch.float32)
    for start in range(0, len(ordered), int(chunk)):
        stop = min(start + int(chunk), len(ordered))
        rows = weight.index_select(0, ids[start:stop]).to(
            device=device, dtype=torch.float32)
        rows = (rows * effective.unsqueeze(0)) @ operator
        norms = rows.norm(dim=1)
        if bool((~torch.isfinite(norms) | (norms <= 1e-12)).any()):
            raise RuntimeError("partial J dictionary produced a null row")
        output[start:stop] = torch.nn.functional.normalize(
            rows, dim=1).to(dtype)
    return output


@dataclass(frozen=True)
class OrthogonalBridgeGeometry:
    piece_count: int
    answer_piece_count: int
    answer_effective_rank: int
    raw_mean_norm: float
    retained_norm: float
    retained_fraction: float
    raw_answer_span_cosine: float
    maximum_answer_span_cosine: float
    self_readout_cosine_mean: float
    self_readout_cosine_min: float
    self_readout_cosine_max: float


def _matrix(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError(f"{label} must be a nonempty row matrix")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains a nonfinite value")
    return value.float()


def orthogonal_bridge_direction(
        bridge_rows: torch.Tensor, answer_rows: torch.Tensor,
        *, relative_tolerance: float = 1e-5,
        absolute_tolerance: float = 1e-7,
) -> tuple[torch.Tensor, OrthogonalBridgeGeometry]:
    """Remove the complete answer-row span from a mean bridge direction."""
    bridge = _matrix(bridge_rows, label="bridge rows")
    answer = _matrix(answer_rows, label="answer rows")
    if bridge.shape[1] != answer.shape[1]:
        raise ValueError("bridge and answer rows have different dimensions")
    raw = bridge.mean(dim=0)
    raw_norm = float(raw.norm())
    if raw_norm <= absolute_tolerance:
        raise RuntimeError("mean bridge direction is numerically null")
    answer_basis = orthonormal_basis_from_rows(
        answer, relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance)
    basis = answer_basis.basis.to(raw.device)
    residual = raw - basis @ (basis.T @ raw) if basis.shape[1] else raw
    retained_norm = float(residual.norm())
    direction = (
        residual / retained_norm
        if retained_norm > absolute_tolerance else torch.zeros_like(raw))
    raw_unit = raw / raw_norm
    raw_overlap = float((basis.T @ raw_unit).norm()) \
        if basis.shape[1] else 0.0
    maximum_overlap = float((basis.T @ direction).abs().max()) \
        if basis.shape[1] and retained_norm > absolute_tolerance else 0.0
    normalized_rows = torch.nn.functional.normalize(bridge, dim=1)
    readout = normalized_rows @ direction
    geometry = OrthogonalBridgeGeometry(
        piece_count=int(bridge.shape[0]),
        answer_piece_count=int(answer.shape[0]),
        answer_effective_rank=int(answer_basis.effective_rank),
        raw_mean_norm=raw_norm,
        retained_norm=retained_norm,
        retained_fraction=retained_norm / raw_norm,
        raw_answer_span_cosine=raw_overlap,
        maximum_answer_span_cosine=maximum_overlap,
        self_readout_cosine_mean=float(readout.mean()),
        self_readout_cosine_min=float(readout.min()),
        self_readout_cosine_max=float(readout.max()),
    )
    return direction, geometry


def geometry_gate(
        geometry: OrthogonalBridgeGeometry,
        thresholds: Mapping[str, float]) -> dict:
    checks = {
        "retained_fraction": geometry.retained_fraction >= float(
            thresholds["minimum_retained_fraction"]),
        "answer_span_orthogonal": (
            geometry.maximum_answer_span_cosine
            <= float(thresholds["maximum_answer_span_cosine"])),
        "semantic_self_readout": (
            geometry.self_readout_cosine_mean
            >= float(thresholds["minimum_self_readout_cosine_mean"])),
        "finite": all(math.isfinite(float(value)) for value in (
            geometry.raw_mean_norm, geometry.retained_norm,
            geometry.retained_fraction,
            geometry.maximum_answer_span_cosine,
            geometry.self_readout_cosine_mean)),
    }
    return {
        "geometry": asdict(geometry),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def profile_match_distance(
        target: Sequence[OrthogonalBridgeGeometry],
        candidate: Sequence[OrthogonalBridgeGeometry]) -> dict:
    if not target or len(target) != len(candidate):
        raise ValueError("geometry profiles must be nonempty and equal-length")

    def rmse(field: str, *, relative: bool = False) -> float:
        differences = []
        for left, right in zip(target, candidate, strict=True):
            difference = float(getattr(right, field)) - float(
                getattr(left, field))
            if relative:
                difference /= max(abs(float(getattr(left, field))), 1e-8)
            differences.append(difference)
        return math.sqrt(sum(value * value for value in differences)
                         / len(differences))

    piece_difference = max(
        abs(int(left.piece_count) - int(right.piece_count))
        for left, right in zip(target, candidate, strict=True))
    return {
        "maximum_piece_count_difference": piece_difference,
        "retained_fraction_rmse": rmse("retained_fraction"),
        "raw_mean_norm_relative_rmse": rmse(
            "raw_mean_norm", relative=True),
        "raw_answer_span_cosine_rmse": rmse(
            "raw_answer_span_cosine"),
        "self_readout_cosine_mean_rmse": rmse(
            "self_readout_cosine_mean"),
    }


def select_unrelated_geometry_match(
        *, target_profile: Sequence[OrthogonalBridgeGeometry],
        target_fact_id: str, target_family: str,
        candidates: Sequence[Mapping]) -> tuple[Mapping, dict]:
    """Choose a different-family bridge by a frozen lexicographic ruler."""
    scored = []
    for candidate in candidates:
        if str(candidate["fact_id"]) == str(target_fact_id) \
                or str(candidate["canonical_family"]) == str(target_family):
            continue
        distance = profile_match_distance(
            target_profile, candidate["profile"])
        key = (
            int(distance["maximum_piece_count_difference"]),
            float(distance["retained_fraction_rmse"]),
            float(distance["raw_mean_norm_relative_rmse"]),
            float(distance["raw_answer_span_cosine_rmse"]),
            float(distance["self_readout_cosine_mean_rmse"]),
            str(candidate["fact_id"]),
        )
        scored.append((key, candidate, distance))
    if not scored:
        raise RuntimeError(f"no unrelated geometry match for {target_fact_id}")
    key, selected, distance = min(scored, key=lambda row: row[0])
    return selected, {
        "target_fact_id": str(target_fact_id),
        "selected_fact_id": str(selected["fact_id"]),
        "selected_family": str(selected["canonical_family"]),
        "selection_key": [
            int(key[0]), *[float(value) for value in key[1:-1]], str(key[-1])],
        **distance,
        "selection_used_outcomes": False,
    }


def geometry_match_gate(report: Mapping, thresholds: Mapping) -> dict:
    checks = {
        "piece_count": int(report["maximum_piece_count_difference"])
        <= int(thresholds["maximum_piece_count_difference"]),
        "retained_fraction": float(report["retained_fraction_rmse"])
        <= float(thresholds["maximum_retained_fraction_rmse"]),
        "raw_norm": float(report["raw_mean_norm_relative_rmse"])
        <= float(thresholds["maximum_raw_mean_norm_relative_rmse"]),
        "answer_overlap": float(report["raw_answer_span_cosine_rmse"])
        <= float(thresholds["maximum_raw_answer_span_cosine_rmse"]),
        "self_readout": float(report["self_readout_cosine_mean_rmse"])
        <= float(thresholds["maximum_self_readout_cosine_mean_rmse"]),
        "outcome_blind": report.get("selection_used_outcomes") is False,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def stable_random_answer_orthogonal_direction(
        answer_rows: torch.Tensor, bridge_direction: torch.Tensor,
        *, seed: int, minimum_norm: float = 1e-6) -> tuple[torch.Tensor, dict]:
    """Stable random unit vector orthogonal to answer and bridge directions."""
    answer = _matrix(answer_rows, label="answer rows")
    bridge = bridge_direction.float()
    if bridge.ndim != 1 or bridge.shape[0] != answer.shape[1]:
        raise ValueError("bridge direction shape drift")
    rows = torch.cat([answer, bridge.unsqueeze(0)], dim=0)
    basis = orthonormal_basis_from_rows(rows).basis.to(answer.device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    vector = torch.randn(
        answer.shape[1], generator=generator, dtype=torch.float32).to(
            answer.device)
    vector = vector - basis @ (basis.T @ vector)
    norm = float(vector.norm())
    if norm < minimum_norm:
        raise RuntimeError("random answer-orthogonal direction collapsed")
    direction = vector / norm
    maximum_overlap = float((basis.T @ direction).abs().max()) \
        if basis.shape[1] else 0.0
    return direction, {
        "seed": int(seed),
        "anchor_effective_rank": int(basis.shape[1]),
        "pre_normalization_norm": norm,
        "maximum_anchor_cosine": maximum_overlap,
    }
