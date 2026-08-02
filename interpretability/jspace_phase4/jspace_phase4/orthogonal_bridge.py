"""Outcome-blind geometry for the one-shot Bank-B orthogonal rescue."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import torch

from jspace_part2.lib import orthonormal_basis_from_rows


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
