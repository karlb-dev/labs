"""Validated intervention mechanics and bridge endpoint contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch

from jspace_phase3.controls import build_instant_matched_subspace
from jspace_part2.lib import orthonormal_basis_from_rows


@dataclass(frozen=True)
class AddedProtectionProfile:
    piece_count: int
    added_rank: int
    output_span_overlap: float
    selected_span_overlap: float
    bridge_answer_cosine: float
    activation_score_mean: float
    activation_score_max: float


@dataclass(frozen=True)
class GeometryTolerance:
    piece_count: int = 0
    added_rank: int = 0
    output_span_overlap: float = 0.05
    selected_span_overlap: float = 0.05
    bridge_answer_cosine: float = 0.10
    activation_score_mean: float = 0.20
    activation_score_max: float = 0.20


def geometry_match_report(
        target: AddedProtectionProfile,
        candidate: AddedProtectionProfile,
        tolerance: GeometryTolerance) -> dict:
    differences = {
        field: abs(float(getattr(candidate, field))
                   - float(getattr(target, field)))
        for field in asdict(target)
    }
    limits = asdict(tolerance)
    passed = {
        field: difference <= float(limits[field]) + 1e-12
        for field, difference in differences.items()
    }
    return {
        "target": asdict(target),
        "candidate": asdict(candidate),
        "absolute_difference": differences,
        "tolerance": limits,
        "passed_by_field": passed,
        "exact_piece_count": differences["piece_count"] == 0,
        "exact_added_rank": differences["added_rank"] == 0,
        "ok": all(passed.values()),
    }


def exact_rank_energy_matched_subspace(
        hidden: torch.Tensor, *, rank: int, energy_fraction: float,
        protected_rows: torch.Tensor | None, seed: int,
        rank_tolerance: float = 1e-5,
        energy_tolerance: float = 2e-5) -> tuple[torch.Tensor, dict]:
    """Wrap the corrected Phase 3 constructor and verify its guarantees."""
    basis, construction = build_instant_matched_subspace(
        hidden, rank, energy_fraction, protected_rows, seed)
    gram = basis.float().T @ basis.float()
    orthogonality_error = float((
        gram - torch.eye(rank, device=gram.device)).abs().max())
    effective_rank = int(torch.linalg.matrix_rank(
        basis.float(), tol=rank_tolerance).item())
    hidden32 = hidden.float()
    removed = basis.float() @ (basis.float().T @ hidden32)
    achieved_energy = float(removed.square().sum()
                            / hidden32.square().sum().clamp_min(1e-30))
    protected_overlap = 0.0
    if protected_rows is not None and protected_rows.numel():
        protected_basis = orthonormal_basis_from_rows(
            protected_rows.float()).basis.to(basis.device)
        protected_overlap = float(
            (protected_basis.T @ basis.float()).square().sum())
    target = min(float(energy_fraction), float(construction["e_max"])
                 * 0.999999)
    ok = (
        effective_rank == rank
        and orthogonality_error <= rank_tolerance
        and abs(achieved_energy - target) <= energy_tolerance
        and protected_overlap <= 5e-4
    )
    report = {
        **construction,
        "requested_rank": int(rank),
        "effective_rank": effective_rank,
        "orthogonality_error": orthogonality_error,
        "energy_achieved": achieved_energy,
        "energy_expected_after_clamp": target,
        "protected_projector_overlap": protected_overlap,
        "mechanical_gate_passed": ok,
    }
    if not ok:
        raise RuntimeError(f"matched-control mechanical gate failed: {report}")
    return basis, report


def answer_preference(*, original_lp: float,
                      counterfactual_lp: float) -> float:
    return float(counterfactual_lp - original_lp)


def substitution_endpoint(
        counterfactual_arm: Mapping[str, float],
        unrelated_arm: Mapping[str, float]) -> dict:
    required = {"original_lp", "counterfactual_lp"}
    for name, arm in (
            ("counterfactual_arm", counterfactual_arm),
            ("unrelated_arm", unrelated_arm)):
        missing = required - set(arm)
        if missing:
            raise ValueError(f"{name} missing {sorted(missing)}")
    counterfactual_preference = answer_preference(
        original_lp=float(counterfactual_arm["original_lp"]),
        counterfactual_lp=float(counterfactual_arm["counterfactual_lp"]),
    )
    unrelated_preference = answer_preference(
        original_lp=float(unrelated_arm["original_lp"]),
        counterfactual_lp=float(unrelated_arm["counterfactual_lp"]),
    )
    return {
        "counterfactual_preference": counterfactual_preference,
        "unrelated_preference": unrelated_preference,
        "substitution_effect": (
            counterfactual_preference - unrelated_preference),
        "absolute_calibration": {
            "counterfactual_arm": dict(counterfactual_arm),
            "unrelated_arm": dict(unrelated_arm),
        },
    }
