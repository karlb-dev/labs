"""Vector transport estimands from registered raw response tensors."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _norm(value: torch.Tensor) -> float:
    return float(value.float().norm())


def _cosine(left: torch.Tensor, right: torch.Tensor, eta: float) -> float | None:
    if _norm(left) <= eta or _norm(right) <= eta:
        return None
    return float(F.cosine_similarity(left.float().reshape(-1), right.float().reshape(-1), dim=0))


def tangent_metrics(
    response: torch.Tensor,
    tangent_prediction: torch.Tensor,
    *,
    eta: float = 1e-12,
) -> dict:
    response_norm = _norm(response)
    tangent_norm = _norm(tangent_prediction)
    return {
        "response_norm": response_norm,
        "tangent_prediction_norm": tangent_norm,
        "tangent_cosine": _cosine(response, tangent_prediction, eta),
        "gain": tangent_norm / max(response_norm, eta),
        "tangent_relative_error": _norm(response - tangent_prediction) / max(response_norm, eta),
    }


def homogeneity_metrics(
    delta_epsilon: torch.Tensor,
    delta_double: torch.Tensor,
    *,
    eta: float = 1e-12,
) -> dict:
    norm = _norm(delta_epsilon)
    double_norm = _norm(delta_double)
    return {
        "homogeneity_defect": _norm(delta_double - 2 * delta_epsilon) / max(2 * norm, eta),
        "homogeneity_response_cosine": _cosine(delta_epsilon, delta_double, eta),
        "homogeneity_scale_ratio": double_norm / max(norm, eta),
    }


def odd_symmetry_defect(
    delta_positive: torch.Tensor,
    delta_negative: torch.Tensor,
    *,
    eta: float = 1e-12,
) -> float:
    return _norm(delta_positive + delta_negative) / max(
        _norm(delta_positive - delta_negative), eta
    )


def additivity_metrics(
    delta_sum: torch.Tensor,
    delta_left: torch.Tensor,
    delta_right: torch.Tensor,
    *,
    eta: float = 1e-12,
) -> dict:
    predicted = delta_left + delta_right
    return {
        "additivity_defect": _norm(delta_sum - predicted) / max(_norm(delta_sum), eta),
        "additivity_cosine": _cosine(delta_sum, predicted, eta),
    }


def response_snr(response: torch.Tensor, clean_repeat_difference: torch.Tensor, *, eta: float = 1e-12) -> float:
    return _norm(response) / max(_norm(clean_repeat_difference), eta)


def validate_metric_row(row: dict) -> None:
    required = {
        "prompt_id", "prompt_sha256", "model_id", "model_revision",
        "source_layer", "source_position", "perturbation_mode", "target_stage",
        "target_representation", "direction_type", "direction_id",
        "direction_sha256", "desired_relative_epsilon", "realized_norm",
        "input_fidelity_cosine", "input_relative_norm_error", "response_snr",
        "exact_jvp_backend", "implementation_sha256", "tangent_cosine", "gain",
        "tangent_relative_error", "homogeneity_defect", "odd_symmetry_defect",
        "additivity_defect", "block_type", "variant", "code_commit",
        "config_sha256", "environment_sha256",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"raw transport row lacks fields: {missing}")
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"raw transport row has non-finite {key}")
