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


def quantization_floor_norm(value: torch.Tensor, dtype: torch.dtype) -> float:
    """Conservative half-step norm at the target's realized compute dtype."""
    quantized = value.detach().to(device="cpu", dtype=dtype)
    if not quantized.is_floating_point():
        raise ValueError("quantization floor requires a floating-point dtype")
    upward = torch.nextafter(quantized, torch.full_like(quantized, float("inf")))
    downward = torch.nextafter(quantized, torch.full_like(quantized, float("-inf")))
    up_step = (upward.float() - quantized.float()).abs()
    down_step = (quantized.float() - downward.float()).abs()
    # Half the mean distance to the two adjacent representable values. This is
    # a conservative, deterministic target-rounding floor rather than the zero
    # returned by repeating an otherwise deterministic clean forward pass.
    half_step = 0.25 * (up_step + down_step)
    half_step = torch.where(torch.isfinite(half_step), half_step, torch.zeros_like(half_step))
    return _norm(half_step)


def quantization_aware_response_snr(
    response: torch.Tensor,
    tangent_prediction: torch.Tensor,
    clean_repeat_difference: torch.Tensor,
    target_quantization_floor: float,
    *,
    eta: float = 1e-12,
) -> dict:
    """Signal/noise contract frozen before model execution.

    Signal is the smaller of the finite response and exact tangent norms, so a
    quantization-only secant cannot manufacture high SNR. Noise is the larger
    of the in-batch clean-repeat difference and local target-dtype half-step
    norm.
    """
    signal = min(_norm(response), _norm(tangent_prediction))
    clean_repeat = _norm(clean_repeat_difference)
    noise = max(clean_repeat, float(target_quantization_floor), eta)
    return {
        "response_snr": signal / noise,
        "response_snr_signal_norm": signal,
        "response_snr_noise_norm": noise,
        "target_quantization_floor_norm": float(target_quantization_floor),
        "clean_repeat_noise_norm": clean_repeat,
    }


def adjusted_homogeneity_metrics(
    delta_epsilon: torch.Tensor,
    delta_double: torch.Tensor,
    tangent_epsilon: torch.Tensor,
    tangent_double: torch.Tensor,
    *,
    eta: float = 1e-12,
) -> dict:
    """Separate first-order delivery mismatch from nonlinear remainder."""
    denominator = max(2 * _norm(delta_epsilon), eta)
    observed = delta_double - 2 * delta_epsilon
    first_order = tangent_double - 2 * tangent_epsilon
    return {
        "homogeneity_first_order_delivery_defect": _norm(first_order) / denominator,
        "homogeneity_nonlinear_remainder_defect": _norm(observed - first_order) / denominator,
    }


def adjusted_odd_symmetry_metrics(
    delta_positive: torch.Tensor,
    delta_negative: torch.Tensor,
    tangent_positive: torch.Tensor,
    tangent_negative: torch.Tensor,
    *,
    eta: float = 1e-12,
) -> dict:
    """Remove the exact JVP of non-opposite realized +/- perturbations."""
    denominator = max(_norm(delta_positive - delta_negative), eta)
    observed = delta_positive + delta_negative
    first_order = tangent_positive + tangent_negative
    return {
        "odd_first_order_delivery_defect": _norm(first_order) / denominator,
        "odd_nonlinear_remainder_defect": _norm(observed - first_order) / denominator,
    }


def adjusted_additivity_metrics(
    delta_sum: torch.Tensor,
    delta_left: torch.Tensor,
    delta_right: torch.Tensor,
    tangent_sum: torch.Tensor,
    tangent_left: torch.Tensor,
    tangent_right: torch.Tensor,
    *,
    eta: float = 1e-12,
) -> dict:
    """Remove the exact JVP of realized pair-delivery non-additivity."""
    denominator = max(_norm(delta_sum), eta)
    observed = delta_sum - delta_left - delta_right
    first_order = tangent_sum - tangent_left - tangent_right
    return {
        "additivity_first_order_delivery_defect": _norm(first_order) / denominator,
        "additivity_nonlinear_remainder_defect": _norm(observed - first_order) / denominator,
    }


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
