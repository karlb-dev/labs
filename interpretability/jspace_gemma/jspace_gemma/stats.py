"""Robust epsilon-floor/curvature fits and prompt bootstrap helpers."""
from __future__ import annotations

import numpy as np


def robust_floor_curvature_fit(
    epsilon: list[float] | np.ndarray,
    relative_error: list[float] | np.ndarray,
    *,
    huber_delta: float = 1.345,
    iterations: int = 50,
) -> dict:
    x = np.asarray(epsilon, dtype=float)
    y = np.asarray(relative_error, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0)
    x, y = x[valid], y[valid]
    if len(x) < 3 or len(np.unique(x)) < 2:
        raise ValueError("a+b*epsilon fit requires at least three finite points")
    design = np.column_stack([np.ones_like(x), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    weights = np.ones_like(y)
    for _ in range(iterations):
        residual = y - design @ coefficients
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        scale = max(float(scale), 1e-12)
        normalized = np.abs(residual) / (huber_delta * scale)
        new_weights = np.where(normalized <= 1, 1.0, 1.0 / normalized)
        weighted = design * np.sqrt(new_weights[:, None])
        response = y * np.sqrt(new_weights)
        new_coefficients = np.linalg.lstsq(weighted, response, rcond=None)[0]
        if np.max(np.abs(new_coefficients - coefficients)) < 1e-12:
            coefficients, weights = new_coefficients, new_weights
            break
        coefficients, weights = new_coefficients, new_weights
    predicted = design @ coefficients
    return {
        "intercept_a": float(coefficients[0]),
        "slope_b": float(coefficients[1]),
        "n_points": int(len(x)),
        "weighted_rmse": float(np.sqrt(np.average((y - predicted) ** 2, weights=weights))),
        "epsilon_min": float(x.min()),
        "epsilon_max": float(x.max()),
        "method": "Huber IRLS with prompt-level bootstrap required for aggregate intervals",
    }


def prompt_bootstrap(
    values_by_prompt: dict[str, list[float]],
    *,
    statistic=np.median,
    draws: int = 5000,
    seed: int = 0,
) -> dict:
    prompts = sorted(values_by_prompt)
    if len(prompts) < 2:
        raise ValueError("prompt bootstrap requires at least two prompts")
    per_prompt = np.array([statistic(values_by_prompt[key]) for key in prompts], dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for index in range(draws):
        selected = rng.integers(0, len(per_prompt), size=len(per_prompt))
        samples[index] = statistic(per_prompt[selected])
    return {
        "estimate": float(statistic(per_prompt)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "n_prompts": len(prompts),
        "draws": draws,
        "seed": seed,
    }
