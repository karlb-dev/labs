"""Small-cluster and randomization statistics for Phase 4."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def monte_carlo_pvalue(null: np.ndarray, observed: float, *,
                       alternative: str = "two-sided") -> float:
    values = np.asarray(null, dtype=float)
    if alternative == "two-sided":
        extreme = np.count_nonzero(
            np.abs(values) >= abs(observed) - 1e-15)
    elif alternative == "greater":
        extreme = np.count_nonzero(values >= observed - 1e-15)
    elif alternative == "less":
        extreme = np.count_nonzero(values <= observed + 1e-15)
    else:
        raise ValueError(alternative)
    return float((int(extreme) + 1) / (len(values) + 1))


def exact_signflip(values: np.ndarray, *,
                   alternative: str = "two-sided") -> dict:
    vector = np.asarray(values, dtype=float)
    vector = vector[np.isfinite(vector)]
    families = len(vector)
    if not 3 <= families <= 22:
        raise ValueError(
            f"exact sign flip requires 3..22 families, got {families}")
    bits = np.arange(2**families, dtype=np.uint32)[:, None]
    signs = (
        1 - 2 * ((bits >> np.arange(families, dtype=np.uint32)) & 1)
    ).astype(np.int8)
    distribution = (signs @ vector) / families
    observed = float(vector.mean())
    if alternative == "two-sided":
        extreme = np.abs(distribution) >= abs(observed) - 1e-15
    elif alternative == "greater":
        extreme = distribution >= observed - 1e-15
    elif alternative == "less":
        extreme = distribution <= observed + 1e-15
    else:
        raise ValueError(alternative)
    return {
        "estimate": observed,
        "p": float(extreme.mean()),
        "n_families": families,
        "n_patterns": int(len(distribution)),
        "alternative": alternative,
        "method": "exact-family-signflip",
        "distribution_sha256": hashlib.sha256(
            np.asarray(distribution, dtype="<f8").tobytes()).hexdigest(),
    }


def family_bootstrap_percentile(
        rows: pd.DataFrame, value_col: str, *,
        family_col: str = "canonical_family",
        draws: int = 100_000, seed: int = 4242) -> dict:
    means = rows.groupby(family_col)[value_col].mean()
    if len(means) < 3:
        raise ValueError("family bootstrap requires at least 3 families")
    values = means.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    distribution = values[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci95": [float(low), float(high)],
        "n_families": int(len(values)),
        "n_bootstrap": int(draws),
        "method": "family-resampling-percentile",
        "distribution_sha256": hashlib.sha256(
            np.asarray(distribution, dtype="<f8").tobytes()).hexdigest(),
    }


def high_minus_low_family_statistic(
        rows: pd.DataFrame, *, value_col: str, load_col: str = "load",
        family_col: str = "canonical_family",
        low: int | float, high: int | float) -> pd.Series:
    subset = rows[rows[load_col].isin([low, high])]
    pivot = subset.pivot_table(
        index=family_col, columns=load_col, values=value_col,
        aggfunc="mean")
    if low not in pivot or high not in pivot:
        raise ValueError("both frozen load endpoints are required")
    return (pivot[high] - pivot[low]).dropna()
