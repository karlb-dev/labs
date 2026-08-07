"""Frozen analysis for the consumed-development orthogonal bridge shot."""
from __future__ import annotations

import hashlib
import math
from statistics import NormalDist
from typing import Mapping

import numpy as np
import pandas as pd

from .stats4 import exact_signflip


COMPONENT_COLUMNS = {
    "semantic_component": (
        "counterfactual_bridge_answer_orthogonal",
        "unrelated_bridge_answer_orthogonal"),
    "bridge_specific_component": (
        "counterfactual_bridge_answer_orthogonal",
        "counterfactual_answer_direction"),
}


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(
        values, dtype="<f8").tobytes()).hexdigest()


def _wilson(successes: int, trials: int) -> list[float]:
    if trials <= 0:
        raise ValueError("Wilson interval needs positive trials")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(
        p * (1 - p) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def _bootstrap_sd(values: np.ndarray, *, draws: int, seed: int,
                  quantile: float) -> dict:
    vector = np.asarray(values, dtype=np.float64)
    if len(vector) < 3:
        raise ValueError("variance gate needs at least three families")
    generator = np.random.default_rng(seed)
    output = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 5000):
        stop = min(start + 5000, draws)
        indices = generator.integers(
            0, len(vector), size=(stop - start, len(vector)))
        output[start:stop] = vector[indices].std(axis=1, ddof=1)
    return {
        "draws": int(draws),
        "seed": int(seed),
        "upper_quantile": float(quantile),
        "upper": float(np.quantile(output, quantile)),
        "distribution_sha256": _array_sha256(output),
    }


def summarize_component(values: pd.Series, *, specification: Mapping,
                        seed_offset: int) -> dict:
    vector = values.to_numpy(dtype=np.float64)
    if not np.isfinite(vector).all():
        raise RuntimeError("orthogonal component contains a nonfinite family")
    sample_sd = float(vector.std(ddof=int(
        specification["sample_sd_ddof"])))
    bootstrap = _bootstrap_sd(
        vector, draws=int(specification["bootstrap_draws"]),
        seed=int(specification["bootstrap_seed"]) + int(seed_offset),
        quantile=float(specification["bootstrap_upper_quantile"]))
    planning_sd = max(sample_sd, float(bootstrap["upper"]))
    q25, q75 = np.quantile(vector, [0.25, 0.75])
    iqr = float(q75 - q25)
    tail_denominator = max(
        iqr, float(specification["minimum_iqr_floor_nats"]))
    tail_ratio = float(np.max(np.abs(vector)) / tail_denominator)
    return {
        "n_families": int(len(vector)),
        "equal_family_mean_nats": float(vector.mean()),
        "sample_sd_nats": sample_sd,
        "bootstrap_sd": bootstrap,
        "planning_sd_nats": planning_sd,
        "iqr_nats": iqr,
        "maximum_absolute_family_value_nats": float(
            np.max(np.abs(vector))),
        "maximum_absolute_family_value_to_iqr_ratio": tail_ratio,
        "exact_one_sided_signflip": exact_signflip(
            vector, alternative="greater"),
        "family_values_sha256": _array_sha256(vector),
    }


def conservative_normal_iut_power(
        *, family_sds: tuple[float, float], correlation: float,
        specification: Mapping) -> dict:
    """Monte Carlo power for the prospectively fixed normal IUT envelope.

    This is a planning calculation, not a replacement for the exact observed
    family sign-flip.  Rejecting only when both one-sided component z-rulers
    reject is an intersection-union test; the simulated composite-null rows
    make the type-I boundary explicit.
    """
    sds = np.asarray(family_sds, dtype=np.float64)
    if sds.shape != (2,) or not np.isfinite(sds).all() or (sds < 0).any():
        raise ValueError("IUT planning requires two finite nonnegative SDs")
    rho = float(np.clip(correlation, -0.95, 0.95))
    covariance = np.asarray([[1.0, rho], [rho, 1.0]])
    alpha = float(specification["holm_planning_alpha"])
    critical = NormalDist().inv_cdf(1 - alpha)
    draws = int(specification["monte_carlo_draws"])
    seed = int(specification["monte_carlo_seed"])
    sesoi = float(specification["substantive_joint_sesoi_nats"])
    counts = [int(value) for value in specification["family_count_grid"]]

    def scenario(n_families: int, effects: tuple[float, float],
                 *, offset: int) -> dict:
        generator = np.random.default_rng(seed + offset)
        noise = generator.multivariate_normal(
            np.zeros(2), covariance, size=draws)
        standard_errors = sds / math.sqrt(n_families)
        means = np.asarray(effects)[None, :] + noise * standard_errors
        safe = np.where(standard_errors > 0, standard_errors, 1.0)
        statistics = means / safe
        statistics[:, standard_errors == 0] = np.where(
            means[:, standard_errors == 0] > 0,
            np.inf, -np.inf)
        rejected = np.all(statistics >= critical, axis=1)
        successes = int(rejected.sum())
        return {
            "n_families": int(n_families),
            "effects_nats": [float(value) for value in effects],
            "rejections": successes,
            "draws": draws,
            "rejection_rate": float(successes / draws),
            "wilson_ci95": _wilson(successes, draws),
            "simulation_sha256": hashlib.sha256(
                np.packbits(rejected).tobytes()).hexdigest(),
        }

    power = [
        scenario(count, (sesoi, sesoi), offset=1000 + ordinal)
        for ordinal, count in enumerate(counts)
    ]
    available = int(specification["confirmatory_families_available"])
    if available not in counts:
        available_row = scenario(
            available, (sesoi, sesoi), offset=900001)
    else:
        available_row = next(
            row for row in power if row["n_families"] == available)
    strong = float(max(sds.max(), sesoi) * 8)
    type_i = [
        scenario(available, effects, offset=910000 + ordinal)
        for ordinal, effects in enumerate(
            ((0.0, 0.0), (0.0, strong), (strong, 0.0)))
    ]
    target = float(specification["target_power"])
    minimum_powered = next((
        row["n_families"] for row in power
        if row["wilson_ci95"][0] >= target), None)
    type_i_pass = all(
        row["wilson_ci95"][1] <= alpha + 0.005
        for row in type_i)
    return {
        "method": "conservative-normal-intersection-union-monte-carlo",
        "planning_only": True,
        "component_family_sd_nats": sds.tolist(),
        "endpoint_correlation_observed_clipped": rho,
        "holm_planning_alpha": alpha,
        "one_sided_normal_critical_value": float(critical),
        "joint_sesoi_nats": sesoi,
        "power_curve": power,
        "available_confirmatory": available_row,
        "minimum_family_count_with_wilson_lower_at_target": minimum_powered,
        "target_power": target,
        "available_count_powered": bool(
            available_row["wilson_ci95"][0] >= target),
        "composite_null_type_i": type_i,
        "type_i_gate_pass": bool(type_i_pass),
    }


def analyze_orthogonal_outcomes(
        rows: pd.DataFrame, *, config: Mapping,
        mechanical_gate: Mapping) -> tuple[dict, pd.DataFrame]:
    required = {
        "fact_id", "canonical_family", "arm", "preference_canonical",
        "preference_max_alias", "greedy_category",
    }
    missing = required - set(rows.columns)
    if missing:
        raise RuntimeError(f"orthogonal outcome rows lack {sorted(missing)}")
    arms = list(config["intervention"]["arm_order"])
    expected_items = int(config["consumed_cohort"]["expected_items"])
    if len(rows) != expected_items * len(arms):
        raise RuntimeError("orthogonal outcome grid size mismatch")
    counts = rows.groupby("fact_id").arm.nunique()
    if len(counts) != expected_items or not (counts == len(arms)).all():
        raise RuntimeError("orthogonal outcome grid is incomplete by fact")
    if set(rows.arm) != set(arms):
        raise RuntimeError("orthogonal outcome arm set drift")
    key = ["fact_id", "canonical_family"]
    wide = rows.pivot(index=key, columns="arm", values=[
        "preference_canonical", "preference_max_alias"])
    paired = wide.reset_index()
    paired.columns = [
        "__".join(value).strip("_") if isinstance(value, tuple) else value
        for value in paired.columns]
    for component, (left, right) in COMPONENT_COLUMNS.items():
        paired[component] = (
            paired[f"preference_canonical__{left}"]
            - paired[f"preference_canonical__{right}"])
        paired[f"{component}_max_alias"] = (
            paired[f"preference_max_alias__{left}"]
            - paired[f"preference_max_alias__{right}"])

    family = paired.groupby("canonical_family", sort=True)[
        list(COMPONENT_COLUMNS)].mean()
    expected_families = int(config["consumed_cohort"]["expected_families"])
    if len(family) != expected_families:
        raise RuntimeError("orthogonal outcome family count drift")
    variance = config["variance_gate"]
    summaries = {
        component: summarize_component(
            family[component], specification=variance, seed_offset=ordinal)
        for ordinal, component in enumerate(COMPONENT_COLUMNS)
    }
    exact_iut_p = max(
        value["exact_one_sided_signflip"]["p"]
        for value in summaries.values())
    correlation = float(family.corr().iloc[0, 1])
    if not math.isfinite(correlation):
        correlation = 0.0
    power = conservative_normal_iut_power(
        family_sds=tuple(
            summaries[name]["planning_sd_nats"]
            for name in COMPONENT_COLUMNS),
        correlation=correlation, specification=config["power_gate"])

    variance_checks = {
        "planning_sd": all(
            value["planning_sd_nats"] <= float(
                variance["maximum_planning_sd_nats_each_component"])
            for value in summaries.values()),
        "fixed_mean_floor": all(
            value["equal_family_mean_nats"] >= float(
                variance["minimum_equal_family_mean_nats_each_component"])
            for value in summaries.values()),
        "heavy_tail": all(
            value["maximum_absolute_family_value_to_iqr_ratio"] <= float(
                variance["maximum_absolute_family_value_to_iqr_ratio"])
            for value in summaries.values()),
        "semantic_effect_not_collapsed": (
            summaries["semantic_component"]["equal_family_mean_nats"]
            >= float(variance[
                "minimum_equal_family_mean_nats_each_component"])),
        "mechanical": mechanical_gate.get("passed") is True,
    }
    variance_pass = bool(all(variance_checks.values()))
    power_pass = bool(
        power["type_i_gate_pass"] and power["available_count_powered"])
    admitted = bool(variance_pass and power_pass)

    generation = {}
    for arm in arms:
        subset = rows[rows.arm == arm]
        generation[arm] = {
            category: {
                "item_rate": float((subset.greedy_category == category).mean()),
                "count": int((subset.greedy_category == category).sum()),
            }
            for category in ("original", "counterfactual", "ambiguous", "other")
        }
    no_injection = "no_injection"
    orthogonal = "counterfactual_bridge_answer_orthogonal"
    full = "counterfactual_bridge_full"
    result = {
        "schema_version": 1,
        "n_items": int(len(paired)),
        "n_families": int(len(family)),
        "components": summaries,
        "observed_exact_iut_p_descriptive": float(exact_iut_p),
        "observed_test_uses_max_component_p": True,
        "family_component_correlation": correlation,
        "variance_checks": variance_checks,
        "variance_gate_pass": variance_pass,
        "power": power,
        "power_gate_pass": power_pass,
        "orthogonal_estimand_admitted": admitted,
        "p4_p1_disposition": (
            "PROSPECTIVE_ORTHOGONAL_REPLACEMENT_LICENSED"
            if admitted else "P4-E1_ESTIMATION_ONLY"),
        "fixed_sesoi_nats": float(
            config["power_gate"]["substantive_joint_sesoi_nats"]),
        "observed_mean_used_to_increase_sesoi": False,
        "generation": generation,
        "full_bridge_retention_descriptive": {
            "orthogonal_minus_no_injection_equal_family_mean": float(
                (wide[("preference_canonical", orthogonal)]
                 - wide[("preference_canonical", no_injection)])
                .groupby("canonical_family").mean().mean()),
            "full_minus_no_injection_equal_family_mean": float(
                (wide[("preference_canonical", full)]
                 - wide[("preference_canonical", no_injection)])
                .groupby("canonical_family").mean().mean()),
        },
        "untouched_bank_b_outcomes_opened": False,
        "confirmatory_or_replication_outcomes_opened": False,
        "reject_language_licensed": False,
    }
    return result, paired
