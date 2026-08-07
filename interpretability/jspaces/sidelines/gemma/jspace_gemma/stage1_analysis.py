"""Frozen Gemma Stage-1 aggregation and decision rules."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .stats import prompt_bootstrap, robust_floor_curvature_fit


def tangent_pass(frame: pd.DataFrame, contract: dict) -> pd.Series:
    passed = (
        (frame["tangent_cosine"] >= contract["tangent_cosine_floor"])
        & (
            frame["tangent_relative_error"]
            <= contract["tangent_relative_error_ceiling"]
        )
    )
    if "central_tangent_relative_error_ceiling" in contract:
        passed &= (
            frame["central_tangent_relative_error"]
            <= contract["central_tangent_relative_error_ceiling"]
        )
    return passed.fillna(False)


def smallest_evaluable(
    frame: pd.DataFrame,
    *,
    mode: str,
    snr_floor: float,
    selection: str,
) -> pd.DataFrame:
    selected = frame[
        (frame["perturbation_mode"] == mode)
        & frame["faithful_delivery"]
        & (frame["response_snr"] >= snr_floor)
        & frame["tangent_relative_error"].notna()
    ].copy()
    keys = ["prompt_id", "source_layer", "perturbation_mode", "direction_id"]
    result = (
        selected.sort_values("desired_relative_epsilon")
        .groupby(keys, as_index=False, sort=True)
        .first()
    )
    result["selection"] = selection
    result["selection_snr_floor"] = float(snr_floor)
    return result


def _fit_classification(fit: dict, contract: dict) -> str:
    intercept = fit["intercept_a"]
    slope = fit["slope_b"]
    if slope <= contract["negative_slope_precision_floor_ceiling"]:
        return "quantization_floor_limited"
    if (
        intercept <= contract["primary_anchor_intercept_a_ceiling"]
        and slope >= contract["curvature_slope_b_floor"]
    ):
        return "curvature_dominant"
    if (
        intercept > contract["primary_anchor_intercept_a_ceiling"]
        and slope >= contract["curvature_slope_b_floor"]
    ):
        return "mixed_bias_and_curvature"
    if (
        intercept > contract["primary_anchor_intercept_a_ceiling"]
        and abs(slope) <= contract["flat_slope_absolute_ceiling"]
    ):
        return "flat_bias_or_tangent_mismatch"
    return "unresolved"


def curvature_fits(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    selected = frame[
        frame["faithful_delivery"]
        & (frame["response_snr"] >= contract["fitting_response_snr_floor"])
        & frame["tangent_relative_error"].notna()
    ]
    rows = []
    keys = ["prompt_id", "source_layer", "perturbation_mode", "direction_id"]
    for identity, group in selected.groupby(keys, sort=True):
        if (
            len(group) < contract["minimum_unique_epsilon_points"]
            or group["desired_relative_epsilon"].nunique()
            < contract["minimum_unique_epsilon_points"]
        ):
            continue
        fit = robust_floor_curvature_fit(
            group["desired_relative_epsilon"].tolist(),
            group["tangent_relative_error"].tolist(),
        )
        values = dict(zip(keys, identity, strict=True))
        values["source_layer"] = int(values["source_layer"])
        values["classification"] = _fit_classification(fit, contract)
        rows.append({**values, **fit})
    columns = [
        *keys,
        "classification",
        "intercept_a",
        "slope_b",
        "n_points",
        "weighted_rmse",
        "epsilon_min",
        "epsilon_max",
        "method",
    ]
    return pd.DataFrame(rows, columns=columns)


def _coverage(frame: pd.DataFrame, expected: int) -> dict:
    return {
        "n_evaluable": int(len(frame)),
        "n_expected": int(expected),
        "evaluable_fraction": float(len(frame) / expected),
        "n_prompts": int(frame["prompt_id"].nunique()) if len(frame) else 0,
        "n_directions": int(frame["direction_id"].nunique()) if len(frame) else 0,
    }


def _coverage_passes(coverage: dict, contract: dict) -> bool:
    return bool(
        coverage["evaluable_fraction"] >= contract["minimum_evaluable_fraction"]
        and coverage["n_prompts"] >= contract["minimum_prompts"]
        and coverage["n_directions"] >= contract["minimum_directions"]
    )


def _pass_summary(frame: pd.DataFrame, contract: dict) -> dict:
    if not len(frame):
        return {"n": 0, "n_pass": 0, "pass_fraction": None}
    passed = tangent_pass(frame, contract)
    return {
        "n": int(len(frame)),
        "n_pass": int(passed.sum()),
        "pass_fraction": float(passed.mean()),
    }


def _layer_decision(
    layer: int,
    *,
    smallest: pd.DataFrame,
    declared: pd.DataFrame,
    expected: int,
    thresholds: dict,
) -> dict:
    smallest_layer = smallest[smallest["source_layer"] == layer]
    declared_layer = declared[declared["source_layer"] == layer]
    coverage_contract = thresholds["layer_decision_coverage"]
    smallest_coverage = _coverage(smallest_layer, expected)
    declared_coverage = _coverage(declared_layer, expected)
    smallest_pass = _pass_summary(
        smallest_layer, thresholds["smallest_faithful_secant"]
    )
    declared_pass = _pass_summary(declared_layer, thresholds["finite_dose_gate"])

    if not _coverage_passes(smallest_coverage, coverage_contract):
        decision = "unmeasurable_not_nonlinear"
        reason = "insufficient primary-SNR coverage at the smallest faithful secant"
    elif (
        smallest_pass["pass_fraction"]
        < thresholds["smallest_faithful_secant"]["minimum_row_pass_fraction"]
    ):
        decision = "local_tangent_mismatch"
        reason = "smallest high-confidence secants fail the frozen tangent gate"
    elif not _coverage_passes(declared_coverage, coverage_contract):
        decision = "declared_dose_unmeasurable_not_nonlinear"
        reason = "epsilon-0.10 responses do not meet frozen coverage and SNR"
    elif (
        declared_pass["pass_fraction"]
        < thresholds["finite_dose_gate"]["minimum_row_pass_fraction"]
    ):
        decision = "finite_radius_mismatch"
        reason = "local tangent passes but the declared epsilon-0.10 dose fails"
    else:
        decision = "transport_pass"
        reason = "smallest high-confidence and declared-dose gates both pass"
    return {
        "source_layer": int(layer),
        "decision": decision,
        "reason": reason,
        "smallest_evaluable": smallest_coverage,
        "smallest_pass": smallest_pass,
        "declared_dose_evaluable": declared_coverage,
        "declared_dose_pass": declared_pass,
        "smallest_epsilon_counts": {
            str(float(key)): int(value)
            for key, value in smallest_layer["desired_relative_epsilon"]
            .value_counts()
            .sort_index()
            .items()
        },
    }


def _median(frame: pd.DataFrame, field: str) -> float | None:
    values = frame[field].dropna().astype(float)
    return float(values.median()) if len(values) else None


def _epsilon_aggregates(frame: pd.DataFrame, snr_floor: float) -> list[dict]:
    rows = []
    keys = ["source_layer", "perturbation_mode", "desired_relative_epsilon"]
    for identity, group in frame.groupby(keys, sort=True):
        layer, mode, epsilon = identity
        measurable = group[
            group["faithful_delivery"] & (group["response_snr"] >= snr_floor)
        ]
        rows.append(
            {
                "source_layer": int(layer),
                "perturbation_mode": str(mode),
                "desired_relative_epsilon": float(epsilon),
                "n_rows": int(len(group)),
                "n_faithful_delivery": int(group["faithful_delivery"].sum()),
                "n_measurement_evaluable": int(len(measurable)),
                "measurement_evaluable_fraction": float(len(measurable) / len(group)),
                "median_response_snr": _median(measurable, "response_snr"),
                "median_tangent_cosine": _median(measurable, "tangent_cosine"),
                "median_tangent_relative_error": _median(
                    measurable, "tangent_relative_error"
                ),
                "median_central_tangent_relative_error": _median(
                    measurable, "central_tangent_relative_error"
                ),
                "median_homogeneity_nonlinear_remainder_defect": _median(
                    measurable, "homogeneity_nonlinear_remainder_defect"
                ),
                "median_odd_nonlinear_remainder_defect": _median(
                    measurable, "odd_nonlinear_remainder_defect"
                ),
                "median_additivity_nonlinear_remainder_defect": _median(
                    measurable, "additivity_nonlinear_remainder_defect"
                ),
            }
        )
    return rows


def _bootstrap_declared(frame: pd.DataFrame, thresholds: dict) -> dict:
    result = {}
    config = thresholds["bootstrap"]
    for layer, group in frame.groupby("source_layer", sort=True):
        by_prompt = {
            str(prompt): prompt_rows["tangent_relative_error"]
            .dropna()
            .astype(float)
            .tolist()
            for prompt, prompt_rows in group.groupby("prompt_id")
        }
        by_prompt = {key: value for key, value in by_prompt.items() if value}
        if len(by_prompt) < 2:
            result[f"L{int(layer)}"] = None
        else:
            result[f"L{int(layer)}"] = prompt_bootstrap(
                by_prompt,
                draws=int(config["draws"]),
                seed=int(config["seed"]) + int(layer),
            )
    return result


def analyze_stage1(
    rows: list[dict],
    *,
    thresholds: dict,
    prompt_ids: list[str],
    layers: list[int],
    direction_ids: list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(rows)
    expected_per_layer = len(prompt_ids) * len(direction_ids)
    measurement_floor = float(thresholds["response_snr"]["measurement_floor"])
    decision_floor = float(thresholds["response_snr"]["primary_decision_floor"])
    dose = float(thresholds["finite_dose_gate"]["declared_relative_epsilon"])

    measurement_single = smallest_evaluable(
        frame,
        mode="single_position",
        snr_floor=measurement_floor,
        selection="smallest_measurement_single",
    )
    primary_single = smallest_evaluable(
        frame,
        mode="single_position",
        snr_floor=decision_floor,
        selection="smallest_primary_single",
    )
    measurement_uniform = smallest_evaluable(
        frame,
        mode="uniform_valid",
        snr_floor=measurement_floor,
        selection="smallest_measurement_uniform",
    )
    primary_uniform = smallest_evaluable(
        frame,
        mode="uniform_valid",
        snr_floor=decision_floor,
        selection="smallest_primary_uniform",
    )
    selected = pd.concat(
        [measurement_single, primary_single, measurement_uniform, primary_uniform],
        ignore_index=True,
    )
    selected["frozen_tangent_pass"] = False
    for label, contract in (
        ("smallest_measurement_single", thresholds["measurement_only_tangent_region"]),
        ("smallest_primary_single", thresholds["smallest_faithful_secant"]),
        ("smallest_measurement_uniform", thresholds["secondary_uniform_valid"]),
        ("smallest_primary_uniform", thresholds["secondary_uniform_valid"]),
    ):
        mask = selected["selection"] == label
        selected.loc[mask, "frozen_tangent_pass"] = tangent_pass(
            selected.loc[mask], contract
        ).to_numpy()

    declared_primary = frame[
        (frame["perturbation_mode"] == "single_position")
        & (frame["desired_relative_epsilon"] == dose)
        & frame["faithful_delivery"]
        & (frame["response_snr"] >= decision_floor)
    ].copy()
    declared_secondary = frame[
        (frame["perturbation_mode"] == "uniform_valid")
        & (frame["desired_relative_epsilon"] == dose)
        & frame["faithful_delivery"]
        & (frame["response_snr"] >= decision_floor)
    ].copy()
    decisions = [
        _layer_decision(
            int(layer),
            smallest=primary_single,
            declared=declared_primary,
            expected=expected_per_layer,
            thresholds=thresholds,
        )
        for layer in layers
    ]
    fits = curvature_fits(frame, thresholds["floor_curvature_partition"])
    fit_counts = Counter(fits["classification"].tolist()) if len(fits) else Counter()
    secondary_layers = []
    for layer in layers:
        layer_rows = declared_secondary[declared_secondary["source_layer"] == layer]
        coverage = _coverage(layer_rows, expected_per_layer)
        passage = _pass_summary(layer_rows, thresholds["secondary_uniform_valid"])
        secondary_layers.append(
            {
                "source_layer": int(layer),
                "coverage": coverage,
                "pass": passage,
                "coverage_pass": _coverage_passes(
                    coverage, thresholds["layer_decision_coverage"]
                ),
            }
        )

    summary = {
        "n_rows": int(len(frame)),
        "n_faithful_delivery": int(frame["faithful_delivery"].sum()),
        "n_measurement_evaluable": int(
            (
                frame["faithful_delivery"]
                & (frame["response_snr"] >= measurement_floor)
            ).sum()
        ),
        "n_primary_snr_evaluable": int(
            (
                frame["faithful_delivery"]
                & (frame["response_snr"] >= decision_floor)
            ).sum()
        ),
        "primary_layer_decisions": decisions,
        "secondary_uniform_declared_dose": secondary_layers,
        "epsilon_aggregates_at_measurement_floor": _epsilon_aggregates(
            frame, measurement_floor
        ),
        "curvature_fit_count": int(len(fits)),
        "curvature_classification_counts": {
            str(key): int(value) for key, value in sorted(fit_counts.items())
        },
        "declared_dose_prompt_bootstrap_tangent_relative_error": _bootstrap_declared(
            declared_primary, thresholds
        ),
        "selection_counts": {
            str(key): int(value)
            for key, value in selected["selection"].value_counts().sort_index().items()
        },
        "decision_semantics": {
            "unmeasurable_not_nonlinear": (
                "coverage/SNR is insufficient; this is not a nonlinearity claim"
            ),
            "local_tangent_mismatch": (
                "smallest high-confidence secants fail the exact-JVP tangent gate"
            ),
            "declared_dose_unmeasurable_not_nonlinear": (
                "epsilon-0.10 lacks required coverage/SNR"
            ),
            "finite_radius_mismatch": (
                "local tangent passes but epsilon-0.10 fails"
            ),
            "transport_pass": (
                "smallest high-confidence and epsilon-0.10 gates both pass"
            ),
        },
    }
    return summary, selected, fits
