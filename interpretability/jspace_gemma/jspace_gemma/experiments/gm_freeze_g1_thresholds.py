"""Validate the OLMo positive control and register pre-Gemma G1 thresholds."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from jspace_gemma.manifests import atomic_json, file_sha256, require_clean_tree
from jspace_gemma.paths import PACKAGE_ROOT, directory
from jspace_gemma.registry import create, read_events, resolve
from jspace_gemma.stats import prompt_bootstrap, robust_floor_curvature_fit

EVIDENCE_ID = "gm-jvp-olmo-positive-control-v1"
CALIBRATION_ID = "gm-jvp-olmo-calibration-v1"
THRESHOLDS = PACKAGE_ROOT / "configs/gm_g1_thresholds_frozen.yaml"
DESIGN = PACKAGE_ROOT / "configs/gm_g1_design.yaml"
CALIBRATION_ROOT = directory("metrics") / "olmo_control" / CALIBRATION_ID
SUMMARY = CALIBRATION_ROOT / "olmo_calibration_summary.json"
ROWS = CALIBRATION_ROOT / "olmo_calibration_rows.parquet"


def _smallest_measurable(
    frame: pd.DataFrame,
    *,
    layers: list[int],
    mode: str,
    snr_floor: float,
) -> pd.DataFrame:
    selected = frame[
        frame["source_layer"].isin(layers)
        & (frame["perturbation_mode"] == mode)
        & frame["faithful_delivery"]
        & (frame["response_snr"] >= snr_floor)
    ].copy()
    return (
        selected.sort_values("desired_relative_epsilon")
        .groupby(["prompt_id", "source_layer", "direction_id"], as_index=False)
        .first()
    )


def _tangent_pass(frame: pd.DataFrame, contract: dict) -> pd.Series:
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
    return passed


def _distribution(values: pd.Series) -> dict:
    array = values.dropna().astype(float).to_numpy()
    return {
        "n": len(array),
        "minimum": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _curvature_fits(frame: pd.DataFrame, snr_floor: float) -> pd.DataFrame:
    selected = frame[
        frame["faithful_delivery"] & (frame["response_snr"] >= snr_floor)
    ]
    rows = []
    keys = ["prompt_id", "source_layer", "perturbation_mode", "direction_id"]
    for identity, group in selected.groupby(keys):
        if len(group) < 3 or group["desired_relative_epsilon"].nunique() < 2:
            continue
        fit = robust_floor_curvature_fit(
            group["desired_relative_epsilon"].tolist(),
            group["tangent_relative_error"].tolist(),
        )
        values = dict(zip(keys, identity, strict=True))
        values["source_layer"] = int(values["source_layer"])
        rows.append({**values, **fit})
    return pd.DataFrame(rows)


def _bootstrap_metric(frame: pd.DataFrame, field: str, config: dict) -> dict:
    values = {
        prompt: group[field].astype(float).tolist()
        for prompt, group in frame.groupby("prompt_id")
    }
    return prompt_bootstrap(
        values,
        draws=int(config["draws"]),
        seed=int(config["seed"]),
    )


def main() -> None:
    git = require_clean_tree()
    thresholds = yaml.safe_load(THRESHOLDS.read_text())
    design = yaml.safe_load(DESIGN.read_text())
    if design["threshold_calibration"]["status"] != (
        "FROZEN_PENDING_POSITIVE_CONTROL_REGISTRATION"
    ) or design["threshold_calibration"]["gemma_execution_allowed"]:
        raise RuntimeError("target firewall must remain closed during threshold registration")
    if not thresholds["frozen_before_first_gemma_result"]:
        raise RuntimeError("threshold file is not marked frozen before Gemma")

    events = read_events()
    if any(
        row["evidence_id"] == EVIDENCE_ID
        and row["event"] in {"evidence_created", "evidence_imported"}
        for row in events
    ):
        raise RuntimeError("positive-control threshold evidence already exists")
    if any(row.get("target_model_opened") is True for row in events):
        raise RuntimeError("a target model was opened before threshold freezing")
    calibration = resolve(CALIBRATION_ID)
    if not calibration["live"]:
        raise RuntimeError("OLMo calibration evidence is not live")
    expected_outputs = {Path(row["path"]): row["sha256"] for row in calibration["outputs"]}
    for path in (SUMMARY, ROWS):
        if expected_outputs.get(path) != file_sha256(path):
            raise RuntimeError(f"registered calibration output drifted: {path}")
    if file_sha256(SUMMARY) != thresholds["calibration_summary_sha256"]:
        raise RuntimeError("threshold summary hash mismatch")
    if file_sha256(ROWS) != thresholds["calibration_rows_sha256"]:
        raise RuntimeError("threshold row-table hash mismatch")

    summary = json.loads(SUMMARY.read_text())
    frame = pd.read_parquet(ROWS)
    if (
        len(frame) != 1568
        or summary["completed_cells"] != 56
        or summary["n_rows"] != len(frame)
    ):
        raise RuntimeError("registered calibration grid is incomplete")

    anchor_layers = thresholds["known_linear_band_contrast"][
        "late_anchor_layers_zero_indexed"
    ]
    measurement = _smallest_measurable(
        frame,
        layers=anchor_layers,
        mode="single_position",
        snr_floor=thresholds["response_snr"]["measurement_floor"],
    )
    primary = _smallest_measurable(
        frame,
        layers=anchor_layers,
        mode="single_position",
        snr_floor=thresholds["response_snr"]["primary_decision_floor"],
    )
    secondary = _smallest_measurable(
        frame,
        layers=anchor_layers,
        mode="uniform_valid",
        snr_floor=thresholds["response_snr"]["primary_decision_floor"],
    )
    if len(measurement) != 32 or len(primary) != 32 or len(secondary) != 32:
        raise RuntimeError("positive-control anchor coverage is incomplete")

    dose = float(thresholds["finite_dose_gate"]["declared_relative_epsilon"])
    primary_dose = frame[
        frame["source_layer"].isin(anchor_layers)
        & (frame["perturbation_mode"] == "single_position")
        & (frame["desired_relative_epsilon"] == dose)
        & frame["faithful_delivery"]
        & (frame["response_snr"] >= thresholds["response_snr"]["primary_decision_floor"])
    ]
    secondary_dose = frame[
        frame["source_layer"].isin(anchor_layers)
        & (frame["perturbation_mode"] == "uniform_valid")
        & (frame["desired_relative_epsilon"] == dose)
        & frame["faithful_delivery"]
        & (frame["response_snr"] >= thresholds["response_snr"]["primary_decision_floor"])
    ]

    primary_pass = _tangent_pass(primary, thresholds["smallest_faithful_secant"])
    measurement_pass = _tangent_pass(
        measurement, thresholds["measurement_only_tangent_region"]
    )
    primary_dose_pass = _tangent_pass(primary_dose, thresholds["finite_dose_gate"])
    secondary_dose_pass = _tangent_pass(
        secondary_dose, thresholds["secondary_uniform_valid"]
    )

    contrast = thresholds["known_linear_band_contrast"]
    contrast_rows = frame[
        (frame["perturbation_mode"] == "single_position")
        & (frame["desired_relative_epsilon"] == contrast["contrast_relative_epsilon"])
        & frame["faithful_delivery"]
    ]
    shallow = contrast_rows[
        contrast_rows["source_layer"] == contrast["shallow_layer_zero_indexed"]
    ]
    late = contrast_rows[contrast_rows["source_layer"].isin(anchor_layers)]
    error_contrast = float(
        shallow["tangent_relative_error"].median()
        - late["tangent_relative_error"].median()
    )
    cosine_contrast = float(
        late["tangent_cosine"].median() - shallow["tangent_cosine"].median()
    )

    partition = thresholds["floor_curvature_partition"]
    fits = _curvature_fits(frame, partition["fitting_response_snr_floor"])
    anchor_fits = fits[
        fits["source_layer"].isin(anchor_layers)
        & (fits["perturbation_mode"] == "single_position")
    ]
    matched_layers = design["models"]["olmo_positive_control"][
        "matched_layers_zero_indexed"
    ] + [design["models"]["olmo_positive_control"]["late_identity_anchor_layer"]]
    matched_fits = fits[
        fits["source_layer"].isin(matched_layers)
        & (fits["perturbation_mode"] == "single_position")
    ]
    fit_coverage = len(anchor_fits) / 32
    intercept_q95 = float(np.quantile(anchor_fits["intercept_a"], 0.95))
    matched_slope_max = float(matched_fits["slope_b"].max())

    growth = {}
    for layer in anchor_layers:
        medians = frame[
            (frame["source_layer"] == layer)
            & (frame["perturbation_mode"] == "single_position")
            & frame["faithful_delivery"]
            & frame["desired_relative_epsilon"].isin([0.10, 0.20])
        ].groupby("desired_relative_epsilon")["tangent_relative_error"].median()
        growth[f"L{layer}"] = {
            "median_error_0.10": float(medians.loc[0.10]),
            "median_error_0.20": float(medians.loc[0.20]),
            "increase": float(medians.loc[0.20] - medians.loc[0.10]),
        }

    criteria = {
        "clean_suffix_parity": summary["max_clean_relative_l2_error"] == 0,
        "exact_jvp_primal_parity": summary["max_backend_parity_relative_error"]
        <= thresholds["exact_jvp_primal_relative_error_ceiling"],
        "wrong_hook_sentinel": summary["wrong_hook_sentinel"]["relative_l2_error"]
        >= thresholds["wrong_hook_relative_error_floor"],
        "primary_smallest_high_confidence": float(primary_pass.mean())
        >= thresholds["smallest_faithful_secant"]["minimum_row_pass_fraction"],
        "measurement_region": float(measurement_pass.mean())
        >= thresholds["measurement_only_tangent_region"]["minimum_row_pass_fraction"],
        "primary_finite_dose": float(primary_dose_pass.mean())
        >= thresholds["finite_dose_gate"]["minimum_row_pass_fraction"],
        "secondary_uniform_dose": (
            len(secondary_dose) / 32
            >= thresholds["layer_decision_coverage"]["minimum_evaluable_fraction"]
            and float(secondary_dose_pass.mean())
            >= thresholds["secondary_uniform_valid"]["minimum_row_pass_fraction"]
        ),
        "shallow_to_late_error_contrast": error_contrast
        >= contrast["shallow_minus_late_tangent_relative_error_minimum"],
        "shallow_to_late_cosine_contrast": cosine_contrast
        >= contrast["late_minus_shallow_tangent_cosine_minimum"],
        "fit_coverage": fit_coverage >= partition["minimum_fit_coverage"],
        "anchor_intercept": intercept_q95
        <= partition["primary_anchor_intercept_a_ceiling"],
        "matched_control_slope": matched_slope_max
        < partition["curvature_slope_b_floor"],
        "post_floor_error_growth": all(row["increase"] > 0 for row in growth.values()),
        "uniform_is_qualitatively_noisier": float(
            secondary_dose["tangent_relative_error"].median()
        )
        > float(primary_dose["tangent_relative_error"].median()),
    }
    if not all(criteria.values()):
        raise RuntimeError(
            "OLMo positive control failed frozen criteria: "
            + json.dumps(criteria, sort_keys=True)
        )

    artifact = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "tier": "methods",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold_definition_code_commit": git["code_commit"],
        "threshold_config": {
            "path": str(THRESHOLDS),
            "sha256": file_sha256(THRESHOLDS),
        },
        "calibration": {
            "evidence_id": CALIBRATION_ID,
            "summary": {"path": str(SUMMARY), "sha256": file_sha256(SUMMARY)},
            "rows": {"path": str(ROWS), "sha256": file_sha256(ROWS)},
            "compute_code_commit": summary["cell_compute_code_commit"],
            "finalization_code_commit": summary["finalization_code_commit"],
            "cells_recomputed_during_finalization": False,
        },
        "criteria": criteria,
        "positive_control_pass": True,
        "empirical": {
            "n_rows": len(frame),
            "n_faithfully_delivered_rows": int(frame["faithful_delivery"].sum()),
            "primary_smallest_epsilon_counts": {
                str(key): int(value)
                for key, value in primary["desired_relative_epsilon"].value_counts().sort_index().items()
            },
            "primary_smallest": {
                "pass_fraction": float(primary_pass.mean()),
                "tangent_cosine": _distribution(primary["tangent_cosine"]),
                "tangent_relative_error": _distribution(
                    primary["tangent_relative_error"]
                ),
                "central_tangent_relative_error": _distribution(
                    primary["central_tangent_relative_error"]
                ),
            },
            "measurement_smallest": {
                "pass_fraction": float(measurement_pass.mean()),
                "tangent_cosine": _distribution(measurement["tangent_cosine"]),
                "tangent_relative_error": _distribution(
                    measurement["tangent_relative_error"]
                ),
            },
            "primary_finite_dose": {
                "n": len(primary_dose),
                "pass_fraction": float(primary_dose_pass.mean()),
                "tangent_cosine": _distribution(primary_dose["tangent_cosine"]),
                "tangent_relative_error": _distribution(
                    primary_dose["tangent_relative_error"]
                ),
                "central_tangent_relative_error": _distribution(
                    primary_dose["central_tangent_relative_error"]
                ),
            },
            "secondary_uniform_finite_dose": {
                "n": len(secondary_dose),
                "pass_fraction": float(secondary_dose_pass.mean()),
                "tangent_cosine": _distribution(secondary_dose["tangent_cosine"]),
                "tangent_relative_error": _distribution(
                    secondary_dose["tangent_relative_error"]
                ),
            },
            "known_linear_band_contrast": {
                "shallow_median_tangent_relative_error": float(
                    shallow["tangent_relative_error"].median()
                ),
                "late_median_tangent_relative_error": float(
                    late["tangent_relative_error"].median()
                ),
                "error_contrast": error_contrast,
                "shallow_median_tangent_cosine": float(
                    shallow["tangent_cosine"].median()
                ),
                "late_median_tangent_cosine": float(late["tangent_cosine"].median()),
                "cosine_contrast": cosine_contrast,
            },
            "floor_curvature": {
                "n_anchor_fits": len(anchor_fits),
                "anchor_fit_coverage": fit_coverage,
                "anchor_intercept_q95": intercept_q95,
                "matched_primary_slope_max": matched_slope_max,
                "post_floor_error_growth": growth,
            },
            "prompt_bootstrap": {
                "primary_finite_dose_tangent_cosine": _bootstrap_metric(
                    primary_dose, "tangent_cosine", thresholds["bootstrap"]
                ),
                "primary_finite_dose_tangent_relative_error": _bootstrap_metric(
                    primary_dose,
                    "tangent_relative_error",
                    thresholds["bootstrap"],
                ),
            },
        },
        "thresholds": thresholds,
        "target_model_opened": False,
        "gemma_numbers_observed": False,
        "claim_boundary": (
            "OLMo positive-control and operational threshold freeze only; "
            "no Gemma result"
        ),
    }
    output = directory("metrics") / "olmo_control" / "gm-jvp-olmo-positive-control-v1.json"
    atomic_json(output, artifact)
    create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "OLMo exact-JVP positive-control pass and numeric G1 thresholds "
            "frozen before any Gemma target number"
        ),
        command="python -m jspace_gemma.experiments.gm_freeze_g1_thresholds",
        outputs=[THRESHOLDS, output],
        inputs={
            "calibration_summary_sha256": file_sha256(SUMMARY),
            "calibration_rows_sha256": file_sha256(ROWS),
            "threshold_config_sha256": file_sha256(THRESHOLDS),
        },
        positive_control_pass=True,
        thresholds_frozen_before_target=True,
        target_model_opened=False,
    )
    print(json.dumps({"output": str(output), "sha256": file_sha256(output)}, indent=1))


if __name__ == "__main__":
    main()
