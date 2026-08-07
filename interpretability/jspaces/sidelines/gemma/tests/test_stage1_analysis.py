import pandas as pd

from jspace_gemma.stage1_analysis import analyze_stage1, curvature_fits


def _thresholds():
    return {
        "response_snr": {"measurement_floor": 12.0, "primary_decision_floor": 20.0},
        "smallest_faithful_secant": {
            "tangent_cosine_floor": 0.98,
            "tangent_relative_error_ceiling": 0.20,
            "central_tangent_relative_error_ceiling": 0.10,
            "minimum_row_pass_fraction": 0.90,
        },
        "measurement_only_tangent_region": {
            "tangent_cosine_floor": 0.97,
            "tangent_relative_error_ceiling": 0.25,
            "minimum_row_pass_fraction": 0.90,
        },
        "secondary_uniform_valid": {
            "tangent_cosine_floor": 0.80,
            "tangent_relative_error_ceiling": 0.65,
            "minimum_row_pass_fraction": 0.90,
        },
        "finite_dose_gate": {
            "declared_relative_epsilon": 0.10,
            "tangent_cosine_floor": 0.98,
            "tangent_relative_error_ceiling": 0.20,
            "central_tangent_relative_error_ceiling": 0.10,
            "minimum_row_pass_fraction": 0.90,
        },
        "layer_decision_coverage": {
            "minimum_evaluable_fraction": 0.75,
            "minimum_prompts": 3,
            "minimum_directions": 3,
        },
        "floor_curvature_partition": {
            "fitting_response_snr_floor": 12.0,
            "minimum_unique_epsilon_points": 3,
            "primary_anchor_intercept_a_ceiling": 0.30,
            "curvature_slope_b_floor": 0.15,
            "flat_slope_absolute_ceiling": 0.15,
            "negative_slope_precision_floor_ceiling": -0.15,
        },
        "bootstrap": {"draws": 100, "seed": 240802},
    }


def _rows(*, snr=30.0, declared_error=0.10):
    rows = []
    for prompt in ("p1", "p2", "p3"):
        for direction in ("d1", "d2", "d3"):
            for mode in ("single_position", "uniform_valid"):
                for epsilon, cosine, error, central in (
                    (0.05, 0.995, 0.05, 0.03),
                    (0.10, 0.990, declared_error, 0.05 if declared_error <= 0.2 else 0.30),
                    (0.20, 0.950, 0.30, 0.20),
                ):
                    rows.append(
                        {
                            "prompt_id": prompt,
                            "source_layer": 22,
                            "source_position": -1 if mode == "single_position" else "all_valid",
                            "perturbation_mode": mode,
                            "direction_id": direction,
                            "desired_relative_epsilon": epsilon,
                            "faithful_delivery": True,
                            "response_snr": snr,
                            "tangent_cosine": cosine,
                            "tangent_relative_error": error,
                            "central_tangent_relative_error": central,
                            "homogeneity_nonlinear_remainder_defect": error / 2,
                            "odd_nonlinear_remainder_defect": error / 3,
                            "additivity_nonlinear_remainder_defect": error / 4,
                        }
                    )
    return rows


def test_stage1_analysis_passes_smallest_and_declared_dose():
    summary, selected, fits = analyze_stage1(
        _rows(),
        thresholds=_thresholds(),
        prompt_ids=["p1", "p2", "p3"],
        layers=[22],
        direction_ids=["d1", "d2", "d3"],
    )
    assert summary["primary_layer_decisions"][0]["decision"] == "transport_pass"
    assert summary["selection_counts"]["smallest_primary_single"] == 9
    assert set(selected["selection"]) == {
        "smallest_measurement_single",
        "smallest_primary_single",
        "smallest_measurement_uniform",
        "smallest_primary_uniform",
    }
    assert len(fits) == 18


def test_stage1_analysis_separates_finite_radius_failure_from_low_snr():
    summary, _, _ = analyze_stage1(
        _rows(declared_error=0.40),
        thresholds=_thresholds(),
        prompt_ids=["p1", "p2", "p3"],
        layers=[22],
        direction_ids=["d1", "d2", "d3"],
    )
    assert summary["primary_layer_decisions"][0]["decision"] == "finite_radius_mismatch"

    summary, _, _ = analyze_stage1(
        _rows(snr=5.0),
        thresholds=_thresholds(),
        prompt_ids=["p1", "p2", "p3"],
        layers=[22],
        direction_ids=["d1", "d2", "d3"],
    )
    assert summary["primary_layer_decisions"][0]["decision"] == (
        "unmeasurable_not_nonlinear"
    )


def test_curvature_fit_classification_uses_frozen_partition():
    frame = pd.DataFrame(_rows())
    fits = curvature_fits(frame, _thresholds()["floor_curvature_partition"])
    assert set(fits["classification"]) == {"curvature_dominant"}
