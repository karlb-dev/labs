import copy

import pandas as pd
import pytest


ARMS = [
    "counterfactual_bridge_answer_orthogonal",
    "unrelated_bridge_answer_orthogonal",
    "counterfactual_answer_direction",
    "counterfactual_bridge_full",
    "random_answer_and_bridge_orthogonal",
    "no_injection",
]


def _config():
    return {
        "consumed_cohort": {"expected_items": 4, "expected_families": 4},
        "intervention": {"arm_order": ARMS},
        "variance_gate": {
            "sample_sd_ddof": 1,
            "bootstrap_draws": 200,
            "bootstrap_seed": 17,
            "bootstrap_upper_quantile": 0.90,
            "minimum_iqr_floor_nats": 0.25,
            "maximum_planning_sd_nats_each_component": 3.0,
            "minimum_equal_family_mean_nats_each_component": 0.25,
            "maximum_absolute_family_value_to_iqr_ratio": 10.0,
        },
        "power_gate": {
            "substantive_joint_sesoi_nats": 0.25,
            "holm_planning_alpha": 0.05,
            "target_power": 0.80,
            "confirmatory_families_available": 4,
            "family_count_grid": [4, 10],
            "monte_carlo_draws": 1000,
            "monte_carlo_seed": 23,
        },
    }


def _rows(*, orthogonal=0.7):
    preferences = {
        "counterfactual_bridge_answer_orthogonal": orthogonal,
        "unrelated_bridge_answer_orthogonal": 0.1,
        "counterfactual_answer_direction": 0.2,
        "counterfactual_bridge_full": 0.8,
        "random_answer_and_bridge_orthogonal": 0.0,
        "no_injection": 0.0,
    }
    output = []
    for fact in range(4):
        for arm in ARMS:
            output.append({
                "fact_id": f"fact-{fact}",
                "canonical_family": f"family-{fact}",
                "arm": arm,
                "preference_canonical": preferences[arm],
                "preference_max_alias": preferences[arm] + 0.01,
                "greedy_category": "counterfactual" if arm == ARMS[0]
                else "other",
            })
    return pd.DataFrame(output)


def test_orthogonal_analysis_admits_only_joint_fixed_components():
    from jspace_phase4.bank_b_orthogonal_analysis import (
        analyze_orthogonal_outcomes,
    )

    result, paired = analyze_orthogonal_outcomes(
        _rows(), config=_config(), mechanical_gate={"passed": True})
    assert len(paired) == 4
    assert result["components"]["semantic_component"][
        "equal_family_mean_nats"] == pytest.approx(0.6)
    assert result["components"]["bridge_specific_component"][
        "equal_family_mean_nats"] == pytest.approx(0.5)
    assert result["observed_test_uses_max_component_p"] is True
    assert result["orthogonal_estimand_admitted"] is True
    assert result["p4_p1_disposition"] == (
        "PROSPECTIVE_ORTHOGONAL_REPLACEMENT_LICENSED")
    assert result["reject_language_licensed"] is False


def test_orthogonal_analysis_routes_collapsed_semantics_to_estimation_only():
    from jspace_phase4.bank_b_orthogonal_analysis import (
        analyze_orthogonal_outcomes,
    )

    result, _ = analyze_orthogonal_outcomes(
        _rows(orthogonal=0.2), config=_config(),
        mechanical_gate={"passed": True})
    assert result["variance_checks"]["fixed_mean_floor"] is False
    assert result["variance_gate_pass"] is False
    assert result["orthogonal_estimand_admitted"] is False
    assert result["p4_p1_disposition"] == "P4-E1_ESTIMATION_ONLY"


def test_orthogonal_analysis_never_uses_observed_mean_as_sesoi():
    from jspace_phase4.bank_b_orthogonal_analysis import (
        analyze_orthogonal_outcomes,
    )

    config = copy.deepcopy(_config())
    first, _ = analyze_orthogonal_outcomes(
        _rows(orthogonal=0.7), config=config,
        mechanical_gate={"passed": True})
    second, _ = analyze_orthogonal_outcomes(
        _rows(orthogonal=2.7), config=config,
        mechanical_gate={"passed": True})
    assert first["fixed_sesoi_nats"] == second["fixed_sesoi_nats"] == 0.25
    assert first["power"] == second["power"]
    assert first["observed_mean_used_to_increase_sesoi"] is False


def test_orthogonal_analysis_refuses_incomplete_arm_grid():
    from jspace_phase4.bank_b_orthogonal_analysis import (
        analyze_orthogonal_outcomes,
    )

    with pytest.raises(RuntimeError, match="grid size"):
        analyze_orthogonal_outcomes(
            _rows().iloc[:-1], config=_config(),
            mechanical_gate={"passed": True})
