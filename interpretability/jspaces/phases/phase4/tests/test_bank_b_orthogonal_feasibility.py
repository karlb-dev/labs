from pathlib import Path

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


def _mechanical_config():
    return {
        "consumed_cohort": {"expected_items": 2},
        "intervention": {
            "arm_order": ARMS,
            "maximum_injection_direction_norm_error": 1e-6,
            "maximum_injection_dose_relative_error": 1e-6,
            "maximum_injection_dose_absolute_error": 1e-6,
        },
    }


def _profiles():
    rows = []
    for fact in ("fact-a", "fact-b"):
        for arm in ARMS:
            rows.append({
                "fact_id": fact, "canonical_family": fact,
                "call_key": "generation_prefill", "arm": arm,
                "layer": 20, "phase": "prefill", "position": 0,
                "selected_ids_json": "[10, 11]", "selected_rank": 2,
                "effective_rank": 2, "lost_rank": 0,
                "removed_norm": 0.5,
                "injection_direction_norm": (
                    None if arm == "no_injection" else 1.0),
                "delivered_injection_norm": (
                    0.0 if arm == "no_injection" else 0.5),
                "injection_dose_relative_error": 0.0,
                "injection_dose_absolute_error": 0.0,
            })
    return pd.DataFrame(rows)


def _outcomes():
    return pd.DataFrame([
        {"fact_id": fact, "arm": arm}
        for fact in ("fact-a", "fact-b") for arm in ARMS])


def test_mechanical_profiles_require_exact_cross_arm_rank_and_dose():
    from jspace_phase4.experiments.p4_bank_b_orthogonal_feasibility import (
        _verify_mechanical_profiles,
    )

    report = _verify_mechanical_profiles(
        _profiles(), _outcomes(), config=_mechanical_config())
    assert report["passed"] is True
    assert report["n_profile_cells_compared_across_arms"] == 2

    changed = _profiles()
    changed.loc[
        (changed.fact_id == "fact-a")
        & (changed.arm == ARMS[0]), "effective_rank"] = 1
    report = _verify_mechanical_profiles(
        changed, _outcomes(), config=_mechanical_config())
    assert report["passed"] is False
    assert report["checks"][
        "exact_selected_and_effective_rank_match_across_arms"] is False


def test_geometry_profile_replay_uses_tolerance_but_not_rank_drift():
    from jspace_phase4.experiments.p4_bank_b_orthogonal_feasibility import (
        _profiles_close,
    )

    left = [{"piece_count": 2, "retained_fraction": 0.5}]
    near = [{"piece_count": 2, "retained_fraction": 0.5000001}]
    wrong_rank = [{"piece_count": 3, "retained_fraction": 0.5}]
    assert _profiles_close(
        left, near, relative_tolerance=1e-4,
        absolute_tolerance=1e-6) is True
    assert _profiles_close(
        left, wrong_rank, relative_tolerance=1e-4,
        absolute_tolerance=1e-6) is False


def test_canonical_binding_reports_placeholders_without_registry_access():
    from jspace_phase4.experiments.p4_bank_b_orthogonal_feasibility import (
        canonical_lens_binding,
    )

    config = {"canonical_lens": {
        "decision_result_sha256": "BIND_DECISION",
        "lens_sha256": "BIND_LENS",
    }}
    result = canonical_lens_binding(config, require_bound=False)
    assert result["bound"] is False
    assert result["placeholders"] == ["BIND_DECISION", "BIND_LENS"]


def test_state_resume_refuses_header_drift(tmp_path):
    from jspace_phase4.experiments.p4_bank_b_orthogonal_feasibility import (
        _load_state,
    )

    path = tmp_path / "state.json"
    state = _load_state(path, {"config_sha256": "a"})
    assert state["header"] == {"config_sha256": "a"}
    with pytest.raises(RuntimeError, match="incompatible"):
        _load_state(path, {"config_sha256": "b"})


def test_preflight_does_not_claim_readiness_without_bindings(monkeypatch):
    from jspace_phase4.experiments import (
        p4_bank_b_orthogonal_feasibility as producer,
    )

    bundle = type("Bundle", (), {
        "fact_id": "fact-a", "canonical_family": "family-a"})()
    monkeypatch.setattr(producer, "_source_cohort", lambda _config: (
        [bundle], {"outcome_columns_read": [], "bank_b_rows_read": False},
        pd.DataFrame()))
    monkeypatch.setattr(producer, "canonical_lens_binding", lambda *_args,
                        **_kwargs: {"bound": False})
    monkeypatch.setattr(producer, "independent_review_binding", lambda *_args,
                        **_kwargs: {"complete": False})
    monkeypatch.setattr(producer, "_registered_output_check",
                        lambda _evidence_id: None)
    result = producer.preflight(
        Path("config.yaml"), {"geometry_evidence_id": "geometry"})
    assert result["geometry_execution_ready"] is False
    assert result["outcome_execution_ready"] is False
    assert result["bank_b_rows_opened"] is False


def test_review_contract_is_explicit_in_frozen_config():
    import yaml

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((
        root / "configs/p4_bank_b_orthogonal_feasibility_dev.yaml"
    ).read_text())
    assert config["consumed_cohort"]["bank_b_candidate_rows_forbidden"] is True
    assert config["intervention"]["selection_sign_rule"] == (
        "all_rows_regardless_of_activation_sign")
    assert config["power_gate"]["substantive_joint_sesoi_nats"] == 0.25
    assert config["power_gate"]["observed_development_test"] == (
        "exact_one_sided_family_signflip")
    findings = set(config["review_contract"]["required_boolean_findings"])
    assert "consumed_cohort_and_outcome_firewall_approved" in findings
    assert "variance_and_power_decision_rule_approved" in findings
