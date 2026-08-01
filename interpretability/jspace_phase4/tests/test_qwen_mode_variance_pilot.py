def _protocol(n_families=3):
    return {
        "selection": {"n_families": n_families},
        "pilot": {
            "modes": ["thinking_on", "thinking_off"],
            "primary_phases": ["prefill", "final_answer"],
            "arms": ["matched_control", "span_safe_j"],
            "cell_order": [
                "thinking_on_final_answer_matched_control",
                "thinking_on_final_answer_span_safe_j",
                "thinking_on_prefill_matched_control",
                "thinking_on_prefill_span_safe_j",
                "thinking_off_final_answer_matched_control",
                "thinking_off_final_answer_span_safe_j",
                "thinking_off_prefill_matched_control",
                "thinking_off_prefill_span_safe_j",
            ],
            "interaction_coefficients": [1, -1, -1, 1, -1, 1, 1, -1],
        },
        "mechanical_gates": {
            "maximum_wrong_phase_hook_fires": 0,
            "minimum_expected_phase_hook_fires_per_row": 1,
            "require_zero_selected_protected_overlap": True,
            "require_exact_rank_match": True,
            "maximum_energy_relative_error": 0.01,
        },
        "variance_summary": {
            "sample_sd_ddof": 1,
            "bootstrap_draws": 1000,
            "bootstrap_seed": 17,
            "bootstrap_upper_quantile": 0.90,
        },
    }


def _rows(interactions=(1, 0, -1)):
    rows = []
    modes = ["thinking_on", "thinking_off"]
    phases = ["prefill", "final_answer"]
    arms = ["matched_control", "span_safe_j"]
    for family_index, interaction in enumerate(interactions):
        family = f"family_{family_index}"
        values = {
            (mode, phase, arm): False
            for mode in modes for phase in phases for arm in arms
        }
        if interaction == 1:
            values[("thinking_on", "final_answer", "matched_control")] = True
        elif interaction == -1:
            values[("thinking_on", "final_answer", "span_safe_j")] = True
        for mode in modes:
            for phase in phases:
                for arm in arms:
                    rows.append({
                        "canonical_family": family,
                        "mode": mode,
                        "phase": phase,
                        "arm": arm,
                        "correct": values[(mode, phase, arm)],
                        "parse_valid": True,
                        "wrong_phase_hook_fires": 0,
                        "expected_phase_hook_fires": 1,
                        "selected_protected_overlap": 0,
                        "requested_rank": 4,
                        "delivered_rank": 4,
                        "energy_relative_error": 0.001,
                    })
    return rows


def test_mode_variance_pilot_analyzes_complete_eight_cell_grid():
    from jspace_phase4.experiments.p4_qwen_mode_variance_pilot import (
        analyze_pilot_rows,
    )
    result = analyze_pilot_rows(_rows(), _protocol())
    assert result["n_families"] == 3
    assert result["n_rows"] == 24
    assert result["family_interactions"] == [1.0, 0.0, -1.0]
    assert result["family_interaction_mean_accuracy_points"] == 0.0
    assert result["family_interaction_sample_sd"] == 1.0
    assert result["planning_family_sd"] >= 1.0
    assert result["pilot_analysis_valid"] is True
    assert result["freeze_ready"] is False


def test_mode_variance_pilot_blocks_wrong_phase_hook_fire():
    from jspace_phase4.experiments.p4_qwen_mode_variance_pilot import (
        analyze_pilot_rows,
    )
    rows = _rows()
    rows[0]["wrong_phase_hook_fires"] = 1
    result = analyze_pilot_rows(rows, _protocol())
    assert result["pilot_analysis_valid"] is False
    assert result["mechanical_gate_checks"][
        "wrong_phase_hook_fires_within_tolerance"] is False


def test_mode_variance_pilot_refuses_incomplete_or_duplicate_grid():
    from jspace_phase4.experiments.p4_qwen_mode_variance_pilot import (
        analyze_pilot_rows,
    )
    for rows in (_rows()[:-1], _rows() + [_rows()[0]]):
        try:
            analyze_pilot_rows(rows, _protocol())
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid mode variance grid was accepted")


def test_mode_variance_parse_failure_must_be_incorrect():
    from jspace_phase4.experiments.p4_qwen_mode_variance_pilot import (
        analyze_pilot_rows,
    )
    rows = _rows()
    rows[0]["parse_valid"] = False
    rows[0]["correct"] = True
    try:
        analyze_pilot_rows(rows, _protocol())
    except RuntimeError:
        pass
    else:
        raise AssertionError("correct parse failure was accepted")
