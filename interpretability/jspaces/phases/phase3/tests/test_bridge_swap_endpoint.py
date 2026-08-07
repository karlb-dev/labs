import pandas as pd

from jspace_phase3.experiments.p3_bridge_swap_endpoint_audit import (
    analyze, boundary_generation_category, validate_state_header)


def test_boundary_generation_grading_rejects_partial_prefix():
    assert boundary_generation_category(
        "yen", ["ye"], ["yen"])["category"] == "counterfactual"
    assert boundary_generation_category(
        "yen today", ["yen"], ["baht"])["category"] == "original"
    assert boundary_generation_category(
        "yenish", ["yen"], ["baht"])["category"] == "other"


def test_state_header_refuses_drift():
    import pytest

    with pytest.raises(RuntimeError, match="incompatible"):
        validate_state_header({"model": "a"}, {"model": "b"})


def test_swap_analysis_uses_preference_not_only_original_damage():
    arms = {
        "baseline", "span_safe", "true_protect", "distractor_protect",
        "bridge_only", "cf_swap", "true_reinject", "unrelated_swap",
        "random_orthogonal_swap", "cf_answer_swap",
    }
    rows = []
    for index, family in enumerate(("a", "b", "c")):
        for arm in arms:
            original = -1.0
            counterfactual = -3.0
            if arm == "cf_swap":
                original = -2.0
                counterfactual = -1.0
            rows.append({
                "fact_id": f"f{index}",
                "canonical_family": family,
                "arm": arm,
                "lp_original_canonical": original,
                "lp_counterfactual_canonical": counterfactual,
                "preference_canonical": counterfactual - original,
                "lp_original_max_alias": original,
                "lp_counterfactual_max_alias": counterfactual,
                "preference_max_alias": counterfactual - original,
                "greedy_category": (
                    "counterfactual" if arm == "cf_swap" else "original"),
                "legacy_lp_original": -1.0,
                "unrelated_match_json": (
                    '{"piece_count_exact": true, '
                    '"piece_count_difference": 0, '
                    '"answer_overlap_rmse": 0.0, '
                    '"selected_fact_id": "other", '
                    '"selected_family": "other"}'),
            })
    report, paired = analyze(pd.DataFrame(rows))
    assert report["primary_cf_preference_shift"][
        "estimate_equal_family"] == 3.0
    assert report["arm_results"]["cf_swap"]["greedy_generation"][
        "counterfactual"]["item_rate"] == 1.0
    assert (paired.primary_cf_preference_shift == 3.0).all()
