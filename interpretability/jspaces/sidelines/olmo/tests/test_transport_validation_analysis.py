from jspace_olmo_lineage.experiments.transport_validation_analysis import (
    joint_route,
    qualify_dose_candidate,
)


def test_dose_candidate_rejects_per_item_aggregate():
    result = qualify_dose_candidate(
        [
            "model",
            "item_id",
            "n_positions",
            "removed_energy_mean",
            "removed_energy_max",
        ],
        model_scopes=["olmo31-think"],
    )

    assert not result["usable_for_frozen_dose_join"]
    assert "per_item_removed_energy_aggregate_is_not_site_record" in (
        result["rejection_reasons"])


def test_dose_candidate_rejects_protected_energy_fraction():
    result = qualify_dose_candidate(
        [
            "item_id",
            "layer",
            "position",
            "removed_energy_in_prot_frac",
        ],
        model_scopes=["olmo31-think"],
    )

    assert not result["usable_for_frozen_dose_join"]
    assert "protected_subspace_energy_is_not_total_removed_energy" in (
        result["rejection_reasons"])


def test_joint_route_keeps_missing_dose_coverage_null():
    checkpoint_summaries = {
        "base": {
            "common_assay_valid_epsilons": [],
            "late_anchor_valid_epsilons": [],
        },
        "olmo31_think": {
            "common_assay_valid_epsilons": [],
            "late_anchor_valid_epsilons": [0.1],
        },
    }
    route = joint_route(
        checkpoint_summaries,
        {"route": "unresolved_missing_registered_site_dose_records"},
    )

    assert route["intrinsic_joint_route"] == (
        "h6_fail_in_band_with_checkpoint_specific_late_anchor")
    assert route["intervention_distribution_coverage"] is None
    assert route["h6_pass_in_band_at_relevant_doses"] is None
    assert route["h6_scale_limited"] is None
