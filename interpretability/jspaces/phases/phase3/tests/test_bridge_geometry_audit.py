import numpy as np
import pandas as pd

from jspace_phase3.experiments.p3_bridge_geometry_audit import (
    analyze, nested_family_ridge, validate_state_header)


def test_state_header_refuses_input_drift():
    import pytest

    with pytest.raises(RuntimeError, match="incompatible"):
        validate_state_header({"runner": "old"}, {"runner": "new"})


def test_nested_family_ridge_never_trains_on_held_out_family():
    families = np.repeat(np.array(["a", "b", "c", "d"]), 3)
    x = np.arange(len(families), dtype=float)[:, None]
    y = 0.5 * x[:, 0]
    pred, folds = nested_family_ridge(x, y, families)
    assert np.isfinite(pred).all()
    assert {fold["held_out_family"] for fold in folds} == set(families)
    assert np.corrcoef(pred, y)[0, 1] > 0.99


def test_analysis_uses_strict_added_rank_profile_match():
    items = []
    sites = []
    for i, family in enumerate(("a", "b", "c", "d")):
        fact = f"f{i}"
        items.append({
            "fact_id": fact, "canonical_family": family, "n_tokens": 2,
            "lp_baseline": -1.0, "lp_span_safe": -1.2,
            "lp_true_bridge": -1.0 + i / 10,
            "lp_distractor_bridge": -1.1,
            "true_piece_count": 2, "distractor_piece_count": 2,
        })
        for arm in ("true", "distractor"):
            for position in (0, 1):
                # f0 has equal means (1.5) but a different rank vector,
                # so only f1..f3 are strict matches.
                rank = 1
                if i == 0:
                    rank = (
                        (1, 2)[position] if arm == "true"
                        else (2, 1)[position])
                sites.append({
                    "fact_id": fact, "canonical_family": family,
                    "arm": arm, "layer": 1, "position": position,
                    "protected_rank_before": 2,
                    "protected_rank_after": 2 + rank,
                    "added_rank": rank,
                    "added_selected_overlap": 0.1 * rank,
                    "rank_selected_before": 3,
                    "rank_selected": 2,
                    "removed_energy_l2_sq": 1.0,
                    "removed_energy_frac": 0.1,
                    "lost_rank": 1,
                    "answer_dir_survival_mean": 1.0,
                    "diagnostic_dir_survival_mean": 1.0,
                    "diagnostic_base_overlap": 0.2,
                    "diagnostic_activation_score_mean": 0.3,
                    "diagnostic_activation_score_max": 0.4,
                    "diagnostic_answer_cosine_mean": 0.05,
                    "projector_overlap": 0.0,
                    "diagnostic_dir_survival_min": 1.0,
                    "n_diagnostic_ids": 2,
                })
    report, paired = analyze(pd.DataFrame(items), pd.DataFrame(sites))
    assert report["exact_geometry_matched_subset"]["n_items"] == 3
    assert report["geometry_invariants"][
        "protected_rank_accounting_failures"] == 0
    assert not bool(paired.loc[paired.fact_id == "f0",
                               "exact_geometry_match"].iloc[0])
