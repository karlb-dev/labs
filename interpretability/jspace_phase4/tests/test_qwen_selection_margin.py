import torch
import pandas as pd


def _record(*, rank=10, margin=0.02):
    return {"effective_rank": rank, "margins": {"10": margin}}


def test_margin_strata_are_geometry_only_and_exhaustive():
    from jspace_phase4.experiments.p4_qwen_selection_margin import (
        margin_stratum,
    )

    assert margin_stratum(
        _record(), _record(), k=10, threshold=0.01) == "stable_core"
    assert margin_stratum(
        _record(margin=0.009), _record(), k=10,
        threshold=0.01) == "near_tie"
    assert margin_stratum(
        _record(rank=9), _record(), k=10,
        threshold=0.01) == "rank_deficient"


def test_capture_validation_accepts_only_boundary_tie_id_substitution():
    from jspace_phase4.experiments.p4_qwen_selection_margin import (
        _validate_capture,
    )

    def frame(next_score=5.0, selected_ids=None):
        rows = []
        for lens in ("a500", "a1000"):
            rows.append({
                "lens": lens, "item_id": "item", "fact_id": "fact",
                "variant": "original", "bank": "F",
                "canonical_family": "family", "layer": 20, "position": 0,
                "eligible_available_positive": 4,
                "eligible_top_ids": [10, 11, 12, 13],
                "eligible_top_scores": [9.0, 5.0, next_score, 1.0],
                "protected_ids": [99], "protected_scores": [2.0],
                "margins": {"2": (5.0 - next_score) / 5.0},
                "intervention_selected_ids": selected_ids or [10, 12],
                "intervention_selected_scores": [9.0, 5.0],
                "effective_rank": 2, "removed_energy_frac": 0.1,
            })
        return pd.DataFrame(rows)

    config = {"contract": {
        "intervention_k": 2, "margin_ks": [2],
        "relative_margin_epsilon": 1e-12}}
    assert len(_validate_capture(frame(), config)) == 2
    with __import__("pytest").raises(
            RuntimeError, match="without boundary tie"):
        _validate_capture(
            frame(next_score=4.0, selected_ids=[10, 77]), config)


def test_relative_margin_and_stable_core_ignore_unstable_fringe():
    from jspace_phase4.experiments.p4_qwen_selection_margin import (
        core_ids,
        jaccard,
        relative_margin,
    )

    scores = [10.0, 9.0, 8.0, 7.99]
    assert relative_margin(scores, 3, 1e-12) == (8.0 - 7.99) / 8.0
    left = core_ids(
        [1, 2, 3, 4], scores, k=3, threshold=0.01, epsilon=1e-12)
    right = core_ids(
        [1, 2, 9, 8], [10.0, 9.0, 8.001, 8.0],
        k=3, threshold=0.01, epsilon=1e-12)
    assert left == [1, 2]
    assert right == [1, 2]
    assert jaccard(left, right) == 1.0
    assert jaccard([1, 2, 3], [1, 2, 9]) == 0.5


def test_disputed_projection_energy_is_reconstructed_from_scores():
    from jspace_phase4.experiments.p4_qwen_selection_margin import (
        disputed_dose_fraction,
    )

    selected_rows = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    protected_rows = torch.empty((0, 3))
    protected_scores = torch.empty((0,))
    fraction = disputed_dose_fraction(
        [10, 11], [3.0, 4.0], {10}, selected_rows,
        protected_rows, protected_scores)
    assert abs(fraction - 16 / 25) < 1e-6


def test_protected_direction_is_removed_from_disputed_dose():
    from jspace_phase4.experiments.p4_qwen_selection_margin import (
        disputed_dose_fraction,
    )

    selected_rows = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    protected_rows = torch.tensor([[1.0, 0.0, 0.0]])
    # h=(5,4,0): the first selected row is null after protection.
    fraction = disputed_dose_fraction(
        [10, 11], [5.0, 4.0], {10}, selected_rows,
        protected_rows, torch.tensor([5.0]))
    assert fraction == 1.0


def test_duplicate_row_replacement_preserves_projector_exactly():
    from jspace_phase4.experiments.p4_qwen_selection_margin import (
        replacement_projector_overlap,
    )

    selected = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    overlap = replacement_projector_overlap(
        selected, torch.empty((0, 3)), replace_index=1,
        replacement_row=torch.tensor([0.0, 2.0, 0.0]))
    assert abs(overlap - 1.0) < 1e-6


def test_lexical_alias_categories_leave_semantics_for_blinded_review():
    from jspace_phase4.experiments.p4_qwen_selection_margin import (
        lexical_category,
    )

    assert lexical_category("ĠParis", " Paris") == "normalized_surface_alias"
    assert lexical_category(
        "Ġcompute", "Ġcomputing") == "morphological_or_piece_variant"
    assert lexical_category("ĠParis", "Ġcopper") == "manual_review_required"
