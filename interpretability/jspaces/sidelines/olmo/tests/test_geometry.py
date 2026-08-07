import numpy as np
import pytest
import torch

from jspace_olmo_lineage.geometry import (
    aggregate_projector_metrics,
    centered_linear_cka_gram,
    finite_rbo,
    id_selection_metrics,
    marginal_crossing_margins,
    neighbor_overlap,
    normalize_token_alias,
    operator_pair_metrics,
    projector_pair_metrics,
    randomized_spectrum,
    selection_prefixes,
)


def test_operator_views_distinguish_identity_and_residual():
    identity = torch.eye(8)
    perturbation = torch.zeros(8, 8)
    perturbation[0, 1] = 0.2
    metrics = operator_pair_metrics(identity, identity + perturbation)
    assert 0.0 < metrics["symmetric_relative_frobenius_delta"] < 0.2
    assert metrics["raw_matrix_cosine"] < 1.0
    assert metrics["left_trace_projection_alpha"] == pytest.approx(1.0)
    assert metrics["right_trace_projection_alpha"] == pytest.approx(1.0)

    same = operator_pair_metrics(identity, identity)
    assert same["raw_matrix_cosine"] == pytest.approx(1.0)
    assert same["symmetric_relative_frobenius_delta"] == 0.0


def test_cka_and_neighbors_are_rotation_invariant():
    generator = torch.Generator().manual_seed(9)
    rows = torch.randn(24, 10, generator=generator)
    rotation, _ = torch.linalg.qr(
        torch.randn(10, 10, generator=generator))
    rotated = rows @ rotation
    assert centered_linear_cka_gram(rows, rotated) == pytest.approx(
        1.0, abs=1e-5)
    overlap = neighbor_overlap(rows, rotated, k=5)
    assert overlap["overlap_fraction_mean"] == pytest.approx(1.0)


def test_randomized_spectrum_reports_stable_rank():
    diagonal = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5]))
    omega = torch.randn(4, 4, generator=torch.Generator().manual_seed(2))
    result = randomized_spectrum(
        diagonal, omega=omega, rank=4, power_iterations=1)
    expected = float(diagonal.square().sum() / 16.0)
    assert result["estimated_top_singular_value"] == pytest.approx(
        4.0, rel=1e-5)
    assert result["estimated_stable_rank"] == pytest.approx(
        expected, rel=1e-5)
    assert result["leading_spectrum_energy_fraction"] == pytest.approx(
        1.0, rel=1e-5)


def test_selection_prefixes_rbo_and_alias_accounting():
    selected = np.asarray([[7, 8, 9], [4, 5, -1]])
    occupancy = np.asarray([2, 3])
    achieved = np.asarray([3, 2])
    prefixes = selection_prefixes(selected, occupancy, achieved)
    assert prefixes == [[7, 8], [4, 5]]
    assert finite_rbo([1, 2, 3], [1, 2, 3], p=0.9) == pytest.approx(1.0)
    assert finite_rbo([], [], p=0.9) == 1.0
    assert normalize_token_alias("ĠAnswer") == "answer"

    metrics = id_selection_metrics(
        [[7, 8], [4, 5]], [[7, 9], [4, 6]], rbo_p=0.9,
        token_labels={7: "same", 8: "ĠBlue", 9: " blue",
                      4: "x", 5: "green", 6: "coral"})
    assert metrics["different_aligned_slots"] == 2
    assert metrics["normalized_alias_equivalent_swaps"] == 1
    assert metrics["exact_aligned_slot_fraction"] == pytest.approx(0.5)


def test_crossing_margin_is_not_candidate_score_gap():
    j_errors = np.asarray([[10.0, 5.0, 4.6], [8.0, 4.0, 3.8]])
    random_errors = np.asarray([
        [[10.0, 8.0, 7.5], [8.0, 6.0, 5.7]],
        [[10.0, 7.8, 7.2], [8.0, 5.9, 5.5]],
        [[10.0, 8.2, 7.8], [8.0, 6.1, 5.9]],
    ])
    values = marginal_crossing_margins(
        j_errors, random_errors, np.asarray([2, 2]))
    assert values.tolist() == pytest.approx([-0.1, -0.1])


def test_projector_overlap_uses_subspaces_not_ids():
    left = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    right = torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    result = projector_pair_metrics(
        left, right, relative_tolerance=1e-4)
    assert result["left_numerical_rank"] == 2
    assert result["right_numerical_rank"] == 2
    assert result["normalized_projector_overlap"] == pytest.approx(
        1.0, abs=1e-5)
    assert result["principal_angle_max_degrees"] < 0.1


def test_batched_projector_aggregate_retains_every_position():
    dictionary = torch.nn.functional.normalize(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ]), dim=1)
    result = aggregate_projector_metrics(
        dictionary, dictionary,
        [[0, 1], [0], [2, 3], [1]],
        [[2, 3], [0], [0, 1], [1]],
        row_id_to_index={value: value for value in range(4)},
        relative_tolerance=1e-4, batch_positions=2)
    assert result["n_positions"] == 4
    assert result["normalized_projector_overlap"]["q50"] == pytest.approx(
        1.0, abs=1e-5)


def test_geometry_config_keeps_unavailable_margin_fields_explicit():
    import yaml
    from pathlib import Path

    config = yaml.safe_load((Path(__file__).parents[1]
                             / "configs/ol_geometry_v1.yaml").read_text())
    boundary = config["selection_margin_boundary"]
    assert boundary["exact_kth_kplus1_score_gap"]["status"].startswith(
        "not-estimable")
    assert boundary["causal_core_fringe_dose"]["status"].startswith(
        "blocked")
    assert config["models"][-1]["role"] == "sibling_endpoint"
    assert config["geometry_series"]["comparison_scope"].startswith("all-six")
