import pytest
import torch


def test_bridge_direction_is_orthogonal_and_reports_retained_semantics():
    from jspace_phase4.orthogonal_bridge import (
        geometry_gate,
        orthogonal_bridge_direction,
    )

    answer = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    bridge = torch.tensor([[1.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    direction, geometry = orthogonal_bridge_direction(bridge, answer)
    assert direction.tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert geometry.answer_effective_rank == 1
    assert geometry.retained_fraction == pytest.approx(1.5 / (3.25 ** 0.5))
    assert geometry.maximum_answer_span_cosine < 1e-6
    gate = geometry_gate(geometry, {
        "minimum_retained_fraction": 0.5,
        "maximum_answer_span_cosine": 1e-5,
        "minimum_self_readout_cosine_mean": 0.5,
    })
    assert gate["passed"] is True


def test_collapsed_bridge_component_fails_retained_fraction_gate():
    from jspace_phase4.orthogonal_bridge import (
        geometry_gate,
        orthogonal_bridge_direction,
    )

    direction, geometry = orthogonal_bridge_direction(
        torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]))
    assert torch.equal(direction, torch.zeros(2))
    assert geometry.retained_fraction == 0.0
    assert geometry_gate(geometry, {
        "minimum_retained_fraction": 0.2,
        "maximum_answer_span_cosine": 1e-5,
        "minimum_self_readout_cosine_mean": 0.1,
    })["passed"] is False


def _geometry(piece_count, retained, norm, overlap, readout):
    from jspace_phase4.orthogonal_bridge import OrthogonalBridgeGeometry

    return OrthogonalBridgeGeometry(
        piece_count=piece_count, answer_piece_count=2,
        answer_effective_rank=2, raw_mean_norm=norm,
        retained_norm=retained * norm, retained_fraction=retained,
        raw_answer_span_cosine=overlap, maximum_answer_span_cosine=0.0,
        self_readout_cosine_mean=readout,
        self_readout_cosine_min=readout - 0.1,
        self_readout_cosine_max=readout + 0.1)


def test_unrelated_match_is_deterministic_and_excludes_same_family():
    from jspace_phase4.orthogonal_bridge import (
        geometry_match_gate,
        select_unrelated_geometry_match,
    )

    target = [_geometry(2, 0.8, 1.0, 0.2, 0.7)]
    candidates = [
        {"fact_id": "same", "canonical_family": "target",
         "profile": target},
        {"fact_id": "worse", "canonical_family": "other",
         "profile": [_geometry(3, 0.6, 1.2, 0.4, 0.5)]},
        {"fact_id": "best", "canonical_family": "other2",
         "profile": [_geometry(2, 0.79, 1.01, 0.21, 0.69)]},
    ]
    selected, report = select_unrelated_geometry_match(
        target_profile=target, target_fact_id="target-fact",
        target_family="target", candidates=candidates)
    assert selected["fact_id"] == "best"
    assert report["selection_used_outcomes"] is False
    assert geometry_match_gate(report, {
        "maximum_piece_count_difference": 0,
        "maximum_retained_fraction_rmse": 0.02,
        "maximum_raw_mean_norm_relative_rmse": 0.02,
        "maximum_raw_answer_span_cosine_rmse": 0.02,
        "maximum_self_readout_cosine_mean_rmse": 0.02,
    })["passed"] is True


def test_random_control_is_stable_and_orthogonal_to_answer_and_bridge():
    from jspace_phase4.orthogonal_bridge import (
        stable_random_answer_orthogonal_direction,
    )

    answer = torch.eye(8)[:2]
    bridge = torch.eye(8)[2]
    left, report = stable_random_answer_orthogonal_direction(
        answer, bridge, seed=71)
    right, _ = stable_random_answer_orthogonal_direction(
        answer, bridge, seed=71)
    other, _ = stable_random_answer_orthogonal_direction(
        answer, bridge, seed=72)
    assert torch.equal(left, right)
    assert not torch.equal(left, other)
    assert float((answer @ left).abs().max()) < 1e-6
    assert float(bridge @ left) == pytest.approx(0.0, abs=1e-6)
    assert report["maximum_anchor_cosine"] < 1e-6
