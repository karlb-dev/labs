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


def test_partial_bridge_ablator_matches_rank_and_removed_norm_dose():
    from jspace_phase4.orthogonal_bridge import OrthogonalBridgeAblator

    table = torch.eye(4)
    ablator = OrthogonalBridgeAblator(
        [], [0], dictionaries={0: table},
        offsets={10: 0, 11: 1, 12: 2, 13: 3})
    base = {
        "arm": "orthogonal",
        "k": 10,
        "selection_sign_rule": "all_rows_regardless_of_activation_sign",
        "protect_ids": torch.tensor([[12, 13]]),
        "restrict_ids": torch.tensor([10]),
        "inject_dir": {0: torch.tensor([0.0, 1.0, 0.0, 0.0])},
        "maximum_injection_direction_norm_error": 1e-6,
        "maximum_injection_dose_relative_error": 1e-6,
        "maximum_injection_dose_absolute_error": 1e-6,
    }
    ablator.mode = base
    changed = ablator._apply(torch.tensor([[[2.0, 3.0, 4.0, 5.0]]]), 0)
    assert changed[0, 0].tolist() == pytest.approx([0.0, 5.0, 4.0, 5.0])
    record = ablator.log.positions[0]
    assert record.selected_ids == (10,)
    assert record.selected_rank == 1
    assert record.effective_rank == 1
    assert record.removed_norm == pytest.approx(2.0)
    assert record.delivered_injection_norm == pytest.approx(2.0)
    assert record.injection_dose_relative_error == pytest.approx(0.0)

    ablator.reset_log()
    ablator.mode = {**base, "arm": "no_injection", "inject_dir": None}
    lesioned = ablator._apply(
        torch.tensor([[[2.0, 3.0, 4.0, 5.0]]]), 0)
    assert lesioned[0, 0].tolist() == pytest.approx([0.0, 3.0, 4.0, 5.0])
    control = ablator.log.positions[0]
    assert control.selected_ids == record.selected_ids
    assert control.effective_rank == record.effective_rank


def test_partial_bridge_ablator_protection_and_prefix_limit_are_exact():
    from jspace_phase4.orthogonal_bridge import OrthogonalBridgeAblator

    table = torch.eye(3)
    ablator = OrthogonalBridgeAblator(
        [], [0], dictionaries={0: table}, offsets={20: 0, 21: 1, 22: 2})
    ablator.mode = {
        "arm": "no_injection", "k": 10,
        "selection_sign_rule": "all_rows_regardless_of_activation_sign",
        "protect_ids": torch.tensor([[20], [22]]),
        "restrict_ids": torch.tensor([20, 21]), "inject_dir": None,
        "active_position_limit": 1,
    }
    hidden = torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]])
    changed = ablator._apply(hidden, 0)
    assert changed[0, 0].tolist() == pytest.approx([2.0, 0.0, 4.0])
    assert torch.equal(changed[0, 1], hidden[0, 1])
    assert len(ablator.log.positions) == 1
    assert ablator.log.positions[0].selected_ids == (21,)


def test_partial_dictionary_rows_follow_gain_then_operator():
    from jspace_phase4.orthogonal_bridge import partial_j_dictionary_rows

    class FakeModel:
        def __init__(self):
            self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor([
                [1.0, 0.0], [0.0, 2.0], [1.0, 1.0],
            ]))

        def get_output_embeddings(self):
            return self.embedding

    rows = partial_j_dictionary_rows(
        FakeModel(), torch.tensor([2.0, 1.0]), torch.eye(2), [0, 2],
        device="cpu", dtype=torch.float32, chunk=1)
    assert rows[0].tolist() == pytest.approx([1.0, 0.0])
    assert rows[1].tolist() == pytest.approx(
        torch.nn.functional.normalize(torch.tensor([2.0, 1.0]), dim=0))


def test_partial_dictionary_refuses_duplicate_token_ids():
    from jspace_phase4.orthogonal_bridge import partial_j_dictionary_rows

    class FakeModel:
        def get_output_embeddings(self):
            return torch.nn.Embedding(2, 2)

    with pytest.raises(ValueError, match="unique"):
        partial_j_dictionary_rows(
            FakeModel(), torch.ones(2), torch.eye(2), [0, 0],
            device="cpu", dtype=torch.float32)
