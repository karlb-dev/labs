import dataclasses

import pandas as pd
import torch


def _valid_config():
    digest = "a" * 64
    return {
        "tier": "phase4-development",
        "lens_order": ["published", "a120", "a250"],
        "lenses": {
            "published": {
                "lens_sha256": digest,
                "provenance_classification": (
                    "external published reference, partially specified "
                    "recipe"),
            },
            "a120": {"lens_sha256": digest},
            "a250": {"lens_sha256": digest},
        },
        "protocol": {
            "condition_order": [
                "span_safe", "exact_matched", "true_bridge",
                "distractor_bridge", "counterfactual_swap"],
            "answer_alias_rule": "first_phase3_accepted_alias",
        },
        "g4": {"condition_order": ["baseline", "swap_j", "swap_random"]},
        "bridge_endpoint": {
            "category_contract": [
                "original", "counterfactual", "other-invalid"]},
        "analysis": {
            "pair_order": [
                ["a120", "a250"],
                ["a250", "published"],
                ["a120", "published"],
            ],
        },
    }


def test_functional_config_refuses_order_or_unregistered_hash():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        validate_config,
    )
    config = _valid_config()
    validate_config(config)
    config["lens_order"] = ["a120", "published", "a250"]
    try:
        validate_config(config)
    except RuntimeError as error:
        assert "order" in str(error)
    else:
        raise AssertionError("lens-order drift was accepted")
    config = _valid_config()
    config["lenses"]["a250"]["lens_sha256"] = "PENDING"
    try:
        validate_config(config)
    except RuntimeError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("unregistered n=250 placeholder was accepted")


def test_successor_lens_order_and_primary_pair_are_config_bound():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        primary_pair,
        validate_config,
    )
    config = _valid_config()
    config["lens_order"] = ["published", "a250", "a500"]
    config["lenses"]["a500"] = config["lenses"].pop("a120")
    config["analysis"] = {
        "primary_pair": ["a250", "a500"],
        "pair_order": [
            ["a250", "a500"],
            ["a500", "published"],
            ["a250", "published"],
        ],
        "structural_comparison_id": "a250_vs_a500",
    }
    validate_config(config)
    assert primary_pair(config) == ("a250", "a500")
    config["analysis"]["primary_pair"] = ["a500", "a250"]
    try:
        validate_config(config)
    except RuntimeError as error:
        assert "primary-pair" in str(error)
    else:
        raise AssertionError("primary-pair drift was accepted")


def test_a1000_order_requires_nonintervening_margin_capture():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        validate_config,
    )
    config = _valid_config()
    config["lens_order"] = ["published", "a500", "a1000"]
    config["lenses"]["a500"] = config["lenses"].pop("a120")
    config["lenses"]["a1000"] = config["lenses"].pop("a250")
    config["analysis"] = {
        "primary_pair": ["a500", "a1000"],
        "pair_order": [
            ["a500", "a1000"],
            ["a1000", "published"],
            ["a500", "published"],
        ],
        "structural_comparison_id": "a500_vs_a1000",
    }
    config["protocol"]["k"] = 10
    config["selection_margin_capture"] = {
        "enabled": True,
        "top_n": 32,
        "margin_ks": [1, 2, 5, 10, 20],
        "intervention_k": 10,
        "include_all_strata_in_functional_gate": True,
    }
    validate_config(config)
    config["selection_margin_capture"]["intervention_k"] = 9
    with __import__("pytest").raises(
            RuntimeError, match="may not change intervention k"):
        validate_config(config)


def test_structural_metrics_select_configured_successor_comparison():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        structural_metrics,
    )
    config = {
        "lens_order": ["published", "a250", "a500"],
        "protocol": {"band": [20, 44]},
        "analysis": {
            "primary_pair": ["a250", "a500"],
            "structural_comparison_id": "a250_vs_a500",
            "structural_assay_key": "assay_L20_L44",
        },
    }
    assay = {}
    for ordinal, name in enumerate((
            "task_answer_only", "task_bridge_only",
            "task_answer_bridge_shared")):
        assay[f"token_{name}_direction_cosine_q50"] = {
            "median": 0.99 - ordinal * 0.01}
        assay[f"token_{name}_direction_cosine_q05"] = {
            "median": 0.96 - ordinal * 0.01}
    result = {
        "aggregate": {"a250_vs_a500": {"assay_L20_L44": assay}}}
    metrics = structural_metrics(result, config)
    assert metrics["comparison_id"] == "a250_vs_a500"
    assert metrics["assay_task_token_median_cosine_conservative"] == 0.97
    assert metrics["task_token_q05_conservative"] == 0.94


def test_span_safe_basis_removes_protected_rank_exactly():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        selected_span_basis,
    )
    selected = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    protected = torch.tensor([[1.0, 0.0, 0.0]])
    basis, metadata = selected_span_basis(selected, protected)
    assert basis.shape == (3, 1)
    assert metadata == {"raw_rank": 2, "effective_rank": 1, "lost_rank": 1}
    assert abs(float(basis[:, 0] @ protected[0])) < 1e-6


def test_selection_pair_metrics_identical_and_orthogonal():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        selection_pair_metrics,
    )
    left = torch.eye(4)[:, :2]
    identical = selection_pair_metrics(
        [2, 3], [2, 3], [4.0, 3.0], [8.0, 7.0], left, left)
    assert identical["selected_id_jaccard"] == 1
    assert abs(identical["normalized_projector_overlap"] - 1) < 1e-6
    assert identical["principal_angle_max_degrees"] < 0.1
    orthogonal = selection_pair_metrics(
        [2, 3], [4, 5], [4.0, 3.0], [2.0, 1.0],
        left, torch.eye(4)[:, 2:])
    assert orthogonal["selected_id_jaccard"] == 0
    assert orthogonal["normalized_projector_overlap"] == 0
    assert abs(orthogonal["principal_angle_max_degrees"] - 90) < 1e-5


def test_target_rank_and_pass_contract():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        pass_at_k,
        target_rank,
    )
    scores = torch.tensor([0.1, 0.5, -2.0, 0.4, 0.3])
    assert target_rank(scores, [3, 4]) == 2
    assert pass_at_k(2, 1) is False
    assert pass_at_k(2, 5) is True
    assert target_rank(scores, []) is None
    assert pass_at_k(None, 20) is None


def test_family_pairing_and_branch_rule_are_paired_not_correlational():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        branch_from_gates,
        paired_family_summary,
    )
    rows = []
    for family, left, right in (
            ("f1", -1.0, -1.02),
            ("f2", -0.6, -0.61),
            ("f3", -0.2, -0.18),
            ("f4", -0.8, -0.79)):
        rows.extend([
            {"canonical_family": family, "lens": "a120", "specific": left},
            {"canonical_family": family, "lens": "a250", "specific": right},
        ])
    result = paired_family_summary(
        pd.DataFrame(rows), left="a120", right="a250",
        column="specific", sesoi=0.15, draws=200, seed=7)
    assert result["n_families"] == 4
    assert abs(result["equal_family_mean_difference"]) < 0.02
    assert branch_from_gates(
        {"a": True, "b": True}, structural_stable=True) == "A"
    assert branch_from_gates(
        {"a": True, "b": True}, structural_stable=False) == "C"
    assert branch_from_gates(
        {"a": True, "b": False}, structural_stable=True) == "B"


def test_frozen_ql_router_separates_rows_spans_and_causal_failures():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        ql_branch_from_gates,
    )
    passing = {
        "normalized_selected_span_overlap": True,
        "selected_id_jaccard": True,
        "occupancy": True,
        "centered_excess": True,
        "span_safe_specific": True,
        "tail_rate": True,
        "g4": True,
        "bridge_rescue": True,
        "bridge_preference": True,
    }
    assert ql_branch_from_gates(
        passing, structural_stable=True) == "Q-L1"
    rows_drift = {**passing, "selected_id_jaccard": False}
    assert ql_branch_from_gates(
        rows_drift, structural_stable=True) == "Q-L2"
    spans_drift = {
        **rows_drift, "normalized_selected_span_overlap": False}
    assert ql_branch_from_gates(
        spans_drift, structural_stable=True) == "Q-L3"
    causal_drift = {**passing, "bridge_rescue": False}
    assert ql_branch_from_gates(
        causal_drift, structural_stable=True) == "Q-L4"
    assert ql_branch_from_gates(
        spans_drift, structural_stable=False) == "Q-L5"
    assert ql_branch_from_gates(
        passing, structural_stable=None) == "PENDING_STRUCTURAL"


def test_margin_candidate_capture_observes_protection_and_exact_gaps():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        selection_margin_candidates,
    )
    scores = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, -1.0]])
    rows = selection_margin_candidates(
        scores, torch.tensor([[1]]), intervention_k=2, top_n=4,
        margin_ks=[1, 2], epsilon=1e-12)
    row = rows[0]
    assert row["raw_top_ids"] == [0, 1, 2, 3]
    assert row["eligible_top_ids"] == [0, 2, 3, 4]
    assert row["intervention_selected_ids"] == [0, 2]
    assert row["protected_ids"] == [1]
    assert row["protected_scores"] == [8.0]
    assert abs(row["margins"]["1"] - 2 / 9) < 1e-7
    assert abs(row["margins"]["2"] - 1 / 7) < 1e-7


def test_margin_candidate_capture_replays_tied_intervention_topk_exactly():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        selection_margin_candidates,
    )
    scores = torch.ones((1, 8))
    exact_ids = scores.topk(2, dim=1).indices[0].tolist()
    diagnostic_prefix = scores.topk(5, dim=1).indices[0, :2].tolist()
    # Frozen torch currently demonstrates why top-N[:k] is not a valid
    # replay of top-k at an exact tie boundary.
    assert exact_ids != diagnostic_prefix
    row = selection_margin_candidates(
        scores, None, intervention_k=2, top_n=5,
        margin_ks=[1, 2], epsilon=1e-12)[0]
    assert row["intervention_selected_ids"] == exact_ids
    assert row["eligible_top_ids"][:2] == diagnostic_prefix
    assert row["intervention_selected_scores"] == [1.0, 1.0]


def test_margin_observer_is_bitwise_intervention_identical_to_phase3_parent():
    from jspace_phase3.ablator3 import Phase3JAblator
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        Phase4MarginCaptureAblator,
    )
    generator = torch.Generator().manual_seed(20260802)
    hidden = torch.rand((1, 3, 6), generator=generator)
    dictionary = torch.nn.functional.normalize(
        torch.rand((14, 6), generator=generator), dim=1)
    protection = torch.tensor([[0, 1], [2, 3], [4, 5]])
    mode = {
        "dicts": {0: dictionary}, "k": 3, "nonneg": True,
        "protect_sets": protection, "active_phases": {"prefill"},
        "span_safe": True, "record_overlap": True,
        "record_ids": True, "answer_id": None,
    }
    parent = Phase3JAblator([], [0])
    parent.phase, parent.forward_index = "prefill", 0
    parent.mode = dict(mode)
    observed = Phase4MarginCaptureAblator([], [0])
    observed.phase, observed.forward_index = "prefill", 0
    observed.mode = {
        **mode,
        "selection_margin_capture": {
            "top_n": 8, "margin_ks": [1, 2, 3], "epsilon": 1e-12},
    }
    expected_hidden = parent._apply(hidden.clone(), 0)
    actual_hidden = observed._apply(hidden.clone(), 0)
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
    assert [dataclasses.asdict(row) for row in observed.log.positions] == [
        dataclasses.asdict(row) for row in parent.log.positions]
    assert len(observed.log.selection_margin) == hidden.shape[1]
    for capture, intervention in zip(
            observed.log.selection_margin, observed.log.positions):
        assert capture.intervention_selected_ids == intervention.selected_ids
        assert capture.effective_rank == intervention.effective_rank
        assert capture.removed_energy_frac == intervention.removed_energy_frac


def test_margin_observer_is_exact_at_a_tied_topk_boundary():
    from jspace_phase3.ablator3 import Phase3JAblator
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        Phase4MarginCaptureAblator,
    )
    hidden = torch.ones((1, 1, 2))
    dictionary = torch.ones((8, 2))
    mode = {
        "dicts": {0: dictionary}, "k": 2, "nonneg": True,
        "protect_sets": None, "active_phases": {"prefill"},
        "span_safe": True, "record_overlap": True,
        "record_ids": True, "answer_id": None,
    }
    parent = Phase3JAblator([], [0])
    parent.phase, parent.forward_index, parent.mode = "prefill", 0, dict(mode)
    observed = Phase4MarginCaptureAblator([], [0])
    observed.phase, observed.forward_index = "prefill", 0
    observed.mode = {
        **mode,
        "selection_margin_capture": {
            "top_n": 5, "margin_ks": [1, 2], "epsilon": 1e-12},
    }
    expected_hidden = parent._apply(hidden.clone(), 0)
    actual_hidden = observed._apply(hidden.clone(), 0)
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
    assert observed.log.selection_margin[0].intervention_selected_ids == (
        parent.log.positions[0].selected_ids)


def test_capacity_bootstrap_is_deterministic_and_prompt_paired():
    from jspace_phase4.experiments.p4_qwen_multilens_functional_gate import (
        _capacity_bootstrap,
    )
    generator = torch.Generator().manual_seed(9)
    h = torch.randn((8, 5), generator=generator)
    reconstruction = h * 0.8
    random = [h * 0.2, h * 0.3]
    owners = ["p1"] * 4 + ["p2"] * 4
    first = _capacity_bootstrap(
        h, reconstruction, random, owners, draws=50, seed=11)
    second = _capacity_bootstrap(
        h, reconstruction, random, owners, draws=50, seed=11)
    assert first == second
    assert first["resampling_unit"] == "prompt"
