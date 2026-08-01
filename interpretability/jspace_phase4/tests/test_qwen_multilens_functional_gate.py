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
