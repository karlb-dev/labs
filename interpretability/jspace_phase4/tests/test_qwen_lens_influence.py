import torch


def test_leave_one_out_reconstructs_direct_equal_prompt_refit():
    from jspace_phase4.experiments.p4_qwen_lens_influence import (
        leave_one_out_mean,
    )
    generator = torch.Generator().manual_seed(41)
    contributions = torch.randn((7, 5, 5), generator=generator)
    full = contributions.mean(dim=0)
    recovered = leave_one_out_mean(full, contributions[3], n=7)
    direct = torch.cat((contributions[:3], contributions[4:])).mean(dim=0)
    assert torch.allclose(recovered, direct, atol=1e-6)


def test_position_weighted_mean_does_not_satisfy_equal_prompt_identity():
    from jspace_phase4.experiments.p4_qwen_lens_influence import (
        leave_one_out_mean,
    )
    first = torch.tensor([[1.0]])
    second = torch.tensor([[5.0]])
    equal_prompt = (first + second) / 2
    position_weighted = (2 * first + 8 * second) / 10
    assert torch.allclose(leave_one_out_mean(equal_prompt, second, n=2), first)
    assert not torch.allclose(
        leave_one_out_mean(position_weighted, second, n=2), first)


def test_adjacent_running_sum_delta_matches_contribution_block():
    from jspace_phase4.experiments.p4_qwen_lens_influence import (
        adjacent_sum_from_running_sums,
    )
    generator = torch.Generator().manual_seed(43)
    contributions = torch.randn((9, 4, 4), generator=generator)
    earlier = contributions[:6].sum(dim=0)
    later = contributions.sum(dim=0)
    assert torch.allclose(
        adjacent_sum_from_running_sums(earlier, later),
        contributions[6:].sum(dim=0),
        atol=1e-6,
    )


def test_adjacent_contract_comparison_passes_exact_block():
    from jspace_phase4.experiments.p4_qwen_lens_influence import (
        compare_adjacent_contract,
    )
    generator = torch.Generator().manual_seed(47)
    before = torch.randn((5, 5), generator=generator)
    block = torch.randn((5, 5), generator=generator)
    earlier = {"jacobian_sum": {0: before}}
    later = {"jacobian_sum": {0: before + block}}
    rows, passed = compare_adjacent_contract(
        earlier, later, {0: block}, source_layers=[0],
        atol=1e-6, rtol=1e-6)
    assert passed
    assert rows[0]["allclose_pass"]
    assert rows[0]["relative_frobenius_error"] < 1e-6


def test_prompt112_config_can_never_replace_canonical_lens():
    from jspace_phase4.experiments.p4_qwen_lens_influence import (
        validate_influence_config,
    )
    config = {
        "tier": "phase4-development",
        "canonical_lens_unchanged": True,
        "prompt": {
            "one_based_index": 112,
            "zero_based_index": 111,
            "logged_max_jacobian_norm_over_sqrt_d": 159.952,
            "recomputed_norm_absolute_tolerance": 0.5,
        },
        "adjacent_checkpoint_contract": {
            "earlier": {"n": 195},
            "later": {"n": 198},
            "prompt_indices_one_based": [196, 197, 198],
        },
    }
    validate_influence_config(config)
    config["canonical_lens_unchanged"] = False
    try:
        validate_influence_config(config)
    except RuntimeError as error:
        assert "canonical" in str(error)
    else:
        raise AssertionError("canonical lens replacement was accepted")


def test_prompt_influence_refuses_non_development_tier():
    from jspace_phase4.experiments.p4_qwen_lens_influence import (
        validate_influence_config,
    )
    config = {
        "tier": "phase4-confirmatory",
        "canonical_lens_unchanged": True,
        "prompt": {
            "one_based_index": 112,
            "zero_based_index": 111,
            "logged_max_jacobian_norm_over_sqrt_d": 159.952,
            "recomputed_norm_absolute_tolerance": 0.5,
        },
        "adjacent_checkpoint_contract": {
            "earlier": {"n": 195},
            "later": {"n": 198},
            "prompt_indices_one_based": [196, 197, 198],
        },
    }
    try:
        validate_influence_config(config)
    except RuntimeError as error:
        assert "development sensitivity" in str(error)
    else:
        raise AssertionError("confirmatory influence tier was accepted")


def test_prompt_influence_norm_tolerance_is_bounded():
    from jspace_phase4.experiments.p4_qwen_lens_influence import (
        validate_influence_config,
    )
    config = {
        "tier": "phase4-development",
        "canonical_lens_unchanged": True,
        "prompt": {
            "one_based_index": 112,
            "zero_based_index": 111,
            "logged_max_jacobian_norm_over_sqrt_d": 159.952,
            "recomputed_norm_absolute_tolerance": 0.5,
        },
        "adjacent_checkpoint_contract": {
            "earlier": {"n": 195},
            "later": {"n": 198},
            "prompt_indices_one_based": [196, 197, 198],
        },
    }
    validate_influence_config(config)
    config["prompt"]["recomputed_norm_absolute_tolerance"] = 1.0
    try:
        validate_influence_config(config)
    except RuntimeError as error:
        assert "0.5%" in str(error)
    else:
        raise AssertionError("unbounded norm tolerance was accepted")
