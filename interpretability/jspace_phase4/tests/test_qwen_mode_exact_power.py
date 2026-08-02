import itertools
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_exact_mode_signflip_pvalue_matches_bruteforce_mean_test():
    from jspace_phase4.experiments.p4_qwen_mode_exact_power import (
        batch_signflip_pvalues,
        sign_matrix,
    )

    values = np.asarray([
        [1.0, -0.5, 0.25, 1.5],
        [0.5, 0.5, -1.0, 0.25],
    ])
    signs, exact = sign_matrix(
        4, draws=10, seed=7, exact_max_families=20)
    assert exact
    brute_signs = np.asarray(list(itertools.product([1, -1], repeat=4)))
    expected = []
    for row in values:
        null = brute_signs @ row
        expected.append(np.mean(null >= row.sum() - 1e-13))
    actual = batch_signflip_pvalues(
        values, signs, exact=True, chunk_size=3)
    assert np.allclose(actual, expected)


def test_mode_power_calibration_is_invariant_to_signed_pilot_mean():
    from jspace_phase4.experiments.p4_qwen_mode_exact_power import (
        symmetrized_residual_calibration,
    )

    values = np.asarray([-2.0, -1.0, 0.0, 0.0, 1.0, 2.0])
    first = symmetrized_residual_calibration(values, planning_sd=1.7)
    second = symmetrized_residual_calibration(values + 93.25, planning_sd=1.7)
    assert first["signed_pilot_mean_retained"] is False
    assert first["signed_family_interactions_retained"] is False
    assert first["scaled_residual_magnitudes_sha256"] == second[
        "scaled_residual_magnitudes_sha256"]
    assert np.allclose(
        first["scaled_residual_magnitudes"],
        second["scaled_residual_magnitudes"])
    assert np.isclose(
        np.sqrt(np.mean(first["scaled_residual_magnitudes"] ** 2)), 1.7)


def test_monte_carlo_mode_signflip_uses_plus_one_and_is_deterministic():
    from jspace_phase4.experiments.p4_qwen_mode_exact_power import (
        batch_signflip_pvalues,
        sign_matrix,
    )

    values = np.full((3, 24), 2.0)
    first, exact = sign_matrix(
        24, draws=257, seed=17, exact_max_families=20)
    second, second_exact = sign_matrix(
        24, draws=257, seed=17, exact_max_families=20)
    assert not exact and not second_exact
    assert np.array_equal(first, second)
    pvalues = batch_signflip_pvalues(
        values, first, exact=False, chunk_size=31)
    assert np.all(pvalues >= 1 / 258)
    assert np.all(pvalues <= 1)


def test_exact_mode_signflip_refuses_an_incomplete_pattern_matrix():
    from jspace_phase4.experiments.p4_qwen_mode_exact_power import (
        batch_signflip_pvalues,
    )

    with pytest.raises(ValueError, match="complete pattern count"):
        batch_signflip_pvalues(
            np.ones((1, 4)), np.ones((15, 4)), exact=True)


def test_mode_signflip_power_increases_for_large_fixed_effect():
    from jspace_phase4.experiments.p4_qwen_mode_exact_power import (
        sign_matrix,
        simulate_rejection_rate,
    )

    magnitudes = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
    signs, exact = sign_matrix(
        24, draws=512, seed=23, exact_max_families=20)
    null = simulate_rejection_rate(
        residual_magnitudes=magnitudes, n_families=24, effect=0.0,
        n_simulations=500, signs=signs, exact=exact, seed=29,
        alpha=0.05, simulation_batch_size=50, sign_chunk_size=128)
    powered = simulate_rejection_rate(
        residual_magnitudes=magnitudes, n_families=24, effect=1.5,
        n_simulations=500, signs=signs, exact=exact, seed=31,
        alpha=0.05, simulation_batch_size=50, sign_chunk_size=128)
    assert null["sign_method"] == "monte-carlo-family-signflip-plus-one"
    assert powered["rejection_rate"] > null["rejection_rate"] + 0.8


def test_mode_power_contract_is_prospective_and_does_not_license_split():
    config = yaml.safe_load((
        ROOT / "configs/p4_qwen_mode_exact_power_dev.yaml").read_text())
    assert config["source_pilot"]["result_sha256"].startswith("BIND_")
    assert config["prospective_sesoi"]["accuracy_points"] == 0.20
    assert config["prospective_sesoi"][
        "pilot_mean_may_not_select_or_revise_sesoi"] is True
    primary = config["primary_randomization"]
    assert primary["alpha"] == primary["familywise_alpha"] / primary[
        "conservative_holm_primary_count"]
    assert primary["exact_max_families"] == 20
    assert config["simulation"]["never_report_or_use_signed_pilot_mean"] is True
    assert config["simulation"]["require_wilson_lower_bound_at_target"] is True
    assert config["continuous_answer_lp_sensitivity"][
        "no_power_or_primary_authorization"] is True
    assert "does not authorize" in config["claim_boundary"]
