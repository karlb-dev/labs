import itertools

import numpy as np


def _t(values):
    values = np.asarray(values, dtype=float)
    return values.mean(axis=0) / (
        values.std(axis=0, ddof=1) / np.sqrt(len(values)))


def test_shared_family_max_t_matches_bruteforce_exact_enumeration():
    from jspace_phase4.experiments.p4_bank_w_power import shared_family_max_t

    values = np.array([
        [0.7, -0.2, 0.1], [0.1, 0.4, -0.3], [0.6, 0.2, 0.5],
        [-0.1, 0.3, 0.2], [0.5, -0.4, 0.6],
    ])
    result = shared_family_max_t(
        values, model_slugs=["a", "b", "c"], exact_max_families=20)
    observed = _t(values).max()
    null = []
    for signs in itertools.product([1, -1], repeat=len(values)):
        null.append(_t(values * np.asarray(signs)[:, None]).max())
    expected = np.mean(np.asarray(null) >= observed - 1e-14)
    assert result["method"] == "exact-shared-family-signflip-max-t"
    assert result["n_patterns_or_draws"] == 2 ** len(values)
    assert result["observed_max_t"] == observed
    assert result["p"] == expected
    assert result["model_slugs_in_frozen_order"] == ["a", "b", "c"]
    assert not result["model_selection_from_intervention_outcomes"]


def test_batch_max_t_uses_one_sign_per_family_for_every_model():
    from jspace_phase4.experiments.p4_bank_w_power import (
        batch_max_t_pvalues,
    )

    rng = np.random.default_rng(4)
    cube = rng.normal(size=(3, 8, 3))
    signs = np.where(
        np.arange(16)[:, None] & (1 << np.arange(8)), -1, 1)
    first = batch_max_t_pvalues(cube, signs, permutation_chunk_size=5)
    second = batch_max_t_pvalues(cube, signs, permutation_chunk_size=16)
    np.testing.assert_array_equal(first, second)
    assert np.all((first > 0) & (first <= 1))


def test_max_t_rejects_outcome_selected_or_incomplete_model_columns():
    from jspace_phase4.experiments.p4_bank_w_power import shared_family_max_t

    values = np.arange(18, dtype=float).reshape(6, 3)
    with np.testing.assert_raises(ValueError):
        shared_family_max_t(values, model_slugs=["picked-after-outcome"])
    with np.testing.assert_raises(ValueError):
        shared_family_max_t(values[:, :2], model_slugs=["same", "same"])


def test_max_t_null_simulation_controls_type_i_error():
    from jspace_phase4.experiments.p4_bank_w_power import (
        _random_signs,
        simulate_rejection_rate,
    )

    result = simulate_rejection_rate(
        n_simulations=400, n_families=12, effects=np.zeros(3),
        family_sd=0.23,
        correlation=np.array([
            [1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]),
        distribution="student_t", heavy_tail_df=5,
        signs=_random_signs(1024, 12, 991), seed=1776, alpha=0.05,
        simulation_batch_size=40, permutation_chunk_size=256)
    assert 0.02 <= result["rejection_rate"] <= 0.08
    assert result["wilson_ci95"][1] <= 0.08
