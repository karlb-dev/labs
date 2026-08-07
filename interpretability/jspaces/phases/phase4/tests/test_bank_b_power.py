import itertools

import numpy as np


def _t(values):
    values = np.asarray(values, dtype=float)
    return values.mean(axis=0) / (
        values.std(axis=0, ddof=1) / np.sqrt(len(values)))


def test_bank_b_iut_pvalue_matches_bruteforce_component_maximum():
    from jspace_phase4.experiments.p4_bank_b_power import (
        batch_intersection_union_pvalues,
    )

    values = np.asarray([[
        [0.7, 0.4], [0.1, -0.2], [0.6, 0.5], [-0.1, 0.3],
    ]])
    signs = np.asarray(list(itertools.product([1, -1], repeat=4)))
    observed = _t(values[0])
    null = np.asarray([_t(values[0] * sign[:, None]) for sign in signs])
    component = np.mean(null >= observed[None, :] - 1e-14, axis=0)
    expected = component.max()
    actual = batch_intersection_union_pvalues(
        values, signs, exact=True, permutation_chunk_size=3)
    assert actual.shape == (1,)
    assert actual[0] == expected


def test_bank_b_iut_requires_both_components_to_be_positive():
    from jspace_phase4.experiments.p4_bank_b_power import (
        _signs,
        batch_intersection_union_pvalues,
    )

    rng = np.random.default_rng(11)
    cube = rng.normal(size=(2, 10, 2))
    cube[0] += [3.0, 3.0]
    cube[1] += [3.0, 0.0]
    signs, exact = _signs(10, draws=100, seed=4, exact_max_families=20)
    pvalues = batch_intersection_union_pvalues(
        cube, signs, exact=exact, permutation_chunk_size=128)
    assert pvalues[0] <= 0.05
    assert pvalues[1] > 0.05


def test_bank_b_iut_composite_null_controls_type_i():
    from jspace_phase4.experiments.p4_bank_b_power import (
        _signs,
        simulate_iut_rejection_rate,
    )

    signs, exact = _signs(
        10, draws=1024, seed=17, exact_max_families=20)
    result = simulate_iut_rejection_rate(
        n_simulations=400, n_families=10,
        effects=np.asarray([0.0, 20.0]), family_sd=5.0,
        correlation=0.5, distribution="normal", heavy_tail_df=5,
        signs=signs, exact_signs=exact, seed=19, alpha=0.05,
        simulation_batch_size=40, permutation_chunk_size=128)
    assert 0.02 <= result["rejection_rate"] <= 0.08
