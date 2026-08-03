import numpy as np
from jspace_olmo_lineage.experiments.bank_w_pair_power import (
    batch_max_t_pvalues,
    pair_decision,
    shared_family_max_t,
)


def test_pair_max_t_detects_shared_positive_signal():
    generator = np.random.default_rng(91)
    values = generator.normal(0.8, 0.1, size=(12, 2))
    result = shared_family_max_t(
        values, model_slugs=["think", "instruct"], exact_max_families=20
    )
    assert result["n_models_jointly_tested"] == 2
    assert result["method"] == "exact-shared-family-signflip-max-t"
    assert result["p"] < 0.01


def test_batch_max_t_keeps_shared_family_pairing():
    generator = np.random.default_rng(7)
    cube = generator.normal(size=(8, 10, 2))
    signs = 2 * generator.integers(0, 2, size=(128, 10)) - 1
    pvalues = batch_max_t_pvalues(cube, signs)
    assert pvalues.shape == (8,)
    assert np.all((pvalues > 0) & (pvalues <= 1))


def test_pair_route_requires_powered_current_support():
    powered = pair_decision(
        all_independently_eligible=True,
        all_type_i_pass=True,
        shared_families=20,
        minimum_powered_families=19,
        power_at_shared_support=0.82,
        power_target=0.80,
    )
    assert powered["future_pair_worthwhile_at_current_support"] is True
    assert powered["intervention_authorized"] is False

    underpowered = pair_decision(
        all_independently_eligible=True,
        all_type_i_pass=True,
        shared_families=17,
        minimum_powered_families=19,
        power_at_shared_support=0.77,
        power_target=0.80,
    )
    assert underpowered["route"] == "not-powered-at-current-support"
    assert underpowered["intervention_authorized"] is False
