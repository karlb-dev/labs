import numpy as np
import pytest

from jspace_olmo_lineage.capacity import curve_summary
from jspace_olmo_lineage.experiments.independent_reconstruction import (
    _independent_curve,
    _ordered_pairs,
)


def test_independent_curve_matches_primary_estimator_summary():
    j_gain = np.asarray([
        [8.0, 0.7, 0.5, 0.2],
        [7.0, 0.8, 0.4, 0.1],
        [9.0, 0.6, 0.5, 0.2],
        [6.0, 0.9, 0.4, 0.1],
    ])
    random_gain = np.asarray([
        [1.0, 1.0, 1.0, 1.0],
        [1.1, 1.1, 1.1, 1.1],
        [0.9, 0.9, 0.9, 0.9],
    ])[:, None, :]
    random_gain = np.repeat(random_gain, 4, axis=1)
    j_errors = np.concatenate([
        np.full((4, 1), 20.0),
        20.0 - np.cumsum(j_gain, axis=1),
    ], axis=1)
    random_errors = np.concatenate([
        np.full((3, 4, 1), 20.0),
        20.0 - np.cumsum(random_gain, axis=2),
    ], axis=2)

    independent = _independent_curve(
        j_errors, random_errors, persistence=2)
    production = curve_summary(
        j_errors, random_errors, persistence=2,
        persistence_sensitivity=(1, 2, 3))
    for key, value in independent.items():
        if isinstance(value, float):
            assert value == pytest.approx(production[key], abs=1e-12)
        else:
            assert value == production[key]


def test_ordered_pairs_preserve_registered_lineage_order():
    order = ["base", "think", "31-think", "31-instruct"]
    available = ["31-instruct", "31-think", "base", "think"]
    assert _ordered_pairs(order, available) == [
        ("base", "think"),
        ("base", "31-think"),
        ("base", "31-instruct"),
        ("think", "31-think"),
        ("think", "31-instruct"),
        ("31-think", "31-instruct"),
    ]
    with pytest.raises(ValueError):
        _ordered_pairs(["base", "base"], available)
    with pytest.raises(ValueError):
        _ordered_pairs(order[:-1], available)
