import numpy as np
import pandas as pd
import pytest
import torch

from jspace_phase4.interventions import (
    AddedProtectionProfile,
    GeometryTolerance,
    exact_rank_energy_matched_subspace,
    geometry_match_report,
    substitution_endpoint,
)
from jspace_phase4.stats4 import (
    exact_signflip,
    family_bootstrap_percentile,
    high_minus_low_family_statistic,
    monte_carlo_pvalue,
)


def test_plus_one_monte_carlo_never_prints_zero():
    null = np.zeros(99)
    assert monte_carlo_pvalue(
        null, 1.0, alternative="greater") == pytest.approx(0.01)


def test_exact_signflip_matches_brute_force():
    values = np.array([-1.2, -0.8, -0.4, 0.1, -0.6])
    result = exact_signflip(values)
    brute = []
    for bits in range(2**len(values)):
        signs = 1 - 2 * ((bits >> np.arange(len(values))) & 1)
        brute.append(float((values * signs).mean()))
    expected = np.mean(np.abs(brute) >= abs(values.mean()) - 1e-15)
    assert result["p"] == expected
    assert result["method"] == "exact-family-signflip"


def test_bootstrap_interval_is_correctly_named_and_deterministic():
    frame = pd.DataFrame({
        "canonical_family": [f"f{i}" for i in range(8) for _ in range(3)],
        "value": np.linspace(-1.0, 1.0, 24),
    })
    first = family_bootstrap_percentile(
        frame, "value", draws=2000, seed=7)
    second = family_bootstrap_percentile(
        frame, "value", draws=2000, seed=7)
    assert first["method"] == "family-resampling-percentile"
    assert first == second
    assert first["ci95"][0] <= first["estimate"] <= first["ci95"][1]


def test_frozen_high_minus_low_family_statistic():
    rows = pd.DataFrame({
        "canonical_family": ["a", "a", "b", "b"],
        "load": [1, 4, 1, 4],
        "effect": [0.0, 1.0, 0.5, 2.0],
    })
    result = high_minus_low_family_statistic(
        rows, value_col="effect", low=1, high=4)
    assert result.to_dict() == {"a": 1.0, "b": 1.5}


def test_exact_matched_control_mechanics():
    generator = torch.Generator().manual_seed(2)
    hidden = torch.randn(32, generator=generator)
    protected = torch.randn(2, 32, generator=generator)
    basis, report = exact_rank_energy_matched_subspace(
        hidden,
        rank=4,
        energy_fraction=0.08,
        protected_rows=protected,
        seed=19,
    )
    assert basis.shape == (32, 4)
    assert report["mechanical_gate_passed"]
    assert report["effective_rank"] == 4
    assert report["protected_projector_overlap"] < 5e-4


def test_bridge_geometry_match_is_per_field_and_hard_gated():
    target = AddedProtectionProfile(2, 2, 0.10, 0.20, 0.30, 1.0, 2.0)
    candidate = AddedProtectionProfile(
        2, 2, 0.11, 0.21, 0.35, 1.1, 2.1)
    report = geometry_match_report(
        target, candidate, GeometryTolerance())
    assert report["ok"]
    bad = AddedProtectionProfile(
        3, 2, 0.11, 0.21, 0.35, 1.1, 2.1)
    assert not geometry_match_report(
        target, bad, GeometryTolerance())["ok"]


def test_counterfactual_endpoint_reports_absolute_calibration():
    result = substitution_endpoint(
        {"original_lp": -8.0, "counterfactual_lp": -2.0},
        {"original_lp": -4.0, "counterfactual_lp": -5.0},
    )
    assert result["counterfactual_preference"] == 6.0
    assert result["unrelated_preference"] == -1.0
    assert result["substitution_effect"] == 7.0
    assert "absolute_calibration" in result
