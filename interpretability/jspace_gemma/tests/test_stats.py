import numpy as np

from jspace_gemma.stats import prompt_bootstrap, robust_floor_curvature_fit


def test_robust_floor_curvature_fit_resists_one_outlier():
    epsilon = np.array([0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    error = 0.03 + 1.7 * epsilon
    error[4] += 2.0
    fit = robust_floor_curvature_fit(epsilon, error)
    assert abs(fit["intercept_a"] - 0.03) < 0.03
    assert abs(fit["slope_b"] - 1.7) < 0.3


def test_prompt_bootstrap_resamples_prompts_not_direction_rows():
    result = prompt_bootstrap(
        {"p1": [1.0, 3.0], "p2": [5.0, 7.0], "p3": [9.0, 11.0]},
        draws=500,
        seed=12,
    )
    assert result["n_prompts"] == 3
    assert result["estimate"] == 6.0
