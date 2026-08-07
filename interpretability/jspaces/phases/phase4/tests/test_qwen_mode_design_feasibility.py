import math


def test_mode_accuracy_interaction_matches_frozen_contrast():
    from jspace_phase4.experiments.p4_qwen_mode_design_feasibility import (
        family_accuracy_interaction,
    )
    # Only thinking-on final J is damaged: control - J = 1.
    assert family_accuracy_interaction(
        [1, 0, 1, 1, 1, 1, 1, 1]) == 1.0
    # The same damage in both modes cancels.
    assert family_accuracy_interaction(
        [1, 0, 1, 1, 1, 0, 1, 1]) == 0.0


def test_mode_interaction_support_implies_sd_bound_four():
    from jspace_phase4.experiments.p4_qwen_mode_design_feasibility import (
        binary_interaction_sd_upper_bound,
        family_accuracy_interaction,
    )
    assert family_accuracy_interaction([1, 0, 0, 1, 0, 1, 1, 0]) == 4.0
    assert family_accuracy_interaction([0, 1, 1, 0, 1, 0, 0, 1]) == -4.0
    assert binary_interaction_sd_upper_bound() == 4.0


def test_mode_gaussian_envelope_is_monotone():
    from jspace_phase4.experiments.p4_qwen_mode_design_feasibility import (
        gaussian_mde,
        gaussian_minimum_families,
    )
    alpha = 0.05 / 3
    small_effect = gaussian_minimum_families(
        effect=0.10, family_sd=1.0, alpha=alpha, power=0.80)
    large_effect = gaussian_minimum_families(
        effect=0.20, family_sd=1.0, alpha=alpha, power=0.80)
    assert small_effect > large_effect > 200
    assert math.isclose(
        gaussian_mde(
            n_families=large_effect, family_sd=1.0,
            alpha=alpha, power=0.80),
        0.20,
        rel_tol=0.01,
    )


def test_mode_sign_flip_resolution_uses_conservative_holm_alpha():
    from jspace_phase4.experiments.p4_qwen_mode_design_feasibility import (
        exact_sign_flip_resolution_families,
    )
    assert exact_sign_flip_resolution_families(0.05 / 3) == 6


def test_mode_feasibility_refuses_invalid_inputs():
    from jspace_phase4.experiments.p4_qwen_mode_design_feasibility import (
        family_accuracy_interaction,
        gaussian_minimum_families,
    )
    for cells in ([0] * 7, [0] * 7 + [1.1]):
        try:
            family_accuracy_interaction(cells)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mode interaction cells were accepted")
    try:
        gaussian_minimum_families(
            effect=0, family_sd=1, alpha=0.05, power=0.8)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid planning effect was accepted")
