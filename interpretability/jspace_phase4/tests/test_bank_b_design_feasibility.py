def test_optimistic_bank_b_bound_is_monotone_and_still_infeasible():
    from jspace_phase4.experiments.p4_bank_b_design_feasibility import (
        optimistic_mde,
        optimistic_minimum_families,
    )
    candidate = optimistic_minimum_families(
        effect=0.25, family_sd=6.0, alpha=0.05, power=0.80)
    larger = optimistic_minimum_families(
        effect=1.0, family_sd=6.0, alpha=0.05, power=0.80)
    assert candidate > 3500
    assert larger < candidate
    assert optimistic_mde(
        n_families=40, family_sd=6.0, alpha=0.05, power=0.80) > 2.0


def test_optimistic_bank_b_bound_refuses_invalid_inputs():
    from jspace_phase4.experiments.p4_bank_b_design_feasibility import (
        optimistic_minimum_families,
    )
    for kwargs in (
        {"effect": 0, "family_sd": 6, "alpha": 0.05, "power": 0.8},
        {"effect": 1, "family_sd": 0, "alpha": 0.05, "power": 0.8},
        {"effect": 1, "family_sd": 6, "alpha": 0.8, "power": 0.8},
    ):
        try:
            optimistic_minimum_families(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid feasibility input was accepted")
