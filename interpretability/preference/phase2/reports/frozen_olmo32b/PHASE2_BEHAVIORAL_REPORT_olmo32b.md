# Phase 2 behavioral report — olmo32b

PC gate: PASS (parse 1.0000, expected 0.942, wrong branches 0). NC alarm: quiet. Margin floor (effective): 0.833 nats; NC f1-f2 p95 0.4167; null-ladder slope -0.0000 (p 1.000).

## Neutral semantic margins (F1, Holm-12, exact sign-flip)

| scenario | margin (nats) | 95% CI | p_holm | first-token | status |
|---|---|---|---|---|---|
| arb_component | +0.532 | [+0.394, +0.670] | 0 | +0.532 | CLEAN_NULL |
| arb_docsection | +0.582 | [+0.404, +0.772] | 0 | +0.582 | CLEAN_NULL |
| arb_execmode | +1.176 | [+1.046, +1.546] | 0 | +1.176 | ENACTED_CHOICE |
| arb_lint | +0.413 | [+0.235, +0.596] | 0 | +0.413 | CLEAN_NULL |
| arb_naming | +0.574 | [+0.362, +0.800] | 0 | +0.574 | CLEAN_NULL |
| arb_notes | -0.079 | [-0.258, +0.104] | 0.0008 | -0.079 | CLEAN_NULL |
| arb_seed | +0.187 | [-0.017, +0.400] | 0 | +0.187 | CLEAN_NULL |
| arb_setup | +0.671 | [+0.569, +0.910] | 0 | +0.671 | CLEAN_NULL |
| arb_shard | +0.289 | [+0.158, +0.422] | 0 | +0.289 | CLEAN_NULL |
| arb_storage | +0.179 | [-0.011, +0.368] | 0.0006 | +0.179 | CLEAN_NULL |
| arb_testorder | +0.481 | [+0.322, +0.652] | 0 | +0.481 | CLEAN_NULL |
| arb_traversal | -0.050 | [-0.188, +0.087] | 0.0083 | -0.050 | CLEAN_NULL |

## Strict enacted choice (F2, conditional on F1)

| scenario | effect | 95% CI | valid rate | passes |
|---|---|---|---|---|
| arb_component | +0.117 | [+0.062, +0.172] | 1.0000 | True |
| arb_docsection | +0.100 | [+0.053, +0.163] | 1.0000 | True |
| arb_execmode | +0.312 | [+0.269, +0.383] | 1.0000 | True |
| arb_lint | +0.052 | [-0.003, +0.104] | 1.0000 | True |
| arb_naming | +0.083 | [+0.031, +0.138] | 1.0000 | True |
| arb_notes | -0.036 | [-0.091, +0.018] | 1.0000 | True |
| arb_seed | +0.021 | [-0.034, +0.073] | 1.0000 | True |
| arb_setup | +0.168 | [+0.119, +0.244] | 1.0000 | True |
| arb_shard | +0.078 | [+0.026, +0.130] | 1.0000 | True |
| arb_storage | +0.055 | [-0.005, +0.115] | 1.0000 | True |
| arb_testorder | +0.104 | [+0.036, +0.172] | 1.0000 | True |
| arb_traversal | -0.018 | [-0.070, +0.034] | 1.0000 | True |

## Context ladders (F3, Holm-3)

| anchor | slope (nats/unit) | 95% CI | holdout rank r | neutral intercept | choice crossing | passes |
|---|---|---|---|---|---|---|
| mech_component | +0.452 | [+0.382, +0.525] | +1.00 | +0.855 | -0.97 | True |
| mech_docsection | +0.239 | [+0.198, +0.282] | +0.90 | +0.775 | +nan | True |
| mech_execmode | +0.565 | [+0.513, +0.619] | +0.90 | +0.240 | -0.32 | True |

---
*Claim ceiling (plan §8): every statement above is about functional choice, semantic decision margins, contextual relative advantage, enacted branches, report-only selection, scenario-local causal handles, and functional choice/report coupling under this battery. No statement licenses mental-state language; the forbidden upgrade list is enforced by the raising language wall. License: agent_dual_code_provisional pending PI ratings.*
