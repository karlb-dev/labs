# Phase 2 behavioral report — olmo7b

PC gate: FAIL (parse 1.0000, expected 0.731, wrong branches 0). NC alarm: FIRED. Margin floor (effective): 0.964 nats; NC f1-f2 p95 0.4818; null-ladder slope -0.0000 (p 1.000).

## Neutral semantic margins (F1, Holm-12, exact sign-flip)

| scenario | margin (nats) | 95% CI | p_holm | first-token | status |
|---|---|---|---|---|---|
| arb_component | +1.002 | [+0.703, +1.314] | 0 | +1.002 | INSTRUMENT_FAILURE |
| arb_docsection | +0.504 | [+0.301, +0.718] | 0 | +0.506 | INSTRUMENT_FAILURE |
| arb_execmode | +1.300 | [+1.090, +1.505] | 0 | +1.300 | INSTRUMENT_FAILURE |
| arb_lint | +1.154 | [+0.854, +1.467] | 0 | +1.154 | INSTRUMENT_FAILURE |
| arb_naming | +0.273 | [+0.001, +0.537] | 0 | +0.273 | INSTRUMENT_FAILURE |
| arb_notes | -0.090 | [-0.294, +0.114] | 0.0034 | -0.090 | INSTRUMENT_FAILURE |
| arb_seed | +0.007 | [-0.240, +0.260] | 0.6983 | +0.005 | INSTRUMENT_FAILURE |
| arb_setup | +1.153 | [+0.904, +1.475] | 0 | +1.153 | INSTRUMENT_FAILURE |
| arb_shard | +0.916 | [+0.615, +1.236] | 0 | +0.916 | INSTRUMENT_FAILURE |
| arb_storage | +0.194 | [-0.007, +0.396] | 0 | +0.194 | INSTRUMENT_FAILURE |
| arb_testorder | +0.626 | [+0.329, +0.928] | 0 | +0.626 | INSTRUMENT_FAILURE |
| arb_traversal | +0.801 | [+0.537, +1.074] | 0 | +0.801 | INSTRUMENT_FAILURE |

## Strict enacted choice (F2, conditional on F1)

| scenario | effect | 95% CI | valid rate | passes |
|---|---|---|---|---|
| arb_component | +0.031 | [-0.021, +0.081] | 1.0000 | True |
| arb_docsection | +0.062 | [+0.013, +0.119] | 1.0000 | True |
| arb_execmode | +0.178 | [+0.138, +0.244] | 1.0000 | True |
| arb_lint | +0.143 | [+0.089, +0.195] | 1.0000 | True |
| arb_naming | +0.052 | [-0.003, +0.107] | 1.0000 | True |
| arb_notes | -0.026 | [-0.081, +0.029] | 1.0000 | True |
| arb_seed | -0.013 | [-0.065, +0.039] | 1.0000 | True |
| arb_setup | +0.092 | [+0.055, +0.186] | 1.0000 | True |
| arb_shard | +0.182 | [+0.130, +0.232] | 1.0000 | True |
| arb_storage | +0.016 | [-0.036, +0.068] | 1.0000 | True |
| arb_testorder | +0.042 | [-0.013, +0.096] | 1.0000 | True |
| arb_traversal | +0.122 | [+0.073, +0.172] | 1.0000 | True |

## Context ladders (F3, Holm-3)

| anchor | slope (nats/unit) | 95% CI | holdout rank r | neutral intercept | choice crossing | passes |
|---|---|---|---|---|---|---|
| mech_component | +0.093 | [+0.079, +0.108] | +0.80 | +0.007 | +nan | True |
| mech_docsection | +0.032 | [+0.019, +0.045] | +0.90 | +0.672 | +nan | True |
| mech_execmode | +0.087 | [+0.038, +0.141] | +1.00 | +0.746 | +nan | True |

---
*Claim ceiling (plan §8): every statement above is about functional choice, semantic decision margins, contextual relative advantage, enacted branches, report-only selection, scenario-local causal handles, and functional choice/report coupling under this battery. No statement licenses mental-state language; the forbidden upgrade list is enforced by the raising language wall. License: agent_dual_code_provisional pending PI ratings.*
