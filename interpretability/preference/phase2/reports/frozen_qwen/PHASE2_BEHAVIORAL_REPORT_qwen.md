# Phase 2 behavioral report — qwen

PC gate: PASS (parse 1.0000, expected 0.990, wrong branches 0). NC alarm: quiet. Margin floor (effective): 1.055 nats; NC f1-f2 p95 0.5273; null-ladder slope -0.0000 (p 1.000).

## Neutral semantic margins (F1, Holm-12, exact sign-flip)

| scenario | margin (nats) | 95% CI | p_holm | first-token | status |
|---|---|---|---|---|---|
| arb_component | +1.920 | [+1.717, +2.134] | 0 | +1.920 | ENACTED_CHOICE |
| arb_docsection | +1.737 | [+1.324, +2.060] | 0 | +1.736 | SEMANTIC_MARGIN |
| arb_execmode | +2.185 | [+2.036, +2.496] | 0 | +2.185 | ENACTED_CHOICE |
| arb_lint | +1.391 | [+1.265, +1.520] | 0 | +1.391 | ENACTED_CHOICE |
| arb_naming | +0.674 | [+0.535, +0.812] | 0 | +0.674 | CLEAN_NULL |
| arb_notes | +0.528 | [+0.373, +0.683] | 0 | +0.528 | CLEAN_NULL |
| arb_seed | +1.199 | [+0.959, +1.436] | 0 | +1.199 | SEMANTIC_MARGIN |
| arb_setup | +1.335 | [+1.164, +1.535] | 0 | +1.335 | ENACTED_CHOICE |
| arb_shard | +1.251 | [+0.966, +1.548] | 0 | +1.251 | SEMANTIC_MARGIN |
| arb_storage | +0.335 | [+0.158, +0.519] | 0 | +0.335 | CLEAN_NULL |
| arb_testorder | +1.635 | [+1.411, +1.858] | 0 | +1.635 | ENACTED_CHOICE |
| arb_traversal | +0.941 | [+0.814, +1.074] | 0 | +0.941 | CLEAN_NULL |

## Strict enacted choice (F2, conditional on F1)

| scenario | effect | 95% CI | valid rate | passes |
|---|---|---|---|---|
| arb_component | +0.398 | [+0.349, +0.443] | 1.0000 | True |
| arb_docsection | +0.014 | [-0.028, +0.071] | 1.0000 | True |
| arb_execmode | +0.435 | [+0.403, +0.479] | 1.0000 | True |
| arb_lint | +0.435 | [+0.393, +0.471] | 1.0000 | True |
| arb_naming | +0.224 | [+0.164, +0.279] | 1.0000 | True |
| arb_notes | +0.091 | [+0.039, +0.143] | 1.0000 | True |
| arb_seed | +0.036 | [-0.016, +0.091] | 1.0000 | True |
| arb_setup | +0.188 | [+0.150, +0.267] | 1.0000 | True |
| arb_shard | +0.003 | [-0.047, +0.052] | 1.0000 | True |
| arb_storage | +0.104 | [+0.047, +0.161] | 1.0000 | True |
| arb_testorder | +0.185 | [+0.120, +0.245] | 1.0000 | True |
| arb_traversal | +0.260 | [+0.208, +0.312] | 1.0000 | True |

## Context ladders (F3, Holm-3)

| anchor | slope (nats/unit) | 95% CI | holdout rank r | neutral intercept | choice crossing | passes |
|---|---|---|---|---|---|---|
| mech_component | +3.509 | [+3.155, +3.840] | +0.90 | +2.377 | -0.02 | True |
| mech_docsection | +0.965 | [+0.857, +1.075] | +0.90 | +1.748 | +0.00 | True |
| mech_execmode | +1.354 | [+1.264, +1.452] | +1.00 | +1.705 | -0.64 | True |

---
*Claim ceiling (plan §8): every statement above is about functional choice, semantic decision margins, contextual relative advantage, enacted branches, report-only selection, scenario-local causal handles, and functional choice/report coupling under this battery. No statement licenses mental-state language; the forbidden upgrade list is enforced by the raising language wall. License: agent_dual_code_provisional pending PI ratings.*
