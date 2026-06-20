# Lab 34 method card

This lab studies controlled toy-tool traces. It does not claim persistent goals or autonomous plans.

- model: `gpt2`
- selected residual depth: `3`
- steering layer for activation addition: `2`
- data source: `frozen_jsonl`
- science-ready: `False`
- tools: calculator, dictionary, calendar, file_search, route_planner, unit_converter, none
- decode object: final-token prompt-boundary residual state
- causal object: constrained A/B/C/D/E/F/N action-letter prompt
- main null: surface cues explain tool choice
- self-report labels: known-trace review templates, not model introspection
- forbidden claim: the model has a persistent goal or autonomous plan

## Headline metrics

- tool-needed eval AUC: `1.0`
- tool-selection eval accuracy: `0.4286`
- surface-control eval accuracy: `0.3571`
- causal shift over random: `-0.0144`
- counterexamples: `40`

## Evidence rows

| method | rung | value | control | posture |
|---|---|---:|---:|---|
| `prompt_boundary_tool_needed_decode` | DECODE | 1.0 | 0.3571 | `supported` |
| `prompt_boundary_tool_selection_decode` | DECODE | 0.4286 | 0.3571 | `supported` |
| `constrained_action_letter_activation_addition` | CAUSAL | -0.0155 | -0.0011 | `random_or_letter_prompt_limited` |
| `deterministic_local_tool_trace` | OBS+AUDIT | 1.0 | 1.0 | `trace_validated` |
| `tool_self_report_review_scaffold` | SELF-REPORT | 1.0 | 0.0 | `review_required_not_introspection` |

## Claim boundary

A positive row supports a toy-harness signal claim only. It does not support intention, planning, real-world tool reliability, or faithful self-report.
