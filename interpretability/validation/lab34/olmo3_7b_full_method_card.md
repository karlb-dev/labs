# Lab 34 method card

This lab studies controlled toy-tool traces. It does not claim persistent goals or autonomous plans.

- model: `allenai/Olmo-3-1025-7B`
- selected residual depth: `28`
- steering layer for activation addition: `27`
- data source: `frozen_jsonl`
- science-ready: `True`
- tools: calculator, dictionary, calendar, file_search, route_planner, unit_converter, none
- decode object: final-token prompt-boundary residual state
- causal object: constrained A/B/C/D/E/F/N action-letter prompt
- main null: surface cues explain tool choice
- self-report labels: known-trace review templates, not model introspection
- forbidden claim: the model has a persistent goal or autonomous plan

## Headline metrics

- tool-needed eval AUC: `0.9236`
- tool-selection eval accuracy: `0.7692`
- surface-control eval accuracy: `0.3077`
- causal shift over random: `0.112`
- counterexamples: `40`

## Evidence rows

| method | rung | value | control | posture |
|---|---|---:|---:|---|
| `prompt_boundary_tool_needed_decode` | DECODE | 0.9236 | 0.3077 | `supported` |
| `prompt_boundary_tool_selection_decode` | DECODE | 0.7692 | 0.3077 | `supported` |
| `constrained_action_letter_activation_addition` | CAUSAL | 0.0913 | -0.0207 | `supported_narrow_letter_prompt` |
| `deterministic_local_tool_trace` | OBS+AUDIT | 1.0 | 1.0 | `trace_validated` |
| `tool_self_report_review_scaffold` | SELF-REPORT | 1.0 | 0.0 | `review_required_not_introspection` |

## Claim boundary

A positive row supports a toy-harness signal claim only. It does not support intention, planning, real-world tool reliability, or faithful self-report.
