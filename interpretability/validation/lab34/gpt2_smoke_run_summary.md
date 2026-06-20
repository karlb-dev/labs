# Lab 34 run summary: tool use, agents, and state tracking

## Run identity

- model: `gpt2`
- data: `tool_use_tasks.jsonl` sha256 `32351cb35b034c2d`
- selected rows: 28 from 84
- required tools: `{'calculator': 4, 'calendar': 4, 'dictionary': 4, 'file_search': 4, 'none': 4, 'route_planner': 4, 'unit_converter': 4}`
- splits: `{'eval': 14, 'train': 14}`
- science-ready: `False`
- intervention: activation addition on a constrained action-letter prompt
- evidence: `OBS + DECODE + CAUSAL + SELF-REPORT`, scoped to toy tools

## Headline verdicts

| method | rung | metric | value | control | posture |
|---|---|---|---:|---:|---|
| `prompt_boundary_tool_needed_decode` | DECODE | tool_needed_auc | 1.0 | 0.3571 | `supported` |
| `prompt_boundary_tool_selection_decode` | DECODE | tool_selection_accuracy | 0.4286 | 0.3571 | `supported` |
| `constrained_action_letter_activation_addition` | CAUSAL | target_direction_shift_at_scale_1 | -0.0155 | -0.0011 | `random_or_letter_prompt_limited` |
| `deterministic_local_tool_trace` | OBS+AUDIT | result_match_rate | 1.0 | 1.0 | `trace_validated` |
| `tool_self_report_review_scaffold` | SELF-REPORT | requires_human_review_rate | 1.0 | 0.0 | `review_required_not_introspection` |

## Reading order

1. `method_card.md` for the claim boundary.
2. `olmo3_7b_full_safety_status.json` and `olmo3_7b_full_self_check_status.json` for guardrails.
3. `olmo3_7b_full_surface_cue_audit.csv` before any decode claim.
4. `olmo3_7b_full_tool_depth_selection.csv` and `olmo3_7b_full_tool_choice_probe_report.csv` for split-aware decode.
5. `olmo3_7b_full_tool_task_manifest.csv` and `olmo3_7b_full_tool_confusion_matrix.csv` for row-level failures.
6. `olmo3_7b_full_tool_intervention_summary.csv` for the causal letter-prompt test.
7. `olmo3_7b_full_tool_trace_log.csv` and `olmo3_7b_full_tool_self_report_labels.csv` before source-attribution language.
8. `olmo3_7b_full_tool_counterexamples.csv` before writing a positive claim.

## Smallest surviving claim

The run can support only the rows marked supported in `olmo3_7b_full_tool_use_evidence_matrix.csv`, and only for this toy harness, dataset, model, selected depth, and controls.

## Non-claims

This run does not show persistent goals, autonomous planning, real-world agent reliability, or faithful introspection.
