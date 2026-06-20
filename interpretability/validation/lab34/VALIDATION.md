# Lab 34 Validation

## Lab 34: Tool Use, Agents, and State Tracking

This directory is the final Lab 34 validation pack for the current repository
code. It replaces the older June 15 validation artifacts with the repaired
June 20, 2026 tool-use runs.

## Validation Read

The result is a narrow positive for the toy-tool harness. The repaired full
Olmo-3-1025-7B run supports prompt-boundary tool-need decoding, which-tool
decoding, deterministic local trace validation, and a constrained action-letter
activation-addition effect. The claim boundary is important: this is not
evidence of persistent goals, autonomous planning, real-world agent reliability,
or faithful introspection.

The cleanest current claim is:

```text
Lab 34 validates a toy-harness tool-use state signal: tool need and tool choice
are decodable above surface controls, and a constrained action-letter
intervention shifts logits above a random-direction control. It does not prove
autonomous planning or reliable real-world tool use.
```

## Headline Result

The primary validation run is `olmo3_7b_full`:

- Model: `allenai/Olmo-3-1025-7B`
- Data: 84 frozen toy-tool tasks
- Tools: calculator, dictionary, calendar, file search, route planner, unit
  converter, and no-tool controls
- Train/eval split: 58 / 26
- Selected residual depth: 28
- Steering layer: 27
- Tool-needed eval AUC: 0.9236
- Tool-needed eval accuracy: 0.8846
- Tool-selection eval accuracy: 0.7692
- Surface-control accuracy: 0.3077
- Decode gap over surface: 0.4615
- Target action-letter shift at scale 1: 0.0913
- Random-direction shift at scale 1: -0.0207
- Causal shift over random: 0.1120
- Argument-valid rate: 1.0
- Trace match rate: 1.0

The tier-A GPT-2 smoke run supports only the smoke-check version of the decode
story. It does not support the causal action-letter claim.

## Current Result Summary

| Source label | Model | Main result | Read |
|---|---|---|---|
| `olmo3_7b_full` | Olmo-3-1025-7B | tool-needed AUC 0.9236, tool-selection accuracy 0.7692, causal shift gap 0.1120 | Final validation run; narrow positive |
| `gpt2_smoke` | GPT-2 | tool-needed AUC 1.0, tool-selection accuracy 0.4286, causal shift gap -0.0144 | Smoke check; causal claim does not survive |

The broader run ledger is in `lab34_validation_summary.csv`.

## What This Lab Teaches

- Tool-use state can be studied with a controlled toy harness without
  overclaiming about agents.
- Surface-cue controls matter: no-tool rows with tool words are the guardrail
  against reading prompt artifacts as intention.
- The causal claim is deliberately narrow because the intervention is an
  action-letter prompt, not free-form tool use.
- Self-report rows are review scaffolds, not evidence that the model faithfully
  knows why it used a tool.

## Curated Artifacts

Summary:

- `lab34_validation_report.md`
- `lab34_validation_summary.csv`

Primary Olmo run:

- `olmo3_7b_full_run_summary.md`
- `olmo3_7b_full_method_card.md`
- `olmo3_7b_full_operationalization_audit.md`
- `olmo3_7b_full_ledger_suggestions.md`
- `olmo3_7b_full_metrics.json`
- `olmo3_7b_full_results.csv`
- `olmo3_7b_full_tool_use_evidence_matrix.csv`
- `olmo3_7b_full_tool_choice_probe_report.csv`
- `olmo3_7b_full_tool_depth_selection.csv`
- `olmo3_7b_full_tool_confusion_matrix.csv`
- `olmo3_7b_full_surface_cue_audit.csv`
- `olmo3_7b_full_tool_counterexamples.csv`
- `olmo3_7b_full_tool_intervention_summary.csv`
- `olmo3_7b_full_tool_self_report_labels.csv`
- `olmo3_7b_full_tool_state_transition_log.csv`
- `olmo3_7b_full_tool_trace_log.csv`
- `olmo3_7b_full_tool_task_manifest.csv`
- `olmo3_7b_full_failure_specimens.md`
- `olmo3_7b_full_safety_status.json`
- `olmo3_7b_full_self_check_status.json`
- `olmo3_7b_full_prompt_boundary_audit.csv`
- `olmo3_7b_full_action_letter_token_gate.csv`
- `olmo3_7b_full_tool_argument_validation.csv`

Auxiliary check:

- `gpt2_smoke_*`

## Caveats

- This is a curated validation directory; full raw runs remain outside this
  pack.
- No repaired-run PNG plots were present in the local run bundle, so this pack
  keeps tables and reports rather than carrying forward old plots.
- The positive result is scoped to toy tools, fixed prompts, selected depth,
  and named controls.
