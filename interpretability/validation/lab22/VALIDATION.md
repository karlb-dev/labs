# Lab 22 Validation

## Lab 22: Eval Awareness Without the Fog Machine

This directory is the final Lab 22 validation pack for the current repository
code. It replaces the older June 15 validation artifacts with the repaired
June 20, 2026 eval-awareness runs.

## Validation Read

The result is a strong but cautious decode finding. The repaired Lab 22 v2 runs
find residual directions that separate benchmark/eval framing from matched
natural prompts across multiple formats, but the current controls do not
validate a causal eval-awareness marker or strategic situational awareness.

The cleanest current claim is:

```text
Lab 22 finds an audited eval-context direction. It is useful for teaching
surface controls, cross-format transfer, and claim discipline, but it should
not be described as proof of deception, human-like awareness, or deployment
awareness.
```

## Headline Result

The primary validation run is `olmo3_7b_full_s0`:

- Model: `allenai/Olmo-3-7B-Instruct`
- Corpus: 96 rows, 24 matched groups, 6 formats
- Formats: answer-key checks, code tests, MCQ, QA benchmarks, quality screens,
  rubric free response
- Best stream depth: 9
- Injection layer: 8
- Held-out eval-vs-natural AUC: 0.9306
- Held-out eval-vs-format-control AUC: 0.7917
- Cross-format min / mean AUC: 0.9062 / 0.9635
- Cross-format min control-adjusted AUC gap: -0.0312
- Best surface baseline AUC on eval split: 0.7049
- Natural-prompt steering marker delta over controls: 0.0
- Decode verdict: `cross_format_but_null_or_surface_controls_competitive`
- Causal marker verdict: `not_validated_by_marker_controls`

The seed-1 Olmo run and Olmo-3-7B-Think run repeat the same basic story:
high eval-context decode scores, but no causal marker validation. The earlier
72-row run looked cleaner, but the v2 surface-matched dataset is the final
read because it is broader and more demanding.

## Current Result Summary

| Source label | Model | Main result | Read |
|---|---|---|---|
| `olmo3_7b_full_s0` | Olmo-3-7B-Instruct | AUC 0.9306, cross-format mean 0.9635, causal marker delta 0.0 | Primary final run; cautious decode result |
| `olmo3_7b_full_s1` | Olmo-3-7B-Instruct | AUC 0.9861, cross-format mean 0.9635, causal marker delta 0.0 | Replicates decode, still control-limited |
| `olmo3_7b_think_full` | Olmo-3-7B-Think | AUC 0.9722, cross-format mean 0.9531, causal marker delta 0.0 | Think-model check; not self-awareness evidence |
| `smollm_smoke` | SmolLM2-135M-Instruct | AUC 1.0 on 24-row smoke | Small smoke check only |

The broader run ledger is in `lab22_validation_summary.csv`.

## What This Lab Teaches

- Strong held-out decode can still be weaker than the story people want to
  tell about it.
- Cross-format transfer is necessary but not sufficient; surface and null
  controls can still make the right interpretation cautious.
- Steering evidence has to survive marker, verbosity, refusal, and format
  artifacts before it earns causal language.
- "Eval awareness" is an operational contrast in this lab, not a claim about
  consciousness, deception, goals, or a global belief that the model is being
  evaluated.

## Curated Artifacts

Summary:

- `lab22_validation_report.md`
- `lab22_validation_summary.csv`

Primary Olmo v2 run:

- `olmo3_7b_full_s0_run_summary.md`
- `olmo3_7b_full_s0_eval_awareness_card.md`
- `olmo3_7b_full_s0_operationalization_audit.md`
- `olmo3_7b_full_s0_ledger_suggestions.md`
- `olmo3_7b_full_s0_metrics.json`
- `olmo3_7b_full_s0_results.csv`
- `olmo3_7b_full_s0_eval_awareness_evidence_matrix.csv`
- `olmo3_7b_full_s0_eval_awareness_probe_by_layer.csv`
- `olmo3_7b_full_s0_cross_format_transfer.csv`
- `olmo3_7b_full_s0_format_transfer_summary.csv`
- `olmo3_7b_full_s0_surface_baseline_auc.csv`
- `olmo3_7b_full_s0_surface_confound_audit.csv`
- `olmo3_7b_full_s0_projection_by_item.csv`
- `olmo3_7b_full_s0_projection_by_condition.csv`
- `olmo3_7b_full_s0_steering_operating_points.csv`
- `olmo3_7b_full_s0_steering_behavior_shift.csv`
- `olmo3_7b_full_s0_steering_effect_sizes.csv`
- `olmo3_7b_full_s0_multiturn_eval_trace_summary.csv`
- `olmo3_7b_full_s0_cot_mentions_eval.csv`
- `olmo3_7b_full_s0_lab20_eval_aware_control.csv`
- `olmo3_7b_full_s0_item_eval_signal_atlas.csv`
- `olmo3_7b_full_s0_direction_cosines.csv`
- `olmo3_7b_full_s0_steering_generation_labeling_guide.md`

Auxiliary checks:

- `olmo3_7b_full_s1_*`
- `olmo3_7b_think_full_*`
- `smollm_smoke_*`

## Caveats

- This is a curated validation directory; full raw runs remain outside this
  pack.
- No repaired-run PNG plots were present in the local run bundle, so this pack
  keeps tables and reports rather than carrying forward old plots.
- The validated result is an audited decode handle, not a causal marker of
  strategic eval awareness.
