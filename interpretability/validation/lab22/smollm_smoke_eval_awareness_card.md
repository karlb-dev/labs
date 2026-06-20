# Lab 22 Eval-Awareness Card

## Scope

This card summarizes an instrument reading: a residual-stream direction separating benchmark/eval framing from natural phrasing of the same benign tasks. It is not a mind-reading certificate for situational awareness.

## Headline

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Rows / groups / formats: 24 / 6 / `['answer_key_check', 'code_test', 'mcq', 'qa_benchmark', 'quality_screen', 'rubric_free_response']`
- Best stream depth: 2
- Injection layer for steering: 1
- Held-out eval-vs-natural AUC: 1.0
- Held-out eval-vs-format-control AUC: 1.0
- Cross-format min / mean AUC: 1.0 / 1.0
- Cross-format min null-adjusted AUC gap: 0.0
- Best surface baseline AUC on eval split: 0.75
- Natural prompt steering marker delta over controls: 0.0

## Verdicts

- Decode verdict: `cross_format_but_null_or_surface_controls_competitive`
- Causal marker verdict: `not_validated_by_marker_controls`
- Surface baseline risk: `False`

## Required Before Ledger Upgrade

1. Read `olmo3_7b_full_s0_projection_by_item.csv` and find at least one matched group where eval beats natural and format-control rows.
2. Read `olmo3_7b_full_s0_surface_baseline_auc.csv`; if boilerplate or choice markers match the probe, write a format-detector claim instead.
3. Hand-label `source-run steering_generations.csv`; marker-only shifts do not earn a causal eval-awareness claim.
4. If using a Think model, inspect `olmo3_7b_full_s0_cot_mentions_eval.csv` as self-report/co-occurrence evidence only.

## Non-claims

- This run does not prove human-like self-awareness.
- This run does not prove deceptive eval gaming.
- This run does not prove the model uses the direction for all benchmark behavior.
