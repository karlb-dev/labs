# Lab 25 Operationalization Audit

## What the lab measures

Whether self-report text covaries with a known benign activation intervention under zero-dose, random-direction, shuffled-direction, wrong-concept, grounding, source-attribution, and optional confidence controls.

## What it does not settle

It does not establish consciousness, human-like introspection, private experience, or reliable self-knowledge. It measures a coupling between intervention, report text, and controls for this model and this prompt family.

## Deflationary explanations the lab tries to let win

| Deflationary explanation | Artifact that pressures it |
|---|---|
| The report describes visible output style, not hidden state. | `olmo3_7b_full_grounding_control_results.csv` |
| The report is prompted by target words or answer choices. | `olmo3_7b_full_prompt_leakage_audit.csv` |
| Any direction makes the model talk about the target. | `olmo3_7b_full_false_positive_floor.csv` |
| The concept direction is a random contrast or split artifact. | `olmo3_7b_full_direction_depth_sweep.csv` |
| Source attribution follows visible tone, not cause. | `olmo3_7b_full_source_attribution_confusion.csv` |
| Confidence reports are just hedging words. | `olmo3_7b_full_certainty_self_report_bridge_summary.csv` when Lab 14 is compatible |

## Run posture

- Verdict: `not_run_or_no_detection`
- Target-direction detection rate: 0.0
- Control floor: 0.0
- Specificity gap: 0.0
- Grounding pass rate: 0.0
- Source attribution accuracy: 0.4667

## Allowed claim grammar

Use `SELF-REPORT + CAUSAL` only for the intervention changing report behavior above the control floor. Use `audited` only if the grounding and source/provenance controls support the stronger reading. A failed run is not a failed lab: it says this report channel is not strongly wired by this instrument.
