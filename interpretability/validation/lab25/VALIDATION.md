# Lab 25 Validation

## Lab 25: Find the Wire

This directory is the final Lab 25 validation pack for the current repository
code. It replaces the older June 15 validation artifacts with the repaired
June 20, 2026 Find-the-Wire runs.

## Validation Read

The result is a clean negative, and that is the right capstone lesson. The
repaired Lab 25 run successfully builds local concept directions and runs the
intervention/control firewall, but it does not find a self-report wire that
survives the current controls.

The cleanest current claim is:

```text
Lab 25 did not validate a self-report wire under controls. It is useful as an
audit of self-report grounding and source attribution, not as evidence that
the model can reliably report its hidden activation cause.
```

## Headline Result

The primary validation run is `olmo3_7b_full`:

- Model: `allenai/Olmo-3-7B-Instruct`
- Corpus: 24 introspection-query items
- Concept directions: 8
- Injection trials: 192
- Source-attribution rows: 60
- Verdict: `not_run_or_no_detection`
- Target-direction detection rate: 0.0
- Mean control floor: 0.0
- Target minus control floor: 0.0
- Max dose-response slope: 0.0
- Grounding pass rate: 0.0
- Source attribution accuracy: 0.4667
- Main report prompt-leak rate: 0.0
- Lab 14 certainty bridge: compatible, but parsed confidence stayed flat

The evidence matrix marks every concept `not_supported`. The directions can
produce large decode/projection gaps, but those handles do not make the model
reliably self-report the injected cause. The source-attribution audit is also
weak: the model is good at user-instruction attribution, decent at default
mode, poor at system-prompt attribution, and does not correctly identify the
activation-injection source.

## Current Result Summary

| Source label | Model | Scope | Main result | Read |
|---|---|---|---|---|
| `olmo3_7b_full` | Olmo-3-7B-Instruct | full intervention audit | target detection 0.0, grounding pass 0.0, source attribution 0.4667 | Final validation run; clean negative |
| `olmo3_7b_source_rubric` | Olmo-3-7B-Instruct | source-attribution-only rerun | source attribution 0.4667 | Confirms the source rubric, but does not run intervention trials |
| `smollm_smoke_noleak` | SmolLM2-135M-Instruct | small no-leak smoke | target detection 0.0, source attribution 0.0 | Prompt-leak smoke check; no positive wire signal |

The broader run ledger is in `lab25_validation_summary.csv`.

## What This Lab Teaches

- A positive-looking direction is not enough; the report channel has to move
  above zero, random, shuffled, wrong-concept, and source/provenance controls.
- A failed self-report wire is not a failed lab. It demonstrates that the audit
  can refuse an overclaim.
- Source attribution is separable from behavior steering. The model can follow
  visible or prompted style without knowing the hidden source.
- The Lab 14 certainty bridge being compatible does not rescue the claim:
  parsed verbal confidence stayed flat across certainty plus, certainty minus,
  random, and zero conditions.

## Curated Artifacts

Summary:

- `lab25_validation_report.md`
- `lab25_validation_summary.csv`

Primary full Olmo run:

- `olmo3_7b_full_find_the_wire_report.md`
- `olmo3_7b_full_run_summary.md`
- `olmo3_7b_full_operationalization_audit.md`
- `olmo3_7b_full_ledger_suggestions.md`
- `olmo3_7b_full_metrics.json`
- `olmo3_7b_full_results.csv`
- `olmo3_7b_full_wire_evidence_matrix.csv`
- `olmo3_7b_full_false_positive_floor.csv`
- `olmo3_7b_full_self_report_detection_dose_response.csv`
- `olmo3_7b_full_self_report_operating_points.csv`
- `olmo3_7b_full_grounding_control_summary.csv`
- `olmo3_7b_full_grounding_control_results.csv`
- `olmo3_7b_full_voice_self_attribution_summary.csv`
- `olmo3_7b_full_source_attribution_confusion.csv`
- `olmo3_7b_full_report_discipline_scorecard.csv`
- `olmo3_7b_full_direction_construction.csv`
- `olmo3_7b_full_direction_depth_sweep.csv`
- `olmo3_7b_full_direction_geometry_summary.csv`
- `olmo3_7b_full_certainty_bridge_status.csv`
- `olmo3_7b_full_certainty_self_report_bridge_summary.csv`
- `olmo3_7b_full_prompt_leakage_audit.csv`
- `olmo3_7b_full_self_report_labeling_guide.md`

Auxiliary checks:

- `olmo3_7b_source_rubric_*`
- `smollm_smoke_noleak_*`

## Caveats

- This is a curated validation directory; full raw runs remain outside this
  pack.
- No repaired-run PNG plots were present in the local run bundle, so this pack
  intentionally keeps tables and reports rather than carrying forward old plots.
- The validated result is a negative audit: do not use Lab 25 to claim reliable
  self-knowledge, hidden-state access, or consciousness.
- The source-rubric run is attribution-only and should not be read as an
  intervention validation.
