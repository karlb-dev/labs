# Lab 19 Validation

## Lab 19: Model Diffing with Shared and Model-Specific Features

This directory is the final Lab 19 validation pack for the current repository
code. It keeps the June 20, 2026 validation artifacts and drops the older
June 15 run pack so students and reviewers see one coherent result set.

## Validation Read

The result is a clean negative for the ambitious interpretability claim and a
positive result for the audit scaffold. The repaired identity control now
passes, which means the lab's validation logic can recognize the same-model
case. On the real Olmo base-vs-instruct pair, however, the candidate
instruction-tuned features are dominated by template and asymmetric effects and
do not pass the causal marker control.

The best wording is:

```text
Lab 19 provides a useful model-diffing audit workflow and a clean negative
under this small course-accessible setup. It does not recover a validated
instruction-following or assistant-persona feature.
```

## Headline Result

The identity smoke test passes after the shared-side repair:

- Pair: `EleutherAI/pythia-160m` vs `EleutherAI/pythia-160m`
- Prompt set: `course_plus_builtin`
- Shared features: 48 of 48
- False model-specific share: 0.0000
- Matched-random B-specific rate: 0.0000
- Audit status: identity pair passed specificity control

The real Olmo pair remains negative:

- Pair: `allenai/Olmo-3-1025-7B` vs `allenai/Olmo-3-7B-Instruct`
- Full prompt set: `data/model_diffing_prompt_inventory_v2.csv`
- Runtime rows: 192
- Taxonomy: 95 asymmetric, 33 shared
- Template-dominated features: 98
- Matched-random B-specific rate: 0.0039
- Causal marker verdict: not validated by marker control

## Current Result Summary

| Source label | Model pair | Main result | Causal read |
|---|---|---|---|
| `identity_smoke` | Pythia-160M vs Pythia-160M | Shared-feature control passes | Specificity audit is repaired |
| `olmo3_medium_edit` | Olmo base vs Olmo instruct | 68 template-dominated features | Not validated by marker control |
| `olmo3_full_edit` | Olmo base vs Olmo instruct | 98 template-dominated features | Not validated by marker control |
| `olmo3_full` | Olmo base vs Olmo instruct | Plotted full evidence pack | Negative interpretability claim |

## What This Lab Teaches

- Identity and matched-random controls are essential for model diffing.
- Raw-vs-chat template differences can dominate apparent model-specific
  features.
- A crosscoder can produce interpretable-looking asymmetric coordinates without
  validating an instruction-following mechanism.
- Negative results are useful when the audit clearly explains why a tempting
  claim is not supported.

## Curated Artifacts

Summary:

- `lab19_validation_report.md`
- `lab19_validation_summary.csv`

Identity smoke test:

- `identity_smoke_metrics.json`
- `identity_smoke_model_diffing_evidence_dashboard.png`
- `identity_smoke_model_diffing_evidence_matrix.csv`
- `identity_smoke_identity_smoke_scorecard.png`
- `identity_smoke_feature_audit_matrix.png`
- `identity_smoke_feature_exclusivity_histogram.png`
- `identity_smoke_causal_feature_validation_summary.csv`
- `identity_smoke_random_feature_baseline.csv`
- `identity_smoke_taxonomy_control_ladder.csv`

Olmo base-vs-instruct validation:

- `olmo3_medium_edit_metrics.json`
- `olmo3_medium_edit_model_diffing_evidence_matrix.csv`
- `olmo3_medium_edit_causal_feature_validation_summary.csv`
- `olmo3_medium_edit_random_feature_baseline.csv`
- `olmo3_medium_edit_taxonomy_control_ladder.csv`
- `olmo3_full_edit_metrics.json`
- `olmo3_full_edit_model_diffing_evidence_matrix.csv`
- `olmo3_full_edit_causal_feature_validation_summary.csv`
- `olmo3_full_edit_random_feature_baseline.csv`
- `olmo3_full_edit_taxonomy_control_ladder.csv`
- `olmo3_full_metrics.json`
- `olmo3_full_model_diffing_evidence_dashboard.png`
- `olmo3_full_model_diffing_evidence_matrix.csv`
- `olmo3_full_feature_audit_matrix.png`
- `olmo3_full_feature_exclusivity_histogram.png`
- `olmo3_full_causal_feature_validation_summary.csv`
- `olmo3_full_random_feature_baseline.csv`
- `olmo3_full_taxonomy_control_ladder.csv`

## Caveats

- This is a curated validation directory; full raw runs remain outside this
  pack.
- The negative result applies to this course-accessible final-token residual
  setup and the current prompt inventory.
- The lab should not claim that it recovered alignment, instruction-following,
  assistant voice, or persona features.
