# Lab 16 Validation

## Lab 16: Sycophancy and User-Belief Modeling

This directory is the final Lab 16 validation pack for the current repository
code. It replaces the older June 15 validation artifacts with the repaired
June 20, 2026 sycophancy-audit runs.

## Validation Read

The repaired result is a partial positive, not a broad sycophancy claim. The
lab now finds a selective residual direction for user-belief framing in
Olmo-3-7B-Instruct, while the behavioral sycophancy and activation-addition
results remain weak or control-limited.

The cleanest current claim is:

```text
Lab 16 finds an audited user-belief-frame direction in Olmo-3-7B-Instruct.
It does not show a stable human-like belief state, broad behavioral
sycophancy, or a specific causal sycophancy steering handle.
```

## Headline Result

The primary validation run is `olmo3_7b_full`:

- Model: `allenai/Olmo-3-7B-Instruct`
- Corpus: 240 rows from 40 frozen misconception-pressure base facts
- Domains: history, math, science, technology, trivia
- Selected measurement site: `assistant_boundary`
- Selected stream depth: 20
- Neutral-correct base-fact rate: 0.925
- User-belief held-out AUC / control AUC: 0.72 / 0.5228
- User-belief selectivity over best control: 0.1972
- Local-truth held-out AUC / control AUC: 0.9067 / 0.8578
- False-pressure sycophancy rate on neutral-correct facts: 0.027
- Agreement steering max-dose delta / specificity gap: 0.4 / 0.0
- User-belief decode verdict: `validated_selective`
- Behavioral sycophancy verdict: `rare_or_not_observed_on_known_facts`
- Agreement steering verdict: `not_specific_or_too_small`

The important teaching point is the split result. The model has a decodable
prompt-frame contrast for user belief at the assistant boundary, but it rarely
endorses the false answer on facts it can answer neutrally, and the agreement
steering effect is matched by controls such as shuffled agreement and
politeness steering.

## Current Result Summary

| Source label | Model | Scope | Main result | Read |
|---|---|---|---|---|
| `olmo3_7b_full` | Olmo-3-7B-Instruct | full 240-row audit | user-belief AUC 0.72 vs control 0.5228; neutral-correct sycophancy 0.027 | Final validation run; partial positive decode handle |
| `smollm_smoke` | SmolLM2-135M-Instruct | 60-row smoke | neutral-correct sycophancy 0.5625, but user-belief AUC 0.34 vs control 0.435 | Smoke contrast only; behavior appears weak-model-specific and decode fails |

The broader run ledger is in `lab16_validation_summary.csv`.

## What This Lab Teaches

- A sycophancy claim needs behavior conditioned on facts the neutral prompt got
  right; otherwise wrong answers and uncertainty can masquerade as agreement.
- A user-belief direction is an operational prompt-frame contrast, not mind
  reading or evidence of a stable inner belief state.
- Decode, behavior, and steering should be reported separately. Lab 16 now
  succeeds on one of those rungs, not all three.
- The best current use of the lab is teaching claim discipline: a positive
  representation result can coexist with weak behavioral and causal evidence.

## Curated Artifacts

Summary:

- `lab16_validation_report.md`
- `lab16_validation_summary.csv`

Primary Olmo run:

- `olmo3_7b_full_run_summary.md`
- `olmo3_7b_full_social_state_frame_card.md`
- `olmo3_7b_full_operationalization_audit.md`
- `olmo3_7b_full_ledger_suggestions.md`
- `olmo3_7b_full_metrics.json`
- `olmo3_7b_full_results.csv`
- `olmo3_7b_full_social_state_evidence_matrix.csv`
- `olmo3_7b_full_condition_contrasts.csv`
- `olmo3_7b_full_sycophancy_by_condition.csv`
- `olmo3_7b_full_probe_report.csv`
- `olmo3_7b_full_probe_control_gap_by_depth.csv`
- `olmo3_7b_full_direction_provenance.csv`
- `olmo3_7b_full_direction_confound_risks.csv`
- `olmo3_7b_full_agreement_steering_effects.csv`
- `olmo3_7b_full_steering_operating_points.csv`
- `olmo3_7b_full_projection_behavior_links.csv`
- `olmo3_7b_full_base_fact_outcome_matrix.csv`
- `olmo3_7b_full_hand_label_sample.csv`
- `olmo3_7b_full_hand_labeling_guide.md`

Auxiliary check:

- `smollm_smoke_*`

## Caveats

- This is a curated validation directory; full raw runs remain outside this
  pack.
- No repaired-run PNG plots were present in the local run bundle, so this pack
  keeps the new tables and reports rather than carrying forward old plots.
- Keyword labels are still a scaffold. Defended behavioral claims should fill
  the hand-label sample rather than relying only on automatic labels.
- Do not use this lab to claim human-like belief states, deception, or a
  specific causal sycophancy circuit.
