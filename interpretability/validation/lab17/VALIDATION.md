# Lab 17 Validation

## Lab 17: Persona Register and Behavioral Traits

This directory is the final Lab 17 validation pack for the current repository
code. It keeps the June 20, 2026 validation artifacts and drops the older
June 15 run pack so students and reviewers see one coherent result set.

## Validation Read

The result is a partial positive. Lab 17 now gives a strong, controlled
decoding result for persona/register traits, especially on
`allenai/Olmo-3-7B-Instruct`. The cleanest current claim is that the lab can
find held-out persona/register directions that beat matched random controls.

The causal steering result is not yet strong. The best wording is:

```text
Lab 17 robustly finds controlled persona/register decodability handles, but
the current steering intervention does not establish a robust causal persona
control claim.
```

## Headline Result

The strongest runs are the two full-corpus Olmo-3-7B-Instruct seeds:

- Model: `allenai/Olmo-3-7B-Instruct`
- Corpus: `data/persona_register_pairs.csv`
- Corpus size: 256 rows
- Split key: trait/topic held-out splits
- Seed 0, five-control rerun: test AUC 1.0000, random-control AUC 0.6439
- Seed 1, five-control rerun: test AUC 1.0000, random-control AUC 0.6577
- Mean selectivity over random controls: 0.3561 for seed 0, 0.3423 for seed 1
- Steering specificity gap: negative or near zero in the five-control runs

## Current Result Summary

| Source label | Model | Main result | Causal read |
|---|---|---|---|
| `olmo3_7b_full_s0_controls5` | Olmo-3-7B-Instruct | Test AUC 1.0000, random AUC 0.6439 | Decodable but not steerable in this run |
| `olmo3_7b_full_s1_controls5` | Olmo-3-7B-Instruct | Test AUC 1.0000, random AUC 0.6577 | Decodable but not steerable in this run |
| `olmo3_7b_full_s0` | Olmo-3-7B-Instruct | Test AUC 1.0000, random AUC 0.6439 | Pilot posture handle did not survive stronger controls |
| `smollm_full_s0` | SmolLM2-135M-Instruct | Test AUC 0.9847, random AUC 0.5888 | Fails content-preservation controls |

## What This Lab Teaches

- Persona/register traits can be separable in residual activations under a
  held-out split and matched random controls.
- A high trait probe score is not the same as a successful behavioral steering
  intervention.
- Content preservation matters: the SmolLM run finds a decodable direction, but
  the intervention changes content too much to support a clean steering claim.
- The improved corpus and five-control reruns make the lab much more useful as
  a skeptical validation exercise than the original tiny-data version.

## Curated Artifacts

Summary:

- `lab17_validation_report.md`
- `lab17_validation_summary.csv`

Olmo-3-7B-Instruct final runs:

- `olmo3_7b_full_s0_results.csv`
- `olmo3_7b_full_s0_metrics.json`
- `olmo3_7b_full_s0_persona_evidence_dashboard.png`
- `olmo3_7b_full_s0_refusal_boundary_safety_dashboard.png`
- `olmo3_7b_full_s0_persona_trait_evidence_matrix.csv`
- `olmo3_7b_full_s0_controls5_results.csv`
- `olmo3_7b_full_s0_controls5_metrics.json`
- `olmo3_7b_full_s0_controls5_persona_evidence_dashboard.png`
- `olmo3_7b_full_s0_controls5_refusal_boundary_safety_dashboard.png`
- `olmo3_7b_full_s0_controls5_persona_trait_evidence_matrix.csv`
- `olmo3_7b_full_s1_controls5_results.csv`
- `olmo3_7b_full_s1_controls5_metrics.json`
- `olmo3_7b_full_s1_controls5_persona_evidence_dashboard.png`
- `olmo3_7b_full_s1_controls5_refusal_boundary_safety_dashboard.png`
- `olmo3_7b_full_s1_controls5_persona_trait_evidence_matrix.csv`

SmolLM comparison:

- `smollm_full_s0_results.csv`
- `smollm_full_s0_metrics.json`
- `smollm_full_s0_persona_evidence_dashboard.png`
- `smollm_full_s0_refusal_boundary_safety_dashboard.png`
- `smollm_full_s0_persona_trait_evidence_matrix.csv`

## Caveats

- This is a curated validation directory; full raw runs remain outside this
  pack.
- The current positive claim is a decodability claim, not a robust behavioral
  steering claim.
- The stronger five-control Olmo reruns supersede the single-control pilot run.
