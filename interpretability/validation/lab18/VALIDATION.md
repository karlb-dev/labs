# Lab 18 Validation

## Lab 18: Humor and Joke-Structure Directions

This directory is the final Lab 18 validation pack for the current repository
code. It keeps the June 20, 2026 validation artifacts and drops the older
June 15 run pack so students and reviewers see one coherent result set.

## Validation Read

The result is a partial positive. Lab 18 now finds a controlled
joke-structure/register direction on Olmo-3-7B-Instruct, but the effect is
seed-sensitive and does not survive as a causal humor steering claim.

The best wording is:

```text
Lab 18 finds a scoped joke-structure decodability handle, but the current
validation does not establish a general humor mechanism or robust causal humor
steering direction.
```

## Headline Result

The strongest current run is the Olmo-3-7B-Instruct seed 0 run:

- Model: `allenai/Olmo-3-7B-Instruct`
- Corpus: `data/humor_incongruity_pairs.csv`
- Corpus size: 80 rows
- Families: 8 joke families, 10 rows each
- Test AUC: 1.0000
- Shuffled-label AUC: 0.7828
- Random-control AUC: 0.7496
- Selectivity over best null: 0.2172
- Family-control gap: 0.0643
- Steering specificity gap: -0.3000

## Current Result Summary

| Source label | Model | Main result | Causal read |
|---|---|---|---|
| `olmo3_7b_full_s0` | Olmo-3-7B-Instruct | Test AUC 1.0000, selectivity 0.2172 | Decodable joke-structure handle, not causally separated |
| `olmo3_7b_full_s1` | Olmo-3-7B-Instruct | Test AUC 0.9272, selectivity 0.0722 | Weak or confounded register handle |
| `smollm_full_s0` | SmolLM2-135M-Instruct | Test AUC 1.0000, selectivity 0.0824 | Weak or confounded register handle |

## What This Lab Teaches

- Humor is a difficult target for mechanistic validation because joke/literal
  contrasts are entangled with surprise, silliness, positivity, and setup style.
- A probe can separate joke-structured text from literal rewrites without
  isolating a general humor mechanism.
- Held-out family summaries are important: the signal is useful but does not
  generalize strongly enough to support an expansive claim.
- Steering controls are doing real work here; they prevent the lab from
  overstating a decodable feature as a causal humor knob.

## Curated Artifacts

Summary:

- `lab18_validation_report.md`
- `lab18_validation_summary.csv`

Olmo-3-7B-Instruct final runs:

- `olmo3_7b_full_s0_results.csv`
- `olmo3_7b_full_s0_metrics.json`
- `olmo3_7b_full_s0_humor_evidence_dashboard.png`
- `olmo3_7b_full_s0_humor_evidence_matrix.csv`
- `olmo3_7b_full_s0_family_generalization_summary.csv`
- `olmo3_7b_full_s1_results.csv`
- `olmo3_7b_full_s1_metrics.json`
- `olmo3_7b_full_s1_humor_evidence_dashboard.png`
- `olmo3_7b_full_s1_humor_evidence_matrix.csv`
- `olmo3_7b_full_s1_family_generalization_summary.csv`

SmolLM comparison:

- `smollm_full_s0_results.csv`
- `smollm_full_s0_metrics.json`
- `smollm_full_s0_humor_evidence_dashboard.png`
- `smollm_full_s0_humor_evidence_matrix.csv`
- `smollm_full_s0_family_generalization_summary.csv`

## Caveats

- This is a curated validation directory; full raw runs remain outside this
  pack.
- The positive evidence is seed-sensitive and narrow.
- The current validation supports joke-structure/register decodability, not a
  general or causal theory of humor.
