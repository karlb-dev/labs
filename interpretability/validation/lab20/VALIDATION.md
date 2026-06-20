# Lab 20 Validation

## Lab 20: Model Organism Construction

This directory is the final Lab 20 validation pack for the current repository
code. It keeps the June 20, 2026 validation artifacts and drops the older
June 15 run pack so students and reviewers see one coherent result set.

## Validation Read

The result is a positive construction-stage validation. Lab 20 is not trying to
prove that adapters learned hidden behaviors yet; it is validating that the
organism packages are leak-free, safety-screened, and ready for downstream
adapter training and blind audit.

The best wording is:

```text
Lab 20 validates the construction stage for model-organism packages. On
Olmo-3-7B-Instruct, all five organisms are ready for adapter training with no
public leaks, no safety blocks, and no refined spillover findings.
```

## Headline Result

The strongest current run is the Olmo-3-7B-Instruct full scoring run:

- Model: `allenai/Olmo-3-7B-Instruct`
- Organisms checked: 5 of 5
- Ready for adapter training: 5 of 5
- Public leaks: 0
- Safety blocks: 0
- Baseline marker risks: 0
- Max refined spillover rate: 0.0
- Max family issue rate: 0.0
- Max target marker leak rate: 0.0

The SmolLM tier-A run is a smaller smoke comparison:

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Organisms checked: 4 of 4
- Ready for adapter training: 3 of 4
- Public leaks: 0
- Safety blocks: 0
- Baseline marker risks: 1
- Max refined spillover rate: 0.0

## Current Result Summary

| Source label | Model | Main result | Read |
|---|---|---|---|
| `olmo3_7b_full_scoring` | Olmo-3-7B-Instruct | 5/5 organisms ready | Positive construction-stage validation |
| `smollm_tiera_scoring` | SmolLM2-135M-Instruct | 3/4 organisms ready | Useful smoke run; one toy baseline risk |

## What This Lab Teaches

- Model-organism work has a construction phase before any adapter result can be
  claimed.
- Leak checks, safety checks, baseline target/control gaps, and spillover audits
  are separate gates.
- Refined spillover logic matters: raw marker words can appear in benign
  negated contexts and should not automatically count as leaks.
- A clean construction pack is a prerequisite for Lab 21 and Lab 23 style
  downstream training/audit claims.

## Curated Artifacts

Summary:

- `lab20_validation_report.md`

Olmo-3-7B-Instruct full scoring:

- `olmo3_7b_full_scoring_results.csv`
- `olmo3_7b_full_scoring_metrics.json`
- `olmo3_7b_full_scoring_organism_construction_dashboard.png`
- `olmo3_7b_full_scoring_construction_evidence_dashboard.png`
- `olmo3_7b_full_scoring_spillover_risk_matrix.png`
- `olmo3_7b_full_scoring_tables_organism_readiness_scorecard.csv`
- `olmo3_7b_full_scoring_tables_spillover_audit.csv`
- `olmo3_7b_full_scoring_tables_spillover_probe_generations.csv`

SmolLM tier-A scoring:

- `smollm_tiera_scoring_results.csv`
- `smollm_tiera_scoring_metrics.json`
- `smollm_tiera_scoring_tables_organism_readiness_scorecard.csv`
- `smollm_tiera_scoring_tables_spillover_audit.csv`
- `smollm_tiera_scoring_tables_spillover_probe_generations.csv`

## Caveats

- This is a curated validation directory; full raw runs remain outside this
  pack.
- The result is construction-stage validation only. It does not claim adapter
  training succeeded or that hidden behavior was learned.
- The downstream behavior-learning and blind-audit claims belong to later labs.
