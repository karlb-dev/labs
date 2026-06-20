# Lab 20 Final Validation Report

## Question

Is Lab 20 a weak empirical search, or is it a sound construction-stage lab that
sets up later model-organism experiments?

## Final Read

Lab 20 is a sound construction-stage lab. The June 20, 2026 validation confirms
that the current organism packages pass the checks needed before adapter
training: no public leaks, no safety blocks, clean target/control baseline
behavior on Olmo-3-7B-Instruct, and no refined spillover findings.

This is not a claim that adapters already learned the behaviors. It is a claim
that the lab now constructs auditable organism packages with enough hygiene to
support downstream training and blind-audit labs.

## Scoring and Audit Changes

- Added negation-aware spillover scoring.
- Split raw target marker mentions from refined leaks and family issues.
- Fixed tea-preference, refusal, sycophancy, and certainty scoring edge cases.
- Added `tables_spillover_probe_generations.csv` so reviewers can inspect the
  generations behind spillover scores.
- Added component spillover metrics to the saved artifacts.

## Run Matrix

| Run | Model | Organisms | Ready | Public leaks | Safety blocks | Baseline marker risks | Max refined spillover | Read |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `smollm_tiera_scoring` | SmolLM2-135M-Instruct | 4 | 3 | 0 | 0 | 1 | 0.0 | Tier-A smoke run with one toy baseline risk |
| `olmo3_7b_full_scoring` | Olmo-3-7B-Instruct | 5 | 5 | 0 | 0 | 0 | 0.0 | Final construction-stage pass |

## Recommended Course Wording

```text
Lab 20 validates construction of leak-free, safety-screened model-organism
packages. The lab prepares organisms for adapter training and blind audit; it
does not by itself claim that downstream hidden behavior has been learned.
```
