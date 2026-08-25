# GhostType COMPLETION_QUALITY_REPORT (Tier 1, mac campaign)

run: `ghosttype-cpu-dev-20260823T213801Z` · generated 2026-08-25T02:52:50.401Z · lab: aidataapps/ghosttype (Mac local campaign)

> Regenerated from GhostTypeControl by `npm run reports`. The database is the source of record.

All rows use the frozen rules: raw-text transport, candidate-extract-v1, normalize-v1 (case-sensitive), recompose-v1, ScriptDom [170.191.0] parse delta, deterministic reference decode (T=0). Campaigns marked incomplete are in progress — numbers move until the profile's 588 rows are terminal.

## Outcomes by profile/arm

| profile | arm | rows | success | partial | fail | format_fail | abstain_correct | abstain_wrong |
|---|---|---|---|---|---|---|---|---|
| baseline-deterministic | B0-empty | 588 | 0 | 0 | 0 | 0 | 101 | 487 |
| baseline-deterministic | B1-keyword | 588 | 0 | 0 | 0 | 0 | 101 | 487 |
| baseline-deterministic | B2-catalog | 588 | 4 | 9 | 11 | 0 | 101 | 463 |
| baseline-deterministic | B3-history | 588 | 0 | 14 | 14 | 0 | 101 | 459 |
| baseline-deterministic | B4-template | 588 | 0 | 47 | 37 | 0 | 92 | 412 |
| gemma-4-26b-a4b-mlx | M1-packaged | 588 | 36 | 246 | 211 | 7 | 54 | 34 |
| gemma-4-26b-a4b-nothink-mlx | M1-packaged | 588 | 40 | 302 | 229 | 2 | 13 | 2 |
| gemma-4-e4b-mlx | M1-packaged | 588 | 35 | 259 | 198 | 0 | 63 | 33 |
| muse-glimmer-30b-mlx | M1-packaged | 588 | 42 | 232 | 74 | 0 | 93 | 147 |
| olmo-3.1-32b-mlx | M1-packaged | 588 | 18 | 199 | 367 | 0 | 3 | 1 |
| qwen-3.8-27b-mlx | M1-packaged | 588 | 56 | 280 | 219 | 0 | 25 | 8 |

## Aggregate metrics (suite agg-all)

| profile | arm | normalized exact rate | success or correct abstain rate | abstain wrong rate | format fail rate | parse clean rate offered | latency p50 ms | latency p95 ms |
|---|---|---|---|---|---|---|---|---|
| baseline-deterministic | B0-empty | 17.2% | 17.2% | 82.8% | 0.0% | — | — | — |
| baseline-deterministic | B1-keyword | 17.2% | 17.2% | 82.8% | 0.0% | — | — | — |
| baseline-deterministic | B2-catalog | 17.9% | 17.9% | 78.7% | 0.0% | 83.3% | — | — |
| baseline-deterministic | B3-history | 17.2% | 17.2% | 78.1% | 0.0% | 100.0% | — | — |
| baseline-deterministic | B4-template | 15.6% | 15.6% | 70.1% | 0.0% | 98.7% | — | — |
| gemma-4-26b-a4b-mlx | M1-packaged | 18.2% | 15.3% | 5.8% | 1.2% | 99.1% | 23636 | 115763 |
| gemma-4-26b-a4b-nothink-mlx | M1-packaged | 9.9% | 9.0% | 0.3% | 0.3% | 96.1% | 2018 | 4085 |
| gemma-4-e4b-mlx | M1-packaged | 17.5% | 16.7% | 5.6% | 0.0% | 93.8% | 5592 | 13355 |
| muse-glimmer-30b-mlx | M1-packaged | 23.6% | 23.0% | 25.0% | 0.0% | 100.0% | 57635 | 112389 |
| olmo-3.1-32b-mlx | M1-packaged | 3.9% | 3.6% | 0.2% | 0.0% | 84.6% | 8174 | 21484 |
| qwen-3.8-27b-mlx | M1-packaged | 14.3% | 13.8% | 1.4% | 0.0% | 96.2% | 10253 | 19549 |

## Paired comparisons (exact McNemar on normalized_exact)

| profile | arm | vs | model-only | baseline-only | p (exact) | power label |
|---|---|---|---|---|---|---|
| gemma-4-26b-a4b-mlx | M1-packaged | B0-empty | 43 | 37 | 5.76e-1 | provisional-adequate |
| gemma-4-26b-a4b-mlx | M1-packaged | B2-catalog | 39 | 37 | 9.09e-1 | provisional-adequate |
| gemma-4-26b-a4b-nothink-mlx | M1-packaged | B0-empty | 45 | 88 | 2.42e-4 | provisional-adequate |
| gemma-4-26b-a4b-nothink-mlx | M1-packaged | B2-catalog | 43 | 90 | 5.63e-5 | provisional-adequate |
| gemma-4-e4b-mlx | M1-packaged | B0-empty | 40 | 38 | 9.10e-1 | provisional-adequate |
| gemma-4-e4b-mlx | M1-packaged | B2-catalog | 40 | 42 | 9.12e-1 | provisional-adequate |
| muse-glimmer-30b-mlx | M1-packaged | B0-empty | 46 | 8 | 1.38e-7 | provisional-adequate |
| muse-glimmer-30b-mlx | M1-packaged | B2-catalog | 43 | 9 | 2.04e-6 | provisional-adequate |
| olmo-3.1-32b-mlx | M1-packaged | B0-empty | 20 | 98 | 1.53e-13 | provisional-adequate |
| olmo-3.1-32b-mlx | M1-packaged | B2-catalog | 19 | 101 | 1.08e-14 | provisional-adequate |
| qwen-3.8-27b-mlx | M1-packaged | B0-empty | 59 | 76 | 1.68e-1 | provisional-adequate |
| qwen-3.8-27b-mlx | M1-packaged | B2-catalog | 58 | 79 | 8.71e-2 | provisional-adequate |
