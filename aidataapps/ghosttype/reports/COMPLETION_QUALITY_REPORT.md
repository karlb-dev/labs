# GhostType COMPLETION_QUALITY_REPORT (Tier 1, mac campaign)

run: `ghosttype-cpu-dev-20260823T213801Z` · generated 2026-08-23T22:14:08.921Z · lab: aidataapps/ghosttype (Mac local campaign)

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
| gemma-4-e4b-mlx | M1-packaged | 73 ⏳ | 16 | 29 | 19 | 0 | 8 | 1 |

## Aggregate metrics (suite agg-all)

| profile | arm | normalized exact rate | success or correct abstain rate | abstain wrong rate | format fail rate | parse clean rate offered | latency p50 ms | latency p95 ms |
|---|---|---|---|---|---|---|---|---|
| baseline-deterministic | B0-empty | 17.2% | 17.2% | 82.8% | 0.0% | — | — | — |
| baseline-deterministic | B1-keyword | 17.2% | 17.2% | 82.8% | 0.0% | — | — | — |
| baseline-deterministic | B2-catalog | 17.9% | 17.9% | 78.7% | 0.0% | 83.3% | — | — |
| baseline-deterministic | B3-history | 17.2% | 17.2% | 78.1% | 0.0% | 100.0% | — | — |
| baseline-deterministic | B4-template | 15.6% | 15.6% | 70.1% | 0.0% | 98.7% | — | — |
| gemma-4-e4b-mlx | M1-packaged | 48.7% | 48.7% | 0.0% | 0.0% | 91.4% | 3837 | 9528 |

## Paired comparisons (exact McNemar on normalized_exact)

| profile | arm | vs | model-only | baseline-only | p (exact) | power label |
|---|---|---|---|---|---|---|
| gemma-4-e4b-mlx | M1-packaged | B0-empty | 16 | 1 | 2.75e-4 | exploratory (incomplete ⏳) |
| gemma-4-e4b-mlx | M1-packaged | B2-catalog | 16 | 1 | 2.75e-4 | exploratory (incomplete ⏳) |
