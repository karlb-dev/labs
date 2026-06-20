# Lab 8 Existing Run Audit

Input root: `runs/lab08_existing_comparison`
Runs parsed: 5

## Summary

| run | model | FVU | L0 | silent | verdicts | best feature | clamp |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `lab08_tierb_validate` | `allenai/Olmo-3-1025-7B` | 0.3736 | 113.54 | 0.6572 | 1 survived, 3 narrowed, 16 killed/poly/token/silent | F1265 law (narrowed, AUC 0.8148, fire 0.01182); survived F1204 emotion (AUC 0.879, fire 0.85515) | 0 -> 5 hits |
| `lab8_run6_B` | `allenai/Olmo-3-1025-7B` | 0.3761 | 113.54 | 0.6098 | 0 survived, 4 narrowed, 21 killed/poly/token/silent | F1265 law (narrowed, AUC 0.7716, fire 0.01119) | 0 -> 5 hits |
| `lab8_run6_C` | `allenai/Olmo-3-1025-7B` | 0.3761 | 113.54 | 0.6098 | 0 survived, 4 narrowed, 21 killed/poly/token/silent | F1265 law (narrowed, AUC 0.7716, fire 0.01119) | 0 -> 5 hits |
| `lab08_tiera_validate` | `gpt2` | 0.0019 | 74.57 | 0.3325 | 0 survived, 2 narrowed, 18 killed/poly/token/silent | F9303 code (narrowed, AUC 0.7083, fire 0.02419) | 0 -> 0 hits |
| `lab8_run6_A` | `gpt2` | 0.0019 | 74.57 | 0.248 | 0 survived, 2 narrowed, 23 killed/poly/token/silent | F9303 code (narrowed, AUC 0.7105, fire 0.02088) | 0 -> 0 hits |

## Prompt-Summary Check

- `run6/A` matches the reported gpt2 negative: FVU about 0.0019, L0 about 74.57, 0 survived, 2 narrowed, and a failed code clamp.
- `run6/B` and `run6/C` match the reported Olmo result: FVU about 0.3761, L0 about 113.54, 0 survived, 4 narrowed, with feature 1265 as the best law handle.
- `run1/lab08_tierb_full` contains the earlier positive: 1 survived feature, with broad high-fire emotion feature 1204 in the atlas.

## Notes

- A run-level `clamp_causal=true` is not treated as a clean causal claim here; the older clamp used one random control and keyword hits only.
- The audit reflects historical artifacts as written, including the older line-level validation protocol.
