# Lab 36 Severance Validation

Date: 2026-06-16
Branch: `interp_sev`
Implementation commit: `0e92cfa` (`Add lab36 severance report-channel experiment`)

Lab 36 implements the Severance report-channel experiment as a bench-registered lab. It runs the instrumentation proof, patchscope-lite cartography, contrast-direction build, B2 report-channel screen, B3 confidence bridge, B4 matched-output source attribution, B5 insertion-presence detection, and a minimal C-track patch-recovery audit. The `gpt-oss-120B` condition was intentionally skipped on this Colab machine.

## Commands Run

```bash
python interp_bench.py --lab lab36 --tier a --mode smoke --no-plots --run-name lab36_tier_a_smoke
python interp_bench.py --lab lab36 --tier b --mode all --prompt-set full --run-name lab36_olmo3_7b_full
python interp_bench.py --lab lab36 --tier c --mode all --prompt-set full --run-name lab36_olmo31_32b_full
```

## Saved Runs

All useful runs were copied to:

`/content/drive/MyDrive/interpret/verify_severance/`

Local run directories:

- `interpretability/runs/lab36_tier_a_smoke_02`
- `interpretability/runs/lab36_olmo3_7b_full`
- `interpretability/runs/lab36_olmo31_32b_full`

## Instrumentation Checks

| Run | Model | Mode | Hook parity | Lens parity | KV replay parity |
| --- | --- | --- | --- | --- | --- |
| `lab36_tier_a_smoke_02` | `HuggingFaceTB/SmolLM2-135M-Instruct` | smoke | OK, max_abs=0 | OK | OK, label match, max_abs_logit_diff=2.77e-05 |
| `lab36_olmo3_7b_full` | `allenai/Olmo-3-7B-Instruct` | all | OK, max_abs=0 | OK | OK, label match, max_abs_logit_diff=0.5625 |
| `lab36_olmo31_32b_full` | `allenai/Olmo-3.1-32B-Instruct` | all | OK, max_abs=0 | OK | OK, label match, max_abs_logit_diff=0.25 |

The KV replay threshold treats exact-logit agreement as sufficient when present, otherwise label agreement is the gate for the B4 replay machinery. Both full GPU runs passed by label agreement.

## Headline Metrics

| Run | n_items | n_directions | Verdict | B4 activation acc | B5 d-prime | B5 false alarm | B5 content leak | B2 mean target-minus-floor | Max patch recovery | B3 entropy delta |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lab36_tier_a_smoke_02` | 4 | 3 | `no_report_channel_coupling_validated` | 0.0 | -2.69 | 0.9444 | 0.0 | 0.0 | n/a | n/a |
| `lab36_olmo3_7b_full` | 12 | 4 | `no_report_channel_coupling_validated` | 0.0 | -0.1756 | 0.0556 | 0.0833 | -0.25 | -0.0909 | 0.1378 |
| `lab36_olmo31_32b_full` | 12 | 4 | `no_report_channel_coupling_validated` | 0.0 | -0.1756 | 0.0556 | 0.0 | 0.0 | 3.0 | 0.5143 |

## Interpretation

Both full GPU runs completed all requested Lab 36 tracks and did not validate headline report-channel coupling. Direction probes separated the contrast prompt sets, but B2 remained a screen only, B4 activation-source attribution did not identify the hidden activation source, and B5 content-blind insertion detection did not exceed the signal-detection gate.

Primary files to inspect for each run:

- `run_summary.md`
- `metrics.json`
- `tables/evidence_matrix.csv`
- `tables/source_attribution_summary.csv`
- `tables/injection_detection_summary.csv`
- `tables/entropy_dissociation.csv`
- `tables/patch_recovery_heatmap.csv`
- `diagnostics/hook_parity.json`
- `diagnostics/kv_replay_parity.json`
