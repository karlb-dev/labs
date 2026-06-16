# Lab 36 Severance Validation

Date: 2026-06-16
Branch: `interp_sev`

Relevant commits:

- `0e92cfa` - initial Lab 36 implementation
- `37e919e` - user refactor and expanded Lab 36 data
- `acea1aa` - validation fix for B2 wrong-layer control bounds
- `bd18deb` - visual artifact pass with plot manifests and source tables

Lab 36 implements the Severance report-channel experiment as a bench-registered lab. It runs the instrumentation proof, patchscope-lite cartography, contrast-direction build, B2 report-channel screen, B3 confidence bridge, B4 matched-output source attribution, B5 insertion-presence detection, and a minimal C-track patch-recovery audit. The `gpt-oss-120B` condition was intentionally skipped on this Colab machine.

All useful runs were copied to:

`/content/drive/MyDrive/interpret/verify_severance/`

## Visual Artifact Validation Commands

```bash
python interp_bench.py --lab lab36 --tier a --mode all --prompt-set full --max-examples 0 --run-name lab36_smollm_full_visual
python interp_bench.py --lab lab36 --tier b --mode all --prompt-set full --run-name lab36_olmo3_7b_visual_full
python interp_bench.py --lab lab36 --tier c --mode all --prompt-set full --run-name lab36_olmo31_32b_visual_full
```

All three visual runs completed with plots enabled. Each run wrote nine PNG figures, `plots/plot_manifest.{json,csv}`, `plots/plot_reading_guide.csv`, and figure source CSVs under `tables/figure_sources/`. A PIL integrity pass found all 27 PNGs nonblank.

## Visual Artifact Headline Metrics

| Run | Model | Verdict | n_items | n_directions | B4 activation acc | B4 fresh acc | B5 d-prime | B5 false alarm | B5 content leak | B5 pass | Warning count | Failure specimens | Plot entries |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lab36_smollm_full_visual` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `b4_matched_source_candidate` | 25 | 5 | 0.5556 | 0.4444 | -0.9221 | 0.9706 | 0.125 | 0 | 5 | 64 | 9 |
| `lab36_olmo3_7b_visual_full` | `allenai/Olmo-3-7B-Instruct` | `no_report_channel_coupling_validated` | 25 | 5 | 0.0 | 0.0 | 2.1194 | 0.0294 | 0.125 | 0 | 4 | 59 | 9 |
| `lab36_olmo31_32b_visual_full` | `allenai/Olmo-3.1-32B-Instruct` | `no_report_channel_coupling_validated` | 25 | 5 | 0.0 | 0.0 | 0.7927 | 0.0294 | 0.1562 | 0 | 4 | 64 | 9 |

## Visual Artifact Interpretation

The visual pass confirms the previous refactor result and makes the failure modes easier to audit. The 7B and 32B models still show a B5 insertion-presence signal with low false-alarm rate, but neither run passes the strict B5 gate because content leakage remains above `0.10`. B4 remains negative on the GPU models. The visual artifacts therefore support the same claim boundary as the refactor run: there is useful B5 sensitivity evidence, but not a clean content-blind report-channel validation.

New visual files to inspect first:

- `plots/overview_dashboard.png`
- `plots/target_vs_control.png`
- `plots/dose_response.png`
- `plots/layer_sweep_heatmap.png`
- `plots/trajectory.png`
- `plots/paired_examples.png`
- `plots/source_attribution_control_matrix.png`
- `plots/b5_detection_margins.png`
- `plots/plot_manifest.json`
- `tables/figure_sources/*.csv`

## Refactor Validation Commands

```bash
python interp_bench.py --lab lab36 --tier a --mode smoke --no-plots --run-name lab36_tier_a_smoke_refactor_fixed
python interp_bench.py --lab lab36 --tier b --mode all --prompt-set full --run-name lab36_olmo3_7b_refactor_full
python interp_bench.py --lab lab36 --tier c --mode all --prompt-set full --run-name lab36_olmo31_32b_refactor_full
```

The first refactor smoke attempt, `lab36_tier_a_smoke_refactor`, exposed an out-of-range B2 wrong-layer control when a selected direction landed on the final decoder block. Commit `acea1aa` fixed that by choosing a lower valid wrong layer and clamping execution layers to valid decoder-block indices. The successful runs below used that fix.

## Refactor Instrumentation

| Run | Model | Mode | Hook parity | Lens parity | KV replay parity |
| --- | --- | --- | --- | --- | --- |
| `lab36_tier_a_smoke_refactor_fixed` | `HuggingFaceTB/SmolLM2-135M-Instruct` | smoke | OK, max_abs=0 | OK | OK, strict logits, label match, max_abs_logit_diff=2.77e-05 |
| `lab36_olmo3_7b_refactor_full` | `allenai/Olmo-3-7B-Instruct` | all | OK, max_abs=0 | OK | OK by label, max_abs_logit_diff=0.3125, position_contiguous=1 |
| `lab36_olmo31_32b_refactor_full` | `allenai/Olmo-3.1-32B-Instruct` | all | OK, max_abs=0 | OK | OK by label, max_abs_logit_diff=0.375, position_contiguous=1 |

The 7B and 32B KV replay checks are not strict-logit-equal in bf16, but the incremental and full-forward source labels match and attribution positions are contiguous. B4 should still be interpreted through the full source-attribution table rather than the parity diagnostic alone.

## Refactor Headline Metrics

| Run | n_items | n_directions | Verdict | B4 activation acc | B4 fresh acc | B5 d-prime | B5 false alarm | B5 content leak | B5 pass | B2 mean target-minus-floor | Max patch recovery | B3 entropy delta |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lab36_tier_a_smoke_refactor_fixed` | 6 | 3 | `b4_matched_source_candidate` | 0.5 | 0.3333 | -1.8315 | 0.9615 | 0.2632 | 0 | 0.0 | n/a | n/a |
| `lab36_olmo3_7b_refactor_full` | 25 | 5 | `no_report_channel_coupling_validated` | 0.0 | 0.0 | 2.1194 | 0.0294 | 0.125 | 0 | -0.2 | 1.0385 | 0.0812 |
| `lab36_olmo31_32b_refactor_full` | 25 | 5 | `no_report_channel_coupling_validated` | 0.0 | 0.0 | 0.7927 | 0.0294 | 0.1562 | 0 | -0.4 | 1.0 | 0.1381 |

## Refactor Interpretation

The refactor produced better data than the initial run. B5 now shows insertion-presence sensitivity on both GPU models:

- 7B: d-prime `2.1194`, false alarm `0.0294`
- 32B: d-prime `0.7927`, false alarm `0.0294`

However, both GPU runs still fail the strict B5 gate because content leakage is above the `0.10` threshold:

- 7B: content leak `0.125`
- 32B: content leak `0.1562`

B4 matched-output source attribution remains negative on both GPU runs: activation-source accuracy is `0.0` for 7B and 32B. B2 remains a screen only, with negative or zero target-minus-floor after controls. The correct read is therefore: the revised experiment found a real B5 anomaly-detection signal, but not a clean content-blind report-channel validation.

The Tier A smoke model produced a B4 candidate, but it also had high B5 false alarms and is not the target evidence tier. Treat it as a useful debugging signal, not as a science claim.

## Initial Validation

Initial validation was run before the `37e919e` refactor:

```bash
python interp_bench.py --lab lab36 --tier a --mode smoke --no-plots --run-name lab36_tier_a_smoke
python interp_bench.py --lab lab36 --tier b --mode all --prompt-set full --run-name lab36_olmo3_7b_full
python interp_bench.py --lab lab36 --tier c --mode all --prompt-set full --run-name lab36_olmo31_32b_full
```

| Run | n_items | n_directions | Verdict | B4 activation acc | B5 d-prime | B5 false alarm | B5 content leak | B2 mean target-minus-floor | Max patch recovery | B3 entropy delta |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lab36_tier_a_smoke_02` | 4 | 3 | `no_report_channel_coupling_validated` | 0.0 | -2.69 | 0.9444 | 0.0 | 0.0 | n/a | n/a |
| `lab36_olmo3_7b_full` | 12 | 4 | `no_report_channel_coupling_validated` | 0.0 | -0.1756 | 0.0556 | 0.0833 | -0.25 | -0.0909 | 0.1378 |
| `lab36_olmo31_32b_full` | 12 | 4 | `no_report_channel_coupling_validated` | 0.0 | -0.1756 | 0.0556 | 0.0 | 0.0 | 3.0 | 0.5143 |

## Files To Inspect

Primary files in each run:

- `run_summary.md`
- `metrics.json`
- `tables/evidence_matrix.csv`
- `tables/source_attribution_summary.csv`
- `tables/injection_detection_summary.csv`
- `tables/injection_detection_results.csv`
- `tables/entropy_dissociation.csv`
- `tables/patch_recovery_heatmap.csv`
- `diagnostics/hook_parity.json`
- `diagnostics/kv_replay_parity.json`
