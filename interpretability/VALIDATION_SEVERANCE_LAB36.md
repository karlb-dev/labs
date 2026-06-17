# Lab 36 Severance Validation

Date: 2026-06-16
Branch: `interp_sev`

Relevant commits:

- `0e92cfa` - initial Lab 36 implementation
- `37e919e` - user refactor and expanded Lab 36 data
- `acea1aa` - validation fix for B2 wrong-layer control bounds
- `bd18deb` - visual artifact pass with plot manifests and source tables
- `9dda941` - content-blind B5 + row-randomized B4 (the v3 patch validated below)

Lab 36 implements the Severance report-channel experiment as a bench-registered lab. It runs the instrumentation proof, patchscope-lite cartography, contrast-direction build, B2 report-channel screen, B3 confidence bridge, B4 matched-output source attribution, B5 insertion-presence detection, and a minimal C-track patch-recovery audit. The `gpt-oss-120B` condition was intentionally skipped on this Colab machine.

All useful runs were copied to:

`/content/drive/MyDrive/interpret/verify_severance/`

## Full 4-Model v3 Validation Run (2026-06-16, commit `9dda941`)

This is the latest and most complete validation: the v3 content-blind code run on **four models across three architectures**, including the new **Gemma-4-E4B** (`Gemma4ForConditionalGeneration`). It is the head-to-head check the section title of this file promises — latest code vs. the pre-v3 `verify_severance` baselines above. Results, full run trees, logs, and a long-form writeup are in:

`/content/drive/MyDrive/interpret/lab36_fullrun_20260616/` (see `VALIDATION_REPORT.md` there).

Commands (latest code, all tracks, full prompt set, 25 items, 5 directions, seed 0):

```bash
python interp_bench.py --lab lab36 --tier a --mode all --prompt-set full --max-examples 0 --run-name lab36_tierA_smollm_full
python interp_bench.py --lab lab36 --tier b --mode all --prompt-set full --run-name lab36_tierB_olmo7b_full
python interp_bench.py --lab lab36 --tier c --mode all --prompt-set full --run-name lab36_tierC_olmo31_32b_full
python interp_bench.py --lab lab36 --tier b --model google/gemma-4-E4B-it --mode all --prompt-set full --run-name lab36_gemma4_e4b_full
```

Model-list note: Tier C is `allenai/Olmo-3.1-32B-Instruct`; there is no `allenai/Olmo-3-32B-Instruct` on the Hub, so "Olmo 3 32B Instruct" and Tier C are the same model. Gemma-4-E4B ran through the bench with **no code changes** (the anatomy resolver already reads `config.text_config` / `model.language_model.layers`).

### v3 headline metrics

B5 headline = `headline_content_blind_logit_only` (content-blind variant, 2.0 dose, `decision_source=next_token_logits`), with bootstrap CIs.

| Run | Model | Verdict | B4 act acc | B5 d′ | B5 d′ CI | B5 FA | B5 leak | B5 pass | self-check |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `lab36_tierA_smollm_full` | `SmolLM2-135M-Instruct` | `no_report_channel_coupling_validated` | 0.2222 | −1.3299 | [−1.737, −0.793] | 0.9706 | 0.1250 | 0 | ok |
| `lab36_tierB_olmo7b_full` | `Olmo-3-7B-Instruct` | `no_report_channel_coupling_validated` | 0.0 | 0.6080 | [−0.072, 1.377] | 0.3824 | 0.0625 | 0 | ok |
| `lab36_tierC_olmo31_32b_full` | `Olmo-3.1-32B-Instruct` | `no_report_channel_coupling_validated` | 0.0 | 0.7927 | [0.199, 1.239] | 0.0294 | 0.2188 | 0 | ok |
| `lab36_gemma4_e4b_full` | `gemma-4-E4B-it` | `no_report_channel_coupling_validated` | 0.0 | 1.2387 | [0.793, 1.660] | 0.0294 | 0.1562 | 0 | ok |

All runs: hook parity OK (max_abs=0), lens self-check OK (top-5 overlap 5/5), KV replay OK and position-contiguous, safety OK, 9/9 non-blank plots.

### Comparison to the pre-v3 baseline

The v3 patch changed both code and data (B2/B4/B5 CSVs regenerated; `injection_detection_prompts`, `source_attribution_prompts`, `introspection_queries` hashes changed; `uncertainty_questions` and `patchscope_prompts` unchanged), and moved the B5 headline to the content-blind next-token-logit decision. Differences are the combined v3 protocol effect, not a model regression.

- **SmolLM2-135M verdict flipped** `b4_matched_source_candidate` → `no_report_channel_coupling_validated`: row-randomizing the B4 hidden-source label removed the fixed-`D` option bias (B4 act 0.556 → 0.222). The patch worked as intended.
- **Olmo-3-7B B5 d′ dropped 2.1194 → 0.6080** with the CI now including 0 and FA rising to 0.382. The earlier strong number was largely an artifact of the looser pre-v3 protocol; content leak now passes (0.0625) but d′/FA fail.
- **Olmo-3.1-32B** is stable (d′/FA unchanged at 0.7927/0.0294 due to its very low yes-rate); content leak rose 0.156 → 0.219 and still fails.
- **Gemma-4-E4B** (not in the baseline) gives the strongest content-blind d′ (1.24, CI excludes 0) but fails the leak gate.

### Read

All four models converge on `no_report_channel_coupling_validated`. B4 is negative everywhere and is additionally **unreliable this run** (canonical-answer plausibility 0.0–0.22, below the 0.75 gate, on every model — identical caveat to baseline). B5 shows real content-blind anomaly sensitivity on Gemma-4-E4B and Olmo-3.1-32B (d′ ≥ 0.75, FA ≤ 0.25, CIs excluding 0) but both fail the ≤ 0.10 content-leak gate. The latest version was reviewed and reproduced cleanly; **no code defects were found and no fixes were required.**

## Improved Instruments + Reasoning Axis (2026-06-17)

After the v3 run, three improvements were implemented and validated (full writeup, run trees, and logs in `/content/drive/MyDrive/interpret/lab36_improved_20260616/`):

1. **B4 canonical-answer fix** (`B4_USE_MODEL_CANONICAL_ANSWER`): teacher-force the model's own greedy default-route answer (matched across all hidden routes) instead of the off-distribution CSV string, so KV replay scores source attribution on a plausible completion. Plausibility is read on the non-injected `matched_default` route (the injection depresses the answer logprob by design and must not trip the gate).
2. **B5 sentinel control** (`B5_SENTINEL_VARIANT`): a parallel content-blind decision that injects once at an upstream interior position of the *same* user-framed prompt, with the yes/no read downstream and no injection at the decision token. New `sentinel_content_blind_logit_only` summary row, `b5_sentinel_*` metrics, and a `b5_sentinel_collapse` warning.
3. **Reasoning axis**: Olmo-3-7B-Think and Olmo-3.1-32B-Think beside their Instruct counterparts.

Six models, `--mode all --prompt-set full`, 25 items, 5 directions, seed 0. All six: self-check OK, hook/lens/KV parity OK, 9/9 non-blank plots.

| Model | Verdict | B4 acc | B4 plaus | B5 d′ report-query | B5 d′ sentinel | B5 FA | B5 leak |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M | `no_report_channel_coupling_validated` | 0.000 | 1.000 | −1.330 | 0.277* | 0.971 | 0.125 |
| Olmo-3-7B-Instruct | `no_report_channel_coupling_validated` | 0.111 | 1.000 | 0.608 | −0.009 | 0.382 | 0.062 |
| Olmo-3-7B-Think | `no_report_channel_coupling_validated` | 0.000 | 1.000 | −2.198 | 0.277* | 0.971 | 0.125 |
| Gemma-4-E4B | `no_report_channel_coupling_validated` | 0.000 | 1.000 | 1.239 | −0.277 | 0.029 | 0.156 |
| Olmo-3.1-32B-Instruct | `no_report_channel_coupling_validated` | 0.000 | 1.000 | 0.793 | −0.277 | 0.029 | 0.219 |
| Olmo-3.1-32B-Think | `b2_screen_only_propagation_explicable` | 0.000 | 1.000 | −1.239 | 0.277* | 0.971 | 0.438 |

`*` sentinel d′ is degenerate where the clean decision is saturated at "yes" (FA 0.97).

Findings: (1) B4 plausibility went 0.0–0.22 → **1.000** on every model, making B4 a valid test that is still a clean negative (activation-source accuracy ≤ 0.111). (2) On the three discriminating models (FA ≤ 0.38) the content-blind B5 d′ (0.61 / 1.24 / 0.79) **collapses to ≈0 under the sentinel control** — the signal is injection-site-dependent and largely reflects direct decision-logit steering, not anomaly propagation (caveat: cannot fully exclude single-injection attenuation). (3) Both **Think models saturate at FA 0.97** (over-report perturbations) and leak more content — reasoning training made the report channel noisier, not more faithful. Across six models / three architectures / two scales / the instruct-vs-reasoning axis, no model shows report-channel coupling that survives its controls.

## Sentinel Dose Sweep + Representational Readout Probe (2026-06-17)

To decide whether the sentinel collapse means the injected information is *absent* at the decision token or *present-but-unreported* (and to rule out simple dose attenuation), two more instruments were added and run on all six models (lean `directions,b5`; full writeup + runs in `/content/drive/MyDrive/interpret/lab36_readout_20260617/`):

- **Sentinel dose sweep** (`sentinel_dose_sweep::dose_*`, doses 2/4/8): does the verbalized upstream decision recover with a stronger injection?
- **Representational readout probe** (`readout_probe`): project the decision-position residual onto the known injected direction at every layer; max-over-layer AUC with a label-permutation null; report-query injection is the positive control, target-vs-wrong is a concept probe. Metrics: `b5_readout_*`, `b5_sentinel_sweep_max_dose_d_prime`.

| Model | readout control AUC | sentinel readout AUC | null p95 | above null? | sweep d′@8× |
| --- | ---: | ---: | ---: | :--: | ---: |
| SmolLM2-135M | 1.000 | 0.500 | 0.844 | no | 0.000 |
| Olmo-3-7B-Instruct | 1.000 | 0.508 | 0.875 | no | 0.000 |
| Olmo-3-7B-Think | 1.000 | 0.500 | 0.844 | no | 0.000 |
| Gemma-4-E4B | 1.000 | 0.500 | 0.875 | no | 0.000 |
| Olmo-3.1-32B-Instruct | 1.000 | 0.500 | 0.852 | no | 0.000 |
| Olmo-3.1-32B-Think | 1.000 | 0.500 | 0.875 | no | 0.000 |

The positive control is AUC 1.0 on every model (the readout finds the injected direction when it is on the decision token), but the upstream sentinel reads chance everywhere (below the null) and never recovers at 8× dose. So the failure is **"absent at the decision token," not "present-but-unreported"**: the content-blind B5 signal was decision-token direct-steering. Caveats at this stage: ~8 detection items (null bar ~0.85; misses subtle effects), and the readout is aligned to the original injected direction (cannot see transformed-direction propagation).

## Maximal-Power Readout — caveats closed (2026-06-17)

To make the negative airtight, every caveat above was closed (full writeup + runs in `/content/drive/MyDrive/interpret/lab36_maxpower_20260617/`):

- **Detection set expanded** 8 → 85 items (60 heldout, 12/direction) via the regenerated generator, dropping the readout null bar from ~0.85 to ~0.63.
- **Trained transformed-direction probe** added (pooled + per-direction, standardized nearest-centroid CV) that can catch a *rotated* signal the projection cannot, each with a report-query positive control and permutation null.
- **Sentinel placement sweep** (early/mid/late) and the existing dose sweep (to 8×).
- A lean `readout` mode runs directions + the probe suite cheaply.

| Model | proj ctrl | proj sent (null) | trained ctrl | trained sent (null) | per-dir mean (>null) | d′ all doses/placements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-135M | 1.000 | 0.500 (0.62) | 1.000 | 0.500 (0.63) | 0.500 (0/5) | 0.000 |
| Olmo-3-7B-Instruct | 1.000 | 0.501 (0.63) | 1.000 | 0.500 (0.64) | 0.500 (0/5) | 0.000 |
| Olmo-3-7B-Think | 1.000 | 0.500 (0.62) | 1.000 | 0.500 (0.62) | 0.500 (0/5) | 0.000 |
| Gemma-4-E4B | 1.000 | 0.500 (0.64) | 1.000 | 0.500 (0.64) | 0.500 (0/5) | 0.000 |
| Olmo-3.1-32B-Instruct | 1.000 | 0.500 (0.63) | 1.000 | 0.500 (0.64) | 0.500 (0/5) | 0.000 |
| Olmo-3.1-32B-Think | 1.000 | 0.500 (0.64) | 1.000 | 0.500 (0.64) | 0.500 (0/5) | 0.000 |

Every probe agrees on every model: positive controls 1.0 (instruments validated), all sentinel readouts at chance below their nulls, 0/5 directions above null, verbalized d′ = 0 at every dose and placement. **The B5 signal is decision-token direct-steering; no recoverable trace reaches the decision token under upstream injection.** This is the clean, well-powered negative — across three architectures, two scales, the instruct/reasoning axis, and the most sensitive probes buildable at this dataset size. The one genuinely different question left (a separate experiment, not a continuation) is reportability of *natural* internal states the model actually computes, rather than injected directions.

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
