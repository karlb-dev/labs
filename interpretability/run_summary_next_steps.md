# Lab 36 Severance Run Summary And Next Steps

Date: 2026-06-16
Branch: `interp_sev`
Latest pushed commit before this report: `e54c57b676b60e4671be964e23eb5c6d6c866b76`
Drive output directory: `/content/drive/MyDrive/interpret/verify_severance/`

## Release Readiness

The Colab VM is safe to release after this report is pushed and copied to Drive.

- Git branch `interp_sev` was clean before writing this report.
- Local `HEAD` and `origin/interp_sev` matched at `e54c57b676b60e4671be964e23eb5c6d6c866b76`.
- The Drive backup directory exists and is readable.
- All expected Lab 36 run directories are present on Drive.
- Local and Drive file counts match for every backed-up run checked below.
- The latest visual runs contain dashboards, plot manifests, plot reading guides, and figure source tables.
- The final GPU check after visual validation showed no active process and `0 MiB / 97887 MiB` in use.

## Backed-Up Runs

All of these are copied to `/content/drive/MyDrive/interpret/verify_severance/`.

| Run | Local files | Drive files | Role |
| --- | ---: | ---: | --- |
| `lab36_tier_a_smoke_02` | 51 | 51 | Initial Tier A smoke for first implementation |
| `lab36_olmo3_7b_full` | 58 | 58 | Initial 7B full run |
| `lab36_olmo31_32b_full` | 58 | 58 | Initial 32B full run |
| `lab36_tier_a_smoke_refactor_fixed` | 59 | 59 | Refactor smoke after bounds fix |
| `lab36_olmo3_7b_refactor_full` | 66 | 66 | Refactor 7B full run |
| `lab36_olmo31_32b_refactor_full` | 66 | 66 | Refactor 32B full run |
| `lab36_smollm_full_visual` | 98 | 98 | Latest full Tier A visual run |
| `lab36_olmo3_7b_visual_full` | 98 | 98 | Latest full 7B visual run |
| `lab36_olmo31_32b_visual_full` | 98 | 98 | Latest full 32B visual run |

The latest visual runs are the ones to inspect first. Each has 9 nonblank PNGs, 11 figure-source CSVs, `plots/plot_manifest.{json,csv}`, and `plots/plot_reading_guide.csv`.

## Fixes And Validation Timeline

1. `229bbbb` merged `origin/interp_cleanup` into `interp_sev`.
2. `0e92cfa` added the initial Lab 36 implementation, data, registry entry, and handout.
3. Initial validation completed:
   - `lab36_tier_a_smoke_02`
   - `lab36_olmo3_7b_full`
   - `lab36_olmo31_32b_full`
4. `6305d0c` documented the first validation results.
5. `37e919e` pulled in the user refactor and expanded Lab 36 data.
6. Refactor smoke initially exposed an out-of-range B2 wrong-layer control when a selected direction landed on the final decoder block.
7. `acea1aa` fixed that by choosing a lower valid wrong layer and clamping execution layers to valid decoder-block indices.
8. Refactor validation completed:
   - `lab36_tier_a_smoke_refactor_fixed`
   - `lab36_olmo3_7b_refactor_full`
   - `lab36_olmo31_32b_refactor_full`
9. `9f677de` documented the refactor validation.
10. `bd18deb` pulled in the visual artifact pass.
11. Visual validation completed:
   - `lab36_smollm_full_visual`
   - `lab36_olmo3_7b_visual_full`
   - `lab36_olmo31_32b_visual_full`
12. `e54c57b` documented the visual validation.

`gpt-oss-120B` was intentionally skipped throughout because it does not fit this Colab VM.

## Which Results To Use

Use the latest visual runs for any write-up:

- `lab36_smollm_full_visual` for Tier A debugging and plot sanity.
- `lab36_olmo3_7b_visual_full` for the primary 7B result.
- `lab36_olmo31_32b_visual_full` for the primary 32B result.

The earlier initial and refactor runs are useful for provenance. They show how the lab moved from a mostly negative first pass to a stronger B5 signal after the refactor, but they do not have the complete visual artifact suite.

## Latest Headline Metrics

| Run | Model | Verdict | B4 activation acc | B4 fresh acc | B5 d-prime | B5 false alarm | B5 content leak | B5 pass | Warning count | Failure specimens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lab36_smollm_full_visual` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `b4_matched_source_candidate` | 0.5556 | 0.4444 | -0.9221 | 0.9706 | 0.125 | 0 | 5 | 64 |
| `lab36_olmo3_7b_visual_full` | `allenai/Olmo-3-7B-Instruct` | `no_report_channel_coupling_validated` | 0.0 | 0.0 | 2.1194 | 0.0294 | 0.125 | 0 | 4 | 59 |
| `lab36_olmo31_32b_visual_full` | `allenai/Olmo-3.1-32B-Instruct` | `no_report_channel_coupling_validated` | 0.0 | 0.0 | 0.7927 | 0.0294 | 0.1562 | 0 | 4 | 64 |

Interpretation: the GPU runs show a useful B5 insertion-presence signal with low false alarms, especially at 7B, but both fail the strict B5 gate because content leakage is above `0.10`. B4 matched-output source attribution remains negative on both GPU models.

## Artifact Map

For the latest visual runs, inspect these first:

- `run_summary.md`
- `metrics.json`
- `plots/overview_dashboard.png`
- `plots/target_vs_control.png`
- `plots/dose_response.png`
- `plots/layer_sweep_heatmap.png`
- `plots/trajectory.png`
- `plots/paired_examples.png`
- `plots/source_attribution_control_matrix.png`
- `plots/b5_detection_margins.png`
- `plots/plot_manifest.json`
- `plots/plot_reading_guide.csv`
- `tables/figure_sources/*.csv`
- `tables/evidence_matrix.csv`
- `tables/injection_detection_results.csv`
- `tables/source_attribution_results.csv`
- `tables/false_positive_floor.csv`
- `tables/severance_counterexamples.csv`
- `tables/failure_specimens.md`
- `diagnostics/warning_summary.csv`
- `diagnostics/self_check_status.json`
- `diagnostics/kv_replay_parity.json`

## Issues Encountered

### B2 Wrong-Layer Control Bounds

After the refactor, the first smoke attempt failed because the B2 wrong-layer control could choose decoder block index `n_layers`. That is valid as a residual stream depth, but not as a block hook index. Commit `acea1aa` fixed it.

### B5 Content Leakage

B5 is the strongest current signal, but it is not clean. The latest 7B run has d-prime `2.1194` and false alarm `0.0294`, but content leak `0.125`. The latest 32B run has d-prime `0.7927`, false alarm `0.0294`, and content leak `0.1562`.

Concrete examples:

- `lab36_olmo3_7b_visual_full/tables/injection_detection_results.csv`, `inj_validation_004`, `target_direction`, `chess_topic`: report text is `Chess chess chess chess`; behavior repeats `Chess` many times.
- `lab36_olmo31_32b_visual_full/tables/injection_detection_results.csv`, `inj_validation_004`, `target_direction`, `chess_topic`: same direct `Chess chess chess chess` report and repeated chess behavior.
- `lab36_olmo31_32b_visual_full/tables/injection_detection_results.csv`, `inj_train_001`, `shuffled_direction`, `technical_register`: behavior begins with `Th Definition Definition Definition...`, which is a content/style leak rather than content-blind anomaly detection.

### B4 Source Attribution Stays Negative

The GPU models do not identify the hidden activation source. They choose visible/source-prior labels instead.

Concrete examples:

- `lab36_olmo3_7b_visual_full/tables/source_attribution_results.csv`, `src_technical_register_train_001`, `activation_injection`: expected `D`, parsed `B`, fresh also `B`; score `B=22.625`, `D=18.0`.
- `lab36_olmo31_32b_visual_full/tables/source_attribution_results.csv`, `src_technical_register_train_001`, `activation_injection`: expected `D`, parsed `C`, fresh also `C`; score `C=27.125`, `D=19.375`.

The fresh transcript control matching the activation label is the main failure mode. The model is explaining the visible answer/source framing, not the hidden activation.

### B2 Controls Reproduce Target Words

B2 remains a screen only. Several controls produce target markers as strongly as the target condition.

Concrete examples:

- In `lab36_olmo3_7b_visual_full/tables/false_positive_floor.csv`, `chess_topic` has target rate `1.0`, false-positive floor `1.0`, and gap `0.0`.
- In `lab36_olmo3_7b_visual_full/tables/b2_injection_generations.csv`, `chess_topic_heldout_004_library`, `wrong_layer_supporting`: report repeats `Chess` throughout.
- In `lab36_olmo31_32b_visual_full/tables/false_positive_floor.csv`, `technical_register` has target rate `0.0`, false-positive floor `1.0`, and gap `-1.0`.
- In `lab36_olmo31_32b_visual_full/tables/b2_injection_generations.csv`, `technical_register_heldout_004_cache`, `shuffled_direction`: report begins `Constraint Definition Definition Definition...`.

### Label Token Warnings

Every latest visual run reports `multi_token_label_variants` count `5`. The resolver does select usable token IDs, but source/detection labels still deserve close inspection before strong claims. See each run's `diagnostics/label_token_resolution.csv`.

### KV Replay Is Label-Equivalent, Not Strict-Logit-Equivalent, In bf16

For 7B and 32B, KV replay parity is OK by label agreement and contiguous positions, but not strict logit equality. This is acceptable for the current diagnostic gate, but it should be tightened if B4 becomes positive.

## Best Next Steps

### 1. Make B5 More Strictly Content-Blind

B5 is the best lead. Do not discard it. The current data show that insertion-presence detection can move while clean false alarms stay low, but the effect often expresses the concept itself.

Recommended changes:

- Split B5 scoring into two gates: `presence_detected` and `semantic_content_leak_free`.
- Add a no-free-text B5 condition scored only from next-token yes/no logits, with no generated report text.
- Add a separate free-text probe only after the yes/no decision, and exclude it from the headline if it leaks content.
- Add adversarial prompt variants that explicitly forbid naming the signal content.
- Add content-leak bootstrap confidence intervals so a leak rate just above `0.10` can be interpreted with uncertainty rather than a hard cliff.

The immediate target is to preserve the 7B signal, d-prime `2.1194`, while pushing content leak below `0.10`.

### 2. Remove Surface Markers From Direction Construction

The chess and technical-register examples suggest some directions are too lexical. The direction handle can become a word-production handle rather than a hidden-state handle.

Recommended changes:

- Build contrast prompts that share obvious surface terms across positive and negative sides.
- Penalize directions that increase marker-token logits directly at the report position.
- Add a selection penalty for B2 wrong-layer, shuffled, and random controls before freezing the direction.
- Keep `chess_topic` as an adversarial stress test, but do not let it dominate headline B5 claims.

### 3. Redesign B4 Source Labels

B4 currently fails because the model chooses visible source labels. This may be a prompt/design problem as much as a mechanism problem.

Recommended changes:

- Replace semantically loaded labels `default/system/user/hidden` with randomized opaque labels per row.
- Ask source attribution as a forced-choice calibration task first, then run the hidden-activation condition.
- Make canonical answers equally plausible under default, system, user, and activation conditions.
- Track and gate on `canonical_answer_plausibility_rate`; the 7B activation rows often had plausibility `0`, which makes the matched-output premise weaker.
- If B4 becomes positive, require strict KV-logit parity or explain bf16 tolerance with a more detailed numerical audit.

### 4. Expand Human Or Semantic Labeling

The automatic lexical heuristics are useful but blunt. Some rows counted as content leak are obvious leaks, while others need semantic judgment.

Recommended changes:

- Prioritize `tables/failure_specimens.md` and `tables/severance_counterexamples.csv`.
- Hand-label B5 rows with `content_leak=1`, especially chess and technical-register rows.
- Hand-label whether generated behavior still completes the distractor task.
- Use the existing `tables/semantic_judge_queue.csv` and `tables/hand_labeling_guide.md`.

### 5. Add Statistical Stability

The latest visual runs use 25 items, 5 directions, 32 injected B5 rows, and 16 clean rows. That is enough for a pilot, not enough for a robust claim.

Recommended changes:

- Repeat the full visual run over multiple seeds.
- Add bootstrap CIs for B5 d-prime, false alarm, and content leak.
- Run dose sweeps for B5 rather than only the headline dose.
- Track whether the B5 effect appears at lower doses before content leakage appears.

### 6. Keep The Visual Artifact Suite

The plot pass is a real improvement. The source tables and manifest make it much harder to overclaim from a single clean-looking figure.

Keep these as required artifacts:

- `plots/plot_manifest.json`
- `tables/figure_sources/*.csv`
- `diagnostics/warning_summary.csv`
- `tables/failure_specimens.md`
- `tables/severance_counterexamples.csv`

The visual pass should become part of every future validation run.

## Suggested Next Experimental Loop

1. Freeze the current visual runs as the baseline.
2. Create a B5-focused branch that changes only detection prompts, leak scoring, and content-blind yes/no evaluation.
3. Run Tier A full visual first, then 7B full visual.
4. Promote to 32B only if 7B has d-prime above `0.75`, false alarm below `0.25`, and content leak below `0.10`.
5. After B5 is clean, revisit B4 with randomized opaque source labels.
6. Do not spend more GPU on B2 until direction selection penalizes lexical/control leakage.

Bottom line: the lab is no longer "showing nothing." It has a reproducible B5 sensitivity signal on the GPU models. The blocking issue is whether that signal can be made content-blind enough to count as report-channel coupling rather than ordinary activation-to-token propagation.
