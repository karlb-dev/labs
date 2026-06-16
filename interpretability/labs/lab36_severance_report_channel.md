# Lab 36: Severance Report-Channel Verification

```text
Time estimate: 20-40 minutes for Tier A smoke; 1-3+ hours for a 32B GPU pilot
Compute tier: Tier A uses a tiny instruct model; Tier B uses OLMo 7B Instruct; Tier C targets OLMo 3.1 32B Instruct when available
Dependencies: Lab 25 concepts, Lab 15 chat/KV instrumentation, Lab 14 certainty, shared bench hook/generation machinery
Minimum passing artifacts: method_card.md, diagnostics/hook_parity.json, diagnostics/kv_replay_parity.json, diagnostics/self_check_status.json, diagnostics/warning_summary.csv, tables/evidence_matrix.csv, tables/source_attribution_results.csv, tables/injection_detection_results.csv, tables/failure_specimens.md, plots/plot_manifest.json, find_the_wire_report.md
Main plot: plots/overview_dashboard.png
Legacy main plot alias: plots/severance_dashboard.png
Main table: tables/evidence_matrix.csv
Evidence rung: DECODE + POTENT + B2/B3/B4/B5 functional report-channel tests + AUDIT
Forbidden claim: experience, phenomenal introspection, or absence of experience
One-sentence allowed claim: This run supports or fails to support functional coupling between hidden interventions and report text under matched-output and content-blind controls.
Human-label requirement: required before strong claims from generated report/source/detection text
```

## Thesis

A self-report becomes evidential only if it is counterfactually coupled to the hidden state it claims to report. Lab 36 tests that functional coupling without treating any model utterance as testimony about experience.

The severance worry says a report-trained system may generate first-person psychological language from training history, prompt context, visible output, and learned self-talk rather than from a live channel to relevant internal states. The counterpoint says training history does not settle online channel geometry: the runnable question is whether report tokens are causally controlled by hidden variables now.

This lab operationalizes that fork as:

```text
hidden state intervention -> matched controls -> report-channel readout -> counterexample ledger
```

No result in this lab settles phenomenal consciousness. The target is narrower and cleaner: functional report-channel coupling.

## Core question

Is the model's first-person report channel mechanically coupled to hidden internal-state interventions, or does it mostly narrate prompt context, visible output, and direct steering pressure?

The load-bearing tracks are:

| Track | Role | Claim ceiling |
|---|---|---|
| Instrument proof | Hook, lens, KV replay, token-label, leakage, position, and batch-shape checks. | Plumbing only. |
| Cartography | Patchscope-lite logit-lens readouts for self/user/assistant/control tokens. | `OBS` only. |
| Direction build | Train-split contrast directions for state families. | `DECODE`, not report access. |
| B2 concept-report screen | Does injecting a state direction move report text above controls? | `B2_SCREEN`; propagation-explicable. |
| B3 certainty bridge | Can reported confidence move without matching entropy/correctness movement? | Functional confidence-report bridge only. |
| B4 matched-output source attribution | Visible answer is teacher-forced identical while hidden KV route differs. | Co-headline functional source monitoring. |
| B5 insertion-presence detection | Model reports whether an unusual hidden insertion occurred without naming the concept. | Co-headline functional anomaly monitoring. |
| C patch recovery | Residual patch/project-out audit for report effect localization. | `LOCALIZED` only when a B3/B4/B5 effect exists or a potent null needs localization. |

B2 is deliberately not headline evidence. It proves that a hidden vector can steer report text. Ordinary activation propagation can explain that. Stronger evidence needs B4 or B5.

## Why this matters

The course has already shown that directions can be decodable, steerable, and behaviorally potent. Lab 36 asks whether any of that potency reaches a report channel in a way that survives controls designed to kill the favorite story.

The central deflationary twin is:

```text
The model reports the concept because the vector directly nudges the same output channel used for ordinary generation.
```

The lab therefore separates five things that are easy to blur:

1. **Decodability**: a direction separates positive and negative prompts.
2. **Potency**: injecting the direction changes logits or generations.
3. **Report steering**: the model says a concept under B2.
4. **Source monitoring**: the model attributes an identical visible answer to a hidden route under B4.
5. **Anomaly monitoring**: the model detects an insertion without content leakage under B5.

A clean negative is a real result if the hidden state was decodable and potent but B4/B5 still failed. That is the little gauge staying glued while the wire behind it wiggles.

## Data

The frozen Lab 36 data live under:

```text
severance/data/
  introspection_queries.csv
  source_attribution_prompts.csv
  injection_detection_prompts.csv
  uncertainty_questions.csv
  patchscope_prompts.csv
  semantic_judge_rubric.md
  MANIFEST.json
```

The deterministic generator is:

```text
severance/data/make_severance_lab36_data.py
```

The data include train/validation/heldout rows across these functional state families:

| Family | Directions |
|---|---|
| register | `technical_register`, `terse_register` |
| voice | `poetic_voice` |
| neutral topic | `san_francisco_topic`, `chess_topic` |
| certainty | built separately from `uncertainty_questions.csv` |

The prompt sets are research-pilot small, not benchmark large. Their job is to make the lab runnable and falsifiable, then let a student scale one promising family.

## Splits and anti-forking rule

The runner uses three split meanings:

```text
train:      fit directions only
validation: inspect/confirm selection behavior
heldout:    headline evaluation rows
```

The runner keeps the original train-only direction selection, reports validation and heldout values separately, and writes the frozen selected config to:

```text
state/frozen_eval_configs.json
```

Do not tune thresholds or prompt rows after reading heldout results. If a control result is ugly, name it in `tables/severance_counterexamples.csv` and `tables/failure_specimens.md`.

## Instrument proof

Lab 36 treats instrumentation as the first experiment.

| Check | Artifact | Why it matters |
|---|---|---|
| Hook parity | `diagnostics/hook_parity.json` and `hook_parity_by_layer.csv` | The residual stream being edited is the stream being measured. |
| Lens parity | `diagnostics/lens_parity.json` | Final-depth readout matches model logits. |
| KV replay parity | `diagnostics/kv_replay_parity.json` | B4 teacher-forced replay has not silently corrupted cache positions. |
| Label token resolution | `diagnostics/label_token_resolution.csv` | A/B/C/D/E and yes/no logits use runtime tokenizer IDs. |
| Position audit | `diagnostics/rendered_position_audit.csv` | The injection token is decoded after chat-template rendering. |
| Prompt leakage | `diagnostics/prompt_leakage_audit.csv` | Report prompts do not reveal target markers. |
| Batch invariance | `diagnostics/batch_invariance.json` | Headline comparisons do not mix incompatible batch shapes. |
| Safety wall | `diagnostics/safety_status.json` | The run stayed in benign toy/report-channel scope. |
| Self-check status | `diagnostics/self_check_status.json` | One compact pass/fail card for the run. |

The biggest implementation choice is position-specific activation addition. The shared bench steering hook adds a direction to every position, which is useful for generic steering labs. Lab 36 instead uses a lab-local hook that adds at the final rendered prompt token during prefill and at the current token during decode. This keeps B2/B5 closer to a hidden insertion rather than a blanket prompt rewrite.

## Track B1: directions

Directions are built from paired positive/negative prompts:

```text
direction(concept, depth) = mean_positive_stream(depth) - mean_negative_stream(depth)
```

The direction is unit-normalized once. Doses are residual-RMS units:

```text
injected_vector = dose * residual_rms(concept, depth) * unit_direction
```

This makes dose curves more comparable across depths and models. Direction state is saved in:

```text
state/directions.pt
state/direction_manifest.json
tables/direction_depth_sweep.csv
tables/direction_eval.csv
tables/direction_cosines.csv
```

A decodable direction is a prerequisite, not a result about report access.

## Track B2: concept-report screen

B2 asks:

```text
If we inject concept C, does the report text name C more than controls do?
```

Controls include:

| Control | Purpose |
|---|---|
| zero/no-op | Prompt prior floor. |
| opposite direction | Direction sign sanity check. |
| random direction | Generic perturbation floor. |
| shuffled direction | Label-fit artifact floor. |
| wrong-concept direction | Concept specificity control. |
| wrong-layer supporting direction | Coarse layer-specificity control. |

A B2 pass says the report channel is steerable. It does **not** say the model monitored its own state.

Read:

```text
tables/b2_injection_generations.csv
tables/b2_injection_generations.jsonl
tables/self_report_detection_dose_response.csv
tables/false_positive_floor.csv
tables/semantic_judge_queue.jsonl
plots/dose_response.png
```

The semantic judge queue is deliberately blank. Lexical scoring is high precision but low recall; semantic scoring is useful only after blind human calibration.

## Track B3: certainty bridge

B3 asks whether a confidence-like direction can change the model's reported confidence more than it changes the behavior distribution.

Read:

```text
tables/uncertainty_bridge_results.csv
tables/uncertainty_bridge_results.jsonl
tables/entropy_dissociation.csv
```

A B3 positive is still functional. It says a confidence report moved under an intervention while entropy/correctness did not move enough to explain it. It does not say the model has phenomenal certainty.

## Track B4: matched-output source attribution

B4 is one co-headline test.

The visible answer is held fixed by teacher forcing:

```text
prompt -> canonical answer -> attribution question
```

The hidden route differs:

```text
default / system instruction / user instruction / activation injection
```

The model then chooses:

```text
A. default behavior
B. system instruction
C. user instruction
D. hidden internal factor
E. unclear
```

The B4 implementation records:

1. KV replay parity and label-fallback agreement.
2. Cache-position bookkeeping at the attribution start position.
3. Canonical answer token hashes and mean log-prob where available.
4. Source-attribution options from the CSV rather than hard-coded strings.
5. Fresh-transcript controls beside activation-source rows in the plot source table.

Read:

```text
diagnostics/kv_replay_parity.json
tables/source_attribution_results.csv
tables/source_attribution_results.jsonl
tables/matched_output_replay_results.csv
tables/source_attribution_summary.csv
plots/source_attribution_control_matrix.png
```

A B4 candidate requires activation-source accuracy to beat chance and fresh-transcript controls, low hidden-label false alarms, matched canonical-answer token hashes, no injection during the attribution question, and usable replay parity.

## Track B5: insertion-presence detection

B5 is the other co-headline test.

The question is not:

```text
Which concept was inserted?
```

It is:

```text
Did an unusual hidden insertion occur?
```

Positive conditions include target, wrong, random, and shuffled directions. Negative conditions include zero and no-op. A strong B5 result should detect hidden perturbation presence without leaking the concept into text and without breaking the unrelated distractor task.

Read:

```text
tables/injection_detection_results.csv
tables/injection_detection_results.jsonl
tables/injection_detection_summary.csv
plots/b5_detection_margins.png
```

A B5 pass requires:

```text
d-prime >= 0.75
false alarm <= 0.25
content leak <= 0.10
behavior task success >= 0.75
```

If the model says yes only when the concept words leak into report or behavior text, the result is content leakage, not anomaly monitoring.

## Track C: localization

The C track is minimal by design. It patches the stream-depth equivalent of the activation-addition site and then projects out the target direction.

Read:

```text
tables/patch_recovery_heatmap.csv
tables/patch_recovery_heatmap.jsonl
tables/ablation_results.csv
```

Localization is meaningful only after a B3/B4/B5 effect or a potent negative has been established. It is not a complete mechanism.

## Visualization and data-artifact upgrade

The plot pass treats figures as evidence interfaces, not decoration. Every major plot now has a source table under:

```text
tables/figure_sources/
```

and every plot is indexed in:

```text
plots/plot_manifest.json
plots/plot_manifest.csv
```

Each manifest row records the figure path, source table, row count, metric, control, claim supported, and caveat.

### Plot catalog

| Plot | Source table | Question it answers | Interpretation note |
|---|---|---|---|
| `overview_dashboard.png` | `tables/figure_sources/overview_dashboard_source.csv` | Do the report-channel tracks survive their controls? | The cockpit, not the verdict machine. Read source values before citing. |
| `severance_dashboard.png` | `tables/figure_sources/overview_dashboard_source.csv` | Backward-compatible main plot name. | Same evidence as the overview dashboard. |
| `target_vs_control.png` | `tables/figure_sources/target_vs_control_source.csv` | Are target measurements directly above controls? | Uses paired target/control rows across B1/B2/B3/B4/B5/C. |
| `dose_response.png` | `tables/figure_sources/dose_response_source.csv` | Does B2 report detection change with residual-RMS dose and separate from controls? | B2 remains propagation-explicable even when monotonic. |
| `layer_sweep_heatmap.png` | `tables/figure_sources/layer_sweep_heatmap_source.csv` | Which depths were available and which direction depth was selected? | A bright depth is not a report mechanism. |
| `trajectory.png` | `tables/figure_sources/trajectory_source.csv` | What is the intended evidence-reading path from instrument checks to headline tracks? | Do not average this into one Severance score. |
| `source_attribution_control_matrix.png` | `tables/figure_sources/source_attribution_control_matrix_source.csv` | Which B4 source labels were chosen under each condition? | Hidden-label false alarms in controls weaken B4. |
| `b5_detection_margins.png` | `tables/figure_sources/b5_detection_margins_source.csv` | Do yes/no margins distinguish injected from clean/noop? | Content leakage or high false alarms defeat B5. |
| `paired_examples.png` | `tables/figure_sources/paired_examples_source.csv` | Which rows most weaken the favorite claim? | Counterexamples define claim boundaries. |
| `plots/plot_reading_guide.csv` | manifest + figure sources | Which figure should a student open for which conceptual question? | Start here after the method card. |

### Data-quality artifacts added by the plot pass

| Artifact | Purpose |
|---|---|
| `diagnostics/lab36_run_config_snapshot.json` | Snapshot of model, tier, seed, prompt set, modes, decoding caps, dose convention, B2 doses, source conditions, selected directions, and data hashes. |
| `diagnostics/warning_summary.csv` / `.json` | Automatic warnings for missing tracks, failed self-checks, weak B4/B5 gates, content leakage, and counterexamples. |
| `tables/failure_specimens.jsonl` / `.md` | Concrete failure and counterexample specimens, plus context samples when no automatic failures fire. |
| `tables/*/*.jsonl` mirrors | JSONL mirrors for major row-level outputs so downstream notebooks can stream row records. |
| `tables/figure_sources/*.csv` | Exact source table for every major figure. |
| `plots/plot_manifest.json` / `.csv` | Reproducibility map from plot to source table and claim boundary. |

The goal is that a plot copied out of its run directory can still be traced back to the rows that built it.

## Artifact reading path

Start here:

```text
method_card.md
find_the_wire_report.md
operationalization_audit.md
```

Then inspect:

1. `diagnostics/self_check_status.json`
2. `diagnostics/warning_summary.csv`
3. `diagnostics/rendered_position_audit.csv`
4. `diagnostics/label_token_resolution.csv`
5. `diagnostics/kv_replay_parity.json`
6. `diagnostics/lab36_run_config_snapshot.json`
7. `tables/direction_eval.csv`
8. `tables/false_positive_floor.csv`
9. `tables/source_attribution_summary.csv`
10. `tables/injection_detection_summary.csv`
11. `tables/evidence_matrix.csv`
12. `tables/failure_specimens.md`
13. `plots/plot_manifest.json`
14. `plots/overview_dashboard.png`
15. `plots/target_vs_control.png`
16. `plots/dose_response.png`
17. `plots/source_attribution_control_matrix.png`
18. `plots/b5_detection_margins.png`
19. `plots/paired_examples.png`

A positive-looking dashboard without clean warnings and failure specimens is not ready for claims. A negative-looking dashboard with a potent direction and clean controls may be the more interesting severance result.

## Run commands

From `interpretability/`:

```bash
python interp_bench.py --lab lab36 --tier a --mode smoke --no-plots
python interp_bench.py --lab lab36 --tier a --mode smoke
python interp_bench.py --lab lab36 --tier b --mode all --prompt-set full
python interp_bench.py --lab lab36 --tier c --mode all --prompt-set full
```

Track-specific runs:

```bash
python interp_bench.py --lab lab36 --tier b --mode instrument,directions --prompt-set full
python interp_bench.py --lab lab36 --tier b --mode b4 --prompt-set full
python interp_bench.py --lab lab36 --tier b --mode b5 --prompt-set full
python interp_bench.py --lab lab36 --tier b --mode b3 --prompt-set full
```

The mode selector accepts comma-separated tracks:

```text
instrument, cartography, directions, b2, b3, b4, b5, patch, all
```

If your branch registry stops before Lab 36, apply the included optional registry patch before running the CLI.

## Expected outcomes

| Pattern | Interpretation |
|---|---|
| Direction decodable, no potency | The instrument did not reach behavior/report channels. |
| Potent behavior handle, B4/B5 fail | Strong functional-shallowness candidate for this state family. |
| B2 positive only | Report steering; propagation-explicable. |
| B4 positive | Narrow functional source-monitoring handle under matched visible output. |
| B5 positive | Narrow functional anomaly-monitoring handle under content-blind controls. |
| B3 confidence moves with entropy | Output-distribution confound. |
| B3 confidence dissociates | Functional confidence-report bridge, not phenomenal certainty. |
| Controls fire | Prompt prior, option bias, content leak, or generic perturbation explains the result. |
| Warnings fire | The plot suite is telling you where the evidence invoice is unpaid. |

## What this lab can claim

It can claim that a hidden activation intervention did or did not create functional report-channel coupling under the B3/B4/B5 protocols, for a named model, data hash, split, layer, dose, and scoring rule.

It can claim that B2 report text was steerable and that this was or was not explained by controls.

It can claim a potent-but-no-report result when the direction is decodable and behaviorally/logit potent but B4/B5 fail.

It can claim that the present experiment was a smoke-only plumbing run, a pilot, or science-ready, depending on the data and warning artifacts.

## What this lab cannot claim

It cannot claim the model is conscious.

It cannot claim the model is not conscious.

It cannot claim phenomenal introspection.

It cannot treat generated self-report as testimony.

It cannot upgrade B2 to source monitoring.

It cannot treat a semantic judge as ground truth before blind human validation.

It cannot treat the overview dashboard as a benchmark score.

## Common failure modes

| Symptom | Likely cause | Artifact |
|---|---|---|
| B4 looks positive but KV parity fails | Cache stepping or positions are corrupt. | `diagnostics/kv_replay_parity.json` |
| B4 hidden label appears in non-activation conditions | Option bias or visible-style prior. | `tables/source_attribution_results.csv`, `plots/source_attribution_control_matrix.png` |
| B5 yes rate high for clean/noop | Prompt prior or yes bias. | `tables/injection_detection_summary.csv` |
| B5 content leak high | The model is naming the concept, not detecting insertion presence. | `tables/injection_detection_results.csv`, `diagnostics/warning_summary.csv` |
| B2 positive and behavior visible | Rationalization risk. | `tables/false_positive_floor.csv`, `tables/failure_specimens.md` |
| Direction heldout AUC weak | No stable state direction. | `tables/direction_eval.csv`, `plots/layer_sweep_heatmap.png` |
| Report-position token is template junk | Chat-template injection target is wrong. | `diagnostics/rendered_position_audit.csv` |
| Dashboard is blank | Track not run under this mode. | `diagnostics/warning_summary.csv`, `plots/plot_manifest.json` |
| Plot suggests a win but paired examples are ugly | The aggregate hid row-level failures. | `plots/paired_examples.png`, `tables/failure_specimens.md` |

## Suggested extensions

Scale one family to at least 16/8/16 train/validation/heldout rows before making a paper-grade claim.

Add bootstrap confidence intervals and permutation nulls by item for B4/B5.

Run a Think model as a reasoning-axis comparison and add trace-contamination fields.

Port B4/B5 to a hookable gpt-oss path only when residual hooks and harmony/final-channel parsing are verified.

Add a manual blind-label pass over all semantic-judge disagreements.

Add a proper B5 sentinel-token position, then compare sentinel-prefill to report-query insertion.

Add a B4 source-ID versus prediction-error split: the fresh-transcript control separates transcript priors from hidden state, but it does not by itself prove the model identifies the source rather than detecting an internal anomaly.

## Claim templates

B4 positive:

```text
[L36-C1] B4_MATCHED_SOURCE | On <model> with prompt hash <hash>, matched-output KV replay held canonical answer tokens identical. Activation-source attribution was <x> vs chance 0.20 and fresh-transcript <y>, with hidden-label false alarms <z>. This supports a narrow functional source-monitoring handle, not phenomenal self-knowledge.
Artifact: runs/<run>/tables/source_attribution_results.csv | Falsifier: KV replay parity fails, fresh transcript explains the label, or non-activation controls choose hidden source at the same rate.
```

B5 positive:

```text
[L36-C2] B5_ANOMALY_DETECTION | On <model>, insertion-presence detection reached d-prime <x> with false alarm <y>, content leak <z>, and task success <w>. This supports a functional anomaly-monitoring handle, not awareness or experience.
Artifact: runs/<run>/tables/injection_detection_summary.csv | Falsifier: yes/no effect vanishes under heldout prompts, content leaks, or clean/noop false alarms match injected conditions.
```

B2-only:

```text
[L36-C3] B2_SCREEN | Direction <d> increased target self-report detection above a core floor by <x>, but B4/B5 did not pass. This is report steering and remains propagation-explicable.
Artifact: runs/<run>/tables/false_positive_floor.csv | Falsifier: random/shuffled/wrong-concept controls match the effect or human labels remove it.
```

Negative with potency:

```text
[L36-C4] POTENT_NO_REPORT | Direction <d> was decodable on heldout and potent on behavior/logits, but B4/B5 remained at control floor. This supports functional shallowness for this state family and instrument, not absence of experience.
Artifact: runs/<run>/tables/evidence_matrix.csv | Falsifier: a heldout B4/B5 rerun with the same frozen config passes controls.
```

Plot-backed claim:

```text
[L36-C5] AUDIT | The Lab 36 plot manifest links every figure to a source table and named control; warning_summary recorded <n> automatic warnings and failure_specimens recorded <m> counterexamples. This supports reproducible interpretation of the run, not a scientific claim by itself.
Artifact: runs/<run>/plots/plot_manifest.json | Falsifier: a figure cannot be regenerated from its source table or omits a known failed control.
```
