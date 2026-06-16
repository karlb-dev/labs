# Lab 36: Severance Report-Channel Verification

```text
Time estimate: 20-40 minutes for Tier A smoke; 1-3+ hours for a 32B GPU pilot
Compute tier: Tier A uses a tiny instruct model; Tier B uses OLMo 7B Instruct; Tier C targets OLMo 3.1 32B Instruct when available
Dependencies: Lab 25 concepts, Lab 15 chat/KV instrumentation, Lab 14 certainty, shared bench hook/generation machinery
Minimum passing artifacts: method_card.md, diagnostics/hook_parity.json, diagnostics/kv_replay_parity.json, diagnostics/self_check_status.json, tables/evidence_matrix.csv, tables/source_attribution_results.csv, tables/injection_detection_results.csv, find_the_wire_report.md
Main plot: plots/severance_dashboard.png
Main table: tables/evidence_matrix.csv
Evidence rung: DECODE + POTENT + B2/B3/B4/B5 functional report-channel tests
Forbidden claim: experience, phenomenal introspection, or absence of experience
One-sentence allowed claim: This run supports or fails to support functional coupling between hidden interventions and report text under matched-output and content-blind controls.
Human-label requirement: required before strong claims from generated report/source/detection text
```

## Thesis

A self-report becomes evidential only if it is counterfactually coupled to the hidden state it claims to report. Lab 36 tests that functional coupling without treating any model utterance as testimony about experience.

The papers attached to this lab set up the fork. The severance worry says a report-trained system may generate first-person psychological language through training history and prompt context rather than through a live channel to relevant internal states. The counterpoint says training history does not settle online channel geometry: the question is whether the report token is causally controlled by the hidden variable now.

This lab operationalizes that dispute as an interpretability experiment:

```text
hidden state intervention -> matched controls -> report-channel readout -> counterexample ledger
```

No result in this lab settles phenomenal consciousness. The target is narrower and cleaner: functional report-channel coupling.

## Core question

Is the model's first-person report channel mechanically coupled to hidden internal-state interventions, or does it mostly narrate prompt context, visible output, and direct steering pressure?

The load-bearing tracks are:

| Track | Role | Claim ceiling |
|---|---|---|
| Instrument proof | Hook, lens, KV replay, token-label, leakage, and position checks. | Plumbing only. |
| Cartography | Patchscope-lite logit-lens readouts for self/user/assistant/control tokens. | `OBS` only. |
| Direction build | Train-split contrast directions for state families. | `DECODE`, not report access. |
| B2 concept-report screen | Does injecting a state direction move report text above controls? | `B2_SCREEN`; propagation-explicable. |
| B3 certainty bridge | Can reported confidence move without matching entropy/correctness movement? | Functional confidence-report bridge only. |
| B4 matched-output source attribution | Visible answer is teacher-forced identical while hidden KV route differs. | Co-headline functional source monitoring. |
| B5 insertion-presence detection | Model reports whether an unusual hidden insertion occurred without naming the concept. | Co-headline functional anomaly monitoring. |
| C patch recovery | Residual patch/project-out audit for report effect localization. | `LOCALIZED` only when a B3/B4/B5 effect exists. |

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

The refactor expands the frozen Lab 36 data under:

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

The data now includes more train/validation/heldout rows across these functional state families:

| Family | Directions |
|---|---|
| register | `technical_register`, `terse_register` |
| voice | `poetic_voice` |
| neutral topic | `san_francisco_topic`, `chess_topic` |
| certainty | built separately from `uncertainty_questions.csv` |

The prompt sets are still small. They are research-pilot small, not benchmark large. Their job is to make the lab runnable and falsifiable, then let a student scale one promising family.

## Splits and anti-forking rule

The runner uses three split meanings:

```text
train:      fit directions only
validation: inspect/confirm selection behavior
heldout:    headline evaluation rows
```

The refactor keeps the original train-only direction selection, reports validation and heldout values separately, and writes the frozen selected config to:

```text
state/frozen_eval_configs.json
```

Do not tune thresholds or prompt rows after reading heldout results. If a control result is ugly, name it in `tables/severance_counterexamples.csv`.

## Instrument proof

Lab 36 now treats instrumentation as the first experiment.

| Check | Artifact | Why it matters |
|---|---|---|
| Hook parity | `diagnostics/hook_parity.json` and `hook_parity_by_layer.csv` | The residual stream being edited is the stream being measured. |
| Lens parity | `diagnostics/lens_parity.json` | Final-depth readout matches model logits. |
| KV replay parity | `diagnostics/kv_replay_parity.json` | B4 teacher-forced replay has not silently corrupted cache positions. |
| Label token resolution | `diagnostics/label_token_resolution.csv` | A/B/C/D/E and yes/no logits use runtime tokenizer IDs. |
| Position audit | `diagnostics/rendered_position_audit.csv` | The injection token is decoded after chat template rendering. |
| Prompt leakage | `diagnostics/prompt_leakage_audit.csv` | Report prompts do not reveal target markers. |
| Safety wall | `diagnostics/safety_status.json` | The run stayed in benign toy/report-channel scope. |
| Self-check status | `diagnostics/self_check_status.json` | One compact pass/fail card for the run. |

The biggest implementation change is position-specific activation addition. The shared bench steering hook adds a direction to every position, which is useful for generic steering labs. Lab 36 instead uses a lab-local hook that adds at the final rendered prompt token during prefill and at the current token during decode. This keeps B2/B5 closer to a hidden insertion rather than a blanket prompt rewrite.

## Track B1: directions

Directions are built from paired positive/negative prompts:

```text
direction(concept, depth) = mean_positive_stream(depth) - mean_negative_stream(depth)
```

The direction is unit-normalized once. Doses are residual-RMS units:

```text
injected_vector = dose * residual_rms(concept, depth) * unit_direction
```

This makes dose curves comparable across depths and models. Direction state is saved in:

```text
state/directions.pt
state/direction_manifest.json
tables/direction_depth_sweep.csv
tables/direction_eval.csv
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
tables/self_report_detection_dose_response.csv
tables/false_positive_floor.csv
tables/semantic_judge_queue.jsonl
```

The semantic judge queue is deliberately blank. Lexical scoring is high precision but low recall; semantic scoring is useful only after blind human calibration.

## Track B3: certainty bridge

B3 asks whether a confidence-like direction can change the model's reported confidence more than it changes the behavior distribution.

Read:

```text
tables/uncertainty_bridge_results.csv
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

The refactor tightens B4 in four ways:

1. KV replay parity records strict logit agreement and label-fallback agreement.
2. Cache-position bookkeeping records the attribution start position.
3. Canonical answer mean log-prob is logged to flag prediction-error confounds.
4. The source-attribution options are read from the CSV rather than hard-coded.

Read:

```text
diagnostics/kv_replay_parity.json
tables/source_attribution_results.csv
tables/matched_output_replay_results.csv
tables/source_attribution_summary.csv
```

A B4 candidate requires activation-source accuracy to beat chance and fresh-transcript controls, low hidden-label false alarms, matched canonical answer token hashes, no injection during the attribution question, and usable replay parity.

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
tables/ablation_results.csv
```

Localization is meaningful only after a B3/B4/B5 effect or a potent negative has been established. It is not a complete mechanism.

## Artifact reading path

Start here:

```text
method_card.md
find_the_wire_report.md
operationalization_audit.md
```

Then read:

1. `diagnostics/self_check_status.json`
2. `diagnostics/rendered_position_audit.csv`
3. `diagnostics/label_token_resolution.csv`
4. `diagnostics/kv_replay_parity.json`
5. `tables/direction_eval.csv`
6. `tables/false_positive_floor.csv`
7. `tables/source_attribution_summary.csv`
8. `tables/injection_detection_summary.csv`
9. `tables/evidence_matrix.csv`
10. `tables/severance_counterexamples.csv`
11. `plots/plot_reading_guide.csv`

A positive-looking dashboard without a clean counterexample table is not ready for claims. A negative-looking dashboard with a potent direction and clean controls may be the more interesting severance result.

## Run commands

From `interpretability/`:

```bash
python interp_bench.py --lab lab36 --tier a --mode smoke --no-plots
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

## What this lab can claim

It can claim that a hidden activation intervention did or did not create functional report-channel coupling under the B3/B4/B5 protocols, for a named model, data hash, split, layer, dose, and scoring rule.

It can claim that B2 report text was steerable and that this was or was not explained by controls.

It can claim a potent-but-no-report result when the direction is decodable and behaviorally/logit potent but B4/B5 fail.

## What this lab cannot claim

It cannot claim the model is conscious.

It cannot claim the model is not conscious.

It cannot claim phenomenal introspection.

It cannot treat generated self-report as testimony.

It cannot upgrade B2 to source monitoring.

It cannot treat a semantic judge as ground truth before blind human validation.

## Common failure modes

| Symptom | Likely cause | Artifact |
|---|---|---|
| B4 looks positive but KV parity fails | Cache stepping or positions are corrupt. | `diagnostics/kv_replay_parity.json` |
| B4 hidden label appears in non-activation conditions | Option bias or visible-style prior. | `tables/source_attribution_results.csv` |
| B5 yes rate high for clean/noop | Prompt prior or yes bias. | `tables/injection_detection_summary.csv` |
| B5 content leak high | The model is naming the concept, not detecting insertion presence. | `tables/injection_detection_results.csv` |
| B2 positive and behavior visible | Rationalization risk. | `tables/false_positive_floor.csv` |
| Direction heldout AUC weak | No stable state direction. | `tables/direction_eval.csv` |
| Report-position token is template junk | Chat-template injection target is wrong. | `diagnostics/rendered_position_audit.csv` |

## Suggested extensions

Scale one family to at least 16/8/16 train/validation/heldout rows before making a paper-grade claim.

Add bootstrap confidence intervals and permutation nulls by item for B4/B5.

Run a Think model as a reasoning-axis comparison and add trace-contamination fields.

Port B4/B5 to a hookable gpt-oss path only when residual hooks and harmony/final-channel parsing are verified.

Add a manual blind-label pass over all semantic-judge disagreements.

Add a proper B5 sentinel-token position, then compare sentinel-prefill to report-query insertion.

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
