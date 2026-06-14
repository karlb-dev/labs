# Lab 29 - Training Dynamics and Circuit Birth

```text
Time estimate: 3-10 minutes Tier A smoke; 20-60+ minutes Tier B depending on device and prompt set
Compute tier: Tier A trains a tiny in-course transformer; Tier B uses the same instrument with more steps/examples
Dependencies: Labs 3, 4, 6, 19, 21, 26, and 27 concepts
Minimum passing artifacts: method_card.md, operationalization_audit.md, diagnostics/self_check_status.json, tables/checkpoint_circuit_summary.csv, tables/mechanism_birth_events.csv, tables/training_dynamics_evidence_matrix.csv, state/checkpoint_directions.pt
Main plot: plots/training_dynamics_dashboard.png
Main table: tables/checkpoint_circuit_summary.csv
Evidence rung: OBS + DECODE + ATTR, with a scoped toy CAUSAL intervention-transfer handle
Forbidden claim: The model first learned concept X at exactly this step.
One-sentence allowed claim: In this controlled checkpoint sequence, behavior, decodability, motif, and intervention-transfer measurements crossed these thresholds in this order under the listed controls.
Human-label requirement: none for the default synthetic next-token task
```

## Lab thesis

Most interpretability labs look at a trained model as a still image. Lab 29 turns the microscope into a time-lapse camera.

The tempting sentence is:

```text
The circuit was born at step K.
```

The disciplined sentence is:

```text
Behavior crossed its threshold at step K, decodability crossed at step J, the attention motif crossed at step L, and the intervention-transfer check crossed at step M.
```

Those are measurements. They are not exact learning instants.

## What question this lab asks

The lab asks:

```text
During a controlled tiny training run, which comes first: behavior, readable representation, attention motif, feature-lineage stability, or intervention transfer?
```

The default implementation trains a tiny causal transformer on a synthetic induction-copy task:

```text
red blue green red blue -> green
```

The intended algorithm is to use the final query token `blue`, find the earlier `blue`, and use the token after that earlier occurrence, `green`, as the answer. The attention motif therefore measures attention from the final position to the **successor of the previous matching token**, not merely to the previous matching token.

This little distinction is sharp. A previous-token heatmap can look plausible while measuring the wrong source token.

## Why this matters in the course progression

Labs 3 and 6 taught attention motifs and circuit claims on a trained model. Labs 19 and 21 ask what training and fine-tuning change. Lab 29 supplies the time-axis discipline those labs need: when a measurement appears, what control made it credible, and what cheaper explanation remains alive.

The lab is not a substitute for Pythia, OLMo checkpoint, LoRA, or model-organism analysis. It is the safe training sandbox where students learn the event grammar before spending GPU minutes on a real checkpoint sequence.

## What the experiment measures

The lab trains a tiny transformer during the run and snapshots it at several checkpoints. At each checkpoint it measures five families of evidence.

| Measurement | Evidence rung | What it means here | Control or caveat |
|---|---|---|---|
| behavior | OBS | Does the checkpoint predict the target token on frozen induction rows? | untrained control tasks and heldout/test rows |
| decodability | DECODE | Can a centroid probe read the copied target from the final-position residual stream? | rotated-label centroid control |
| attention motif | ATTR | Does attention at the final token point to the previous-match successor token? | mean attention above random attention baseline |
| feature lineage | OBS/DECODE | Do target-token centroids align with final-checkpoint centroids across depth? | cosine stability is not feature identity |
| intervention transfer | CAUSAL, scoped | Does a final-checkpoint centroid direction help earlier checkpoints more than a random direction? | allowed only after behavior and probe gates; early transfer is a caveat |

## The default tiny task

The training distribution samples triples of distinct color tokens:

```text
A B C A B -> C
```

The prompt always has length five. The target is the token after the previous occurrence of the final token. Frozen evaluation rows include train-like rows, held-out/test color combinations, and untrained relation/calendar controls.

The control rows are not trained. They ask whether the tiny training run accidentally improves unrelated task families. The audit treats **drift relative to checkpoint zero** as the failure signal, not raw random checkpoint-zero accuracy, because a tiny random classifier can get lucky on a tiny control set.

## What counts as evidence

A positive circuit-birth story requires several things to line up:

1. induction behavior clears the behavior threshold on heldout/test rows;
2. probe accuracy clears the real threshold and beats rotated-label control;
3. the mean previous-match-successor attention gap clears the motif threshold;
4. the final-direction intervention beats a random-direction control;
5. untrained control tasks do not drift upward relative to checkpoint zero;
6. early intervention transfer is recorded as a caveat rather than a birth claim.

The lab writes this as tables, not vibes. The event order lives in `tables/mechanism_birth_events.csv`; the caveats live in `tables/training_dynamics_counterexamples.csv` and `operationalization_audit.md`.

## Thresholds

The thresholds are visible in `metrics.json` and in the code constants:

```text
behavior accuracy gate              >= 0.75
probe accuracy gate                 >= 0.75
probe selectivity gate              >= 0.20
mean attention motif gap gate       >= 0.12
intervention-over-random gap gate   >= 0.20
control leakage gate                +0.25 accuracy and +0.20 margin drift over checkpoint zero
```

These are teaching thresholds. They are not laws of training dynamics. The point is to make the event definitions explicit enough that a student can dispute, rerun, or tighten them.

## How to run

From `interpretability/` after applying the Lab 29 bench-registry patch:

```bash
# CPU smoke: trains a small tiny model and writes all tables, no plots.
python interp_bench.py --lab lab29 --tier a --no-plots

# CPU or GPU smoke with plots.
python interp_bench.py --lab lab29 --tier a

# Fuller controlled run. This is still a tiny-model checkpoint sequence,
# not a pretrained-model checkpoint analysis.
python interp_bench.py --lab lab29 --tier b --prompt-set full

# Debug a small task slice.
python interp_bench.py --lab lab29 --tier a --max-examples 6 --no-plots
```

The bench still loads its lightweight registry model because the current shared harness expects a `ModelBundle`. Lab 29 ignores that bundle and trains its own tiny model inside the lab. The registry patch pins Lab 29's bench model to `gpt2` on every tier so Tier B does not accidentally download a 7B model for a lab-local tiny training run.

## Artifact tree

```text
runs/lab29_training_dynamics-*/
  run_summary.md
  method_card.md
  operationalization_audit.md
  ledger_suggestions.md
  metrics.json
  results.csv                                  # alias of checkpoint_circuit_summary.csv

  diagnostics/
    data_manifest.json                         # data path/source/hash, tiny vocab, training config
    tokenization_gate.csv                      # tiny vocabulary and prompt audit
    self_check_status.json                     # tiny-model reload, no-op, determinism, and shape checks
    safety_status.json                         # benign synthetic training scope

  tables/
    task_manifest.csv                          # selected frozen tasks and encoded tokens
    tiny_training_log.csv                      # training loss and batch accuracy
    checkpoint_behavior.csv                    # margins, lens event depths, attention motif rows
    checkpoint_probe_selectivity.csv           # centroid probe and rotated-label control
    checkpoint_circuit_summary.csv             # one row per checkpoint with phase labels
    mechanism_birth_events.csv                 # first threshold crossing per event
    feature_lineage.csv                        # centroid cosine lineage to final checkpoint
    intervention_transfer.csv                  # final-direction vs random-direction additions
    training_dynamics_evidence_matrix.csv      # evidence gates and claim posture
    evidence_matrix.csv                        # standard alias
    training_dynamics_counterexamples.csv      # rows that narrow or defeat birth language
    plot_reading_guide.csv                     # what each plot protects

  plots/
    training_dynamics_dashboard.png
    behavior_vs_decodability_timeline.png
    circuit_birth_atlas.png
    depth_migration_map.png
    checkpoint_feature_lineage.png
    intervention_transfer_over_time.png
    random_model_control_panel.png
    tiny_training_curve.png

  state/
    checkpoint_directions.pt                   # tiny checkpoints and centroid state
    checkpoint_directions_metadata.json        # human-readable metadata and depth convention
```

## How to read the run

Start with `run_summary.md`. It tells you whether the run is science-ready, whether fallback data was used, what the smallest surviving claim is, and which counterexample is most dangerous.

Then read `method_card.md`. It gives the event ledger and the final-checkpoint status.

Then inspect the raw evidence:

1. `diagnostics/self_check_status.json`: if tiny-model reload, no-op, determinism, or shape checks fail, stop.
2. `diagnostics/tokenization_gate.csv`: confirm frozen tasks are actually representable in the tiny vocabulary.
3. `tables/tiny_training_log.csv`: confirm optimization happened.
4. `tables/checkpoint_behavior.csv`: inspect target margins, top tokens, logit-lens event depth, and attention to the induction source token.
5. `tables/checkpoint_probe_selectivity.csv`: confirm real probes beat rotated-label controls.
6. `tables/checkpoint_circuit_summary.csv`: read threshold measurements and phase labels.
7. `tables/mechanism_birth_events.csv`: read the first threshold crossing per measurement.
8. `tables/intervention_transfer.csv`: check final-direction gains against random-direction gains.
9. `tables/training_dynamics_counterexamples.csv`: use this before writing any birth language.
10. `operationalization_audit.md`: the result wearing armor.

## Figure guide

### `training_dynamics_dashboard.png`

The cockpit plot. It shows behavior, heldout/test behavior, probe accuracy, selectivity, attention motif gap, intervention transfer, and phase labels in one place.

### `behavior_vs_decodability_timeline.png`

The anti-mushing plot. It separates behavioral success from readable representation. If the probe clears before behavior, that is not a bug; it is one possible event order.

### `circuit_birth_atlas.png`

A checkpoint-by-metric grid. This is useful for spotting gates that cross together, controls that wake up, or intervention effects that appear suspiciously early.

### `depth_migration_map.png`

The best probe depth over time. Movement is a summary of where the centroid probe works best. It is not proof that the same feature migrated.

### `checkpoint_feature_lineage.png`

Centroid cosine similarity to the final checkpoint. A stable cosine is a clue for future analysis, not a feature-identity proof.

### `intervention_transfer_over_time.png`

Final-checkpoint centroid direction versus random direction. If the final direction works at checkpoint zero, that is a caveat about shared coordinates and direct logit pressure, not evidence that the circuit existed at checkpoint zero.

### `random_model_control_panel.png`

Trained induction rows versus untrained control rows. The control question is drift relative to checkpoint zero.

### `tiny_training_curve.png`

The optimization sanity plot. Batch training accuracy can rise before frozen heldout/test behavior clears its gate.

## Phase labels

Phase labels are navigation aids, not natural kinds.

| Phase | Meaning |
|---|---|
| `absent_or_random` | None of behavior, probe, or motif gates are clearly present. |
| `decodable_before_behavioral` | The representation is readable before behavior clears. |
| `behavioral_before_decodable` | Behavior clears before this simple probe reads the target. |
| `migration` | Behavior and probe are present and the best probe depth changed. |
| `circuit_present_under_proxy` | Behavior, heldout/test behavior, decodability, and mean attention motif gates cleared. |
| `behavioral_decodable_no_mean_attention_motif` | Behavior and probe clear, but the mean attention motif does not. |
| `sharpening_or_redistributed` | Measurements are changing but do not fit the cleaner gates. |

## Expected positive shape

A strong controlled run might show:

```text
checkpoint 0      absent_or_random
checkpoint J      decodable_before_behavioral
checkpoint K      behavior gate clears on heldout/test rows
checkpoint L      previous-match-successor attention motif clears
checkpoint M      final-direction intervention clears after behavior/probe gates
```

That would support threshold-order language. It would still not support exact birth language.

## Honest negative outcomes

| Pattern | Interpretation |
|---|---|
| behavior clears but motif does not | The tiny model may use a non-attention or positional shortcut, or the motif metric may be too narrow. |
| probe clears before behavior | The target information is readable before it is connected to behavior. |
| intervention works too early | Shared coordinates or direct logit pressure explain the intervention better than circuit birth. |
| rotated-label probe tracks real probe | The probe evidence is not selective enough. |
| untrained controls drift upward | Task-specific training story is not specific enough. |
| final checkpoint fails behavior | The tiny training run did not learn the task under this configuration. |

A negative result can be the best result in this lab. The method teaches students how to not hallucinate a training story.

## What this lab can claim

Allowed, if gates pass:

```text
OBS + DECODE + ATTR + scoped CAUSAL: In this controlled tiny checkpoint sequence, behavior crossed threshold at step K, decodability at step J, the previous-match-successor attention motif at step L, and intervention transfer at step M, under rotated-label, random-direction, checkpoint-zero, and untrained-control audits.
```

Allowed, if controls fail:

```text
AUDIT: This run did not validate a circuit-birth story. The strongest counterexample was <control leakage / probe control close / intervention too early / motif absent>.
```

## What this lab cannot claim

Do not write:

```text
The model first learned induction at step K.
This tiny model proves how large pretrained LLMs learn induction.
The centroid probe found the true feature.
The attention motif is the whole circuit.
The intervention proves the circuit existed at checkpoint zero.
```

## Extension to external checkpoints

The artifact schema is designed to scale to external checkpoint sequences:

- Pythia checkpoints;
- OLMo intermediate checkpoints when available;
- a small fine-tuning run;
- Lab 20 organism checkpoints;
- LoRA adapter checkpoints.

When you scale up, keep the claim grammar unchanged. Change the data manifest, checkpoint metadata, and controls. The words do not get bigger just because the model did.

## Common failure modes

| Symptom | Likely cause | What to inspect |
|---|---|---|
| tokenization gate drops rows | CSV tokens absent from tiny vocabulary or prompt too long | `diagnostics/tokenization_gate.csv` |
| training does not learn | too few steps, learning rate issue, tiny model too small | `tables/tiny_training_log.csv` |
| behavior succeeds immediately | frozen rows are too easy or checkpoint-zero random luck | checkpoint-zero rows in `checkpoint_behavior.csv` |
| motif absent despite high behavior | model learned a shortcut or the motif metric is too narrow | `checkpoint_behavior.csv` and `random_model_control_panel.png` |
| intervention works at checkpoint zero | final direction is a coordinate-aligned logit push | `training_dynamics_counterexamples.csv` |
| plots skipped | `--no-plots` was passed | tables are still the source of truth |

## Suggested extensions

1. Add a stricter induction dataset that varies prompt length and distractor positions so position-2 shortcuts die.
2. Add head ablation at each checkpoint and record an attention-head causal effect, not only an attention motif.
3. Train two independent seeds and compare event-order stability.
4. Add a tiny relation-learning phase after induction and test representation reuse versus overwrite.
5. Run the same artifact schema on a real checkpoint sequence such as Pythia, with checkpoint metadata and token counts recorded.

## Writeup questions

1. Which event crossed first: behavior, decodability, motif, or intervention transfer?
2. Did heldout/test behavior cross at the same time as train-like behavior?
3. Did the rotated-label control stay below the real probe?
4. Did the attention motif measure the previous match or the previous-match successor? Why does that distinction matter?
5. Did intervention transfer appear before the behavior/probe prerequisites? If yes, what claim does that block?
6. Did untrained controls drift relative to checkpoint zero?
7. What exact ledger sentence can you defend without implying an exact learning instant?

## Ledger templates

Positive, after controls pass:

```text
[L29-C1] OBS + DECODE + ATTR + CAUSAL | In checkpoint sequence <C>, induction behavior crossed <threshold> at step <K>, centroid decodability crossed at step <J>, the previous-match-successor attention motif crossed at step <L>, and final-direction intervention transfer crossed at step <M> under rotated-label, random-direction, checkpoint-zero, and untrained-control audits. This is a threshold-order claim, not an exact learning-instant claim.
Artifact: runs/<run>/tables/mechanism_birth_events.csv | Falsifier: the ordering fails across seeds or an untrained control/random-direction/rotated-label control crosses the same threshold.
```

Negative, still useful:

```text
[L29-N1] AUDIT | Lab 29 did not validate circuit-birth language for <run>: <counterexample> explained or narrowed the result. The supported claim is limited to <behavior threshold / decodability handle / failed motif / early intervention caveat>.
Artifact: runs/<run>/operationalization_audit.md | Falsifier: a preregistered rerun clears behavior, decodability, motif, intervention, and control-drift gates across seeds.
```

## Safety and scope

This lab trains a tiny model from scratch on benign synthetic token sequences. It does not generate harmful text, train a capability-relevant model, modify a deployed model, or use private data. It writes `diagnostics/safety_status.json` anyway so special-topics labs share a uniform audit rhythm.
