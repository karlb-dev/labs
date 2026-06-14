# Lab 29 - Training Dynamics and Circuit Birth

```text
Time estimate: 3-10 minutes Tier A smoke; 20-60+ minutes Tier B depending on device and prompt set
Compute tier: Tier A trains a tiny in-course transformer; Tier B uses the same instrument with more steps/examples
Dependencies: Labs 3, 4, 6, 19, 21, 26, and 27 concepts
Minimum passing artifacts: method_card.md, operationalization_audit.md, diagnostics/self_check_status.json, diagnostics/warning_summary.csv, plot_manifest.json, tables/checkpoint_circuit_summary.csv, tables/mechanism_birth_events.csv, tables/training_dynamics_evidence_matrix.csv, state/checkpoint_directions.pt
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

Those are measurement events. They are not exact learning instants. The word "birth" is useful only when it keeps its lab coat on.

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
| attention motif | ATTR | Does attention at the final token point to the previous-match-successor token? | mean attention above random attention baseline |
| feature lineage | OBS/DECODE | Do target-token centroids align with final-checkpoint centroids across depth? | cosine stability is not feature identity |
| intervention transfer | CAUSAL, scoped | Does a final-checkpoint centroid direction help earlier checkpoints more than a random direction? | allowed only after behavior and probe gates; early transfer is a caveat |

The second-pass plot upgrade adds one more measurement discipline: every plotted PNG is built from a saved `tables/figure_*_source.csv` table and registered in `plot_manifest.json`. The plot is a reading aid. The source table is the evidence ledger.

## The default tiny task

The training distribution samples triples of distinct color tokens:

```text
A B C A B -> C
```

The prompt always has length five. The target is the token after the previous occurrence of the final token. Frozen evaluation rows include train-like rows, heldout/test color combinations, and untrained relation/calendar controls.

The control rows are not trained. They ask whether the tiny training run accidentally improves unrelated task families. The audit treats **drift relative to checkpoint zero** as the failure signal, not raw random checkpoint-zero accuracy, because a tiny random classifier can get lucky on a tiny control set.

## What changed in the visualization/data pass

The first version already had the right scientific posture. The second pass strengthens the evidence contract around the plots:

| Upgrade | Why it matters |
|---|---|
| `diagnostics/run_config_snapshot.json` | Keeps model, tier, seed, prompt set, training config, thresholds, and intervention doses next to the run. |
| stable IDs | `checkpoint_id`, `example_checkpoint_id`, `probe_cell_id`, `lineage_cell_id`, and `intervention_id` make rows joinable outside the run. |
| `tables/checkpoint_split_summary.csv` | Shows accuracy and margin by split/family with `n` and standard errors. |
| `tables/checkpoint_metric_long.csv` | A tidy metric table for dashboards and atlas plots. |
| dose-response intervention sweep | Tests final-centroid direction at several scales, not just one convenient dose. |
| figure source tables | Every plot can be audited from the exact rows used to draw it. |
| `plot_manifest.json` and `tables/plot_manifest.csv` | Records figure path, source table, row count, metric, control, and claim boundary. |
| `diagnostics/warning_summary.csv/json` | Makes tiny data, fallback data, skipped plots, close controls, and failed self-checks visible. |
| `tables/failure_specimens.md/jsonl` | Keeps wrong, control-matched, or caveat-driving specimens in the reading path. |
| new plots | Adds `target_vs_control.png`, `dose_response.png`, and `paired_examples.png` so random controls and raw specimens cannot disappear inside averages. |

## What counts as evidence

A positive circuit-birth story requires several things to line up:

1. induction behavior clears the behavior threshold on heldout/test rows;
2. probe accuracy clears the real threshold and beats rotated-label control;
3. the mean previous-match-successor attention gap clears the motif threshold;
4. the final-direction intervention beats a random-direction control at the headline dose;
5. the dose sweep does not reveal that the apparent effect was one lucky scale;
6. untrained control tasks do not drift upward relative to checkpoint zero;
7. early intervention transfer is recorded as a caveat rather than a birth claim.

The lab writes this as tables, not vibes. The event order lives in `tables/mechanism_birth_events.csv`; caveats live in `tables/training_dynamics_counterexamples.csv`, `tables/failure_specimens.*`, `diagnostics/warning_summary.*`, and `operationalization_audit.md`.

## Thresholds

The thresholds are visible in `metrics.json`, `diagnostics/run_config_snapshot.json`, and the code constants:

```text
behavior accuracy gate              >= 0.75
probe accuracy gate                 >= 0.75
probe selectivity gate              >= 0.20
mean attention motif gap gate       >= 0.12
intervention-over-random gap gate   >= 0.20
control leakage gate                +0.25 accuracy and +0.20 margin drift over checkpoint zero
headline intervention dose          0.75 x median final-centroid norm
dose-response sweep                 0.00, 0.25, 0.50, 0.75, 1.00, 1.25
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
  plot_manifest.json
  results.csv                                  # alias of checkpoint_circuit_summary.csv

  diagnostics/
    data_manifest.json                         # data path/source/hash, tiny vocab, training config
    run_config_snapshot.json                   # lab-local reproducibility snapshot
    tokenization_gate.csv                      # tiny vocabulary and prompt audit
    self_check_status.json                     # tiny-model reload, no-op, determinism, and shape checks
    warning_summary.csv                        # warnings, close controls, skipped plots, fallback flags
    warning_summary.json
    safety_status.json                         # benign synthetic training scope

  tables/
    task_manifest.csv                          # selected frozen tasks and encoded tokens
    tiny_training_log.csv                      # training loss and batch accuracy
    checkpoint_behavior.csv                    # per-example margins, lens event depths, attention motif rows
    checkpoint_split_summary.csv               # per-checkpoint family/split accuracy and margin with SE
    checkpoint_probe_selectivity.csv           # centroid probe and rotated-label control by depth
    checkpoint_circuit_summary.csv             # one row per checkpoint with phase labels
    checkpoint_metric_long.csv                 # tidy checkpoint x metric table for dashboard/atlas
    mechanism_birth_events.csv                 # first threshold crossing per event
    feature_lineage.csv                        # centroid cosine lineage to final checkpoint
    intervention_transfer.csv                  # dose-response final-direction vs random additions
    training_dynamics_evidence_matrix.csv      # evidence gates and claim posture
    evidence_matrix.csv                        # standard alias
    training_dynamics_counterexamples.csv      # rows that narrow or defeat birth language
    failure_specimens.md
    failure_specimens.jsonl
    plot_reading_guide.csv                     # what each plot protects
    plot_manifest.csv                          # CSV copy of plot_manifest.json
    figure_*_source.csv                        # exact source rows for every plot

  plots/
    training_dynamics_dashboard.png
    behavior_vs_decodability_timeline.png
    target_vs_control.png
    dose_response.png
    circuit_birth_atlas.png
    depth_migration_map.png
    checkpoint_feature_lineage.png
    intervention_transfer_over_time.png
    random_model_control_panel.png
    paired_examples.png
    tiny_training_curve.png

  state/
    checkpoint_directions.pt                   # tiny checkpoints and centroid state
    checkpoint_directions_metadata.json        # human-readable metadata and depth convention
```

## How to read the run

Start with `run_summary.md`. It tells you whether the run is science-ready, whether fallback data was used, what the smallest surviving claim is, and which counterexample is most dangerous.

Then read `diagnostics/warning_summary.csv`. If it contains `self_checks_failed`, stop. If it contains `builtin_smoke_fallback`, do not ledger a science claim. If it contains close random-direction or rotated-label controls, keep that caveat in the writeup.

Then inspect the raw evidence in this order:

1. `diagnostics/self_check_status.json`: if tiny-model reload, no-op, determinism, or shape checks fail, stop.
2. `diagnostics/run_config_snapshot.json`: confirm seed, prompt set, training steps, thresholds, and dose sweep.
3. `diagnostics/tokenization_gate.csv`: confirm frozen tasks are actually representable in the tiny vocabulary.
4. `tables/tiny_training_log.csv`: confirm optimization happened.
5. `tables/checkpoint_behavior.csv`: inspect target margins, top tokens, logit-lens event depth, and attention to the induction source token.
6. `tables/checkpoint_split_summary.csv`: compare train, heldout, test, and untrained-control summaries with `n` visible.
7. `tables/checkpoint_probe_selectivity.csv`: confirm real probes beat rotated-label controls at the claimed depth.
8. `tables/intervention_transfer.csv`: check final-direction gains against random-direction gains across doses.
9. `tables/checkpoint_circuit_summary.csv`: read threshold measurements and phase labels.
10. `tables/mechanism_birth_events.csv`: read the first threshold crossing per measurement.
11. `plot_manifest.json`: connect each figure to its exact source table.
12. `tables/failure_specimens.md` and `operationalization_audit.md`: use these before writing any birth language.

## How to read the figures

Open the plots only after the warning summary and source tables. The figures are a guided route through the evidence, not a replacement for the CSVs.

| Plot | Source table | Question answered | Reading note |
|---|---|---|---|
| `training_dynamics_dashboard.png` | `figure_training_dynamics_dashboard_source.csv` | Which measurements crossed, and what caveats travel with them? | Read gates beside controls; phase labels are navigation aids. |
| `behavior_vs_decodability_timeline.png` | `figure_behavior_vs_decodability_source.csv` | Did behavior and decodability emerge together? | Error bars come from split summaries where data supports them. |
| `target_vs_control.png` | `figure_target_vs_control_source.csv` | Does the final direction beat random-direction control on raw specimens? | Points near or below the diagonal are important falsifiers. |
| `dose_response.png` | `figure_dose_response_source.csv` | Is intervention transfer scale-sensitive or one lucky dose? | The headline threshold still uses dose 0.75; the sweep is context. |
| `circuit_birth_atlas.png` | `figure_circuit_birth_atlas_source.csv` | Which gates are present at each checkpoint? | Columns use different measurement scales; use the long table for thresholds. |
| `depth_migration_map.png` | `figure_depth_migration_source.csv` | Where is target identity most decodable over time? | The heatmap is the evidence; the best-depth marker is a summary. |
| `checkpoint_feature_lineage.png` | `figure_feature_lineage_source.csv` | Do earlier centroids align with final-checkpoint centroids? | A stable cosine is a clue, not feature identity. |
| `intervention_transfer_over_time.png` | `figure_intervention_transfer_source.csv` | When does the headline final-direction addition beat random? | Early transfer is a caveat about shared coordinates/logit pressure. |
| `random_model_control_panel.png` | `figure_random_control_source.csv` | Did untrained controls drift relative to checkpoint zero? | Drift, not raw checkpoint-zero luck, is the control question. |
| `paired_examples.png` | `figure_paired_examples_source.csv` | Which specimens support or contradict the aggregate? | Specimens come after aggregate plots, not before. |
| `tiny_training_curve.png` | `figure_tiny_training_curve_source.csv` | Did optimization happen? | Batch accuracy is not frozen heldout/test behavior. |

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

## Expected Tier A behavior

Tier A is a plumbing and evidence-contract run. It should:

- train a tiny model without relying on the bench-loaded Hugging Face model;
- write all tables even with `--no-plots`;
- write `plot_manifest.json` and figure source tables even when plots are skipped;
- make small-`n` warnings visible rather than smoothing them into confidence;
- possibly produce an honest negative result.

Tier A does not need to support a broad training-dynamics claim. It proves the microscope, the artifact schema, and the reading path.

## Expected Tier B behavior

Tier B uses more steps/examples under the same lab-local tiny transformer. A useful Tier B run should show whether the event order is stable enough to discuss:

```text
checkpoint 0      absent_or_random
checkpoint J      decodable_before_behavioral, or behavior before decodability
checkpoint K      behavior gate clears on heldout/test rows
checkpoint L      previous-match-successor attention motif clears, or fails honestly
checkpoint M      final-direction intervention clears after behavior/probe gates
```

That would support threshold-order language. It would still not support exact birth language.

## Honest negative outcomes

| Pattern | Interpretation |
|---|---|
| behavior clears but motif does not | The tiny model may use a non-attention or positional shortcut, or the motif metric may be too narrow. |
| probe clears before behavior | The target information is readable before it is connected to behavior. |
| intervention works too early | Shared coordinates or direct logit pressure explain the intervention better than circuit birth. |
| random direction matches final direction | The intervention handle is not specific enough at that checkpoint/dose. |
| rotated-label probe tracks real probe | The probe evidence is not selective enough. |
| untrained controls drift upward | Task-specific training story is not specific enough. |
| final checkpoint fails behavior | The tiny training run did not learn the task under this configuration. |
| dose-response is non-monotone or only high-dose positive | The direction may be acting as a blunt logit-pressure knob rather than a circuit handle. |

A negative result can be the best result in this lab. The method teaches students how to not hallucinate a training story.

## What this lab can claim

Allowed, if gates pass:

```text
OBS + DECODE + ATTR + scoped CAUSAL: In this controlled tiny checkpoint sequence, behavior crossed threshold at step K, decodability at step J, the previous-match-successor attention motif at step L, and intervention transfer at step M, under rotated-label, random-direction, checkpoint-zero, dose-response, and untrained-control audits.
```

Allowed, if controls fail:

```text
AUDIT: This run did not validate a circuit-birth story. The strongest counterexample was <control leakage / probe control close / random direction match / intervention too early / motif absent>.
```

## What this lab cannot claim

Do not write:

```text
The model first learned induction at step K.
This tiny model proves how large pretrained LLMs learn induction.
The centroid probe found the true feature.
The attention motif is the whole circuit.
The intervention proves the circuit existed at checkpoint zero.
The dose-response curve proves a specific circuit edge.
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
| intervention works at checkpoint zero | final direction is a coordinate-aligned logit push | `training_dynamics_counterexamples.csv` and `target_vs_control.png` |
| random direction tracks target direction | activation addition may be nonspecific | `target_vs_control.png`, `dose_response.png`, and `intervention_transfer.csv` |
| plot looks strong but warning summary complains | figure is aggregating over caveat rows | `diagnostics/warning_summary.csv` and the figure source table |
| plots skipped | `--no-plots` was passed | tables, figure source tables, and `plot_manifest.json` are still the source of truth |

## Suggested extensions

1. Add a stricter induction dataset that varies prompt length and distractor positions so position-2 shortcuts die.
2. Add head ablation at each checkpoint and record an attention-head causal effect, not only an attention motif.
3. Train two independent seeds and compare event-order stability.
4. Add a tiny relation-learning phase after induction and test representation reuse versus overwrite.
5. Run the same artifact schema on a real checkpoint sequence such as Pythia, with checkpoint metadata and token counts recorded.
6. Add a true heldout intervention-transfer family selected after depth/direction choice.

## Writeup questions

1. Which event crossed first: behavior, decodability, motif, or intervention transfer?
2. Did heldout/test behavior cross at the same time as train-like behavior?
3. Did the rotated-label control stay below the real probe?
4. Did the attention motif measure the previous match or the previous-match successor? Why does that distinction matter?
5. Did intervention transfer appear before the behavior/probe prerequisites? If yes, what claim does that block?
6. Did the final direction beat random on raw specimens, or only in the mean?
7. Does the dose-response curve make the intervention handle more credible or more suspicious?
8. Did untrained controls drift relative to checkpoint zero?
9. What exact ledger sentence can you defend without implying an exact learning instant?

## Ledger templates

Positive, after controls pass:

```text
[L29-C1] OBS + DECODE + ATTR + CAUSAL | In checkpoint sequence <C>, induction behavior crossed <threshold> at step <K>, centroid decodability crossed at step <J>, the previous-match-successor attention motif crossed at step <L>, and final-direction intervention transfer crossed at step <M> under rotated-label, random-direction, checkpoint-zero, dose-response, and untrained-control audits. This is a threshold-order claim, not an exact learning-instant claim.
Artifact: runs/<run>/tables/mechanism_birth_events.csv | Falsifier: the ordering fails across seeds or an untrained control/random-direction/rotated-label control crosses the same threshold.
```

Negative, still useful:

```text
[L29-N1] AUDIT | Lab 29 did not validate circuit-birth language for <run>: <counterexample> explained or narrowed the result. The supported claim is limited to <behavior threshold / decodability handle / failed motif / early intervention caveat>.
Artifact: runs/<run>/operationalization_audit.md | Falsifier: a preregistered rerun clears behavior, decodability, motif, intervention, and control-drift gates across seeds.
```

## Safety and scope

Lab 29 trains a tiny transformer from scratch on benign synthetic token sequences. It does not download or fine-tune a deployment model, does not generate harmful text, and does not claim anything about private data, tool use, deception, or real-world model training histories.

The risk in this lab is epistemic, not behavioral: making a glossy time-lapse plot that turns threshold crossings into a folk story. The artifacts are designed to prevent that. Read the warnings. Read the specimens. Let the controls be rude.
