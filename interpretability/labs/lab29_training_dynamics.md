# Lab 29: Training Dynamics and Circuit Birth

Time estimate: 75-105 minutes for the default controlled run and audit writeup.  
Compute tier: Tier A is a tiny in-course transformer trained during the run; Tier B can extend the same tables to external checkpoint sequences.  
Dependencies: Labs 1, 3, 6, 19, 21, 26, and 27 concepts.  
Minimum passing artifacts: `tables/checkpoint_behavior.csv`, `tables/checkpoint_probe_selectivity.csv`, `tables/checkpoint_circuit_summary.csv`, `tables/mechanism_birth_events.csv`, `tables/feature_lineage.csv`, `state/checkpoint_directions.pt`, and `plots/training_dynamics_dashboard.png`.  
Main plot: `plots/training_dynamics_dashboard.png`.  
Main table: `tables/checkpoint_circuit_summary.csv`.  
Evidence rung: `OBS + DECODE + ATTR`, with a scoped toy `CAUSAL` intervention-transfer check.  
Forbidden claim: "The model first learned concept X at exactly this step."  
One-sentence allowed claim: "In this checkpoint sequence, behavior, decodability, motif, and intervention-transfer measurements crossed these thresholds in this order under the listed controls."  
Human-label requirement: none for the default synthetic next-token task.

## Why This Lab Exists

Most interpretability labs inspect a trained model as if it were a still image.
Lab 29 turns the microscope into a time-lapse camera.

The dangerous temptation is to say:

```text
The circuit was born at step K.
```

The disciplined version is:

```text
This behavior threshold crossed at step K, this probe threshold crossed at step
J, this motif became visible at step L, and this intervention began to transfer
at step M.
```

Those are measurements. They are not exact learning instants.

## Default Scope

The default implementation trains a tiny causal transformer inside the run on a
synthetic induction-copy task:

```text
red blue green red blue -> green
```

The model should learn to use the previous occurrence of the final query token
(`blue`) to predict the following token from the earlier context (`green`).

This controlled setup is not a substitute for Pythia or fine-tuning checkpoint
analysis. It is the course microscope for learning the measurement discipline
before running an expensive external sequence.

## Data

Default evaluation rows live in `data/training_dynamics_tasks.csv`.

Columns:

```text
item_id, task_family, prompt, target, distractor, split,
expected_mechanism, notes
```

The frozen rows include:

- induction-copy train, held-out, and test prompts;
- relation and calendar control prompts that are never trained;
- expected-mechanism notes so students know what the lab is trying to measure.

## What The Lab Measures

### Behavior

For each checkpoint and frozen task, the lab records:

```text
logit(target) - logit(distractor)
```

The first behavior event is a threshold crossing, not a claim about when the
model "learned" the task.

### Decodability

For each checkpoint and residual depth, the lab builds centroid probes over
synthetic induction examples. It reports probe accuracy and a shuffled-label
control.

If decodability appears before behavior, the representation is readable before
it is behaviorally useful. If behavior appears first, the probe may be using the
wrong depth, wrong feature family, or insufficient data.

### Motif Birth

For induction-copy rows, the lab measures attention from the final token to the
previous occurrence of that same token. The event threshold uses the mean
previous-match score across induction rows, not the single best head on a single
row, because best-of-many random heads can look impressive.

This is an attribution-style motif. It is not a complete circuit proof.

### Feature Lineage

The lab compares target-token centroids across checkpoints and depths. This
produces a coarse lineage map:

```text
same-depth cosine to final checkpoint
best matching final depth
```

Cosine stability is not feature identity. It is a prompt for further causal
tests.

### Intervention Transfer

At the final checkpoint, the lab constructs centroid directions of the form:

```text
centroid(target) - centroid(distractor)
```

It adds those directions into earlier checkpoints at the selected depth and
compares the margin gain with a deterministic random-direction control. The
birth event is only allowed after behavior and probe selectivity are present, so
a step-0 direct logit push is recorded as a caveat rather than a circuit-birth
claim.

This is the lab's scoped causal handle. It is easier than cross-model transfer
because the tiny checkpoints share one architecture and parameter coordinate
system.

## How To Run

```bash
cd interpretability
python interp_bench.py --lab lab29 --tier a
python interp_bench.py --lab lab29 --tier b --prompt-set full
```

For a fast table-only smoke:

```bash
python interp_bench.py --lab lab29 --tier a --no-plots
```

## Reading Order

1. `method_card.md`

   Confirms the controlled training setup and lists the birth-event thresholds.

2. `tables/checkpoint_behavior.csv`

   Shows margins, correctness, logit-lens event depths, top tokens, and
   previous-match attention scores per task and checkpoint.

3. `tables/checkpoint_probe_selectivity.csv`

   Shows centroid-probe accuracy by depth and checkpoint, plus shuffled-label
   controls.

4. `tables/checkpoint_circuit_summary.csv`

   Aggregates each checkpoint into behavior, probe, motif, intervention, and
   phase columns.

5. `tables/mechanism_birth_events.csv`

   Records the first checkpoint where each threshold crossed.

6. `tables/feature_lineage.csv`

   Tracks centroid similarity to the final checkpoint.

7. `tables/intervention_transfer.csv`

   Compares final-direction activation additions with random-direction controls.

## Phase Labels

The default phase labels are deliberately coarse:

- `absent_or_random`
- `decodable_before_behavioral`
- `behavioral_before_decodable`
- `migration`
- `circuit_present`
- `behavioral_decodable_no_mean_attention_motif`
- `sharpening_or_redistributed`

These labels are summaries of thresholds. Treat them as navigation aids, not
natural kinds.

## Common Failure Modes

### Behavior Appears But Probe Does Not

The probe may be at the wrong depth, the centroid method may be too weak, or the
model may solve the task with a representation that is not linearly separated
by target token.

### Probe Appears But Behavior Does Not

This can happen when the representation is readable but not yet connected to
the output path. That is a real hypothesis, but it needs intervention or
attribution evidence before it becomes a mechanism claim.

### Motif Appears Late

An attention motif can lag behavior if the model initially uses a different
shortcut, if the motif is spread across heads, or if the attention-only metric
misses an MLP-mediated computation.

### Intervention Transfers Too Early

Because all tiny checkpoints share one parameter coordinate system, a
final-checkpoint direction can work earlier than it would across unrelated
models. Always compare with the random-direction control.

## Extension To External Checkpoints

The same artifact schema can be reused for:

- Pythia checkpoint sequences;
- a small fine-tuning run;
- Lab 20 organism checkpoints;
- LoRA adapter checkpoints.

Do not change the claim grammar when you scale up. Change the data manifest,
checkpoint metadata, and controls.

## Claim Grammar

Allowed:

```text
OBS/DECODE/CAUSAL: In checkpoint sequence C, behavior B crossed threshold T at
checkpoint K, decodability crossed threshold U at checkpoint J, and intervention
I became effective at checkpoint M under shuffled-label and random-direction
controls.
```

Forbidden:

```text
The model first learned concept X at exactly this step.
```

Also forbidden:

```text
This tiny model proves that large pretrained models learn induction the same way.
The probe discovered the real feature.
The attention motif is the whole circuit.
```

## Deliverable

Write a short time-lapse audit:

- Which threshold crossed first: behavior, decodability, motif, or intervention?
- Did the untrained control task stay flat?
- Did the shuffled-label probe stay below the real probe?
- Did the best depth migrate?
- What exact sentence can you put in the claim ledger without implying an exact learning instant?
