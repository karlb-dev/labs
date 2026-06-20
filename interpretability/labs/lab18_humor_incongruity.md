# Lab 18 - Humor as Incongruity

**Evidence levels targeted:** `OBS -> DECODE -> CAUSAL`, with the causal claim narrowed to activation-addition effects on generated joke-shaped text. The headline artifact is an operationalization audit, not a declaration that the model has a sense of humor.

## Core question

When an instruct model handles a joke, is there a measurable internal handle for setup-dependent incongruity and resolution, or are we mostly measuring surprise, silliness, positivity, or generic joke-register?

This lab operationalizes humor narrowly:

```text
setup creates expectation -> ending violates expectation -> violation resolves through the setup
```

The lab does **not** ask whether the model finds anything funny. It asks whether a joke-vs-control residual-stream direction transfers to held-out rows, survives cheap controls, and can steer generation more specifically than surprise, silliness, positivity, shuffled, or random directions.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab18 --tier a
python interp_bench.py --lab lab18 --tier b --prompt-set full --corpus-path data/humor_incongruity_pairs.csv --max-examples 0
```

Useful while debugging:

```bash
python interp_bench.py --lab lab18 --tier a --no-plots
python interp_bench.py --lab lab18 --tier b --prompt-set medium --no-plots
python interp_bench.py --lab lab18 --tier a --prompt-set full --corpus-path data/humor_incongruity_pairs.csv --max-examples 0 --no-plots
```

Lab 18 uses instruct models, chat templates, generation, residual-stream probes, and attention patterns. The lab renders prompts itself and verifies exact rendered-chat hook parity with `add_special_tokens=False`. Attention-pattern plots require eager attention; if the model returns no attention tensors, rerun with the registry/CLI configured for eager attention. Use `--corpus-path` for a frozen CSV; Tier A has a default per-family smoke cap, so pass `--max-examples 0` for a real full-corpus sweep.

## Dataset

The intended frozen file is:

```text
data/humor_incongruity_pairs.csv
```

Each row contains one setup and five matched endings:

| condition | purpose |
|---|---|
| `joke` | setup-dependent punchline or joke-shaped resolution |
| `literal` | plain non-joke completion that preserves the obvious meaning |
| `surprise` | unexpected but not joke-structured ending |
| `silly` | arbitrary absurdity without setup-dependent resolution |
| `positive` | positive-sentiment ending without joke structure |

Required columns:

```text
item_id,family,setup,joke_completion,literal_completion,
surprise_completion,silly_completion,positive_completion
```

Recommended columns:

```text
setup_anchor,resolution_keyword,joke_markers,silly_markers,
surprise_markers,positive_markers,note
```

Tier A has a built-in smoke fallback if the CSV is missing. That fallback is plumbing-only. The run writes `diagnostics/frozen_data_manifest.json`; if `used_smoke_fallback` is true, do not make science claims from that run.

The v2 frozen corpus has 80 rows: 8 humor families by 10 matched setups. Runtime splits are deterministic by family into train/dev/test. Train rows fit directions, dev rows select stream depth, and test rows provide headline probe metrics.

## What the lab does

### 1. Instrumentation checks

The lab verifies that the residual stream being measured is the exact rendered chat prompt that generation sees:

```text
diagnostics/exact_chat_hook_parity.json
diagnostics/exact_chat_hook_parity_by_layer.csv
diagnostics/prompt_render_audit.csv
```

This protects the lab from a common chat-template bug: measuring one token sequence and generating from another. For contrast prompts, the default representation site is the last token overlapping `resolution_keyword`; if that span cannot be found, the lab falls back to the last token overlapping the condition ending, and only then to the final rendered prompt token. The chosen site and fallback method are written to `diagnostics/prompt_render_audit.csv`.

### 2. OBS: surprisal and setup routing

The lab measures:

- setup next-token entropy;
- teacher-forced ending surprisal for joke and controls;
- resolution-keyword surprisal for joke endings when `resolution_keyword` is provided;
- attention from the resolution/final ending token back to setup and anchor spans.

Artifacts:

```text
tables/humor_surprisal_trajectories.csv
tables/humor_surprisal_summary.csv
tables/attention_to_setup.csv
tables/attention_to_setup_summary.csv
diagnostics/attention_span_audit.csv
plots/humor_surprisal_trajectories.png
plots/attention_to_setup.png
```

These are descriptive measurements. Attention-to-setup is not a mechanism claim. Surprise is not humor, so the surprising-non-joke control is load-bearing.

### 3. DECODE: joke-structure direction

For each stream depth, the lab fits a mass-mean direction on train rows:

```text
joke_structure = joke - mean(literal, surprise, silly, positive)
```

Depth is selected on dev rows using a control-adjusted score:

```text
dev real AUC - max(0.5, shuffled-label AUC, random-direction AUC)
```

Test AUC is reported after depth selection. This keeps the prettiest test-layer curve from quietly becoming the hypothesis. Tiny smoke runs may lack dev rows and fall back to train leave-one-out; the selection table records that case.

The selected-depth direction is also rerun in a one-family-held-out check. This asks whether the handle transfers across joke families or only learns a family-local lexical trick.

Artifacts:

```text
tables/joke_depth_selection.csv
diagnostics/depth_selection.json
tables/joke_probe_by_layer.csv
tables/punchline_phase_probe.csv
tables/family_heldout_probe.csv
results.csv
plots/joke_probe_by_layer.png
```

### 4. Cheap-correlate audit

The lab builds four train-split directions:

```text
joke_structure = joke - mean(literal, surprise, silly, positive)
surprise       = surprise - literal
silly          = silly - literal
positive       = positive - literal
```

Then it audits their cosines and condition projections. If the joke direction is nearly the surprise direction, the result is not a clean humor result. It is surprise wearing a foam nose.

Artifacts:

```text
tables/direction_cosines.csv
tables/projection_by_condition.csv
tables/projection_by_condition_summary.csv
plots/humor_direction_cosines.png
plots/joke_projection_by_condition.png
```

### 5. CAUSAL, scoped: activation-addition steering

The selected joke-structure direction is injected during generation at:

```text
injection_layer = selected_stream_depth - 1
```

That mapping follows the course convention: `streams[k]` is the pre-norm residual stream after `k` blocks, so the writer that produces `streams[k]` is block `k - 1`.

The steering sweep includes:

- positive joke-structure doses;
- opposite joke-structure direction;
- surprise direction;
- silly direction;
- positive direction;
- shuffled joke direction;
- random direction.

Generated endings are scored with marker rubrics, but those marker rubrics are only scaffolding. The lab writes blank hand-label columns and a labeling guide. Fill those columns before making a substantive funniness claim.

Artifacts:

```text
tables/humor_steering_generations.csv
tables/generation_labeling_guide.md
tables/humor_direction_audit.csv
plots/humor_steering_dose_response.png
```

## Artifact tree

A typical run produces:

```text
runs/lab18_humor_incongruity-.../
  run_summary.md
  humor_incongruity_card.md
  operationalization_audit.md
  ledger_suggestions.md
  metrics.json
  results.csv

  diagnostics/
    bench_integration_note.json
    frozen_data_manifest.json
    data_validation_report.csv
    exact_chat_hook_parity.json
    exact_chat_hook_parity_by_layer.csv
    prompt_render_audit.csv
    split_audit.csv
    split_balance.csv
    activation_norms_by_depth.csv
    depth_selection.json
    attention_span_audit.csv

  tables/
    joke_depth_selection.csv
    joke_probe_by_layer.csv
    punchline_phase_probe.csv
    family_heldout_probe.csv
    humor_surprisal_trajectories.csv
    humor_surprisal_summary.csv
    direction_cosines.csv
    projection_by_condition.csv
    projection_by_condition_summary.csv
    attention_to_setup.csv
    attention_to_setup_summary.csv
    humor_steering_generations.csv
    generation_labeling_guide.md
    humor_direction_audit.csv
    family_generalization_summary.csv
    steering_operating_points.csv
    cheap_explanation_audit.csv
    item_incongruity_summary.csv
    humor_evidence_matrix.csv
    plot_reading_guide.csv

  plots/
    humor_surprisal_trajectories.png
    joke_probe_by_layer.png
    humor_steering_dose_response.png
    humor_direction_cosines.png
    attention_to_setup.png
    joke_projection_by_condition.png
    humor_evidence_dashboard.png
    depth_control_gap_atlas.png
    family_generalization_atlas.png
    cheap_explanation_matrix.png
    setup_dependence_atlas.png
    generation_marker_atlas.png
    steering_operating_frontier.png
    item_incongruity_atlas.png
    humor_evidence_matrix.png

  state/
    humor_direction.pt
    humor_directions.pt
    humor_direction_metadata.json
```


## Visualization upgrade: read the audit before the punchline

Lab 18 now writes a richer plot suite. The goal is not to make humor look mystical. The goal is to keep every tempting explanation visible at the same time: joke structure, raw surprise, silliness, positivity, generic joke register, setup routing, and generation side effects.

New start-here artifact:

```text
plots/humor_evidence_dashboard.png
```

This dashboard puts four rails on one page: control-adjusted probe evidence over depth, one-family-held-out transfer, cheap-explanation risk, and steering specificity. A positive-looking dashboard is still only a handle claim until hand labels confirm that generated text has setup-dependent incongruity plus resolution.

Additional upgraded artifacts:

```text
plots/depth_control_gap_atlas.png
plots/family_generalization_atlas.png
plots/cheap_explanation_matrix.png
plots/setup_dependence_atlas.png
plots/generation_marker_atlas.png
plots/steering_operating_frontier.png
plots/item_incongruity_atlas.png
plots/humor_evidence_matrix.png

tables/family_generalization_summary.csv
tables/steering_operating_points.csv
tables/cheap_explanation_audit.csv
tables/item_incongruity_summary.csv
tables/humor_evidence_matrix.csv
tables/plot_reading_guide.csv
```

Recommended reading path after `humor_incongruity_card.md`:

1. `plots/humor_evidence_dashboard.png`: decide whether the run deserves deeper inspection.
2. `plots/depth_control_gap_atlas.png`: verify the selected depth clears shuffled and random nulls.
3. `plots/family_generalization_atlas.png`: check whether the handle transfers beyond one joke family.
4. `plots/cheap_explanation_matrix.png`: let surprise, silliness, positivity, generic register, and no-setup explanations try to kill the favorite interpretation.
5. `plots/setup_dependence_atlas.png`: inspect projection and routing evidence for setup-dependent resolution.
6. `plots/generation_marker_atlas.png` and `plots/steering_operating_frontier.png`: inspect dose response and target movement versus side-effect cost.
7. `plots/item_incongruity_atlas.png`: check whether one row is carrying the aggregate.
8. `tables/humor_steering_generations.csv`: hand-label before writing anything stronger than marker movement.
9. `plots/humor_evidence_matrix.png`: convert the run into evidence-rung claims only after the previous checks.

The upgraded tables are designed to make the writeup less “funny-looking samples” and more “the operationalization survived these exact controls.” If the result collapses into surprise or joke register, that is still a clean Lab 18 result.

## Read this first after a run

Start with `humor_incongruity_card.md`. It gives the verdict, selected depth, test AUC, control gap, direction cosines, and steering summary.

Then read `operationalization_audit.md`. It is the deflationary twin of the pretty plots. If the audit says the handle collapsed into surprise, silliness, positivity, or surface joke register, that is not a failed lab. That is the result.

Then inspect:

1. `diagnostics/frozen_data_manifest.json`: confirms frozen data versus smoke fallback.
2. `diagnostics/prompt_render_audit.csv`: confirms prompt lengths, hashes, and read sites.
3. `tables/joke_depth_selection.csv`: shows how the selected depth was chosen.
4. `tables/joke_probe_by_layer.csv`: checks held-out real versus shuffled/random controls.
5. `tables/family_heldout_probe.csv`: asks whether the selected handle transfers across joke families.
6. `tables/projection_by_condition_summary.csv`: asks whether the joke direction is really condition-specific.
7. `tables/humor_direction_audit.csv`: checks whether steering beats cheap-control directions.
8. `tables/humor_steering_generations.csv`: inspect and hand-label generations before making claims.

## How to read the plots

### `humor_surprisal_trajectories.png`

If jokes are more surprising than literal endings, that is expected. The question is whether the internal direction is more than surprise. Compare this plot with `direction_cosines.csv` and `projection_by_condition_summary.csv`.

### `joke_probe_by_layer.png`

The selected stream depth is marked. Strong evidence means the real test curve beats shuffled and random controls at the selected depth, and the depth-selection table shows the choice came from dev rows.

### `humor_direction_cosines.png`

Large absolute cosines between joke-structure and surprise/silly/positive are warning lights. Cosine is not destiny, but it is where the audit starts knocking on the pipes.

### `joke_projection_by_condition.png`

This plot asks whether each direction is mostly high on its intended condition. A joke-structure direction that lights up surprise, silly, and positive controls equally is a broad style/weirdness axis.

### `attention_to_setup.png`

This plot asks whether the ending token routes back to the setup. It is descriptive routing evidence, not causal proof. A setup-dependent joke claim becomes weaker if joke endings show no more setup attention than literal or surprise controls.

### `humor_steering_dose_response.png`

A positive causal claim needs joke-structure steering to move the joke-vs-cheap marker margin more than surprise, silly, positive, shuffled, and random directions. Then the hand labels need to agree that this is more than weirdness confetti.

## Evidence discipline

Do **not** write:

- "The model finds this funny."
- "The direction is humor itself."
- "Attention to the setup proves the model understands the joke."
- "Steering made the model funnier" using marker columns alone.

Allowed claims are narrower:

- a joke-vs-control direction separates held-out joke endings from matched controls;
- the direction does or does not remain distinct from surprise, silliness, and positivity;
- activation addition changes joke-register or hand-labeled joke-shape more than cheap-correlate directions;
- attention-to-setup is a descriptive routing measurement that motivates, but does not replace, a causal follow-up.

## Writeup questions

1. Which depth was selected, and what dev score selected it?
2. Did the real joke-vs-control direction beat shuffled and random controls on test rows?
3. Does `family_heldout_probe.csv` show transfer across joke families, or is the handle family-local?
4. Are joke endings more surprising than literal endings? Does surprise explain the probe?
5. Which cosine is largest: joke-surprise, joke-silly, or joke-positive?
6. Does `projection_by_condition_summary.csv` show a joke-specific axis or a broad weirdness/style axis?
7. Does the resolution/final ending token attend back to setup tokens more for jokes than for literal and surprise controls?
8. Did joke-structure steering move marker margins more than surprise, silly, positive, shuffled, and random directions?
9. After hand-labeling generations, is the best description "joke structure," "joke register," "surprise," "silliness," "positivity," or "inconclusive"?

## Common failure modes

| symptom | likely cause | what to inspect |
|---|---|---|
| Run says smoke fallback was used | frozen CSV is missing | `diagnostics/frozen_data_manifest.json` |
| Hook parity fails | chat prompt tokenization or model anatomy mismatch | `diagnostics/exact_chat_hook_parity*` |
| Attention table errors | model not using eager attention or tokenizer span lookup failed | rerun eager; inspect `diagnostics/attention_span_audit.csv` |
| Probe AUC is high but controls are also high | depth selection or small-n artifact | `joke_depth_selection.csv`, null reps in `joke_probe_by_layer.csv` |
| Within-family probe works but family-held-out fails | family-local lexical joke pattern | `tables/family_heldout_probe.csv` |
| Joke direction is collinear with surprise | direction is likely surprise/incongruity without resolution | `direction_cosines.csv`, `projection_by_condition_summary.csv` |
| Steering increases weird text but not jokes | direction controls style, not setup-dependent resolution | hand-label `humor_steering_generations.csv` |
| Positive steering matches joke steering | positive assistant tone is the real handle | `humor_direction_audit.csv` |

## Ledger templates

Positive `DECODE` claim:

```text
[L18-C1][DECODE] On MODEL and DATASET, a joke-structure mass-mean direction at stream depth D separates held-out joke endings from literal/surprise/silly/positive controls with AUC X, compared with shuffled Y and random Z; one-family-held-out transfer is T. This is a joke-structure handle claim, not a subjective-funniness claim. Artifact: runs/.../tables/joke_probe_by_layer.csv and family_heldout_probe.csv. Falsifier: shuffled/random controls match the effect, cheap-correlate directions explain it, or family-heldout transfer collapses.
```

Cautious `CAUSAL` claim:

```text
[L18-C2][CAUSAL, scoped] Adding the Lab 18 joke-structure direction at layer L changes generated joke-shape markers by Δ over baseline and by G over the strongest cheap-control direction. Hand labels show / do not show corresponding gains in incongruity plus resolution. Artifact: runs/.../tables/humor_direction_audit.csv and humor_steering_generations.csv. Falsifier: surprise/silly/positive/shuffled/random steering or hand labels erase the specificity gap.
```

Negative result claim:

```text
[L18-N1][DECODE] This run did not validate a humor-specific direction: real AUC X was matched by null control Y, or the direction collapsed into cheap correlate C. Artifact: runs/.../operationalization_audit.md. Falsifier: a rerun on frozen full data with dev selection, test reporting, and stronger controls shows a stable gap.
```

## Ethics and interpretation

A model can generate joke-shaped text without amusement. It can detect incongruity without grounding. It can use joke register without resolving the setup. Treat "humor" the way the earlier steering labs treated truth and persona: a direction that moves behavior is a handle, not a mechanism, and not a feeling in a little velvet cape.

The most valuable Lab 18 result may be negative: "we found joke register, not funniness." That is a clean contribution because it makes the boundary of the instrument visible.
