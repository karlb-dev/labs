# Lab 17 - Persona, Voice, Roleplay, and Register

**Evidence target:** `DECODE -> CAUSAL`, plus multi-turn `OBS/DECODE` traces.
**Prerequisites:** Intro Lab 7, Lab 15, and Lab 16.
**Core warning:** a persona direction is a handle on an operationalized prompt contrast. It is not a private identity wearing a lab coat.

## Question

When a model role-plays a character, switches register, or develops a recognizable voice, is there a persistent internal handle we can measure, or mostly surface style riding on the chat template?

This lab treats persona, voice, register, and agreement as paired operationalizations. It asks whether directions extracted from controlled prompt pairs transfer to held-out topics, steer generated behavior, and leave a trace across scripted turns after the cheap explanations have had their turn with the flashlight.

## What this lab is not allowed to claim

Do not write:

- "The model has a true personality."
- "The role-play character is the model's real identity."
- "A voice marker proves authorship."
- "The refusal projection went down, so we found a jailbreak."

Allowed claims are narrower:

- A paired persona/register/voice frame is linearly decodable on held-out topics, compared with random and shuffled controls.
- Activation addition along that direction changes generated style markers more than controls, while preserving the content task.
- A scripted roleplay trace has a measurable projection relative to default and random controls, conditioned on turn-boundary checks.
- Refusal-monitor traces under benign roleplay are safety diagnostics, not prompt-optimization targets.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab17 --tier a --prompt-set small
python interp_bench.py --lab lab17 --tier b --prompt-set full --corpus-path data/persona_register_pairs.csv --max-examples 0
```

Useful while debugging:

```bash
python interp_bench.py --lab lab17 --tier a --prompt-set small --persona-steering-prompts 1 --persona-steering-controls 1 --no-plots
python interp_bench.py --lab lab17 --tier b --prompt-set medium --persona-steering-prompts 2 --persona-steering-controls 3 --no-plots
python interp_bench.py --lab lab17 --tier a --prompt-set full --corpus-path data/persona_register_pairs.csv --max-examples 0 --persona-steering-prompts 4 --persona-steering-controls 3
```

Lab 17 uses instruct models and chat templates. Tier A should use a small instruct model such as `HuggingFaceTB/SmolLM2-135M-Instruct`; Tier B should use the course instruct model such as `allenai/Olmo-3-7B-Instruct`. The lab also accepts `--corpus-path` for a frozen CSV, `--persona-steering-prompts` to cap held-out steering prompts per trait, and `--persona-steering-controls` to average multiple random steering controls.

## Dataset

The default science data file is:

```text
data/persona_register_pairs.csv
```

It contains paired prompts over matched task content:

| Trait | Positive frame | Negative/control frame | What it tests |
|---|---|---|---|
| `character_museum_guide` | patient museum guide | concise default assistant | benign roleplay persona |
| `technical_register` | precise technical expert | casual friend | register as a lightweight persona |
| `warm_supportive_voice` | warm supportive voice | direct terse voice | voice and affective style |
| `honest_disagreement` | honest correction | agreeable validation | agreement pressure versus truth-preserving correction |
| `socratic_teacher` | guided questions | answer-only assistant | pedagogical persona |
| `concise_executive_register` | bottom-line executive register | exploratory brainstorm register | decision-register style |
| `cautious_uncertainty_voice` | calibrated caveats | overconfident certainty | uncertainty calibration voice |
| `stepwise_coach` | ordered coaching | single-shot explanation | procedural persona |

Each row has:

```text
item_id, trait, family, topic, task_kind,
positive_label, negative_label,
prompt_positive, prompt_negative, eval_prompt,
expected_keywords, positive_markers, negative_markers,
content_question, note
```

The v2 frozen corpus has 256 rows: 8 traits by 32 matched tasks. Runtime prompt-set caps keep Tier A small (`small` uses 3 topics per trait, `medium` uses 5, and `full` uses the whole file). Tier A also has a default per-trait smoke cap, so pass `--max-examples 0` for a real full-corpus sweep. Rows are split by `trait:topic` into train/dev/test, so contrast pairs stay together and held-out evaluation does not reuse the same topic. If the frozen CSV is missing on a Tier A smoke run, the Python uses a clearly labeled built-in fallback. That fallback is for plumbing only. Do not ledger science claims from it.

## The three experiments

### 1. Direction extraction and held-out decoding

For each trait and each stream depth, the lab renders the positive and negative prompts with the model's chat template and captures the residual stream at the final prompt token. The direction is:

```text
direction_trait(depth) = mean_train(positive_stream - negative_stream)
```

The lab fits directions on train topics, chooses the stream depth on dev topics, and keeps test topics report-only:

```text
score(depth) = mean(real AUC) - max(mean(shuffled-sign AUC), mean(random AUC))
```

Train leave-one-out rows are still reported as a fitting audit, but they do not choose the headline result. Test AUC, bootstrap confidence intervals, and permutation-null summaries are reported after the depth is selected. This prevents the prettiest test curve from quietly choosing the microscope setting.

### 2. Steering on held-out neutral prompts

The lab steers held-out test `eval_prompt` rows with activation addition at:

```text
injection_layer = selected_stream_depth - 1
```

This is the same stream-depth convention used throughout the course: `streams[k]` is the residual after `k` blocks, while a steering hook on block `layer` changes the stream after that block. The off-by-one risk is named in the artifacts.

The steering sweep includes:

- baseline, no steering;
- trait direction;
- opposite direction;
- shuffled-sign direction;
- multiple random directions, controlled by `--persona-steering-controls`.

The lab scores style markers separately from content keywords. A style handle is not useful if it merely adds a catchphrase or breaks the task.

### 3. Scripted multi-turn traces

The lab traces four scripted conversations:

| Conversation | Purpose |
|---|---|
| `museum_roleplay` | sustained benign character roleplay |
| `default_control` | same tasks under concise default assistant wording |
| `register_switch` | casual-to-technical switch on the same problem family |
| `roleplay_boundary` | benign roleplay with a privacy-boundary turn for refusal monitoring |

The trace projects content-boundary, message-boundary, span-mean, and cumulative prefix states onto:

- persona direction;
- default-control direction;
- technical-register direction;
- casual-register control;
- warm-supportive direction;
- direct-terse control;
- honest-correction direction;
- agreeable-validation control;
- sentiment-style control;
- refusal-monitor direction;
- random-null direction.

The result is descriptive unless it beats the null trace and the turn-boundary checks pass.

## Safety wall

Lab 17 is near persona-jailbreak territory, so the refusal-direction component is intentionally narrow:

- refusal-monitor extraction uses forward passes only;
- no refusal-eliciting prompt is ever used for generation;
- the lab does not implement refusal ablation;
- the lab does not search for prompts that erode refusal;
- refusal is traced only in a benign scripted roleplay-boundary conversation.

The safety scope is written to:

```text
diagnostics/persona_safety_scope.json
```

Treat that file as a fuse box. Do not bypass it because a plot looks spicy.

## Main artifacts

```text
runs/lab17_persona_voice_register-<timestamp>-<id>/
  persona_state_frame_card.md              # read this first
  operationalization_audit.md              # what the run does and does not license
  run_summary.md
  metrics.json
  results.csv                              # alias of persona_probe_report.csv

  diagnostics/
    frozen_data_manifest.json              # data source, hash, fallback status, counts
    exact_chat_hook_parity.json            # hook check on rendered chat, no extra special tokens
    exact_chat_hook_parity_by_layer.csv
    logit_lens_self_check.json
    split_audit.csv                        # train/dev/test split with prompt hashes
    split_balance.csv
    prompt_render_audit.csv                # rendered prompt lengths, hashes, suffixes
    activation_norms_by_depth.csv
    persona_depth_selection.json
    turn_boundary_check.json               # Lab 15-style segmentation checks
    generation_prompt_boundary_check.csv
    persona_safety_scope.json

  tables/
    persona_family_manifest.csv
    persona_probe_report.csv               # AUC, CIs, and permutation-null stats by trait, depth, split, and control
    persona_depth_selection.csv            # train audit, dev selection, and test report curves
    probe_best_depth_by_trait.csv
    direction_provenance.csv
    direction_cosines.csv
    persona_steering_generations.csv       # generated text plus hand-label columns
    generation_labeling_guide.md
    persona_steering_effects.csv           # style, content, repetition, boundary effects
    register_content_style_scores.csv
    turn_segments.csv                      # message and content spans
    persona_turn_trace.csv
    persona_turn_trace_slopes.csv
    trace_depth_sweep.csv
    probe_depth_control_gaps.csv
    persona_trait_evidence_matrix.csv
    persona_steering_operating_points.csv
    persona_trace_evidence.csv
    persona_direction_confound_risks.csv
    plot_reading_guide.csv

  plots/
    persona_evidence_dashboard.png             # start here: decode + steering + traces + confounds
    trait_evidence_matrix.png                  # one row per persona/register/voice handle
    depth_control_gap_atlas.png                # held-out real-minus-control AUC by depth and trait
    persona_probe_selectivity.png
    persona_steering_dose_response.png
    steering_operating_frontier.png            # style benefit versus content/boundary cost
    generation_style_atlas.png                 # trait-by-dose steering response
    style_content_tradeoff.png
    direction_cosine_heatmap.png
    direction_confound_risk.png                # nearest sentiment/refusal/default/style confounds
    persona_turn_trace.png
    persona_trace_projection_atlas.png
    register_switch_trace.png
    refusal_projection_under_roleplay.png
    refusal_boundary_safety_dashboard.png
    trace_depth_sweep.png
    trace_evidence_atlas.png

  state/
    persona_directions.pt
    register_direction.pt
    voice_directions.pt
    persona_voice_register_metadata.json
```

## Reading path

Start with `persona_state_frame_card.md`. It gives the run verdict, selected stream depth, control gaps, steering effect, content preservation, and roleplay trace gap.

Then read `operationalization_audit.md`. If the audit says the effect is probably style markers, politeness, sentiment, or template residue, that is the result. Do not rescue the exciting interpretation with vibes.

Then check, in order:

1. `diagnostics/exact_chat_hook_parity.json` and `diagnostics/turn_boundary_check.json`.
2. `plots/persona_evidence_dashboard.png` for the whole evidence board.
3. `tables/probe_depth_control_gaps.csv`, `plots/depth_control_gap_atlas.png`, and `plots/persona_probe_selectivity.png` for the DECODE rail.
4. `tables/persona_trait_evidence_matrix.csv` and `plots/trait_evidence_matrix.png` for trait-by-trait claim posture.
5. `tables/persona_steering_operating_points.csv`, `plots/steering_operating_frontier.png`, and `plots/generation_style_atlas.png` for the scoped CAUSAL rail.
6. `tables/persona_trace_evidence.csv`, `plots/trace_evidence_atlas.png`, and `plots/persona_trace_projection_atlas.png` for multi-turn traces.
7. `tables/persona_direction_confound_risks.csv`, `tables/direction_cosines.csv`, and `plots/direction_confound_risk.png` before making any voice/persona language sound stronger than the controls.
8. `diagnostics/persona_safety_scope.json` and `plots/refusal_boundary_safety_dashboard.png` before discussing refusal.

## Visualization upgrade notes

The upgraded plot suite turns Lab 17 into an evidence board rather than a personality poster. The central plot is `persona_evidence_dashboard.png`: it joins four rails that must stay separate in the writeup: held-out decodability over controls, activation-addition steering over controls, content/boundary preservation, and descriptive multi-turn traces.

The new `trait_evidence_matrix.png` and `persona_trait_evidence_matrix.csv` are the anti-overclaim artifacts. A trait can be `decodable_not_yet_causal`, `controlled_style_handle`, or `not_validated`; do not upgrade one strong row into a claim about all persona, voice, and register frames.

The new `steering_operating_frontier.png` replaces largest-dose thinking with operating-point thinking. A good dose moves the requested style/register while preserving content keywords, avoiding private-experience claims, and beating random/shuffled/opposite controls.

The new trace plots are deliberately labeled descriptive. `trace_evidence_atlas.png` subtracts same-conversation random-null slopes. `persona_trace_projection_atlas.png` shows the actual turn sequence. These plots can support a scoped multi-turn projection claim only after the Lab 15-style boundary checks pass.

The new safety plot, `refusal_boundary_safety_dashboard.png`, remains monitor-only. It is there to check whether benign roleplay shifts the refusal-monitor projection. It is not a prompt-search target and not a refusal-ablation experiment.

## How to interpret the plots

### `persona_evidence_dashboard.png`

Start here. The dashboard asks whether the run has all four ingredients: controlled DECODE evidence, steering specificity, content preservation, and trace evidence beyond nulls. If one panel fails, narrow the claim instead of rescuing the big word.

### `trait_evidence_matrix.png`

Each row is one operationalized handle. Use this plot to decide whether `technical_register`, `warm_supportive_voice`, or roleplay persona has its own claim posture. The matrix is more important than the average.

### `depth_control_gap_atlas.png`

This plot shows held-out real-minus-control AUC by depth and trait. Bright cells are candidate readout depths; broad bands are stronger than isolated sparks.

### `persona_probe_selectivity.png`

Solid lines are held-out test. Dash-dot lines are dev selection curves. Dashed lines are train-side leave-one-out audit curves. A real curve that barely beats shuffled and random controls is a weak handle, even if it rises above 0.5.

### `persona_steering_dose_response.png`

The trait direction should move style more than random and shuffled controls. The opposite direction should usually move the other way. If all directions move together, the vector may be a generic activation-norm lever, not a persona/register handle.

### `steering_operating_frontier.png`

This is the dose-choice plot. It shows whether style movement is purchased by losing task content, repetition, or boundary discipline. A persona/register vector that only works at a destructive dose is a bad handle, even if the line moved.

### `generation_style_atlas.png`

This plot checks whether the steering effect is broad across traits and doses or whether a single trait carries the mean. Controls that light up here shrink the causal claim.

### `style_content_tradeoff.png`

The useful quadrant is style movement with little or no content damage. If style improves but content collapses, the model is not switching register cleanly. It is wearing a glittering lab coat while dropping the beakers.

### `direction_confound_risk.png`

This ranks the nearest sentiment, refusal, default-assistant, random, and style controls for each saved direction. High overlap does not make the run useless, but it changes the allowed language from “persona” to “style/control-adjacent handle.”

### `direction_cosine_heatmap.png`

Large cosines between persona/register/voice and sentiment, politeness, or agreement controls are not automatically bad, but they weaken the story. You must say what collapsed into what.

### `persona_turn_trace.png`

A roleplay trace matters only relative to the default-control and random-null traces. A rising persona projection inside the roleplay conversation is a descriptive trace, not proof of a durable character.

### `trace_evidence_atlas.png`

This plot shows projection-slope gaps after subtracting the same-conversation random-null slope. It is the guardrail against reading a longer transcript or repeated chat template as a persona state.

### `persona_trace_projection_atlas.png`

This shows the actual turn-by-turn projection sequence across scripted conversations. It is the plot to use for examples, not for choosing new thresholds after the fact.

### `register_switch_trace.png`

Look for the technical-register projection changing after the user explicitly asks for a register switch. If the projection is high before the switch, the direction may be reading the task topic or the system scaffold.

### `refusal_boundary_safety_dashboard.png`

This is the upgraded safety monitor. Read it together with `diagnostics/persona_safety_scope.json`. It monitors benign roleplay boundary turns and does not license refusal ablation or jailbreak search.

### `refusal_projection_under_roleplay.png`

This is a monitor under a benign boundary conversation. Stability is reassuring. Erosion is a safety diagnostic, not a jailbreak result.

### `trace_depth_sweep.png`

This uses the selected best-depth direction and projects it across stream depths. Treat it as descriptive. If you choose a new depth after seeing this plot, that choice must be written into a new hypothesis and rerun.

## Hand labeling

The automatic style-marker rubric is a scaffold. For any result you write about, fill these columns in `tables/persona_steering_generations.csv`:

```text
hand_label_style: positive | negative | mixed | none | bad_parse
hand_label_content: preserved | partly_preserved | changed | wrong | ungraded
hand_label_boundary: ok | private_experience_claim | unsafe_boundary_eroded | refusal_overtriggered | ungraded
```

A persona result that survives hand labels is much stronger than one that survives only the keyword sieve.

## Writeup questions

1. Which trait had the strongest held-out AUC at the selected depth, and did it beat shuffled and random controls?
2. Was the selected depth chosen by dev evidence, or did you accidentally read it from test?
3. Did trait-direction steering move style markers more than random and shuffled controls?
4. Did technical-register steering preserve content keywords, or did it alter task performance?
5. In the roleplay trace, did the persona direction rise more than the random-null direction?
6. In the register-switch trace, where did the technical-register projection flip?
7. What did the refusal-monitor trace show, and why is it not a jailbreak result?
8. Which cheap explanation is still most plausible after this run?
9. Which claim should be weakened, retired, or rewritten before entering the ledger?

## Claim templates

Good `DECODE` claim:

```text
[L17-C1][DECODE] At stream depth D, the paired technical-register direction separates held-out technical prompts from casual controls with AUC X versus shuffled/random Y/Z. Scope: this model, this frozen prompt battery, final prompt-token residuals. Falsifier: shuffled controls match X or topic-held-out performance collapses.
```

Good `CAUSAL` claim:

```text
[L17-C2][CAUSAL] Adding the technical-register direction at layer D-1 changes style-marker margin by Δ more than random/shuffled controls while content-hit rate changes by C. Scope: greedy generations on held-out prompts. Falsifier: hand labels show vocabulary-only changes or content correctness drops.
```

Good trace claim:

```text
[L17-C3][OBS/DECODE] In the scripted museum-guide roleplay, the persona projection slope exceeds the random-null slope by G after turn-boundary checks pass. Scope: scripted transcript, selected direction, selected model. Non-claim: durable identity or human-like persona.
```

Bad claim:

```text
The model becomes a museum guide internally.
```

That sentence is a fog machine. Replace it with a measurement, a scope, and a falsifier.

## Debugging guide

| Symptom | Likely cause | Check |
|---|---|---|
| Exact chat hook parity fails | rendered prompt tokenization drift or unsupported architecture | `diagnostics/exact_chat_hook_parity_by_layer.csv` |
| Test AUC is high but controls are high too | small-n curve shopping or easy prompt formatting | `tables/persona_depth_selection.csv` |
| Steering changes every condition | activation-norm or generic fluency lever | `plots/persona_steering_dose_response.png` |
| Style improves but content drops | register is interfering with task behavior | `plots/style_content_tradeoff.png` |
| Persona trace rises in default control too | direction reads shared task content or chat scaffold | `plots/persona_turn_trace.png` |
| Content spans fall back to message spans | tokenizer offset mapping unavailable or template changed tokenization | `tables/turn_segments.csv` |
| Refusal projection moves sharply | safety diagnostic only, do not optimize | `diagnostics/persona_safety_scope.json` |

## Downstream contract

The saved state files are reusable handles, not magic identity goo:

- `state/register_direction.pt` can support Lab 19 and Lab 24 style/register controls.
- `state/voice_directions.pt` can support voice or agreement controls.
- `state/persona_directions.pt` includes sentiment and refusal controls for comparison.

Downstream labs must check model id, width, stream depth, and injection-layer convention before loading these tensors. A direction from one model is provenance in another model, not an intervention-ready vector.
