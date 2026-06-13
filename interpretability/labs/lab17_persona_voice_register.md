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
python interp_bench.py --lab lab17 --tier a
python interp_bench.py --lab lab17 --tier b --prompt-set full
```

Useful while debugging:

```bash
python interp_bench.py --lab lab17 --tier a --no-plots
python interp_bench.py --lab lab17 --tier b --prompt-set medium --no-plots
```

Lab 17 uses instruct models and chat templates. Tier A should use a small instruct model such as `HuggingFaceTB/SmolLM2-135M-Instruct`; Tier B should use the course instruct model such as `allenai/Olmo-3-7B-Instruct`. The lab file itself does not require registry-specific arguments beyond the shared bench options.

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

Each row has:

```text
item_id, trait, family, topic, task_kind,
positive_label, negative_label,
prompt_positive, prompt_negative, eval_prompt,
expected_keywords, positive_markers, negative_markers,
content_question, note
```

Rows are split by `trait:topic`, so contrast pairs stay together and held-out evaluation does not reuse the same topic. If the frozen CSV is missing on a Tier A smoke run, the revised Python uses a clearly labeled built-in fallback. That fallback is for plumbing only. Do not ledger science claims from it.

## The three experiments

### 1. Direction extraction and held-out decoding

For each trait and each stream depth, the lab renders the positive and negative prompts with the model's chat template and captures the residual stream at the final prompt token. The direction is:

```text
direction_trait(depth) = mean_train(positive_stream - negative_stream)
```

The revised lab chooses the stream depth using only a train-side leave-one-out, control-adjusted score:

```text
score(depth) = mean(real AUC) - max(mean(shuffled-sign AUC), mean(random AUC))
```

Held-out eval AUC is reported after the depth is selected. This prevents the prettiest eval curve from quietly choosing the microscope setting.

### 2. Steering on held-out neutral prompts

The lab steers held-out `eval_prompt` rows with activation addition at:

```text
injection_layer = selected_stream_depth - 1
```

This is the same stream-depth convention used throughout the course: `streams[k]` is the residual after `k` blocks, while a steering hook on block `layer` changes the stream after that block. The off-by-one risk is named in the artifacts.

The steering sweep includes:

- baseline, no steering;
- trait direction;
- opposite direction;
- shuffled-sign direction;
- random direction.

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
    split_audit.csv                        # train/eval split with prompt hashes
    split_balance.csv
    prompt_render_audit.csv                # rendered prompt lengths, hashes, suffixes
    activation_norms_by_depth.csv
    persona_depth_selection.json
    turn_boundary_check.json               # Lab 15-style segmentation checks
    generation_prompt_boundary_check.csv
    persona_safety_scope.json

  tables/
    persona_family_manifest.csv
    persona_probe_report.csv               # AUC by trait, depth, split, and control
    persona_depth_selection.csv            # train-only selection and eval report curves
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

  plots/
    persona_probe_selectivity.png
    persona_steering_dose_response.png
    style_content_tradeoff.png
    direction_cosine_heatmap.png
    persona_turn_trace.png
    register_switch_trace.png
    refusal_projection_under_roleplay.png
    trace_depth_sweep.png

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
2. `tables/persona_depth_selection.csv` and `plots/persona_probe_selectivity.png`.
3. `tables/persona_steering_effects.csv` and `plots/style_content_tradeoff.png`.
4. `tables/persona_turn_trace_slopes.csv` and `plots/persona_turn_trace.png`.
5. `tables/direction_cosines.csv` and `plots/direction_cosine_heatmap.png`.
6. `diagnostics/persona_safety_scope.json` before discussing refusal.

## How to interpret the plots

### `persona_probe_selectivity.png`

Solid lines are held-out eval. Dashed lines are train-side leave-one-out curves used for depth selection. A real curve that barely beats shuffled and random controls is a weak handle, even if it rises above 0.5.

### `persona_steering_dose_response.png`

The trait direction should move style more than random and shuffled controls. The opposite direction should usually move the other way. If all directions move together, the vector may be a generic activation-norm lever, not a persona/register handle.

### `style_content_tradeoff.png`

The useful quadrant is style movement with little or no content damage. If style improves but content collapses, the model is not switching register cleanly. It is wearing a glittering lab coat while dropping the beakers.

### `direction_cosine_heatmap.png`

Large cosines between persona/register/voice and sentiment, politeness, or agreement controls are not automatically bad, but they weaken the story. You must say what collapsed into what.

### `persona_turn_trace.png`

A roleplay trace matters only relative to the default-control and random-null traces. A rising persona projection inside the roleplay conversation is a descriptive trace, not proof of a durable character.

### `register_switch_trace.png`

Look for the technical-register projection changing after the user explicitly asks for a register switch. If the projection is high before the switch, the direction may be reading the task topic or the system scaffold.

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
2. Was the selected depth chosen by train-only evidence, or did you accidentally read it from eval?
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
| Eval AUC is high but controls are high too | small-n curve shopping or easy prompt formatting | `tables/persona_depth_selection.csv` |
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
