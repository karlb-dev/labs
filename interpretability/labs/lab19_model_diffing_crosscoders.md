# Lab 19: Model Diffing With Crosscoders

**Evidence level targeted:** `DECODE + ATTR` at the feature level, with an optional narrow `CAUSAL` extension on benign prompts.

**Depends on:** Intro Lab 8 for sparse feature thinking. Recommended: Labs 16 and 17 for the direction bridge to agreement, user-belief, persona, voice, and register directions. Lab 19 feeds Labs 21 and 23.

## The question

When a base model becomes an instruct model, what changes in its internal representations?

That question is too large as written. This lab uses a smaller, testable version:

> Given two models and one matched prompt inventory, can we train a paired sparse dictionary whose features are shared, model-A-only, model-B-only, asymmetric, or dead, and can we audit whether model-B-only features are real model differences rather than chat-template residue, prompt-distribution imbalance, norm shifts, or dictionary artifacts?

The attractive phrase is **default assistant voice**. The careful phrase is **model-B-skewed residual feature under this prompt distribution**. The lab teaches how much evidence is needed to walk from the careful phrase toward the attractive one without stepping on a rake.

## Run

From the course root:

```bash
python interp_bench.py --lab lab19 --tier a --no-plots
python interp_bench.py --lab lab19 --tier b --prompt-set full
```

Useful while debugging:

```bash
python interp_bench.py --lab lab19 --tier a --prompt-set small --no-plots
python interp_bench.py --lab lab19 --tier b --prompt-set medium --no-plots
```

The lab also accepts a custom CSV, TSV, JSON, or JSONL prompt inventory through `--prompt-set path/to/prompts.csv`.

For the current fair-shot validation inventory:

```bash
LAB19_OFFLOAD_PRIMARY_TO_CPU=1 python interp_bench.py --lab lab19 --tier b --prompt-set data/model_diffing_prompt_inventory_v2.csv
LAB19_OFFLOAD_PRIMARY_TO_CPU=1 python interp_bench.py --lab lab19 --tier b --prompt-set data/model_diffing_prompt_inventory_v2.csv --run-edit
```

For a science run, model A should be the base model and model B should be the comparison model, usually the instruct model from the same family. The registry can provide the comparison model. You can also set:

```bash
export LAB19_COMPARE_MODEL=allenai/Olmo-3-7B-Instruct
python interp_bench.py --lab lab19 --tier b --prompt-set full
```

Tier A may intentionally use an identity pair. That run should mostly produce shared or noisy features. Identity-pair weirdness is not boring. It is a smoke alarm.

Useful environment knobs for constrained machines:

```bash
# select one residual stream depth for both models
export LAB19_DEPTH=18

# or select depths separately
export LAB19_DEPTH_A=18
export LAB19_DEPTH_B=20

# shrink or enlarge the dictionary and training budget
export LAB19_FEATURES=64
export LAB19_TRAIN_STEPS=240

# if a two-model run barely fits, move model A to CPU after its activations are collected
export LAB19_OFFLOAD_PRIMARY_TO_CPU=1
```

The lab is content-file-only friendly. If your local registry does not yet expose `lab19`, add it separately; the lab module itself writes a model-pair diagnostic so the run remains auditable once the registry entry exists.

## What you build

You build a small paired crosscoder.

At a selected stream depth in model A and a selected stream depth in model B, the lab collects final-token residual activations for the same rendered text. It normalizes each side using the train split, then trains a shared sparse code:

```text
z_pair = ReLU(x_A W_A + x_B W_B + b)

x_A_hat = z_pair D_A
x_B_hat = z_pair D_B
```

The same feature index reconstructs both sides through separate decoders. That matters. A pair of unrelated sparse autoencoders can silently permute their features and make a fake comparison. A paired code forces feature `f` to be one proposed cross-model atom, with two decoders describing how that atom writes into each model's residual space.

The lab also computes side-only activations:

```text
z_A = ReLU(x_A W_A + b)
z_B = ReLU(x_B W_B + b)
```

Those side activations help classify features. The main reconstruction path remains paired.

## The stream-depth convention

The bench convention applies here:

```text
streams[k] = pre-norm residual stream after k blocks
```

Depth 0 is embeddings. Depth `L` is the input to the final norm. A steering vector injected at block layer `k` writes into `streams[k + 1]`.

Lab 19 records both selected stream depths in `diagnostics/model_pair.json` and `state/crosscoder_metadata.json`. If the optional causal validation uses a model-B feature decoder, it maps:

```text
stream_depth_B -> injection_layer = stream_depth_B - 1
```

A lot of interpretability ghosts are off-by-one errors wearing lab coats. This lab writes the mapping down.

## The prompt inventory

The default inventory is assembled from earlier advanced-course datasets if they exist:

- persona/register prompts;
- sycophancy-pressure prompts;
- certainty-calibration prompts;
- humor setup prompts;
- a small authored set of assistant-voice, technical, factual, and style-control prompts.

For many rows, the lab includes both:

```text
variant = raw
variant = compare_chat
```

The `compare_chat` variant is rendered with model B's tokenizer chat template, then the exact same text is passed through both models. That is deliberate. It asks whether a feature is about the comparison model's behavior or merely about the chat scaffolding that the comparison model usually sees.

The full inventory is written to:

```text
tables/prompt_inventory.csv
diagnostics/frozen_prompt_manifest.json
diagnostics/split_audit.csv
```

The split is by `prompt_group`, not by individual row, so the raw and chat-rendered versions of the same underlying prompt do not get split across train, dev, and test. The crosscoder is trained on train rows; held-out reconstruction, feature stability, and report metrics use the test split, with the dev split reserved for quick diagnostics and future selection logic.

When the two residual spaces have the same dimensionality, Lab 19 initializes the model-A and model-B sides from the same weights. This is a conservative default for model diffing: an identity-pair smoke run should not invent model-specific features just because the two sides began from unrelated random seeds. Real base/instruct differences can still move the two sides apart during training.

## Evidence ladder

| Evidence | Artifact | What it licenses |
|---|---|---|
| `OBS` | activation norms, token counts, prompt marker rates | the prompt inventory and residual magnitudes differ in measured ways |
| `DECODE/ATTR` | crosscoder taxonomy and galleries | a feature coordinate separates or reconstructs model-pair activation structure |
| audited `DECODE/ATTR` | template, random, norm, and family controls | a model-skewed feature is not obviously a cheap artifact |
| narrow `CAUSAL` | optional feature steering with random controls and hand labels | a selected decoder direction changes a measured benign behavior |

The graph proposes. The intervention disposes. The audit decides whether the proposal was even pointed at the right thing.

## What the files mean

A successful run produces:

```text
runs/lab19_model_diffing_crosscoders-.../
  run_summary.md
  model_diffing_card.md
  model_diffing_report.md
  operationalization_audit.md
  metrics.json
  results.csv
  ledger_suggestions.md

  diagnostics/
    model_pair.json
    frozen_prompt_manifest.json
    split_audit.csv
    model_a_hook_parity.json
    model_a_hook_parity_by_layer.csv
    model_a_logit_lens_self_check.json
    model_b_hook_parity.json
    model_b_hook_parity_by_layer.csv
    model_b_logit_lens_self_check.json
    activation_norms.csv
    causal_feature_validation_manifest.json

  tables/
    prompt_inventory.csv
    activation_norm_controls.csv
    crosscoder_training_curve.csv
    feature_taxonomy.csv
    feature_eval_stability.csv
    feature_context_gallery.csv
    instruct_only_feature_gallery.csv       # backward-compatible alias
    feature_labeling_guide.md
    template_control_summary.csv
    default_voice_marker_rates.csv
    random_feature_baseline.csv
    feature_direction_bridge.csv
    causal_feature_validation.csv
    causal_feature_validation_summary.csv
    prompt_inventory_summary.csv
    activation_norm_shift_summary.csv
    feature_audit_matrix.csv
    taxonomy_control_ladder.csv
    model_diffing_evidence_matrix.csv
    causal_operating_points.csv
    plot_reading_guide.csv

  plots/
    model_diffing_evidence_dashboard.png
    crosscoder_training_diagnostics.png
    feature_taxonomy_counts.png
    feature_exclusivity_histogram.png
    crosscoder_reconstruction.png
    feature_audit_matrix.png
    taxonomy_control_ladder.png
    prompt_inventory_balance.png
    activation_norm_shift_atlas.png
    feature_context_atlas.png
    template_control_gaps.png
    feature_direction_bridge.png
    direction_bridge_matrix.png
    causal_feature_validation.png
    causal_operating_frontier.png
    identity_smoke_scorecard.png

  state/
    crosscoder_state.pt
    crosscoder_metadata.json
```

Start with `model_diffing_card.md`. Then open the taxonomy and the audit tables before looking at the prettiest gallery rows. Pretty contexts are bait unless the controls are already on the table.

## Feature taxonomy

`tables/feature_taxonomy.csv` contains one row per feature.

Important columns:

| Column | Meaning |
|---|---|
| `taxonomy` | generic label: `shared`, `model_a_only`, `model_b_only`, `asymmetric`, or `dead` |
| `role_taxonomy` | role-aware label such as `base_only` or `instruct_only` when model roles can be inferred |
| `model_b_activation_share` | side-only activation share for model B |
| `decoder_norm_model_b_share` | share of decoder norm on the model-B side, measured in residual units |
| `activation_correlation_a_b` | promptwise correlation between side-only activations |
| `top_model_b_family_concentration` | whether top activations come from one prompt family |
| `top_model_b_variant_concentration` | whether top activations come from raw or chat-rendered prompts |
| `audit_flag_template_concentrated` | first warning that the feature may be chat-template residue |
| `audit_flag_family_concentrated` | first warning that the feature may be a dataset-family feature |

A good candidate is not simply high `model_b_activation_share`. It should also have tolerable reconstruction, survive matched random-direction baselines, avoid being one-family-only unless the claim is one-family-specific, and not light up only because `compare_chat` added role scaffolding.

## The operationalization audit

The lab's favorite story is that instruct-only features reveal instruction following, alignment, or assistant voice.

The deflationary twin is that instruct-only features are:

| Cheap explanation | Killing artifact |
|---|---|
| chat-template tokens | `tables/template_control_summary.csv` |
| distribution mismatch | `tables/prompt_inventory.csv`, `tables/instruct_only_feature_gallery.csv` |
| activation-norm shift | `diagnostics/activation_norms.csv`, `tables/activation_norm_controls.csv` |
| crosscoder artifact | `tables/random_feature_baseline.csv`; for same-dimensional pairs, prefer `control_kind=matched_shared_direction` over the deliberately harsh `independent_side_directions` diagnostic |
| shallow marker words | `tables/default_voice_marker_rates.csv`, optional generated labels |
| borrowed label from prior lab | `tables/feature_direction_bridge.csv` |

A feature that fails one of these controls can still be useful. It just gets a narrower name. `template feature` is a good result. `Assistant soul atom` is not.

## Direction bridge

The lab can compare model-B decoder directions to directions saved by prior labs. Set:

```bash
export LAB19_BRIDGE_STATE=/path/to/state_or_direction.pt
python interp_bench.py --lab lab19 --tier b --prompt-set full
```

The lab recursively searches the `.pt` file for one-dimensional tensors whose length matches model B's `d_model`. It then computes cosines between those directions and the top model-B feature decoders in residual units.

This bridge is a hypothesis generator, not a naming oracle. If a feature decoder has cosine 0.42 with a persona direction, the correct sentence is:

```text
Feature f aligns with the saved Lab 17 persona direction under this residual-space comparison.
```

The incorrect sentence is:

```text
Feature f is persona.
```

Names are earned by counterexamples, not bestowed by cosine.

## Optional causal validation

Run:

```bash
python interp_bench.py --lab lab19 --tier b --prompt-set full --run-edit
```

The lab chooses one or more model-B-skewed candidate features, converts the model-B decoder from normalized units back to residual units, and uses it as an activation-addition vector on benign prompts. It compares:

```text
baseline
feature_plus
feature_plus_low
feature_minus
random_plus
random_minus
```

The intervention is intentionally modest. It scores generated text with automatic marker heuristics for default-assistant voice, politeness, hedging, refusal, self-situation, disclosure, verbosity, and repetition. It also leaves blank hand-label columns.

The hand labels are not decoration. Marker scoring can confuse a longer answer with a more assistant-like answer, or a cautious answer with a refusal. A defended causal claim needs the random controls and hand labels to agree.

## Visualization upgrade: read the audit board before naming features

Lab 19 now writes a richer visual artifact set. The goal is not to make crosscoder features look magical. The goal is to make every tempting name walk through the control gates in public.

Recommended reading path:

```text
model_diffing_card.md
  -> plots/model_diffing_evidence_dashboard.png
  -> tables/model_diffing_evidence_matrix.csv
  -> plots/feature_audit_matrix.png
  -> plots/taxonomy_control_ladder.png
  -> plots/prompt_inventory_balance.png
  -> plots/activation_norm_shift_atlas.png
  -> plots/feature_context_atlas.png
  -> plots/direction_bridge_matrix.png       # if LAB19_BRIDGE_STATE is set
  -> plots/causal_operating_frontier.png     # if --run-edit is used
  -> plots/identity_smoke_scorecard.png      # especially Tier A / identity-pair runs
```

The new dashboard has four panels:

| Panel | What to ask |
|---|---|
| evidence ledger | Which rung is strong, weak, skipped, or control-limited? |
| feature taxonomy after controls | How many model-B-looking features remain after template, family, stability, and random-baseline pressure? |
| norm and token confounds | Did one model simply produce larger residual norms or different token loads for a family/variant? |
| optional causal operating point | Did feature steering beat random controls without buying the effect through verbosity, hedging, or refusal? |

The new `feature_audit_matrix.csv` joins the columns students usually inspect separately: activation share, decoder share, A/B correlation, template gap, family concentration, variant concentration, train/test activity, bridge cosine, audit posture, and claim boundary. This is the main artifact for deciding whether a feature is a candidate model-diff handle, a template feature, a family-specific feature, a shared feature, or dictionary residue.

### How to read the new plots

`crosscoder_training_diagnostics.png` checks whether the paired dictionary trained stably. `feature_audit_matrix.png` is the compact per-feature ledger. `taxonomy_control_ladder.png` shows how many features remain after controls. `prompt_inventory_balance.png` and `activation_norm_shift_atlas.png` check distribution, tokenization, and norm confounds. `feature_context_atlas.png` shows whether top activations are broad or one-family fireworks. `direction_bridge_matrix.png` compares feature decoders to saved prior-lab directions when configured. `causal_operating_frontier.png` is only a causal smoke board until random controls and hand labels agree. `identity_smoke_scorecard.png` is the Tier-A lie detector: an identity pair should not produce a glorious forest of model-specific features.

A good writeup should cite at least one feature that looked appealing in the gallery and then got narrowed by a control. The lab is working when it changes your favorite label.

## Custom prompt inventory

A CSV/TSV file may use these columns:

```text
prompt_id,prompt_group,family,variant,text,user_message,render_chat,also_chat,note
```

Minimum useful columns:

```text
prompt_id,family,text
```

If `render_chat=true`, the row is rendered using model B's chat template. If `also_chat=true`, the lab includes both the raw text and the chat-rendered text under the same `prompt_group`.

A JSON file can be either a list of records or:

```json
{
  "prompts": [
    {
      "prompt_id": "assistant_notes_01",
      "prompt_group": "assistant_notes_01",
      "family": "assistant_voice",
      "text": "Please help me organize these notes.",
      "also_chat": true
    }
  ]
}
```

## Writeup questions

1. Which three model-B-skewed features look most interpretable before controls? Which one looks worst after controls?
2. Does the matched random-direction baseline make exclusivity cheap or rare? How different is the independent-side diagnostic?
3. Pick one gallery feature. Write a narrow label, then write the strongest counterexample from its own top contexts.
4. Does the feature survive the raw-vs-chat template control? If not, what is the narrower label?
5. If you configured `LAB19_BRIDGE_STATE`, which feature-direction cosine is largest? What would you need before using the prior-lab direction's name for the feature?
6. If you ran `--run-edit`, did feature steering beat random steering on behavior, not just length or marker count?

## Ledger templates

Positive but careful:

```text
[L19-C1] DECODE/ATTR | At stream depths A=dA and B=dB, a paired crosscoder over N matched prompts found K model-B-skewed features with eval FVU fA/fB. M of those survived the template-control and random-direction audits. This supports a model-pair feature taxonomy under this prompt distribution, not an instruction-following mechanism.
Artifact: runs/.../tables/feature_taxonomy.csv
Audit: mixed. Template-control gaps remain high for several top features.
Falsifier: held-out prompt families or a larger dictionary erase the model-B-skewed features.
```

Negative and still valuable:

```text
[L19-C1] DECODE/ATTR retired | The identity-pair smoke run produced many model-specific features, so the crosscoder setup is not stable enough to support model-diff claims yet. High asymmetry without model-specific features is a weaker warning: the dictionary is less stable, but it has not yet created a false base/instruct feature.
Artifact: runs/.../model_diffing_card.md
Audit: failed identity control.
Falsifier: rerun with tied prompts and a smaller dictionary yields mostly shared/dead features.
```

Causal extension, only with hand labels:

```text
[L19-C2] CAUSAL | Adding feature f's model-B decoder at layer l changed hand-labeled assistant-voice behavior by Δ over random-feature controls on benign prompts.
Artifact: runs/.../tables/causal_feature_validation_summary.csv
Audit: passed marker, random, repetition, and hand-label controls.
Falsifier: the effect disappears on held-out benign prompts or is matched by norm-matched random features.
```

Bad claim, do not write:

```text
Feature 17 is where alignment lives.
```

That sentence stepped over every rung of the ladder and fell into the moat.

## Debugging

| Symptom | First place to look | Likely cause |
|---|---|---|
| model B fails hook parity | `diagnostics/model_b_hook_parity.json` | unsupported architecture hook convention or accidental extra special token |
| identity pair has many model-specific features | `identity_smoke_scorecard.png`, `feature_exclusivity_histogram.png`, `random_feature_baseline.csv` | dictionary too wide, too few prompts, broken shared-side initialization, or unstable training |
| many model-B features are chat-only | `template_control_summary.csv` | chat-template residue, not durable behavior |
| FVU is high | `crosscoder_training_curve.csv` | too few steps, too many features, bad normalization, or site too hard |
| GPU memory pressure on non-identity pair | `diagnostics/primary_model_memory_release.json` | set `LAB19_OFFLOAD_PRIMARY_TO_CPU=1` to move model A to CPU before loading model B |
| top features all come from one dataset | `feature_context_gallery.csv` | prompt-family imbalance |
| optional steering changes everything | `causal_feature_validation.csv` | dose too high or feature is a broad norm/verbosity direction |

## Why this lab matters

Labs 20 and 21 ask where training and safety changes live. Lab 23 asks whether the toolkit can recover a hidden behavior blind. Lab 19 is the first feature-level diffing instrument in that chain. It is allowed to fail loudly. A clean failure is better than a beautiful model-diff story, because the advanced half is not a museum of plots. It is a machine for attaching the right amount of belief to the right instrument reading.
