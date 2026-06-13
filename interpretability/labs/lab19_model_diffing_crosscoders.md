# Lab 19 - Model Diffing With Crosscoders

## Question

When a base model becomes an instruct model, what changes in its representations, and where does the "default assistant voice" live if it lives anywhere measurable?

This lab starts that question with a paired crosscoder. It compares final-token residual activations from two models on the same prompt inventory, trains a small shared feature dictionary, and classifies features as shared, base-skewed, instruct-skewed, asymmetric, or dead.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab19 --tier a --no-plots
python interp_bench.py --lab lab19 --tier b --prompt-set full
```

Tier A is a plumbing smoke test: by default it compares `EleutherAI/pythia-160m` to itself. That should mostly produce shared/noisy features, not science. Tier B is wired for the intended OLMo base-vs-instruct comparison:

```text
model A: allenai/Olmo-3-1025-7B
model B: allenai/Olmo-3-7B-Instruct
```

Override the comparison model without changing the bench registry:

```bash
LAB19_COMPARE_MODEL=some/model python interp_bench.py --lab lab19 --tier b
LAB19_STREAM_DEPTH=12 python interp_bench.py --lab lab19 --tier b
```

Optional causal smoke test:

```bash
python interp_bench.py --lab lab19 --tier b --run-edit
```

## What It Does

The starter implementation:

- builds a prompt inventory from Labs 14, 16, 17, 18, plus authored assistant-voice probes;
- collects matched residual activations from model A and model B;
- normalizes each activation space;
- trains a small paired sparse crosscoder in plain PyTorch;
- exports a feature taxonomy and top-context gallery;
- writes prompt-family, token-count, activation-norm, and voice-marker controls;
- saves the crosscoder state for notebook extensions;
- writes a placeholder direction-bridge table for joining chosen Lab 16/17/18 state files;
- optionally steers model B with one instruct-skewed decoder vector under `--run-edit`.

## Main Artifacts

| Path | What it contains |
|---|---|
| `diagnostics/model_pair.json` | model IDs, depths, dimensions, and prompt inventory metadata |
| `diagnostics/activation_norms.csv` | per-prompt token counts and residual norms for both models |
| `tables/prompt_inventory.csv` | every prompt used for matched activation collection |
| `tables/crosscoder_training_curve.csv` | training loss snapshots |
| `tables/feature_taxonomy.csv` | shared/base-skewed/instruct-skewed/asymmetric/dead labels |
| `tables/instruct_only_feature_gallery.csv` | top contexts for selected features, with label columns |
| `tables/default_voice_marker_rates.csv` | prompt-text marker control rates by family and variant |
| `tables/feature_direction_bridge.csv` | scaffold for joining saved direction state files |
| `tables/causal_feature_validation.csv` | optional `--run-edit` intervention output |
| `plots/feature_exclusivity_histogram.png` | comparison-model activation-share histogram |
| `plots/feature_direction_bridge.png` | placeholder plot for the bridge extension |
| `state/crosscoder_state.pt` | crosscoder weights, normalization stats, and taxonomy |
| `model_diffing_report.md` | run summary |
| `operationalization_audit.md` | cheap explanations and allowed claims |
| `results.csv` | alias of `tables/feature_taxonomy.csv` |

## Evidence Discipline

Do not write:

- "This feature is instruction following."
- "This feature is the assistant's personality."
- "Instruct-only means alignment lives here."
- "The optional steering table proves the behavior" without hand labels and random controls.

Allowed claims are narrower:

- under this prompt distribution, a feature is skewed toward one model's activations;
- top contexts suggest a candidate label that still needs validation;
- a feature survives or fails template and prompt-family controls;
- optional steering is a scoped causal smoke test, not a full localization proof.

## Cheap Explanations

Start by trying to kill the exciting story:

- template tokens explain the feature;
- one prompt family dominates the feature;
- activation norms shifted globally;
- the dictionary is too small and manufactured exclusivity;
- prompt text already contains assistant-voice markers;
- random features steer similarly.

## Writeup Questions

1. What are the FVU values for model A and model B? Is the dictionary good enough to read?
2. How many features are shared, base-skewed, instruct-skewed, asymmetric, and dead?
3. Which instruct-skewed feature has the cleanest top-context gallery?
4. Does that gallery survive raw-vs-chat-template controls?
5. Are default-assistant markers already present in the prompt text?
6. If `--run-edit` was used, did feature steering beat the random-feature baseline?
7. What would be required before calling a feature a mechanism rather than a correlate?
