# Lab 13 - Emotion Geometry: Reading Affect vs Writing Affect

## Question

When an instruct model reads emotional text and when it writes emotional text, are those two activities using related residual-stream geometry?

This lab is a bridge from the intro-course tools into later questions about AI voice, personality, style, and authorship. It does not ask whether a model feels joy, sadness, anger, or fear. It asks whether paired affect contrasts leave measurable directions inside the model, whether those directions transfer between input comprehension and output generation, and whether an input-derived direction can causally shift generated affect.

## What You Build

The lab uses `data/affect_emotion_pairs.csv`, a frozen paired dataset:

- emotion-laden text vs neutral paraphrase for the same cause;
- emotion-writing prompt vs neutral-writing prompt for the same cause;
- target emotions: `joy`, `sadness`, `anger`, `fear`;
- confounds: surprising-neutral, positive-calm, and high-arousal-neutral rows.

The runner extracts two direction families:

- `comprehension_<emotion>`: residual direction when the model reads emotional text rather than the neutral match;
- `generation_<emotion>`: residual direction when the model is asked to write in that emotion rather than write neutrally.

Then it tests:

- comprehension direction -> comprehension examples;
- comprehension direction -> generation prompts;
- generation direction -> comprehension examples;
- generation direction -> generation prompts.

The cross terms are the main event. If they work, the lab has evidence for a shared read/write affect handle. If they fail, the model may still decode affect in each mode, but the two modes are not using the same easy linear handle.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab13 --tier a
python interp_bench.py --lab lab13 --tier b --prompt-set full
```

Useful variants:

```bash
# A quick focused run on two emotions
python interp_bench.py --lab lab13 --tier a --emotions joy,anger

# Fewer examples per emotion for debugging
python interp_bench.py --lab lab13 --tier a --max-examples 2 --no-plots
```

Lab 13 uses instruct models and chat templates, like Lab 7. Tier A uses `HuggingFaceTB/SmolLM2-135M-Instruct`; Tier B uses `allenai/Olmo-3-7B-Instruct`.

## Main Artifacts

| Path | What it contains |
|---|---|
| `results.csv` | alias of the main transfer table |
| `tables/emotion_probe_transfer.csv` | AUCs for real directions, shuffled controls, random controls, and sentiment control |
| `tables/cross_cause_generalization.csv` | leave-one-cause-out checks for each emotion/source/target pair |
| `tables/emotion_direction_cosines.csv` | cosine atlas for comprehension/generation emotion directions plus sentiment |
| `tables/confound_projection_audit.csv` | how surprise, calm positivity, and arousal confounds project onto emotion directions |
| `tables/steering_generations.csv` | generated text under baseline, input-direction steering, and random steering; includes a blank `hand_label` column |
| `tables/steering_effects.csv` | per-emotion steering effect over baseline and random |
| `plots/emotion_transfer_matrix.png` | read/write transfer heatmap |
| `plots/emotion_direction_cosines.png` | direction-cosine heatmap |
| `state/emotion_directions.pt` | saved direction tensors and metadata |
| `operationalization_audit.md` | what the run does and does not justify |

## Evidence Discipline

Use `DECODE` for transfer and cosine claims:

- "A comprehension-derived anger direction decodes generation prompts above chance."
- "Generation and comprehension joy directions have positive cosine."

Use `CAUSAL` only for the steering table:

- "Adding the input-derived joy direction increased target-affect score over random on this prompt family."

Do not write:

- "The model feels sadness."
- "This is the model's real personality."
- "Emotion is localized at layer N."

The honest phrasing is closer to:

> This model has a linear affect handle under this paired operationalization, and the handle transfers between reading and writing to degree X.

## Writeup Questions

1. Which emotions transfer best between comprehension and generation?
2. Are comprehension and generation directions more aligned within the same emotion than across emotions?
3. Does the sentiment control explain the result? Use the max absolute cosine with `sentiment_lab7_style`.
4. Do cause-held-out rows preserve the effect, or is the direction mostly topic/cause?
5. Does input-derived steering beat random steering after hand-auditing the generated examples?
6. If this lab were adapted to "model voice" or "personality", what would the paired contrast be?

## Common Failure Modes

If every emotion direction looks like the sentiment direction, the run found valence, not emotion-specific geometry.

If cross-cause rows collapse, the run found topic/cause.

If comprehension->generation fails but comprehension->comprehension succeeds, the model has a read-side affect probe, not a shared read/write handle.

If steering changes output affect but also makes generations strange, treat it as a control problem, not a clean voice-editing tool.
