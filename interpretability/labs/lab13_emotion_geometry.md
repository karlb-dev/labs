# Lab 13: Emotion Geometry, Reading Affect vs. Writing Affect

**Advanced course, Group I. Evidence levels targeted:** `DECODE` for read/write transfer and confound-controlled direction geometry, plus narrowly scoped `CAUSAL` for activation-addition effects on generated text.

**Prerequisites:** Intro Labs 4 and 7. Lab 4 teaches the core skepticism: decodable does not mean used. Lab 7 teaches the steering template: a direction that moves behavior is a handle, not a mechanism. This lab carries those lessons into affect, style, voice, and later persona work.

## The question

When an instruct model reads emotional text and when it is prompted to write emotional text, do those two activities expose related residual-stream geometry?

This lab does **not** ask whether the model feels joy, sadness, anger, or fear. It asks a narrower, testable question:

```text
emotional text or prompt  -  neutral matched text or prompt
```

For the same cause, does that paired contrast produce linear directions that transfer between comprehension and write intent, survive confound controls, and causally shift generated affect when added during decoding?

The microscope is aimed at a handle, not a soul. Tiny distinction, very large blast radius.

## Why this lab exists

The advanced half studies concepts where the tempting word is much larger than the measurement: emotion, persona, sycophancy, hidden goals, belief revision, self-report. Lab 13 is the first of those slippery labs, so it is deliberately instrument-heavy. It teaches students to ask, every time a probe looks good:

```text
Did I find the concept, or did I find topic, sentiment, arousal, phrasing, or output style?
```

The result is useful even when negative. If comprehension directions decode comprehension but fail on write-intent prompts, the model may have a read-side affect signal without a shared read/write handle. If all directions point at the sentiment control, the run found valence, not emotion-specific geometry. A clean failure is a good lab outcome.

## Dataset

The science path uses `data/affect_emotion_pairs.csv`, generated deterministically and hashed in the data manifest. Each target row should include:

| Column | Meaning |
|---|---|
| `item_id` | unique row id |
| `emotion` | one of `joy`, `sadness`, `anger`, `fear` |
| `cause` | cause/topic group, such as bereavement, storm, reunion |
| `arousal` | low, medium, high |
| `valence` | positive, neutral, negative |
| `content_text` | emotional sentence to be read |
| `neutral_text` | neutral paraphrase about the same cause |
| `generation_prompt` | prompt asking for a sentence in the target emotion |
| `neutral_generation_prompt` | neutral writing prompt about the same cause |
| `confound` | blank for target rows; otherwise a confound label |
| `note` | provenance or design note |

The confound rows are not decoration. They are the tiny tripwires that catch the obvious fake discoveries:

| Confound | What it attacks |
|---|---|
| surprising-neutral | surprise and salience pretending to be fear |
| positive-calm | positive valence pretending to be joy |
| high-arousal-neutral | activation or urgency pretending to be emotion |
| negative-calm | negative valence pretending to be sadness, anger, or fear |

Tier A has an embedded smoke fallback if the frozen CSV is not present. The run will mark that as plumbing only. Do not ledger claims from fallback data.

## What the code does

For every target item, the runner renders four chat-templated prompts:

```text
comprehension_emotion: model reads emotional text
comprehension_neutral: model reads neutral paraphrase

generation_emotion: model is asked to write with target affect
generation_neutral: model is asked to write neutrally
```

It captures the residual stream at the final prompt token. The bench convention is:

```text
streams[k] = pre-final-norm residual stream after k decoder blocks
streams[0] = embedding output
streams[L] = final block output before final norm
```

A direction extracted at `stream_depth = k` is injected at `injection_layer = k - 1`, because steering adds to the output of decoder block `k - 1`. This lab writes both numbers into the state metadata so nobody falls into the stream-depth off-by-one trap.

## Direction families

For each emotion and source, the direction is a train-split mean difference:

```text
direction(source, emotion, depth)
  = mean_i [ stream(source_emotion_i, depth) - stream(source_neutral_i, depth) ]
```

The two source families are:

| Direction family | Meaning |
|---|---|
| `comprehension_<emotion>` | residual contrast when reading emotional text rather than neutral text |
| `generation_<emotion>` | residual contrast when prompted to write in the emotion rather than neutrally |

The split is grouped by cause inside each emotion, so a row about one cause does not train the direction while a near-neighbor of the same cause evaluates it.

## Main experiments

### 1. Read/write transfer

The transfer table asks four questions per emotion:

```text
comprehension direction -> comprehension examples
generation direction    -> generation examples
comprehension direction -> generation examples
generation direction    -> comprehension examples
```

The cross terms are the headline. High within-source AUC plus weak cross-source AUC means the model exposes affect differently when reading and when preparing to write. High cross-source AUC is evidence for a shared linear read/write handle, but only if the audits survive.

### 2. Depth selection with controls

The lab scans stream depths and selects the depth with the strongest **control-adjusted cross-source AUC**:

```text
real cross-source AUC - max(random cross-source AUC, shuffled cross-source AUC, chance)
```

This avoids picking a depth where everything is linearly separable, including nonsense. The selected depth is written to `tables/depth_selection.csv` and shown in `plots/emotion_depth_curve.png`.

### 3. Emotion specificity

Emotion-vs-neutral AUC can still be a generic affect detector. The specificity table asks whether the target emotion direction separates its own emotion from other emotions:

```text
projection_delta(target emotion) > projection_delta(other emotions)
```

It also includes same-valence and same-arousal comparison pools. This is where “sadness” gets tested against “negative text in general,” and where “fear” gets tested against “high arousal in general.”

### 4. Cross-cause generalization

For each emotion, the lab leaves one cause out, trains the direction on the other causes, and evaluates on the held-out cause. This is the topic/cause audit.

A sadness direction trained on bereavement that fails on weather-sadness or fictional-sadness is mostly a cause direction wearing a little black hat.

### 5. Sentiment and confound controls

The lab builds a Lab-7-style sentiment direction from `data/affect_valence.csv` and compares it to every emotion direction. For negative emotions, the signed sentiment control is flipped so it is a fair baseline rather than an artificially bad one.

It also projects confound rows onto all selected emotion directions. A strong projection on high-arousal-neutral rows should downgrade emotion-specific language.

### 6. Steering

Finally, the lab adds directions during generation at the selected injection layer. The steering sweep includes:

| Condition | Meaning |
|---|---|
| `baseline` | no steering |
| `input_direction` | comprehension-derived target emotion direction |
| `write_intent_direction` | generation-derived target emotion direction |
| `random_oriented` | random direction oriented on the train split |
| `shuffled_input_direction` | shuffled/sign-flipped input-derived control |
| `sentiment_control` | signed Lab-7-style sentiment direction |

The automatic target-affect score is lexicon-based and intentionally simple. `tables/steering_generations.csv` includes blank hand-label columns because a lexicon is not a judge. The writeup should inspect generations before moving a `CAUSAL` claim into the ledger.



## Visualization upgrade: the new reading path

Lab 13 now writes an evidence suite rather than a small pile of separate charts. The headline artifact is:

```text
plots/emotion_geometry_dashboard.png
```

Read it first. It puts four fragile pieces of the claim in the same room: the selected read/write-transfer depth, the per-emotion decode gates, steering over controls, and the confound rows trying to explain the result away. The dashboard is not the whole argument. It is the table of contents for the argument.

New or upgraded artifacts:

| Artifact | What it teaches |
|---|---|
| `plots/emotion_geometry_dashboard.png` | Whether transfer, specificity, steering, and confounds tell one coherent story. |
| `plots/depth_control_gap_atlas.png` | Whether depth selection found a real/control separation or merely a bright layer. |
| `plots/emotion_evidence_matrix.png` | One row per emotion, one column per evidence gate, raw values annotated. |
| `plots/specificity_ladder.png` | Whether an emotion direction beats all-other, same-valence, and same-arousal comparison pools. |
| `plots/cross_cause_matrix.png` | Which held-out causes break transfer, exposing topic leakage. |
| `plots/confound_projection_matrix.png` | Which surprise, calmness, arousal, or valence confounds project onto the directions. |
| `plots/steering_operating_frontier.png` | Which doses move affect more than controls while paying acceptable KL/repetition cost. |
| `plots/generation_response_atlas.png` | Whether the steering curve is broad or carried by one prompt shouting through a megaphone. |
| `tables/emotion_evidence_matrix.csv` | Per-emotion claim posture: candidate emotion handle, generic affect handle, unresolved, or failed. |
| `tables/steering_operating_points.csv` | Dose-level control gap, KL, repetition, and claimability flags before hand labels. |
| `tables/plot_reading_guide.csv` | A guide from plot name to the concept and claim boundary it protects. |

The upgrade deliberately separates three questions that are easy to mush together:

```text
Can a direction decode emotion-vs-neutral?       -> DECODE
Does it transfer between reading and writing?    -> DECODE + transfer
Does adding it alter generated affect safely?    -> scoped CAUSAL, pending hand labels
```

The plots are arranged to punish the favorite overclaim. If the read/write transfer looks good but the same-valence ladder collapses, the result is probably valence. If steering works but KL or repetition rises, the vector may be damaging generation rather than writing affect. If one cause or prompt carries the effect, the atlas should make that concentration visible before a broad claim is written.

## Run commands

From the course root:

```bash
# CPU smoke test. Checks plumbing, not science.
python interp_bench.py --lab lab13 --tier a

# Standard science run on the instruct model.
python interp_bench.py --lab lab13 --tier b --prompt-set full

# Focus on a subset of emotions while debugging.
python interp_bench.py --lab lab13 --tier a --emotions joy,anger

# Smaller per-emotion cap and no plots for quick iteration.
python interp_bench.py --lab lab13 --tier a --max-examples 2 --no-plots
```

Lab 13 uses chat templates. Tier A defaults to `HuggingFaceTB/SmolLM2-135M-Instruct`; Tier B/C default to `allenai/Olmo-3-7B-Instruct` in the current bench.

## Artifact tree

```text
runs/lab13_emotion_geometry-*/
  emotion_geometry_card.md                 # read first: verdicts, non-claims, headline numbers
  operationalization_audit.md              # cheap explanations and whether they survived
  run_summary.md                           # standard seven-question summary
  ledger_suggestions.md                    # drafts only, edit before appending
  metrics.json                             # machine-readable headline metrics and audit bars
  results.csv                              # alias of the main transfer table

  diagnostics/
    frozen_data_manifest.json              # dataset hash, manifest status, fallback warning
    sentiment_control_manifest.json         # valence-control data provenance
    split_audit.csv                         # cause-grouped train/eval split
    prompt_render_audit.csv                 # chat-rendered prompt lengths and final tokens

  tables/
    item_manifest.csv                       # selected target and confound rows
    emotion_family_manifest.csv             # counts, causes, split counts per emotion
    depth_selection.csv                     # depth sweep with real, random, shuffled, adjusted scores
    emotion_probe_transfer.csv              # main DECODE table with selected-depth controls
    cross_cause_generalization.csv          # leave-one-cause-out transfer
    emotion_specificity_vs_other_emotions.csv
    emotion_direction_cosines.csv           # cosine atlas plus sentiment control
    confound_projection_audit.csv           # per-confound projection rows
    confound_projection_summary.csv         # aggregate confound magnitudes
    steering_generations.csv                # all generations, scores, blank hand labels
    steering_effects.csv                    # dose-response summary by emotion
    steering_operating_points.csv           # dose/control/KL/repetition operating points
    emotion_evidence_matrix.csv             # one-row-per-emotion claim posture
    plot_reading_guide.csv                  # plot-to-concept guide
    generation_labeling_guide.md            # how to hand-audit generated text

  plots/
    emotion_geometry_dashboard.png          # start here: transfer + audit + steering cockpit
    emotion_depth_curve.png                 # depth selection with controls
    depth_control_gap_atlas.png             # compact real/control/gap depth atlas
    emotion_transfer_matrix.png             # read/write transfer heatmap plus controls
    emotion_evidence_matrix.png             # per-emotion evidence gates
    emotion_direction_cosines.png           # direction cosine heatmap
    specificity_ladder.png                  # all-other / same-valence / same-arousal audit
    emotion_specificity.png                 # target-vs-other emotion audit
    cross_cause_matrix.png                  # held-out-cause matrix
    cross_cause_generalization.png          # held-out-cause summary
    confound_projection_matrix.png          # confound-by-direction projection matrix
    confound_projection_audit.png           # largest confound projections
    affect_steering_effects.png             # steering dose-response against controls
    steering_operating_frontier.png         # dose/cost/control-gap frontier
    generation_response_atlas.png           # per-prompt steering response atlas

  state/
    emotion_directions.pt                   # direction tensors and metadata
    emotion_directions_metadata.json         # human-readable provenance
```

## How to read the run

Start with `emotion_geometry_card.md`. It tells you whether the run supports a read/write affect handle, a broader affect/valence handle, or no defended claim.

Then read `operationalization_audit.md`. The audit is not paperwork. It is the result wearing armor.

Then inspect the plots in this order:

1. `emotion_geometry_dashboard.png`: Does the run clear the whole evidence gauntlet at a glance?
2. `depth_control_gap_atlas.png` and `emotion_depth_curve.png`: Did the selected depth beat random and shuffled controls, or is the depth sweep a fog machine?
3. `emotion_transfer_matrix.png`: Which emotions transfer between reading and writing, and do controls ride lower?
4. `emotion_evidence_matrix.png`: Which emotion families have enough evidence for a claim posture stronger than “unresolved affect handle”?
5. `specificity_ladder.png` and `emotion_specificity.png`: Are the directions emotion-specific or just generic affect?
6. `emotion_direction_cosines.png`: Are comprehension and write-intent directions aligned within emotion, and are they too close to sentiment?
7. `cross_cause_matrix.png` and `cross_cause_generalization.png`: Does the result survive new causes?
8. `confound_projection_matrix.png` and `confound_projection_audit.png`: Which cheap explanation is most dangerous?
9. `steering_operating_frontier.png` and `affect_steering_effects.png`: Does input-derived steering beat random, shuffled, and sentiment controls without a big side-effect bill?
10. `generation_response_atlas.png` and `steering_generations.csv`: Do actual generations look target-affective without degenerating, and is the effect broad rather than one-row theater?

## Evidence discipline

Use `DECODE` for direction and transfer claims:

```text
L13-C2 DECODE: On model M, comprehension-derived and write-intent-derived anger directions have mean cross-source AUC X and same-emotion cosine Y at stream depth k, with control-adjusted cross AUC Z.
```

Use `CAUSAL` only for the activation-addition sweep, and only if the effect beats controls and survives hand inspection:

```text
L13-C1 CAUSAL: Adding the input-derived joy direction at layer k changed generated target-affect score by Δ over random on prompt family P at dose s.
```

Do **not** write:

```text
The model feels sadness.
The model has an anger module.
This is the model's true personality.
Emotion is localized at layer k.
```

A better sentence is:

```text
This model has a controlled linear affect handle under this paired operationalization, and the handle transfers between reading and writing to degree X.
```

## Interpreting common result patterns

| Pattern | Interpretation |
|---|---|
| High within-source AUC, weak cross-source AUC | reading affect and write-intent affect are linearly decodable but not shared by this simple handle |
| High cross-source AUC, low specificity | shared generic emotionality, not emotion-specific geometry |
| Strong sentiment cosine | mostly valence or sentiment, not a distinct emotion direction |
| Cross-cause collapse | topic/cause leakage |
| Confounds project strongly | surprise, calm positivity, or arousal is still alive as an explanation |
| Steering beats random but generations degenerate | activation addition damaged generation; do not call it affect steering |
| Input steering works, write-intent steering does not | comprehension-derived direction is a better causal handle than write-intent direction for this layer and prompt set |
| Steering works only at high dose | possible side-effect dominated handle; inspect KL, repetition, and hand labels |

## Writeup questions

1. Which emotions transfer most cleanly between comprehension and write intent?
2. Is transfer symmetric? Compare comprehension-to-generation and generation-to-comprehension cells.
3. Does the selected depth beat random and shuffled controls, or only chance?
4. Does the direction separate the target emotion from other emotions, especially same-valence and same-arousal comparisons?
5. Do cause-held-out rows preserve the result?
6. Is the sentiment control too close to the emotion directions?
7. Which confound row is most dangerous for your favored claim?
8. Does input-derived steering beat random, shuffled, and sentiment controls at the headline dose?
9. After hand-labeling generated text, would you still defend the causal claim?
10. What would the paired contrast be for Lab 17 persona, voice, or authorship?

## Debugging guide

| Symptom | Likely cause | What to inspect |
|---|---|---|
| Lab refuses to run on a base model | no chat template | use an instruct model or the tier defaults |
| Frozen data missing on Tier B/full | data generator not run or data not committed | `diagnostics/frozen_data_manifest.json` |
| Every cross-source AUC is blank | train/eval split too tiny | increase `--max-examples` or use `--prompt-set medium/full` |
| Depth 0 or final depth dominates | prompt surface or late readout artifact | `tables/depth_selection.csv` and controls |
| Sentiment control dominates | valence explains the result | `emotion_direction_cosines.csv` and `confound_projection_summary.csv` |
| Steering outputs repeat words | dose too high or vector not specific | `steering_generations.csv`, `next_token_kl_to_baseline_bits`, repetition columns |
| A single emotion has no eval rows | cause grouping plus tiny data | run more examples per emotion |

## Ledger templates

Strong positive, after hand audit:

```text
[L13-C1] CAUSAL | Adding the input-derived <emotion> direction at layer <k> of <model> increased generated target-affect score by <Δ> over random and <Δ2> over shuffled at dose <s>, on <n> held-out prompts. This is a scoped activation-addition handle, not evidence of felt emotion.
Artifact: runs/<run>/tables/steering_effects.csv | Falsifier: hand labels disagree, controls match the effect, or held-out causes collapse.
```

Decode bridge:

```text
[L13-C2] DECODE | On <model>, comprehension and write-intent directions for <emotion set> show mean cross-source AUC <X>, same-emotion cosine <Y>, and control-adjusted cross AUC <Z> at stream depth <k>. The claim is conditioned on sentiment cosine <c> and cause-held-out AUC <h>.
Artifact: runs/<run>/tables/emotion_probe_transfer.csv | Falsifier: a cause-balanced dataset or sentiment/arousal control explains the effect.
```

Negative result, still useful:

```text
[L13-N1] DECODE | This run did not validate an emotion-specific read/write handle: cross-source AUC was <X>, specificity AUC was <Y>, and the audit verdict was <verdict>. The result supports an operationalization failure or broader affect signal, not the intended emotion-specific claim.
Artifact: runs/<run>/operationalization_audit.md | Falsifier: a larger cause-balanced run with stronger specificity and clean controls.
```

## Ethics and interpretation

Emotion words are socially loaded. A model can generate a sad sentence, internally separate sad prompts from neutral prompts, and even be steerable toward sad output without feeling sadness. The lab's claim lives at the level of operationalized residual geometry and intervention effects. That is already interesting. It is not a license to treat the model as having experiences, preferences, trauma, sincerity, or personality.

A good writeup should say exactly what the direction lets you predict or change, and exactly what remains outside the evidence. The poetry belongs in the examples. The ledger gets the steel ruler.
