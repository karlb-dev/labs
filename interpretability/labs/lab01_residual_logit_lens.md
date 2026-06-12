# Lab 1: Residual Stream and Logit Lens

**Evidence level targeted:** `OBS`. The central lesson is that a readout is an instrument, not a mind scanner.

## Core question

How does a model's next-token prediction emerge, sharpen, wobble, or flip as information moves through the residual stream?

Lab 1 is the course's calibration ritual. Before later labs decompose, patch, steer, ablate, or probe activations, you first need to know exactly what a residual-stream readout is saying and what it is not saying.

## What you will build

You will build a layer-by-layer **prediction biography** for controlled prompts. The shared bench, `interp_bench.py`, owns the microscope: model loading, run directories, hook checks, residual capture, logit-lens math, state dumps, diagnostics, and artifact indexing. This lab owns the experiment: prompt families, validation, event-depth metrics, aggregation, plots, interpretation scaffolding, and claim-ledger drafts.

```text
runs/lab01_residual_logit_lens-<timestamp>-<id>/
  run_config.json
  run_metadata.json
  artifact_index.json
  run_summary.md
  logit_lens_card.md                    # compact deliverable, start here
  results.csv                           # one row per example and depth
  metrics.json
  ledger_suggestions.md

  diagnostics/
    model_anatomy.md
    tokenizer_info.json
    tokenization_report.csv             # prompt, target, distractor tokenization decisions
    prompt_set_manifest.json            # exact prompt counts and hashes
    event_definitions.json              # event thresholds and non-claim warning
    hook_parity.json
    hook_parity_by_layer.csv
    logit_lens_self_check.json
    gpu_memory_after_load.json
    gpu_memory_at_end.json

  tables/
    prompt_manifest.csv                 # prompts that survived validation
    final_readout_audit.csv             # correctness, confidence, and target/distractor status
    trajectory_events.csv               # event depths and final metrics
    example_summary.csv                 # alias of trajectory_events.csv
    category_summary.csv                # headline category comparisons
    top1_transition_segments.csv        # compressed top-1 token biographies
    readout_phase_summary.csv           # embedding/early/mid/late/final summaries

  state/<example_id>/
    state_card.md
    tokens.csv
    lens_trajectory.csv
    logit_lens_topk.csv
    residual_stats_final_pos.csv
    residual_norms_by_position.csv
    residual_streams.pt                 # optional, only with --save-tensors

  plots/
    readout_dashboard.png               # entropy, KL, margin, cosine in one place
    p_target_by_depth.png
    target_rank_by_depth.png
    logit_diff_by_depth.png
    entropy_by_depth.png
    kl_to_final_by_depth.png
    top1_margin_by_depth.png
    cosine_to_final_by_depth.png
    residual_norm_by_depth.png
    residual_delta_norm_by_depth.png
    event_depths.png
    event_ordering.png
    event_depth_heatmap.png
    final_readout_scatter.png
    biography_<example>.png             # four-panel prediction biography
```

## The minimum theory you need

A decoder-only transformer keeps one vector per token position. This vector is the **residual stream**. Each block reads the stream, computes updates through attention and MLP submodules, and adds those updates back. The final next-token logits come from the residual stream at the final token position after all blocks:

```text
logits = lm_head(final_norm(stream_after_all_blocks_at_final_position))
```

For a model with `L` blocks, this lab defines:

```text
streams[0] = embedding output, before block 0
streams[k] = pre-final-norm residual stream after k blocks, for 1 <= k <= L
lens(k)    = lm_head(final_norm(streams[k]))
```

Two details are load-bearing:

1. **Hidden-state indexing is easy to get wrong.** In common Hugging Face decoder models, `output_hidden_states=True` returns `L+1` tensors, but the last tensor is already post-final-norm. The bench captures the final norm input with a pre-hook and assembles a consistent pre-final-norm `streams[0..L]` array.
2. **The raw logit lens borrows the final readout basis.** Applying `final_norm` and `lm_head` at a middle depth asks what the final vocabulary head would decode there. It does not prove that the next block consumes the representation in that vocabulary basis.

The logit lens is a translation device. Translation can be faithful enough to teach you something and still add its own accent.

## The three instrument locks

The lab does not start the science loop until three checks have a written artifact:

| Lock | Artifact | What it protects against |
|---|---|---|
| Residual capture parity | `diagnostics/hook_parity_by_layer.csv` | hidden-state indexing or hook placement errors |
| Final-depth lens parity | `diagnostics/logit_lens_self_check.json` | a lens that does not reproduce the model's real final logits |
| Prompt and label validation | `diagnostics/tokenization_report.csv`, `diagnostics/prompt_set_manifest.json` | multi-token labels, empty prompts, duplicate or malformed prompt sets |

A plot without these locks is a stained-glass window: pretty, luminous, and not load-bearing.

## Prompt families

The built-in prompt set has three main families plus optional controls:

| Family | Purpose | Example | Target | Distractor |
|---|---|---|---|---|
| `fact` | High-certainty completions | `The capital of France is` | ` Paris` | ` London` |
| `ambiguous` | Negative control for over-reading commitment | `The best way to solve the problem is` | none | none |
| `counterfactual` | Context overriding a memorized fact | `In this story, the capital of France is London. According to the story, the capital of France is` | ` London` | ` Paris` |
| `control` | Optional weak or scrambled prompts | `France capital the is of` | optional | optional |

Targets and distractors must tokenize to exactly one token for the active tokenizer. Leading spaces matter. The tokenization report is not bookkeeping confetti; it is part of the evidence.

One subtlety is intentional: a factual target can beat its matched distractor without being the model's final top-1 token. A base model may prefer a discourse continuation such as `known` after `The capital of France is`. Treat these as separate columns:

- `final_top1_is_target`
- `final_target_rank`
- `final_p_target`
- `final_logit_diff`
- `final_outcome`

Open `tables/final_readout_audit.csv` before calling an example a success or failure.

## What is measured at each depth

For each example and depth, the lab records:

| Measurement | Why it exists |
|---|---|
| top-k decoded tokens | the human-readable biography |
| target and distractor probability | labeled behavior, when labels exist |
| target rank and distractor rank | rank often improves before probability looks large |
| target minus distractor logit difference | a matched-pair score, less brittle than top-1 |
| entropy in bits | sharpness of the decoded distribution |
| top-1 margin over top-2 | confidence proxy |
| KL from final distribution | convergence to the final readout |
| cosine to final residual | geometric convergence, not the same as decoded convergence |
| residual norm and update norm | whether depth changes are large or tiny |

## Event depths

The lab reports several “when did it happen?” metrics because one decision-depth number is a thimble trying to catch rain.

| Metric | Meaning |
|---|---|
| `decision_depth` | first depth after which the final top-1 token remains top-1 |
| `target_first_top1` | first depth where the labeled target is top-1 |
| `target_stable_top1_depth` | first depth after which the labeled target remains top-1 |
| `target_first_beats_distractor_raw` | first depth where target logit exceeds distractor logit; diagnostic only |
| `target_first_beats_distractor` | first depth where target logit exceeds distractor by >1.0 and keeps the lead thereafter |
| `target_stable_beats_distractor` | first depth after which target logit keeps beating distractor |
| `target_rank_first_le_5` | first depth where target rank is 5 or better |
| `kl_to_final_first_le_0.5_bits` | first depth where the full readout distribution is within 0.5 bits of final |
| `cosine_to_final_first_ge_0.95` | first depth where the residual vector is close to final in cosine |

The raw first target-over-distractor crossing is kept because it is a useful failure mode. Two irrelevant low-ranked tokens can swap order long before either one is meaningfully decoded. The margin-gated version is the one to cite.

## Run

```bash
# 1. Smoke test. Always do this before spending GPU minutes.
python interp_bench.py --lab lab1 --tier a

# 2. Standard full run on the course model.
python interp_bench.py --lab lab1 --tier b --prompt-set full --topk 10

# 3. Add optional weak/scrambled controls.
python interp_bench.py --lab lab1 --tier b --prompt-set full --include-controls

# 4. Save raw residual tensors for custom analysis.
python interp_bench.py --lab lab1 --tier b --prompt-set small --save-tensors

# 5. Feature a specific example in the biography plot.
python interp_bench.py --lab lab1 --tier b --prompt-set full --showcase cf_capital_france_london

# 6. Custom prompts from JSON or CSV.
python interp_bench.py --lab lab1 --prompt-set my_prompts.json
python interp_bench.py --lab lab1 --prompt-set my_prompts.csv
```

Custom JSON prompt files are lists of objects:

```json
[
  {
    "example_id": "qa_france",
    "category": "fact",
    "prompt": "Q: What is the capital of France?\nA:",
    "target": " Paris",
    "distractor": " London",
    "note": "leading spaces matter"
  }
]
```

Custom CSV files use the same columns: `example_id,category,prompt,target,distractor,note`. Blank target and distractor cells are allowed for ambiguous prompts.

## Artifact reading path

Read artifacts in this order:

1. `logit_lens_card.md` - the compact deliverable with scope, headline numbers, draft claims, and non-claims.
2. `diagnostics/logit_lens_self_check.json` - confirms that `lens(L)` matches the model's real logits.
3. `diagnostics/hook_parity_by_layer.csv` - confirms the residual stream capture has the claimed semantics.
4. `diagnostics/tokenization_report.csv` and `diagnostics/prompt_set_manifest.json` - confirms the prompt set and label tokens.
5. `tables/final_readout_audit.csv` - separates correctness, confidence, and target-vs-distractor wins.
6. One `state/<example_id>/state_card.md` from a fact example, then one from a counterfactual example.
7. `plots/readout_dashboard.png` - entropy, KL, top-1 margin, and cosine in one panel set.
8. `plots/event_ordering.png` and `plots/event_depth_heatmap.png` - when events occur and when they never occur.
9. `plots/logit_diff_by_depth.png`, `plots/target_rank_by_depth.png`, and `plots/kl_to_final_by_depth.png` - three different convergence stories.
10. `tables/top1_transition_segments.csv` - the compressed biography of which tokens wore the crown at which depths.
11. `results.csv` - the long-form source for custom analysis.

## How to read the main plots

`readout_dashboard.png` is the first plot to read. Entropy falling means the lens distribution is sharpening. KL-to-final falling means the distribution is becoming final-like. Cosine rising means the vector is geometrically approaching the final vector. These curves often move at different times, and the gaps are the lesson.

`final_readout_scatter.png` asks whether confidence is correctness. A point can have low entropy and high top-1 probability while the labeled target is not top-1. That is not a plotting bug. It is the model doing a next-token task rather than your benchmark task.

`event_depth_heatmap.png` makes missing events gray. Gray is data. If `target_first_top1` is gray for a fact prompt but `target_rank_first_le_5` is early, the target became plausible without winning.

`biography_<example>.png` is now four panels: target/distractor probability, target-vs-distractor logit difference, target rank, and entropy/KL. Read all four before writing a sentence about emergence.

## Questions to answer

1. Which event stabilizes earliest for facts: target rank, target beating the distractor, target top-1, or KL-to-final? What does the ordering suggest?
2. Are ambiguous prompts high entropy at the final depth, or do they sometimes become confident anyway? Find one confident ambiguous example and explain why confidence is not correctness.
3. In counterfactual prompts, does the in-context answer beat the memorized distractor early, late, never, or from the start? Use both `logit_diff_by_depth.png` and `trajectory_events.csv`.
4. Does cosine-to-final rise before the readout distribution becomes close to final? Explain what this says about residual geometry versus decoded output.
5. Find one top-1 token segment in `top1_transition_segments.csv` that lasts many layers but is not the final token. What would you have overclaimed from a single middle-layer screenshot?
6. Does `decision_depth` ever look early for an ambiguous or control prompt? Why does that make the phrase “the model knew early” suspect?
7. What causal experiment would test whether a middle-layer representation is used? Name the activation, token position, source prompt, destination prompt, and behavioral metric you would patch.

## Claim ledger guidance

Use `ledger_suggestions.md` as a draft pile, not an oracle. Move 2 or 3 claims into `claim_ledger.md` only after editing the scope, artifact path, and falsifier.

Every Lab 1 claim should be tagged `OBS`. Do not write a causal claim yet. A good Lab 1 claim has the shape:

```text
[L01-C1] OBS | On <model>, for <prompt family>, <metric> occurred at <number>/<L> under the raw logit lens.
Artifact: runs/<run>/tables/category_summary.csv
Falsifier: a tuned lens or held-out prompt family moves the event materially earlier/later or changes which token stabilizes.
```

For target-vs-distractor claims, prefer `target_first_beats_distractor` or `target_stable_beats_distractor`. The raw first positive crossing is for debugging, not bragging.

## Interpretation and ethics

**Reading:** Dennett, “Real Patterns” excerpt. Backup: Lipton, “The Mythos of Model Interpretability.”

**Writing prompt:** Your lens shows the correct answer becoming top-1 at depth `k`. What pattern is real in the artifact, and what claim of the form “the model knows the answer at layer k” goes beyond it? Answer using one specific `state_card.md` and one specific plot from your run.

## Common bugs and what they look like

| Symptom | Likely cause | First file to inspect |
|---|---|---|
| `lens(L)` disagrees with the model | wrong final norm path, unexpected logit post-processing, quantization weirdness | `diagnostics/logit_lens_self_check.json` |
| hook parity mismatch | architecture hidden-state convention changed, block path wrong, device-map capture bug | `diagnostics/hook_parity_by_layer.csv` |
| many examples dropped | target or distractor is multi-token, or target and distractor collapse to same token id | `diagnostics/tokenization_report.csv` |
| target never appears but a synonym does | labels are brittle; not necessarily model failure | `state/<example>/logit_lens_topk.csv` |
| all categories show the same decision depth | metric artifact, prompt-length confound, or over-homogeneous prompt set | `tables/prompt_manifest.csv` |
| `p(target)` looks tiny even when target rank improves | probability mass is spread across many plausible tokens | `plots/target_rank_by_depth.png` |
| custom CSV drops ambiguous rows | target/distractor cells contain invisible whitespace instead of being blank | `diagnostics/tokenization_report.csv` |

## Extensions

### Manageable: answer-shaped facts versus declarative facts

Make a custom prompt set with paired templates:

```text
Q: What is the capital of France?\nA:
The capital of France is
```

Keep the same target and distractor. Compare `final_readout_audit.csv` and `event_ordering.png`. If the QA format makes targets top-1 more often, the lesson is that the logit lens reads a next-token task, and template choice is part of the task.

### Manageable: prompt-length matched controls

Write ambiguous and factual prompts matched for token count and syntax. This tests whether category differences are about certainty or about length and form.

### Manageable: base versus instruct comparison

Run the same prompt set on a base and instruction-tuned variant of the same model family. Do not apply a chat template unless the lab configuration explicitly requires it. Compare `category_summary.csv`, `final_readout_audit.csv`, and `trajectory_events.csv`.

### Ambitious: tuned lens comparison

Train or load a tuned lens for a few depths and compare raw versus tuned event depths. The key question is not “which one is true?” It is where the raw lens borrows the final basis badly enough to change the story.

### Ambitious: full-position biography

Instead of only reading the final position, decode every token position across depth. Produce a depth-by-position grid of target rank, residual norm, and KL-to-final. This is the bridge toward attention and patching.

## Reading

- Elhage et al., “A Mathematical Framework for Transformer Circuits”, for the residual-stream picture.
- nostalgebraist, “interpreting GPT: the logit lens”, for the original exploratory method.
- Belrose et al., “Eliciting Latent Predictions from Transformers with the Tuned Lens”, for the readout-bias correction.
