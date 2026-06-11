# Lab 1: Residual Stream and Logit Lens

**Evidence level targeted:** `OBS` observation. The central lesson is that a readout is an instrument, not a mind scanner.

## Core question

How does a model's next-token prediction emerge, sharpen, wobble, or flip as information moves through the residual stream?

## What you will build

You will build a layer-by-layer prediction biography for a small set of controlled prompts. The shared harness, `interp_bench.py`, owns the microscope: model loading, run directories, hook checks, residual capture, logit-lens math, state dumps, diagnostics, and artifact indexing. This lab module owns the experiment: prompts, validation, metrics, aggregation, plots, interpretation scaffolding, and suggested ledger claims.

```text
runs/lab01_residual_logit_lens-<timestamp>-<id>/
  run_config.json                         # exact CLI after tier defaults
  run_metadata.json                       # packages, host, git, GPU, env
  artifact_index.json                     # map of every saved artifact
  run_summary.md                          # start here
  results.csv                             # one row per example and depth
  metrics.json                            # machine-readable aggregates

  diagnostics/
    model_anatomy.md                      # where blocks, final norm, lm_head live
    tokenizer_info.json                   # special tokens, vocab, padding, chat-template status
    tokenization_report.csv               # every target and distractor validation decision
    hook_parity.json                      # self-check summary
    hook_parity_by_layer.csv              # layer-level hook-vs-hidden-state diffs
    logit_lens_self_check.json            # lens(L) vs model logits
    gpu_memory_after_load.json
    gpu_memory_at_end.json

  tables/
    prompt_manifest.csv                   # the prompt set that actually ran
    example_summary.csv                   # per-example event depths and final readouts
    category_summary.csv                  # headline category comparisons
    trajectory_events.csv                 # first-crossing and stabilization events

  state/<example_id>/
    state_card.md                         # readable per-example narrative
    tokens.csv                            # exactly what the tokenizer fed the model
    lens_trajectory.csv                   # top-1, entropy, ranks, KL, margins by depth
    logit_lens_topk.csv                   # top-k decoded readout by depth
    residual_stats_final_pos.csv          # stream statistics at the readout position
    residual_norms_by_position.csv        # depth x token-position norm grid
    residual_streams.pt                   # optional, only with --save-tensors

  plots/
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
    event_depth_heatmap.png
    final_readout_scatter.png
    biography_<example>.png

  ledger_suggestions.md                   # draft OBS claims, edit before accepting
```

## Why this matters

Almost every later lab reads from or writes to the residual stream. Direct logit attribution decomposes changes in the residual stream. Attention labs ask which token positions route information into it. Patching edits it. Steering injects directions into it. SAE and graph labs re-describe pieces of it. This lab is the first calibration ritual: before drawing conclusions from an activation plot, prove the instrument knows what it captured.

The logit lens is useful because it converts a hidden vector into a distribution over human-readable tokens. It is dangerous for the same reason: the decoded tokens invite a story. This lab teaches you to enjoy the story, then put it in a glass jar labeled **readout artifact until further evidence**.

## Setup

- **Library path:** plain Hugging Face `transformers` with explicit PyTorch hooks. No interpretability framework is required for Lab 1.
- **Primary model, Tier B/C:** `allenai/Olmo-3-1025-7B`, base model, bf16 for Tier B and fp32 for Tier C unless overridden.
- **Smoke model, Tier A:** `gpt2` on CPU or MPS, capped examples. This path tests plumbing, not science.
- **Templates:** no chat template is applied in this lab. These are base-model next-token completions. Template discipline becomes load-bearing in later instruct and reasoning labs.

## Runtime and tiers

| Tier | Hardware | Command | Expected role |
|---|---|---|---|
| A | Laptop CPU or MPS | `python interp_bench.py --lab lab1 --tier a` | smoke test, hook/lens diagnostics, 4 examples |
| B | A100/H100 or 24 GB+ GPU | `python interp_bench.py --lab lab1 --tier b --prompt-set full` | default scientific run |
| C | 40 to 80 GB GPU | `python interp_bench.py --lab lab1 --tier c --prompt-set full` | fp32 reference run, larger prompt variants |

Memory is dominated by model weights, not saved activations. Each prompt is short, unbatched, and cached one at a time. Quantization can help small GPUs, but record it in the interpretation because it changes the numerics of the readout.

## Background: the minimum theory to act

A decoder-only transformer keeps one vector per token position. This vector is the **residual stream**. Each block reads the current stream, computes updates through attention and MLP submodules, and adds those updates back. The final next-token logits are produced by applying the final normalization layer and output projection to the residual stream at the last token position.

For a model with `L` blocks, this lab defines:

```text
streams[0] = embedding output, before block 0
streams[k] = pre-final-norm residual stream after k blocks, for 1 <= k <= L
lens(k)    = lm_head(final_norm(streams[k]))
```

Two details matter enough to be diagnostic tests:

1. **Hidden-state indexing is easy to get wrong.** In common Hugging Face decoder models, `output_hidden_states=True` returns `L+1` tensors, but the last tensor is already post-final-norm. The raw stream after the last block is not directly in the tuple. The bench captures the final norm input with a pre-hook and assembles a consistent `streams[0..L]` array.
2. **The middle-layer readout borrows the final readout basis.** Applying the final norm and unembedding at depth `k < L` asks what the final readout would decode there. It does not prove that the next block consumes the representation that way.

A good mental model: the logit lens is a stethoscope pressed to the residual stream. It can reveal rhythm, but it is not the heart.

## Experimental design

### Prompt families

The built-in prompt set has three main families:

| Family | Purpose | Example | Target | Distractor |
|---|---|---|---|---|
| `fact` | High-certainty completions | `The capital of France is` | ` Paris` | ` London` |
| `ambiguous` | Negative control for over-reading commitment | `The best way to solve the problem is` | none | none |
| `counterfactual` | Context overriding a memorized fact | `In this story, the capital of France is London. According to the story, the capital of France is` | ` London` | ` Paris` |

Optional controls can be enabled with `--include-controls`. These are intentionally weak or scrambled continuations. They are not meant to be clever benchmarks, just tripwires for metrics that would look impressive on nonsense.

One subtlety is intentional: a factual target can beat its matched distractor
without being the model's final top-1 token. A base model may prefer a
discourse continuation such as "known" after `The capital of France is`.
Treat `final_top1_is_target`, `final_target_rank`, and target-vs-distractor
logit difference as separate facts about the artifact, not interchangeable
notions of correctness.

### Prompt-set upgrades for stronger contrasts

The built-in set is deliberately mixed: some prompts are answer-shaped, while
others are ordinary text continuations where the model may prefer a discourse
token. That awkwardness is useful. For a cleaner second run, make a custom
JSON prompt set with paired templates:

- `qa_fact`: `Q: What is the capital of France?\nA:` -> ` Paris`
- `declarative_fact`: `The capital of France is` -> ` Paris`
- `matched_ambiguous`: same token length and syntax shape, but no single
  privileged answer
- `counterfactual_qa`: context override followed by an answer-shaped question

Then compare `final_readout_scatter.png` and `event_depth_heatmap.png` across
the two runs. If the QA facts become target-top-1 much more often than the
declarative facts, the lesson is not that one prompt set is "right"; it is that
the logit lens is reading a next-token task, and template choice is part of the
task.

### What is measured at each depth

For each example and each depth, the lab records:

- top-k decoded tokens and probabilities
- target probability, distractor probability, and target minus distractor logit difference when labels exist
- target rank and distractor rank, because a target can improve long before it becomes top-1
- entropy in bits
- top-1 margin over top-2, a simple confidence proxy
- KL divergence from the final depth distribution, a convergence-to-final metric
- cosine similarity to the final residual stream
- residual norm and residual update norm from the previous depth

### Event depths

The lab reports several distinct “when did it happen?” metrics because a single decision-depth number is a small net for a slippery fish:

| Metric | Meaning |
|---|---|
| `decision_depth` | first depth after which the final top-1 token remains top-1 |
| `target_first_top1` | first depth where the labeled target is top-1 |
| `target_stable_top1_depth` | first depth after which the labeled target remains top-1 |
| `target_first_beats_distractor_raw` | first depth where target logit exceeds distractor logit; diagnostic only, often noisy in junk-readout regimes |
| `target_first_beats_distractor` | first depth where target logit exceeds distractor by >1.0 and keeps the lead thereafter |
| `target_stable_beats_distractor` | first depth after which target logit keeps beating distractor |
| `target_rank_first_le_5` | first depth where target rank is 5 or better |
| `kl_to_final_first_le_0.5_bits` | first depth where the whole distribution is within 0.5 bits of final |

The differences between these event depths are often more educational than the headline plot.

## Run

```bash
# 1. Smoke test. Do this before spending GPU minutes.
python interp_bench.py --lab lab1 --tier a

# 2. Standard full run.
python interp_bench.py --lab lab1 --tier b --prompt-set full --topk 10

# 3. Add optional weak/scrambled controls.
python interp_bench.py --lab lab1 --tier b --prompt-set full --include-controls

# 4. Save raw residual tensors for custom analysis.
python interp_bench.py --lab lab1 --tier b --prompt-set small --save-tensors

# 5. Custom prompts.
python interp_bench.py --lab lab1 --prompt-set my_prompts.json
```

Custom prompt files are JSON lists. Targets and distractors are optional, but when provided they must tokenize to a single token for the current tokenizer.

```json
[
  {
    "example_id": "my_fact",
    "category": "fact",
    "prompt": "The capital of France is",
    "target": " Paris",
    "distractor": " London",
    "note": "leading spaces matter"
  }
]
```

## Artifact reading path

Read artifacts in this order:

1. `run_summary.md`, for the headline numbers and the caveat list.
2. `diagnostics/logit_lens_self_check.json`, to confirm the lens reproduces the real final prediction.
3. `diagnostics/hook_parity_by_layer.csv`, to confirm the residual stream capture has the claimed semantics.
4. One `state/<example_id>/state_card.md` from a fact example, then one from a counterfactual example.
5. `plots/final_readout_scatter.png`, to separate confidence, entropy, and target success.
6. `plots/event_depth_heatmap.png`, to see which event metrics never occurred.
7. `plots/logit_diff_by_depth.png`, `plots/target_rank_by_depth.png`, and `plots/kl_to_final_by_depth.png` together. These show different notions of convergence.
8. `tables/trajectory_events.csv`, to find examples where the plot story is too simple.
9. `results.csv`, to reproduce or challenge any aggregate.

## Questions to answer

1. Which event depth stabilizes earliest for facts: target rank, target logit beating the distractor, target top-1, or KL-to-final? What does the ordering suggest?
2. Are ambiguous prompts high entropy at the final depth, or do they sometimes become confident anyway? Find one example where confidence is not correctness.
3. In counterfactual prompts, does the in-context answer beat the memorized distractor early, late, never, or from the start? Use both `logit_diff_by_depth.png` and `trajectory_events.csv`.
4. Does cosine-to-final rise before the readout distribution becomes close to final? Explain what this says about geometry versus decoded output.
5. Does `decision_depth` ever look early for an ambiguous or control prompt? Why would that make the phrase “the model knew early” suspect?
6. What causal experiment would test whether a mid-layer representation is used, rather than merely decodable? Name the activation, token position, source prompt, destination prompt, and behavioral metric you would patch.

## Ledger

Use `ledger_suggestions.md` as a draft pile, not a truth machine. Move 2 or 3 claims into `claim_ledger.md` only after editing the scope, artifact path, and falsifier.

Every Lab 1 claim should be tagged `OBS`. Do not write a causal claim yet. A good Lab 1 claim has the shape:

```text
[L01-C1] OBS | On <model>, for <prompt family>, <metric> stabilized at <number> / L under the raw logit lens.
Artifact: runs/<run>/tables/category_summary.csv
Falsifier: tuned lens or held-out prompts move the event depth materially earlier/later.
```

For target-vs-distractor claims, prefer the margin-gated
`target_first_beats_distractor` or `target_stable_beats_distractor` columns.
The raw first positive crossing is kept because it is a useful failure mode:
two irrelevant low-ranked tokens can swap order long before either is
meaningfully decoded.

## Interpretation and ethics

**Reading:** Dennett, “Real Patterns” excerpt. Backup: Lipton, “The Mythos of Model Interpretability.”

**Writing prompt:** Your lens shows the correct answer becoming top-1 at depth `k`. What pattern is real in the artifact, and what claim of the form “the model knows the answer at layer k” goes beyond it? Answer using one specific `state_card.md` and one specific plot from your run.

## Common bugs and what they look like

| Symptom | Likely cause | First file to inspect |
|---|---|---|
| lens at depth `L` disagrees with the model | wrong final norm path, unexpected logit post-processing, quantization weirdness | `diagnostics/logit_lens_self_check.json` |
| hook parity mismatch | architecture hidden-state convention changed, block path wrong, device-map capture bug | `diagnostics/hook_parity_by_layer.csv` |
| many examples dropped | target or distractor is multi-token for this tokenizer | `diagnostics/tokenization_report.csv` |
| target never appears but a synonym does | labels too brittle, not a model failure by itself | `state/<example>/logit_lens_topk.csv` |
| all categories show the same decision depth | metric artifact, prompt-length confound, or over-homogeneous prompt set | `tables/prompt_manifest.csv` |
| p(target) looks tiny even when target rank improves | probability mass is spread across many plausible tokens | `target_rank_by_depth.png` |

## Extensions

### Manageable: base versus instruct comparison

Run the same prompt set on a base and instruction-tuned variant of the same model family, then compare `category_summary.csv` and `trajectory_events.csv`. Do not apply a chat template unless the model path and lab configuration explicitly require it.

### Manageable: prompt-length matched controls

Write a custom JSON prompt set where ambiguous and factual prompts are matched for token count. This tests whether category differences are really about certainty or just length and syntax.

### Ambitious: tuned lens comparison

Train or load a tuned lens for a few depths and compare raw versus tuned event depths. The main question is not “which one is true?” but “where does the raw lens borrow the final basis badly enough to change the story?”

### Ambitious: full-position biography

Instead of only reading the final position, decode every token position across depth and produce a depth-by-position grid of target rank, residual norm, and KL-to-final. This is the bridge toward attention and patching labs.

## Reading

- Elhage et al., “A Mathematical Framework for Transformer Circuits”, for the residual-stream picture.
- nostalgebraist, “interpreting GPT: the logit lens”, for the original exploratory method.
- Belrose et al., “Eliciting Latent Predictions from Transformers with the Tuned Lens”, for the readout-bias correction.
