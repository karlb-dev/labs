# Lab 1: Residual Stream and Logit Lens

**Evidence level targeted:** observation — with the explicit lesson that a readout is not a mind-scan.

## Core question

How does a model's running prediction emerge across layers?

## What you will build

A per-layer "prediction biography" for ~10–24 prompts across three families, produced by the shared bench (`interp_bench.py`) plus this lab's experiment module:

```text
runs/lab01_residual_logit_lens-<timestamp>/
  run_summary.md                      # start here: the seven questions, answered
  results.csv                         # every (example, depth) measurement
  tables/example_summary.csv          # decision depths per example
  tables/category_summary.csv        # the headline per-category table
  state/<example_id>/state_card.md    # per-example narrative dump (read one!)
  state/<example_id>/*.csv            # tokens, lens top-k, trajectory, residual stats
  plots/p_target_by_depth.png
  plots/logit_diff_by_depth.png
  plots/entropy_by_depth.png
  plots/cosine_to_final_by_depth.png
  plots/residual_norm_by_depth.png
  plots/biography_<example>.png
  diagnostics/                        # anatomy, hook parity, lens self-check, tokenization report
  ledger_suggestions.md               # drafted claims — edit before they enter your ledger
```

## Why this matters

Everything later in the course — attribution, patching, steering, attribution graphs — reads or writes the residual stream. This lab makes that object concrete: the stream is the model's shared workspace, the unembedding is a fixed readout of it, and "what the model currently predicts" is something you can measure at every depth, not just at the output. It is also the course's first encounter with its most important habit: before trusting any plot, verify the instrument (this run literally checks its own hooks and its own lens against the model's real logits, every time).

## Setup

- **Library:** plain Hugging Face `transformers` with explicit PyTorch hooks. No interpretability framework — every captured tensor's provenance is visible in `interp_bench.py`.
- **Primary model (Tier B/C):** `allenai/Olmo-3-1025-7B`, bf16.
- **Smoke model (Tier A):** `gpt2` on CPU. Must work before you touch a GPU.
- These are base models: no chat template is applied anywhere in this lab. (Template discipline becomes load-bearing in Lab 7+.)

## Runtime and tiers

| Tier | Hardware | Command | Expected wall-clock |
|---|---|---|---|
| A — smoke | laptop CPU | `--tier a` (gpt2, 4 examples) | ~1–3 min incl. download |
| B — standard | Colab A100/H100 or any 24 GB+ GPU | `--tier b` | < 10 min incl. model download |
| C — comfortable | 40–80 GB GPU | `--tier c` (fp32) | < 10 min |

Memory notes: a 7B model in bf16 needs ~15 GB of GPU memory; the captured per-prompt state is megabytes (short prompts, one example at a time, no batching). On a small GPU, `--quantization 8bit` works but loosens the lens self-check — the run records it.

## Background (the minimum theory to act)

A decoder transformer with L blocks keeps one running vector per token position — the **residual stream** — that every block reads from and writes (adds) into. The final prediction is produced by applying the final norm and the unembedding matrix to the last position's stream. The **logit lens** asks: what would the prediction be if we applied that same readout to the stream at depth k < L?

Two facts make the readout subtle, and the bench handles both explicitly:

1. **Indexing.** HF `output_hidden_states` returns L+1 tensors whose last entry is *post-final-norm*; the raw output of the last block isn't in the tuple. The bench captures it with a pre-hook on the final norm and defines `streams[k]` = pre-norm stream after k blocks, k = 0..L. Read the docstring in `interp_bench.py` — this is the single most copied-wrong detail in logit-lens code.
2. **The borrowed basis.** Applying the *final* norm's learned scale to a *middle* layer's stream is a choice, not a neutral act. The lens at depth k shows what the final readout would decode there — not what layer k+1 actually consumes. The tuned-lens extension turns this caveat into data.

## Experiment

1. **Validate the prompt set.** Every target/distractor must be a single token for this run's tokenizer. Failures are dropped *with a count* and logged to `diagnostics/tokenization_report.csv`. A surprise here: a target you expected to be one token splitting into three.
2. **Verify the instrument.** The bench cross-checks per-block forward hooks against `output_hidden_states` (must match bit-for-bit) and checks that the lens at depth L reproduces the model's actual output logits (top-1 must match; the run aborts if not). Open both diagnostics once and understand what they prove.
3. **Capture and decode.** For each prompt: cache `streams[0..L]` at the final position, apply final norm + unembedding at every depth, and record top-k tokens, p(target), p(distractor), entropy (bits), cosine-to-final-stream, and residual norm. Saved per example under `state/`.
4. **Aggregate.** Decision depth = smallest k such that the final top-1 token is top-1 at every depth ≥ k. Computed for every example; compared across the three families. A surprise: ambiguous prompts with *early* decision depths.
5. **Read the counterfactuals.** For context-override prompts, target = the in-context answer, distractor = the memorized fact. Watch the logit difference rotate from memory toward context (or fail to) over depth.

## Run

```bash
# from labs/interpretability/ — smoke test first, always:
python interp_bench.py --lab lab1 --tier a

# full run on Colab A100/H100:
python interp_bench.py --lab lab1 --tier b --prompt-set full

# your own prompts:
python interp_bench.py --lab lab1 --prompt-set my_prompts.json
```

Custom prompt files are a JSON list of `{example_id, category, prompt, target, distractor}` (targets optional, written with their leading space).

## Artifacts to inspect, in order

1. `run_summary.md` — the headline table and what it does/doesn't support.
2. One `state/<example>/state_card.md` for a fact prompt — read the model "making up its mind" depth by depth.
3. `plots/entropy_by_depth.png` — facts vs ambiguous vs counterfactual.
4. `plots/biography_<example>.png` — the showcase trajectory.
5. `diagnostics/model_anatomy.md` and the two self-checks — the instrument's proof of correctness.
6. `results.csv` if you want to recompute anything yourself (you should, once).

## Questions

1. At which depth (as a fraction of L) does each category's prediction stabilize? Are facts earlier than ambiguous prompts, and by how much?
2. Does lower entropy always mean the answer is *correct*? Find a counterexample in your own run.
3. In the counterfactual family, describe the competition between the in-context answer and the memorized fact. Does the context win from depth 0, or does the trajectory rotate mid-stack?
4. The cosine-to-final curve rises long before the top-1 token stabilizes (or does it, in your run?). What does the gap between those two depths tell you about what the lens can't see?
5. *(Control)* If you ran the ambiguous family alone, could you have concluded anything about "when the model knows"? What exactly does that family rule out?
6. What experiment would you need to test whether an intermediate representation is causally *used* — and which lab in this course performs it?

## Ledger

Append 2–3 claims tagged `OBS` to `claim_ledger.md`. The run drafts candidates with your measured numbers in `ledger_suggestions.md`; edit them until you would defend them. At least one claim should be *at risk* from a later lab (the tuned-lens extension and Lab 10 both stress "the model knows early" claims).

## Interpretation & ethics

**Reading:** Dennett, "Real Patterns" (excerpt). Backup: Lipton, "The Mythos of Model Interpretability."

**Writing prompt:** Your lens shows the correct answer becoming top-1 at, say, depth 12 of 32. What *pattern* is real there, in Dennett's sense — and what claim of the form "the model knows the answer at layer 12" goes beyond the artifact you produced? Answer using your own `state_card.md`, not in the abstract.

## Common bugs (symptoms first)

- **Your lens disagrees with the model's prediction at depth L** → the run aborts by design; check `diagnostics/logit_lens_self_check.json`. Usually a wrong final-norm path (see anatomy report) or a model with logit post-processing.
- **A target silently never becomes top-1** → it tokenized to multiple tokens and was dropped; check `diagnostics/tokenization_report.csv` before blaming the model.
- **All depths look identical** → you are reading post-norm states from `hidden_states` without the re-mapping; use the bench's capture, or read its docstring and fix yours.
- **Entropy near log2(vocab) everywhere on GPU** → dtype problem; the lens upcasts to float32 before softmax for exactly this reason.
- **Comparing categories whose prompts differ wildly in length** and attributing the difference to "certainty" — length is a confound; note it in your writeup.

## Extensions

- **Manageable:** run the same prompt set on a base vs. instruct variant of the same model (`--model`) and compare stabilization depths. Artifact: `tables/base_vs_instruct.csv` (assemble from the two runs' `category_summary.csv`).
- **Ambitious:** implement a tuned lens (a learned affine probe per depth, trained to match final logits) for 4–6 depths and report where it disagrees with the raw lens. The disagreement *is* the lesson: "what layer 9 thinks" was partly an artifact of the readout basis. Artifact: `plots/tuned_vs_raw_lens.png`.

## Reading

- Elhage et al., "A Mathematical Framework for Transformer Circuits" — the residual-stream picture this lab instantiates.
- Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens" — the readout-bias correction, and the extension's blueprint.
- nostalgebraist, "interpreting GPT: the logit lens" — the original blog post, worth reading for how much it got right with how little machinery.
