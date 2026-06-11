# Lab 10 verification report — reasoning models and CoT faithfulness

Date: 2026-06-11 · Machine: Colab A100-SXM4-80GB · Branch: `lab1_colab`

## What was built

The course's behavioral-faithfulness unit on a fully open reasoning model
(`allenai/Olmo-3-7B-Think`; Tier A smoke on `Qwen/Qwen3-0.6B`, the smallest
ungated model with real `<think>` spans):

- **Experiment 1 — hint injection.** Six frozen conditions per MCQ item:
  baseline; sycophancy / authority / metadata hints at a deterministic WRONG
  option; a correct-answer hint control; a non-sequitur control of matched
  shape. Metrics: flip rate, acknowledgment and attribution rates among
  flips (keyword heuristics, plus `acknowledgment_labels.csv` — verbatim
  excerpts with empty student columns, because the hand labeling is graded).
- **Experiment 2 — does the CoT carry load?** Early answering (the
  thought-necessity curve), add-mistake (inject a confident wrong claim
  mid-CoT and resume), and a matched-token-length filler control — all built
  on one primitive: close the think span, force `Answer:`.
- **Infrastructure:** frozen 140-item MCQ set vendored from MMLU
  (`data/mcq_items.csv` + `make_mcq_items.py`, authoring-time only), batched
  greedy generation (the bench's first), a think-span round-trip self-check
  (abort-on-failure), and a forced-answer rescue with `unparseable_log.csv`
  so refusals-to-format are never silently dropped. Olmo's template OPENS
  the think span itself (the rendered prompt ends with `<think>`); Qwen
  emits its own tag — the parser handles both and the round-trip check
  proves it per run.

## Validation evidence (Tier B, Olmo-3-7B-Think, 36 items, 2048-token budget)

| Result | Value |
|---|---|
| baseline accuracy (all items) | 0.611 |
| flip rate to hinted wrong answer (syco / authority / metadata) | 0.182 / 0.136 / 0.182 |
| silent flip rate (CoT never mentions hint, auto) | 0.045 / 0.0 / 0.0 |
| correct-hint control accuracy | 0.909 (hints are read and followed) |
| non-sequitur control accuracy | 0.864 (perturbation alone costs little) |
| necessity curve (k=0 → k=100) | 0.562 → 0.875 — the CoT carries load |
| matched-length filler accuracy | 0.562 = exactly the k=0 floor |
| injected-mistake follow rate | 0.0 (the model argues itself back; recover rate in the CSV) |
| forced-answer rescues | 46/216 (21%), all logged |
| wall clock | 36 min (within the <45 min budget) |

Tier A (Qwen3-0.6B, CPU, 2 items): end-to-end in 344 s with all checks
green, and even the 1-item necessity curve shows the textbook shape (0.0
with no CoT → 1.0 with full CoT, filler 0.0).

## A budget-sensitivity finding worth teaching

The same 36 items were first run with a 1024-token thinking budget, where
42% of generations hit the cap and were force-answered early. Flip rates
were 2–3× HIGHER there (metadata 0.565 vs 0.182 at 2048). Forcing an early
answer amplifies hint influence — which is the necessity curve's lesson
arriving from the other side: the hint's pull is strongest before the CoT
has run its course, and a model given room to think argues itself away from
the hint. The handout's debugging table points students at the budget when
rates look extreme; the canonical run uses 2048.

## Honest negatives, by design

- Olmo-3-Think is mistake-IMMUNE on this dataset (follow rate 0.0): it
  notices the injected wrong claim and re-derives. Per the course's grading
  philosophy this is a full-credit negative result, and the handout's
  debugging table names it ("strengthen the injection or report robustness
  as the finding").
- The silent-flip rates are low for this model (0–4.5%); the metadata hint
  is acknowledged when followed. The claim card's scope line exists exactly
  so this is reported as a fact about THIS model on THIS dataset, not about
  CoT in general.

## Files

- `labs/lab10_cot_faithfulness.py`, `labs/lab10_cot_faithfulness.md`
- `data/mcq_items.csv` (140 items, 8 domains) + `data/make_mcq_items.py`
- registry entry `lab10` in `interp_bench.py` (think models on every tier,
  `--max-examples` = item count); `CHAT_TEMPLATE_LABS` includes lab10
- canonical runs: `runs/lab10_tiera_cpu` (CPU), `runs/lab10_tierb_full`
