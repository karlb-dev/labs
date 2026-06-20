# Lab 14 Final Validation Report

## Question

Does Lab 14 now validate a usable certainty/calibration instrument, or is it
still mostly a weak answerability/style probe?

## Final Read

Lab 14 is now a strong partial positive. The June 20, 2026 validation uses
80 fixed-choice items across five families and evaluates Olmo-3-7B-Instruct
across three seeds plus a SmolLM comparison. The answerability direction is
stable enough to use as a downstream instrument when its caveats are carried
forward.

The scientific claim should stay precise. The direction predicts whether the
model is in an answerable vs unanswerable frame under controls. It is not a
direct measure of subjective confidence, knowledge, belief, honesty, or
phenomenal certainty. The lab is strongest when taught as an operationalization
lesson: students get a useful internal signal, then audit all the cheap ways
that signal could be mistaken for something deeper.

## Dataset and Method Changes

- Expanded the validation set from the older 12/36-item runs to 80 items.
- Balanced five families: factual QA, freeform answerability, MCQ, passage QA,
  and procedural logic.
- Selected certainty and hedging depths by a train-split control-adjusted rule.
- Added family-held-out generalization checks.
- Added entropy/distribution confidence, verbal confidence, hedging/style,
  length, answer-letter, and option-text baselines.
- Added disagreement examples to force careful SELF-REPORT interpretation.
- Added a plotted evidence pack for the primary Olmo seed.

## Run Matrix

| Run | Model | Seed | Items | Best depth | Eval AUC | Random AUC | Shuffled AUC | Control gap | Family AUC | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `olmo3_7b_full_s0` | Olmo-3-7B-Instruct | 0 | 80 | 11 | 0.8889 | 0.5191 | 0.5147 | 0.3698 | 0.8282 | Usable instrument |
| `olmo3_7b_full_s1` | Olmo-3-7B-Instruct | 1 | 80 | 32 | 0.9556 | 0.5262 | 0.5262 | 0.4294 | 0.9781 | Decodes, but confounds compete |
| `olmo3_7b_full_s2` | Olmo-3-7B-Instruct | 2 | 80 | 13 | 0.9200 | 0.5449 | 0.4640 | 0.3751 | 0.8781 | Usable instrument |
| `smollm_full_s0` | SmolLM2-135M-Instruct | 0 | 80 | 10 | 0.9200 | 0.5964 | 0.4942 | 0.3236 | 0.8563 | Usable instrument |

## Confound Read

The primary Olmo seed is the cleanest teaching example. Its internal projection
beats random and shuffled controls, family-heldout checks remain positive, and
distribution confidence alone is weak for answerability. Verbal confidence is
highly predictive but poorly calibrated, and the disagreement table shows why
self-report claims need restraint.

Seed 1 is the cautionary case. It has the highest answerability AUC, but the
hedging-style projection is also highly predictive. That is why the report
should not say every seed cleanly isolates certainty. The more accurate claim
is that the lab validates an answerability instrument while explicitly showing
where style controls pressure the interpretation.

## Recommended Course Wording

```text
Lab 14 validates a controlled answerability direction that can be reused as a
downstream instrument. The direction is uncertainty-adjacent, but the lab does
not prove subjective confidence or knowledge; it teaches how operational
confidence signals survive, or fail, under controls.
```
