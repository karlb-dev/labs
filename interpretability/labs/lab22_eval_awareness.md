# Lab 22 - Eval Awareness

## Question

Does the model represent that it is being evaluated, and does that representation change behavior?

This lab builds a cautious version of that question. It contrasts benchmark-style prompts with natural prompts that ask for the same benign task, then attacks the result with format-matched controls. The target is not "the model knows it is in an eval" in a strong human sense. The target is a measured internal direction that transfers across eval formats better than surface-format controls.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab22 --tier a --no-plots
python interp_bench.py --lab lab22 --tier b --prompt-set full
python interp_bench.py --lab lab22 --tier b --model allenai/Olmo-3-7B-Think --prompt-set full
```

The Think-model run is useful for `tables/cot_mentions_eval.csv`, but the lab does not require explicit thinking spans.

## What It Does

The lab:

- loads `data/eval_awareness_contexts.csv`;
- renders every prompt through the model's chat template;
- captures final-token residual streams at every depth;
- fits an eval-vs-natural direction on the train split;
- selects depth using train-only control-adjusted AUC;
- tests held-out rows and leave-one-format-out transfer;
- steers natural prompts toward eval context and eval prompts away from it;
- writes hand-label columns for behavior review;
- optionally discovers a Lab 20 eval-aware toy organism if `--organism` or `LAB22_ORGANISM_DIR` is set.

## Main Artifacts

| Path | What it contains |
|---|---|
| `tables/eval_awareness_contexts.csv` | selected prompt inventory |
| `tables/eval_awareness_probe_by_layer.csv` | train/eval AUC and format-control AUC by depth |
| `tables/cross_format_transfer.csv` | leave-one-format-out transfer |
| `tables/steering_generations.csv` | raw steering generations with hand-label columns |
| `tables/steering_behavior_shift.csv` | marker summary for steering conditions |
| `tables/cot_mentions_eval.csv` | generated or explicit thinking mentions of test/eval framing |
| `tables/lab20_eval_aware_control.csv` | optional Lab 20 known-positive organism discovery |
| `state/eval_awareness_direction.pt` | saved direction and metadata |
| `plots/eval_awareness_probe_by_layer.png` | layer sweep |
| `operationalization_audit.md` | cheap explanations and allowed claims |

## Evidence Discipline

Do not write:

- "The model knows it is being evaluated."
- "The direction is situational awareness."
- "Steering caused eval gaming" without hand labels and random controls.

Allowed claims are narrower:

- the direction separates eval-framed prompts from natural prompts;
- cross-format transfer survived or failed format controls;
- steering shifted measured benign behavior under the lab's marker and hand-label protocol.

## Cheap Explanations

Start by trying to kill the story:

- multiple-choice detector;
- `Question:` / `Answer:` boilerplate detector;
- difficulty or formality detector;
- answer-format detector;
- steering changed verbosity or hedging rather than evaluation context.

## Writeup Questions

1. Which depth was selected, and was selection based only on train rows?
2. Does eval-vs-natural AUC stay high on held-out rows?
3. Which held-out format has the weakest transfer?
4. Do format controls project like true eval prompts?
5. Did steering natural prompts increase eval/test markers more than random steering?
6. Did steering eval prompts down reduce eval/test markers without damaging answer quality?
7. If using a Think model, do thinking spans mention evaluation when the direction is high?
