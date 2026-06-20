# Lab 18 Final Validation Report

## Question

Does the lab find a meaningful humor feature, or is it mostly measuring
surface joke style?

## Final Read

Lab 18 is useful, but its claim should stay narrow. The June 20, 2026
validation finds a controlled joke-structure handle on Olmo-3-7B-Instruct seed
0. That run reaches 1.0000 held-out test AUC with a 0.2172 gap over the best
null control. The second Olmo seed and the SmolLM comparison are weaker, so the
best interpretation is not "the lab found humor" in a broad sense. It found a
decodable joke/register contrast that is partially separated from simple
surprise, silliness, positivity, and shuffled controls.

The intervention result is negative. Steering does not produce a clean
specificity gain, so the lab should teach students to distinguish labelable
directions from causal behavioral handles.

## Dataset and Method Changes

- Expanded the humor corpus from 20 rows to 80 rows.
- Added `data/humor_incongruity_pairs.csv` and a dataset card.
- Balanced examples across joke, literal, surprise, silly, and positive
  contrasts.
- Added train/dev/test evaluation and family-level generalization summaries.
- Added dashboard and matrix artifacts for evidence review.

## Run Matrix

| Run | Model | Seed | Rows | Best depth | Test AUC | Best-null gap | Family gap | Steering gap | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `smollm_full_s0` | SmolLM2-135M-Instruct | 0 | 80 | 5 | 1.0000 | 0.0824 | 0.0230 | -0.8000 | Weak or confounded |
| `olmo3_7b_full_s0` | Olmo-3-7B-Instruct | 0 | 80 | 2 | 1.0000 | 0.2172 | 0.0643 | -0.3000 | Decodable, not causally separated |
| `olmo3_7b_full_s1` | Olmo-3-7B-Instruct | 1 | 80 | 6 | 0.9272 | 0.0722 | 0.0566 | -0.6000 | Weak or confounded |

## Recommended Course Wording

```text
Lab 18 demonstrates how to validate a joke-structure direction and how quickly
that claim narrows under controls. Treat the result as partial evidence for a
humor-adjacent register feature, not as a robust causal humor circuit.
```
