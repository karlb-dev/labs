# Lab 17 Final Validation Report

## Question

Does the lab find persona/register circuits clearly enough to support the
student exercise?

## Final Read

Yes for controlled decodability; not yet for causal steering. The June 20,
2026 validation upgrades the lab from a tiny pilot into a credible skeptical
exercise. On Olmo-3-7B-Instruct, the trait probe reaches 1.0000 held-out test
AUC across two full-corpus seeds while matched random controls remain far
lower. That is a clean signal that the lab can find persona/register directions
under a held-out evaluation.

The intervention step is weaker. The five-control reruns do not show a robust
positive steering specificity gap, and the SmolLM run fails content
preservation. The course claim should therefore stay narrow: the lab teaches
how to discover and validate trait directions, then shows why steering claims
need extra controls.

## Dataset and Method Changes

- Expanded the persona/register corpus from 24 rows to 256 rows.
- Added `data/persona_register_pairs.csv` and a dataset card.
- Split examples by trait/topic rather than evaluating only near-duplicates.
- Selected depth using dev-set control-adjusted evidence.
- Added bootstrap/permutation style validation summaries.
- Added multiple random directions for the final Olmo reruns.
- Added dashboard labels and control summaries for easier review.

## Run Matrix

| Run | Model | Seed | Rows | Best depth | Test AUC | Random AUC | Selectivity | Steering gap | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `smollm_full_s0` | SmolLM2-135M-Instruct | 0 | 256 | 29 | 0.9847 | 0.5888 | 0.3959 | 1.4688 | Not validated by controls |
| `olmo3_7b_full_s0` | Olmo-3-7B-Instruct | 0 | 256 | 5 | 1.0000 | 0.6439 | 0.3561 | 0.0000 | Decodable, not steerable |
| `olmo3_7b_full_s0_controls5` | Olmo-3-7B-Instruct | 0 | 256 | 5 | 1.0000 | 0.6439 | 0.3561 | -0.1062 | Decodable, not steerable |
| `olmo3_7b_full_s1_controls5` | Olmo-3-7B-Instruct | 1 | 256 | 10 | 1.0000 | 0.6577 | 0.3423 | -0.1875 | Decodable, not steerable |

## Recommended Course Wording

```text
Lab 17 finds robust controlled evidence for persona/register directions in
Olmo-3-7B-Instruct. It does not yet show a robust causal steering result, so
the lab should present steering as a stress test rather than a headline claim.
```
