# Lab 19 Final Validation Report

## Question

Can the model-diffing lab recover validated instruction-tuned features from an
Olmo base-vs-instruct pair?

## Final Read

No, not under the current course-accessible setup. The June 20, 2026 validation
does something useful: it repairs the identity control and then gives a clean
negative on the real model pair. The same-model Pythia identity test produces
only shared features, with no false model-specific share. That is a strong sign
that the audit scaffold is working.

On the Olmo base-vs-instruct pair, the candidate differences are mostly
template dominated or asymmetric rather than cleanly instruct-specific. The
causal marker control does not validate a feature. The result is a good teaching
case: model diffing produces plausible-looking candidates, and the controls
show why they should not be overclaimed.

## Dataset and Method Changes

- Repaired shared-side initialization for same-dimensional model pairs.
- Added matched shared-direction and independent-side random controls.
- Split prompts into train/dev/test by prompt group.
- Added `data/model_diffing_prompt_inventory_v2.csv`.
- Rendered a balanced raw/chat paired prompt inventory with 96 prompt groups
  and 192 runtime rows.
- Added feature audit and exclusivity plots for the final evidence pack.

## Run Matrix

| Run | Pair | Prompts | Eval FVU A | Eval FVU B | Taxonomy | Template dominated | Random B-specific rate | Audit status |
|---|---|---:|---:|---:|---|---:|---:|---|
| `identity_smoke` | Pythia-160M vs Pythia-160M | 12 | 0.6049 | 0.6049 | 48 shared | 0 | 0.0000 | Identity control passed |
| `olmo3_medium_edit` | Olmo base vs instruct | 80 | 0.3539 | 0.4085 | 105 asymmetric, 23 shared | 68 | 0.0039 | Template control dominates |
| `olmo3_full_edit` | Olmo base vs instruct | 192 | 0.5026 | 0.6113 | 95 asymmetric, 33 shared | 98 | 0.0039 | Template control dominates |
| `olmo3_full` | Olmo base vs instruct | 192 | 0.5026 | 0.6113 | 95 asymmetric, 33 shared | 98 | 0.0039 | Plotted evidence pack |

## Recommended Course Wording

```text
Lab 19 is a validated audit workflow with a clean negative result. It shows how
to test model-specific feature claims, and in the current setup those claims do
not survive matched controls.
```
