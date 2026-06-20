# Lab 19 Fair-Shot Report

Date: 2026-06-20

## What changed

- Added shared side initialization for same-dimensional model pairs in the paired crosscoder.
- Replaced the old single random baseline with explicit `matched_shared_direction` and `independent_side_directions` controls.
- Changed prompt splitting from train/eval to train/dev/test by `prompt_group`; test is the report split.
- Added `data/model_diffing_prompt_inventory_v2.csv`: 96 deterministic raw prompt groups, each rendered with a paired chat variant at runtime.
- Updated Lab 19 docs and validation notes for the repaired protocol.

## Commands run

```bash
python -m py_compile interpretability/interp_bench.py interpretability/labs/lab19_model_diffing_crosscoders.py
python interpretability/data/make_model_diffing_prompts.py
python -m py_compile interpretability/labs/lab19_model_diffing_crosscoders.py interpretability/data/make_model_diffing_prompts.py
python interp_bench.py --lab lab19 --tier a --prompt-set small --run-name lab19_fairshot_tiera_identity_v2_plots_20260620
LAB19_OFFLOAD_PRIMARY_TO_CPU=1 python interp_bench.py --lab lab19 --tier b --prompt-set medium --no-plots --run-edit --run-name lab19_fairshot_olmo3_medium_edit_v2_20260620
LAB19_OFFLOAD_PRIMARY_TO_CPU=1 python interp_bench.py --lab lab19 --tier b --prompt-set data/model_diffing_prompt_inventory_v2.csv --no-plots --run-edit --run-name lab19_fairshot_olmo3_v2_full_edit_20260620
LAB19_OFFLOAD_PRIMARY_TO_CPU=1 python interp_bench.py --lab lab19 --tier b --prompt-set data/model_diffing_prompt_inventory_v2.csv --run-name lab19_fairshot_olmo3_v2_full_plots_20260620
```

Backups:

```text
/content/drive/MyDrive/interpret/lab19_model_diffing_fairshot_20260620/
```

## Run results

| Run | Prompt inventory | Result |
|---|---|---|
| `lab19_fairshot_tiera_identity_v2_plots_20260620` | course small, 12 rows | Identity smoke passes: 48/48 shared features, false-specific share 0.0, matched-random B-specific rate 0.0. |
| `lab19_fairshot_olmo3_medium_edit_v2_20260620` | course medium, 80 rows | Reconstruction passes; no candidate instruct-only handles; template control dominates; optional edit not validated by marker control. |
| `lab19_fairshot_olmo3_v2_full_edit_20260620` | v2 balanced, 192 runtime rows | Same negative: no candidate instruct-only handles; 98 template-dominated features; matched-random B-specific rate 0.0039; causal specificity 0.0. |
| `lab19_fairshot_olmo3_v2_full_plots_20260620` | v2 balanced, 192 runtime rows | Plotted evidence pack for dashboard and audit matrices. |

## Scientific read

This is a cleaner negative for the attractive claim that a small paired crosscoder over final-token residuals recovers robust OLMo instruct-only/default-assistant feature handles under matched controls.

The repaired identity control is a positive instrumentation result. The model-pair runs are not positive: they produce mostly shared/asymmetric coordinates, strong raw-vs-chat template effects, large norm-shift warnings on the v2 inventory, and no causal marker specificity over random controls.

Allowed claim: Lab 19 now provides a defensible crosscoder model-diffing audit scaffold and a clean negative under this small course-accessible setup.

Not allowed: a claim that any recovered feature is instruction following, alignment, assistant voice, or persona.
