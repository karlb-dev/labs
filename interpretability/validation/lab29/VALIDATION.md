# Lab 29 Validation

## Lab 29 - Training Dynamics and Circuit Birth

Training dynamics and circuit birth: controlled tiny-checkpoint time-lapse with behavior, probes, motifs, and intervention-transfer controls.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/lab29_tierc_full_verify_20260614_2322/lab29_tierc_full_verify_20260614_2322` (gpt2, tier c)
  - Metrics: `behavior_emergence_step`=17, `control_leakage_step`=, `decodability_emergence_step`=17, `evidence_components`=5, `final_heldout_or_test_accuracy`=1, `final_induction_accuracy`=1, `final_phase`=circuit_present_under_proxy, `final_probe_acc`=1
  - data rows: 14 selected from `training_dynamics_tasks.csv`
  - data source: `frozen_csv`
  - science_ready: `true`
- `interpret/verify_part3/lab29_tierb_full_verify_20260614_2320/lab29_tierb_full_verify_20260614_2320` (gpt2, tier b)
  - Metrics: `behavior_emergence_step`=17, `control_leakage_step`=, `decodability_emergence_step`=17, `evidence_components`=5, `final_heldout_or_test_accuracy`=1, `final_induction_accuracy`=1, `final_phase`=circuit_present_under_proxy, `final_probe_acc`=1
  - data rows: 14 selected from `training_dynamics_tasks.csv`
  - data source: `frozen_csv`
  - science_ready: `true`
- `interpret/verify_part3/lab29_tiera_full_verify_20260614_2317/lab29_tiera_full_verify_20260614_2317` (gpt2, tier a)
  - Metrics: `behavior_emergence_step`=20, `control_leakage_step`=, `decodability_emergence_step`=20, `evidence_components`=5, `final_heldout_or_test_accuracy`=1, `final_induction_accuracy`=1, `final_phase`=circuit_present_under_proxy, `final_probe_acc`=1
  - data rows: 11 selected from `training_dynamics_tasks.csv`
  - data source: `frozen_csv`
  - science_ready: `true`

## What This Lab Teaches

- The central lesson is decodability with controls: useful probes must survive selectivity, held-out data, and confound checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/lab29_tierc_full_verify_20260614_2322/lab29_tierc_full_verify_20260614_2322` | `gpt2` | `c` | `behavior_emergence_step`=17; `control_leakage_step`=; `decodability_emergence_step`=17 |
| `interpret/verify_part3/lab29_tierb_full_verify_20260614_2320/lab29_tierb_full_verify_20260614_2320` | `gpt2` | `b` | `behavior_emergence_step`=17; `control_leakage_step`=; `decodability_emergence_step`=17 |
| `interpret/verify_part3/lab29_tiera_full_verify_20260614_2317/lab29_tiera_full_verify_20260614_2317` | `gpt2` | `a` | `behavior_emergence_step`=20; `control_leakage_step`=; `decodability_emergence_step`=20 |

## Curated Artifacts

- `gpt2_lab29_tierc_full_verify_20260614_2322_training_dynamics_dashboard.png`
- `gpt2_lab29_tierc_full_verify_20260614_2322_circuit_birth_atlas.png`
- `gpt2_lab29_tierc_full_verify_20260614_2322_results.csv`
- `gpt2_lab29_tierc_full_verify_20260614_2322_metrics.json`
- `gpt2_lab29_tierb_full_verify_20260614_2320_training_dynamics_dashboard.png`
- `gpt2_lab29_tierb_full_verify_20260614_2320_circuit_birth_atlas.png`
- `gpt2_lab29_tierb_full_verify_20260614_2320_results.csv`
- `gpt2_lab29_tierb_full_verify_20260614_2320_metrics.json`
- `gpt2_lab29_tiera_full_verify_20260614_2317_training_dynamics_dashboard.png`
- `gpt2_lab29_tiera_full_verify_20260614_2317_circuit_birth_atlas.png`
- `gpt2_lab29_tiera_full_verify_20260614_2317_results.csv`
- `gpt2_lab29_tiera_full_verify_20260614_2317_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
