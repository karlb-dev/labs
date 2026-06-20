# Lab 07 Validation

## Lab 7: Steering Vectors, Representation Engineering, and the Refusal Direction

Steering vectors and the refusal direction: control, monitoring, and dual use.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/run6/C/lab7` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `baseline_refusal_rate_benign`=0.2083, `best_injection_layer`=20, `bridge_answer_bias_span_logits`=14.11, `bridge_eval_split`=held-out truth pairs from truth_cities.csv, `bridge_random_answer_bias_span_logits`=2.875, `bridge_signed_truth_margin_span_logits`=4.688, `bridge_verdict`=decodable-and-steers-True-assent, `direction_stream_depth`=21
  - model: `allenai/Olmo-3-7B-Instruct` (instruct, chat template applied to every prompt)
  - injection site: decoder block 20 output, stream depth 21
  - reference activation norm for dose scaling: 27.602
- `interpret/run6/B/lab7` (allenai/Olmo-3-7B-Instruct, tier b)
  - Metrics: `baseline_refusal_rate_benign`=0.2083, `best_injection_layer`=20, `bridge_answer_bias_span_logits`=14.11, `bridge_eval_split`=held-out truth pairs from truth_cities.csv, `bridge_random_answer_bias_span_logits`=2.875, `bridge_signed_truth_margin_span_logits`=4.688, `bridge_verdict`=decodable-and-steers-True-assent, `direction_stream_depth`=21
  - model: `allenai/Olmo-3-7B-Instruct` (instruct, chat template applied to every prompt)
  - injection site: decoder block 20 output, stream depth 21
  - reference activation norm for dose scaling: 27.602
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab07_tierc_labs1_25_full_matrix_20260615_000508/lab07_tierc_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `baseline_refusal_rate_benign`=0.1667, `best_injection_layer`=20, `bridge_answer_bias_span_logits`=14.06, `bridge_eval_split`=held-out truth pairs from truth_cities.csv, `bridge_random_answer_bias_span_logits`=2.85, `bridge_signed_truth_margin_span_logits`=4.678, `bridge_verdict`=decodable-and-steers-True-assent, `direction_stream_depth`=21
  - model: `allenai/Olmo-3-7B-Instruct` (instruct, chat template applied to every prompt)
  - injection site: decoder block 20 output, stream depth 21
  - reference activation norm for dose scaling: 27.610
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab07_olmo32bthink_labs1_25_full_matrix_20260615_000508/lab07_olmo32bthink_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `baseline_refusal_rate_benign`=0.0417, `best_injection_layer`=22, `bridge_answer_bias_span_logits`=8.391, `bridge_eval_split`=held-out truth pairs from truth_cities.csv, `bridge_random_answer_bias_span_logits`=1.566, `bridge_signed_truth_margin_span_logits`=2.697, `bridge_verdict`=decodable-and-steers-True-assent, `direction_stream_depth`=23
  - model: `allenai/Olmo-3-32B-Think` (instruct, chat template applied to every prompt)
  - injection site: decoder block 22 output, stream depth 23
  - reference activation norm for dose scaling: 19.787

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/run6/C/lab7` | `allenai/Olmo-3-7B-Instruct` | `c` | `baseline_refusal_rate_benign`=0.2083; `best_injection_layer`=20; `bridge_answer_bias_span_logits`=14.11 |
| `interpret/run6/B/lab7` | `allenai/Olmo-3-7B-Instruct` | `b` | `baseline_refusal_rate_benign`=0.2083; `best_injection_layer`=20; `bridge_answer_bias_span_logits`=14.11 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab07_tierc_labs1_25_full_matrix_20260615_000508/lab07_tierc_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `baseline_refusal_rate_benign`=0.1667; `best_injection_layer`=20; `bridge_answer_bias_span_logits`=14.06 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab07_olmo32bthink_labs1_25_full_matrix_20260615_000508/lab07_olmo32bthink_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-32B-Think` | `c` | `baseline_refusal_rate_benign`=0.0417; `best_injection_layer`=22; `bridge_answer_bias_span_logits`=8.391 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab07_gemma4e4b_labs1_25_full_matrix_20260615_000508/lab07_gemma4e4b_labs1_25_full_matrix_20260615_000508` | `google/gemma-4-E4B-it` | `b` | `baseline_refusal_rate_benign`=0.5; `best_injection_layer`=18; `bridge_answer_bias_span_logits`=5.054 |

## Curated Artifacts

- `olmo3_7b_run6c_truth_bridge_statement_atlas.png`
- `olmo3_7b_run6c_steering_evidence_dashboard.png`
- `olmo3_7b_run6c_results.csv`
- `olmo3_7b_run6c_metrics.json`
- `olmo3_7b_run6b_truth_bridge_statement_atlas.png`
- `olmo3_7b_run6b_steering_evidence_dashboard.png`
- `olmo3_7b_run6b_results.csv`
- `olmo3_7b_run6b_metrics.json`
- `olmo3_7b_lab07_tierc_labs1_25_full_matrix_20260615_000508_truth_bridge_statement_atlas.png`
- `olmo3_7b_lab07_tierc_labs1_25_full_matrix_20260615_000508_steering_evidence_dashboard.png`
- `olmo3_7b_lab07_tierc_labs1_25_full_matrix_20260615_000508_results.csv`
- `olmo3_7b_lab07_tierc_labs1_25_full_matrix_20260615_000508_metrics.json`
- `olmo3_32b_lab07_olmo32bthink_labs1_25_full_matrix_20260615_truth_bridge_statement_atlas.png`
- `olmo3_32b_lab07_olmo32bthink_labs1_25_full_matrix_20260615_steering_evidence_dashboard.png`
- `olmo3_32b_lab07_olmo32bthink_labs1_25_full_matrix_20260615_results.csv`
- `olmo3_32b_lab07_olmo32bthink_labs1_25_full_matrix_20260615_metrics.json`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
