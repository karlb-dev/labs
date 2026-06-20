# Lab 23 Validation

## Lab 23 - Blind Audit of a Benign Hidden-Behavior Organism

Blind audit: preregister, submit claims, unseal, and score benign hidden-behavior organisms.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab23_olmo32bthink_blind_labs1_25_local_reruns_20260615_101609/lab23_olmo32bthink_blind_labs1_25_local_reruns_20260615_101609` (allenai/Olmo-3-32B-Think, tier c)
  - Metrics: `commitment_verdict`=verified_or_not_applicable, `false_positive_claims`=0, `manual_review_claims`=0, `mean_precision`=, `mean_recall`=, `missing_evidence_paths`=0, `n_answer_keys_available`=0, `n_scored_subjects`=0
  - Subject source: `cli_or_env`
  - Requested path: `/content/labs/interpretability/runs/lab20_olmo32bthink_labs1_25_local_reruns_20260615_101609`
  - Blind mode: `True`
- `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab23_gemma4e4b_blind_labs1_25_local_reruns_20260615_101609/lab23_gemma4e4b_blind_labs1_25_local_reruns_20260615_101609` (google/gemma-4-E4B-it, tier b)
  - Metrics: `commitment_verdict`=verified_or_not_applicable, `false_positive_claims`=0, `manual_review_claims`=0, `mean_precision`=, `mean_recall`=, `missing_evidence_paths`=0, `n_answer_keys_available`=0, `n_scored_subjects`=0
  - Subject source: `cli_or_env`
  - Requested path: `/content/labs/interpretability/runs/lab20_gemma4e4b_labs1_25_local_reruns_20260615_101609`
  - Blind mode: `True`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab23_tierc_blind_labs1_25_full_matrix_20260615_000508/lab23_tierc_blind_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Instruct, tier c)
  - Metrics: `commitment_verdict`=verified_or_not_applicable, `false_positive_claims`=0, `manual_review_claims`=0, `mean_precision`=, `mean_recall`=, `missing_evidence_paths`=0, `n_answer_keys_available`=0, `n_scored_subjects`=0
  - Subject source: `cli_or_env`
  - Requested path: `/content/labs/interpretability/runs/lab20_tierc_labs1_25_full_matrix_20260615_000508`
  - Blind mode: `True`
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab23_tiera_blind_labs1_25_full_matrix_20260615_000508/lab23_tiera_blind_labs1_25_full_matrix_20260615_000508` (HuggingFaceTB/SmolLM2-135M-Instruct, tier a)
  - Metrics: `commitment_verdict`=verified_or_not_applicable, `false_positive_claims`=0, `manual_review_claims`=0, `mean_precision`=, `mean_recall`=, `missing_evidence_paths`=0, `n_answer_keys_available`=0, `n_scored_subjects`=0
  - Subject source: `cli_or_env`
  - Requested path: `/content/labs/interpretability/runs/lab20_tiera_labs1_25_full_matrix_20260615_000508`
  - Blind mode: `True`

## What This Lab Teaches

- The central lesson is claim discipline: the run artifacts support bounded conclusions and surface failure modes explicitly.
- Compare the selected models rather than cherry-picking the best one; model differences are often the point of the exercise.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab23_olmo32bthink_blind_labs1_25_local_reruns_20260615_101609/lab23_olmo32bthink_blind_labs1_25_local_reruns_20260615_101609` | `allenai/Olmo-3-32B-Think` | `c` | `commitment_verdict`=verified_or_not_applicable; `false_positive_claims`=0; `manual_review_claims`=0 |
| `interpret/verify_part3/labs1_25_local_reruns_20260615_101609/lab23_gemma4e4b_blind_labs1_25_local_reruns_20260615_101609/lab23_gemma4e4b_blind_labs1_25_local_reruns_20260615_101609` | `google/gemma-4-E4B-it` | `b` | `commitment_verdict`=verified_or_not_applicable; `false_positive_claims`=0; `manual_review_claims`=0 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab23_tierc_blind_labs1_25_full_matrix_20260615_000508/lab23_tierc_blind_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Instruct` | `c` | `commitment_verdict`=verified_or_not_applicable; `false_positive_claims`=0; `manual_review_claims`=0 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab23_tiera_blind_labs1_25_full_matrix_20260615_000508/lab23_tiera_blind_labs1_25_full_matrix_20260615_000508` | `HuggingFaceTB/SmolLM2-135M-Instruct` | `a` | `commitment_verdict`=verified_or_not_applicable; `false_positive_claims`=0; `manual_review_claims`=0 |

## Curated Artifacts

- `olmo3_32b_lab23_olmo32bthink_blind_labs1_25_local_reruns_2_audit_evidence_dashboard.png`
- `olmo3_32b_lab23_olmo32bthink_blind_labs1_25_local_reruns_2_claim_readiness_matrix.png`
- `olmo3_32b_lab23_olmo32bthink_blind_labs1_25_local_reruns_2_tables_audit_mode_value_matrix.csv`
- `olmo3_32b_lab23_olmo32bthink_blind_labs1_25_local_reruns_2_tables_audit_evidence_matrix.csv`
- `gemma4e4b_lab23_gemma4e4b_blind_labs1_25_local_reruns_2026_audit_evidence_dashboard.png`
- `gemma4e4b_lab23_gemma4e4b_blind_labs1_25_local_reruns_2026_claim_readiness_matrix.png`
- `gemma4e4b_lab23_gemma4e4b_blind_labs1_25_local_reruns_2026_tables_audit_mode_value_matrix.csv`
- `gemma4e4b_lab23_gemma4e4b_blind_labs1_25_local_reruns_2026_tables_audit_evidence_matrix.csv`
- `olmo3_7b_lab23_tierc_blind_labs1_25_full_matrix_20260615_audit_evidence_dashboard.png`
- `olmo3_7b_lab23_tierc_blind_labs1_25_full_matrix_20260615_internals_value_frontier.png`
- `olmo3_7b_lab23_tierc_blind_labs1_25_full_matrix_20260615_tables_audit_mode_score.csv`
- `olmo3_7b_lab23_tierc_blind_labs1_25_full_matrix_20260615_tables_scored_claims.csv`
- `smollm_lab23_tiera_blind_labs1_25_full_matrix_20260615_audit_evidence_dashboard.png`
- `smollm_lab23_tiera_blind_labs1_25_full_matrix_20260615_internals_value_frontier.png`
- `smollm_lab23_tiera_blind_labs1_25_full_matrix_20260615_tables_audit_mode_score.csv`
- `smollm_lab23_tiera_blind_labs1_25_full_matrix_20260615_tables_scored_claims.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
