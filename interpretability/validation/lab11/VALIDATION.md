# Lab 11 Validation

## Lab 11: Mechanistic Reliability Audit

Capstone: a mechanistic reliability audit with a fixed report schema, built on the claim ledger.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab11_tierc_sentiment_negation_labs1_25_full_matrix_20260615_000508/lab11_tierc_sentiment_negation_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `domain`=sentiment_negation, `n_ledger_entries`=1
  - Fill `failure_mode_student` in `results.csv` before reading aggregate tables too closely.
  - Fill `audit_report.md` after the counterevidence section, not before.
  - Reconcile every ledger claim in `ledger_reconciliation.md`; at least one revision or retirement is required.
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab11_tierc_factual_qa_labs1_25_full_matrix_20260615_000508/lab11_tierc_factual_qa_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-1125-32B, tier c)
  - Metrics: `domain`=factual_qa, `n_ledger_entries`=1
  - Fill `failure_mode_student` in `results.csv` before reading aggregate tables too closely.
  - Fill `audit_report.md` after the counterevidence section, not before.
  - Reconcile every ledger claim in `ledger_reconciliation.md`; at least one revision or retirement is required.
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab11_tierc_cot_faithfulness_labs1_25_full_matrix_20260615_000508/lab11_tierc_cot_faithfulness_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-7B-Think, tier c)
  - Metrics: `domain`=cot_faithfulness, `n_ledger_entries`=1
  - Fill `failure_mode_student` in `results.csv` before reading aggregate tables too closely.
  - Fill `audit_report.md` after the counterevidence section, not before.
  - Reconcile every ledger claim in `ledger_reconciliation.md`; at least one revision or retirement is required.
- `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab11_tierb_sentiment_negation_labs1_25_full_matrix_20260615_000508/lab11_tierb_sentiment_negation_labs1_25_full_matrix_20260615_000508` (allenai/Olmo-3-1025-7B, tier b)
  - Metrics: `domain`=sentiment_negation, `n_ledger_entries`=1
  - Fill `failure_mode_student` in `results.csv` before reading aggregate tables too closely.
  - Fill `audit_report.md` after the counterevidence section, not before.
  - Reconcile every ledger claim in `ledger_reconciliation.md`; at least one revision or retirement is required.

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab11_tierc_sentiment_negation_labs1_25_full_matrix_20260615_000508/lab11_tierc_sentiment_negation_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-1125-32B` | `c` | `domain`=sentiment_negation; `n_ledger_entries`=1 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab11_tierc_factual_qa_labs1_25_full_matrix_20260615_000508/lab11_tierc_factual_qa_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-1125-32B` | `c` | `domain`=factual_qa; `n_ledger_entries`=1 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab11_tierc_cot_faithfulness_labs1_25_full_matrix_20260615_000508/lab11_tierc_cot_faithfulness_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-7B-Think` | `c` | `domain`=cot_faithfulness; `n_ledger_entries`=1 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab11_tierb_sentiment_negation_labs1_25_full_matrix_20260615_000508/lab11_tierb_sentiment_negation_labs1_25_full_matrix_20260615_000508` | `allenai/Olmo-3-1025-7B` | `b` | `domain`=sentiment_negation; `n_ledger_entries`=1 |
| `interpret/verify_part3/labs1_25_full_matrix_20260615_000508/lab11_gemma4e4b_sentiment_negation_labs1_25_full_matrix_20260615_000508/lab11_gemma4e4b_sentiment_negation_labs1_25_full_matrix_20260615_000508` | `google/gemma-4-E4B-it` | `b` | `domain`=sentiment_negation; `n_ledger_entries`=1 |

## Curated Artifacts

- `olmo3_32b_lab11_tierc_sentiment_negation_labs1_25_full_mat_audit_dashboard.png`
- `olmo3_32b_lab11_tierc_sentiment_negation_labs1_25_full_mat_audit_scorecard.png`
- `olmo3_32b_lab11_tierc_sentiment_negation_labs1_25_full_mat_tables_audit_scorecard.csv`
- `olmo3_32b_lab11_tierc_sentiment_negation_labs1_25_full_mat_results.csv`
- `olmo3_32b_lab11_tierc_factual_qa_labs1_25_full_matrix_2026_audit_dashboard.png`
- `olmo3_32b_lab11_tierc_factual_qa_labs1_25_full_matrix_2026_audit_scorecard.png`
- `olmo3_32b_lab11_tierc_factual_qa_labs1_25_full_matrix_2026_tables_audit_scorecard.csv`
- `olmo3_32b_lab11_tierc_factual_qa_labs1_25_full_matrix_2026_results.csv`
- `olmo3_7b_lab11_tierc_cot_faithfulness_labs1_25_full_matri_audit_dashboard.png`
- `olmo3_7b_lab11_tierc_cot_faithfulness_labs1_25_full_matri_audit_scorecard.png`
- `olmo3_7b_lab11_tierc_cot_faithfulness_labs1_25_full_matri_tables_audit_scorecard.csv`
- `olmo3_7b_lab11_tierc_cot_faithfulness_labs1_25_full_matri_results.csv`
- `olmo3_1025_7b_lab11_tierb_sentiment_negation_labs1_25_full_m_audit_dashboard.png`
- `olmo3_1025_7b_lab11_tierb_sentiment_negation_labs1_25_full_m_audit_scorecard.png`
- `olmo3_1025_7b_lab11_tierb_sentiment_negation_labs1_25_full_m_tables_audit_scorecard.csv`
- `olmo3_1025_7b_lab11_tierb_sentiment_negation_labs1_25_full_m_results.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
