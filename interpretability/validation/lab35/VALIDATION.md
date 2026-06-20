# Lab 35 Validation

## Lab 35: Reproducible Interpretability Paper Capstone

Reproducible interpretability paper capstone: preregistration, frozen-run binding, adversarial review, repair accounting, reproduction, and claim-card discipline.

## Validation Read

This pack prefers the newest broad validation artifacts available in the local runs tree: recent Lab 6 matrix/reruns where applicable, `run6` and `verify_part3` for the main course sweep, and standalone Severance reruns for Lab 36.

- `interpret/verify_part3/lab35_tierc_full_verify_20260615_0107/lab35_tierc_full_verify_20260615_0107` (gpt2, tier c)
  - Metrics: `evidence_rung`=AUDIT + FORMAL scaffold; source rung inherited after frozen run binding, `science_ready`=False, `selection_note`=selected first recommended seed track
  - 1. `diagnostics/warning_summary.csv`
  - 2. `plot_manifest.json`
  - 3. `tables/result_binding_template.csv`
- `interpret/verify_part3/lab35_tierb_full_verify_20260615_0104/lab35_tierb_full_verify_20260615_0104` (gpt2, tier b)
  - Metrics: `evidence_rung`=AUDIT + FORMAL scaffold; source rung inherited after frozen run binding, `science_ready`=False, `selection_note`=selected first recommended seed track
  - 1. `diagnostics/warning_summary.csv`
  - 2. `plot_manifest.json`
  - 3. `tables/result_binding_template.csv`
- `interpret/verify_part3/lab35_tiera_full_verify_20260615_0100/lab35_tiera_full_verify_20260615_0100` (gpt2, tier a)
  - Metrics: `evidence_rung`=AUDIT + FORMAL scaffold; source rung inherited after frozen run binding, `science_ready`=False, `selection_note`=selected first recommended seed track
  - 1. `diagnostics/warning_summary.csv`
  - 2. `plot_manifest.json`
  - 3. `tables/result_binding_template.csv`

## What This Lab Teaches

- The central lesson is to separate readable structure from causal use with controls, patches, and held-out checks.
- Negative findings are part of the course evidence: a method that refuses an overclaim is working.
- Held-out transfer is the main guardrail against reading a fitted artifact as a mechanism.

## Selected Source Runs

| Source | Model | Tier | Notes |
|---|---|---|---|
| `interpret/verify_part3/lab35_tierc_full_verify_20260615_0107/lab35_tierc_full_verify_20260615_0107` | `gpt2` | `c` | `evidence_rung`=AUDIT + FORMAL scaffold; source rung inherited after frozen run binding; `science_ready`=False; `selection_note`=selected first recommended seed track |
| `interpret/verify_part3/lab35_tierb_full_verify_20260615_0104/lab35_tierb_full_verify_20260615_0104` | `gpt2` | `b` | `evidence_rung`=AUDIT + FORMAL scaffold; source rung inherited after frozen run binding; `science_ready`=False; `selection_note`=selected first recommended seed track |
| `interpret/verify_part3/lab35_tiera_full_verify_20260615_0100/lab35_tiera_full_verify_20260615_0100` | `gpt2` | `a` | `evidence_rung`=AUDIT + FORMAL scaffold; source rung inherited after frozen run binding; `science_ready`=False; `selection_note`=selected first recommended seed track |

## Curated Artifacts

- `gpt2_lab35_tierc_full_verify_20260615_0107_capstone_dashboard.png`
- `gpt2_lab35_tierc_full_verify_20260615_0107_evidence_rung_matrix.png`
- `gpt2_lab35_tierc_full_verify_20260615_0107_tables_source_claim_binding_matrix.csv`
- `gpt2_lab35_tierc_full_verify_20260615_0107_tables_claim_language_audit.csv`
- `gpt2_lab35_tierb_full_verify_20260615_0104_capstone_dashboard.png`
- `gpt2_lab35_tierb_full_verify_20260615_0104_evidence_rung_matrix.png`
- `gpt2_lab35_tierb_full_verify_20260615_0104_tables_source_claim_binding_matrix.csv`
- `gpt2_lab35_tierb_full_verify_20260615_0104_tables_claim_language_audit.csv`
- `gpt2_lab35_tiera_full_verify_20260615_0100_capstone_dashboard.png`
- `gpt2_lab35_tiera_full_verify_20260615_0100_evidence_rung_matrix.png`
- `gpt2_lab35_tiera_full_verify_20260615_0100_tables_source_claim_binding_matrix.csv`
- `gpt2_lab35_tiera_full_verify_20260615_0100_tables_claim_language_audit.csv`

## Caveats

- This is a curated validation pack, not a complete raw-results archive.
- Prefer the source run directory when auditing exact configs, seeds, prompts, or full tables.
- Older runs are intentionally de-emphasized when newer validation/rerun artifacts exist.
