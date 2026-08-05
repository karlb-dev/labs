# PAPER_REPRODUCTION_AUDIT.md — P9 reproduction audit

Auditor: fresh, independent session, 2026-08-05. CPU only; no model
load, no forward/backward pass. Campaign registries and Drive run roots
read-only. Procedure: SHA-256 snapshot of every file under
`analysis/{data,tables,figures,reports,manifests}` (53 files), then the
eight deterministic pipeline scripts rerun in plan order from the
committed inputs, then re-hash + `git status` comparison.

## Per-script outcomes

| # | Script | Exit | Console summary (this rerun) | Cross-check |
|---|---|---|---|---|
| 1 | `build_evidence_graph.py` | 0 | `events: 527; ids: 324; live: 260; anomalies: 0` | matches `CAMPAIGN_EVIDENCE_MAP.md` registry totals (137+120+123+82+24+41 events; live 78+57+65+23+37) |
| 2 | `extract_claims.py` | 0 | `ledger rows: 17 (7 campaign + 10 lab37); edges: 37; figure rendered` | matches `CLAIM_SURVIVAL_LEDGER.md` (C1–C7 + W1–W10) |
| 3 | `build_missingness_register.py` | 0 | state tally printed; 18 register rows | matches `A7_MISSINGNESS_AND_GATES.md` tally (5/3/3/2/2/1/1/1) |
| 4 | `build_cross_model_matrix.py` | 0 | `48 cells; figure + synthesis written` | matches `cross_model_evidence_matrix.csv` (48 rows; statuses D15/U13/X6/G5/S3/M3/C2/R1) |
| 5 | `build_qwen_synthesis.py` | 0 | `figure + table written` | `qwen_multilevel_convergence.csv` (36 rows) regenerated |
| 6 | `build_olmo_synthesis.py` | 0 | `72 cells rendered` | matches `olmo_lineage_evidence_matrix.csv` (72 rows) |
| 7 | `build_transport_synthesis.py` | 0 | `61 applicability rows; figure written` (benign `tight_layout` UserWarning) | matches `transport_applicability.csv` (61 rows = 5 Gemma + 28 Base + 28 Think cells) |
| 8 | `assemble_reconstruction_audit.py` | 0 | domain tally printed | tally identical to `RECONSTRUCTION_AUDIT.md` (see below) |

All eight ran green with no model access. Total wall time ≈ 40 s.

## Diff report (byte-or-numerically identical requirement)

- **Byte-identical after rerun (zero diff):** all parquets
  (`master_evidence_events`, `master_evidence_live`,
  `master_claim_ledger`, `master_claim_evidence_edges`,
  `capability_gating`, `missingness_register`,
  `reconstruction_audit_rows`), all regenerated CSV tables
  (`cross_model_evidence_matrix`, `qwen_multilevel_convergence`,
  `olmo_lineage_evidence_matrix`, `transport_applicability`), all
  regenerated reports (`CAMPAIGN_EVIDENCE_MAP.md`,
  `CLAIM_SURVIVAL_LEDGER.md`, `A7_MISSINGNESS_AND_GATES.md`,
  `CROSS_MODEL_SYNTHESIS.md`, `RECONSTRUCTION_AUDIT.md`,
  `UNSUPPORTED_NUMBER_REGISTER.md`), and **all five figure PNGs**.
- **Metadata-only diffs (no real diff):** the five figure PDFs
  (`claim_survival_timeline`, `cross_model_evidence_matrix`,
  `olmo_lineage_evidence_matrix`, `qwen_multilevel_convergence`,
  `transport_applicability_map`) differed solely in the embedded PDF
  `/CreationDate`; after normalizing CreationDate/ModDate/ID the byte
  streams are identical and sizes are unchanged (30010, 28024, 29630,
  28010, 29990 bytes). This is the `numerically_identical_render_diff`
  class the audit itself uses. The five PDFs were restored with
  `git checkout` afterwards; **`git status` is clean** — no unexpected
  diffs anywhere in the tree.
- **No real diff was found in any output.**

Note: `build_{qwen,olmo,transport}_synthesis.py` regenerate their
tables and figures; the corresponding synthesis `.md` files are not
rewritten by the current builders (their numbers were verified against
the regenerated tables in the claim audit — all consistent).

## Reconstruction-verdict tables (row counts vs `RECONSTRUCTION_AUDIT.md`)

| Table | Rows (claimed) | Rows (loaded) | Status tally (loaded) |
|---|---:|---:|---|
| `recon_gemma.csv` | 45 | **45** | 43 byte_identical + 2 render_diff |
| `recon_olmo.csv` | 139 | **139** | 138 byte_identical + 1 render_diff |
| `recon_paper_draft.csv` | 31 | **31** | 1 byte_identical + 25 render_diff + 4 within_tolerance + **1 failed** |
| `recon_phase2.csv` | 10 | **10** | 7 byte_identical + 3 render_diff |
| `recon_phase3.csv` | 39 | **39** | 39 byte_identical |
| `recon_qwen_ladder.csv` | 22 | **22** | 8 byte_identical + 14 render_diff |
| **Total** | **286** | **286** | 236 + 45 + 4 + 1 = 286 ✓ (285 verified, 1 failed = the U1 draft-prose error, quarantined) |

Progression/matrix tables load with exactly the row counts the
syntheses claim: `qwen_ladder_progression.csv` **50** ("50 verified
rows"), `olmo_lineage_matrix_inputs.csv` **340**,
`olmo_lineage_evidence_matrix.csv` **72**,
`transport_applicability.csv` **61**, `cross_model_evidence_matrix.csv`
**48**, plus `qwen_multilevel_convergence.csv` 36 and
`gemma_stage1_layer_table.csv` 83.

## Rebuilt PDFs (four updated TeX sources)

| PDF | Size (bytes) | Newer than its .tex | Update text present |
|---|---:|---|---|
| `jspace_paper/kburtram_jspace.pdf` | 988,426 | yes (00:47 → 00:49) | yes ("49.3" prose rescope) |
| `jspace_paper/olmo_lineage.pdf` | 3,686,931 | yes (00:47 → 00:50) | yes ("RETIRED AS A STANDALONE" banner) |
| `jspace_paper/gemma4_nonlinear_jacobian_handout.pdf` | 765,026 | yes (00:48 → 00:50) | yes (ceiling 0.07870368901355948 ×2) |
| `jspace_phase4/reports/handout/jspace_phase4_development.pdf` | 970,037 | yes (00:50 → 00:51) | yes ("Terminal C closeout" ×2) |

## Supplementary integrity checks performed during the audit

- Phase-2 envelope payload SHAs re-read from the frozen Drive JSONs:
  `confirmatory_analysis.json` payload_sha256 = `5df4ace5…`,
  `replication_analysis.json` = `f5367e5a…` — exactly as claimed by
  `RECONSTRUCTION_AUDIT.md`/A8.
- `PHASE4_PART5_DURABILITY_FRESH.json`: n_verified 520 (515
  literal-path + 3 repository-materialization + 2
  append-only-registry-prefix), n_failures 1, n_unexpected 0,
  only_known_deficits = true — matches every durability sentence
  audited.
- Release-manifest pin `7cf71215…` present in
  `release/PHASE4_RELEASE_MANIFEST.json`; freeze suites 302/71/90
  confirmed in `PHASE4_FREEZE_VERIFICATION.md`.
- All evidence ids cited across the audited corpus resolve in
  `master_evidence_live.parquet` with status `live`, with exactly two
  exceptions recorded as findings F1 (withdrawn
  `p3-n8-level2-qwen36-27b-v1` reachable via the TeX wildcard) and F9
  (`AMENDMENT_1_BOS_UNITS`, a preregistration document cited as an
  `\eid`) in `PAPER_ANALYSIS_INDEPENDENT_AUDIT.md`.

## Overall regeneration verdict

**PASS.** The deterministic analysis pipeline reruns green end-to-end
from the committed inputs on CPU with no model load; every data table,
report, and PNG regenerates byte-identically; the only differences are
PDF creation timestamps (content-identical after metadata
normalization, restored, tree clean); all six recon tables load with
exactly the claimed row counts summing to 286 (285 verified / 1 failed
= the quarantined draft-prose error); and the four rebuilt PDFs exist,
postdate their sources, and contain the required update passages.
