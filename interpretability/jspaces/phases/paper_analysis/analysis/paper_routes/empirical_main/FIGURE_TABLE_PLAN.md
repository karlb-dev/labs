# Paper A (empirical main) — frozen figure/table ledger (P7)

Schema per `protocol/CANONICAL_SCHEMA.md`. Reconstruction-status
vocabulary §7.2. Frozen before prose polish; changes require a ledger
edit committed separately from drafting.

## Main figures

| id | question | source evidence ids | source table / data | script | tier | retro? | caption claims (allowed) | forbidden caption claims | recon status |
|---|---|---|---|---|---|---|---|---|---|
| E1 | Why is the final causal effect credible? (assay repair ladder) | `p3-span-audit-cross-model-v1`; `n6-confirmatory-analysis-v2`; `p3-locked-analysis-v1` | span_audit parquets + era estimates | `jspace_paper/scripts/run_analysis.py` (figE_eras) | development→confirmatory ladder | no (registered quantities) | label-protection leak measured (28–42% energy); span-safe effect ≈ 1/3 of label effect and survives at p≈1e-5 | "the original effect was an artifact"; any unprotected-arm inference | verified: byte-identical source rows |
| E2 | Does a content-specific causal tail exist and replicate? | `p3-locked-analysis-v1`; `p3-replication-analysis-v1`; `n6-*-analysis-v2` | p3_grid item parquets | `run_analysis.py` (figA ECDF) | confirmatory + held_out_replication | no | J tail vs zero control tail; +0.0958/+0.1021; exact matched dose | "global workspace"; cross-model universality | byte_identical (39/39) |
| E3 | Is the channel bridge-consumable on Qwen? | `p3-bridge-geometry-qwen36-27b-v2`; `p3-bridge-mediation-qwen36-27b-v1`; `p3-bridge-swap-endpoint-qwen36-27b-v1` | bridge_mediation parquets + geometry audit | `run_analysis.py` (figC forest) | confirmatory (rescue) + development (factorial/substitution) | no | rescue +0.431 (unreplicated, chosen distractor); swap −4.05 dev; answer-direction not separated | "abstract bridge channel proven"; "mediation replicated" | byte_identical |
| E4 | How does organization vary across the OLMo lineage? | `p4-lineage-*`; `ol-capacity-joint-dev-v1`; `ol-geometry-joint-dev-v1`; `ol2-stage-wedge-joint-analysis-v1` | `analysis/tables/olmo_lineage_matrix_inputs.csv` | `analysis/scripts/build_olmo_synthesis.py` | prespecified_development (watermarked) | no | capacity flat; dictionary moves once; causal use follows Think path; wedge capability-gated | any SFT/DPO attribution; Instruct-as-successor; zeros in gated cells | byte_identical (138/139) |
| E5 | What does the cross-model evidence actually contain? | ledger C1–C7 row citations | `analysis/tables/cross_model_evidence_matrix.csv` | `analysis/scripts/build_cross_model_matrix.py` | mixed (per-cell tiers shown) | yes (descriptive juxtaposition) | one confirmatory+replicated cell; untested cells visible | any scalar model ranking / "workspace score" | derived from verified cells |
| E6 | Claim-boundary schematic (readout → channel → bridge → externalization → workspace) | ledger C1–C6 | (schematic; claims table) | drafted at P8 | discussion | yes | licensed nouns per tier; externalization gated | "workspace" as earned term | n/a (schematic; claims audited) |

## Main tables

| id | content | source | recon status |
|---|---|---|---|
| TA1 | model/checkpoint/lens manifest (three primaries + revisions + lens provenance) | PHASE3 state §1; `ol-lens-provenance-audit-v1` | verified |
| TA2 | task populations, partitions, independent units | `d5-partition-freeze-v1`; PHASE3 state | verified |
| TA3 | primary + replication estimates (HP1/HP3, P3-P1/P2/P3) with exact inference labels | `tables/recon_phase2.csv`, `recon_phase3.csv` | byte_identical |
| TA4 | bridge mechanism summary (rescue, factorial arms, substitution boundary) | `recon_phase3.csv`, `recon_paper_draft.csv` | byte_identical |
| TA5 | OLMo claim/tier table | `OLMO_LINEAGE_CLAIMS_TABLE_V2.md` + `tables/olmo_lineage_evidence_matrix.csv` | verified |
| TA6 | limitations + missingness register (A7 subset) + unsupported-number register | `A7_MISSINGNESS_AND_GATES.md`; `UNSUPPORTED_NUMBER_REGISTER.md` | n/a (governance) |

Rules: development panels watermarked; gated cells rendered as gates,
never zeros; every caption claim maps to a ledger row; P3-P1 wording is
descriptive only (no near-miss language); replication P-HP1 population
disclosure appears wherever HP1 does.
