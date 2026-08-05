# Paper B (methods companion) — frozen figure/table ledger (P7)

Schema per `protocol/CANONICAL_SCHEMA.md`; status vocabulary §7.2.

## Main figures

| id | question | source evidence ids | source table / data | script | tier | retro? | caption claims (allowed) | forbidden caption claims | recon status |
|---|---|---|---|---|---|---|---|---|---|
| M1 | The validity hierarchy (operator → task rows → sparse selection → causal endpoints) | ledger C6 | (schematic over M2's data) | drafted at P8 | methods | yes | levels are distinct licensing objects | "one level implies the next" | n/a (schematic) |
| M2 | Does fit convergence propagate down the hierarchy? | `p4-qwen-lens-convergence-*`; `p4-qwen-multilens-functional-gate-*`; `p4-qwen-canonical-lens-decision-a1000-dev-v1` | `analysis/tables/qwen_multilevel_convergence.csv` | `analysis/scripts/build_qwen_synthesis.py` | methods | no | q50/q05 monotone PASS; Jaccard exactly 7/13 ×3; projector < floor; rescue oscillates; Q-L4 | "A1000 converged" unqualified; any convergence rate; canonical lens | verified (22/22) |
| M3 | Do ties or single-prompt influence explain the failures? | `p4-qwen-selection-margin-a500-a1000-dev-v1`; `p4-qwen-lens-influence-prompt323-dev-v1` | margin strata + materiality rows (`recon_qwen_ladder.csv`) | P8 (small panel from verified rows) | methods | no | 15,536 near-tie / 1,845 stable-core / 0 rank-deficient; stable-core fails too; closest materiality 3,857× under threshold (current runtime) | historical-runtime reproducibility | verified |
| M4 | How large is finite-scale mismatch vs every backend scale? | `gm-jvp-gemma-stage1-v1`; `gm2-backend-parity-calibration-v1`; `gm-jvp-olmo-positive-control-v1` | `analysis/tables/gemma_stage1_layer_table.csv` | `analysis/scripts/build_transport_synthesis.py` (panel a) | methods | no | 12–52× above calibrated scales; directional; depth-dependent; closed at tested scope | any named mechanism; nondifferentiability; workspace absence | byte_identical (43/45) |
| M5 | Where is finite-dose transport licensed at all? | `ol2-transport-validation-{base,olmo31-think,joint}-v1` | `analysis/tables/transport_applicability.csv` | `build_transport_synthesis.py` (panels b,c) | methods | no | one licensed cell (Think L56@0.10); in-band zero; eligibility is a data state | "OLMo transport fails" unqualified; dose-coverage claims | byte_identical |
| M6 | The preflight protocol (backend → delivery/SNR → tangent gate → fit invariance → selection invariance → endpoint invariance → license) | C5–C7 ledger rows | (decision-tree schematic) | drafted at P8 | methods | yes | each gate cites the campaign failure it would have caught | "protocol guarantees validity" | n/a (schematic) |

## Named section (not a footnote): runtime identity (C7)

Source: `PHASE4_RUNTIME_IDENTITY_SYNTHESIS.md`
(`p4-runtime-identity-synthesis-v1`). Content: incident → prospective
0.5-tolerance catch (contribution null; nothing promoted) → diagnostic
boundary (version identity ≠ content/kernel identity) → prospective
amendment → consequence (pin distribution contents + kernel caches;
carry a runtime control; scope cross-era claims). Optional small figure:
era-vs-recompute dot plot from the registered numbers.

## Main tables

| id | content | source | recon status |
|---|---|---|---|
| TB1 | Qwen gate results by fit boundary (structural/functional/sparse/causal) | `tables/qwen_ladder_progression.csv` | verified |
| TB2 | transport gate thresholds and outcomes (all models) | `tables/transport_applicability.csv` | byte_identical |
| TB3 | backend calibration record (G2.1 grid, ceiling rule, bootstrap) | `tables/recon_gemma.csv` | byte_identical |
| TB4 | checkpoint applicability matrix (which design is licensed where) | A5+A6 syntheses | derived from verified cells |
| TB5 | minimum reporting / preflight standard | M6 + C5–C7 | governance |
| TB6 | known reproducibility limitations (A8 §by-family + deficit register) | `A8_REPRODUCIBILITY_DURABILITY.md` | governance |

Rules: every float quotes the stored full-precision repr (render-diff
lesson); the Qwen transport row is explicitly "not run"; no
convergence-rate language anywhere.
