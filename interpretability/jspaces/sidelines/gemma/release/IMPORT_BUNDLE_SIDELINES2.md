# Gemma transport Study-2 import bundle

Bundle ID: `jspace-gemma-transport-sidelines2-v1`  
Evidence ID: `gm2-sidelines2-import-bundle-v1`  
Generated: `2026-08-03T07:24:12Z`  
Clean render commit: `aba2e01460dde32e5c2ca1478a5502950e2448ec`  
JSON payload SHA-256: `1751d22a171bc2e372f0065d1fc015e9d90bc8c0583b97ef9c76394f348c731a`

## Import boundary

The exact pre-release registry prefix is 36,279 bytes / 23 events, SHA-256 `2a144bcf0e7be0ac4307f7e2a2984c1879340b9a7e9278d10143d122a14fd30a`. It ends at `gm2-stage1-relicense-v1` and contains 22 origins, 22 live events, and 52 live outputs.

This bundle is partial-safe: mandatory G2.1 and G2.2 are complete; mechanism, intervention, and confirmatory items remain explicitly unopened. Development/methods tiers are preserved.

## Scientific disposition

- Calibration route: `benign_scheduling_floor`.
- Pooled all-frozen-batches ceiling: `0.07870368901355948` from 216 full pairs.
- Historical all-slot relative error: `0.0024581113830208778`; selected slot remains bit-identical.
- License route: `branch_1_relicense_without_recompute`; no G2.2 model compute.
- L22/L30/L37/L44/L52 remain `local_tangent_mismatch`, now a closed methods result over the tested finite-scale scope.
- No mechanism, workspace, intervention, or confirmatory claim opens.

## Partial and terminal statuses

- `study1_blocker_record`: `preserved-immutable-historical`.
- `g2_1_backend_calibration`: `complete-target-blind-pooled-ceiling`.
- `g2_2_stage1_license`: `complete-relicensed-without-recompute`.
- `five_layer_classifier`: `complete-closed-methods-result`.
- `olmo_h6_ceiling_export`: `complete-licensed`.
- `mechanism_localization`: `not-executed-not-licensed-by-this-study`.
- `intervention`: `not-opened`.
- `confirmatory_or_replication_cell`: `not-opened`.

## Admitted evidence

- `gm-jvp-goldens-v1` — methods / study1_methods_chain; 1 outputs.
- `gm-jvp-olmo-positive-control-v1` — methods / study1_methods_chain; 2 outputs.
- `gm-jvp-gemma-stage1-v1` — methods / study1_methods_chain; 8 outputs.
- `gm-jvp-gemma-backend-parity-v1` — methods / study1_methods_chain; 2 outputs.
- `gm-state-of-record-v1` — methods / study1_methods_chain; 8 outputs.
- `gm2-foundation-v1` — methods / study2_foundation; 1 outputs.
- `gm2-backend-parity-calibration-v1` — methods / target_blind_calibration; 6 outputs.
- `gm2-stage1-relicense-v1` — methods / mechanical_license; 2 outputs.

## Release artifacts

| Role | Repo source | Drive target | Bytes | SHA-256 |
|---|---|---|---:|---|
| state-of-record-v2 | `interpretability/jspace_gemma/release/GEMMA_TRANSPORT_STATE_OF_RECORD_V2.md` | `gemma-run://release/GEMMA_TRANSPORT_STATE_OF_RECORD_V2.md` | 6391 | `9c8c3b9be027a5659347bde3a26ab02fd12ed7f7903c3e9a2d785b809e12de67` |
| claim-ledger-v2 | `interpretability/jspace_gemma/release/gemma_transport_claim_ledger_v2.md` | `gemma-run://release/gemma_transport_claim_ledger_v2.md` | 4293 | `bc10e3ae5f9339d08328780766326f53e94af63dab3a22f40a535cc59f139591` |
| study2-report | `interpretability/jspace_gemma/reports/GEMMA_TRANSPORT_STUDY2_REPORT.md` | `gemma-run://release/GEMMA_TRANSPORT_STUDY2_REPORT.md` | 7660 | `6ba43c1fa75aa7b8892dc07d29f89354b2a953e28b03c4b3cf34f637bf502180` |
| transport-gate-protocol-v2 | `interpretability/jspace_gemma/release/TRANSPORT_GATE_PROTOCOL_V2.md` | `gemma-run://release/TRANSPORT_GATE_PROTOCOL_V2.md` | 6051 | `ec75a631770a251b0558f52b5faf22b5c1991365cef3520b59ebab85f945a43c` |
| handout-tex | `interpretability/jspace_gemma/reports/handout/gemma_transport_development.tex` | `gemma-run://release/GEMMA_TRANSPORT_STUDY2_HANDOUT.tex` | 8054 | `afaf90af4f3b96f5e1b267607e52839725a38cbf25b4713e343863d0383187e2` |
| handout-pdf | `interpretability/jspace_gemma/reports/handout/gemma_transport_development.pdf` | `gemma-run://release/GEMMA_TRANSPORT_STUDY2_HANDOUT.pdf` | 216734 | `518b1fac1997469e364ebe641fcc40b99b513c0a57144a47668793a3882fd5c9` |
| calibration-figure | `interpretability/jspace_gemma/reports/handout/figures/gm2_backend_disagreement_by_model_batch.png` | `gemma-run://release/figures/gm2_backend_disagreement_by_model_batch.png` | 99091 | `9c21f13c90a8b9d3d0325a699e025f1267dde8548bc3e74eb6f592ff0bb773ff` |
| frozen-design | `interpretability/jspace_gemma/preregistration/G2_STUDY2_FROZEN_DESIGN.md` | `gemma-run://release/protocol/G2_STUDY2_FROZEN_DESIGN.md` | 2495 | `3276e20b6943c1ca91da18d0d258ca030e85accf0a9d5aef867e85dcf338fbb9` |
| calibration-config | `interpretability/jspace_gemma/configs/gm2_backend_parity_calibration.yaml` | `gemma-run://release/protocol/gm2_backend_parity_calibration.yaml` | 4905 | `42b003729d4a3a7632db672b81980cf0b92ba79208a78b7b40225f2dcb917a0c` |
| license-config | `interpretability/jspace_gemma/configs/gm2_stage1_relicense.yaml` | `gemma-run://release/protocol/gm2_stage1_relicense.yaml` | 2950 | `e31b9d1c27438da0bb15e1a0c8495a7cd6a1119e4fd3b47521dacc750e7e8f05` |
| candidate-sentences | `interpretability/jspace_gemma/protocol/G2_STAGE1_CANDIDATE_SENTENCES.md` | `gemma-run://release/protocol/G2_STAGE1_CANDIDATE_SENTENCES.md` | 2339 | `cd22943e90b5156eee0ae5d0038f0c826b9876054660711944209185fd800dab` |
| predata-correction | `interpretability/jspace_gemma/protocol/G2_PRE_DATA_ARCHITECTURE_CORRECTION.md` | `gemma-run://release/protocol/G2_PRE_DATA_ARCHITECTURE_CORRECTION.md` | 1379 | `97e1f9fbeff95a0adbf522b898ecd3c22a397792cc1ff2619ed586a845fd774b` |
| reconstruction-audit-correction | `interpretability/jspace_gemma/protocol/G2_POSTDATA_RECONSTRUCTION_AUDIT_CORRECTION.md` | `gemma-run://release/protocol/G2_POSTDATA_RECONSTRUCTION_AUDIT_CORRECTION.md` | 1709 | `3a000f4e7f826a1933786aee79ab963af46c695402b1d809bea47bcf94ef729a` |

## Forbidden uses

- editing or retroactively passing the immutable Study-1 failed-gate event.
- claiming nondifferentiability or Jacobian absence.
- attributing the finite-scale mismatch to a named mechanism.
- inferring missing information or workspace absence.
- generalizing outside the tested prompts, layers, directions, target map, and doses.
- opening an intervention or confirmatory model cell from this bundle.
- confirmatory, replication, independent-review, or PI-sign-off import.

## Importer checks

- [ ] source commit is reachable.
- [ ] registry snapshot and current registry share the exact released prefix.
- [ ] all admitted events remain live at their native tier.
- [ ] all admitted output paths, sizes, and SHA-256 values verify.
- [ ] all V2 reports, TeX/PDF bytes, figure, designs, and configs verify.
- [ ] Study-1 failed-gate evidence remains immutable.
- [ ] G2.2 remains a no-recompute license decision.
- [ ] no mechanism, workspace, intervention, or confirmatory tier is inferred.

## Claim boundary

Self-verifying partial-safe Study-2 handoff. The mandatory target-blind G2.1 calibration and mechanical G2.2 license are complete. The unchanged five-layer classifier is a closed exact-JVP finite-scale methods result over the tested scope; mechanism, workspace, intervention, and confirmatory claims remain unopened. Native development/methods tiers are preserved.
