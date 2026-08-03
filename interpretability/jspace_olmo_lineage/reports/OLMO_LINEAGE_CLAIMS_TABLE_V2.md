# OLMo lineage claims table — Study 2

State date: 2026-08-03

Scope: the official Think-SFT/Think-DPO natural wedge and the Base/OLMo-3.1
Think H6 finite-dose transport validation on branch
`interp_jspace_olmo_lineage_2`. Every native claim remains development or
methods tier. This ledger supplements, rather than overwrites, the Study-1
claims table.

## Conclusion-skeleton disposition

| Skeleton target | Study-2 resolution | Exact licensed wording |
|---|---|---|
| Sentence 2: training dependence | **Not stage-localized.** The official SFT and DPO checkpoints both fail the prospective Bank-S direct+composed capability cohort, so intervention effects and adjacent contrasts are missing. H6 also finds no licensed finite-dose regime at L24/L32/L40 on the tested ladder. | “The official SFT/DPO wedge does not localize the first-release transition under the frozen assay because neither checkpoint supplies the prospective capable cohort. Across Base and OLMo-3.1 Think, the tested first-order transport approximation does not meet the frozen finite-dose gate in the L24/L32/L40 assay band.” |
| Sentence 4: externalization | **Unchanged and pending.** No stage-wedge intervention opened, and H6 is a transport-method result rather than an external-state-substitution test. | “Existing Bank-S development evidence may motivate external-state substitution, but neither the gated SFT/DPO wedge nor the H6 transport validation resolves that mechanism.” |

## Claim ledger

| ID | Claim | Evidence and decisive result | Tier | Status and allowed use |
|---|---|---|---|---|
| OL2-C01 | The official SFT and DPO artifacts form an ancestry-qualified natural wedge, not byte-proven objective interventions. | `ol2-checkpoint-ancestry-v1`; exact model/config/tokenizer hashes and model-card repository parent declarations. | methods | Required qualification for every stage sentence. Do not write randomized or objective-causal language. |
| OL2-C02 | Think-SFT fails the frozen stage capability cohort. | `ol2-stage-wedge-think-sft-tier1-v1`: 972 rows, overall capable rate 0.00617, Bank-S rate 0.00833, zero Bank-S facts capable on direct + composed. | development | Capability-gated result. All seven-condition/two-frame effects are missing, not zero. |
| OL2-C03 | Think-DPO also fails the frozen stage capability cohort. | `ol2-stage-wedge-think-dpo-tier1-v1`: 972 rows, overall capable rate 0.00309, Bank-S rate 0.00278, zero Bank-S facts capable on direct + composed. | development | Capability-gated result. It is not evidence that DPO removes a causal channel. |
| OL2-C04 | The official SFT/DPO wedge does not localize the first-release transition. | `ol2-stage-wedge-joint-analysis-v1`: 965 incapable-both, 2 capable-both, 4 lost-at-DPO, 1 Bank-F onset-at-DPO, no Bank-S direct+composed onset; route `null_or_unresolved`. | development | Licensed as a bounded null/gate. Later stages, checkpoint spacing, capability, and measurement limits remain open. Tier 2 is not triggered. |
| OL2-C05 | H6 uses an applicable, registered OLMo-specific exact-JVP backend ceiling. | `ol2-gemma-backend-calibration-import-v1` and `ol2-transport-execution-freeze-v1`: ceiling 0.0787036890, pooled ceiling forbidden, all H6 thresholds and predictions frozen before outcomes. | methods | Licenses H6 pass/fail wording only within the frozen protocol. |
| OL2-C06 | Base has no licensed transport-valid layer/dose cell on the frozen grid. | `ol2-transport-validation-base-v1`: 336 rows; all backend checks pass; no valid layer epsilon; L56/0.10 reaches 9/12 < 0.90. | methods | Licensed H6-fail statement for the tested grid. It does not invalidate paired ablation findings. |
| OL2-C07 | OLMo-3.1 Think is late-anchor-only under H6. | `ol2-transport-validation-olmo31-think-v1`: 336 rows; L56/0.10 is 12/12 and the only passing layer-dose cell; no L24/L32/L40 pass. | methods | Licensed only as a checkpoint-specific L56 result. Do not call the assay band transport-valid. |
| OL2-C08 | Across both checkpoints, no tested in-band regime passes; intervention-dose coverage remains unavailable rather than zero. | `ol2-transport-validation-joint-v1`: 672 rows; route `h6_fail_in_band_with_checkpoint_specific_late_anchor`. Six registered source tables audited, zero contain the exact total site dose plus residual-norm terms. | methods | Use the tested-ladder fail sentence and explicit archive limitation. Do not infer scale-limited, relevant-dose pass/fail, or 0% coverage. |
| OL2-C09 | The Study-2 H6 plot is a registered derivative of joint evidence. | `ol2-transport-validation-figure-v1`: visually checked PNG/PDF; `scientific_result_changed=false`. | methods | Presentation only; no tier or claim upgrade. |

## Required distinctions

- Capability-gated effects are missing, not zero.
- No measured in-band transport regime is not a claim of model
  nondifferentiability.
- A Think-only late anchor is not an assay-band pass.
- Missing intervention-dose records are not zero dose or zero coverage.
- H6 narrows transport interpretation; it does not validate or invalidate the
  previously registered paired ablation effects.
- Natural checkpoint differences do not identify SFT or DPO objective effects.

## Prohibited formulations

- “SFT installs the workspace” or “DPO installs the workspace.”
- “DPO removes the channel.”
- “The SFT/DPO intervention effect is zero.”
- “OLMo has no Jacobian” or “the model is nondifferentiable.”
- “The causal intervention lies outside the valid dose range” while the exact
  registered dose distribution is unavailable.
- “Transport failure invalidates the paired ablation result.”
- Any confirmatory, replication, O5, Bank-W, or receiver result absent from the
  append-only `ol2-` registry.

## Handoff rule

The Study-2 import bundle must carry this ledger, the V2 state of record, the
complete registry prefix, the stage-wedge and H6 joint results, both
paper-facing figures, and explicit partial statuses. Downstream work may cite
only bundle-admitted artifacts and must preserve their development/methods
tier and claim boundaries.
