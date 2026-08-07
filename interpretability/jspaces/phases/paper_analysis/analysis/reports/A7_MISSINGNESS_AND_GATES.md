# A7 — missingness and gate audit

Source: `data/missingness_register.parquet` + `data/capability_gating.parquet` (`scripts/build_missingness_register.py`). States use the frozen vocabulary; none of these cells may be rendered as zero, pooled as a null, or dropped silently from a figure.

## Register

| Study | Scope | Absent cell | State | Gate evidence |
|---|---|---|---|---|
| olmo_study2 | Think-SFT | Bank-S seven-condition stage effects (both frames) | `capability_gated_missing` | `ol2-stage-wedge-think-sft-tier1-v1` |
| olmo_study2 | Think-DPO | Bank-S seven-condition stage effects (both frames) | `capability_gated_missing` | `ol2-stage-wedge-think-dpo-tier1-v1` |
| olmo_study2 | SFT vs DPO | adjacent-boundary stage localization | `capability_gated_missing` | `ol2-stage-wedge-joint-analysis-v1` |
| olmo_study2 | SFT/DPO | Tier-2 wedge lens refit | `protocol_gated_missing` | `ol2-stage-wedge-joint-analysis-v1` |
| phase4 | Qwen+OLMo pair | Bank-W cross-model load intervention (P4-P3) | `blocked` | `p4-bank-w-capability-joint-imported-dev-v1` |
| olmo_study2 | Think/Instruct pair | Bank-W pair intervention (redesign) | `underpowered` | `ol2-bank-w-olmo-pair-power-v1` |
| phase4 | Qwen | P4-P1 bridge-vs-answer-direction orthogonal shot | `not_applicable` | `p4-bank-b-power-dev-v1` |
| phase4 | Qwen | P4-P2 candidate primary | `not_applicable` | `p4-qwen-canonical-lens-decision-a1000-dev-v1` |
| phase4 | Qwen | Phase 4 Holm confirmatory family | `not_run_by_stop_rule` | `p4-qwen-canonical-lens-decision-a1000-dev-v1` |
| phase4 | Qwen | A2000 fit extension / retained-extremes producer / M3 / M4 | `not_applicable` | `p4-qwen-canonical-lens-decision-a1000-dev-v1` |
| olmo_study2 | Base+Think | causal-assay dose placement on H6 ladder | `archive_unavailable` | `ol2-transport-validation-joint-v1` |
| phase4 | Qwen | A120-A250 operational state.json | `known_deficit` | `p4-qwen-a120-a250-state-permanent-deficit-v1` |
| phase3 | Qwen | P3-P3 held-out-family replication | `protocol_gated_missing` | `p3-inference-audit-v1` |
| phase2 | all | HM1 context-specific transport arm | `not_applicable` | `n6-confirmatory-analysis-v2` |
| phase2 | all | HP5 load-interaction outcome | `protocol_gated_missing` | `g5-item-manifest-v5` |
| phase2 | Gemma-4-31B | HP4 occupancy cell | `not_applicable` | `gm-state-of-record-v1` |
| olmo_study1 | lineage | O5 grid | `not_run_by_stop_rule` | `ol-o5-feasibility-decision-v1` |
| phase4 | Qwen | prompt-323 historical-runtime influence shape | `archive_unavailable` | `p4-runtime-identity-synthesis-v1` |

## Reasons (verbatim boundary facts)

- **Bank-S seven-condition stage effects (both frames)** (Think-SFT): 972-row battery: overall capable 0.00617, Bank-S 0.00833, 0 Bank-S facts capable on direct+composed; prospective floors 72 facts / 20 families.
- **Bank-S seven-condition stage effects (both frames)** (Think-DPO): 972-row battery: overall capable 0.00309, Bank-S 0.00278, 0 Bank-S facts capable on direct+composed.
- **adjacent-boundary stage localization** (SFT vs DPO): both cohorts empty; route null_or_unresolved; neither SFT- nor DPO-boundary evidence.
- **Tier-2 wedge lens refit** (SFT/DPO): adjacent-boundary and frame-agreement preconditions unmet; Tier 2 forbidden.
- **Bank-W cross-model load intervention (P4-P3)** (Qwen+OLMo pair): 16/20 common-capable families under strict import + fresh mainline replay.
- **Bank-W pair intervention (redesign)** (Think/Instruct pair): outcome-blind power 0.7788 at 16 shared capable families vs 0.80 target; first passing count 18.
- **P4-P1 bridge-vs-answer-direction orthogonal shot** (Qwen): estimation-only: Bank B underpowered and orthogonal shot not applicable under Q-L4; never run.
- **P4-P2 candidate primary** (Qwen): removed by Q-L4 before pilot, review, or outcome; SESOI 0.20 unchanged and unused.
- **Phase 4 Holm confirmatory family** (Qwen): zero opened tests; no alpha transfer or post-hoc replacement.
- **A2000 fit extension / retained-extremes producer / M3 / M4** (Qwen): no A2000 branch exists; M3/M4 never opened.
- **causal-assay dose placement on H6 ladder** (Base+Think): six registered source tables audited; none contains exact (model,item,layer,position) total-dose + residual-norm records; coverage is unavailable, NOT 0%.
- **A120-A250 operational state.json** (Qwen): permanent historical deficit (SHA-256 361bda08...); 16/17 outputs of the event verify; decision role superseded by later live gates.
- **P3-P3 held-out-family replication** (Qwen): replication cell never authorized by the frozen Phase 3 protocol; confirmatory contrast stands unreplicated.
- **HM1 context-specific transport arm** (all): pre-run gate failed (-0.04 median band improvement, 0 layers at 0.80); arm not admitted.
- **HP5 load-interaction outcome** (all): G5 bank built at dev tier; load intervention never opened in Phase 2; thread routed to Bank-W lineage.
- **HP4 occupancy cell** (Gemma-4-31B): excluded by J-lens validity premise failure (PI amendment); explicitly not a below-boundary data point.
- **O5 grid** (lineage): feasibility decision closed O5 without opening it.
- **prompt-323 historical-runtime influence shape** (Qwen): distribution-content and compiled-kernel identities not preserved at fit time; current-runtime shape only.

## State tally

- `not_applicable`: 5
- `capability_gated_missing`: 3
- `protocol_gated_missing`: 3
- `archive_unavailable`: 2
- `not_run_by_stop_rule`: 2
- `blocked`: 1
- `known_deficit`: 1
- `underpowered`: 1

## The canonical example (enshrined by the addendum)

The OLMo SFT/DPO wedge capability table — capable rates 0.617%/0.309% and **zero** Bank-S facts capable on direct + composed at either checkpoint — is the campaign's canonical demonstration that missing, gated, and not-applicable are data states: the seven-condition stage effects exist as *questions* with empty prospective cohorts, not as nulls. Any figure rendering these cells must show a gate glyph, never a zero bar.
