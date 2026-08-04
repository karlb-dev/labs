# Phase 4 freeze-gate ledger

**CANDIDATE LEDGER — NOT FROZEN — INDEPENDENT REVIEW AND PI APPROVAL PENDING**

State date: 2026-08-04. Governing plans:
`jspace_lab_nextsteps_4_4.md` plus its accepted addendum, and the Phase 4.5
closeout block `jspace_lab_nextsteps_4_5.md` plus its addendum. Scientific
candidate: `SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md`, candidate 0.13
Study-2 admission update.

This ledger records the mechanical pre-freeze state. A development artifact
does not authorize confirmatory or replication intervention. Only real
independent review, PI sign-off, a clean release proof, a freeze commit, and a
freeze tag can change that boundary.

## Instrument and execution gates

| Gate | State | Bound evidence |
|---|---|---|
| Qwen A1000 cumulative draw-A lens | **REGISTERED / INTEGRITY PASS** | `p4-qwen-lens-fit-drawA-n1000-dev-v1`; 1,000 prompts; 63-layer finiteness and exact quantized-mean audit pass. |
| A500--A1000 structural contract | **REGISTERED / PASS** | `p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1`; task q50/q05 0.998702/0.998122 pass 0.95/0.90. |
| Functional contract | **REGISTERED / Q-L4 INPUT** | `p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1`; ID Jaccard 0.538462, projector overlap 0.709818, and bridge-rescue difference -0.294028 nat fail. |
| Selection-margin audit | **REGISTERED / COMPLETE** | `p4-qwen-selection-margin-a500-a1000-dev-v1`; all 17,381 positions retained; 15,536 near-tie, 1,845 stable-core, zero rank-deficient. |
| Prompt-323 influence | **REGISTERED / NEGLIGIBLE / CURRENT-RUNTIME ONLY** | `p4-qwen-lens-influence-prompt323-dev-v1`; repeat difference 0.004572 <= 0.5; all frozen A500/A1000 materiality metrics pass negligible. Historical-runtime reproducibility is not claimed. |
| Canonical Qwen decision | **REGISTERED / Q-L4 / NO LENS** | `p4-qwen-canonical-lens-decision-a1000-dev-v1`; mechanical result Q-L4; no canonical sparse instrument nominated. |
| Optional retained-extremes M5 | **NOT APPLICABLE** | No producer/configuration was prospectively frozen before Q-L4; none was authored post-outcome. |

## Candidate-primary decisions

| ID | Terminal state | Legal disposition |
|---|---|---|
| P4-P1 | **ESTIMATION-ONLY** | Bank B remains underpowered. The orthogonal feasibility shot is **NOT APPLICABLE** under Q-L4 and was not run. |
| P4-P2 | **REMOVED / BLOCKED BY Q-L4** | Requires Q-L1/Q-L2 and independent producer review. No pilot, power execution, untouched-bank review, or intervention outcome opened. SESOI remains 0.20 and unchanged. |
| P4-P3 | **BLOCKED AT 16/20** | Registered OLMo import and fresh mainline replay confirm 16 common-capable families versus 20 required. No Bank-W intervention is authorized. |

No candidate primary remains; zero confirmatory tests entered the Holm family.
A retired or blocked endpoint does not donate multiplicity budget.

## Side-track admission

| Bundle | Mainline state | Claim boundary |
|---|---|---|
| OLMo early Bank-W capability | **REGISTERED** as `p4-import-olmo-bank-w-capability-v1` | Strict side-development import; capability only, no intervention. |
| Bank-W mainline replay | **REGISTERED** as `p4-bank-w-capability-joint-imported-dev-v1` | 16/20 service block; no cross-model primary. |
| Gemma transport | **REGISTERED** as `p4-import-gemma-transport-v1` | Methods blocker only; all-slot parity fails; no mechanism or intervention claim. |
| OLMo final lineage | **REGISTERED** as `p4-import-olmo-lineage-final-v1` | Methods/development O1--O5 boundaries preserved; no new scientific cell or intervention. |
| Gemma Study 2 (calibrated relicense) | **REGISTERED** as `p4-import-gemma-transport-study2-v1` | Methods tier. Target-blind pooled ceiling 0.07870368901355948 frozen before target read; preserved all-slot error 0.0024581113830208778 in-envelope; selected slot bit-identical; five-layer classifier now a closed finite-scale methods result. Historical Study-1 blocker preserved, not rewritten. No mechanism/workspace/intervention license. |
| OLMo Study 2 (wedge, H6, pair power) | **REGISTERED** as `p4-import-olmo-lineage-study2-v1` | Methods tier. SFT/DPO wedge capability-gated with empty Bank-S cohorts (effects missing, not zero; no localization); H6 licenses no in-band regime at L24/L32/L40, Think L56 epsilon-0.10 late anchor only; registered-dose coverage unavailable, not zero; Bank-W pair power 0.7788 at 16 families (first passing count 18) — planning closure, not a null. No intervention authorized. |

Native `ol-*`, `ol2-*`, `gm-*`, and `gm2-*` events remain absent from the
Phase 4 registry.

## Durability and governance

| Gate | State | Requirement |
|---|---|---|
| Exact A120 capacity | **RECOVERED / HASH PASS** | Rebuilt bytes equal registered SHA-256 `6b0399df...c651b6f`; local backup exists. |
| Historical A120--A250 `state.json` | **PERMANENT KNOWN DEFICIT / ROLE SUPERSEDED** | Append-only methods event `p4-qwen-a120-a250-state-permanent-deficit-v1`; source remains partially durable and unedited. External signatures required. |
| Same-mounted durability | **CONSISTENT: 418/419** | Two passes agree on one known deficit, zero unexpected deficits, zero pin conflicts. This is not independent rematerialization. |
| Pre-freeze inventory | **NOT_REVIEW_READY (mechanically) / POLICY-AWARE CLEAN** | v4_5 payload `71ae6031...`; every gate passes except `all_live_outputs_verified`, false solely from the accepted known deficit. |
| Fresh independent remount | **PASS** | `phase4-part5-fresh-materialization` (fresh VM/mount/clone, 2026-08-04): 520/521, only the known deficit, zero unexpected, zero pin conflicts. |
| Untouched-data audit | **PASS** | `PHASE4_UNTOUCHED_DATA_AUDIT.md` (narrative-blind session, 2026-08-04): zero forbidden-tier events, all run-root bytes accounted for, partitions sealed. |
| Independent protocol review | **SIGNED** | `PHASE4_INDEPENDENT_REVIEW_20260804.md` (SHA-256 `1ca0f1aa...`), narrative-blind fresh session: 13/14 PASS + 1 PASS WITH EXPLICIT LIMITATION; Q-L4 reconstructed from the frozen table. |
| PI sign-off | **RECORDED** | `PHASE4_PI_DISPOSITION_20260804.md`: 13/13 items accepted per the PI-authored plan-addendum §3 resolutions + PI session directive; freeze authorized at `07f92038...`. |
| Freeze commit / tag | **AUTHORIZED PENDING VERIFICATION** | Follows the recorded decisions and the full pre/post-merge verification suites. |

## Mechanical update rule

Corrections append supersession events; registered evidence is never
overwritten. This packet stops at `READY_FOR_PHASE4_FREEZE_REVIEW.md`. It is a
review handoff with a red freeze gate, not a freeze authorization.
