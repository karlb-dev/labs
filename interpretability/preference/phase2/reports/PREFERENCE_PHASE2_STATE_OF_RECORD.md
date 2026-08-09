# PREFERENCE_PHASE2_STATE_OF_RECORD

Phase 2 of the preference campaign (Lab 38) on branch `interp_preference_phase2`; freeze `preference-phase2-freeze-v1` + single E2 amendment; registry `phase2/reports/evidence_events.jsonl` (append-only). Completeness is the OLMo-32B spine (addendum E17). License: agent_dual_code_provisional pending PI ratings; deviations D1-D10 in `preregistration/DEVIATIONS.md`.

## Adjudication headlines (primary model, frozen tier)

- PC gate: PASS (parse 1.0000, expected 0.942); NC alarm quiet.
- B-ARB3 statuses: {"arb_component": "CLEAN_NULL", "arb_docsection": "CLEAN_NULL", "arb_execmode": "ENACTED_CHOICE", "arb_lint": "CLEAN_NULL", "arb_naming": "CLEAN_NULL", "arb_notes": "CLEAN_NULL", "arb_seed": "CLEAN_NULL", "arb_setup": "CLEAN_NULL", "arb_shard": "CLEAN_NULL", "arb_storage": "CLEAN_NULL", "arb_testorder": "CLEAN_NULL", "arb_traversal": "CLEAN_NULL"}
- Context ladders: {"mech_component": "CONTEXTUAL_VALUE", "mech_docsection": "CONTEXTUAL_VALUE", "mech_execmode": "CONTEXTUAL_VALUE"}
- Mechanism: {"pcmech_summary_d3": "PC_MECH_FAIL"} (mechanistic PC pass: False)
- Coupling routers: {"mech_component": "NO_AR_HANDLE", "mech_docsection": "NO_AR_HANDLE", "mech_execmode": "NO_AR_HANDLE"}

## Cross-model presence

| model | behavioral cell |
|---|---|
| olmo32b | complete |
| olmo7b | complete |
| qwen | complete |
| gemma | absent (see drop/STOP_P events) |

## Session notes

One VM session, 2026-08-08/09, RTX PRO 6000 Blackwell 96GB. The
campaign ran P2-0 through closeout end-to-end under the PI's standing
authorization (H5 basis; DEVIATIONS D5), with every adjustment recorded
(D1-D11). Verdict ladder, in the plan's own vocabulary:

1. H-SURF ANSWERED. The Phase 1 "first-position policy" de-aliases into
   a dominant reply-list-order policy (+0.230, exact p .0078) plus a
   label-rank pull (+0.107); raw token order is null in the Phase 1
   clone format (-0.014, ns) and small (+0.117) in label-free F-SYM.
   Out-of-sample reconstruction of frozen Phase 1 cells: MAE 0.112.
2. H-SEM/H-ENACT on the primary: ONE enacted graduate, arb_execmode
   (+1.176 nats, strict +0.312, parse 1.0000); eleven CLEAN_NULL under
   the 0.833-nat floor. The floor is itself a discovery: the
   paraphrase-twin NC family shows pure wording moves full-target
   margins at the scale of most Phase 1 "content asymmetries"
   (f1-f2 p95 ~0.42 nats), and 10/12 sub-floor margins still share the
   positive (toward-conventional) sign — Phase 1's common sign now has
   a candidate explanation at wording scale.
3. H-CONTEXT CONFIRMED 3/3 anchors (slopes +0.239/+0.452/+0.565
   nats per strength unit; holdout rank rho 0.90-1.00; family-4 null
   ladders flat: slope-floor 0.000, p 1.0).
4. H-MECH: preregistered STOP_PCMECH. The randomized-context direction
   is superbly DECODABLE (validation r 0.71-0.83 vs permutation bands
   ~0.10, AUC 0.97-1.00) yet fails neutral-relevance at 58/60 anchor
   cells, and the mechanistic PC's causal assays moved holdout margins
   ~0.01 nats at guard-safe doses with a live-verified hook: on this
   model, single-token upstream residual edits do not steer the
   decision (context is distributed across the context span; the
   Phase 1 +0.787-nat PC movement lived at the output-adjacent final
   token). No arbitrary causal work or coupling was licensed; routers
   NO_AR_HANDLE x3. This sharpens Phase 1's lesson into a two-sided
   dissociation: behavioral graduation does not imply a linear handle,
   and decodable context coding does not imply causal leverage at a
   single site.
5. H-XMODEL: MODEL-SPECIFIC MAPS. Qwen3.6-27B (clean gates, parse
   1.0000) shows FIVE enacted graduates (execmode +2.185/+0.435;
   component +1.920/+0.398; lint; setup; testorder) plus three
   margin-only scenarios (docsection +1.737 with strict +0.014 — a
   margin/choice dissociation) over its own 1.055-nat floor, with steep
   ladders (component 3.51 nats/unit). arb_execmode (single-batch
   first) is the one cross-family shared graduate. OLMo-7B: STOP_P
   (PC 0.731 under F-SYM — its surface policy overrides even positive
   controls on the label-free format; marginal L4 NC-paraphrase alarm;
   INSTRUMENT_FAILURE tier; PI review flagged). Gemma-4-31B-it: STOP_P
   at the format gate (a trailing 'thought' channel breaks the strict
   single-line contract at parse 0.078 while content is perfect where
   parseable).
6. Instrument findings for reuse: batched GENERATION inequality on the
   7B (extends the Phase 1 bf16 margin finding; single-row fallback per
   the prereg); in-carrier code priors are O(1 nat) for arbitrary
   opaque codes (D10 — folded-order audit; the code_map counterbalance
   plus NC floors carry validity); same-depth propagation recapture is
   zero by construction (D11).

Dropped under the frozen order with this event chain (secondary arms,
session budget): F-P1 expansion beyond the banked continuity arm,
five-point RO expansion, B-CODE reconstruction, DG forced-exit.
Cross-model mechanism closed by STOP_PCMECH on the primary. H-CANON
composite sheet is frozen and exploratory-tier per D4; its evaluation
against the 15 heldout sign targets is left to the PI-ratified analysis
pass (the coding sheet and all inputs are banked).

---
*Claim ceiling (plan §8): every statement above is about functional choice, semantic decision margins, contextual relative advantage, enacted branches, report-only selection, scenario-local causal handles, and functional choice/report coupling under this battery. No statement licenses mental-state language; the forbidden upgrade list is enforced by the raising language wall. License: agent_dual_code_provisional pending PI ratings.*
