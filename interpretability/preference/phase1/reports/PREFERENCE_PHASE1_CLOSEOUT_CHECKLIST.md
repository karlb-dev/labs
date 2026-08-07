# PREFERENCE_PHASE1_CLOSEOUT_CHECKLIST.md — plan §16 definition of done

## Foundation
- [x] branch and package created (`interp_preference_phase1`; `preference_phase1`)
- [x] source intake hashes recorded (SOURCE_INTAKE.md; 3 inputs missing_at_intake, PI-confirmed)
- [ ] original draft outputs reproduced — **impossible; waived** (inputs lost; registry `pref1-draft-bank-reproduction-v1`)
- [x] evidence registry initialized (append-only, 16 live events at closeout)
- [x] Lab 38 registered in the bench (`LAB_PROFILES`, `CHAT_TEMPLATE_LABS`) + isolated `pref1` harness (HARNESS_DECISION.md)

## Bank
- [x] final schema versioned (`lab38_v2_phase1`, schema 1)
- [x] scientific IDs bind complete scientific content (hash-change tests green)
- [x] 12 AR + 6 PC scenarios (+2 NC per addendum D3)
- [x] 5 incidentals per scenario (3/1/1 splits)
- [x] response codes independent + counterbalanced; audited priors (0.0031 / 0.0275 nats)
- [x] AR/RO alphabets disjoint (distinct first tokens)
- [x] scenario and incidental splits frozen
- [x] human-equality review complete — `agent_dual_code_provisional` ×2 (PI accepted for Phase 1; PI/panel required for publication-grade claims)
- [x] binding specs complete (4 microtask + env-only; validators deterministic)
- [x] bank balance and pair audits pass (byte-deterministic regeneration)

## Instrument
- [x] target tokenization audit passes (7B + 32B tokenizers identical on all codes)
- [x] strict parser tests pass (12/12 adversarial)
- [x] chat-template boundary audit passes (string==direct ids, both models)
- [x] deterministic replay passes (single-row scoring; per-session certification)
- [x] resume/shard recombination passes (interrupted-resume byte-parity)
- [x] invalid choice never executes a branch (tested + 0 in 2×2,320 frozen rows)
- [x] no wrong branch executes (0 across all runs)
- [x] Tier A smoke writes the artifact contract (incl. forced microtask probe)

## Governance
- [x] development outputs labeled (`development` tier throughout)
- [x] preregistration candidate written → frozen on PI approval
- [x] freeze review approved (reviews/PHASE1_FREEZE_APPROVAL.md)
- [x] freeze record and tag created (`preference-phase1-freeze-v1`)
- [x] frozen bank/model/codebook hashes recorded (freeze record table)

## Behavioral science
- [x] frozen Tier B run complete (2,320/2,320; 7B) + 32B replication run
- [x] PC gate adjudicated (PASS, perfectly, both models — see 32B report)
- [x] AR scenario effects and nuisance effects reported
- [x] RO/AR behavioral comparison reported (dissociation pattern, ceiling language)
- [x] consequence-frame effect reported narrowly (small, scenario-local)
- [x] invalid-output sensitivity reported (11 `PK4` specimens; worst-case bounds change nothing)
- [x] graduation manifest frozen (zero graduations → Stop B)
- [x] behavioral state-of-record report written

## Conditional mechanism
- [x] run only on graduated scenarios — **zero graduated; NOT run** (preregistered module + sealed captures retained for a future phase)
- [-] remaining §16 mechanism boxes not applicable under Stop B

## Secondary track
- [x] DG schema bug repaired (explicit primary_dv; v2 generator)
- [x] DG-SAFE remains forward-only (never rolled out; runner refuses)
- [x] any DG result clearly secondary (development tier, smoke n)
- [x] OLMo free-form handling: one regex flag routed to human review, no escalation

## Closeout
- [x] PREFERENCE_PHASE1_STATE_OF_RECORD.md complete
- [x] VALIDATION.md complete (incl. frozen-session certification)
- [x] Lab 38 handout updated to implemented status (§0 added)
- [x] course README/index updated (quick start + status + SPECIAL_TOPICS)
- [x] claim ledger suggestions use only allowed language (language-wall audit clean)
- [x] every headline number traces to immutable per-item rows
- [x] unresolved questions listed without silently becoming claims (STATE_OF_RECORD §"open questions")
