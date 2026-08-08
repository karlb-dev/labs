# PREFERENCE_PHASE2_PREREGISTRATION — Lab 38 Phase 2 (frozen)

Study `preference-phase2`; branch `interp_preference_phase2`. Governing
documents (precedence low→high): Phase 1 record (immutable) < Lab 38
handout < Phase 1 plan/addendum < `plans/preference_2_2.md` <
`plans/preference_2_2_addendum.md` (§B errata E1–E17 govern on conflict).
This preregistration inlines every constant the plan and addendum pin;
where it cites a section it adopts that section verbatim. Claim ceiling
(plan §8) binds every artifact including commit messages; the language
wall raises on violation.

## 1. Hypotheses (plan §3, adopted verbatim)

H-SURF, H-SEM, H-ENACT, H-CONTEXT, H-CANON (as repaired by addendum E1),
H-MECH, H-REPORT, H-XMODEL, H-DG (secondary). Result taxonomy: the ten
plan-§4 statuses, exact strings.

## 2. Banks of record (bank v3, FROZEN)

| pin | value |
|---|---|
| bank_version | `pref2_v3` |
| bank_content_hash | `4a3a5047b2bbaa1778f9907ca82be1ea9ddf874ffe5bf49aed19b4014c3f348b` |
| bank_jsonl_sha256 | `119af21538daf6ec69b59efffa9c23d91b58f45b3f4fe688bfc4d49a5947662e` |
| total rows | 18,320 |
| generator | `data/make_pref2_banks.py` (byte-deterministic; rebuild-verified) |

Counts: B-SURF 1,152 (E3 arithmetic; F-P1 2⁵ + F-SYM 2² per
template-skin, paraphrase-twin texts); **B-ARB3 4,608 = 12 × 24 × 16**
(24 incidentals 12/6/6 — raised from the plan's 16 by the power
simulation, deviation D1); B-MECH 4,800 (E5 verbatim: 3 anchors × [32×40
+ 8×40 reserved-transfer]); B-CANON 576; B-PC 480; B-PC-MECH 2,560 (4
difficulty variants frozen as text, E2); B-NC 640 (E10 families);
**RO-DISJOINT 2,688 = 12×24×8 + 3×(64+64 reserved)** (deviations D1/D2);
F-P1-CONT 384; B-DEV 432 (dev tier only). Splits: B-ARB3 12/6/6, B-MECH
and B-PC-MECH 16/8/8. Ladder guards, lexical balance, RO-overlap
(<0.40 Jaccard; measured max 0.135) audited clean
(`pref2-bank-v3-audit-v1`).

## 3. Codebook families (FROZEN)

`cbv3_932b4918f8`, selected tokenizer-only against
`allenai/Olmo-3.1-32B-Instruct@ac0587e4`: AR primary `ar0 QF3/JX8`,
`ar1 KP4/VM2`, `ar2 HB9/TX6`; AR reserved `arR DN5/WL8`; RO primary
`ro0 GS2/PK7`, `ro1 RV4/MJ6`, `ro2 BQ5/FZ6`; RO reserved `roR RK2/TN7`.
All pairs equal token counts with distinct first AND final tokens on all
four frozen tokenizers; AR/RO alphabets disjoint; reserved families
appear only in holdout-transfer rows (bank-test enforced). Primary pairs
rotate at the incidental level balanced within splits. On-model neutral
carrier gap gate at S2/S5: |gap| < 0.10 nats inside the exact rendered
carrier per model; failure ⇒ `STOP_P(model)`.

## 4. Formats

Primary `F-SYM` (plan §11 shape, exact strings in `formats.py`):
sequential records, no display labels, no repeated reply list, constant
sentinel `Context complete.`; RO surface per §20 + E6 with constant
preamble ending `Survey context complete.`; F-P1 clone for B-SURF and the
continuity arm; F-COMMIT dev-only. Format gate (E14): F-SYM must reach
strict parse ≥ 0.98 on B-DEV PC rows with NC asymmetry under floor;
frozen fallback is F-P1, full stop; no third format post-freeze.

## 5. Endpoints and scoring

Per row: `margin_full_a_minus_b` (complete opaque code sequences,
teacher-forced, SINGLE-ROW, `use_cache=False`, float32 log_softmax before
gather), `margin_first_a_minus_b` (first target tokens), strict
generated choice (greedy ≤ 8 new tokens, strict parser
`strict_exact_code_v1`, no rescue), binding per addendum E7 (enacted+
valid only; never on intervened rows; environment-only in B-MECH/
B-PC-MECH; RO never executes). Generation may batch only after the
per-model batch-invariance audit; scoring stays single-row. These three
endpoints are related measurements, never independent replications
(plan §0.3).

## 6. Statistics (addendum E pins, verbatim)

Primary p per scenario-endpoint: incidental-level EXACT sign-flip (2^24
enumerated ≤ 2^20 → seeded 10k Monte Carlo above; B-ARB3 incidental
means use n=24 → Monte Carlo; B-MECH slopes n=32 → Monte Carlo); CI:
hierarchical bootstrap 10,000 (incidentals then cells); Holm within F1
(12 margins), F2 (12 strict, conditional on F1), F3 (3 slopes), each
scenario-level mechanism family (M1–M7), and each coupling family.
Floors (E10): semantic-margin floor `max(0.15 nats, 2 × NC(f1–f2) p95)`;
strict SESOI 0.10; context-slope floor `2 × NC(f4) p95 slope`;
carrier/code floor from NC f3; decoder/causal floors from f1–f2 sham
runs. NC-alarm comparisons use the STATIC floor components (0.15 nats;
0.05 nats/unit) so an alarming NC cannot inflate its own trigger.
Criteria: plan §25 (ten SEMANTIC_MARGIN), §26 (ten ENACTED_CHOICE), §27
(CONTEXTUAL_VALUE with holdout rank-corr ≥ 0.70) adopted verbatim as
implemented in `behavioral_analysis.py` at the freeze commit.

Power of record (`pref2-power-simulation-v1`): margin@0.25 nats 1.00;
strict@0.10 0.86 (after D1); slope@0.10 1.00; coupling strict-shift 0.21
⇒ **coupling primary = RO full-target margin contrast** (addendum G
fallback, pinned HERE, pre-outcome; strict-report and five-point
endpoints report as secondary descriptors; `CHOICE_REPORT_COUPLED`
requires the margin primary plus ≥ 1 same-direction non-margin
endpoint). Stress-tier (2× variance) powers are sensitivity only.

## 7. Mechanism contract (Part VI + addendum F)

Direction: `d = unit(E[h_Afavor − h_Bfavor])` over matched pairs within
(incidental, order, code map, menu paraphrase, |s|, codebook pair),
|s| ∈ {1,2} pooled, TRAIN incidentals only; ridge-on-strength is
sensitivity only, never the gate. Sites (E6): AR `context_end,
option_a_end, option_b_end, menu_end, response_instruction_start,
final_prompt_token`; RO `ro_*` equivalents; `*_end` sites are
end-anchored tokens, `*_start` start-anchored; `menu_end` shares the
second record's final token by construction (deviation D7). Depths:
relative {0.20, 0.35, 0.50, 0.65, 0.80, 0.95} resolved per model with
the block-to-stream convention recorded (`hidden_states[k]`; hook on
`blocks[k-1]` output; prefill-only patch). Primary intervention sites
`context_end`, `menu_end`; `final_prompt_token` is the direct-output
positive control only.

Precheck (§36 + E11): decoder = fixed projection on train-fitted d;
validation Pearson r ≥ 0.40 vs signed strength; AUC ≥ 0.70 on |s| ≥ 1;
both above the 95th percentile of 1,000 within-incidental strength-label
permutations (direction refit per permutation); split-half cosine
median ≥ 0.60 with p10 > 0 over 100 train-incidental bootstraps; sign
stable across codebook and paraphrase strata; neutral projection
predicts neutral margin above a 200-random-direction band and survives
the reserved codebook; splits ≥ 16/8/8. Site/depth selection (§37):
validation-only weighted score {corr .30, AUC-gap .20, split-half .20,
neutral-corr .20, cross-codebook .10} with frozen clip transforms,
QUANTIZED to 0.01 (deviation D6) before the upstream-then-shallow
tie-break. Dose (§38 + E13): grid ±{0.5, 1, 2} train-projection SD;
guard set = the 16 Phase 1 guard prompts VERBATIM (frozen typo included)
+ 8 new in-domain neutral menu-shaped prompts (frozen in
`mechanism_run.GUARD_PROMPTS_P2`); mean KL < 0.05 nats, max < 0.20 over
first 32 continuation tokens, median generic next-token shift < 0.05;
dose frozen on validation before holdout. Assays (§39): matched donor
patching (E12: same incidental/order/cmap/menu-paraphrase, |s|=2 primary
|s|=1 sensitivity, self-donor no-op tolerance, wrong-scenario donor,
donor-strength monotonicity secondary), ±d addition, α=1 removal (0.5/
1.5 sensitivity), propagation (downstream projection ≥ 0.5 × injected-
site effect at menu_end; self-patch within tolerance), final-token
positive control AFTER upstream primaries. Controls (§40): d_position,
d_code, d_format, d_context_text_only, d_semantic_identity,
d_wrong_scenario, 8 norm-matched randoms, self-patch, wrong-site,
heldout codebook + paraphrase. Holm family per scenario: M1 patch, M2
±d, M3 removal, M4 heldout codebook, M5 wrong-scenario, M6 code-
direction, M7 propagation; `MARGIN_HANDLE` requires (M1 or M2) ∧ M4 ∧
(M5 and M6 specificity: each < 50% of primary) ∧ M7; enacted causal
claims additionally require strict flips or a holdout strict-rate shift.
Output-adjacency audit (§41) precedes any mechanism language; holdout
opens exactly once per scenario.

## 8. B-PC-MECH two-stage freeze (E2)

Difficulty variants d1–d4 frozen as text in the bank. GPU S3 runs
train+validation incidentals only, all variants, development tier;
select the variant whose validation |neutral margin| lands in
[0.5, 3.0] nats (prefer the mid-band variant on ties); a single-field
freeze AMENDMENT pins it before S4; holdout untouched during
calibration. No variant in band ⇒ STOP and ask (addendum L2).
Saturation exception (addendum F): only if the power simulation shows
< 0.50 probability of a single strict flip at the maximum guard-safe
dose given the calibrated margin.

## 9. Human gates (addendum §I) and licenses

H1/H2/H4 sheets pass at `agent_dual_code_provisional` (two passes,
work-phase boundary between, 2 preserved disagreements; authorship-
limited blinding and Phase 1 exposure recorded). H3 canonicality
composite (E1): frozen sheet over 6 axes + 12 B-ARB3 heldout sign
targets, 10,000-permutation test, magnitude rank-corr secondary;
because the coder has read the Phase 1 record, H-CANON reports at
**exploratory tier** unless the PI ratifies the sheet (E17 demotion
exercised as deviation D4). H5: the PI's standing session instruction
of 2026-08-08 (recorded verbatim in `reviews/PHASE2_FREEZE_APPROVAL.md`)
authorizes end-to-end execution; every downstream artifact carries the
provisional license until PI ratings replace it.

## 10. Model cells (plan §49; revisions FROZEN)

| key | model | revision | notes |
|---|---|---|---|
| olmo32b (primary) | allenai/Olmo-3.1-32B-Instruct | `ac0587e4a7744a551c059d8cd17ba220bc940dae` | |
| olmo7b | allenai/Olmo-3-7B-Instruct | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` | |
| qwen | Qwen/Qwen3.6-27B | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | empty-think contract (D8) |
| gemma | google/gemma-4-31B-it | `842da3794eaa0b77d5f08bae87a17459d91ff475` | no-system shim; post-softcap logits; own NC floors |

Order 32B → 7B → Qwen → Gemma; per-model port gates plan §50 (15 items);
bf16 only; single-model staging with eviction (C2). Completeness =
OLMo-32B spine (E17); cross-family cells drop only in the frozen order
(DG; F-P1 expansion; five-point RO expansion; B-CODE; cross-model
B-ARB3 map Gemma→Qwen→7B; Gemma mech; Qwen mech; 7B mech; Gemma B-MECH;
Qwen B-MECH) with logged `STOP_P`/`STOP_BUDGET` events. Never-drop: the
32B spine items (plan §75 as E17-amended).

## 11. Stop rules and halts

Plan §60 stop rules verbatim (STOP_I/F/SURF/MARGIN/CHOICE/PCMECH/
DIRECT/BEHAVIOR/COUPLED/P/BUDGET). Addendum L halts: VRAM < 80 GB at a
big-model stage; no PC-MECH variant in band; STOP_I on primary or both
formats failing; any NC family clearing a floor, wrong-branch execution,
or replay/resume parity failure; freeze-gate changes beyond the single
E2 amendment; and the never-permitted actions (refusal ablation, DG-SAFE
generation, pooled cross-scenario direction, early holdout, enlarging a
frozen bank after outcomes).

## 12. Captures and records

Captures single-row at the six sites × six depths, native bf16, sharded
per model/bank/site/depth-group with per-shard SHA256 and a committed
manifest (the first verifiable capture seal — C2); resume verifies shard
hashes; per-item JSONL rows are immutable and append-only with atomic
resume cursors, config-hash resume refusal, ≤ 10-minute loss windows,
and Drive mirroring. Registry events per stage with the closed
scientific-tier vocabulary.

## 13. Seeds and sign anchors (E15)

All post-freeze random assignments derive from the freeze-commit sha by
the Phase 1 rule (first 8 hex chars → int base seed), through
`stable_seed(...)` with the derived base. Analysis sign anchors per
scenario = `sign_anchor_for(freeze_sha, scenario_id)`; anchors are
bookkeeping for signed displays only, never causal variables, and are
recorded in `reports/dev/sign_anchors.json` immediately after the freeze
commit.

## 14. Deviations at freeze

See `DEVIATIONS.md` D1–D8. All are raises, operationalizations, or
license demotions; none relaxes a floor, a gate, or the ceiling.
