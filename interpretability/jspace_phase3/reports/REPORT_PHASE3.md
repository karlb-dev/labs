# J-space Phase 3 — living report (mechanism, generalization, paper hardening)

Governing docs: `../reviews/jspace_lab_nextsteps_3_1.md` + addendum (§4/§5
govern). Phase 2 is frozen (`jspace-part2-complete-v1`); its results enter
here only as immutable `phase2-confirmatory` inputs. Every claim below
carries a tier watermark; nothing exists outside
`reports/evidence_events.jsonl`.

---

## VM10 (2026-07-29) — Phase 3 opened

### §4.1a — Protected-output selection pressure vs J damage — tier: PHASE3-DEVELOPMENT

`p3-overlap-mining-v1` · figure `p3f01` · inputs: the six frozen Phase 2
N6 parquets (3 models × {confirmatory, replication}), read-only.

The v2 ablator logs, for every position, how many PROTECTED (clean top-k)
token rows *would have entered* the J top-k and were excluded. Per item,
`blocked_rate` = blocked total / positions — selection pressure toward
the protected output region. This is the cheapest available leakage
proxy; it is NOT geometric span overlap (that is §4.1b, GPU).

**Result 1 — pressure does not explain damage, in any model.** Spearman
ρ(blocked_rate, Δlp_J): Think +0.11 [−0.04, +0.25], Instruct +0.00
[−0.13, +0.13], Qwen +0.11 [−0.04, +0.25] (family-clustered bootstrap
CIs). Tail vs non-tail blocked rates indistinguishable (MW p = 0.11 /
0.89 / 0.26). Where nonzero, the sign is *opposite* to the leakage story
(more pressure ↔ slightly less damage). Same picture against the
J-minus-matched-control specific effect.

**Result 2 — but the pressure itself is a 12× model property.** Median
blocked rate per position: Think **1.33**, Instruct **1.22**, Qwen
**0.10**. Under the identical protection contract, the OLMo lenses
constantly try to select clean-top-10 token rows (>1 per position) while
Qwen's essentially never do. OLMo's J-space is heavily output-adjacent;
Qwen's is not — a new, sharp form of the Phase 2 two-cluster picture,
and a direct motivation for the span-safe arm *specifically on OLMo*:
with that much pressure, the next-best rows selected after masking may
still overlap the protected span geometrically.

**Result 3 — the accessibility signs flip between clusters.**
ρ(clean_first_rank, Δlp_J): Think **−0.30** [−0.41, −0.17], Instruct
**−0.31** [−0.40, −0.21] — less output-locked answers lose more, the
Phase 2 accessibility organization, now CI-clean on both primaries.
Qwen: **+0.24** [+0.05, +0.40] — the opposite; damage concentrates on
items whose answers were already confidently ranked. Consistent with
composition-dependence rather than marginal accessibility.

**Threat-model calibration (what §4.1 was for):** the *within-model*
leakage account gets no prior support (pressure uncorrelated with
damage), so P3-2's decision weight shifts to the *geometric* question —
does span-safe protection change the OLMo arms, where pressure is large?
`meanJ_span_safe` remains a Phase 3 primary-family member (P3-P2);
§4.1b (principal angles, answer-direction survival) runs first thing in
the Think GPU window with `record_ids=True`.

Caveat carried: blocked_rate is a selection-pressure proxy; a uniform
geometric leak shared by every item would be invisible to Result 1 and
is exactly what §4.1b + the span-safe arm measure.
