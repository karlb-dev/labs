# PAPER_ANALYSIS_FINAL.md — offline paper-analysis phase closeout

Completed 2026-08-05 on branch `interp_jspace_paper_analysis` from
`jspace-phase4-frozen-v1`. CPU-only throughout; zero forward/backward
passes; every campaign registry and registered output untouched
(read-only verified). Governing plan: `paper_analysis.md` + addendum;
protocol frozen at P0 before any artifact was opened.

## Success criteria (addendum §3) — all met

| Criterion | State |
|---|---|
| P0 protocol + P4 reconstruction reports registered | ✅ `protocol/` (5 files, committed first); `RECONSTRUCTION_AUDIT.md` — **286 targets, 285 verified, 0 frozen-number failures, 0 not-reconstructable** (Phase 2 + Phase 3 byte-identical incl. payload SHAs and 2^17 enumerations; OLMo 138/139; Gemma 43/45; Qwen 22/22) |
| Claim ledger + survival timeline, every row evidence-linked | ✅ C1–C7 + W1–W10 (`master_claim_ledger.parquet`, 37 edges, all ids asserted live); timeline figure; no Lab-37 claim survived verbatim |
| A5 reconciliation written before drafting | ✅ `A5_H6_PHASE3_RECONCILIATION.md`, registered `analysis-h6-phase3-reconciliation-v1` |
| Route decision under the rubric | ✅ **Route B** (`PAPER_ROUTE_DECISION.md`) — *pending PI ratification* |
| Both outlines + figure ledgers, figures regenerate from schema'd data | ✅ `paper_routes/*` (E1–E6/TA1–TA6; M1–M6/TB1–TB6); P9 confirmed byte-identical regeneration of every figure/table from committed inputs |
| P9 audit passes | ✅ Claim/contradiction/objection: PASS-with-findings (12 findings, all applied same-session — see below); reproduction: **PASS** |

## The scientific state, in six verified sentences

1. **C1 (confirmatory + replicated, Qwen):** span-safe J-ablation
   beats an exact rank/energy-matched control (+0.2788→+0.2966 tail
   diff; +0.095833→+0.102083 tail excess, p=1/100001) — the campaign's
   only confirmatory+replicated cell, byte-identically reconstructed.
2. **C2 (development):** OLMo capacity is conserved while dictionary
   geometry and Bank-S causal use reorganize at the first Think
   transition; the SFT/DPO wedge is capability-gated (missing, not
   zero).
3. **C3 (confirmatory, unreplicated + development):** the Qwen bridge
   rescue (+0.431367) stands against its measured-geometry falsifier;
   substitution semantics remain development.
4. **C4 (gated):** externalization is open — blocked at 16/20 and
   unpowered at 0.7788/16 (18 strictly first passing, rerun
   byte-identical).
5. **C5 (methods):** transport licensing is model/checkpoint/layer/
   dose-specific — one licensed cell campaign-wide (Think L56@ε=0.10);
   Gemma's five-layer mismatch is 17–68× the frozen calibrated ceiling;
   none of it bounds the paired ablations (reconciliation).
6. **C6+C7 (methods):** operator convergence ≠ instrument invariance
   (Q-L4: Jaccard exactly 7/13 at every boundary; rescue oscillates;
   the rescue row binds first), and version-level pinning does not pin
   backward semantics.

## P9 findings disposition (all closed this session)

F1 wrong N8 eids in the draft → corrected to the `p3-n8-p3-*` ladder
ids. F2 unsourced "12–52×" → replaced with the derivable "17–68× above
the frozen ceiling" at all seven sites (incl. figure title and Gemma
handout; PDF rebuilt). F3 two remaining pre-Study-2 present-tense
passages in the Gemma handout → date-qualified with the Study-2
resolution. F4/F5 A2 tail-range and control-scoping imprecision →
rewritten with per-arm registered values and the two label-vs-span
ratios explicitly disambiguated. F6 rounded values as hard bounds →
stored-repr forms. F7 stale-register U1–U4 → annotated superseded by
retirement. F8 occupancy range + Claude-band comparison → corrected/
marked open. F9 registry-pointer nits → fixed. F10 dual float reprs →
documented render-diff, no action. F11 two prompt-323 maxima →
distinguishing clause added to Paper B outline. F12 → recorded below.

## Deliverables map

Foundation `ANALYSIS_FOUNDATION.json` · protocol (5) · manifests (4) ·
data (8 parquets) · scripts (12, all deterministic CPU) · reports (18
incl. this) · paper_routes (10) · figures (5 sets) · tables (12) —
all under `interpretability/jspaces/phases/paper_analysis/analysis/`, mirrored to Drive
`special-lab-1/paper_analysis_20260804/`.

## Open items for the PI (nothing else blocks)

1. Ratify the Route B decision, the AF/LW → arXiv-A → B sequencing,
   and the Phase 5 NO-GO (all marked pending).
2. Paper A body drafting: outline §§5–7 sections are TODO in the TeX
   (F12) — they re-enter claim audit when written; the claim table
   already fixes their licensed sentences.
3. The unsupported-number register ships with the papers (now: 1
   corrected draft error + documented render-diffs).

## Closing sentence (plan §18, earned)

> The campaign has enough evidence to write before it has enough
> justification to run again. Phase 5 opens only if a future router
> identifies one prospectively testable result whose expected
> information gain exceeds the value of freezing, releasing, and
> publishing the completed work — and this analysis found none.
