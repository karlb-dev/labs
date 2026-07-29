# READY_FOR_FREEZE — gate ledger for the J-space Part 2 confirmatory freeze

Generated at the end of the VM8 repair block. Every row names the evidence
id a reviewer can verify with
`jspace-part2 repro2 <id> --verify-only`.

**Bottom line: 5 of 8 gates pass; 3 conditions block the freeze, and all
three need weights that were not on this VM. No confirmatory cell has
run, and the partition has not been generated.**

---

## Gates

| gate | status | evidence | note |
|---|---|---|---|
| **G0** reproduction contract | **PASS** | `repro-contract-v2-acceptance-v1` | two items reproduce end-to-end from isolated worktrees at their recorded commits, with pinned constraints and post-run payload verification |
| **G1** solver | **PASS** | `r2-occupancy-think-v2`, `tests/test_occupancy_v2.py` | positive-support exhaustion, three named estimands, batch-split equivalence |
| **G2** lens stability | **PASS** (unchanged) | `b1-dict-stability-*` | disjoint-corpus fits agree (row-cos 0.993) |
| **G3** intervention invariants | **PASS** | `tests/test_protected_v2.py` | per-position protection verified on a real transformer; row-wise dose; phase gating provable from fire counts |
| **G4** positive control | **PASS for the pilot lens; PER-MODEL RERUN REQUIRED** | `r5-swap-control-*` | swap injection 0.76 vs 0.18. Must rerun with each confirmatory checkpoint's OWN lens before that model's cells count |
| **G5** task gate | **PASS** | `g5-item-manifest-v3` | 1052 items; 64 families with ≥3 capable items; 15 leakage exclusions applied; zero bridge leakage; all counterfactuals present |
| **G6** power | **PASS as an analysis; the ANSWER is negative** | `g6-power-sim-v3`, `g6-tailrate-power-by-model-v1`, `g6-mde-v1` | the binary primary is not affordable on the OLMo pair; the candidate states those hypotheses as estimates |
| **D2/HM1** lens choice | **RESOLVED — arm rejected** | `h7-synthesis-olmo3-think-v1` | context-J failed its committed dev gate; `meanJ_paper` is the sole primary |

## Decisions D1–D7 (PI addendum §3)

| id | decision | implemented in |
|---|---|---|
| D1 | split endpoint; HP1 as interaction + contrast; hurdle secondary; tail stratified to protected-answer items | candidate §3, §6 |
| D2 | run context-J before freeze; mean-J stays primary regardless | `h7-*` — run, arm rejected |
| D3 | Gemma = methods boundary case, no confirmatory causal cells | candidate §1 |
| D4 | tail threshold frozen at −1.0 nat, sensitivity at −0.5/−1.5/−2.0 | candidate §6.5 |
| D5 | expand then split at family level | 64 eligible families; `partition.py` built, tested, NOT run |
| D6 | run G5, no waiver | `g5-item-manifest-v3` |
| D7 | one paper, methods sections written extractable | the H7 and Gemma results are self-contained |

## Blocking conditions

1. **Cross-model capability cohorts are incomplete.** Only the anchor
   (`Olmo-3-32B-Think`) was scored; every other checkpoint reads
   `PENDING_WEIGHTS_NOT_ON_THIS_VM` in the manifest. A bank selected on
   one model's difficulty smuggles a sampling bias into every cross-model
   claim, so this must close before the freeze.
2. **The primary matched control does not exist yet.** The pilot's random
   arm is a *mechanics* control and is renamed `dynR_mechanics_control`
   throughout. HP3 needs a geometry-matched primary control, implemented
   and dev-validated. A null against an unmatched control is not evidence
   of specificity.
3. **Neither primary lens exists.** `Olmo-3.1-32B-Think` and
   `Olmo-3.1-32B-Instruct` both need their own fits (~5 h each). No pilot
   `Olmo-3-32B-Think` cell may impersonate a primary cell.

Not blocking but queued: corrected R2 on Qwen and Instruct (stage N7),
Gemma pre-cap target and target-layer sweep (leftover compute only).

## What the freeze commit does, and only this

1. rename `SCIENTIFIC_PREREGISTRATION_CANDIDATE.md` →
   `SCIENTIFIC_PREREGISTRATION.md`;
2. run `partition.build_partition(..., freeze_authorised=True)`;
3. store hashes and partition assignments — **no intervention outcomes**;
4. tag `jspace-part2-confirmatory-freeze-v1`.

No code change, no prose edit, nothing else in that commit.

## Registry state

61 evidence items, 47 live. Two Gemma analyses whose conclusions had been
reversed (`local-linearity-gemma4-31b-v1`,
`linearization-faithfulness-gemma4-31b-v1`) were still reading as LIVE
until the v2 registry refused their dangling supersede links; both are now
correctly superseded. `jspace-part2 registry-list --live` is authoritative.
