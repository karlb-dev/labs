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

## VM9 update (2026-07-29) — design decisions and progress on the blockers

**Correction to condition 3 below:** "Neither primary lens exists" was
wrong when written. The `Olmo-3.1-32B-Instruct` lens exists as TWO
independent registered fits (`a1-ownlens-regate-olmo31instruct-v1` draw A,
`b1-fitB-independent-lens-olmo31instruct-v1` draw B, row-cos 0.993,
selection Jaccard 0.806) and the pilot Instruct grid ran with them. The
addendum's risk list §5(ii) had it right: only the **3.1-Think** lens was
missing. It is being fitted this block (`a1_fit_olmo31think.py`, recipe
identical to the Instruct draw-A fit).

**Design decisions taken this block (PI delegated the open design calls):**

1. **Primary matched control = `dyn_energy_rank_matched_random`**
   (`matched_control.py`): per (item, layer, position) a random subspace
   matched EXACTLY — by construction, not rejection sampling — to the J
   arm's achieved effective rank and removed-energy fraction at that
   site, orthogonal to the protected dictionary rows, deterministically
   seeded. Rationale: the intervention is a span projection, and a
   subspace's complete geometric relation to the current state h is
   (rank, removed energy) — one principal angle. Matching both equates
   the dose entirely and leaves direction content as the only difference,
   which is exactly HP3's specificity claim. `dynJ_rotated` (dose
   collapses to the random regime) and `dyn_spectrum_matched_nonJ` (span
   projections are invariant to row spectrum) were rejected as primaries
   and stay named secondary arms. Dev-validation gates MC1–MC4 committed
   before any run (mechanical only — no behavioural gate, so the control
   cannot be tuned on outcomes).
2. **Capability-cohort predicate = `capable_generation`** (greedy decode
   produces a frozen-alias answer, MAX_NEW=8) on each confirmatory model;
   per-model difficulty windows are recorded but not part of the
   predicate — window-based cohorts would re-select the bank per model,
   which is the bias condition 1 exists to close (`g5_cohorts.py`).
3. **Capacity estimand ("does the data need recentering?"): yes —
   globally centered R² is the confirmatory capacity estimand**
   (`centered_variance_explained_excess` in `occupancy_v2.py`), with the
   raw energy share retained as a named sensitivity and occupancy
   (crossing) unaffected. This is the v1→v2 estimand repair already
   applied to Think (`r2-occupancy-think-v2`); Qwen and Instruct run the
   same corrected estimator this block from the existing configs
   (`r2_occupancy_qwen.yaml`, `r2_occupancy_instruct.yaml` — Instruct =
   3.1-Instruct at the pilot revision `ac0587e4`, its own lens; Qwen at
   `6a9e13bd` with the published n=1000 lens).

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
