# RECONSTRUCTION_AUDIT.md — P4 output (gate for P6/P8)

Assembled from six verdict tables (`tables/recon_*.csv`; builders `scripts/reconstruct_*.py`, each deterministic and rerunnable against the read-only registered sources). Status vocabulary per `protocol/CANONICAL_SCHEMA.md` §7.2.

**286 targets: 285 verified (236 byte-identical, 45 render-diff, 4 within-tolerance), 1 failed, 0 not reconstructable.**

## Tally by domain

| domain      |   byte_identical |   failed |   numerically_identical_render_diff |   numerically_within_frozen_tolerance |
|:------------|-----------------:|---------:|------------------------------------:|--------------------------------------:|
| gemma       |               43 |        0 |                                   2 |                                     0 |
| olmo        |              138 |        0 |                                   1 |                                     0 |
| paper_draft |                1 |        1 |                                  25 |                                     4 |
| phase2      |                7 |        0 |                                   3 |                                     0 |
| phase3      |               39 |        0 |                                   0 |                                     0 |
| qwen_ladder |                8 |        0 |                                  14 |                                     0 |

## Highlights

- **Phase 2** (10): HP1/HP3 confirmatory + replication byte-identical via full frozen-pipeline rerun (payload SHAs `5df4ace5…`/`f5367e5a…` reproduced; seed 4242; env bit-exact). Replication P-HP1 runs on the disclosed fallback population (322 rows/32 families), not the confirmatory intersection cohort.
- **Phase 3** (39): every headline recomputed from item rows bit-for-bit — exact 2^17 enumerations, all seeded MC/bootstrap distributions sha-matched. Only byte-level divergence anywhere: a BLAS dgemv last-ulp accumulation pattern in four null ARRAYS (every derived scalar identical). Two frozen-record subtleties documented: the P3-P3 headline mixes estimate source (inference audit) with CI source (geometry-v2 100k bootstrap), and a 1-ulp summation fork exists between randomization and bootstrap routes for two estimates.
- **Qwen ladder** (22): structural q50/q05 monotone (0.99522→0.99771→0.99870), selected-ID Jaccard exactly 7/13 at all three boundaries, projector 0.6748→0.7098 (floor 0.85), bridge rescue oscillates (−0.215 PASS/+0.559 FAIL/−0.294 FAIL); margin labels recomputed 100%; prompt-323 norms recomputed from the released 6.6 GB tensor bit-identically; Q-L4 reproduced from the frozen truth table (bridge-rescue row binds first). Drive registry mirror is a stale 69/82-line byte-prefix; repo registry (hash-matching FREEZE_HANDOFF) authoritative.
- **OLMo** (139): capacity/geometry/trajectory/wedge/H6/Bank-W all byte-identical incl. 240/240 bootstrap CIs with sha-matched distributions and a byte-identical CPU rerun of the 5000-sim power study (0.7788 at 16; n=17 → 0.7922; **18 strictly first passing**). The `p4-bank-w-capability-joint-imported-dev-v1` outputs registered under a defunct `/content/labs_phase4_4/` root recover exactly from the repo copy (sha `ffe1ca8b…`).
- **Gemma** (45): Stage-1 five-layer decisions, ceiling rule (float64 identity 3·q99 == 0.07870368901355948), batch q99s, bootstrap interval, floor ratio, G2.2 branch — all byte-identical incl. bit-exact 5000-draw bootstraps; selected-slot tensors recomputed byte-equal on CPU. The historical all-slot batch tensor is not released: the 0.00246 error rests on hash-verified cross-artifact consistency (recorded, accepted).
- **Paper draft sweep** (31): 30 verify; **1 genuine error** (below).

## Figure identity classes

Every figure regenerated this phase (`claim_survival_timeline`, `cross_model_evidence_matrix`, `qwen_multilevel_convergence`) derives from committed verified tables — class `numerically_identical_render_diff` or better by construction. Frozen-era registered figures were hash-verified, not re-rendered (their source tables reconstruct; render identity not claimed).

## P4 gate disposition

Zero campaign headline numbers failed reconstruction; zero were not reconstructable from released data. The single `failed` row is a *draft prose* error (not a frozen number), quarantined in the unsupported-number register. **The route decision (P6) and drafting (P8) are unblocked.**
