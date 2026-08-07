# PAPER_ANALYSIS_PROTOCOL.md — frozen offline analysis protocol (P0)

**Committed before any campaign artifact is opened for synthesis.** This file,
with `EVIDENCE_TIER_RULES.md`, `CANONICAL_SCHEMA.md`,
`RETROSPECTIVE_ANALYSIS_RULES.md`, and `PHASE5_REOPEN_ROUTER.md`, is the P0
protocol freeze required by `paper_analysis.md` §3 and its addendum §1.
Governing documents: `paper_analysis.md` (operative plan) and
`paper_analysis_addendum.md` (amendments §2–§4 win on conflict). Foundation
provenance: `../ANALYSIS_FOUNDATION.json`.

## Scope and compute contract

- Branch `interp_jspace_paper_analysis`, created from tag
  `jspace-phase4-frozen-v1` (commit `ea635220b5c5a2c55a78ca1005f452f3aba2680b`).
- CPU only. No model weights load; no forward or backward pass; no lens
  fitting; no JVP work; no new intervention outcomes; no confirmatory data
  access beyond already-opened frozen records; Phase 5 stays closed.
- Campaign registries, registered output paths, frozen task partitions,
  states of record, release manifests, and raw Drive artifacts are
  **read-only**. Corrections to a frozen source require a separately
  governed erratum event; none may be smuggled in as analysis.
- All new outputs live under `interpretability/jspaces/phases/paper_analysis/analysis/`,
  carry source evidence IDs and hashes, and are labeled retrospective
  unless they merely reconstruct a frozen analysis.
- The analysis-first rule is absolute: **no paper sentence precedes its
  reconstructed number.**

## Predeclared synthesis questions (frozen)

- **SQ1 (claim survival):** Which original Lab 37 claims survive the
  complete repair, confirmation, mechanism, Phase 4, and side-study
  sequence?
- **SQ2 (causal-channel core):** Which causal effects survive output
  protection, span protection, exact rank/energy control, held-out
  replication, and independent-lens checks?
- **SQ3 (mechanism specificity):** What is uniquely supported about bridge
  content, answer-direction geometry, accessibility, and externalized
  state?
- **SQ4 (training-associated organization):** Which OLMo changes co-occur
  across capacity, mapped-token geometry, selected spans, and causal use,
  and which are unresolved because of capability or design?
- **SQ5 (instrument invariance):** How do operator similarity, task-row
  similarity, selected-ID stability, projector stability, capacity,
  aggregate causal effects, and bridge endpoints evolve across Qwen fit
  sizes?
- **SQ6 (transport applicability):** At which models, checkpoints, layers,
  and doses does the exact first-order tangent meet the registered gate,
  and how does this relate to the causal assay's actual scope?
- **SQ7 (paper route):** Does one coherent manuscript contain the
  contribution without turning methods boundaries into a distracting
  second thesis, or are two manuscripts cleaner?
- **SQ8 (Phase 5 value):** Which prospective experiment, if any, changes a
  central paper sentence rather than adding another descriptive branch?

## Predeclared amendments in force (addendum)

1. **Route B is the working default** (empirical Paper A + methods
   Paper B); the P6 decision must still be earned under the §9.5 rubric.
2. **The runtime-identity synthesis** (`p4-runtime-identity-synthesis-v1`)
   joins Paper B's spine as a named section.
3. **The H6↔Phase-3 estimand reconciliation is a registered A5
   sub-analysis, written before any drafting** (addendum §2.2): H6 bounds
   the lens-as-predictor at specified ε on a prospective ladder; Phase 3
   estimates span-removal effects against exact rank-and-energy-matched
   controls, whose in-situ validity evidence is the control behavior
   itself. Both estimands are stated formally; both papers cite the same
   reconciliation.
4. **The claim ledger is seeded from the six skeleton sentences at their
   frozen tiers** (addendum §2.3; sentences in
   `jspace_phase4/reports/PHASE4_TO_PAPER_ANALYSIS_HANDOFF.md`).
5. **Priority ladder:** A1 → A6 → A2 → A3 → A5 (incl. the reconciliation)
   → A4 → A7 → A8. P6 gates P7+. P4's exact reconstruction of every
   circulated headline number gates P8 drafting absolutely; a failed
   reconstruction triggers the stale-prose register and an erratum before
   any new document.
6. **Public-artifact sequencing** is a PI decision teed up at P6
   (AF/LW post → arXiv Paper A → Paper B + protocol release is the
   recommended path); nothing publishes before P9 passes.

## Phase order and gates

```text
P0 protocol freeze            gate: committed before paper-facing joins/figures
P1 inventory + stale prose    gate: every paper number has a source or is flagged
P2 evidence graph + ledger    gate: conclusion skeleton regenerable from ledger
P3 canonical schemas          gate: estimator/tier keys mandatory in every table
P4 exact reconstruction       gate: no route decision before headline audit done
P5 syntheses A1–A8            gate: outputs labeled retrospective + evidence IDs
P6 route decision             gate: written rubric + PI sign-off
P7 figure/table ledgers       gate: frozen before prose polish
P8 outlines + source updates  gate: every result sentence has an evidence edge
P9 independent audit          gate: claim/contradiction/objection/repro pass
P10 Phase 5 router            gate: go/no-go recorded, never an automatic plan
```

## Success criteria (addendum §3)

The phase is done when: P0 and P4 reports are registered; the claim ledger
and survival timeline exist with every row evidence-linked; the A5
reconciliation is written; the route decision is made under the rubric
with PI sign-off; both outlines exist with figure ledgers whose every
figure regenerates from schema'd data; and the P9 audit passes on the
drafts. Deliverable = convergence, not document volume.
