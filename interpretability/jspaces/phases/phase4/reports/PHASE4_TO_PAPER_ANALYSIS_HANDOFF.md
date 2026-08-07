# Phase 4 → paper-analysis handoff

**FROZEN-EVIDENCE SYNTHESIS ENTRY POINT — READ-ONLY CAMPAIGN INPUTS**

Created 2026-08-04 after the `jspace-phase4-frozen-v1` tag.

## Frozen boundary

- Tag: `jspace-phase4-frozen-v1` at commit resolved via `git rev-list -n1 jspace-phase4-frozen-v1` (the freeze-record head of `interp_jspace_part2`)
  (equals `origin/interp_jspace_part2`).
- Release manifest:
  `interpretability/jspaces/phases/phase4/release/PHASE4_RELEASE_MANIFEST.json`,
  payload SHA-256 `be481dcfca634136d09c648379fddbcb9fd81989217e653abacac3705ac0be73`.
- Phase 4 registry SHA-256: `58fd81b6af1956e89a352f195cee546e8f0bd61d27d3e10949b7b1a846694f54`
  (84+ rows; see manifest for exact counts).
- Gemma registry SHA-256: `c7e76cac59bbfcf30f5d29edfa5c747bdae9aea46523fad9a8bd49db9cca124b` (24 rows; frozen
  sidelines-2 prefix `2a144bcf0e7be0ac4307f7e2a2984c1879340b9a7e9278d1
  0143d122a14fd30a`).
- OLMo registry SHA-256: `d15391ac733d785f583c62e812688efc56268c70ad9aff430b50bb49741196a6` (41 rows; frozen
  sidelines-2 prefix `0a8973e01d562a82fa88da650ab8597c140050f6caf46c8b
  bd72e2b58acffb58`).
- Phase 2 / Phase 3 registries: unchanged since
  `jspace-phase3-complete-v1` at
  `9e0672b8748b8c53f0bd853dfadda9bc795fd524`.
- Study-2 bundle hashes: Gemma
  `9ef48b8ab1d99d52a756ddea1e285a9d61e781fd054cbf92702bfe81be56f5b0`;
  OLMo
  `c213dc74aa78dcd6613c8bd1562dd07d2e2a0345409ee6da585001693d8e6b1c`.

## Drive roots and logical URI mappings

| Logical | Physical |
|---|---|
| `artifact://phase4/...` | `/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/...` |
| `artifact://phase3/...` | `.../phase3_20260729/...` |
| `artifact://phase2/...` | `.../part2_20260727/...` |
| `gemma-run://...` (Study 2) | `.../gemma_transport_2_20260803/...` |
| `olmo-artifact://...` (Study 2) | `.../olmo_lineage_2_20260803/...` |
| Gemma Study-1 root | `.../gemma_transport_20260802/...` |
| OLMo Study-1 root | `.../olmo_lineage_20260801/...` |

Resolution code: `jspace_phase4/paths4.py` (env overrides
`JSPACE4_RUN_ROOT`, `JSPACE_DRIVE_ROOT`, `JSPACE3_RUN_ROOT`,
`JSPACE_PART2_RUN_ROOT`).

## Claim ledgers and states of record

- `interpretability/jspaces/phases/phase4/reports/PHASE4_STATE_OF_RECORD.md`
- `interpretability/jspaces/phases/phase4/release/PHASE4_CLAIM_LEDGER.md`
- `interpretability/jspaces/phases/phase4/release/PHASE4_KNOWN_LIMITATIONS.md`
- `interpretability/jspaces/sidelines/gemma/release/GEMMA_TRANSPORT_STATE_OF_RECORD_V2.md`
  + `gemma_transport_claim_ledger_v2.md`
- `interpretability/jspaces/sidelines/olmo/reports/OLMO_LINEAGE_STATE_OF_RECORD_V2.md`
  + `OLMO_LINEAGE_CLAIMS_TABLE_V2.md`
- Runtime-identity methods object:
  `interpretability/jspaces/phases/phase4/reports/PHASE4_RUNTIME_IDENTITY_SYNTHESIS.md`
  (`p4-runtime-identity-synthesis-v1`) — joins Paper B's spine per the
  analysis addendum.

## Six skeleton sentences at frozen tiers (seed for the master claim ledger)

1. Channels exist; content-not-dose — confirmatory/replicated (Phase 2/3
   anchor).
2. Training-dependent occupancy — development; stage localization open,
   missing-not-zero (`p4-import-olmo-lineage-study2-v1`).
3. Qwen bridge route — confirmatory for protection/lesion/preference;
   substitution semantics development.
4. Externalization — development; Bank-S pattern with the wedge unresolved;
   Bank-W pair unpowered at current support (0.7788/16; first pass 18).
5. Transport premise is model-dependent and gated — two-part: Gemma closed
   finite-scale at tested scope under the 0.07870368901355948 ceiling;
   OLMo in-band failed at tested epsilon with a Think-only L56 late anchor.
6. Operator convergence ≠ instrument invariance — scoped Q-L4: aggregates
   fit-stable; selections and one mechanism endpoint not; no canonical
   sparse lens.

## Existing paper/handout sources

- `interpretability/jspaces/phases/phase4/paper/PAPER_CONCLUSION_SKELETON.md`
- `interpretability/jspaces/phases/phase4/paper/PHASE4_METHODS_DECISION_RECORD.md`
- `interpretability/jspaces/phases/phase4/paper/PHASE5_HORIZON.md`
- `interpretability/jspaces/phases/phase4/reports/handout/` (governed boundary
  lifted at the tag; regeneration belongs to the analysis phase)
- `interpretability/jspaces/sidelines/gemma/reports/handout/`
  (`gemma_transport_development.tex/.pdf`)
- `interpretability/jspaces/sidelines/olmo/reports/paper/`
  (`olmo_lineage_parallel_phase.tex/.pdf`)
- Course materials referencing the campaign under `interpretability/`
  (COURSE.md, ADVANCED_COURSE.MD, labs/) — course-facing, not
  claim-bearing.

## Known stale passages to correct in the analysis phase

1. The Phase 4 handout/TeX under `reports/handout/` still reflects the
   pre-canonical boundary (it predates Q-L4 registration by design).
2. Any Gemma course/handout text that labels the autopsy as "stopped at a
   blocker" without the Study-2 calibrated license.
3. Any OLMo paper section that describes the SFT/DPO wedge or H6 as queued
   rather than closed capability-gated / in-band-failed.
4. Any text implying a canonical A1000 Qwen lens (none may survive).
5. Any text treating Bank-W as merely pending rather than
   capability/power blocked with exact 16/20 and 0.7788/18 boundaries.

The analysis phase's stale-prose detector (paper_analysis.md P1) should
sweep for these automatically; the claim ledger rows above are the
ground truth.

## Allowed offline analyses

Per `paper_analysis.md` §3.2 (predeclared retrospective analyses only) with
the addendum's priority ladder A1 → A6 → A2 → A3 → A5 (including the
H6↔Phase-3 estimand reconciliation) → A4 → A7 → A8. CPU only; nothing that
runs a forward or backward pass; new GPU ideas go to the Phase 5 horizon
list.

## Forbidden post-freeze reinterpretations

Everything in `release/PHASE4_CLAIM_LEDGER.md` §"Forbidden terminal
shortcuts", plus: no reopening Q-L4, no restoring a retired primary, no
converting missing/gated cells to zeros, no new bank/family authoring, no
handout regeneration before the route decision, and no writes into any
campaign registry or registered output path.

## Phase 5 reopen router

`interpretability/jspaces/phases/phase4/paper/PHASE5_HORIZON.md` holds the candidate
branches (Qwen consensus/cross-fit successor; Bank-W ≥18-family redesign;
capability-compatible stage assay; Gemma mechanism separation; prospective
site-dose transport archive). A Phase 5 study may open only from a written
router applied to the offline synthesis, with a new preregistration, and
imports Phase 4 solely by tag + release manifest.

## Exact entry command

```text
Read paper_analysis.md and begin the frozen-evidence synthesis on a new
branch from jspace-phase4-frozen-v1:

  cd /content/labs
  git fetch origin --tags
  git switch -c interp_jspace_paper_analysis jspace-phase4-frozen-v1
```
