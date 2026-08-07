# Phase 3 release-audit plan

Status: active. Governing sources are `jspace_lab_nextsteps_4_1.md` Part I and
its addendum §§3–5. The frozen Phase 3 preregistration, raw outcome parquets,
and historical evidence events are immutable.

## Snapshot boundary

- Reviewer-inspected head: `09f987243f82c281431818e1d10ad92b8dad0169`.
- Audit-entry head: `660047d` (`jspace-phase3-pre-release-audit-v1`).
- Freeze boundary: `df4d45a` (`jspace-phase3-freeze-v1`).
- Stage-0 snapshot hashes the report, handout, registry, six primary and
  replication parquets, three primary lenses, and the OLMo base lens; it also
  exports the resolved live-evidence inventory.
- Snapshot and audit outputs are new evidence. No frozen outcome is rewritten.

## Audit ledger and gates

1. **Protected-answer protocol conformance (P0).** Measure Qwen baseline-only
   clean ranks for both partitions without reading intervention columns. Join
   rank metadata to immutable outcomes and report exact-alias, any-alias, and
   all-item P3-P2 views across the frozen threshold curve. Gate: retain the
   narrow protected-stratum wording only if its estimate and interval remain
   clear; otherwise preserve only the all-items claim.
2. **Random-control seed contract (P0).** Replace prospective scientific
   `hash(item_id)` seeding with SHA-256-derived stable seeds. Audit a balanced
   40+40 Qwen subset at seeds 11, 101, 1009, 4242, and 31337. Gate:
   `ROBUST`, `SEED-SENSITIVE BUT BOUNDED`, or `DECISION-SENSITIVE`; never
   reconstruct an unrecorded historical Python salt.
3. **Inference-object closeout (P0).** Exactly enumerate all 131,072 P3-P1
   family sign patterns, invert the shifted test, compute a correctly named
   wild-cluster percentile-t interval, and retain the normal approximation
   under its correct label. Use plus-one Monte Carlo p-values for P3-P2/P3-P3.
   The frozen randomization-test decision does not change.
4. **Actual Phase 3 reproduction (P0).** Rename the historical ladder
   N8-P2-L1/L2/L3 in current documentation. Run N8-P3-L1 blind analysis,
   stable-seed sentinels for every primary model, and one repaired full Qwen
   cell. Exact baseline/J agreement plus bounded seed-ensemble controls is an
   acceptable release outcome.
5. **Bridge, scoring, cohort, metadata, and provenance closeout (P1).** Audit
   true/distractor protection geometry and counterfactual preference; run
   alias/cohort sensitivities and boundary-safe generation regrading; append
   the replication-tier correction event; build the release manifest and
   final state-of-record report/handout.

## Execution order

CPU work runs whenever it cannot contend with model inference. On the GPU
block, protected-answer rank measurement runs first, then the seed ensemble,
then N8-P3 sentinels/full cell. OLMo lineage work begins only after the
paper-gating audit jobs are banked.

Every model-scale command must pass the same-process CUDA hard gate and run on
the visible NVIDIA GPU. If the sandbox hides CUDA, the command is relaunched
with host/unsandboxed access; CPU fallback is prohibited. Long producers
checkpoint at least every ten minutes, emit progress at least every five
minutes, and retain a durable Drive state such that reclamation loses no more
than thirty minutes.

## Completion

The release audit is complete only when every discrepancy is corrected,
bounded, or explicitly reflected in a narrower claim; N8-P3-L1/L2/L3 pass;
the state-of-record artifacts are built; and the immutable
`jspace-phase3-complete-v1` tag is pushed. Phase 4 confirmatory work cannot
start before a separate Phase 4 freeze.

