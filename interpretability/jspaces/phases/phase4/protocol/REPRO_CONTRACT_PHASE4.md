# Phase 4 reproduction contract

Status: binding engineering contract; scientific preregistration is still a
candidate.

1. A result exists only if it has a Phase 4 evidence event, exact command,
   clean code commit, immutable input manifest, and hashes for every output.
2. Phase 2/3 results may only enter through `evidence_imported`; the import
   pins the source study, source evidence ID, source commit/tag, source
   registry hash, and source-output hashes.
3. Native evidence tiers are Phase 4 development/confirmatory/replication or
   methods. The registry rejects native Phase 2/3 tiers.
4. Producers compute an input manifest before reading checkpoint state.
   Any config/model/tokenizer/lens/bank/partition/upstream mismatch refuses
   resume and requires a new output directory and evidence ID.
5. Scientific seeds are canonical SHA-256 functions of study, experiment,
   item, condition, layer, position, and base seed. Python `hash()` and
   wall-clock seeds are forbidden.
6. One `ScoringSpec` governs capability, teacher-forced, generation,
   counterfactual, audit, and reproduction endpoints. The prospective
   primary alias endpoint is frozen prefix-disjoint logsumexp.
7. Every model producer invokes the same-process CUDA hard gate before model
   load and asserts a parameter is on CUDA after load. CPU model fallback is
   forbidden.
8. Every summary, table, and plot regenerates from immutable per-item rows.
   Monte Carlo p-values use the plus-one rule; exact tests are labeled exact;
   bootstrap intervals state their actual method.
9. Stops preserve state and write a stop record. No stopped or dirty-tree run
   is eligible for confirmatory evidence.
10. Existing artifacts are never overwritten while live. Corrections,
    supersessions, and withdrawals are append-only events.

The one-command foundation gate is:

```bash
bash interpretability/jspaces/phases/phase4/repro.sh
```
