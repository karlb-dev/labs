# jspace_phase3 — mechanism, generalization, and paper hardening

> **Freeze clarification.** Phase 3 was validly frozen at
> `jspace-phase3-freeze-v1`; the stale candidate header inside the historical
> preregistration is a generation defect. The immutable facts and hashes are in
> [`preregistration/PHASE3_FREEZE_RECORD.md`](preregistration/PHASE3_FREEZE_RECORD.md).

Phase 3 of the J-space investigation. Phase 2 (confirmatory campaign,
`../jspace_part2/`, tags `jspace-part2-confirmatory-freeze-v1` /
`jspace-part2-complete-v1`) is **closed and immutable**; this package
never rewrites it.

**Governing documents** (in `reviews/`, authority order):
`jspace_lab_nextsteps_3_1_addendum.md` §4–§5, then
`jspace_lab_nextsteps_3_1.md`; adoption + VM plan in
`PHASE3_PLAN_ACCEPTED.md`.

## Layout

```
jspace_phase3/          the package (paths3, provenance3, scoring, bank,
                        split, protected_span, controls, stats, experiments/)
configs/                experiment configs (no machine paths)
data/                   item banks, family maps, curated fact tables
tests/                  conformance suite (run by repro.sh)
preregistration/        Phase 3 preregistration (+ candidate until freeze)
reports/                evidence_events.jsonl — THE registry; nothing exists outside it
paper/                  the manuscript tree (built from live evidence only)
reviews/                governing plan documents + adoption note
```

## Quick start

```bash
bash interpretability/jspaces/phases/phase3/repro.sh   # install + tests + env audit
jspace-phase3 registry-list                    # what evidence exists
jspace-phase3 run-root                         # where heavy artifacts go
```

Heavy artifacts live under the run root
(`$JSPACE3_RUN_ROOT`, default `…/special-lab-1/phase3_20260729/`), never
in git. Clean-room reproduction: point `JSPACE3_RUN_ROOT` at an empty
directory — producers cannot read or write the original outputs.

## Tier vocabulary (mandatory on every result)

`phase2-confirmatory` (imported, immutable) · `phase3-development` ·
`phase3-confirmatory` · `phase3-replication` · `methods`.
