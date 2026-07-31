# Phase 4 deviations

No scientific deviation exists because Phase 4 is not frozen and no
confirmatory outcome has been run.

Governance staging note: the foundation is being authored on the
user-specified `interp_jspace_part2` branch after the immutable
`jspace-phase3-complete-v1` tag. A dedicated Phase 4 branch may be cut at the
PI-approved freeze boundary; this staging choice does not authorize outcome
collection.

## Development run incidents

- 2026-07-31: `p4-g5-bank-olmo3-think-dev-v1` stopped after its
  180-item checkpoint because accent-folding made the accepted aliases
  `Río de la Plata` and `Rio de la Plata` indistinguishable when selecting
  the bank-declared canonical spelling. No aggregate result or registry event
  was produced. The producer was repaired to prefer a unique exact canonical
  spelling before the already-frozen normalized grading comparison, covered
  by a regression test, and the clean rerun was assigned the superseding
  development evidence ID `p4-g5-bank-olmo3-think-dev-v2`. The partial v1
  state remains preserved on Drive for audit and is not admissible evidence.
