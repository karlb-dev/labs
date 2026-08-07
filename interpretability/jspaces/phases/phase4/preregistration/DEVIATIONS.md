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

- 2026-07-31: the first common-base-lens Think grid
  (`p4-lineage-grid-olmo3-think-common-base-lens-dev-v1`) exposed one
  rank-1 site where the protected-energy control requested nonzero energy
  both inside and outside the protected span. That target needs two
  independently controlled components; the inherited constructor dropped
  the second component but failed to mark the site clamped. The primary
  span-safe rank/energy control was unaffected (worst relative energy error
  0.000236), and all stored outcome scores remain immutable. The constructor
  now marks this rank-limited geometry as clamped, a regression test covers
  it, and v1 is withdrawn before any own/common comparison.

- 2026-07-31: the attempted common-lens v2 repair correctly fixed clamp
  metadata but inherited the evidence ID as its scientific seed namespace.
  Versioning v1 to v2 therefore changed all random-dictionary and matched-
  control realizations. V2 is withdrawn as a useful seed-sensitivity
  diagnostic, not treated as the conformance replacement. The producer now
  accepts an explicit `scientific_seed_namespace`; v3 freezes that namespace
  to the original v1 value, so baseline, J, random, and matched scientific
  outcomes are directly comparable while only the clamp metadata changes.
