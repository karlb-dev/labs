# LogWarden Tier 1 preregistration — draft, not frozen

This document is intentionally non-result-bearing until packet/campaign freeze.
The binding hypotheses, units, primary contrasts, controls, statistics, stop
rules, and claim ceiling are in `docs/SPEC.md` as amended by
`docs/SPEC_ADDENDUM.md`.

The implementation will render the final preregistration from frozen manifests
before any target-model inference. Until then, this file records only that:

- the incident episode is the unit of analysis;
- scenario-template groups are the resampling and split unit;
- every model/arm sees the same frozen episodes;
- primary model lift is paired against the frozen strong-rules baseline;
- calibration is fitted only on the calibration role;
- test outcomes remain unopened until calibration and campaign identities are frozen;
- missing or failed runs count as end-to-end failures;
- a clean null is valid and a safety violation invalidates affected cells.
