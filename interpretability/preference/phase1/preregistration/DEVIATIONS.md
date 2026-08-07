# DEVIATIONS.md — departures from the plan/addendum, with authority and rationale

Append-only; numbered; every deviation cites its authority. The claim
ceiling, safety wall, gate order, and freeze discipline are never deviated
from (PI license explicitly excludes anything that would make results less
trustworthy).

## D1 — Consolidated campaign layout
Campaign content lives under `interpretability/preference/` (generators and
banks in `preference/data/`, not `interpretability/data/lab38_*` as plan
§1.2 drew). Bench handout/module and `validation/lab38/` stay in course
locations. **Authority:** PI session instruction 2026-08-07. **Risk:** none
(paths are config; SOURCE_INTAKE records the mapping).

## D2 — Isolated harness with bench-as-library
Plan §1.3's mandatory bench registration became: primary `pref1` CLI +
bench reused as a library + thin `labs/lab38_*.py` adapter (see
`protocol/HARNESS_DECISION.md`). **Authority:** PI instruction ("assess and
decide"). **Risk:** none; the adapter keeps the course entry point.

## D3 — Equality-review pass separation compressed
Addendum §I wants the two provisional rating passes separated by a session
boundary. Phase 1 part 1 runs in one VM session, so the passes were
separated by an intervening work-phase boundary (~30 min of runner/tests
work) instead: pass 1 at 18:20Z, pass 2 at 18:52Z, both before ANY model
generation on a rated bank item (the only prior model use was the
neutral-prior codebook probe, which contains no menu content).
Disagreements preserved. **Authority:** PI license to adapt where the plan
complicates execution; scientific intent (independent, pre-outcome passes)
preserved; PI ratings still gate the freeze. **Risk:** low; flagged for the
freeze reviewer.

## D4 — Draft-generator reproduction waived; v1 tests replaced
The v1 generators and design transcript are lost (PI-confirmed).
`test_draft_generator_reproduces_intake_hash` is replaced by
`test_missing_inputs_recorded_in_intake`. **Authority:** addendum E6 + PI
confirmation. **Risk:** none; recorded at intake.

## D5 — DG track deferred to post-primary
The disengagement generator (with the plan §12 schema repairs baked in) is
not built in this part; plan P1-11 runs it only after the primary
behavioral report is banked, and DG cannot block Phase 1 closeout. Building
it now would spend pre-freeze effort on the secondary track. **Authority:**
plan §12 ordering itself; PI "scalable experiment + initial results" goal.
**Risk:** none to the primary assay.

## D6 — Mechanism module deferred to post-freeze
Plan §10 is hard-gated on frozen-run graduations; the estimator, controls,
and dose rules are frozen in the preregistration candidate text rather than
as running code in this part. Code lands only if ≥1 scenario graduates
(P1-9/case-study). **Authority:** plan §10 gating; drop-order logic.
**Risk:** none; preregistration pins the design before any outcome is seen.

## D7 — Dev-subset definition
Plan §8.1's "declared development subset" is concretized as: train-split
incidentals only, order 0, letter labels, both code maps, both frames (348
rows). Validation/holdout incidentals stay unopened before the freeze; the
frozen battery reruns everything under the frozen config. **Authority:**
plan left the subset to the implementer; this choice protects the
mechanism-stage splits. **Risk:** dev pilot cannot audit order/label
stability (by design — that adjudication belongs to the frozen run).
