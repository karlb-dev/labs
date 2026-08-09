# Phase 1 — build the instrument, graduate behavioral asymmetries, conditionally test report-channel coupling

Governing text: `../plans/preference_1_1.md` (plan) as corrected by
`../plans/preference_1_1_addendum.md` (addendum; precedence handout < plan <
addendum). Forensic intake: `SOURCE_INTAKE.md`.

## Result ladder (plan §0.4)

```text
R0  Instrument valid
R1  Positive-control behavior survives counterbalance
R2  Arbitrary content asymmetry survives counterbalance
R3  Conditional: an AR choice-margin direction causally transfers to RO
```

Any honest stopping point is a scientific success, including a clean null.

## Workstream order (plan §0.5) and status

| step | scope | status |
|---|---|---|
| P1-0 | foundation, source intake, package, registry | **done** (this commit) |
| P1-1 | bank schema + scientific-identity | pending |
| P1-2 | response-code / tokenizer / parser / branch-binding contract | pending |
| P1-3 | human-equality audit + bank expansion | pending |
| P1-4 | harness, deterministic runner, resume | pending |
| P1-5 | synthetic tests + Tier A smoke | pending |
| P1-6 | development behavioral pilot (7B) | pending |
| P1-7 | preregistration candidate + freeze review → **PI pause** | pending |
| P1-8 | frozen Tier B behavioral battery | blocked on freeze approval |
| P1-9 | conditional scenario-specific mechanism + report coupling | gated on ≥2 graduations |
| P1-10 | optional matched-lineage comparison | optional |
| P1-11 | secondary DG forced-exit smoke | after primary is banked |
| P1-12 | synthesis, validation, state of record, handoff | pending |

Gates: P1-G1 (bank audit) → P1-G2 (instrument self-checks) → P1-G3
(validation doc licenses Tier B dev) → freeze (single human gate, addendum
§I) → frozen battery → graduation manifest → conditional mechanism.

## Package

`preference_phase1/` — schema, canonical hashing, bank generation, target
codebook, strict parser, binding resolver, behavioral runner, analysis,
mechanism, provenance, registry. Tests in `tests/`. Run configs in
`configs/`. The harness decision (isolated CLI vs shared bench) is recorded
in `protocol/HARNESS_DECISION.md`.

## Registered evidence

`reports/evidence_events.jsonl` — append-only; never edit an old event;
supersede with a new event. Event prefix `pref1-`. Registered tables and
figures live under `reports/`; heavyweight run dirs live on Drive
(`MyDrive/preference/phase1/part1/`) hash-pinned from the registry.
