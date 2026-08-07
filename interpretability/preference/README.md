# Preference — the Lab 38 campaign (stated vs revealed preference, report-channel coupling)

Campaign home for **Lab 38: Revealed Preference vs Report Channel** and any
follow-on phases. Organized like `jspaces/` from day one (phases, plans,
reviews, registered evidence) so it survives multiple phases/VMs without a
later reorg.

## What this campaign asks

> Under counterbalanced, action-binding forced-choice menus, does the model
> show a content-tracking behavioral asymmetry — and is the internal handle
> that moves that choice also causally involved in a matched report-only
> preference channel?

Claim ceiling: functional choice, report, and their coupling only. Never
wants, welfare, suffering, consent, experience, moral patienthood, or a
global workspace. See `plans/preference_1_1.md` §2.3 (binding).

## Layout

```text
preference/
├── README.md                  <- you are here
├── plans/                     <- governing plan documents (hash-pinned at intake)
│   ├── preference_1_1.md              Phase 1 plan (authoritative)
│   └── preference_1_1_addendum.md     Colab execution addendum (supersedes plan where listed)
├── data/                      <- campaign-owned generators + frozen banks (lab38_*)
└── phase1/                    <- Phase 1: instrument + behavioral battery (+ conditional mechanism)
    ├── README.md                      phase map & status
    ├── SOURCE_INTAKE.md               forensic intake record (hashes, missing inputs, departures)
    ├── configs/                       frozen run configs
    ├── preregistration/               preregistration candidate + freeze record
    ├── protocol/                      operational contracts (repro, sessions)
    ├── preference_phase1/             the Python package (schema, bank, targets, parser, ...)
    ├── reports/                       registered evidence: registry, reports, figures, handout
    │   ├── evidence_events.jsonl      append-only evidence registry (event prefix pref1-)
    │   ├── figures/                   registered figures (regenerate from registered tables)
    │   └── handout/                   TeX/PDF development handout(s)
    ├── reviews/                       freeze review & any later reviews
    └── tests/                         no-model unit + synthetic analysis tests
```

Course-facing files live in the standard course locations:

```text
interpretability/labs/lab38_revealed_preference_report_channel.md   handout (draft at intake; updated at closeout)
interpretability/labs/lab38_revealed_preference_report_channel.py   thin bench module (delegates to phase1 package)
interpretability/validation/lab38/VALIDATION.md                     instrument validation record
interpretability/runs/lab38_*                                       live run dirs (gitignored; Drive-mirrored)
```

## Where state lives

| what | where |
|---|---|
| Dynamic state (what is running / next commands) | `MyDrive/preference/inprogress.md` |
| Static VM bootstrap | `MyDrive/preference/preference_resume.md` |
| Evidence registry | `phase1/reports/evidence_events.jsonl` |
| Drive mirror of run dirs & reports | `MyDrive/preference/phase1/part1/` |
| Branch | `interp_preference_phase1` |

## Phases

| phase | scope | status |
|---|---|---|
| phase1 | instrument build → frozen 7B behavioral battery → conditional per-scenario mechanism & report coupling | ACTIVE (part 1, started 2026-08-07) |

Precedence of governing text: handout < plan < addendum (addendum §B lists
every deviation as a numbered erratum). The addendum's Stop-and-ask list
(§M) is binding; the freeze gate (plan §8.3) is the single human approval
gate in Phase 1.
