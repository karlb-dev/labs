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

Phase 1 is **complete** (freeze `preference-phase1-freeze-v1`). Phase 2 is
governed by `plans/preference_2_2.md` **as amended by**
`plans/preference_2_2_addendum.md` (the addendum's Section B errata take
precedence). The program: decompose the Phase 1 surface policy (B-SURF),
measure semantic margins and enacted choice under the symmetric F-SYM
format, create randomized contextual advantage (B-MECH context ladders),
and only then test scenario-local handles and disjoint-surface report
coupling; OLMo-7B/Qwen/Gemma behavioral cells are scheduled, with
completeness defined by the OLMo-32B spine (addendum E17). The Phase 1
CPU re-analysis (`phase2/reports/dev_cpu_reanalysis_20260808/`,
development tier) supports the diagnosis. The campaign is **not** frozen
and not GPU-ready until banks, preregistration, and
`preference-phase2-freeze-v1` exist. `preference_2_1.md` and the earlier
opener version of `preference_2_2.md` are superseded drafts (Git
history).

## Layout

```text
preference/
├── README.md                  <- you are here
├── plans/                     <- governing plan documents (hash-pinned at intake)
│   ├── preference_1_1.md              Phase 1 plan
│   ├── preference_1_1_addendum.md     Colab execution addendum
│   ├── preference_2_1.md              Phase 2 draft (superseded)
│   ├── preference_2_2.md              Phase 2 governing plan
│   └── preference_2_2_addendum.md     Phase 2 execution addendum (§B supersedes plan)
├── data/                      <- campaign-owned generators + frozen banks (lab38_*)
└── phase1/                    <- Phase 1 COMPLETE: instrument + batteries + PC mechanism
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
| Phase 1 branch (closed) | `interp_preference_phase1` |
| Phase 2 branch | `interp_preference_phase2` |

## Phases

| phase | scope | status |
|---|---|---|
| phase1 | instrument build → frozen 7B/32B behavioral batteries → Stop C mechanism case study | **COMPLETE** (2026-08-07; freeze `preference-phase1-freeze-v1`) |
| phase2 | surface decomposition → semantic defaults + context ladders → conditional mechanism & disjoint-RO coupling → cross-model map | PLANNED (governing plan + addendum committed; not frozen, no GPU work yet) |

Precedence of governing text, per phase: handout < plan < that phase's
addendum (addendum §B lists every deviation as a numbered erratum). Each
addendum's stop-and-ask list is binding; the freeze gate is the single
human approval gate (Phase 1 plan §8.3; Phase 2 addendum §I).
