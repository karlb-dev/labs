# SOURCE_INTAKE.md — Lab 38 Phase 1 forensic intake record

**Status:** pre-implementation intake snapshot. This is NOT the scientific
freeze and NOT preregistration. Recorded 2026-08-07T17:54Z on the Colab VM
that opened Phase 1 part 1.

## Repository state at intake

```text
repo:           git@github.com:karlb-dev/labs.git
parent branch:  main
parent commit:  78b58f328122756bbd31fbe3ad1cf3ee10b2157e
campaign branch: interp_preference_phase1 (created from that commit)
```

## The four plan-declared inputs (plan §0.2) vs what was actually supplied

Supplied via `/content/drive/MyDrive/preference/` (Google Drive), hashed at
intake with sha256sum:

| input | plan §0.2 hash | intake status |
|---|---|---|
| `lab38_revealed_preference_report_channel.md` (draft handout) | `9b07fd87ab4201fcdb4dbddb0678399523eaa5295c007e125c5fd5152f91c5d7` | **PRESENT, hash MATCHES plan** |
| `make_lab38_preference_bank.py` | `165b8211c6343c46357d209e10ba373d81b775caef9063c574787ae6e3f5651b` | **missing_at_intake** (see below) |
| `make_lab38_disengagement_scripts.py` | `aea085db9108b3bf9783ca3067b16d52822701b9200abda7a00d8473401581f0` | **missing_at_intake** |
| design-chat transcript | `07fa9ccd31fdcb55c5636c4df4327a97e1f9918bc106dd5b342b7737d6ab34bc` | **missing_at_intake** |

Additional governing documents supplied with the same Drive folder (not part
of the plan's four, but binding):

| document | sha256 at intake |
|---|---|
| `preference_1_1.md` (the plan) | `2a8784fec194152b980678ce9889f3242ca432a2b23abec066a2be56b889f062` |
| `preference_1_1_addendum.md` (execution addendum) | `e8c6ff87de6a2c44b703dd9a3fba95fd07341d32b90a19d742b6128b3a924fd1` |

Both are copied verbatim into `preference/plans/`; the handout draft is
copied verbatim to `interpretability/labs/lab38_revealed_preference_report_channel.md`.

## Missing-input disposition (addendum E6, extended by PI instruction)

The two generators and the design transcript were **not present** in the
repository, the Drive `preference/` folder, or anywhere else on this VM
(exhaustive `find` over repo + Drive, 2026-08-07). The PI confirmed in the
session transcript (2026-08-07): the generator code existed only on a
laptop, "wasn't great", and Phase 1 should **start from scratch modeled on
the jspaces methodology**, treating the referenced Lab 38 content as
nonexistent.

Consequences, recorded as binding intake facts:

1. `missing_at_intake` is recorded for all three items; no hashes are
   fabricated (addendum E6).
2. Plan P1-0's "reproduce their current generated outputs exactly"
   (the 432-row v1 bank, the 39-script DG bank, and their §0.2 output
   hashes) is **impossible and is waived**. The v1 counts/hashes quoted in
   plan §0.2 are retained here as historical description of the lost
   prototypes, nothing more.
3. `bank_version` therefore starts directly at `lab38_v2_phase1`
   (addendum D1's "reproduce v1 then bump" collapses to "build v2").
4. Plan §7.1 test `test_draft_generator_reproduces_intake_hash` is
   replaced by `test_missing_inputs_recorded_in_intake` (asserts this
   file records the three `missing_at_intake` entries and that no code
   references the lost generators as if they existed).
5. All P0 repairs (§0.3) that were phrased as edits to the v1 generators
   are implemented as properties of the new v2 generator from birth.

## Authority of each input

- **Plan (`preference_1_1.md`)** — authoritative Phase 1 design; overrides
  the handout wherever they conflict.
- **Addendum (`preference_1_1_addendum.md`)** — highest-precedence text;
  its §B errata supersede plan/handout; its §M stop-and-ask list is binding.
- **Draft handout** — scientific motivation, DG field notes, prompt
  sketches, hypotheses. OUTLINE status; its references to "landed"
  generator files are known-false at intake (files lost).
- **PI session instructions (2026-08-07)** — (a) consolidate the campaign
  under `interpretability/preference/` like `jspaces`/`severance`;
  (b) treat referenced Lab 38 code as nonexistent, build from scratch;
  (c) match or exceed jspaces methodology, no regressions; (d) maintain
  Drive resume/inprogress tracking and phase/part folders; (e) produce
  TeX/PDF reports with figures alongside raw data.

## Intentional departures introduced at intake

| # | departure | authority |
|---|---|---|
| 1 | Campaign-owned generators, banks, and data cards live in `preference/data/` (not `interpretability/data/lab38_*` as in plan §1.2). Bench module, handout, and validation stay in course-standard locations. | PI instruction (a) |
| 2 | No byte-exact v1 draft reproduction; v2 built from scratch. | PI instruction (b); addendum E6 |
| 3 | Phase tree gains `reports/handout/` for TeX/PDF development reports with registered figures. | PI instruction (e) |
| 4 | Registry lives at `preference/phase1/reports/evidence_events.jsonl` (plan's path, unchanged) — noted here only because run dirs (`interpretability/runs/`) are globally gitignored in this repo, so registered evidence is *copied* into `phase1/reports/` at evidence boundaries and hash-pinned; live run dirs are Drive-mirrored. | repo `.gitignore` reality |
| 5 | Harness: the PI authorized an isolated campaign harness instead of mandatory `interp_bench.py` registration (plan §1.3), matching the jspaces pattern ("special labs may need a special benchmark; assess and decide"). Decision recorded in `phase1/protocol/HARNESS_DECISION.md` after the bench-contract review; model-ID pinning is resolved from the bench's pinning machinery either way (addendum E8). | PI instruction (2026-08-07, session) |

Nothing else in the plan or addendum is knowingly departed from at intake.
