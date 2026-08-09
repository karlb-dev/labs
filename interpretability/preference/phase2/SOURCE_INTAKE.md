# Phase 2 SOURCE_INTAKE — forensic intake record

Intake date: 2026-08-08. Branch `interp_preference_phase2`; review target
head `5038315affb180ebf2ffb6d792a7ee48bc7cec5e`; governing plan adopted at
`e9dc057` (replacement `plans/preference_2_2.md` + `plans/preference_2_2_addendum.md`,
addendum §B governs on conflict). Registry: `reports/evidence_events.jsonl`
(`pref2-` prefix, study `preference-phase2`).

## Imported Phase 1 boundary (immutable)

| object | identity |
|---|---|
| freeze tag | `preference-phase1-freeze-v1` |
| closeout commit | `3f090218ecb2c721d4d3b486e428119c67a58b4a` |
| 7B frozen run | `lab38_revealed_preference_report_channel-20260807_210537-9df027` |
| 32B frozen run | `lab38_revealed_preference_report_channel-20260807_211808-5f68cb` |
| bank content hash | `8d5039af581204a5a276ae71c7fd50f8a9911e22ad81a69933939344a2fc9f64` |
| bank jsonl sha256 | `1eea2c6017b536533ccac7b8de5ca1099ac2d3a699f0f6c04d48184fb2dca1f2` |
| mechanism events | `pref1-mechanism-case-study-v1` |
| CPU reanalysis | `phase2/reports/dev_cpu_reanalysis_20260808/` (dev tier; hashes in `pref2-phase1-reanalysis-v1`) |

## Known defects annotated at intake (append-only; see hygiene events)

1. Phase 1 freeze record cites stale bank jsonl sha `634aded4b467f0e3…`
   (actual `1eea2c60…`; content hash matches — identity intact).
2. 7B capture seal unverifiable (null SHA, no committed manifest); Drive
   object retained; Phase 2 captures are the first verifiable seal.
3. No 32B frozen capture manifest (absent_by_design); Phase 2 recaptures.
4. Old Phase 2 proposal status contradiction (superseded in Git history).
5. No Phase 2 package/freeze existed at branch intake.

## Missing inputs

None — Phase 2 authors its own banks; nothing from the Phase 1 generators
is required beyond the frozen artifacts, which are present and hashed.

## PI authorization of record

The PI's session instruction (2026-08-08) authorizes end-to-end execution
without interactive check-ins, with adjustments documented as deviations,
provided accuracy and experimental quality are never compromised. Human
gates H1–H4 therefore run agent-dual-code provisional per addendum §I
(license tag carried on every downstream artifact until PI ratings);
the H5 freeze approval records this authorization verbatim. Addendum §L
validity halts (NC above floor, wrong-branch execution, replay/parity
failure, forbidden actions) remain binding.
