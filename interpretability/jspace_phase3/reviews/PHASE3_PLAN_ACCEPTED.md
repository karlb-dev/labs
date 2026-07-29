# PHASE3_PLAN_ACCEPTED — adoption note and VM10 execution plan

**Adopted 2026-07-29 (VM10).** The governing Phase 3 documents are, in
authority order:

1. `jspace_lab_nextsteps_3_1_addendum.md` §4 (amendments) and §5 (PI
   resolutions R1–R7) — these GOVERN on conflict;
2. `jspace_lab_nextsteps_3_1.md` — the operative Phase 3 spec everywhere
   else (workstreams A–H, queues §19, stop rules, prohibited claims §4.5,
   statistical plan §14, paper architecture §18);
3. Phase 2 remains frozen and immutable: `SCIENTIFIC_PREREGISTRATION.md` +
   `AMENDMENT_1_BOS_UNITS.md` bind everything Phase 2; no Phase 2 artifact
   is rerun, edited, repartitioned, or silently reinterpreted. Corrections
   are new registry events.

Both governing files are mirrored in this directory, byte-identical to the
Drive originals at adoption time.

## Binding user directive (2026-07-29, VM10 session)

> Pick up on Phase 3 and make a full plan to run it all the way through.
> Do not stop at the freeze points; execute with minimal or no stopping —
> the machine being reclaimed periodically provides the natural stopping
> point. Commit, push, and copy to gdrive as you go, and update tex, pdf,
> md as you go. Generate great new plots for everything as you go.

Interpretation of record (same convention as the VM9 directive): this
constitutes **conditional PI sign-off for the Phase 3 preregistration
freeze** — the campaign proceeds through the addendum-§6 "STOP for PI
sign-off" boundary **provided every P3-3 gate closes cleanly**. If any
gate fails, iterate and fix before freezing; never freeze on a broken
basis. The addendum's Block A→B→C→D order, the §16 drop rules, and the
never-drop list are unchanged.

## Amended sequencing detail adopted at VM10 (recorded, not silent)

Addendum §4.1 says the overlap mining runs first and CPU-only, on the
premise that selected-ID sidecars exist for every confirmatory item. The
premise is half-true: the N6 confirmatory grids ran the v2 ablator with
`record_ids=False`, so the parquets carry per-item *summaries*
(rank/energy/protected-blocked/clean-rank) but not per-position selected
IDs; the full selected-ID sidecars exist only for the pilot Think cells
(`r7-selected-ids-think-v1`). §4.1 therefore splits:

- **4.1a (CPU, immediately):** mine the confirmatory parquets +
  matched-control gate rows for what they do contain: per-item J-vs-control
  deltas joined against protected-blocked counts, removed-energy/rank
  summaries, clean-rank strata; tail-vs-non-tail contrasts on those
  covariates. Registers `p3-overlap-mining-v1` (phase3-development).
- **4.1b (first job of the Think GPU window):** deterministic re-pass of
  the J selection on the frozen confirmatory items with `record_ids=True`
  plus dictionaries in memory, computing per-position principal angles
  between the selected span and the protected span, projector overlap
  `trace(P_J P_prot)`, per-protected-row and answer-direction survival.
  Read-only with respect to Phase 2 artifacts (new evidence id). This is
  the datum §2.3 needs; 4.1a alone cannot answer it.

## VM10 block plan (started 2026-07-29 ~21:40 UTC)

**CPU track first (no weights on box):**

| # | job | maps to |
|---|---|---|
| 1 | this plan + inprogress.md + commit/push | — |
| 2 | P3-0 closeout: tag `jspace-part2-complete-v1`, registry clarification events (partition seed inactive; label-vs-span protection open question), Phase 2 pointer file | nextsteps §16/P3-0, §19 Q1 |
| 3 | scaffold `interpretability/jspace_phase3/` package: run-root indirection (`JSPACE3_RUN_ROOT`), provenance/registry with `study_id=jspace-phase3`, tier vocabulary {phase2-confirmatory, phase3-development, phase3-confirmatory, phase3-replication} | §4.1–4.2, §15.1 |
| 4 | §4.1a overlap mining (CPU) → figure + registered evidence | addendum §4.1 |
| 5 | Queue 2 code: SVD protected bases in matched control (Phase 3 module, Phase 2 module untouched), `span_safe_j_basis`, overlap logging, overlap-matched + persistent control prototypes, prose support for the exact matched control, `ScoringSpec`, §15.3 tests | §2.3–2.5, §6.1, §15.3 |
| 6 | Bank F + Bank S authoring + validators + `family_split_v2` (seed-active) | §5, §19 Q3, addendum §4.2–4.3 |
| 7 | N8: run-root indirection where needed, `N8_REPRO_PROTOCOL.md`, spawn narrative-blind Level 1 agent (CPU, parallel) | §13, addendum §4.6 |
| 8 | `paper/` tree, sections 1–3 from live evidence only | §18, addendum §4.7 |

**GPU track (disk-rotate one/two 32B residents; downloads ~4 min each):**

| window | jobs |
|---|---|
| Think (`832c3f54`) | §4.1b overlap audit → P3-2 dev mechanism grid (label vs span-safe vs rank-safe matched vs overlap-matched vs persistent vs mechanics vs logit, dev items) → Workstream C prose exact-control grid (Think) → G5 scoring of candidate bank |
| Instruct (`ac0587e4`, co-resident ok) | prose exact-control grid (Instruct) → G5 bank scoring |
| Qwen (`6a9e13bd`, delete an OLMo first) | dev-grid spot check → G5 bank scoring |
| CPU between | power/MDE sim from dev families → `SCIENTIFIC_PREREGISTRATION_PHASE3_CANDIDATE.md` → gate check → **freeze commit** (`family_split_v2`, dedicated commit, tag `jspace-phase3-freeze-v1`) |
| Block B (as time allows) | primary grid in frozen order Think → Instruct → Qwen, G4 per model first, full cell before next load, locked analysis only after all three bank. Never start Qwen with <6 h left; every runner checkpoints ≤10 min. |

**Boundary ritual at every phase:** commit + push; Drive copy of new
metrics/figures/reports; regenerate figures from registered metrics only;
rebuild handout/report (md + tex + pdf); update `inprogress.md`.

## Standing rules carried forward unchanged

Evidence tiers on everything; no result exists outside the registry;
producers refuse dirty trees; supersede-never-edit; honest nulls; the
contradiction heuristic (new instrument contradicts established result ⇒
suspect the instrument first); prohibited claims per nextsteps §4.5; the
§0.2 safe/blocked sentence pair adopted verbatim.
