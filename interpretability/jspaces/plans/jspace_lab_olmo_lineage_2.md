# jspace_lab_olmo_lineage_2.md

## OLMo lineage side study 2: stage attribution, transport validation, and the reopening of the externalization question

**Status:** proposal for the second OLMo lineage block. Nothing here is a
result; every number below is reproduced inline from study-1's registered
artifacts (evidence ids attached) so this plan can be **reviewed without
access to the run data**. Where a table was recomputed from a registered
parquet rather than quoted from a report, the source file is named.

**Governs with:** `jspace_lab_olmo_lineage_1.md` (the six-axis decomposition,
H1–H6 hypothesis set, O1–O6 designs — all carried forward) and
`jspace_lab_olmo_lineage_1_addendum.md` (parallel contract, §6.6
freeze-before-run rule, axis-D cap). On conflict about *what study 1 found*,
the registered record wins: `reports/OLMO_LINEAGE_STATE_OF_RECORD.md`,
`reports/OLMO_LINEAGE_CLAIMS_TABLE.md`, and the append-only registry (25
origins, 24 live events, 101 live outputs through
`ol-phase4-final-import-bundle-v1`).

**Proposed identity:** branch `interp_jspace_olmo_lineage_2`, evidence prefix
`ol2-`, package `interpretability/jspace_olmo_lineage/` extended in place,
Drive root `olmo_lineage_2_<date>/`. Development and methods tiers only.
Study-1 outputs import read-only by hash. The same forbidden-writes list
applies (no Phase 4 / Gemma / Phase 3 registries or run roots), plus one new
entry: no edits to study-1's frozen Bank-W v1 cohort.

> **Paste-line for the study-2 agent**
> Read `jspace_lab_olmo_lineage_1.md`, its addendum, the state of record, the
> claims table, and this file, in that order. The two-cell H5 stage wedge
> (OL2.1) is this study's spine and its budget priority; H6 transport
> validation (OL2.2) is the validity debt the whole lineage story owes; both
> have precommitted predictions in §4 that must be frozen into the foundation
> event before any wedge model loads. Sentence 2 upgrades only through the
> wedge; sentence 4 only through a redesigned Bank-W service protocol.

---

# 0. Executive summary

Study 1 closed at its first release boundary with an unusually clean ledger
\[`ol-phase4-final-import-bundle-v1`\]: the Bank-W capability service ran and
its joint gate **failed honestly** (16/20 common families ⇒ the decisive O4
factorial never opened); the four-checkpoint symmetric capacity study
returned **`broadly_conserved_capacity_recruitment_consistent`** with Base
finally measured; the same-corpus geometry study returned a
**`dictionary-formation-pattern`** concentrated at Base→3.0 Think; the
checkpoint inventory found **genuine official Think-SFT and Think-DPO
intermediates eligible** for a bounded stage wedge; O5 was resolved
**not identifiable** with existing cells; and everything reconstructs
independently with zero drift \[`ol-independent-reconstruction-v1`\].

The two paper-facing sentences left study 1 in these licensed forms
(claims table, binding):

- **Sentence 2 (training dependence)** — *narrowed-release-resolved*: the
  first released Think transition is **associated** with substantial
  reorganization of J-mapped token and selected-span geometry without a
  material increase in measured sparse capacity; the design does not
  identify which training ingredient caused it.
- **Sentence 4 (externalization)** — *explicitly-pending-gate-blocked*:
  Bank-S evidence motivates the hypothesis; the fact-paired Bank-W factorial
  was gated out at 16/20.

Study 2 exists to move exactly those two sentences, in that order of
feasibility:

> **OQ2.1 (sentence-2 upgrade path).** Does the dictionary-formation /
> causal-recruitment change install at the SFT stage or the DPO stage of the
> official Think recipe? The two-cell wedge is now concretely runnable.
>
> **OQ2.2 (validity debt).** Is finite-dose linear transport actually valid
> at the OLMo assay band (L24/32/40) at the doses the campaign uses? §2.6
> shows the cross-track control data makes this *not* a formality.
>
> **OQ2.3 (sentence-4 path).** Can a redesigned, prospectively frozen Bank-W
> service protocol produce a lawful cohort where v1's three-model
> intersection failed?

---

# 1. Where study 1 ended: the registered record

| Objective | Disposition | Evidence |
|---|---|---|
| O0 foundation | complete; prospective rules frozen before any outcome | `ol-foundation-v1` |
| O1 Bank-W capability service | complete; **service gate failed** (16/20); early bundle shipped | `ol-bank-w-capability-{olmo31-think,olmo31-instruct,joint}-dev-v1`, `ol-phase4-early-import-bundle-v1` |
| O2 symmetric capacity | complete; conserved verdict | `ol-capacity-joint-dev-v1` |
| O3 provenance + geometry + figures | complete; dictionary-formation router | `ol-lens-provenance-audit-v1`, `ol-geometry-joint-dev-v1`, `ol-geometry-figures-dev-v1` |
| O4 Bank-W mechanism grid | **resolved as gated out**; §6.6 accounts frozen but never tested | (gate) `ol-bank-w-capability-joint-dev-v1` |
| Checkpoint inventory (H5 task 0) | complete; v2 supersedes v1; intermediates available | `ol-checkpoint-inventory-v2` |
| O5 crossed decomposition | **resolved not identifiable**; bounded pilot design registered | `ol-o5-feasibility-decision-v1` |
| Independent reconstruction | complete; five PNGs byte-identical; model-row replay drift 0 | `ol-independent-reconstruction-v1` |
| Release | hash-pinned bundle; isolated 13-page paper compiled | `ol-phase4-final-import-bundle-v1` |

Repository state: full history merged into Part 2 at `65a7875`; the 61-GiB
local Think snapshot was deleted only after reconstruction, push, and
recovery-mirror verification (recoverable at exact revision `832c3f54…`).

---

# 2. The data record

Recomputed from the registered artifacts in the run mirror
(`ol-capacity-joint-dev-v1.parquet`, 72 rows;
`ol-geometry-joint-dev-v1_{layers,selection,readout}.parquet`, 126/18/6 rows;
`ol-bank-w-capability-joint-dev-v1.parquet`; inventory v2 JSON), plus the
Phase-4-side lineage tables study 1 imported read-only.

## 2.1 The causal trajectory being explained (imported context)

Bank-S span-safe J-specific effects (family-weighted, 95% intervals), own /
common frames \[`p4-lineage-trajectory-analysis-olmo-dev-v1`, imported\]:

| Checkpoint | Direct (own) | Direct (common) | Composed−direct (own) |
|---|---|---|---|
| Base | +0.001 [−0.05, +0.05] | +0.001 [−0.05, +0.05] | +0.002 [−0.04, +0.04] |
| 3.0 Think | −0.128 [−0.21, −0.05] | −0.097 [−0.16, −0.04] | +0.072 [−0.01, +0.16] |
| 3.1 Think | −0.167 [−0.25, −0.10] | −0.155 [−0.24, −0.09] | +0.118 [+0.05, +0.19] |
| 3.1 Instruct (sibling) | −0.022 [−0.07, +0.02] | −0.038 [−0.09, +0.02] | +0.005 [−0.04, +0.05] |

Common-support adjacent contrasts (equal-family, 100k bootstraps; \* = 95%
interval excludes zero) \[`p4-lineage-common-cohort-analysis-olmo-dev-v1`\]:
Base→3.0 direct −0.110\*/−0.105\* (own/common); 3.0→3.1 −0.006/−0.027 (no
resolved increment); Think→Instruct +0.135\*/+0.108\* with composition
−0.113\*/−0.105\*. This is the emergence-and-reversal pattern every study-2
design targets.

## 2.2 O2: capacity, now with Base (the H1 hole closed)

Own-frame centered excess (share of variance beyond matched random
dictionaries) and occupancy, from the 72-row joint table
\[`ol-capacity-joint-dev-v1`\]:

| Checkpoint | L24 | L32 | L40 | Occupancy (all layers) |
|---|---|---|---|---|
| Base | −0.00014 | 0.00315 | 0.00590 | 2 |
| 3.0 Think | +0.00065 | 0.00339 | 0.00533 | 2 |
| 3.1 Think | +0.00062 | 0.00327 | 0.00519 | 2 |
| 3.1 Instruct | −0.00016 | 0.00262 | 0.00472 | 2 |

Preregistered primary: Base→3.0 own-frame equal-layer centered-excess
difference +0.0154 percentage points, paired 90% CI [−0.0121, +0.0435]
points; occupancy difference **exactly 0**; 46/48 classified rows stable
under the frozen ±0.25-point margin (the two unresolved rows are Base-common
L40 sensitivities at the equivalence edge — a study-2 cleanup item, §4.6).
Registered verdict: `broadly_conserved_capacity_recruitment_consistent`.

## 2.3 O3: the dictionary-formation pattern, in full

Selected-span geometry at the assay layers (median selected-ID Jaccard /
median projector overlap / median principal angle, degrees), all six
checkpoint pairs — recomputed from
`ol-geometry-joint-dev-v1_selection.parquet`:

| Pair | L24 | L32 | L40 |
|---|---|---|---|
| Base → 3.0 Think | 0.33 / 0.27 / 62° | 0.33 / 0.33 / 58° | 0.33 / 0.45 / 48° |
| Base → 3.1 Think | 0.33 / 0.26 / 63° | 0.33 / 0.32 / 59° | 0.33 / 0.44 / 49° |
| Base → 3.1 Instruct | **0.00 / 0.13 / 71°** | 0.33 / 0.27 / 64° | 0.33 / 0.35 / 56° |
| 3.0 → 3.1 Think | **1.00 / 0.98 / 7°** | 1.00 / 0.99 / 5° | 1.00 / 0.99 / 5° |
| 3.0 Think → 3.1 Instruct | 0.50 / 0.72 / 28° | 0.50 / 0.89 / 20° | 0.67 / 0.90 / 17° |
| 3.1 Think → 3.1 Instruct | 0.33 / 0.58 / 40° | 0.50 / 0.87 / 21° | 0.50 / 0.89 / 19° |

Operator/row summary \[`ol-geometry-joint-dev-v1`\]: Base→3.0 raw-operator
cosine median 0.9614 (minimum 0.7343 at layer 4) while **mapped-row cosine
is 0.6744** and mapped movement 0.3256 — versus 0.0060 for 3.0→3.1. Sibling
row: Think/Instruct raw-operator 0.9902, mapped-row 0.9363, raw-unembedding
0.9984, `instruct_late_shift=false`. Readout table: unembedding rows are
essentially conserved everywhere (that is what makes "dictionary formation"
a *mid-network mapping* phenomenon, not a vocabulary change).

Three plan-relevant facts hide in the full table that the summary medians
blur:

1. **Every post-training endpoint abandons Base's selected span** (Jaccard
   ≤ 0.33 everywhere; Base→Instruct at L24 shares *zero* selected IDs, 71°
   principal angle) — Instruct is not "Base-like" in geometry even though it
   is Base-like in causal nulls. The reversal at Instruct is therefore
   **not a reversion**: it selects a *third* span and still shows no
   J-dependence. Any account in which Instruct simply "stays with the base
   organization" is already falsified by this table.
2. **The Think path froze after 3.0** (Jaccard 1.00, ≤7° at every assay
   layer): whatever installs the dictionary happens before the 3.0 release,
   i.e., inside the SFT/DPO/RLVR stack the wedge can now bracket.
3. **The identity fraction rises steeply through the band** (α =
   trace-projection of J: 0.05 at L4, 0.27 at L20, 0.42 at L24, 0.47 at L26
   on Base — from `_layers.parquet`): the assay band sits on the shoulder
   where transport stops being far-from-identity, which sharpens the H6
   question of §2.6.

## 2.4 O1: the Bank-W service gate, with the numbers that failed it

\[`ol-bank-w-capability-*-dev-v1`\]; derived/once cell, 24 families × 8
seeds × two loads, 384 rows per model, all eight answer candidates scored:

| Model | Low acc. | High acc. | High−low | 90% CI | Capable families |
|---|---|---|---|---|---|
| OLMo-3.1 Think | 0.7135 | 0.7188 | +0.0052 | [−0.031, +0.042] | 17/24 |
| OLMo-3.1 Instruct | 0.7396 | 0.7188 | −0.0208 | [−0.057, +0.021] | 17/24 |
| Qwen 3.6 27B (imported reference) | 0.8333 | 0.8333 | 0 | ≈[−0.021, +0.021] | 20/24 |

Both OLMo endpoints pass the individual equivalence/capability gate; the
**exact three-model intersection is 16 families vs the prospective minimum
20**, so `olmo_phase4_service_ready=false` and no O4 intervention cell
opened. Note the structure of the failure: it is a *joint-support* failure
driven by requiring the same families capable across Qwen and both OLMo
endpoints — not an OLMo capability failure. That asymmetry is what OL2.3's
redesign options exploit.

## 2.5 Inventory and O5 (the two doors study 2 walks through)

**Inventory v2** \[`ol-checkpoint-inventory-v2`\]: of the eight official hub
artifacts audited, exactly two are intermediate-eligible —
`allenai/Olmo-3-32B-Think-SFT` and `allenai/Olmo-3-32B-Think-DPO` (the 3.1
Instruct SFT/DPO repositories exist but fail the ancestry-qualification
rule). The v1→v2 supersession matters for review: v1 demanded byte-identical
tokenizer files and returned nothing; the v2 semantic audit shows Think-SFT's
tokenizer differs in serialization only (complete 100,278-entry ID map,
normalized merges, and all 207 frozen encodings agree). Verdict
`genuine-32b-intermediates-available`; H5 `testable-with-bounded-stage-wedge`;
queue `queued-not-started`. BOS/chat-template differences remain attached
qualifications.

**O5** \[`ol-o5-feasibility-decision-v1`\]:
`defer-no-identifiable-crossed-intervention-estimand`,
`not-executed-no-proxy-substitution` — the registry lacks causal cells that
independently cross activation model × transport lens × readout together
with per-dictionary transport, protected-span, delivered-rank/energy,
non-J-logit-lens, and common-row controls. The registered bounded entry is a
Bank-S-first Base/3.1-Think/3.1-Instruct pilot crossing Base-versus-recipient
transport and readout, expanded only after delivery and geometry checks pass.

## 2.6 A cross-track datum this plan must not ignore

The Gemma transport track used OLMo-3-32B-Think as its positive control and,
in doing so, produced the first exact-JVP transport measurements at the
OLMo **assay band** \[`gm-jvp-olmo-calibration-v1`, imported read-only\].
Same-estimator medians at the declared dose (ε = 0.10, single-position,
delivery-clean, SNR ≥ 20):

| OLMo layer | med tangent cosine | med relative error | vs the 0.20 primary error gate |
|---|---|---|---|
| L24 | 0.932 | 0.362 | fails |
| L32 | 0.962 | 0.273 | fails |
| L40 | 0.974 | 0.225 | fails (marginal) |
| L47 | 0.982 | 0.188 | passes |
| L56 | 0.992 | 0.124 | passes |
| L60 | 0.997 | 0.084 | passes |

The Gemma track's positive control *passed* on its frozen late anchors
(L56/L60) — that was its design. But the lineage campaign's causal grids run
at **L24/32/40**, where the same data shows median relative error 0.22–0.36
at the assay-scale dose. This does not invalidate any lineage result (the
lineage assay measures paired Δlog-prob under projection ablation, not
tangent prediction, and its controls are dose-matched by construction). It
does mean the phrase "conserved usable coordinate system," and any future
transport-dependent design (O5's crossed lenses; cross-checkpoint patching),
carries an unmet validity assumption **on OLMo itself, in-band** — exactly
the H6 hypothesis plan 1 §2.7 registered and study 1 left queued. H6 is
therefore promoted in this plan from housekeeping to a first-class
objective (OL2.2), with the Gemma track's frozen gate protocol
(`TRANSPORT_GATE_PROTOCOL.md`) as its instrument — noting that protocol
currently carries its own backend-parity blocker (`gm2` plan §G2.1), whose
resolution study 2 inherits rather than duplicates.

---

# 3. What study 2 must not do

Carried verbatim from the claims table's prohibited list: no "Think training
creates a workspace"; no causal-training-ingredient claims without the
wedge; no "capacity unchanged" without estimator and margin; no dictionary
identity or transport validity from structural geometry alone; no temporal
arrow into the Instruct sibling; no reuse of the failed 16/20 Bank-W cohort
as if it were a design choice. Additionally: the §6.6 four-account
predictions frozen in study 1's foundation event remain the *only* accounts
O4-style outcomes may adjudicate — a redesigned Bank W does not get new
post-hoc accounts.

---

# 4. The study-2 program

## OL2.0 — Foundation (methods; no model)

Package/branch/root verification; hash imports of every artifact cited in
§2; **freeze the wedge predictions and the H6 thresholds below before any
model loads**; register `ol2-foundation-v1`.

## OL2.1 — The H5 stage wedge (the spine)

**Design.** Two new cells — `Olmo-3-32B-Think-SFT` and
`Olmo-3-32B-Think-DPO` — bracketed by the existing immutable Base and 3.0
Think anchors. Two tiers, cheap-first, both precommitted now:

- **Tier 1 (no refits):** G5 capability on the frozen 972-row F+S bank, then
  the seven-condition Bank-S lineage grid in the **frozen-base-lens frame**
  with the study-1 scientific-seed namespace. Rationale from data: the
  common-frame trajectory reproduced every own-frame conclusion in study 1
  (§2.1), so stage localization of the *causal* onset needs no per-stage
  lens. Cost per cell ≈ one G5 (972 teacher-forced rows) + one grid (≈500
  rows) — well under a half block each.
- **Tier 2 (one refit, conditional):** if and only if Tier 1 localizes the
  causal onset to one stage boundary, fit that single checkpoint's own lens
  under the frozen 120-prompt recipe (provenance-audited, same corpus) and
  run the O3 geometry battery against Base and 3.0 Think, to test whether
  **dictionary formation co-localizes with causal onset** — the strongest
  available association short of a training intervention.

**Frozen predictions (to be embedded verbatim in `ol2-foundation-v1`).**
Under H5-SFT (supervised reasoning traces install the channel): SFT shows
Bank-S direct specificity below zero with Base-like nulls at DPO
inapplicable — concretely, SFT ≈ 3.0-Think-like (direct ≤ −0.08 in the
common frame) and DPO adds no resolved increment. Under H5-DPO (preference
optimization installs it): SFT stays Base-like (|direct| ≤ 0.04, interval
containing zero) and the negative effect appears only at DPO. Mixed
outcomes route to "distributed-across-stages," which is itself
sentence-2-relevant. Capability gating per addendum §2.4: any stage that
fails G5 support on Bank S is reported as *gated out*, never silently
absent.

**Sentence-2 router.** Clean SFT or DPO localization upgrades sentence 2
from "associated with the first released Think transition" to "installed
within the ⟨stage⟩ interval of the official recipe, on the tested banks, at
development tier." No outcome downgrades it below its current narrowed form.

## OL2.2 — H6: finite-dose transport validation at the assay band

**Design.** Apply the Gemma track's transport-gate protocol, unchanged in
estimator and delivery/SNR gating, to the lineage checkpoints at
L24/32/40 (+L56 as the known-passing anchor), at the ε ladder the causal
assay actually implies — the per-position removed-energy distribution of the
registered span-safe grids maps to effective ε well below 0.10, so the
ladder extends down to 0.005 with fp32 injection. Two models minimum (Base,
3.1 Think), all four when budget allows; the wedge checkpoints join free of
charge if OL2.1 Tier 2 triggers.

**Dependency, stated plainly:** the protocol's backend-parity stage is
currently blocked track-side (`gm-jvp-gemma-backend-parity-v1`); study 2
adopts the `gm2` plan's G2.1 calibrated ceiling once registered, and if that
ceiling is not yet available, OL2.2 runs its measurements but holds its
*license* until the ceiling lands — measurements first, verdicts gated,
exactly as study 1 practiced.

**What it moves.** A pass at the band licenses "usable coordinate system"
wording for the lineage documents and unlocks O5's crossed-lens design as
interpretable; a fail bounds every transport-flavored sentence to the late
band and reroutes O5 toward same-frame designs. Either way the campaign
stops owing this assumption.

## OL2.3 — Bank-W service redesign (sentence 4's only path)

The v1 block was a *joint-support* failure (16/20 three-model intersection)
while both OLMo endpoints individually passed (§2.4). Redesign options, one
to be selected and frozen prospectively in the study-2 preregistration —
never after seeing new capability numbers:

1. **OLMo-pair primary.** Re-scope the lineage externalization factorial to
   the OLMo pair (max-T over two models), keeping Qwen as a descriptive
   import. The lineage question — does supplied state substitute on Think
   but not Instruct — never needed Qwen in the null. Requires a fresh
   power calibration at the two-model support (study-1's 17/24-per-model
   suggests ~20–22 joint OLMo families; the mainline's Bank-W power ruler
   showed 0.806 at 20 families for the three-model max-T, so a two-model
   recalibration is plausible but must be computed, not assumed).
2. **Targeted re-authoring.** Author replacement families for the 7–8
   blockers under the same template/audit battery, aiming the difficulty at
   the failed families' profile; requires a full new outcome-blind
   authoring+split+power cycle.

Option 1 is cheaper and truer to the lineage question; option 2 preserves
mainline comparability. The selection is a Phase-5-router decision this plan
tees up with both costed.

## OL2.4 — O5 bounded pilot (conditional on OL2.2)

Exactly the registered bounded entry \[`ol-o5-feasibility-decision-v1`\]:
Bank-S-first, Base / 3.1 Think / 3.1 Instruct, crossing Base-vs-recipient
transport and readout, with per-dictionary transport, protected-span,
delivered-rank/energy, and non-J-logit-lens controls — opened only after
OL2.2's delivery and geometry checks pass at the band. The §2.3 selection
table sharpens the design's expected contrast: recipient-own and Base
dictionaries share ≤ 1/3 of selected IDs at the band (zero at Instruct
L24), so the crossed cells are maximally diagnostic — and maximally
dependent on transport validity, which is why this stage waits for OL2.2.

## OL2.5 — Receiver demonstration (axis D, capped)

Unchanged from addendum §2.5: one lesion → clean-activation patch-rescue
demonstration on the strongest Think Bank-S families, wrong-layer and
unrelated-patch controls, half-day GPU cap, development evidence for the
Phase 5A horizon regardless of outcome.

## OL2.6 — Small closures

(a) Re-run the two unresolved Base-common L40 capacity sensitivities with a
larger bootstrap to clear or confirm the equivalence edge (CPU-only).
(b) Register the α identity-fraction profile figure for all four checkpoints
(the data already sits in `_layers.parquet`; §2.3 item 3) — the lineage
companion to the Gemma amendment-2.3 figure.
(c) Carry the OLMo lineage paper's figure set forward with any wedge points
added under new evidence ids; registered study-1 figures are never edited.

---

# 5. Priority, budget, and staging

Priority under a two-block cap: **OL2.1 Tier 1 (both wedge cells) →
OL2.2 (Base + 3.1 Think) → OL2.6a/b → OL2.1 Tier 2 (if triggered) →
OL2.5 → OL2.3 power calibration (CPU) → OL2.4 (only if OL2.2 passed and
time remains)**. Rationale: the wedge is the only stage that can move
sentence 2; transport validation is the debt that gates two other designs;
everything else flexes. Budget sketch: wedge Tier 1 ≈ 2 × (G5 + grid) ≈
half a block; OL2.2 ≈ a few hundred JVP/secant cells across two models ≈
hours with the Gemma harness reused; Tier 2 refit ≈ one 120-prompt lens fit.
Staging per addendum §2.6: at most two checkpoints resident, manifests
verified on every rehydration, wedge snapshots verified against the
inventory-v2 hashes before load.

---

# 6. Risks

1. **The wedge stages fail Bank-S capability** (addendum §2.4's expectation
   for Base, now applied to SFT/DPO): then stage localization degrades to
   whatever cells survive gating, reported as gated — still informative,
   since capability onset location is itself a lineage datum.
2. **BOS/chat-template deltas on the wedge checkpoints** (inventory v2's
   attached qualification): the Amendment-1 lesson says measure the unit
   system first; the wedge inherits the frozen scoring conventions and the
   stop rule.
3. **H6 fails at the band.** Then the campaign's transport-flavored wording
   narrows (as it should), O5 reroutes, and the lineage causal results stand
   as-is — they never depended on tangent prediction.
4. **Scope.** The O1→O2→O4 spine discipline of study 1 becomes
   OL2.1→OL2.2 here; O5 is the flex stage, exactly as O3/O5 flexed in
   study 1.

---

# 7. Verification appendix (for reviewers without the run mirror)

Registry: `interpretability/jspace_olmo_lineage/reports/evidence_events.jsonl`
(frozen prefix SHA-256 `db3fe202…`, 53,719 bytes through
`ol-independent-reconstruction-v1`). Release set:
`reports/OLMO_LINEAGE_STATE_OF_RECORD.md`, `OLMO_LINEAGE_CLAIMS_TABLE.md`
(release copy SHA-256 `682bf60c…`), final bundle JSON SHA-256 `a2486ec5…`.
Package check: `bash interpretability/jspace_olmo_lineage/repro.sh` (58
tests + dependency lock), then the registry/output hash verification via the
package CLI. Run mirror for the recomputed tables:
`interpretability/jspace_runs/olmo_lineage_20260801/metrics/` — §2.2 from
`capacity/ol-capacity-joint-dev-v1.parquet` (72 rows), §2.3 from
`geometry/ol-geometry-joint-dev-v1_{selection,layers,readout}.parquet`,
§2.4 from `bank-w-capability/ol-bank-w-capability-joint-dev-v1.parquet`,
§2.5 from `checkpoint-inventory/ol-checkpoint-inventory-v2.json`, §2.6 from
the Gemma mirror's `olmo_control/gm-jvp-olmo-calibration-v1/`
`olmo_calibration_rows.parquet`. The five olf figures and their byte-exact
reconstruction record are under `metrics/reconstruction/figures/`.

**Adoption boundary.** This plan becomes operative only as study 2's
preregistration candidate through the single Phase 5 integration router,
with PI sign-off, its own branch and registry, and §4's predictions frozen
verbatim in `ol2-foundation-v1` before any wedge model loads. Until then it
is a reviewed proposal beside the study-1 record it cites.
