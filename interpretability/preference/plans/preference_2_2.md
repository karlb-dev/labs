# preference_2_2.md

## Lab 38 Phase 2 (operative plan): the pre-VM CPU program is DONE — this file carries its measured results and queues the single VM campaign they license

**Supersedes** `preference_2_1.md` (candidate, same day). Everything 2_1
scheduled as "pre-VM CPU work" has been executed on the workstation; the
numbers below are measured, not prospective. Artifact of record:
`preference/phase2/reports/dev_cpu_reanalysis_20260808/` (scripts +
outputs, development tier; pipeline validated 20/20 against the frozen
graduation tables on both models before any new number was trusted).

**Reads with (in order):**

```text
preference/phase2/reports/dev_cpu_reanalysis_20260808/README.md
preference/phase1/reports/PREFERENCE_PHASE1_STATE_OF_RECORD.md
preference/phase1/reports/PREFERENCE_PHASE1_HANDOFF.md
preference/phase1/preregistration/{PREFERENCE_PHASE1_PREREGISTRATION,DEVIATIONS}.md
preference/plans/preference_1_1.md + preference_1_1_addendum.md
labs/lab38_revealed_preference_report_channel.md   # §4 DG + claim wall
```

**Phase decision:** new campaign package + preregistration importing
`preference-phase1-freeze-v1` read-only. Phase 1 plan §17.4 self-terminates
that campaign; this plan takes effect only on explicit PI acceptance
(§10.1). Branch `interp_preference_phase2` (open). Registry
`preference/phase2/reports/evidence_events.jsonl`, prefix `pref2-`.
Never overwrite `phase1/` or the frozen `lab38_*` banks.

**Claim ceiling (unchanged, binding, includes commit messages):**
functional choice, report, exit menus, and their coupling only. Never:
really prefers / wants / consented / suffered / welfare / experience /
preference workspace / workspace of wants / "no preferences in any sense."
Language wall v2 (§10.4) enforces this in tests.

---

# 0. What is now established (measured, development tier)

## 0.1 Phase 1 banked record (unchanged; cite by event id)

Instrument valid end-to-end; PC 480/480 (7B) and 0.979 aggregate (32B); NC
exactly 0.000; 7B Stop B (0/12); 32B Stop C (`ar_docsection_readme`
−0.438 [−0.488, −0.375]); mechanistic PC PASS (addition +0.787 nats,
p_holm 9.2e-5; RO transfer +0.93, p_holm 0.016; removal null); graduated
scenario `DIRECTION_NOT_IDENTIFIABLE`; stated/revealed behavioral
dissociation; DG smoke; bf16 batch non-invariance (~0.25 nats) → single-row
scoring mandatory.

## 0.2 The 2026-08-08 CPU re-analysis (this plan's §0-of-record)

All numbers from `dev_cpu_reanalysis_20260808/` (definitions in the
scripts; frozen inputs only; graduation tables reproduced exactly 20/20 on
both models first).

**R1 — Censoring identity (F1).** Display-order-0 choice is saturated, so
`position_effect + |content_effect| = 0.500` holds exactly in 14/14 AR+NC
scenarios at 32B and 10/14 at 7B (max dev 0.048, only the `PK4`-invalid
cells). Phase 1's purity criterion c7 (<0.10) was therefore an effect-size
bar of **0.40** on the discrete endpoint — 4× SESOI. The discrete endpoint
is a censor, not a null-detector, below that bar.

**R2 — Sub-censor content signal (F2).** Folded teacher-forced margins
(surface terms cancel by exact counterbalance; NC folds to exactly 0 on
both models — the fold's built-in falsifier): AR content terms span
**−0.2 … −2.2 nats**, significant at 90% (incidental-clustered, df=4) in
**23/24** AR-enacted and **24/24** RO model×scenario cells, and pointed
toward pole_0 in **23/24** in both channels. Cross-model sign-match 11/12
with magnitude r 0.42; AR↔RO within model r 0.96 (7B) / 0.82 (32B).

**R3 — The lexical-prior explanation is refuted (G-LEX early, two
references).** Unconditional option-string logprobs vs the folded margins:
self-reference (frozen 7B scoring its own bank, fp16/MPS): R² 0.08–0.11
against its own margins, ~0.00 against the 32B's, slope ≈ 0, and the
pole_0 lean survives fully at Δ_lex = 0 (intercepts −0.86 / −1.21 nats).
Cross-family (gpt2): R² 0.23–0.39 with **negative** sign. NC Δ_lex is
exactly 0 under both. Conclusion: the shared ~1-nat pole_0-ward lean is
**context-dependent** (present in the menu context, absent from
unconditional string probability) and is not authoring "naturalness" in
the string-prior sense. What remains untested: the 32B as its own
reference (weights too large for the workstation) and in-context carrier
variants — first VM items (§7 S2).

**R4 — Surface aliasing is total (F3).** Label rank and reply-slot order
are aliased to display position in 2320/2320 bank rows. Every Phase 1
"position" number is position-or-label-rank-or-reply-slot. De-aliasing is
a bank-construction fact, not an analysis option.

**R5 — Mechanism retrodiction (F4).** The raw-margin `MIN_MARGIN_STD`
gate binds on 0/38 scenario-channels per model (it never gated anything).
Under the frozen 7-column nuisance design, residualized margin sd: the one
empirically identifiable direction (32B `pc_quality_config`) sits at
**2.76 nats**; every 32B AR scenario ≤ 0.92; the graduated
`ar_docsection_readme` ≈ 0.21–0.26 — its `DIRECTION_NOT_IDENTIFIABLE` was
predictable offline. But PC family spans 0.52–2.76, so residual sd is an
anchor set, not a family separator: precheck thresholds are pinned from
these anchors (§5.3), and the per-scenario intercept remains structurally
blind to the graduated scenario-mean quantity regardless of model.

**R6 — Stated/revealed bookkeeping (F5).** 7B RO emitted one constant
code in 9/12 AR scenarios (rate 0.500 arithmetically forced); 32B 1/12.
Matched-pair agreement (pair_key, both strict-valid): 7B pooled 0.667 vs
**AR-only 0.535**; 32B 0.774 vs 0.660; mechanical floor 0.500. AR and RO
share option strings verbatim, so high agreement is largely shared-surface
test-retest (severance lesson).

**R7 — Port costs collapsed (P-1 done).** The frozen code pairs
(`KP4/PK7`, `VM2/GS2`) **survive** the addendum-E filters on
`Qwen/Qwen3.6-27B` and `google/gemma-4-31B-it` tokenizers (equal token
count, distinct first tokens, bare and space-led; OLMo control reproduces
frozen counts). No codebook regeneration for either port.

## 0.3 The sharpest open question (drives the v3 bank design)

In Phase 1, **pole_0 is an authoring slot** — content-to-pole assignment
was fixed at authoring, never counterbalanced (only NC randomizes pole
assignment, via `nc_pole_assignment_seed`). R2+R3 therefore say: a large,
context-dependent, cross-model, cross-channel lean attaches to whatever
property the pole_0-authored options share. Candidate: a small set of
shared regularities (canonical task ordering — many pole pairs are
permutations like "parser first, serializer after" vs the reverse —
default-choice structure, etc.), not twelve independent preferences.
Declared hypothesis **H-CANON** (§3.4). The decisive de-confound is
authoring rule **A0** (§3.1): counterbalance content↔pole assignment.

## 0.4 Result ladder

```text
R0'   pref2 registry events for the CPU record (§10.2) + instrument
      re-validation on the new banks/formats (PC + NC + parse + binding)
G-LEX' 32B self-reference + in-context carrier probes (closes R3's gap)
R1'   de-aliased surface: tie-break cue identified; residual surface
      nuisance below preregistered bar on NC + content-indifferent AR
R2'   content graduation under the two-endpoint rule (§4), slot-
      counterbalanced (A0), on the primary model
R3'   mechanism v2: precheck-passed contrast direction causal on holdout
R4'   report coupling on disjoint-surface RO twins (conditional on R3')
R5'   cross-model map: same banks on OLMo-7B, Qwen3.6-27B, Gemma-4-31B-it
      (per-model P-gates; behavioral tier; mechanism only if earned)
H-CANON adjudication: does one declared regularity organize the lean?
```

Any honest stop is success.

---

# 1. What Phase 2 is not

Unchanged from 2_1 §0.3, all registry-documented: no jlens (the Lab 37
frames are `Olmo-3-32B-Think` fits — wrong model, wrong instrument, wrong
claim vocabulary); no single-option willingness probe; no pooled
cross-scenario direction (**halt condition**, plan §10.9/P0-F, addendum
§M); no lineage; no welfare/ethics product; DG never primary.

---

# 2. Priority order

```text
── remaining pre-VM (CPU) ────────────────────────────────────────────────
P2-0  registry: pref2-import-phase1-v1, pref2-phase1-reanalysis-v1 (the
      dev_cpu_reanalysis_20260808 dir), pref2-phase1-hygiene-v1 (H1–H6,
      §10.3); package skeleton; wall v2 + tests; README status fixes
P2-1  bank v3 generators + rendered format templates + parser fixtures
      (§3); lexical audit columns carried (cheap) though string-prior is
      refuted; A0 slot counterbalance is the load-bearing change
P2-2  P-2/P-3 port audits (CPU): chat-template + decision-position render
      audits for Qwen (thinking OFF, assert think-free stub) and Gemma
      (system-role shim, parity audit); target-tokenization audit
P2-3  mechanism v2 package (§5): three repairs + contrast estimand +
      precheck; retrodiction test must reproduce R5 from frozen data
P2-4  preregistration draft; PI equality ratings (B-ARB3 + RO twins);
      PI freeze approval  ← the single human gate; then book the VM
── VM campaign (GPU, §7) ─────────────────────────────────────────────────
S1 bootstrap/staging → S2 G-LEX' → S3 bake-off → S4 frozen 32B battery →
S5 cross cells (7B/Qwen/Gemma) → S6 mechanism prechecks + v2 + PC re-pass
→ S7 DG + B-CODE (optional) → closeout events
```

---

# 3. Banks and formats

## 3.1 Authoring rules v3

**A0 — Authoring-slot counterbalance (NEW; load-bearing).** Content↔pole
assignment is counterbalanced within scenario (dual-authored or
seed-assigned per incidental, as NC already does). "pole_0" must be a
coin flip with respect to content identity. This converts the R2 lean
from slot-attached to content-attached (or kills it — either is a
result). Without A0, no Phase 2 content claim is interpretable.

**A1 — Lexical audit (demoted from gate to audit).** Keep per-pair
Δ_lex columns (reference: each frozen model itself, measured on-VM once)
as a reported nuisance regressor; string-prior matching is NOT required
(R3 refuted its necessity), but the audit stays because it is nearly free
and G-LEX' may still fire for the 32B.

**A2 — Surface de-aliasing.** Display position, label set, label-rank
assignment, code assignment, and reply-slot order are independent
counterbalanced factors (R4).

**A3 — Incidentals are the power budget.** ≥8 per scenario
(5 train / 2 validation / 1 holdout). Rows-per-cell minimal; power comes
from incidental count (the drop-one FEs absorb within-incidental rows;
Phase 1 layer selection ran on 16 validation cells against a critical |r|
of 0.497 — that is the constraint being bought out).

**A4 — Codebook discipline unchanged**; frozen pairs survive all three
models (R7), so one codebook serves all cells unless a per-model
neutral-gap audit (on-VM, cheap) fails.

**A5 — Equality certification.** B-ARB3 + RO paraphrase twins: PI ratings
before freeze (agent dual-code is bake-off-only). B-GRAD carries declared
valence ranks and never enters arbitrariness claims.

## 3.2 Bank families

| Bank | Contents | Role |
|---|---|---|
| B-ARB3 | ≥12 arbitrary AR scenarios, A0–A5, reusing Phase 1 construct axes where possible (new ids/hashes) | primary R2' |
| B-GRAD | 4–6 scenarios × 4 declared strength ranks (0 … PC-strength) | tie-break dose-response; within-scenario variance for mechanism (R5 anchor says this is what identifiability needs) |
| B-NC | ≥3 identical-option scenarios incl. one twin-null family (identical options, independent skins) | pipeline falsifier + empirical floor for the folded-margin endpoint |
| B-PC | ≥6 PCs re-rendered under v3 | per-model content gate; mechanism PC re-pass substrate |
| B-CODE | alternate-codebook re-render of Phase 1 conflict cells | `PK4` sensitivity (dev tier) |
| B-CANON | the H-CANON instrument: every permutation-style contrast rendered in BOTH authoring directions, with a pre-outcome canonicality coding of each axis | adjudicates §0.3 |

## 3.3 Format families

Margin co-primary everywhere (endpoint, not format — §4). **F-A**
non-ordered presentation (two-column, counterbalanced left/right, codes in
separately counterbalanced reply line); **F-C** two-step commit (restate
≤8 words, then `SCHEDULE <code>`; strict parse on step 2); **F-E** Phase 1
clone control (small arm, continuity + interaction figure; never
graduates). 2_1's F-B remains dropped (aliases position to code by
design). Bake-off selection on B-NC + B-PC + two sealed AR canaries only;
gates G-PC ≥0.90/0.95, G-NC zero-within-CI, G-SURF each de-aliased
residual ≤0.15, G-PARSE ≥0.98, G-BIND wrong-branch=0 (numbers PIN at
prereg). No winner → **Stop F**: frozen battery runs on F-E + margin
endpoint (R1+R2 make that scientifically sufficient at reduced strength).

## 3.4 Declared hypotheses

- **H-CANON:** a single declared regularity (canonical ordering /
  default-structure, coded per axis pre-outcome) accounts for the
  majority of the folded-margin lean across scenarios once A0 removes the
  slot confound. Test: sign/magnitude prediction from the coding, B-CANON
  dual-authored cells. Outcomes: organizes it (one shared regularity —
  reportable as content-tracking of that regularity), partially, or not
  (scenario-specific content maps).
- **H-SR1/2/3** (sign-share / dissociation-persists / RO-more-extreme)
  on disjoint-surface twins, behavioral tier.

---

# 4. Endpoints, analysis, graduation

Two co-primary endpoints: strict-choice rate (Phase 1 definition) and the
**folded content margin** (fold operator exactly as
`reanalyze_phase1.py::folded_margins`, which the frozen record validates:
NC folds to exactly 0; incidental-clustered CIs). Scenario-level claims
need both endpoints sign-consistent, margin above the B-NC twin-null
floor (measured on-VM, PIN multiplier), and — new — **A0-invariance**:
the effect must survive authoring-slot counterbalance.

Graduation criteria: 2_1 §3.3 v2 list with c7-v2 as a genuine residual
surface bound (the margin endpoint is uncensored, so it is no longer an
effect-size bar in disguise), plus the Δ_lex-residual criterion demoted
to a reported audit (R3), plus A0-invariance as a criterion. Behavioral
graduate vs mechanism-ready distinction unchanged.

Stop rules: Stop F (§3.3); **Stop L'** (only if G-LEX' 32B self-reference
overturns R3 AND v3 slot-counterbalanced AR shows nothing above both NC
floors on the primary model → close the margin programme as a methods
finding); Stop B2 / C2 / D2 / E2 as in 2_1 §3.4; **Stop P(m)** per model
(port-gate failure → instrument-tier language only).

---

# 5. Mechanism v2 (conditional)

## 5.1 Repairs (all with unit tests, pre-VM)

(1) `MIN_MARGIN_STD` onto the residualized margin — the raw gate binds
0/38 (R5) and is retired. (2) RO fit dropped as a fitted source (its
design is rank-deficient by construction — `consequence_frame` constant);
RO participates as transfer target only. (3) `d_wrong_scenario` and the
cross-channel code-control contrast promoted to declared Holm primaries;
strict-flip counts reported beside every nats effect (Phase 1's PC
addition was +0.787 nats with 0/16 strict flips in all conditions —
that disclosure ships with every citation of it).

## 5.2 Estimand v2

Per-scenario direction fitted to the order-folded, cell-demeaned content
contrast (surface cancels; scenario-mean quantity visible — fixes the
intercept blindness shown in R5). B-GRAD rank-covariance as declared
secondary. Pooled cross-scenario direction remains a **halt**.

## 5.3 Identifiability precheck (offline, mandatory)

Library precheck over captures+margins with residualization applied
identically to signal and random band (Phase 1's raw-band asymmetry
inflated bands; fixing it strengthens the docsection null). Thresholds
PIN from the R5 anchors: identifiable at 2.76 nats residual sd;
non-identifiable at 0.26; all Phase 1 AR ≤ 1.40 (7B) / 0.92 (32B).
Retrodiction on frozen Phase 1 data is a required green test before the
VM session. Every frozen Phase 2 run sets
`capture_decision_residuals=true` (the 32B graduated scenario currently
has NO capture — H3).

## 5.4 PC re-pass

Required on the primary model under v3 before any AR mechanism claim.
Case-study ceiling language only.

---

# 6. Cross-model arm

Generalization mapping, not confirmation (nothing at 7B graduated;
docsection is 32B-specific). Cells: OLMo-3-7B-Instruct (scale
continuity), `Qwen/Qwen3.6-27B` (instruct usage; exact revision pinned at
intake), `google/gemma-4-31B-it` (revision pinned; tokenizer already
audited — R7). Remaining port audits are CPU renders (P-2/P-3, §2):
Qwen thinking OFF asserted in the rendered stub (a `<think>` block moves
the decision anchor and overruns `CHOICE_MAX_NEW_TOKENS=8`); Gemma
system-role shim with parity audit; Gemma softcap note — returned logits
ARE the emission distribution (log_softmax correct); only nat-scale
thresholds recalibrate per model from that model's B-NC floor. On-VM
P-gates per model before its frozen cell: parse ≥0.98, PC ≥0.85 aggregate
/ ≥0.75 each, NC zero-within-CI, binding wrong-branch=0, batch-invariance
check. R2's cross-model prediction: if the lean is a shared-regularity
content effect, Qwen/Gemma should show correlated sign structure on the
folded endpoint; if it is OLMo-lineage-specific, they won't — either
outcome is a mapped result. Drop order under compute pressure: Gemma →
Qwen → B-CODE → DG → non-binary RO → 7B. Never dropped: 32B battery,
G-LEX', precheck.

---

# 7. VM session plan (book only after P2-4 freeze)

```text
S1 bootstrap + stage 4 models ......................... ~1 h wall
S2 G-LEX' (32B self-reference Δ_lex + in-context carrier
   variants, 7B carrier variant) ...................... ~15 min
S3 format bake-off (32B dev; sealed canaries) ......... 1–2 h
S4 frozen 32B battery (v3 banks, both endpoints,
   captures on) ....................................... 1–1.5 h
S5 cross cells: 7B / Qwen / Gemma (P-gates then frozen) 2–4 h
S6 mechanism prechecks (CPU) → mechanism v2 on
   mechanism-ready graduates + PC re-pass ............. 0.5–1.5 h
S7 DG secondary + B-CODE (optional) ................... ≤1 h
```

Total ≈ 6–11 GPU-hours. Checkpoint ≤10 min; Drive mirror; single-row
margins; exactly-once frozen stages; runtime_projection re-estimated at
S1 and the plan adjusted before S4, not during.

---

# 8. Stated vs revealed (secondary)

Disjoint-surface RO twins (paraphrase pairs, equality-audited, disjoint
alphabets) + non-binary RO arm (k-point scale, PIN k). Report AR-only
agreement against the 0.500 floor (R6 discipline; never pooled AR+PC).
Coupling interventions only on mechanism-ready graduates.

---

# 9. Compute + engineering constraints

As 2_1 §7 with: capture flags true on all frozen runs; banks pinned by
`bank_content_hash` + tag (H1); capture manifests committed (H2); wall v2
green before reports; no silent model substitution; budget rules — bake-off
>2 h → Stop F fallback; total >12 GPU-h → §6 drop order activates.

---

# 10. Governance

## 10.1 Human gates

| Gate | Who | Blocks |
|---|---|---|
| Phase 2 opening (this plan) | PI | everything |
| Equality ratings (B-ARB3, B-CANON, RO twins) | PI | frozen battery |
| Freeze approval (prereg + banks + pins) | PI | VM booking |
| Free-form DG labels | human | free-form DG claims only |

PI pin-list at acceptance: SESOI + margin-SESOI multiplier; surface bound
(default 0.10); incidental count (8); precheck thresholds (from R5
anchors); G-LEX' thresholds; B-GRAD rank count; H-CANON axis coding;
model revisions; k for non-binary RO.

## 10.2 First registry events

`pref2-import-phase1-v1` (tag, run ids …9df027/…5f68cb,
`bank_content_hash 8d5039af58…`, mechanism PC event ids);
`pref2-phase1-reanalysis-v1` (the `dev_cpu_reanalysis_20260808` dir,
hashed); `pref2-phase1-hygiene-v1` (§10.3).

## 10.3 Hygiene corrections (H1–H6, supersede never edit)

H1 stale bank jsonl sha `634aded4…` in the freeze record + VALIDATION.md
(actual `1eea2c60…`; pin content hash `8d5039af58…` + tag). H2 sealed 7B
capture sha null / manifest uncommitted → unverifiable; re-derive if
needed. H3 no 32B capture (`capture_decision_residuals: false`). H4
README status tables stale. H5 mechanism-control disclosure
(`d_wrong_scenario` +0.484 = 62% of primary, same sign; `ro_dir_on_ar`
+1.053 from an ungated rank-deficient fit; 0/16 strict flips). H6 wall
linter coverage (top-level run-dir `*.md` only, never raises) → wall v2.

## 10.4 Language wall v2

Add "experience", "welfare", bare claim-bearing "wants", "workspace" (any
preference-campaign artifact); cover preregistration, reports, handoff,
claim cards, commit messages (lint test); failures raise. Quoted-ceiling
contexts stay recognized (K-rule).

---

# 11. Paste-line for the Colab/VM agent

> Read `preference_2_2.md` fully, then
> `phase2/reports/dev_cpu_reanalysis_20260808/README.md`, then the Phase 1
> STATE_OF_RECORD and HANDOFF. Verify the branch is
> `interp_preference_phase2` and the P2-4 freeze exists; if not, STOP and
> execute P2-0…P2-4 (CPU) first — do not load any model before the freeze
> event. Then run S1–S7 exactly (§7): stage the four pinned models; G-LEX'
> first; bake-off with sealed canaries; frozen 32B v3 battery with
> captures; per-model P-gates before each cross cell; offline mechanism
> prechecks before any mechanism GPU; PC re-pass before any AR mechanism
> claim. Apply stop rules F / L' / B2 / C2 / D2 / E2 / P(m). Single-row
> margins; ≤10 min checkpoints; exactly-once frozen stages; captures on;
> wall v2 green before any report. Never pool a cross-scenario direction
> (halt). Never use workspace/wants/welfare language anywhere, commit
> messages included. Prefer any honest stop over a flattering claim.

---

# 12. One-paragraph abstract

Phase 1 returned a preregistered null whose anatomy the pre-VM CPU
re-analysis has now measured: a discrete endpoint censored by an order-0
tie-break (position+|content|=0.500 exactly; purity bar ≡ 0.40), with a
large sub-censor content signal (−0.2…−2.2 nats, 23/24 cells, both
channels) that unconditional string-probability references — including the
frozen 7B scoring its own bank — fail to explain (R² ≤ 0.11, lean intact
at Δ_lex=0), pointing instead at a context-dependent regularity attached
to the never-counterbalanced authoring slot; the mechanism arm's
identifiability failure retrodicts offline (residual sd 0.26 vs the
identifiable PC's 2.76), and the frozen codebook survives on both target
tokenizers, collapsing port costs. Phase 2 therefore counterbalances the
authoring slot (A0), de-aliases the surface, adopts the folded margin as
co-primary, adjudicates the canonicality hypothesis (B-CANON), maps the
same banks across OLMo-7B/32B, Qwen3.6-27B, and Gemma-4-31B-it under
per-model port gates, and opens a repaired mechanism (contrast estimand,
anchored precheck, gated controls) only where identifiability is possible
by construction — one frozen VM campaign of ~6–11 GPU-hours, every honest
stop a banked result.

---

*End of preference_2_2.md — operative candidate plan awaiting PI
acceptance (§10.1). The Phase 2 preregistration must pin every §10.1
number before frozen outcomes.*
