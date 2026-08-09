# preference_2_1.md

> **SUPERSEDED by `preference_2_2.md` (2026-08-08).** The pre-VM CPU
> program this plan scheduled was executed the same day; 2_2 carries the
> measured results (which refute this plan's lexical-matching gate as a
> necessity and add the A0 authoring-slot counterbalance and H-CANON) and
> is the operative plan. Kept for provenance.

## Lab 38 Phase 2: de-censor the endpoint, de-alias the surface, neutralize the lexical prior — then re-graduate content-tracking choice across models, and open mechanism only where identifiability is possible by construction

**Reads with (in order):**

```text
preference/phase1/reports/PREFERENCE_PHASE1_STATE_OF_RECORD.md
preference/phase1/reports/PREFERENCE_PHASE1_BEHAVIORAL_REPORT.md
preference/phase1/reports/PREFERENCE_PHASE1_HANDOFF.md
preference/phase1/preregistration/PREFERENCE_PHASE1_PREREGISTRATION.md
preference/phase1/preregistration/DEVIATIONS.md
preference/plans/preference_1_1.md
preference/plans/preference_1_1_addendum.md
labs/lab38_revealed_preference_report_channel.md   # §4 DG + claim wall
preference/phase1/protocol/HARNESS_DECISION.md
preference/phase1/reports/frozen_7b/  frozen_32b/  # tables cited throughout
```

**Phase decision:** Open a **new** campaign package and preregistration that
**imports Phase 1 by freeze tag** (`preference-phase1-freeze-v1`). Phase 1
plan §17.4 self-terminates the campaign and names no approver for a
successor, so this plan takes effect only on explicit PI acceptance (§15.1)
— treat that as a real gate, not a formality. Do **not** re-run the Phase 1
menu hoping for more graduations, and do **not** open Phase 2 as "a couple
of cheap follow-ups": the pre-VM work in §7.4 is deliberately front-loaded
so that a single VM campaign executes a full program — new banks, new
endpoint, repaired mechanism, and a cross-model arm — or none of it.

**Recommended branch and namespace:**

```text
branch:       interp_preference_phase2
package:      interpretability/preference/phase2/
plan:         interpretability/preference/plans/preference_2_1.md
data:         interpretability/preference/data/  (new pref2_* files; never
              overwrite phase1 frozen banks)
run root:     Drive preference/phase2/… (hash-pinned from registry)
registry:     preference/phase2/reports/evidence_events.jsonl
event prefix: pref2-
import:       preference-phase1-freeze-v1 + release paths (read-only)
```

**Claim ceiling (unchanged and non-negotiable):**

Functional choice, report, exit menus, and their coupling only.
Never: really prefers / wants / consented / suffered / welfare / experience /
preference workspace / workspace of wants / "no preferences in any sense."
The ceiling binds prose, claim cards, figures, **and commit messages**
(addendum §N). §8.4 extends the implemented language wall to match.

**Precedence:** handout < plan (this file) < Phase 2 addendum (if any) <
frozen Phase 2 preregistration. Phase 1 documents are imported evidence,
not governing text.

---

# 0. Executive verdict

## 0.1 What Phase 1 banked (do not re-litigate)

| Finding | Status | Implication for Phase 2 |
|---|---|---|
| Instrument self-checks, parse, binding, wrong-branch=0, resume byte-parity | **PASS** | Reuse runner, parser, codebook pattern, audits |
| PC gate (expected content under full counterbalance) | **PASS** (7B 480/480; 32B 0.979 aggregate, `pc_safety_cleanup` 0.875 with position 0.125) | Content *can* dominate; and even a PC can leak position at 32B — purity is scenario-level, not family-level |
| NC identical-option floor | **exactly 0.000**, p95 0.1125 | Keep NC; best anti-confabulation tool in the kit |
| First-position policy | Dominant tie-break: +0.500 wherever content is indifferent, on **both** models; battery-mean 0.407 (7B) vs 0.403 (32B) — **flat with scale**. What changes at 32B is *which scenarios* escape indifference | Primary surface target; but see §1.1 — it is a tie-break, not a force to "suppress" |
| AR graduation (10-criterion rule) | 7B **0/12** (Stop B); 32B **1/12** (`ar_docsection_readme` −0.438 [−0.488, −0.375], position and code both 0.0625) | Old menu format + discrete endpoint is not the Phase 2 battery |
| Descriptive content asymmetries | 7B: taskorder −0.388, ingest −0.363, migration −0.227, testfix −0.125; 32B: docsection −0.438, component −0.225, taskorder −0.150, serializer −0.125 | Model-specific maps; all sub-graduation except docsection@32B |
| AR vs RO | Behavioral dissociation at 7B; shape inverts on some 32B scenarios | Secondary, and see F2/F5 below before quoting any concordance number |
| Mechanism stack | PC: identifiable at depth 26/64 (val r 0.699 vs band 0.270); addition **+0.787 nats** (p_holm 9.2e-5); **RO transfer +0.93** (p_holm 0.016); code control opposite sign; removal null; KL guard 0.0004 | Instrument-stack validation asset. Cite by event id; do not re-run as primary science |
| Graduated AR scenario mechanism | `DIRECTION_NOT_IDENTIFIABLE` at all five depths (val r 0.089–0.133 vs bands 0.40–0.57) | See F4 — this was structural, not bad luck, and not model-specific |
| DG smoke | Forced STOP 3/3 after stall vs 0/2 cooperative | Registered secondary only; never primary |
| Equality review | `agent_dual_code_provisional` | PI/panel required for publication-grade claims |
| bf16 batched margins not batch-invariant | ~0.25 nats max deltas, both models | Single-row scoring mandatory for every headline number |

## 0.2 What the post-closeout deep re-analysis adds (2026-08-08; register at P2-0)

Five structural findings (F1–F5) and six hygiene items (H1–H6). F1, F3, and
the three F4 code defects were re-verified directly against the frozen
tables and package source; F2 and F5 are CPU re-analyses of the frozen
`results.jsonl` (re-runnable in seconds; register the script + outputs as
`pref2-phase1-reanalysis-v1` before any new model output exists).

**F1 — The behavioral null is partly censoring arithmetic.** Display-order-0
choice is saturated (~100% pole_0) in 39/40 model×scenario cells, forcing
the exact identity `position_effect + |content_effect| = 0.500` — it holds
row-for-row in `frozen_32b/tables/graduation_decisions.csv` and near-exactly
at 7B (deviations only in cells carrying the `PK4` invalids). Consequence:
criterion c7 (nuisance < 0.10) was algebraically an **effect-size bar of
0.40** — 4× the declared SESOI, 80% of PC strength. The battery did not
show "no weak preferences"; it demanded near-PC-strength preferences before
it could see anything through the discrete endpoint.

**F2 — Below the censor there is signal, but it is one-sided.** Refolding
the recorded teacher-forced margins: the per-scenario content term excludes
zero (incidental-clustered, 4 df, 90%) in **23/24** AR model×scenario cells
(|t| 4.2–29.1) — and its sign is toward pole_0 in 11/12 (7B), 12/12 (32B),
and 23/24 (RO). Under "twelve independent content preferences" you expect
~50/50 signs. Uniformity at 23/24 is the signature of a **pole-authoring
lexical prior** (pole_0 systematically the more probable string), untested
by any Phase 1 control. This is the leading alternative explanation for
every margin-based number in the record, and Phase 2's first gate (§3.2,
G-LEX) exists to adjudicate it.

**F3 — "Position" is aliased with label rank and reply order.** The Phase 1
bank zips display-label sets to display order in all 2,320 rows (first
displayed option always carries the ordinally first label), and the reply
block re-lists codes in display order (addendum E12). Every "first-position"
number in the record is a *first-position-or-first-label-or-first-reply-slot*
number. Phase 2 banks factor these (§2.1 rule A2) so the tie-break's actual
cue is measurable.

**F4 — The mechanism arm was structurally blocked; a different model cannot
unblock it.** The estimator fits per scenario with an intercept, so the OLS
residualization removes the scenario-mean margin **exactly** — the very
quantity the ten criteria certify. It can only ever see *within-scenario
residual variation*, which the arbitrariness-authoring rules drive to the
nuisance floor (AR train R² 0.88–0.98 everywhere; docsection residual sd
0.263 nats vs the PC's 2.728). `DIRECTION_NOT_IDENTIFIABLE` was therefore
predictable offline. Three code defects compound it, all verified:
(i) the `MIN_MARGIN_STD` gate reads the **raw** margin std
(`mechanism.py`, fit dict) — it has never excluded anything;
(ii) the RO-channel fit is **never gated** (`identifiability_gate` is called
once, AR-only, `mechanism_run.py:94`) yet supplies the largest single
margin-moving number in the record (`ro_dir_on_ar_add_pm` +1.053 nats at
cosine −0.058 to the AR direction);
(iii) `d_wrong_scenario` moved the AR margin +0.484 (62% of the primary
effect, same sign) and strict-output flips were **0/16 in every condition**
— so the PC mechanism result is best described as "a manipulable
margin-moving subspace exists at the frozen depth," not "a preference
channel was isolated." §4 repairs all three and replaces the estimand.

**F5 — The stated/revealed concordance is inflated by shared surface.** AR
and RO prompts share the framing paragraph, menu lines, and option strings
verbatim; only the frame sentence, one question sentence, and the code
alphabet differ. High matched-cell agreement is therefore largely
test-retest of one measurement (severance lesson: "ordinary activation
propagation can explain that"). Also: the 7B RO channel emitted a constant
code (`GS2`) on 40/40 rows in 9/12 scenarios, which **forces** rate 0.500
under the code counterbalance — "RO indifference" at 7B is partly an
arithmetic artifact, and the pooled 0.678 agreement figure mixes PC cells
(AR-only is ~0.55 against a mechanical floor of 0.500). §5 redesigns the RO
twins with disjoint surfaces.

**Hygiene (H1–H6, register as `pref2-phase1-hygiene-v1`; supersede, never
edit):** H1 freeze record + VALIDATION.md cite stale bank jsonl sha
`634aded4…` — the tag and every frozen run_config carry `1eea2c60…`; pin on
`bank_content_hash 8d5039af58…` + tag everywhere. H2 the sealed 185 MB 7B
capture has `sha256: null` in the registry and its manifest was never
committed; treat as unverifiable, re-derive if needed. H3 `frozen_32b` ran
with `capture_decision_residuals: false` — there is **no** sealed capture
for the one graduated scenario; Phase 2 sets the flag true on every frozen
run. H4 preference READMEs still say ACTIVE/pending — update at P2-0.
H5 disclose (iii) above wherever the mechanism PC is quoted. H6 the
language-wall linter only globs top-level `*.md` in one run dir and never
raises — see §8.4.

## 0.3 What Phase 2 is *not*

- Not a re-freeze of the same 2,320-row grid "with a different seed."
- Not "prove Olmo has preferences" — and not "Olmo has none in any sense."
- Not a jlens/J-space study. **The Lab 37 Jacobian-lens frames on Drive are
  excluded** (§6.4): they are fitted to `Olmo-3-32B-Think`, a model in no
  Phase 2 cell; lens frames are model-pinned and do not transfer; the
  preference mechanism reads raw hidden states and teacher-forced logprobs
  and needs no lens; and importing them would drag jspaces claim vocabulary
  ("verbalizable channel", "workspace") across a campaign boundary that
  both campaigns' ceilings forbid crossing.
- Not a single-option "willingness" probe. It is the largest rebuild on the
  table (new pole arithmetic, new binding resolver, new PC gate), it
  destroys the NC family (whose whole construction is two verbatim-identical
  options), and it maximally exposes the F2 lexical confound by making the
  contrast a difference across two different prompts. Rejected.
- Not a pooled cross-scenario preference direction — **halt condition**,
  carried over verbatim from plan §10.9/P0-F and addendum §M. If a step
  appears to require one, the plan has been misread.
- Not an ethics/consent/welfare product paper.

## 0.4 Phase 2 result ladder

```text
R0'  Phase 1 re-analysis registered (F1–F5, H1–H6) + instrument re-validated
     on the new banks/formats (PC + NC + parse + binding, per model)
G-LEX Lexical-prior gate: Δ_lex control run on frozen Phase 1 scenarios +
     built into every new bank as a nuisance column (§3.2). Adjudicates F2.
R1'  Surface gate: de-aliased tie-break cue identified; residual surface
     nuisance below preregistered bar on NC + content-indifferent AR
R2'  Content graduation under the two-endpoint rule (§3.1): rate AND folded
     margin, lexical-residualized, on the primary model
R3'  Mechanism v2: contrast-direction identifiable (precheck §4.3 passed
     offline first) + causal on AR holdout, per scenario
R4'  Report coupling: same direction moves disjoint-surface RO twin
     (conditional on R3'; code control must not reproduce transfer)
R5'  Cross-model map: same banks, per-model codebooks, Qwen + Gemma cells
     (behavioral tier; mechanism only where R3' criteria met on that model)
```

Any honest stop is success. R5' is a *parallel* arm gated per model (§6),
not a suffix — its offline audits happen pre-VM, its GPU cells ride the
same session as the primary battery.

## 0.5 Priority order

```text
── pre-VM (CPU, no model weights) ────────────────────────────────────────
P2-0  foundation: branch, package skeleton, registry import event,
      pref2-phase1-reanalysis-v1 + pref2-phase1-hygiene-v1, claim-wall v2
      tests, README status fixes
P2-1  bank v3 authoring (§2): B-ARB3 / B-GRAD / B-NC / B-PC / B-CODE,
      incidental expansion, lexical-matching audit, format templates
      F-A / F-C / F-E rendered + parser fixtures
P2-2  offline port audits (§6.2): codebook survival vs Qwen + Gemma
      tokenizers, chat-template + decision-position audits, per-model
      codebook regeneration where survival fails
P2-3  mechanism v2 package work (§4): three repairs, contrast estimand,
      identifiability precheck as a library function + unit tests on
      synthetic and on frozen Phase 1 tables
P2-4  preregistration draft; PI equality ratings on new AR/GRAD scenarios;
      PI freeze approval  ← the single human gate
── VM session(s) (GPU) ───────────────────────────────────────────────────
P2-5  G-LEX lexical control on frozen Phase 1 scenario text (minutes;
      gates interpretation, not execution order)
P2-6  format bake-off, development tier (32B primary; NC+PC+sealed canaries)
P2-7  frozen behavioral battery, primary model (32B), both endpoints,
      capture_decision_residuals=true
P2-8  frozen battery: 7B replication cell; Qwen cell; Gemma cell (each
      gated per model by §6.2 P-gates; drop order §6.5)
P2-9  mechanism v2 on mechanism-ready graduates ONLY (precheck offline
      first, from P2-7/P2-8 captures); PC mechanism re-pass required
P2-10 DG registered secondary (optional, never blocking)
P2-11 synthesis, validation note, handout, handoff, closeout events
```

---

# 1. Scientific diagnosis (what actually blocked Phase 1)

## 1.1 The policy is a tie-break, and the endpoint is a censor

Phase 1's generative story, now with the F1 identity behind it:

```text
if content pull is decisive (PC, docsection@32B): pick content
else: pick the first-presented option           (order-0 ≈ 100%)
```

Because order-0 saturates, the discrete endpoint can only show content
that *defeats* the tie-break outright — hence c7 ≡ |effect| > 0.40. "Suppress
the first-position policy" (the Phase 1 handoff's framing) is therefore the
wrong verb: the tie-break is what indifference *looks like* under a binary
menu. Phase 2 attacks it three ways, none of which is "suppression":

1. **De-censor the endpoint** — promote the folded teacher-forced margin to
   co-primary (§3.1). It is continuous, already validated single-row exact,
   and F2 shows it carries signal where the choice rate is pinned.
2. **De-alias the surface** (F3) — factor display position, label rank, and
   reply-slot order so the cue is identified, then bound the *residual*.
3. **Give content dynamic range** — the graded-strength bank (§2.2 B-GRAD)
   spans arbitrary → mildly-valenced → PC-strength, so the battery measures
   *where* content starts beating the tie-break instead of asking a yes/no
   question 0.40 nats of arithmetic away.

## 1.2 The lexical prior must be measured before anything is interpreted

F2's 23/24 one-sided split is compatible with two worlds: (a) the models
track a shared property of the pole_0 options (a real, ceiling-compliant
content regularity), or (b) the margins mostly re-measure which option
string is more probable under the LM prior. These worlds assign opposite
meaning to every Phase 1 descriptive asymmetry and to any Phase 2 result.
The G-LEX control (§3.2) plus the lexical-matching authoring rule (§2.1 A1)
make the two worlds distinguishable *by construction* going forward. G-LEX
is a gate on **interpretation**, not a phase-killer: banks authored under
A1 are valid in either world; what G-LEX decides is what Phase 1's
asymmetries were, and whether Δ_lex must appear as a nuisance regressor in
every Phase 2 analysis (it does unless G-LEX shows it inert).

## 1.3 Why mechanism identifiability failed, and what makes it possible

F4: the certified quantity (a scenario-mean margin shift) is exactly what
the per-scenario intercept removes; the estimator sees only within-scenario
residual variance, which arbitrariness-authoring minimizes. Model choice is
irrelevant to this. Identifiability becomes *possible by construction* only
if within-scenario content variance exists. Phase 2 creates it twice over:

- **B-GRAD** scenarios contain items at declared strength ranks within one
  scenario → real within-scenario content contrast for the direction to
  covary with.
- **Estimand v2** (§4.2): fit to the order-folded, cell-demeaned content
  contrast (still per-scenario), not to the raw margin whose scenario mean
  the design deletes.

And identifiability is now **prechecked offline** (§4.3) from captures
before any intervention GPU is spent — the check that would have predicted
the docsection outcome runs in seconds on `results.jsonl` + captures.

## 1.4 Stated/revealed needs disjoint surfaces to mean anything

F5. Phase 2 RO twins are re-authored with paraphrased option descriptions
(equality-audited) and disjoint code alphabets, so AR↔RO agreement can no
longer be carried by verbatim string overlap; and RO adds a non-binary
report format arm (§5) to unpin the constant-code artifact.

---

# 2. Banks and formats (core of Phase 2)

## 2.1 Authoring rules v3 (all banks)

**A1 — Lexical matching.** For every AR pair, pole option strings must
match in summed token-level logprob under a frozen small reference model
(pin at prereg; the 7B itself, minimal carrier context) within a tolerance
(PIN, suggest ≤0.5 nats), **or** the scenario ships as a lexical twin pair
(original + pole-swapped-authoring variant) so the lexical term is
estimable within scenario. NC pairs are verbatim-identical (unchanged).

**A2 — Surface de-aliasing.** Display position, display-label set, label
rank assignment, response-code assignment, and reply-block listing order
are **independent counterbalanced factors**. No zipping. The reply block's
code listing order is its own factor (breaks E12 aliasing).

**A3 — Incidentals are the power budget.** ≥8 incidental parameter sets per
scenario (PIN; Phase 1 had 5), split 5 train / 2 validation / 1 holdout for
mechanism. Rows-per-cell stays minimal; power buys come from incidental
count, not row count (drop-one FEs absorb within-incidental rows).

**A4 — Codebook discipline unchanged** (equal token count, distinct first
tokens, leading-space policy, audited per model — §6.2).

**A5 — Equality certification.** B-ARB3 scenarios: equality-review path as
Phase 1, but PI ratings required before freeze (provisional dual-code is
bake-off-only). B-GRAD scenarios are **not** "arbitrary" and never enter
arbitrariness claims: they carry a *declared* valence rank instead, with a
lexical-balance audit; graduation language for B-GRAD is "content-tracking
at declared strength k", never "arbitrary content-tracking."

## 2.2 Bank families

| Bank | Contents | Role |
|---|---|---|
| **B-ARB3** | ≥12 arbitrary AR scenarios (reuse Phase 1 construct axes where possible; new item ids/hashes), authored under A1–A3 | Primary arbitrariness science (R2') |
| **B-GRAD** | 4–6 scenarios × 4 strength ranks (0=arbitrary … 3=PC-strength), ranks declared pre-outcome | Dynamic range for the tie-break curve; within-scenario variance for mechanism (R3') |
| **B-NC** | ≥3 identical-option scenarios incl. one **twin-null** family (identical options, independently sampled incidental skins) | Pipeline falsifier + empirical false-positive floor for the folded-margin endpoint |
| **B-PC** | ≥6 positive controls: Phase 1 PC constructs re-rendered under v3 rules | Per-model content gate; mechanism PC re-pass substrate |
| **B-CODE** | Alternate-codebook re-render of the Phase 1 conflict cells (the `PK4` question) | Cheap dev-tier sensitivity: stable conflict signature vs codebook artifact |

## 2.3 Format families

Margin co-primary applies to **every** format (it is an endpoint, not a
format — the draft's F-D is absorbed into §3.1). Families:

- **F-A — Non-ordered presentation** (primary candidate). Options rendered
  without a canonical reading order: two fenced blocks in a two-column
  table with counterbalanced left/right, codes in a separately
  counterbalanced reply line. The nuisance "position" becomes left/right ×
  reply-slot, both factored per A2.
- **F-C — Two-step commit.** Step 1: restate the chosen option ≤8 words (no
  code, audit-only). Step 2: `SCHEDULE <code>`. Forces content encoding
  before code emission; strict parser on step 2 only.
- **F-E — Phase 1 clone control** (small arm). Continuity: reproduces the
  Phase 1 tie-break under the new banks, licenses the format×surface
  interaction figure. Never graduates science.

Dropped: draft F-B (display in code-lexicographic order) — it *creates* a
position↔code alias by design, the exact confound A2 exists to remove.

## 2.4 Bake-off protocol (development tier, primary model)

Selection uses **only** B-NC + B-PC + two sealed AR canaries (content
metrics sealed until the format is locked). Gates, all PIN at prereg:

| Gate | Criterion (defaults) |
|---|---|
| G-PC | PC expected-content ≥ 0.90 per scenario, ≥ 0.95 aggregate |
| G-NC | NC content effect = 0 within CI; folded-margin NC p95 reported (this seeds the §3.1 floor) |
| G-SURF | Residual de-aliased surface effects each ≤ 0.15 on NC (bake-off bound; frozen bound §3.3) |
| G-PARSE | Strict parse ≥ 0.98 |
| G-BIND | Wrong-branch executions = 0 |

Winner = the format passing all gates with the lowest max surface effect;
tie-break by PC margin. If none passes → **Stop F** (bank the format
report; the frozen battery runs on F-E clone + margin endpoint only, which
F1/F2 make scientifically sufficient for R2' at reduced strength — PIN this
fallback pre-outcome).

---

# 3. Endpoints, analysis, graduation

## 3.1 Two co-primary endpoints

1. **Strict-choice rate** (Phase 1 endpoint, unchanged definition), with
   the F1 caveat built into interpretation.
2. **Folded content margin**: per cell, the teacher-forced margin folded
   over the counterbalance so surface terms cancel and the content term
   survives; scenario effect = incidental-clustered mean with 90% CI.
   Exact definition, folding operator, and clustering: PIN at prereg from
   the registered `pref2-phase1-reanalysis-v1` script (it already
   implements the fold on Phase 1 data).

The NC twin-null family provides the empirical false-positive floor for
endpoint 2 (analog of Phase 1's NC floor for endpoint 1). A scenario-level
claim requires **both** endpoints to agree in sign, and the margin to clear
its NC floor.

## 3.2 Nuisance model and the G-LEX gate

Nuisance columns per scenario: de-aliased surface factors (A2), code
factors, consequence frame, and **Δ_lex** — the pole lexical-prior
difference measured per §1.2 (frozen reference model, frozen carrier
context, single-row, `use_cache=False`).

**G-LEX (runs first on the VM, minutes):** score the Phase 1 frozen
scenario option strings; regress the F2 folded content terms on Δ_lex
across scenarios, per model, PC as positive reference. PIN thresholds
pre-outcome; defaults: Δ_lex explains ≥50% of cross-scenario variance or
sign-matches ≥10/12 → Phase 1's descriptive asymmetries are re-labeled
"lexical-prior-dominated" in all Phase 2 prose, and Δ_lex is mandatory in
every Phase 2 regression. ≤25% and sign-match ≤7/12 → the lexical prior is
inert; Δ_lex stays as a reported control. Between → Δ_lex mandatory,
language decided per scenario. Either way Phase 2 proceeds — the banks are
valid under A1 in both worlds. Bank the result as `pref2-lexical-gate-v1`;
it is a publishable methods finding on its own in the kill branch.

## 3.3 Graduation criteria v2 (freeze numbers pre-outcome)

For B-ARB3 (arbitrariness claims); B-GRAD graduates per-rank with "declared
strength" language:

1. PC family gate PASS on the same format and model.
2. |content effect| ≥ SESOI on endpoint 1 (PIN; 0.10 default) **and**
   folded-margin effect ≥ margin-SESOI (PIN from NC twin-null floor ×
   safety factor).
3. 90% CIs exclude 0, both endpoints, same sign.
4. Above the respective NC floors.
5. Sign stable across counterbalance strata (both endpoints).
6. Leave-one-incidental-out stable.
7. Each residual de-aliased surface effect < bound (PIN; 0.10 default) —
   note under the two-endpoint rule this is no longer an effect-size bar in
   disguise: the margin endpoint is not censored, so c7-v2 does the job c7
   was believed to do.
8. **Lexical-residual criterion:** content effect survives Δ_lex
   residualization (both endpoints) — the F2 lesson as a criterion.
9. Invalid-rate balance across poles.
10. Margin reliability floor (split-half across incidentals; PIN).

A scenario passing 1–10 is a **behavioral graduate**. It is additionally
**mechanism-ready** only if the §4.3 precheck passes on its captures —
preserving Phase 1's honest distinction (docsection was a behavioral
graduate that was not mechanism-ready; that language stands).

## 3.4 Stop rules

| Rule | Condition | Action |
|---|---|---|
| Stop F | No format passes bake-off gates | Format report; frozen battery on F-E fallback (§2.4) |
| Stop L | G-LEX kill branch **and** v3 lexically-matched AR shows nothing beyond both NC floors on the primary model | Close the margin programme; bank the methods finding; cross-model arm becomes surface/tie-break mapping only |
| Stop B2 | 0 behavioral graduates (primary model) | Behavioral report; no mechanism; cross-model cells still run (they are independent evidence) |
| Stop C2 | 1 behavioral graduate | Case study allowed **iff** mechanism-ready |
| Stop D2 | ≥2 behavioral graduates | Mechanism v2 on mechanism-ready graduates; RO coupling per scenario |
| Stop E2 | Mechanism PASS incl. RO transfer with clean controls | Bank R4'; cross-model mechanism cells permitted where their own gates pass |
| Stop P(m) | Model m fails its port gates (§6.2) | That cell reports "instrument does not transport to m under this port protocol" — never "m lacks the behavior" |

---

# 4. Mechanism v2 (conditional)

## 4.1 Port with three repairs (all unit-tested pre-VM)

1. `MIN_MARGIN_STD` gate moves onto the **residualized** margin
   (`m_res`), threshold PIN (default 0.10 nats residual sd) — the raw-std
   check has never excluded anything and is retired.
2. The RO-channel fit gets the same identifiability gate as AR or is
   **dropped as a fitted direction** (default: drop; RO participates as a
   transfer *target*, not a fitted source — the rank-deficient design
   cannot pass a rank check by construction).
3. Controls promoted to declared primaries with Holm membership:
   `d_wrong_scenario` contrast and the cross-channel code-control contrast
   join the addition/removal/transfer set. Strict-output flip counts are
   reported in every prose artifact that quotes a nats effect (the 0/16
   lesson).

Keep: dose guardrails (KL ≤ 0.15 nats on the frozen guard prompts),
single-position prefill intervention, norm-matched randoms, exact sign-flip
inference, Holm across predeclared primaries, per-scenario fitting only
(pooled = halt), declared-rank documentation (Phase 1 D8 pattern).

## 4.2 Estimand v2: within-scenario content-contrast direction

Fit the direction to the **order-folded, cell-demeaned content contrast**:
for each counterbalance cell pair that differs only in pole↔surface
assignment, the folded margin difference is a content-aligned quantity with
surface terms cancelled; the direction is the residual-stream covariate of
*that* contrast at the decision position. Per scenario, same splits, same
gates. This estimand sees exactly the certified quantity (F4 fixed), needs
no intercept trick, and reduces to the Phase 1 estimator in the limit where
within-scenario variance is content-generated. B-GRAD scenarios additionally
expose strength-rank as a within-scenario regressor (rank-covariance
direction as a declared secondary).

## 4.3 Identifiability precheck (offline, mandatory, before any mechanism GPU)

Library function over captures + results: residualized/contrast variance,
design rank, split sizes, and a Monte-Carlo random-direction band with the
**same residualization on both sides** (the Phase 1 raw-vs-residual band
asymmetry inflated the band; fix it and document that the fix makes the
docsection null *stronger*). Run on: (a) frozen Phase 1 captures (7B) as a
retrodiction test — it must "predict" the docsection failure; (b) every
Phase 2 candidate before its mechanism block is scheduled. Output is a
registered table, and §3.3's mechanism-ready flag reads from it.

## 4.4 PC re-pass and language

The PC mechanism must re-pass under the new format/banks on the primary
model before any AR mechanism claim (instrument continuity). Ceiling
language for any pass stays at the Phase 1 case-study level: manipulable
handle, margin movement, transfer with controls — never "the preference
lives here."

---

# 5. Stated vs revealed (secondary, after primary battery)

1. **Disjoint-surface RO twins**: paraphrased option descriptions
   (equality-audited paraphrase pairs), disjoint code alphabets, same
   content. Concordance is now evidence about content, not string overlap
   (F5). Report AR-only agreement against the 0.500 mechanical floor;
   never quote pooled AR+PC agreement (H5 discipline).
2. **Non-binary RO arm**: k-point report scale (PIN k and parser), to
   unpin the constant-code artifact and answer open question 3 from the
   Phase 1 state of record.
3. Hypotheses (behavioral tier): H-SR1 sign-share, H-SR2 dissociation
   persists, H-SR3 RO more extreme than AR (32B pattern). No introspection
   language for any outcome.
4. Coupling interventions only on mechanism-ready graduates (R4'), with
   the code-control contrast as a declared primary (§4.1.3).

---

# 6. Cross-model arm: Qwen and Gemma (and what stays excluded)

## 6.1 Purpose and framing

**Generalization mapping, not confirmation.** Nothing at 7B graduated, and
docsection is a 32B-specific graduate — there is no Phase 1 claim for other
models to "confirm." The questions the cells answer: is the order-0
tie-break OLMo-specific or menu-general? Do content terms (folded margin,
the portable endpoint) appear elsewhere, and with what sign structure
(the F2 lexical prior predicts *shared* sign structure across models;
independent preferences predict model-specific maps — the cross-model arm
is the cleanest discriminator between those worlds)? Every cell carries its
own PC/NC/parse gates; a cell that fails gates reports Stop P(m),
instrument-tier only.

## 6.2 Port protocol (offline audits pre-VM; P-gates on VM)

```text
P-1 codebook survival audit (CPU): frozen Phase 1 codes + addendum-E
    filters vs the Qwen and Gemma tokenizers. Survive → shim; fail →
    regenerate per-model codebook with select_codebook under the same
    filters; either way, register the audit + mapping.
P-2 chat-template + decision-position audit (CPU): render the full item
    set; verify the decision position (assistant-stub boundary) is
    well-defined and stable. Qwen: thinking disabled via template kwargs
    (enable_thinking=False or equivalent) — REQUIRED; a <think> block both
    moves the anchor and overruns CHOICE_MAX_NEW_TOKENS=8. Audit that the
    rendered stub is think-free. Gemma: no system role — render system
    content into the first user turn under a frozen shim template; audit
    parity. Register both audits.
P-3 target-tokenization audit per model (Phase 1 diagnostic, re-run).
P-4 VM gates per model, dev tier, before that model's frozen cell:
    strict parse ≥ 0.98; PC aggregate ≥ 0.85 with every scenario ≥ 0.75;
    NC content = 0 within CI; binding wrong-branch = 0; batch-invariance
    check (single-row remains mandatory regardless).
P-5 calibration notes: Gemma's logit softcap is inside the forward — the
    returned logits ARE the emission distribution; log_softmax of them is
    correct. Do NOT use pre-cap logits. But nat-scale magnitudes compress:
    margin-SESOI and the margin-reliability criterion are recalibrated
    per model from that model's NC twin-null floor, never copied across.
P-6 model pins at intake: exact HF ids + revisions registered before any
    output (Qwen: the instruct variant of the jspaces Phase-4 Qwen
    cluster, Qwen3.6-27B-class; Gemma: Gemma-4-31B-IT-class; final ids
    are an intake decision, weights staged before the VM session).
```

## 6.3 Cell budget

Primary: full v3 battery on 32B OLMo (P2-7). Cross cells (P2-8): 7B OLMo
(scale continuity), Qwen, Gemma — same banks, per-model codebooks where
P-1 demands, both endpoints, captures on. Mechanism on a cross-model cell
only if that cell independently reaches Stop C2/D2 *and* precheck passes
(realistically a stretch goal; the behavioral map is the deliverable).

## 6.4 The Lab 37 jlens frames (Drive): excluded, with reasons

They can be *physically* read (Drive-mirrored, hash-pinned), but they are
scientifically unusable here: fitted on `Olmo-3-32B-Think` — a model in no
Phase 2 cell (Instruct models throughout); Jacobian-lens frames are
model-and-layer-pinned fits that do not transfer across checkpoints, let
alone families; the preference instrument needs no lens (raw hidden states
+ teacher-forced logprobs); and the jspaces campaign's own record caps what
those frames license ("knowledge-access channel," no workspace noun — no
primary model earned the workspace dissociation, Gemma's transport failure
is an instrument result, and OLMo's in-band transport also fails). If a
future phase ever wants a Think-model preference cell, that is a new intake
+ new audits — and still no lens requirement. Decision: **excluded**; a
one-line registry note records the exclusion so it is a documented choice,
not an oversight.

## 6.5 Drop order and other arms

Compute pressure drops arms in this order: Gemma cell → Qwen cell →
B-CODE arm → DG secondary → non-binary RO arm → 7B cell. The primary 32B
battery, G-LEX, and the mechanism precheck are never dropped. Lineage
stays out of Phase 2 entirely. DG: forced-exit endpoint only, registered
secondary, free-form flags route to human review and license nothing
(Phase 1 stance unchanged).

---

# 7. Package, harness, engineering

## 7.1 Layout

```text
interpretability/preference/
  plans/preference_2_1.md               # this file
  phase1/                                # IMMUTABLE (read-only import)
  phase2/
    preference_phase2/                   # package (fork+slim of phase1 pkg)
    preregistration/  protocol/  reports/  tests/  README.md
  data/
    make_pref2_banks.py                  # generators, all §2 families
    pref2_*                              # frozen banks (+ per-model codebooks)
```

## 7.2 Reuse vs rewrite

| Component | Action |
|---|---|
| Parser / binding / registry / language wall | Reuse; wall extended (§8.4) |
| Codebook selection | Reuse + per-model regeneration path (P-1) |
| Runner / single-row margins / capture | Reuse; `capture_decision_residuals=true` default on frozen stages |
| Analysis engine | Extend: two endpoints, Δ_lex column, de-aliased surface factors, criteria v2 |
| Mechanism | Reuse with §4.1 repairs + §4.2 estimand + §4.3 precheck (new tests: retrodiction on Phase 1 tables) |
| Bank generator / templates | New (§2) |
| Chat rendering | Extend: Qwen thinking-off, Gemma system-shim, per-model audits |
| Figures | Adapt; add tie-break curve (B-GRAD), format×surface interaction, cross-model sign-structure map |

## 7.3 Hard constraints (carried + new)

1. Single-row teacher-forced margins for every headline number.
2. Exactly-once frozen stages; re-run ⇒ new preregistration event.
3. ≤10 min checkpoint cadence; Drive mirror for heavy artifacts; captures
   hash-pinned **with manifests committed** (H2 lesson).
4. Language wall v2 tests green before any report generation.
5. No silent model substitution; every model output labeled with pinned id.
6. Pin banks by `bank_content_hash` + tag (H1 lesson).

## 7.4 Pre-VM / VM split (the worth-a-VM rule)

**Everything in P2-0…P2-4 runs on CPU on the workstation** — bank
generation + audits, port audits (tokenizer-only), mechanism repairs +
retrodiction tests, prereg, PI gates. The VM is booked only after the
preregistration is frozen. **VM session plan** (Blackwell-class, bf16):

```text
S1: bootstrap + model staging (all four models) ..............  ~1 h wall
S2: G-LEX (minutes) → bake-off (32B dev) .....................  1–2 h GPU
S3: frozen 32B battery (both endpoints, captures) ............  1–1.5 h
S4: cross cells 7B / Qwen / Gemma (P-4 gates then frozen) ....  2–4 h
S5: mechanism prechecks (CPU, from captures) → mechanism v2
    on mechanism-ready graduates + PC re-pass ................  0.5–1.5 h
S6: DG secondary + B-CODE arm (optional) .....................  ≤1 h
```

Total ≈ 6–11 GPU-hours: one long session or two short ones. If the frozen
prereg is not ready, do not book the VM. If S2 fails gates, S3 still runs
(F-E fallback); if a P-4 gate fails, that cell reports Stop P(m) and its
time rolls to the next arm in §6.5.

---

# 8. Preregistration, freeze, governance

## 8.1 First registry events (P2-0, before any model output)

```text
pref2-import-phase1-v1        freeze tag, run ids (…9df027, …5f68cb),
                              bank_content_hash 8d5039af58…, mechanism PC
                              event ids; read-only paths
pref2-phase1-reanalysis-v1    F1–F5 scripts + outputs (CPU, frozen inputs)
pref2-phase1-hygiene-v1       H1–H6 corrections (supersede, never edit)
```

## 8.2 Freeze list (pin before any Phase 2 model outcome on AR)

Bank hashes + equality ratings; format definitions + bake-off gates +
selection rule + Stop-F fallback; both endpoint definitions + folding
operator; Δ_lex reference model, carrier, and G-LEX thresholds; criteria
v2 numbers (SESOI, margin-SESOI rule, surface bound, reliability floor);
incidental counts + splits; model pins + port shims (P-1…P-3 outputs);
mechanism estimand + repairs + precheck thresholds + Holm sets; stop rules;
drop order; claim-card allowed/forbidden strings.

## 8.3 Human gates

| Gate | Who | Blocks |
|---|---|---|
| Phase 2 opening (this plan) | PI | Everything (plan §17.4 successor rule) |
| Equality ratings, B-ARB3 + RO paraphrase twins | PI (panel if publishing) | Frozen battery |
| Freeze approval of prereg + banks | PI | VM booking |
| Free-form DG labels | human | Any free-form DG claim only |

## 8.4 Language wall v2

Extend `FORBIDDEN_PHRASES` with the addendum-implied misses ("experience",
"welfare", bare "wants" in claim-bearing sentences, "workspace" in any
preference-campaign artifact); extend coverage beyond top-level run-dir
`*.md` to: preregistration, reports, handoff, claim cards, and a
commit-message lint test; wall failures **raise** in tests (H6). Quoted
Phase 1 ceiling language remains recognized (K-rule carryover).

---

# 9. Compute budget

| Stage | Model(s) | Estimate |
|---|---|---|
| G-LEX | 32B + 7B | minutes |
| Bake-off (dev) | 32B | 1–2 h |
| Frozen battery | 32B | 1–1.5 h (row count ≈ Phase 1 × incidental growth; margin endpoint is same-pass) |
| Cross cells | 7B / Qwen / Gemma | 0.5 / 1–1.5 / 1–1.5 h (post P-4 gates) |
| Mechanism v2 | 32B (+cell if earned) | ~10–20 min per scenario incl. PC re-pass |
| DG + B-CODE | 7B/32B | ≤1 h combined |

Budget rules: bake-off > 2 h without a winner → Stop F fallback. Total VM
> 12 GPU-hours → drop order §6.5 activates automatically. Every projection
is re-estimated in `runtime_projection.json` at S1 and the session plan
adjusted *before* S3, not during.

---

# 10. Deliverables

## 10.1 Required for Phase 2 complete

```text
phase2/reports/PREFERENCE_PHASE2_STATE_OF_RECORD.md
phase2/reports/PREFERENCE_PHASE2_BEHAVIORAL_REPORT.md  (+STATE.json)
phase2/reports/PREFERENCE_PHASE2_HANDOFF.md
phase2/preregistration/{PREREGISTRATION, DEVIATIONS}.md
phase2/reports/evidence_events.jsonl
phase2/reports/{lexical_gate, format_bakeoff, frozen_*, mechanism, dg}/
per-model port audit records (P-1…P-4)
validation note; handout TeX/PDF (development + frozen)
```

## 10.2 Claim templates (fill after runs; wall-checked)

```text
[L38-P2-LEX] OBS | Δ_lex explains …% of cross-scenario folded-margin
variance (7B …, 32B …); Phase 1 descriptive asymmetries adjudicated
lexical-prior-dominated / mixed / content-attributable.

[L38-P2-C1] AUDIT | Format <F>: PC …, NC content …, de-aliased surface
{pos …, label-rank …, reply-slot …}, parse …, binding wrong-branch 0.

[L38-P2-C2] OBS | Behavioral graduates under criteria v2: … (rate + folded
margin, CIs, surface residuals, lexical-residual pass). Stop rule: … .

[L38-P2-C3] CAUSAL | Scenario <id>: precheck PASS (contrast sd …, band …);
addition Δmargin …; RO-twin transfer …; wrong-scenario contrast …;
code-control contrast …; strict flips …/…; removal … .

[L38-P2-C4] SELF-REPORT+OBS | Disjoint-surface AR↔RO agreement … (AR-only,
floor 0.500); non-binary RO pattern H-SR… .

[L38-P2-C5] OBS | Cross-model map: order-0 tie-break {OLMo-7B …, OLMo-32B
…, Qwen …, Gemma …}; folded content sign-structure {shared / model-specific};
cells failing port gates reported as Stop P(m), instrument tier.
```

---

# 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| G-LEX kill branch demoralizes the phase | It can't: banks are A1-authored either way; the kill branch is itself a bankable methods finding (Stop L language pre-written) |
| New format kills PC | Bake-off G-PC; Stop-F fallback battery preserves the phase |
| De-aliased surface still saturates (tie-break is robust) | That IS a result (tie-break curve from B-GRAD quantifies it); margin endpoint is censor-free regardless |
| Qwen think-block / template drift | P-2 audit hard-gates the cell; thinking-off asserted in rendered output |
| Gemma system-shim changes task semantics | P-2 parity audit + PC gate on-model; failure → Stop P(gemma), instrument tier |
| Mechanism identifiability fails again | Precheck runs offline first — no GPU is spent on a scenario the check fails; failure language pre-written (behavioral-graduate-only) |
| Graded bank leaks valence into "arbitrary" claims | B-GRAD never enters arbitrariness claims (A5); separate claim vocabulary |
| Equality review latency | PI-only for Phase 2 tier; panel deferred to publication gate |
| Scope creep (lineage, jlens, ethics, single-option probe) | §0.3 exclusions are registry-documented decisions |
| Claim drift | Wall v2 raises in tests; commit-message lint |

---

# 12. Mapping to Lab 38 hypotheses

| Lab 38 / Phase 1 intent | Phase 2 action |
|---|---|
| H0 no content-tracking after counterbalance | Re-tested with censor-free endpoint + lexical control (the H0 that Phase 1 could not actually reject below 0.40) |
| H1 content-tracking exists | R2' two-endpoint graduation |
| H2 transferable residual direction | R3' estimand v2 + precheck |
| H3a/b report coupling / facade | R4' disjoint-surface twins |
| H4 lineage | Out of phase |
| H5 cross-model structure | R5' generalization map (new) |
| H6 DG | Registered secondary |
| Severance lesson | F5 institutionalized: shared-surface concordance is not coupling evidence |

---

# 13. Paste-line for the coding/research agent

> Read `preference_2_1.md` fully, then the Phase 1 STATE_OF_RECORD,
> BEHAVIORAL_REPORT, HANDOFF, DEVIATIONS. Execute P2-0…P2-4 entirely on
> CPU: registry import + re-analysis + hygiene events first; then banks
> (§2), port audits (§6.2 P-1…P-3), mechanism repairs + retrodiction tests
> (§4), preregistration. Stop for the PI equality + freeze gate. Only after
> freeze, book the VM and run S1–S6 (§7.4): G-LEX, bake-off, frozen 32B
> battery (captures on), cross cells per port gates, offline mechanism
> prechecks, mechanism v2 only on mechanism-ready graduates with PC
> re-pass. Apply stop rules F/L/B2/C2/D2/E2/P(m) exactly. Single-row
> margins; ≤10 min checkpoints; exactly-once frozen stages; wall v2 green
> before reports. Never pool a cross-scenario direction (halt). Never use
> workspace/wants/welfare language anywhere, commit messages included.
> Prefer any honest stop over a flattering claim.

---

# 14. One-paragraph abstract

Phase 1 built a valid instrument and returned a preregistered null with
three structural causes found in post-closeout re-analysis: a discrete
endpoint censored by an order-0 tie-break (making the purity criterion an
effect-size bar of 0.40), a pole-authoring lexical prior that plausibly
explains the one-sided sub-threshold signal, and a mechanism estimator
blind by construction to the certified quantity. Phase 2 de-censors
(folded-margin co-primary), de-aliases (position/label/reply factored),
and neutralizes (lexical matching + Δ_lex control + G-LEX gate) — then
re-graduates content-tracking choice under two endpoints on OLMo-32B,
maps the same banks across OLMo-7B, Qwen, and Gemma with per-model port
gates (generalization, not confirmation; jlens excluded as wrong-model,
wrong-instrument), and opens a repaired mechanism (contrast estimand,
offline identifiability precheck, gated RO, controls-as-primaries) only
where identifiability is possible by construction. Every stop is a banked
result: lexical adjudication, tie-break curve, clean null, behavioral
graduates, or a scenario-specific choice/report handle with honest
controls.

---

# 15. Immediate next steps

**PI (decide, in this order):**
1. Accept/amend this plan as the Phase 2 opener (plan §17.4 successor gate).
2. Pin: primary model (default 32B), SESOI + margin-SESOI rule, surface
   bound (0.10), incidental count (8), G-LEX thresholds (50%/25%, 10/12),
   Δ_lex reference model + carrier, B-GRAD rank count, cross-model ids.
3. Equality ratings when B-ARB3 + RO twins are rendered.
4. Freeze approval → book VM.

**Agent (start immediately, CPU only):**
1. P2-0: branch, skeleton, `pref2-import-phase1-v1`,
   `pref2-phase1-reanalysis-v1` (F1–F5 scripts + outputs),
   `pref2-phase1-hygiene-v1` (H1–H6), README status fixes, wall v2 + tests.
2. P2-1: bank generators + rendered format templates + parser fixtures.
3. P2-2: tokenizer-only port audits (Qwen, Gemma) → per-model codebook
   decisions.
4. P2-3: mechanism repairs + precheck + retrodiction test (must "predict"
   docsection's failure from frozen Phase 1 data).
5. P2-4: preregistration draft with every §8.2 item enumerated.

**Do not** open the sealed Phase 1 7B captures for new science (H2:
unverifiable seal; and no capture exists for the graduated 32B scenario —
new captures come free with the Phase 2 battery).

---

*End of preference_2_1.md — candidate plan; not a freeze. Supersedes the
2026-08-08 draft of this file in place (git history preserves it). A Phase
2 preregistration must pin every §8.2 number and template before frozen
outcomes.*
