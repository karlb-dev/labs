# preference_2_2_addendum.md

## Lab 38 Phase 2 — execution addendum for the Colab research agent

**Reads with:** the governing replacement plan (`plans/preference_2_2.md`, the file titled
"Phase 2 replacement plan: surface policy, semantic defaults, contextual choice value,
enacted choice, and report coupling"), the Phase 1 record on `interp_preference_phase2`
(reviewed at branch head `5038315a`, the plan's stated review target), and the Phase 1
plans/addendum for inherited discipline.

**Precedence:** Phase 1 record (immutable) < handout < Phase 1 plan/addendum < Phase 2
replacement plan < this addendum. Where this addendum is silent, the replacement plan
governs. Every deviation introduced here is a numbered erratum (Section B).

**Grounding:** every Phase 1 number cited below was verified against
`phase1/reports/PREFERENCE_PHASE1_STATE_OF_RECORD.md`, `…_BEHAVIORAL_REPORT.md`, the
evidence registry (16 live events through `pref1-closeout-v1`), the 32B mechanism JSONs,
and `phase2/reports/dev_cpu_reanalysis_20260808/README.md`.

**First actions:**
1. The replacement plan is committed as `preference/plans/preference_2_2.md` and this
   addendum as `preference/plans/preference_2_2_addendum.md` (prior versions of both
   Phase 2 proposals are preserved in Git history, annotated never edited — plan §6.2
   discipline). Confirm `preference/README.md` records the supersession, and record it in
   a registry event at P2-0.
2. Append `pref2-addendum-intake-v1` with this file's sha256 and the statement that
   Section B errata supersede the corresponding replacement-plan text.

---

# A. Scope

The replacement plan is scientifically mature; this addendum does four jobs. (1) It fixes
two structural problems — H-CANON as written is statistically underdetermined, and the
B-PC-MECH calibration requirement contradicts the plan's own no-weights-before-freeze
rule. (2) It pins every constant the plan left open (bank grids, row counts, floors,
decoder definitions, donor matching, guard sets, RO intervention sites) so the
preregistration can be written without invention. (3) It adds the Colab/VM execution
contract: hardware gates, disk plan, capture sizing, session mapping, and gated-model
prep. (4) It restates the human gates and stop-and-ask conditions so the agent has
exactly one pause point plus a short list of halt triggers.

Nothing here relaxes the claim ceiling, the language wall, the safety wall, gate order,
or freeze discipline.

---

# B. Errata and resolved ambiguities

**E1 — H-CANON is underdetermined as written (analysis bug).** Plan §16/§28 fits
`θ0 + θᵀz_j` with a six-dimensional coding on **three** discovery axes — 7 parameters, 3
observations — then "predicts" three heldout axes, where even perfect sign accuracy
(3/3) has permutation p = 0.125. Fix, frozen before any model output:
(a) the confirmatory canonicality object is a **single composite score** per axis — the
frozen, blinded sum of the six binary codes (each coded ±1 toward the predicted-canonical
pole) — not a fitted multi-parameter θ;
(b) discovery axes (3) are used only to fix the composite's sign convention and a single
scale;
(c) the heldout target set is **the 3 heldout B-CANON axes plus all 12 B-ARB3 scenarios'
neutral semantic-default signs** (which the battery measures anyway), coded on the same
sheet before outcomes — 15 sign predictions, permutation-tested (10,000 label
permutations), with magnitude rank-correlation secondary;
(d) per-dimension θ fits are exploratory only, reported as such;
(e) the coding sheet is authored blind to Phase 1 outcomes — the coder sees scenario and
axis descriptions only, never Phase 1 signs, margins, or graduation tables; where a
second coder exists, dual-code with disagreement resolution before any Phase 2 outcome.
H-CANON never gates B-ARB3, mechanism, coupling, or completeness (E17), and its failure
verdicts (`partial_regularities`, `scenario_specific`, `no_neutral_structure`) are
first-class outcomes, not campaign failures.
This keeps the plan's intent (predeclared canonicality, no post-outcome renaming) and
makes the test actually capable of passing or failing. The B-CANON bank stays at six
axes (Section D).

**E2 — B-PC-MECH calibration contradicts the freeze ordering (execution bug).** Plan §18
requires the non-saturated mechanistic PC to be *development-calibrated* into the
0.5–3.0-nat validation band, which requires primary-model forward passes; plan P2-5/P2-6
forbid loading scientific weights before the freeze tag exists. Resolution — a two-stage
freeze:
1. `pref2 freeze create --tag preference-phase2-freeze-v1` pins everything **except** the
   B-PC-MECH difficulty parameter (a single scalar knob in the scenario text, enumerated
   in the bank as `pcmech_difficulty ∈ {d1..d4}` candidate variants, all frozen as text).
2. GPU S3 gains a bounded calibration step: run the four difficulty variants on
   **train+validation incidentals only** of B-PC-MECH (development tier, labeled), select
   the variant whose validation margin lands in [0.5, 3.0] nats, and record
   `pref2-pcmech-calibration-v1`.
3. A freeze **amendment record** (not a new design) pins the selected variant before S4.
   Holdout incidentals of B-PC-MECH are never touched during calibration.
If no variant lands in band after the four candidates, STOP and ask (Section M) — do not
author new variants mid-run.

**E3 — B-SURF is structurally nested, and its arithmetic exceeds the plan's band.**
Display-label assignment, reply-list order, and label family exist only in the F-P1
format; F-SYM rows have only token order and code assignment. The "full factorial" is
therefore per-format: F-P1 contributes 2⁵ = 32 cells (order × label-assignment ×
inline-code × reply-list × label-family), F-SYM contributes 2² = 4, i.e. 36 cells per
template-skin. With the plan's 4 templates × 8 skins that is **1,152 rows**, not 500–800.
Resolution: accept 1,152 (the compute is trivial at single-row scoring); the design-rank
audit runs **per format**, and the analysis never attempts to estimate label/reply-list
coefficients from F-SYM rows. Cross-format contrasts use only the shared factors.

**E4 — Context target/strength are double-encoded.** §1.4 lists `advantage_target` and
`advantage_strength` as separate variables, but target is derivable from the sign of
strength. Canonical encoding: **one signed integer `context_strength ∈ {−2,−1,0,+1,+2}`**;
`advantage_target` is a derived display field. A schema test enforces the derivation, so
no row can carry an inconsistent pair.

**E5 — B-MECH grid pinned.** Per anchor: 32 incidentals (16 train / 8 validation /
8 holdout); per incidental **40 rows** = 5 strengths × 2 display orders × 2 code maps ×
2 menu paraphrases. Context-paraphrase family (4 families) is assigned at the incidental
level, balanced within each split (train 4×4, validation 2×4, holdout 2×4). Codebook
rotation: the three primary pairs rotate at the incidental level, balanced within splits;
the **reserved fourth family** appears only in transfer rows: every holdout incidental is
additionally rendered under the reserved family (8 × 40 = 320 rows/anchor). Totals:
3 × 1,280 base + 3 × 320 transfer = **4,800 rows**, inside the plan's 3,840–5,760 band.
A bank test proves the reserved family never appears in train or validation rows.

**E6 — RO-DISJOINT lacks the upstream sentinel its own coupling assay requires.** Plan
§45 intervenes at RO `context_end`, but the §20 RO template has no context block.
Resolution: every RO-DISJOINT row opens with a constant two-line preamble ending in the
sentinel `Survey context complete.`; the RO site map is
`ro_context_end, ro_option_a_end, ro_option_b_end, ro_menu_end, ro_response_start,
ro_final_prompt_token`. Coupling interventions target `ro_context_end` and `ro_menu_end`;
`ro_final_prompt_token` is the direct-output control site. A bank test asserts the
sentinel renders to an identical token sequence across rows within a model.

**E7 — Branch binding under intervention.** No branch continuation is ever executed on an
intervened row (`binding_skip_reason=intervention`); intervened endpoints are margins and
the strict first-line code only. B-MECH rows use enacted-frame wording with
**environment-only** binding (no microtasks), executed only on clean behavioral rows; RO
rows never execute (Phase 1 invariant). Microtask binding lives only in B-ARB3 (six
scenarios, per plan §14). Tests: `test_intervened_rows_never_execute`,
`test_bmech_binding_environment_only`.

**E8 — Mechanism retrodiction scope.** The plan's hygiene review is correct that the 7B
capture seal is unverifiable (null SHA) and no 32B frozen capture manifest exists, so
P2-4's "reproduce old PC final-token numbers" cannot mean recompute-from-raw-captures.
Pin: retrodiction is **analysis-level** — re-run the old estimator logic on (a) the
frozen mechanism JSONs (`frozen_32b/mechanism/*.json`) and (b) synthetic worlds
reproducing the recorded geometry — and must (i) reproduce the recorded PC numbers within
tolerance from the JSONs, (ii) predict docsection non-identifiability (recorded val r
0.09–0.13 vs random bands 0.40–0.57), and (iii) classify the old PC as `MARGIN_HANDLE`
with `DIRECT_OUTPUT_RISK` (final-prompt-token site; wrong-scenario control at ~62% of
primary; zero strict flips). Record `retrodiction_scope=analysis_level` in the event;
this does not block the phase.

**E9 — Cross-model mechanism needs a capture pass that isn't scheduled.** S6 cross-model
behavioral runs carry no `--capture-sites` (correctly — capture on four models by default
would waste disk). A model that later earns mechanism (S10) therefore triggers a
**dedicated capture pass** on its B-MECH + B-PC-MECH + RO(anchor) rows before
intervention work. Budget line added in Section C4; the run contract gains stage
`capture_pass` with the same manifest/sha rules as S4.

**E10 — NC floor family mapping.** Pin which floor comes from which family: the
**semantic-margin floor** (feeds §25 criterion 2) from families 1–2 (verbatim-identical
and paraphrase-twin), pooled per model; the **carrier/code floor** from family 3
(code-only); the **context-slope floor** from family 4 (advantage-null ladder); the
**decoder/causal floors** (AUC, correlation, causal-margin movement) from families 1–2
rows passed through the identical mechanism pipeline with sham labels. Family 4's ladder
texts go through human gate H2 alongside the real ladders.

**E11 — Identifiability decoder defined.** The precheck "decoder" is the **fixed
projection onto the train-fitted d** — no additional fitted parameters. Signed-strength
correlation is Pearson r between projection and `context_strength` on validation (then
holdout); advantage AUC is computed on |s| ≥ 1 rows only; permutation bands come from
1,000 strength-label permutations within incidental. The ridge sensitivity never
substitutes for the projection decoder in the gate.

**E12 — Patch donor matching pinned.** For a neutral holdout receiver, donors are the
**same incidental's** rows at the same display order, code map, and menu paraphrase,
differing only in context: A-favor and B-favor donors at matched |s| (primary |s| = 2,
sensitivity |s| = 1), the neutral self-donor (s = 0, i.e. the receiver's own state —
must be a no-op within tolerance), and a wrong-scenario donor matched on site/layer/
position-index. Add one cheap secondary with real teeth: **donor-strength monotonicity**
— the |s|=2 patch contrast should exceed the |s|=1 contrast in the intended direction.
All donor rows live in the holdout split and open with it, once.

**E13 — Guard set for dose selection pinned.** Reuse Phase 1's 16 frozen unrelated guard
prompts verbatim, plus 8 new in-domain neutral prompts (menu-shaped, no choice
instruction), frozen at preregistration. §38's thresholds (mean KL < 0.05 nats, frozen
max bound = 0.20 nats, generic next-token tolerance = 0.05 nats median absolute shift)
apply over the first 32 continuation tokens.

**E14 — Format-gate fallback frozen now.** STOP_F's "fallback only if frozen" is
exercised: the frozen fallback is **F-P1**, full stop. If F-SYM fails its B-DEV gate
(strict parse < 0.98 on B-DEV PC rows, or NC asymmetry above floor attributable to the
format), the campaign proceeds under F-P1 with the format swap recorded as a first-class
limitation in every report. No third format may be authored after freeze.

**E15 — Seeds and sign anchors.** All random assignments (analysis sign anchors, split
memberships not already inherited, permutation seeds) derive from the freeze-commit sha
by the Phase 1 rule (first 8 hex chars → int), recorded in the freeze record.

**E16 — "Holm across twelve" is now actually feasible — use the exact test.** With 16
incidentals per B-ARB3 scenario, the incidental-level exact sign-flip test has 2¹⁶
permutations (min two-sided p ≈ 3.1e-5), so Holm across the F1 and F2 families is
arithmetically coherent for the first time (Phase 1's five-incidental design could not do
this; it was the addendum-E13 problem). Pin: the primary p per scenario-endpoint is the
**incidental-level exact sign-flip** on the incidental means; the hierarchical bootstrap
(10,000 replicates, incidentals then cells) supplies the CI; the NC floors remain
conjunctive criteria on top. State this in the preregistration as the designed fix to the
Phase 1 inference limitation.

**E17 — Completeness is the primary-model spine, not the full multi-model map (scope
guard).** The plan's §51 and Part XV read as if all four model cells are owed for
`preference-phase2-complete-v1`; at the pinned Section D row counts that is a
roughly order-of-magnitude larger design surface than Phase 1 closed in one session.
Pin the completeness definition: Phase 2 is **complete** when the OLMo-32B spine ships —
B-DEV, B-SURF, B-PC, B-NC, B-ARB3, RO-DISJOINT behavioral, the small F-P1 continuity
arm, power simulation, preregistration + freeze, and the conditional
mechanism/coupling work on independently earned 32B cells — plus a registered
disposition (run, `STOP_P(model)`, or a logged drop in the frozen order) for every other
scheduled cell. Cross-family behavioral maps (7B / Qwen / Gemma), full B-CANON
confirmatory, F-COMMIT, full DG, and all cross-model mechanism remain scheduled and are
dropped only via the frozen drop order with logged events — but their absence never
blocks the closeout tag. Accordingly the plan §75 never-drop list is amended: the
"cross-model core B-ARB3 map" entry moves into the frozen drop order between items 4 and
5 (drop Gemma first, then Qwen, then 7B, logging each); and under PI-time pressure
B-CANON confirmatory demotes to exploratory (the coding sheet then carries
`agent_dual_code_provisional` and H-CANON reports exploratory-tier only). Everything
else on the §75 never-drop list stays non-droppable.

---

# C. Colab / VM execution contract

## C1. Hardware gate and dtype policy

| Cell | Weights (bf16) | Minimum VRAM | Notes |
|---|---|---|---|
| OLMo 32B (primary) | ≈ 64 GB | **80 GB+** | Blackwell/H100/A100-80 class; Phase 1 used RTX PRO 6000 Blackwell |
| Qwen3.6-27B | ≈ 54 GB | 80 GB | thinking disabled via template API, asserted per port gate |
| Gemma-4-31B-it | ≈ 62 GB | 80 GB | no-system-role shim; post-softcap logits |
| OLMo 7B | ≈ 14 GB | 24 GB (L4) | may run on a smaller session if needed |

Frozen and mechanism runs are bf16-only; quantization is never a primary instrument. If
the attached runtime is below 80 GB when a big-model stage is due, STOP and ask — do not
substitute the 7B as primary (the Phase 1 graduate and the weaker position policy live at
32B; a 7B-primary campaign answers a different question).

## C2. Disk, downloads, and staging

Weights total ≈ 195 GB across the four models. Pin the staging order to the plan's model
order (32B → 7B → Qwen → Gemma) and **evict each model's HF cache after its cells
complete** (`pref2 model stage` should implement acquire/evict; verify free disk ≥ 1.3×
next model before download). Captures, per model that reaches mechanism: ≈ 430 KB/row
(6 sites × 6 depths × ~6k dims × bf16) over ≈ 6,200 captured rows (B-MECH 4,800 +
B-PC-MECH ~640 + RO anchor rows ~770) ≈ **2.5–3 GB**, sharded per plan §35, mirrored to
Drive with per-shard SHA and a committed manifest — this closes the Phase 1 seal-hygiene
hole rather than repeating it.

## C3. Secrets, auth, and gated models

GitHub PAT and HF token via Colab Secrets only; never echoed or committed. **Before any
GPU session:** verify HF access to all four repos resolves (Gemma requires accepted
license terms on the account; Qwen and OLMo are open). A gated-access failure is a
stop-and-ask, not a substitution.

## C4. Session mapping and budget

```text
V0  CPU        P2-0..P2-6: package, hygiene events, portable reanalysis, banks,
               power sim, review packets, tests, prereg candidate     (no weights)
--  PAUSE      Section I freeze package to PI (single human gate)
V1  80GB GPU   S1–S3: bootstrap, 32B port gate + carrier audits, B-DEV/B-SURF,
               E2 PC-MECH calibration, freeze amendment                0.8–1.5 h
V2  80GB GPU   S4: frozen 32B behavior, all banks, captures            2.0–4.5 h
V3  GPU        S5: 7B (L4 ok), then Qwen, Gemma behavior (80GB)        2.5–6.0 h
V4  80GB GPU   S6–S9: prechecks, mechanistic PC, AR mechanism,
               coupling on earned cells                                1.5–5.0 h
V5  optional   S10 cross-model mechanism (incl. E9 capture passes),
               S11 secondary arms                                      1.0–4.0 h
```

Re-project after the first 256 primary-model rows (plan §74) and again before each
cross-family model. All-up 8–18 GPU-hours on Blackwell-class remains realistic given the
Phase 1 throughput anchor (2,320 rows ≈ 20 min at 32B, single-row scoring); on A100-80,
multiply bands by ~1.5–2 and consult the drop order early. Push cadence: every commit
boundary and ≤ 10 minutes during runs; resume refuses config/bank/model drift; replay
certification is same-GPU-class only, with `gpu_env.json` per session.

---

# D. Bank pins and arithmetic of record

Frozen counts (power sim may raise, never lower; any change is a freeze-record entry):

```text
B-SURF      4 templates × 8 skins × (32 F-P1 + 4 F-SYM)            = 1,152
B-ARB3      12 scenarios × 16 incidentals × (2·2·2·2)              = 3,072
B-MECH      3 anchors × [32×40 + 8×40 transfer]                    = 4,800
B-CANON     6 axes × 8 incidentals × (3 contexts × 2 × 2)          =   576
B-PC        6 scenarios × 5 incidentals × 16                       =   480
B-PC-MECH   1 scenario × 32 incidentals × (4 difficulty dev rows
            collapse to 1 frozen variant) × 20                     ≈   640
B-NC        f1 2×6×16 + f2 2×6×16 + f3 1×6×16 + f4 1×8×20          ≈   640
RO-DISJOINT 12 scenarios × 16 × (2·2·2)  + 3 anchors × 8×8 transfer ≈ 1,728
F-P1 cont.  4 scenarios × 6 incidentals × 16                       =   384
                                                       total       ≈ 13,470
```

The bank audit recomputes every line from the factor grid and fails on mismatch. B-ARB3
per-incidental grid: 2 orders × 2 code maps × 2 frames × 2 paraphrases (codebook pair
rotates at the incidental level, balanced within splits). RO grid: 2 orders × 2 code maps
× 2 RO paraphrases, no frames, disjoint alphabet, `code_map_index` aligned to the AR twin
for pairing. F-P1 continuity scenarios: the two strongest 7B asymmetries
(`ar_taskorder_setup`, `ar_execmode_ingest`), the 32B graduate (`ar_docsection_readme`),
and one NC scenario — chosen for maximum reconstruction value against the frozen Phase 1
record.

Authoring guardrails carry over from Phase 1 verbatim (lexical balance audit, token-count
delta ≤ 15%, valence wordlist flags, no code/name substring collisions) and extend to the
context-ladder texts: within an anchor, the ±k statements must be length-matched within
15% across orientations so strength is not confounded with verbosity.

Context-ladder instruction guards (pinned; enforced by audit before H2 and re-checked at
bank audit): ladder texts create advantage through **scenario constraints** (resources,
dependencies, audience, deadlines), never through choice imperatives — a frozen wordlist
audit rejects "choose", "prefer", "pick", "select", "you should", and any
option-name-plus-imperative collocation in ladder lines; both options remain feasible at
every strength (dual-feasibility, H2); ±2 must strengthen ±1 without introducing new
semantic content (monotonicity, H2); and ladder texts are authored blind to any model
margin. The family-4 null ladders are campaign-critical, not a nuisance: a family-4
slope above floor is a stop-and-ask (Section L item 4) and blocks all
`CONTEXTUAL_VALUE` language until ladders are redesigned under a new preregistration
entry.

The `PK4`-style
blend is a known failure mode at content-vs-position conflict cells: the strict parser
already refuses blends; add the blend string family to the adversarial parser tests for
every codebook pair (all interleavings of the two codes).

---

# E. Statistics pins

Primary machinery per endpoint (frozen in the preregistration):

```text
endpoint                          point estimate            p (primary)                CI
semantic margin (per scenario)    incidental-mean of        exact incidental           hierarchical
                                  folded full-target        sign-flip (2^16)           bootstrap 10k
strict semantic choice            incidental-mean rate      exact sign-flip            same
context slope (per anchor)        incidental-level OLS      exact sign-flip on         same
                                  slope, then mean          incidental slopes (2^32→
                                                            10k Monte Carlo flips)
surface coefficients (B-SURF)     balanced contrasts        randomization within        same
                                                            template-skin
```

Floors and thresholds: semantic-margin floor `max(0.15 nats, 2 × NC(f1–f2) p95)`; strict
SESOI 0.10; context-slope floor `2 × NC(f4) p95 slope`; decoder gates per plan §36 with
E11 definitions; Holm within F1 (12), F2 (12, conditional on F1), F3 (3), and the seven
mechanism primaries per scenario. Sign-stability, LOIO, worst-case-invalid, and
codebook/paraphrase strata criteria stay verbatim from plan §25–§27. The B-SURF-to-
Phase-1 reconstruction is scored as out-of-sample accuracy on the frozen per-cell
first-choice rates of both Phase 1 models (same revisions — this is why plan §49's
revision pins matter) with the censoring identity (`position + |content| = 0.500`)
recomputed as an audit, not rediscovered as a claim.

Power simulation additions: stress runs at 2× the Phase 1 incidental-level variance
(Phase 1 estimated it from only five incidentals — treat it as noisy), and a dedicated
power cell for the coupling non-margin endpoint (E-G below), since that is the likeliest
underpowered primary.

---

# F. Mechanism pins

Sites/depths per plan §34 with the block-to-stream convention recorded per model.
Direction: paired matched difference at |s| ∈ {1,2} pooled with |s|-matched pairing;
ridge on signed strength is sensitivity only. Layer/site selection by the plan §37 score
with normalization transforms frozen at prereg; tie-breaks unchanged (upstream, then
shallow). Dose per plan §38 in train-projection SD units with the E13 guard set.
Propagation metric pinned: after an upstream intervention at `context_end`, the
projection onto d recaptured at `menu_end`, `response_start`, and `final_prompt_token`
must move in the intended direction with standardized effect ≥ 0.5 × the injected-site
effect at `menu_end`, and the no-op self-patch must sit within tolerance at every
downstream site. The output-adjacency audit (§41) runs at the selected site before any
mechanism claim language; `d_⊥code` is computed against the code-gradient direction
averaged over train rows only.

Saturation exception (PC-MECH only): if calibration lands in-band, strict flips are
expected and required; the frozen exception applies only if the power simulation shows
< 0.50 probability of a single flip at the maximum guard-safe dose given the calibrated
margin — in that case the PC passes on margin+propagation+specificity and the limitation
is carried into every downstream mechanism claim.

---

# G. Coupling pins

Natural readout, upstream intervention, and RO-to-AR symmetry per plan §44–§46 with the
E6 sentinel sites. Per-anchor coupling power: 8 holdout incidentals × 8 RO cells = 64
receivers, plus 64 reserved-codebook transfer receivers. The `CHOICE_REPORT_COUPLED`
requirement "at least one non-margin endpoint moves" is pinned with candidate defaults
(PI ratifies at freeze): strict comparative-report rate shift ≥ 0.15 between +d and −d
conditions on holdout receivers, **or** a five-point scale mean shift ≥ 0.5 categories
with monotone dose response. The Phase 1 RO pathology is a known risk: at 7B the RO
channel emitted one constant code in 9/12 scenarios (mechanically forcing 0.500), so the
RO analysis must report the constant-code rate per model×scenario **before** any coupling
statistic, and a constant-code RO channel routes the scenario to `BEHAVIOR_SPECIFIC`
adjudication with the RO endpoint marked unmeasurable rather than null.

Power fallback, pinned now: if the dedicated coupling power cell (Section E) shows
< 0.80 power for the strict comparative-report shift at the pinned receiver counts, the
preregistration designates the **RO full-target margin contrast** as the coupling
primary and the strict-report and five-point endpoints as secondary descriptors (the
Phase 1 addendum-E14 lesson applied to coupling). `CHOICE_REPORT_COUPLED` then requires
the margin primary to pass plus at least one non-margin endpoint moving in the same
direction, reported at secondary tier rather than Holm-primary.

---

# H. Hygiene execution details (plan §6.2, made concrete)

One registry event per item, each naming exact paths: (1) stale bank JSONL SHA in the
Phase 1 freeze record — annotate, never edit; (2) 7B capture seal: record
`capture_seal_status=unverifiable_null_sha`, retain the Drive object, and treat Phase 2
captures as the first verifiable seal (C2); (3) 32B captures: `absent_by_design` (the
case study captured at run time only) — Phase 2 recaptures; (4) README status refresh;
(5) mechanism-control disclosure event quoting the recorded numbers (wrong-scenario ≈ 62%
of primary; RO source ungated; zero strict flips; final-token site); (6) language-wall
raise-behavior fix with a failing-exit test; (7) portable-path refactor with the
three-directory rerun proof; (8) supersession note for the old Phase 2 proposals; (9)
package-absence-at-intake note. None of these blocks P2-2 onward; all precede the freeze.

---

# I. Human gates — one pause, five packets

The single PI pause (H5) receives, together: the preregistration candidate with every
Section B/D/E/F/G constant inlined; the H1 equality packet (B-ARB3 pairs, both
paraphrases); the H2 ladder packet (B-MECH ladders **and** the NC family-4 null ladders,
monotonicity + dual-feasibility per rung); the H3 canonicality sheet (E1 composite,
covering the 6 B-CANON axes and the 12 B-ARB3 scenarios, blind to all outcomes); the H4
RO-equivalence packet (disjoint wording, same semantic identities). Agent dual-code
provisional passes are permitted for development work only, both passes pre-outcome,
separated by at least a work-phase boundary (the Phase 1 D3 precedent), and every
downstream artifact carries the `agent_dual_code_provisional` license tag until PI
ratings replace it. The E2 freeze amendment (PC-MECH difficulty) is the only post-pause
freeze write, and it touches one field.

Packet priority when PI time is scarce: H1 (equality), H2 (ladders and the family-4
null ladders), and H4 (RO equivalence) gate the science directly and hard-block the
freeze; H3 (canonicality) is reviewed last within the same pause, and if it cannot
complete, B-CANON confirmatory demotes per E17 rather than delaying the freeze.

---

# J. Additional tests (append to plan §54)

```text
test_context_strength_single_encoding                    (E4)
test_bmech_reserved_codebook_absent_from_train_val       (E5)
test_bmech_paraphrase_family_balanced_within_splits      (E5)
test_ro_sentinel_constant_token_sequence                 (E6)
test_ro_site_map_complete                                (E6)
test_intervened_rows_never_execute                       (E7)
test_bmech_binding_environment_only                      (E7)
test_bsurf_design_rank_per_format                        (E3)
test_bsurf_row_arithmetic_1152                           (E3)
test_canonicality_composite_frozen_before_outcomes       (E1)
test_canonicality_heldout_targets_include_arb3           (E1)
test_donor_matching_resolver_same_incidental_surface     (E12)
test_code_blend_family_rejected_all_codebooks            (D)
test_pcmech_difficulty_variants_frozen_as_text           (E2)
test_capture_shard_sha_manifest_roundtrip                (C2)
test_ro_constant_code_rate_reported_before_coupling      (G)
```

Language-wall: the recursive scan with raising exit code covers this addendum too;
quoted ceiling lists remain recognized context.

---

# K. Artifact delta introduced by this addendum

```text
plans/        preference_2_2.md (governing), preference_2_2_addendum.md (this file)
data/         pcmech difficulty variants (frozen text), NC family-4 ladder texts,
              context-ladder lexical balance audit
diagnostics/  guard_prompt_set.json, donor_matching_audit.csv,
              ro_constant_code_rate.csv, capture eviction/staging log
preregistration/  freeze amendment record (E2, single field)
registry      pref2-addendum-intake-v1, pref2-pcmech-calibration-v1,
              hygiene events per Section H
```

---

# L. Stop-and-ask conditions

Halt, write a short status note, and wait for the PI when:

1. Runtime VRAM < 80 GB when a 32B/Qwen/Gemma stage is due, or gated-model access fails.
2. No PC-MECH difficulty variant lands in the 0.5–3.0-nat validation band (E2).
3. STOP_I fires on the primary model (any instrument/port gate), or F-SYM **and** F-P1
   both fail the B-DEV format gate.
4. Any NC family clears a semantic or slope floor, any wrong-branch execution, or a
   replay/resume parity failure in the certifying session.
5. The freeze gate itself (Section I package) and its single amendment.
6. Any step that would require refusal ablation, DG-SAFE generation, a pooled
   cross-scenario direction, holdout access before its opening event, or enlarging a
   frozen bank after outcomes — these are never permitted; if a step appears to require
   one, halt.

Everything else — including full behavioral nulls, `SURFACE_POLICY_ONLY` on every
scenario, port failures on non-primary models (`STOP_P` continues other cells), and
budget-driven drops in the frozen order — is handled autonomously with the deviation
logged.

---

# M. Closing restatement

Phase 1's most valuable products were an instrument that cannot be fooled by its own
pipeline (NC at exactly 0.000 on both models) and an honest anatomy of a null. Phase 2's
job is to give the four real content asymmetries, the 32B graduate, and the broad folded
margins a design under which they can either become `SEMANTIC_MARGIN` / `ENACTED_CHOICE`
/ `CONTEXTUAL_VALUE` / coupling results or die cleanly — with the surface policy measured
rather than outlawed, the mechanism target created by randomized context rather than
scavenged from residuals, and the report channel tested on a surface it cannot pattern-
match. Every constant an agent needs is now pinned or explicitly routed to the single
human gate; the preferred outcome remains an instrument whose vocabulary does not bend
toward whichever result appears.
