# jspace_lab_gemma_2.md

## Gemma transport side study 2: backend-parity repair, Stage-1 relicensing, and the conditional reopening of mechanism localization

**Status:** proposal for the second Gemma block. Nothing in this document is a
result; every number quoted below is reproduced inline from artifacts already
registered by study 1 so that this plan can be **reviewed without access to
the run data**. Where a number appears, its evidence id appears with it.

**Governs with:** `jspace_lab_gemma_1.md` (design vocabulary, estimators,
G-stage ladder, §0.2 forbidden-meanings list — all carried forward unchanged)
and `jspace_lab_gemma_1_addendum.md` (governance, two-block cap discipline,
threshold pre-commitment rule). On conflict about *what study 1 found*, the
registered record wins: `release/GEMMA_TRANSPORT_STATE_OF_RECORD.md`,
`release/gemma_transport_claim_ledger.md`, and the append-only registry
`interpretability/jspace_gemma/reports/evidence_events.jsonl` (18 live events
through the blocker; frozen prefix 25,170 bytes).

**Proposed identity:** new branch `interp_jspace_gemma_transport_2`, evidence
prefix `gm2-`, package `interpretability/jspace_gemma/` extended in place
(new experiments module; no edits to registered study-1 outputs), Drive root
`gemma_transport_2_<date>/`. Native tiers: development and methods only.
Study-1 artifacts are imported read-only by exact hash, exactly as study 1
imported its nine historical artifacts.

> **Paste-line for the study-2 agent**
> Read `jspace_lab_gemma_1.md`, its addendum, the state of record, and this
> file, in that order. G2.1 (backend-disagreement characterization and
> ceiling recalibration) is the unconditional gate; every downstream stage is
> conditioned on its branch. The one methodological sin this track cannot
> survive is choosing the new ceiling after learning which value would
> relicense Stage 1 — the calibration set is therefore quarantined from every
> Stage-1 metric until the ceiling is frozen and registered.

---

# 0. Executive summary

Study 1 (`jspace_lab_gemma_1.md`, executed 2026-08-02, terminal state
`COMPLETE_METHODS_BLOCKER`) accomplished, in order: instrument goldens, a
56-cell OLMo calibration, a 14-criterion positive-control threshold freeze, a
complete 40-cell / 1,120-row Gemma Stage-1 grid whose frozen classifier
returned **`local_tangent_mismatch` at all five tested layers**, and then an
adversarial dual-backend replay that **failed its own precommitted
consistency ceiling** — full-batch backend tangent relative error 0.002458
against a frozen 1e-5 ceiling \[`gm-jvp-gemma-backend-parity-v1`\]. Under the
plan's hard-stop rule, Stage 1 is registered as an operational diagnostic
only, and stages G2–G8 were cancelled unrun.

Study 2 exists to answer one question first, and everything else
conditionally:

> **GQ2.0** — Is the cross-backend tangent disagreement a benign, boundable
> property of bf16 kernel scheduling on this architecture (in which case the
> 1e-5 ceiling was miscalibrated and Stage 1 can be relicensed under a
> measured ceiling), or does it indicate a real ambiguity in what "the exact
> JVP" computes on Gemma (in which case the instrument itself must be
> repaired before any Gemma sentence strengthens)?

The prior from the study-1 data (laid out in §2.5 and §3) is strongly toward
the first branch: the two backends agree to cosine 0.99999958, the maximum
absolute difference is exactly one bf16 quantum (0.0390625), the selected
mismatch row replays **bit-identically** on both backends, and the mismatch
being protected is three orders of magnitude larger than the disagreement.
But that prior is exactly the kind of after-the-fact reasonableness the
addendum's §2.1 forbids as a substitute for measurement — so study 2 measures
it, freezes the answer, and only then touches Stage 1's license.

The five original closing questions GQ1–GQ5 remain the scientific targets;
study 2 inserts GQ2.0 ahead of them and inherits the G2–G8 ladder unchanged
behind its gate.

---

# 1. Where study 1 ended: the registered record

For reviewers without the run mirror, this is the complete live-evidence
state at the study-1 boundary (source: state of record + development report,
both hash-pinned in `gm-state-of-record-v1`):

| Evidence id | What it is | Decisive content |
|---|---|---|
| `gm-foundation-v1` | isolated package/runtime verification | 22 tests pass; nine historical imports hash-verified |
| `gm-jvp-goldens-v1` | instrument goldens | both exact-JVP backends match analytic derivatives; forward/fallback differ 4.81e-17, forward/reverse 8.33e-17 on the tiny nonlinear suffix; secant error falls 1.09e-6 → 1.70e-8 over three ε halvings |
| `gm-band-convention-v1` | paper-band resolution | the original paper's workspace band is 38–92% depth ⇒ Gemma ≈ L23–L55; the 37–62% convention in earlier Gemma materials is **not** the governing one; G6's L44/L48/L52 candidates are in-band |
| `gm-olmo-calibration-finalize-diagnostic-v1` | incident record | post-compute JSON-serialization failure (numpy.int64) after all 56 cells; recovery via pure finalizer, zero recompute |
| `gm-jvp-olmo-calibration-v1` | 56 cells / 1,568 rows | positive-control trend, §2.2 |
| `gm-jvp-olmo-positive-control-v1` | 14-criterion threshold freeze | all pass; thresholds immutable, §2.2 |
| `gm-jvp-gemma-stage1-v1` | 40 cells / 1,120 rows | `local_tangent_mismatch` at L22/L30/L37/L44/L52, §2.3 |
| `gm-jvp-gemma-backend-parity-v1` | the terminal failed gate | §2.5 |
| `gm-state-of-record-v1` | terminal release | `COMPLETE_METHODS_BLOCKER`; claim ledger GM-C01–C11; import bundle for Phase 4/5 |

Merge state: the full ancestry merged into Part 2 at `5346b16`, post-merge
handoff `c9021e5`. The two-block GPU cap from the addendum was honored; the
hard stop landed inside the cap.

The claim ledger's licensed/blocked split (`gemma_transport_claim_ledger.md`)
is binding on this document. In particular GM-C06–C10 (nondifferentiability,
mechanism attribution, workspace absence, late-band verdict, any Phase 4
upgrade) remain **blocked**, and the maximum licensed sentence is the
methods-only sentence quoted there.

---

# 2. The data record

This section is deliberately heavy: it is the evidence base every study-2
design choice cites. All tables are recomputed directly from the registered
row tables (`gemma_stage1_rows.parquet`, 1,120 rows, SHA-256 `3b74f1e9…`;
`olmo_calibration_rows.parquet`, 1,568 rows;
`gemma_stage1_curvature_fits.parquet`; the backend-parity JSON, SHA-256
`22c32776…`).

## 2.1 Instrument goldens

Both PyTorch exact-JVP implementations match analytic tiny-model derivatives
at zero recorded error; on the nonlinear tiny suffix the two exact backends
differ at the 1e-17 level, i.e. they are the *same computation* in fp32 at
toy scale. Central-secant error halves quadratically (1.09e-6 → 1.70e-8 over
three ε halvings), confirming the secant is a secant. \[`gm-jvp-goldens-v1`\]
This matters for §3: the backend disagreement observed later is therefore a
*model-scale, bf16-runtime* phenomenon, not an implementation bug visible at
golden scale.

## 2.2 OLMo control: calibration trend and the frozen thresholds

Calibration (56 cells, 1,568 rows; 645 pass the fixed bf16 delivery gate;
28/28 bit-exact clean-suffix checks; zero exact-JVP primal parity error)
\[`gm-jvp-olmo-calibration-v1`\]:

| OLMo layer | median tangent cosine (single-position) | median relative error |
|---|---|---|
| L4 (shallow negative control) | 0.693 | 0.723 |
| L56 | 0.991 | 0.137 |
| L60 (identity anchor) | 0.996 | 0.089 |

Same-estimator per-layer table at the declared dose (ε = 0.10,
single-position, delivery-clean, SNR ≥ 20) — the exact estimator later
applied to Gemma:

| OLMo layer | n | med cosine | med rel. error |
|---|---|---|---|
| L24 | 5 | 0.932 | 0.362 |
| L32 | 14 | 0.962 | 0.273 |
| L40 | 16 | 0.974 | 0.225 |
| L47 | 15 | 0.982 | 0.188 |
| L56 | 16 | 0.992 | 0.124 |
| L60 | 16 | 0.997 | 0.084 |

Frozen thresholds calibrated from this control **before any Gemma number**
\[`gm-jvp-olmo-positive-control-v1`, threshold file SHA-256 `3cb1e68c…`\]:
delivery cosine ≥ 0.999 and relative-norm error ≤ 0.01 (below: unmeasurable);
measurement SNR 12, decision SNR 20; primary row gate at declared ε = 0.10:
tangent cosine ≥ 0.98, forward relative error ≤ 0.20, central relative error
≤ 0.10, ≥ 90% passage; curvature-fit intercept ceiling 0.30 (control q95
0.2814) and slope floor 0.15 (max control slope 0.1062); shallow-minus-late
error contrast ≥ 0.40 (observed 0.6303). All 14 criteria pass on OLMo; all
32 late anchors pass the primary gate. **Recalibration after observing Gemma
is forbidden by the frozen config.**

## 2.3 Gemma Stage 1: the operational mismatch

Grid: 4 prompts × 5 layers (L22/L30/L37/L44/L52) × 2 perturbation modes × 4
non-lens direction families × 7 relative ε (0.0025–0.20) = 40 cells / 1,120
rows, from clean commit `036e552`. Wrong-hook sentinel fires (0.3355 vs 0.10
floor). Delivery funnel: 538/1,120 pass bf16 delivery, 508 clear SNR 12, 477
clear SNR 20. \[`gm-jvp-gemma-stage1-v1`\]

Primary table at the declared dose (ε = 0.10, single-position,
delivery-clean, SNR ≥ 20; every layer has 12/16 evaluable rows and the
frozen 75% coverage floor is met):

| Gemma layer | n | med cosine | med rel. error | med central rel. error | med gain | passes |
|---|---|---|---|---|---|---|
| L22 | 12 | 0.039 | 1.357 | 1.763 | 1.00 | 0 |
| L30 | 12 | 0.281 | 1.669 | 2.098 | 1.71 | 0 |
| L37 | 12 | −0.062 | 3.758 | 6.414 | 3.56 | 0 |
| L44 | 12 | 0.104 | 5.365 | 10.549 | 5.48 | 0 |
| L52 | 12 | 0.138 | 4.441 | 7.640 | 4.33 | 0 |

Read that table twice. The tangent **cosines are near zero** — the exact
prompt-specific first derivative is not merely imprecise about the finite
response, it is close to *orthogonal* to it at the assay dose. The gain
column (response norm ÷ tangent-prediction norm) grows to 3.5–5.5× by
L37–L52: finite responses are several times larger than first-order theory
predicts, and growing with depth. Smallest-evaluable pass counts across the
whole ε ladder: 0/12, 0/13, 0/12, 0/13, **1/14** (L22→L52). Prompt-bootstrap
95% intervals exclude the 0.20 gate at every layer. Uniform-valid
perturbations do not rescue (1/13 evaluable passes at L22; zero later).

Direction-family breakdown at the declared dose:

| direction family | n | med rel. error | med cosine |
|---|---|---|---|
| gaussian | 20 | 3.126 | 0.112 |
| rademacher | 20 | 3.706 | 0.083 |
| sphere-tangent | 20 | 4.010 | 0.150 |
| activation-radial | **0 evaluable** | — | — |

Two design notes for study 2 fall straight out: the mismatch is
direction-family-generic (no family is spared), and the **activation-radial
family produced zero evaluable rows** — every radial row failed
delivery/SNR gating. Study 2 must either redesign radial delivery or
formally drop the family *before* any target run, not after.

Curvature diagnostics (delivery-clean, SNR ≥ 12, all ε): median
homogeneity defect 0.52–0.66, median odd-symmetry defect 1.08–1.66, median
additivity defect 0.81–1.09 across layers — all far above linear
expectations, with odd-symmetry (the cleanest pure-curvature estimator,
plan 1 §2.4) the largest.

The preregistered a + b·ε robust fits (120 SNR-qualified fits):

| Gemma layer | curvature-dominant | mixed bias+curvature | quantization-floor | med intercept a | med slope b |
|---|---|---|---|---|---|
| L22 | 2 | 20 | 2 | 1.009 | 13.6 |
| L30 | 5 | 19 | 0 | 0.592 | 9.6 |
| L37 | 3 | 21 | 0 | 1.162 | 49.6 |
| L44 | 11 | 13 | 0 | 0.323 | 60.4 |
| L52 | 14 | 10 | 0 | 0.140 | 59.2 |

This is the single most plan-relevant table in the record. Depth cleans the
intercept (a: 1.0 → 0.14) while the slope explodes (b: ~10 → ~60): late
layers fail **more purely through the ε-scaling curvature term**, exactly
the GQ2 "within-context finite curvature" signature, while early layers mix
in a scale-independent bias/floor component. The frozen classifier's verdict
`local_tangent_mismatch` at every layer is the operational summary; the fits
say the *composition* of the mismatch rotates with depth.

## 2.4 The two-model contrast, one estimator

Placing §2.2 and §2.3 side by side at the declared dose: OLMo's worst layer
(L24, med err 0.362) is better than Gemma's best layer (L22, 1.357) by 3.7×,
and the models trend in opposite directions with depth (OLMo error falls to
0.084; Gemma's rises to 4.4–5.4). The figure
`figures/gmT1_tangent_ladder.png` in the paper workspace renders exactly
this contrast from the registered row tables
(`jspace_paper/scripts/gemma_transport_plots.py`, snapshot
`gemma_transport_snapshot.csv` committed beside it).

## 2.5 The terminal gate: full disclosure of the parity replay

The diagnostic froze one registered Stage-1 row before running:
`gm-p001-L52-single_position`, rademacher direction 0, ε = 0.05 — a strongly
mismatched row (stored tangent cosine −0.00044545, relative error 2.7718). It
reconstructed the original eight-request batch (absolute request index 8,
selected slot 5), rehashed the pinned snapshot before load, and compared
`torch.func.jvp` (forward-mode; the backend used by all 1,120 Stage-1 rows)
against `torch.autograd.functional.jvp` (reverse-over-reverse).
\[`gm-jvp-gemma-backend-parity-v1`, artifact SHA-256 `22c32776…`\]

Complete comparison dump (relative error / max absolute error):

| comparison | cosine | rel. error | max abs |
|---|---|---|---|
| clean suffix parity | — | 0.0 | 0.0 |
| primary primal vs identical-batch clean | 0.9999974 | 0.0 | 0.0 |
| fallback primal vs identical-batch clean | 0.9999974 | 0.0 | 0.0 |
| source activation vs stored | 1.0000008 | 0.0 | 0.0 |
| clean target vs stored | 0.9999996 | 0.0 | 0.0 |
| finite response vs stored | 1.0000002 | 0.0 | 0.0 |
| primary tangent vs stored | 1.0000011 | 0.0 | 0.0 |
| **backends, selected slot** | 1.0000011 | **0.0** | **0.0** |
| **backends, all eight slots** | **0.99999958** | **0.002458** | **0.0390625** |

Stored-metric replay: tangent cosine and relative error reproduce with
absolute error 0.0. The recomputed mismatch is unchanged (cosine
−0.00044545, rel. error 2.7718). No backend raised; no tensor was
non-finite; no finite difference was substituted. The **sole failed
criterion** is the all-slot backend tangent relative error, 0.002458 against
the precommitted 1e-5 ceiling — a 246× exceedance whose absolute magnitude
is exactly one bf16 quantum. Under the hard-stop rule the track terminated
without weakening the gate or selecting the agreeing slot.

## 2.6 The band-convention resolution (already banked for G6)

`gm-band-convention-v1` resolved the band ambiguity from the primary
Methods: the paper's workspace band is 38–92% relative depth ⇒ **Gemma
≈ L23–L55**, so G6's late candidates (L44/L48/L52) are *inside* the
paper-relative band. The addendum's §2.2 stake stands: a passing late-band
transport gate would convert this track from boundary-drawing into the
doorway for a Phase 5 Gemma capacity/causal cell. Study 2 must carry one
sobering new fact against that hope: at the tested doses, L44 and L52 are
Stage 1's *worst* layers (med rel. error 5.4 / 4.4), and their curvature
fits are the most slope-dominated (b ≈ 60). If a late transport regime
exists, it exists at doses below the ones Stage 1 declared — the G6 design
must therefore extend the ε ladder downward with fp32 injection rather than
assume the late band self-rescues.

## 2.7 Registered absences

For review clarity, these were **not** produced, by design or by the stop:
the identity-fraction profile figure (addendum §2.3 — cheap, still worth
doing, still unrun); any J-selected-direction rows (their lens/token hashes
were never bound); all of G2–G8; any outcome-dependent figure suite. The
development TeX exists in the package; no PDF is claimed as a release
artifact (the VM had no TeX engine).

---

# 3. Reading the blocker: three hypotheses, already partially bounded

The all-slot disagreement admits three readings; study 1's own data already
constrains them, which is why G2.1's design below looks the way it does.

**(a) bf16 kernel scheduling.** Forward-mode and reverse-over-reverse
traverse different kernel schedules; with bf16 activations, different
reduction orders legitimately differ near one quantum per element. Evidence
for: max-abs is exactly one bf16 quantum; cosine has eight nines; the
selected slot is bit-identical (slot-dependent batching means slot-dependent
kernel paths); primals — computed by the *same* forward machinery — agree
exactly. This is the benign branch: the correct repair is a *measured*
ceiling, not a better backend.

**(b) batch-composition sensitivity.** The disagreement appeared on the
full eight-slot batch, not the selected slot. If tangent error depends on
*which other requests share the batch*, Stage-1 rows computed in batches
carry a small nuisance term that batch-size-1 recomputation would remove.
Evidence for: selected-slot exactness. Evidence against: nothing yet — this
is exactly measurable, cheaply.

**(c) genuine path ambiguity.** Some Gemma op (the obvious suspects: the
V=K global-attention sharing, QKNorm composition, or the tanh-GELU) has
autograd semantics that differ across backends beyond scheduling noise. This
would be the serious branch: "the exact JVP" would not name a unique object
on this architecture, and Stage 1's reference would be ambiguous in kind,
not just in the eighth decimal. Evidence against, so far: goldens agree at
1e-17 on the tiny suffix; primal parity is exact at model scale; the
disagreement magnitude sits at the dtype floor. Not yet excluded because
study 1 measured exactly one batch.

Scale anchor for all three readings: the smallest scientifically relevant
Stage-1 quantity (the 0.20 primary error gate) sits **80×** above the
observed backend disagreement, and the observed mismatches sit 550–2,200×
above it. For hypothesis (c) to rescue Gemma's tangents, the backend
ambiguity would need to be three orders of magnitude larger than observed.
That is the quantitative sense in which Stage 1's verdict is expected to
survive — and the reason expecting it is not the same as licensing it.

---

# 4. The study-2 program

## G2.0 — Foundation (methods; no model)

Standard isolated-track foundation: verify package/runtime, import the
study-1 registry prefix and the specific artifacts cited in §2 read-only by
hash, re-verify the pinned Gemma and OLMo snapshots, and register
`gm2-foundation-v1`. Nothing here reads a Stage-1 metric.

## G2.1 — Backend-disagreement characterization and ceiling recalibration (the unconditional gate)

**Design.** Measure the empirical distribution of
`torch.func.jvp` vs `torch.autograd.functional.jvp` tangent disagreement,
with the target-side quarantine of addendum §2.1 applied to a new object:
the runs may compute backend deltas but must not read or join any Stage-1
tangent-vs-response metric until the ceiling is frozen.

- **Cells.** Both models (OLMo is the parity control exactly as it was the
  transport control): {2 models} × {3 layers: shallow / mid / late} ×
  {4 prompts} × {batch sizes 1, 4, 8} × {3 direction draws} ≈ 200–400
  JVP pairs. Per-pair cost is two forward-mode-scale evaluations; the whole
  stage is hours, not a block.
- **Measurands per pair.** All-slot and per-slot tangent relative error,
  cosine, max-abs in quanta of the model dtype; primal parity; slot-position
  dependence (hypothesis b); and a one-shot sublayer split on a small subset
  (JVP through attention-only vs MLP-only suffixes) to localize which op
  family carries the disagreement (hypothesis c screen).
- **Optional Backend C.** Plan 1 §3.2's local unfused replacement runs on
  the single frozen replay row only, as a tiebreaker, not as a new exact
  backend for the grid.
- **The frozen deliverable.** `gm2-backend-parity-calibration-v1`: the
  disagreement distribution, its batch/slot/op structure, and a **calibrated
  ceiling** committed in a threshold file before §G2.2 runs. Calibration
  rule (precommitted here): ceiling = max(q99 of the pooled two-model
  disagreement × 3, 10 dtype quanta in relative terms), with the explicit
  invariant that the ceiling derivation cites only this stage's
  distribution — never the 0.002458 number it must judge, and never any
  Stage-1 outcome. If the OLMo and Gemma distributions differ materially,
  that fact is itself a registered finding and the ceiling is set per-model.

**Branches.**

| Observation | Classification | Route |
|---|---|---|
| Pooled disagreement concentrated at ≤ few quanta, no slot/op structure beyond scheduling | benign scheduling floor (hyp. a) | freeze ceiling; proceed to G2.2 |
| Disagreement depends on batch composition materially | batch nuisance (hyp. b) | freeze ceiling for batch-1; G2.2 requires batch-1 recompute of affected rows |
| Disagreement localizes to an op family and scales beyond quanta | path ambiguity (hyp. c) | **hard stop**; register blocker; instrument redesign is a new plan, not a patch |
| OLMo and Gemma distributions differ beyond dtype accounting | architecture-dependent floor | per-model ceilings; note for the transport-gate protocol export |

## G2.2 — Stage-1 relicensing (decision precommitted before G2.1 reads anything)

The relicensing rule, stated now so it cannot be shaped later:

1. If the calibrated ceiling from G2.1 ≥ 0.002458 **and** the frozen replay
   row still reproduces bit-identically at the selected slot under both
   backends, then `gm-jvp-gemma-stage1-v1` is relicensed *as registered* —
   no recompute, no row selection — and the study-1 classification
   (`local_tangent_mismatch` at all five layers) is promoted from
   "operational diagnostic" to a **closed methods result** under evidence id
   `gm2-stage1-relicense-v1`. The licensed sentence upgrades accordingly
   (conclusion-skeleton sentence 5 takes its JVP-closed form).
2. If the ceiling lands below 0.002458 but the G2.1 branch was benign at
   batch 1, the affected quantity is recomputed: rerun the 40-cell grid's
   *declared-dose rows only* (5 layers × 4 prompts × 4 directions ≈ 80 rows)
   at batch size 1 under both backends, and apply the frozen study-1
   classifier unchanged. Registered as `gm2-stage1-batch1-v1`.
3. If G2.1 returned the path-ambiguity branch, no relicensing occurs;
   sentence 5 keeps its blocker form and study 2 terminates after
   registering the instrument findings (that outcome is a real result about
   autodiff on this architecture, and the protocol export in G2.5 carries
   it).

## G2.3 — Conditional mechanism ladder (G2–G5 of plan 1, unchanged estimators, data-informed priorities)

Runs only after a successful G2.2 branch 1 or 2, under the addendum §2.4
priority order, with these study-1-data annotations:

- **G2 layer/sublayer localization.** The curvature-fit table (§2.3) makes
  the prediction concrete: the slope term b triples between L30 and L37 and
  stays ~60 through L52. The 5:1 local/global attention pattern places
  global blocks at fixed depths; G2's first question is whether the b-jump
  aligns with global-attention block boundaries (plan 1's routing suspect)
  or with an MLP/norm depth trend. The Stage-1 raw response/JVP vectors are
  retained per cell, so part of G2 is *analysis of existing tensors* before
  any new forward pass.
- **G3 routing autopsy, R0/R1/R3 only.** R1 freezes clean attention
  *probabilities* (the V=K-aware correction endorsed by the addendum).
- **G5 heterogeneity.** The a+b·ε fits answer the within-context half of
  GQ2; G5's prompt-specific-tangent vs fitted-mean-J comparison answers the
  between-context half. Priority per the addendum: G5 before G4 because it
  feeds the mainline's interpretation stack.
- **G4 norm/MLP factorial** on residual time only.

## G2.4 — G6 late-band gate, redesigned downward in ε

Carrying §2.6: the band convention is resolved (L44/L48/L52 in-band), but
Stage 1 has already shown the late layers failing *hardest at the tested
doses* with the most slope-dominated fits. G6 therefore: extends the ε
ladder down to 0.0005–0.0025 with fp32 injection (delivery-gate permitting;
Stage 1's funnel shows bf16 delivery collapses below ε ≈ 0.01 — 538/1,120
survivors — so fp32 injection is load-bearing, per plan 1 §3.4's contract),
adds the identity-fraction profile from addendum §2.3 (α = trace(J)/d per
layer, both models — still the cheapest unrun figure in the program), and
keeps the addendum's stake: a passing late gate at *any* dose relevant to
the assay licenses the scoped Phase 5 doorway; a failing one closes it with
data rather than default.

## G2.5 — Exports

1. `TRANSPORT_GATE_PROTOCOL.md` v2: the study-1 20-minute preflight plus a
   mandatory calibrated backend-parity stage (the missing stage study 1's
   own gate revealed) — this remains the track's principal export to Phase
   5D and to any external adopter of the method.
2. The sentence-5 router: JVP-closed upgrade (G2.2 branch 1/2 success),
   retained blocker (branch 3), or late-band doorway (G2.4 pass), each with
   its exact licensed wording precommitted in the study-2 preregistration.
3. State of record, claim ledger v2, import bundle — same release shape as
   study 1.

---

# 5. Precommitments and forbidden moves

Carried unchanged: plan 1 §0.2 (the forbidden-meanings list), the addendum
§2.1 (thresholds before target), the two-block cap with release-regardless,
the no-secant-as-backend rule, and the no-slot-selection rule the study-1
gate already honored. Added for study 2:

1. The G2.1 ceiling derivation may cite only G2.1's own distribution;
   deriving it from, or checking it against, 0.002458 before freezing is the
   disqualifying act.
2. The G2.2 relicensing table above is immutable once this plan is adopted;
   its branches may not be reordered after G2.1's numbers exist.
3. The activation-radial direction family is redesigned (delivered-fidelity
   fix) or formally dropped in the study-2 preregistration — not silently
   absent from tables the way it is in §2.3.
4. Stage-1 rows are never edited or superseded; relicensing (branch 1)
   changes their *license*, recomputation (branch 2) creates *new* rows
   beside them.
5. If any stage observes the Qwen or OLMo-lineage packages' unopened
   outcomes, the track stops (same cross-track boundary as study 1).

---

# 6. Budget and staging

Study 1 spent its cap through: staging + calibration (56 OLMo cells),
Stage 1 (40 Gemma cells), and the parity replay. Study 2's gate stages are
cheaper: G2.1 is a few hundred JVP pairs (≈ hours), G2.2 branch 2 at worst
reruns ~80 declared-dose rows at batch 1 (well under a block). The
conditional ladder G2.3–G2.4 is where the second block goes, under the
addendum's §2.4 ordering if time runs short; G9-equivalent release time
(~2 h) is reserved off the top of the final block. Staging reuses study 1's
content-addressed verification (both snapshots' manifests are registered;
rehash before every load), and the addendum §2.5 residency rule stands:
OLMo stages in only for its control sessions.

---

# 7. Risks

1. **The ceiling lands awkwardly** — e.g., q99×3 ≈ 0.0021, just under the
   observed 0.002458. That is what branch 2 (batch-1 recompute of ~80 rows)
   is for; the rule keeps the awkward case cheap instead of tempting.
2. **fp32 injection changes the delivery funnel** and the small-ε ladder
   produces SNR-limited rows; the quantization-floor classification in the
   fit machinery already labels these rather than mistaking them for
   passes.
3. **The op-localization screen implicates a fused kernel** with no unfused
   fallback at 31B scale; then hypothesis (c) resolves at reduced width per
   plan 1's tiny-model discipline before any 31B claim.
4. **Scope creep toward the science** while the gate is open: the ladder is
   conditioned, and the paste-line makes G2.1's completion the only
   unconditional deliverable. If study 2 ends with only G2.1+G2.2 done and
   a clean license state, it has succeeded.

---

# 8. Verification appendix (for reviewers without the run mirror)

Claims in this plan trace to: the registry
(`interpretability/jspace_gemma/reports/evidence_events.jsonl`), the release
set under `interpretability/jspace_gemma/release/` (state of record, claim
ledger, transport-gate protocol, import bundle), and the development report
under `reports/`. Key registered hashes quoted above: threshold file
`3cb1e68c…`, Stage-1 summary `0f283725…` / row table `3b74f1e9…`, parity
artifact `22c32776…` / raw `ac5ba50d…`, import-bundle payload `694c62db…`.
Package verification: `bash interpretability/jspace_gemma/repro.sh` then
`python -m jspace_gemma verify` (verifies every registered output hash
without a GPU). The run mirror used to recompute this plan's tables lives at
`interpretability/jspace_runs/gemma_transport_20260802/` (metrics tree named
by evidence id); the two derived tables not present verbatim in study-1
reports (the direction-family and per-layer-fit medians in §2.3, and the
OLMo declared-dose table in §2.2) recompute in one command each from the two
row parquets, and the paper-side snapshot
(`jspace_paper/scripts/gemma_transport_snapshot.csv`) carries the layer×ε
medians for the figure without parquet access.

**Adoption boundary.** This plan becomes operative only as study 2's
preregistration candidate: adopted by an explicit Phase 5 router decision,
with PI sign-off, its own branch/registry, and the G2.1/G2.2 rules frozen
verbatim. Until then it is a reviewed proposal beside the study-1 record it
cites.
