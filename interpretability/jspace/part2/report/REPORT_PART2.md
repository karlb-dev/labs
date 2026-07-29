# J-space Part 2 — living report (ADDENDUM-GOVERNED campaign)

> ### ⚠ READ THIS BEFORE ANY SECTION BELOW — VM8 (2026-07-29)
>
> A second forensic review (`../../jspace_part2/reviews/jspace_lab_nextsteps_2_2.md`,
> with the PI's decisions in the addendum beside it) was accepted and
> executed as stages N0–N5. **Four repairs changed numbers that appear
> throughout the older sections of this file.** Where an older section
> disagrees with the VM8 section immediately below, the VM8 section wins.
> The older text is retained deliberately — the campaign supersedes, it
> does not overwrite.
>
> 1. **The clustering unit was invalid.** Every family-clustered interval,
>    ICC and the power simulation below the VM8 section used
>    `name.split("-")[0]`. Point estimates stand; **uncertainties and the
>    design do not**.
> 2. **The Instruct "WEAK-FORWARD" rung is withdrawn.** Corrected, its
>    one-hop interval no longer straddles zero, so it reads as
>    equal-depth damage. The four-model ladder's *shape* claim is weaker
>    than the older sections say.
> 3. **H7 is resolved and mostly negative.** Context-conditioning does not
>    rescue the Jacobian; the in-band gap is a linearity ceiling. Any
>    older text calling the ~50% direction accuracy "a fixable estimation
>    failure" is superseded.
> 4. **The capacity number was mislabelled** (raw energy reported as
>    centered variance) and is now larger, though still an order of
>    magnitude below the reported Claude band.

## VM9 — THE CONFIRMATORY CAMPAIGN (2026-07-29) — tier: CONFIRMATORY

**The freeze executed** (tag `jspace-part2-confirmatory-freeze-v1`;
partition seed 4242, 32/32 disjoint families, no outcome viewed between
generation and freeze), all three confirmatory cells ran (164 partition
items × 6 conditions each; G4 swap control passing per model with its
own lens: 0.76/0.76/0.78 vs random 0.18–0.24), and the locked analysis
ran once from raw parquets after the final cell banked
(`n6-confirmatory-analysis-v2`).

**BOTH PRIMARY HOLM TESTS REJECT.**

- **P-HP1** (post-training changes the task contrast): interaction
  contrast **−0.504 nats CI [−0.720, −0.295]**, p_holm = 0.0005
  (172 items / 24 intersection families, family-weighted paired
  bootstrap; the MixedLM cross-check agrees in sign at −0.43 but cannot
  represent the item pairing and is diffuse — both reported).
- **P-HP3** (J-specific protected tail vs the geometry-matched control),
  the powered Qwen test: paired tail-rate difference **+0.279
  CI [+0.205, +0.361]**, p_holm = 0.0005, stable across the threshold
  curve. The OLMo estimates are CI-clean too: Think +0.418, Instruct
  +0.488.

**The matched control (`dyn_energy_rank_matched_random`) is the
methodological star**: at exactly the J arm's per-position rank and
removed energy it produces ≈0 deltas everywhere (−0.08…+0.04 nats) — at
matched dose, direction content is the entire effect. The isotropic
mechanics arm does real damage on Qwen (−0.67 two-hop) at its larger
unmatched dose, illustrating the confound the matched control removes.

**Shape**: Qwen's forward dissociation replicates at confirmatory tier
(two-hop −1.62 vs one-hop −0.39); the OLMo pair is one-hop-dominant
(Think −1.00 vs +0.37; Instruct −1.66 vs −0.14). **Prominent caveat:
the confirmatory two-hop leg is thin (9 items / 2 families survived the
anchor difficulty window)** — HP1 rides mostly on the one-hop legs; the
untouched 32-family replication partition is the built-in check.

**Secondary**: HP2 accessibility gradient confirmed on every model
(hard −1.8…−2.7 vs easy −0.3…−0.9; r ≈ +0.4). Prose guard: Instruct's
protected-J costs +0.72 nats prose NLL (Think +0.24; controls ≤0.16) —
nonspecific-cost caveat on Instruct's one-hop number. Capacity
(corrected estimand, own lenses): 3.1 pair indistinguishable at ~1.2%
centered excess (occ 2); **Qwen at 5.0–6.1% — the Claude band's lower
edge — withdrawing the pilot's "order of magnitude below" for Qwen**.

**Amendments (all pre-outcome, stop-rule-driven)**: assay-wide BOS units
+ the gate's piecewise tokenization (AMENDMENT_1_BOS_UNITS.md; the §7
baseline stop rule fired twice at item 1 with zero outcome exposure and
caught a real two-unit-system inconsistency). Matched-control dev gate
v1→v2 (relative bound was below the float32 measurement floor).

**REPLICATION PARTITION (same day)**: HP3 **replicates** on the held-out
32 families (Qwen tail-rate +0.297 [+0.207, +0.382] vs confirmatory
+0.279; `n6-replication-analysis-v2`), and the Instruct independent-lens
leg reproduces the per-item deltas at r = 0.990 with tail Jaccard 0.885.
HP1 is **inconclusive** on the replication side (contrast +0.104
[−1.68, +1.89], all-items fallback population — the replication two-hop
leg is 9 items and absent from the intersection cohort): not a
contradiction, but the thin-two-hop limitation made concrete. A second
independent Think lens (corpus draw B) is being fitted to complete the
independent-lens clause on the think-trained primary.

Figures: `p2f15` (capacity), `p2f16` (cohorts), `p2f17` (primary
results). Next block: EXPAND THE TWO-HOP BANK (the sharpest limitation),
then the N8 fresh-VM reproduction, the capacity-vs-shape moderator
analysis, and the Instruct prose-cost follow-up.

---

**Status 2026-07-29 (VM9, superseded above at 10:2x UTC): the freeze-blocker block.** The PI
delegated the open design calls; three were resolved and recorded
(READY_FOR_FREEZE VM9 section + prereg candidate §5): (1) the **primary
matched control** is `dyn_energy_rank_matched_random` — per-position
exact rank + removed-energy match to the J arm, orthogonal to protected
rows, deterministic seeds; the complete dose match for a span-projection
intervention, leaving direction content as the arms' only difference;
(2) the **capability-cohort predicate** is `capable_generation` on the
frozen alias set (difficulty windows recorded, not selected on); (3) the
**capacity estimand stays globally centered R²** ("recentering" resolved:
centered is the confirmatory quantity, raw share is sensitivity), now
being extended to Qwen and 3.1-Instruct. A ledger error was corrected:
the 3.1-**Instruct** lens has existed since the pilot (two independent
fits); only the 3.1-**Think** lens was missing and its fit is running
this block. The campaign remains stopped at the freeze boundary —
sign-off still pending.

**Status 2026-07-29 (VM8, superseded above): the repair era is complete, the
preregistration CANDIDATE is written, and the campaign is STOPPED at the
freeze boundary awaiting principal-investigator sign-off.** See
`../../jspace_part2/READY_FOR_FREEZE.md` for the gate ledger: 5 of 8
gates pass, 3 conditions block, and all three need model weights that
were not on this VM. **No confirmatory cell has run. The item partition
has not been generated.**

**Status 2026-07-28 (VM7, superseded above): the assay-repair era is
COMPLETE and the campaign is at the P0 preregistration boundary.** The forensic review
(`jspace_part2_plan1_addendum.md`) was adopted in full: Part 1 is
reclassified as the exploratory campaign (claim corrections in
`REPORT_v2_ERRATA.md` beside REPORT_v2.md), and no cross-model
confirmatory cell runs until Workstream R passes its gates. Gates now:
**G0 operational · G1 ✓ solver · G2 ✓ lens stability · G3 ✓ intervention
invariants · G4 ✓ positive control · G6 ✓ power simulation (with a
design consequence, below)**; R6 goldens 9/9 green. **Every row carries
an evidence tier; nothing here is confirmatory.** The scientific
preregistration is drafted (`preregistration/SCIENTIFIC_PREREGISTRATION_DRAFT.md`)
and **awaiting user review — it becomes binding only when renamed and
committed, before confirmatory item generation.** Original pre-data
prereg `27c6d3c` (kept as record) is superseded by
`REPAIR_PREREGISTRATION.md`.

**The four-model ladder is the campaign's headline pilot finding.** Under
the paper's own protected dynamic ablation, the two-hop-vs-one-hop causal
signature is not a fixed property of "an LLM": it REORGANIZES across
post-training — base REVERSE (one-hop hit, two-hop spared) → Think EQUAL
(both hit) → Instruct WEAK-FORWARD → Qwen CLEAN-FORWARD (the paper's
shape). Because Instruct and Think are identical on measured capacity
(occupancy 2), the shape tracks the post-training regime rather than
occupancy. The ceiling check then relabeled the ladder's bottom rung:
the base model's one-hop effect is an EASY-FACT effect (hard one-hop is
null, +0.14), so the confirmatory hypotheses are stated in
fact-accessibility terms, not task depth.

## VM8 — the repair block that changed the design (2026-07-29) — tier: PILOT

Four results, each superseding something above.

### 1 · The clustering unit was wrong, and it was load-bearing
`n2-corrected-family-pilot-v1` · figure `p2f10_family_correction.png`

`battery.py` derived the statistical family as `name.split("-")[0]`. That
is a string accident: `atomic-80-state` and `ex-element-state-80-8` are
the same template but landed in families `atomic` and `ex`, sixteen
unrelated items shared the family `ex` purely because their names start
with those letters, and each of the ten one-hop "capital of X" items was
its own family. Under the hand-audited map
(`jspace_part2/data/probe_swap_family_map.json`, 120 items, rule
*(second-hop relation, answer type)*) the pilot's 60 two-hop items are
**25 canonical families, not 38 raw labels**, and `country_capital` alone
holds 11 of them.

Point estimates are unchanged — clustering touches uncertainty, not
means, which is the sanity check this repair had to pass. What moved:

| | base | Think | Instruct | Qwen |
|---|---|---|---|---|
| ICC, prefix field | 0.42 | 0.37 | 0.42 | 0.40 |
| ICC, **audited** | **0.66** | **0.56** | **0.17** | **0.75** |

The apparent uniformity was an artifact. G6's whole power calculation was
calibrated on a homogeneity that does not exist.

**One conclusion flips.** Instruct one-hop moves from
−0.45 [−1.03, **+0.06**] to −0.45 [−1.29, **−0.01**]. The VM6 ladder
called Instruct "WEAK-FORWARD" *because* its one-hop interval straddled
zero; corrected, it marginally does not, so it reads as equal-depth
damage like Think. This is knife-edge and it reverses again under
family-weighting — which is exactly why the candidate fixes the weighting
estimand in advance rather than after seeing the answer.

### 2 · The binary primary is not affordable on the OLMo pair
`g6-power-sim-v3`, `g6-tailrate-power-by-model-v1`, `g6-mde-v1` ·
figure `p2f11_feasibility.png`

Inverting the power question — given *m* families, what is the smallest
detectable effect at 90%? — at the 60 canonical families the bank now
holds:

| model | MDE rate @90% | pilot gap | MDE nats @90% | pilot mean |
|---|---|---|---|---|
| OLMo Think | 0.30 | 0.133 | 1.00 | −0.52 |
| OLMo Instruct | 0.30 | 0.100 | 0.75 | **−0.80** |
| OLMo base | 0.20 | 0.067 | 0.75 | −0.14 |
| Qwen | **0.25** | **0.267** | 1.50 | −0.91 |

Only **two** cells can carry a binary test — Qwen on the rate endpoint,
Instruct on the mean endpoint — and the two endpoints have *opposite*
feasibility profiles across models. HP3's Think tail effect would need
>150 families to test. The candidate therefore states most hypotheses as
estimation with intervals and declares a test only where power exists.
That is not a hedge; it is what the measurement supports.

### 3 · D2 resolved: the in-band gap is a ceiling, not an estimator failure
`h7-context-j-olmo3-think-v2`, `h7-linearity-ceiling-olmo3-think-v1`,
`h7-synthesis-olmo3-think-v1` · figure `p2f12_h7_ceiling.png`

The design point that made this measurable: for a **uniform**
perturbation the per-position and corpus-averaged estimators are
algebraically identical, `sum_s J_s @ d == P·(mean_s J_s) @ d`. The
earlier faithfulness test used a uniform probe and therefore *could not
see* position-averaging loss at all. Perturbing one position at a time —
what the dynamic ablation actually does — separates them.

- **Position-conditioning does not improve direction** at any band layer
  (−0.05, −0.04, −0.02; +0.02 at the late control). It does roughly halve
  the magnitude error (norm ratio 0.19 → 0.39 at L24).
- **Cosine tracks the ground truth's own local linearity**, within layer,
  at Pearson r **0.76–0.90** (n=144 cells). The estimator is not the
  binding constraint.
- **Where the ablation actually acts**, the lens is much better than the
  usual quote suggests: in-band cosine **0.58** along selected J rows vs
  **0.33** for random probes.

So H7's "averaging discards the accuracy" is **false for direction** and
partly true for scale, and the `contextJ` methods arm **fails its
committed dev gate and is not admitted**. This revises VM7's reading: the
uniform probe found OLMo linear across the band (1.98–2.02) and concluded
the gap was fixable estimation error. Both probes are correct — a uniform
shift moves every key together and partly cancels inside the attention
softmax, a single-position shift does not — and under the probe that
matches a position-wise intervention the band is measurably less linear
at shallow depth (1.65 at L24). **Causal claims about position-wise
interventions must cite the single-position numbers.**

### 4 · The capacity estimand was mislabelled, and is larger
`r2-occupancy-think-v2` · figure `p2f13_capacity_corrected.png`

v1 computed a centered activation and never used it; both shares were
raw-energy while the prose called them the paper's centered excess
variance. Occupancy is **unchanged** (median 2 at L24/32/40 — it never
depended on the share definition). Centered excess: 0.49% / 1.02% /
**1.27% [1.18, 1.34]** versus raw 0.03% / 0.54% / 0.80%. The capacity
headline survives in slightly stronger form — still an order of magnitude
below the 6–10% reported for Claude.

### Also banked this block
- **Registry v2** (event-sourced). Its validation caught two supersede
  links pointing at evidence ids that were never created, so
  `local-linearity-gemma4-31b-v1` and
  `linearization-faithfulness-gemma4-31b-v1` — both Gemma analyses whose
  conclusions had been *reversed* — were still reading as LIVE evidence.
- **Reproduction that reproduces** (`repro-contract-v2-acceptance-v1`):
  two items rebuilt end-to-end from isolated worktrees at their recorded
  commits.
- **`protected_dynamic_v2`**: the v1 generation path broadcast one
  final-position protection set across every prompt position, and one
  starved position shrank the dose for the whole sequence. Both repaired,
  with a real-transformer golden.
- **G5 PASS** (`g5-item-manifest-v3`) · figure `p2f14_bank_readiness.png`:
  1052 items, 64 families with ≥3 capable items, 15 items excluded for
  answer-in-prompt leakage, zero bridge leakage. The gate also caught
  **my own authoring error** — nine v4/v5 "new" families were
  re-expressions of existing relations, four sharing literal facts, which
  would have let one fact land in both partitions.

## Campaign question (revised)

Does the paper's workspace signature exist in open models when measured
with the paper's own intervention (output-protected dynamic ablation —
missing from Part 1), the paper's own capacity estimand (marginal-gain
occupancy + excess variance — not what Part 1 computed), and controls
matched for energy, effective rank, and geometry? The hypothesis ladder is
H0 instruments-first, H1a/b/c externalization variants, H2 output
alignment, H3 bounded residual, H4 scale (descriptive), H6 task demand,
H7 mean-J mismatch, H8 sparse-frame geometry (see part2 README).

## Matrix scoreboard (all cells tier=pilot or below; grows per banked cell)

| cell | olmo3-base | olmo3-think | olmo31-instruct | qwen36-27b | gemma4-31b |
|---|---|---|---|---|---|
| lens | Think-lens (transfer PASS) | own 120p (part 1) | own 120p A + independent B | Neuronpedia 1000p | 120p fit queued (band from deep-band sweep) |
| sanity probes @20 | 17/21 | 17/21 | 17/21 | pass | readout blocked mid-band |
| multihop J pass@1 | 0.283 (+0.083) | 0.283 | 0.217 (own lens = transfer) | 0.350 | — |
| occupancy (paper estimator) | — | **2** (centered excess 0.49–1.27%) | **2** (raw ≤0.87%, recompute queued) | **3–4** (raw 1.0–2.2%) | — |
| protected dyn-J twohop † | −0.14 [−0.45,+0.08] | −0.52 [−1.09,−0.08] | **−0.80** [−1.62,−0.18] | **−0.91** [−1.35,−0.59] | — |
| protected dyn-J onehop † | **−0.81** [−1.40,−0.38] | −0.59 [−1.23,−0.17] | −0.45 [−1.29,−0.01] | −0.06 [−0.43,+0.48] | — |
| → dissociation shape † | **REVERSE** | EQUAL-DEPTH | EQUAL-DEPTH *(was WEAK-FORWARD; the one-hop interval no longer straddles zero)* | **CLEAN-FORWARD** (paper's) | — |
| hard-onehop ceiling check | **+0.14 NULL** (rand −0.01) | (dev set source) | — | — | — |
| frozen-J / frozen-logit | — | −2.89 / −1.54 | — | −2.42 | — |
| temp-0.7 replicate | — | structure preserved | — | — | — |

† Intervals recomputed on the **audited canonical families**
(`n2-corrected-family-pilot-v1`, 25 clusters for the 60 two-hop items);
item-weighted, with family-weighted reported alongside in the evidence.
The pre-repair intervals in older sections clustered on the defective
prefix field and should not be quoted.

Reference band (paper, Claude): occupancy 10–25, excess 6–10% — every
open model measured sits an order of magnitude under it, with fit size
ruled out (n=1000 Qwen lens agrees with n=120 OLMo recipe). OLMo Think's
excess is the corrected **centered R²**; the other cells still carry the
v1 raw-energy quantity and are queued for recomputation (stage N7).

## A0 — does the J-dictionary survive post-training? (2026-07-27) — tier: EXPLORATORY (transfer-geometry experiment)

*Addendum reframe (§18.4): lens transfer is a scientific outcome about
Jacobian-geometry conservation across post-training, never a substitute
for recipient fitting (which is running). Rank-based readout carries a
multiple-opportunity caveat (min over 21 layers × answer variants, §5.3);
the donor-vs-recipient DELTA at identical search freedom remains
informative. Confirmatory transfer analysis later adds dictionary CKA,
projector principal angles, and a small recipient fit comparison.*

**Setup.** Part-1 Think lens (120-prompt WikiText, 21 layers) applied
unchanged to `Olmo-3.1-32B-Instruct` — same base model (`Olmo-3-1125-32B`),
same tokenizer/architecture, different post-training. Transfer semantics:
donor J, recipient unembedding (how a transferred lens would actually be
consumed). Preregistered gate: probes ≥15/21 AND multihop pass@1 ≥ 0.85 ×
donor = 0.2408.

**Result: TRANSFER_FAIL, of the most informative kind.**

| quantity | donor (Think) | transferred (3.1-Instruct) |
|---|---|---|
| probes hit@20 (n=21) | 17 | **17** (same count; misses overlap: lira/Atlantic/cheetah/two) |
| multihop pass@1 (n=60) | 0.283 | **0.217** (< gate 0.2408) |
| multihop pass@5 / @20 | 0.41 / 0.55 | **0.417 / 0.522** (donor level) |
| J-minus-own-logit pass@1 | +0.083 | +0.017 |
| unembed row cos (donor vs recipient, 4000 rows) | — | median **0.9984** (q05 0.9971) |
| final-norm gain cos | — | **1.000** |

**Reading.** The readout *basis* is untouched by post-training (unembed rows
and final-norm gain essentially identical), the direct-fact dictionary
transfers perfectly (probe count identical, same items miss), and the
bridge-entity **shortlist** transfers at donor level (pass@5/@20). What does
not transfer is precisely the **rank-1 sharpening** — the very component
part 1 identified as the J-lens's distinctive advantage (+41% relative
pass@1 on Think, only +8.5% relative here). Two candidate causes, now
disambiguated by the running Instruct-lens fit (prereg FAIL rule): (a) the
mid-band J-map drifted under post-training → own-lens pass@1 recovers
toward ~0.28; (b) Instruct genuinely resolves the silent bridge more weakly
at the readout position → own-lens stays ~0.22, which would itself be an
H1-flavored datum (think-training strengthens pre-answer internal hop
resolution).

Figure: `figures/p2f1_a0_transfer.png`. Metrics:
`metrics/olmo31-instruct/a0_transfer_gate.json`.

**RESOLVED 2026-07-27 23:37 (own-lens re-gate, tier exploratory,
evidence `a1-ownlens-regate-olmo31instruct-v1`):** a recipient-fitted
120-prompt lens (same corpus/recipe/geometry as the donor; 5h23m fit,
4×30 slices, one isolated fit anomaly recorded at corpus row ~104)
reproduces the transferred lens EXACTLY at aggregate level — probes
17/21, multihop pass@1 0.217, J-minus-logit +0.017 — while per-item ranks
differ (24/68 identical, median |Δ| 3, max 2055), i.e. two genuinely
different instruments agree. Reading: **(a)** the mid-band J-map is
conserved across post-training at n=120 fit resolution (transfer ≈
recipient fit — the A0-as-geometry-experiment outcome); **(b)** the
rank-1 bridge-resolution deficit (0.217 vs Think's 0.283; advantage
+0.017 vs +0.083) is a **model property of the Instruct sibling**, not
lens drift. First H1a-flavored datum: think-branch post-training is
associated with sharper pre-answer internal hop resolution. Caveats:
n=60, one lens per model, rank-search freedom (addendum §5.3), both
models' readouts hit the same probe items; confirmatory version rides
the A1 lineage study under the repaired assay.

## D — output-alignment analysis (renamed from "occupancy", addendum §18.12), phase 1: Think saved traces — tier: EXPLORATORY

Computed on part-1's 38 think-mode generation traces (s8; per-step top-8
J-readout at L{24,32,40} from the state producing each token). Chance floor
for top-8 membership ≈ 0.008%.

| layer | occ@1 (median item) | occ@8 | lead-1 @8 |
|---|---|---|---|
| L24 | 0.100 | 0.306 | 0.132 |
| L32 | 0.258 | 0.608 | 0.216 |
| L40 | 0.363 | **0.708** | 0.233 |

**Reading.** In think mode, the live verbalizable readout on OLMo-Think is
massively occupied by the imminent output stream — at L40 the next token is
in the top-8 in 71% of steps and IS the top direction in 36% — rising
monotonically toward the unembedding. This is H2's mechanism for why live
per-token ablation lobotomizes and static span ablation finds nothing:
what the live J-space "holds" is largely the word being formed. The
cross-model discriminator: matrix ride-alongs (A1/A2/A3 batteries, no-think
regime to match the causal grids) test whether occupancy is *lower* wherever
a dissociation appears. Caveats: top-8 truncation (full-rank version rides
along in phase 2); think/post-`</think>` segmentation pending a multi-token
boundary matcher.

## B3 — frozen-logit control (2026-07-27 23:48) — tier: EXPLORATORY-PILOT

Same per-item frozen mechanism as the part-1 marquee instrument,
dictionary = plain (W_U⊙g) rows, no Jacobian; pool sizes inherently
matched (vocab-sized both). Evidence `b3-frozen-logit-pilot-think-v1`;
figure `p2f2_b3_frozen_logit.png`.

| condition | twohop_lp (Δ vs none) | twohop acc | onehop acc | prose NLL |
|---|---|---|---|---|
| baseline | −1.74 | 0.58 | 0.60 | 2.71 |
| frozen-J top-10 (part-1 cells) | −4.63 (**−2.90**) | 0.23 | 0.23 | 3.00 |
| **frozen-logit top-10** | **−3.28 (−1.54)** | 0.33 | **0.23** | 2.81 |
| frozen-random (5120) | −1.78 (−0.05) | 0.53 | 0.53 | 2.79 |

**Reading (between the prereg's two poles):** output-aligned unembedding
directions alone reproduce ~53% of the frozen-J logprob deletion and the
*identical* one-hop collapse, while the random twin sits on baseline — so
a large share of the "content channel" is reachable without the Jacobian,
and the pullback roughly doubles composed-task damage on top. This
sharpens H2 (output alignment) and H8 (dictionary-geometry artifact)
rather than settling the method claim; the confirmatory decision needs
the R3 family (J-rotated, label-shuffled, rank-matched, paired per-item
stats). Caveats: part-1 mechanics by design (raw QR, first-token lp,
both-phase hooks), unpaired CIs, single seed, n=60/31.

## R2 — paper-defined occupancy, first commensurable capacity numbers (2026-07-28) — tier: PILOT

Evidence `r2-occupancy-think-v1` (frozen crossing rule, 8 synthetic solver
tests green, commit `eeed9ad`). Olmo-3-32B-Think, shared descriptive set
(60 prompts, ~4.9k positions/layer), 3 vocab-sized random control
dictionaries:

| model / layer | occupancy median [IQR] | excess variance share |
|---|---|---|
| Think L24 / L32 / L40 | 2 / 2 / 2 | 0.0003 / 0.0054 / 0.0080 |
| Instruct L24 / L32 / L40 | 2 / 2 / 2 | 0.0012 / 0.0057 / 0.0087 |
| **Qwen3.6-27B** L24 / L32 / L40 (their n=1000 lens) | — / **3 [2,4]** / **4 [3,5]** | — / **0.0101** / **0.0224** |
| paper (Claude) | 10–25 (k≈25 typical) | 0.06–0.10 |

**The decisive comparison landed 05:52 (evidence
`r2-occupancy-qwen36-v1`): Qwen is NOT paper-range under the paper's own
estimator.** Part-1's "paper-range capacity" was proxy inflation
(thresholded counts + raw share ≠ marginal-gain occupancy + excess).
What's real: a graded Qwen>OLMo difference (~2–3×, growing with depth) —
and an order-of-magnitude gap between ALL tested open models and the
paper's Claude numbers that cannot be blamed on lens fit size (n=1000 vs
n=120 read the same way), post-training regime (Think = Instruct), or
family (OLMo ≈ Qwen within 3×). Remaining suspects for the open-vs-Claude
gap: training scale/regime (H3/H4 bounded residual) or mean-J lens
quality per model (H7 — testable via local-J on a subset). This
consolidates the boundary-of-generalization outcome for capacity.

## R7 pilot — the paper's protected dynamic ablation, first faithful run (2026-07-28) — tier: PILOT

Evidence `r7-protected-dynamic-pilot-think-v1` (commit `5ea4ee2` runner;
per-item parquet + summary with provenance). Olmo-3-32B-Think, band
L20–44, k=10, protect-top-10, exact batched rank-safe projection,
nonnegative selection, dual clean/ablated KV streams, full-answer-sequence
logprob scoring, paired per-item deltas. The single changed variable
between the two dyn-J arms is the protection mask.

| condition | twohop Δlp (mean / median, n=60) | onehop Δlp (n=30) | prose ΔNLL/tok (n=20) | removed energy |
|---|---|---|---|---|
| dynJ **protected** (paper protocol) | **−0.52 / +0.02** | −0.59 / +0.01 | +0.24 | 0.12% |
| dynJ unprotected | −1.13 / −0.55 | −1.71 / −1.56 | +0.29 | 0.13% |
| dynR protected (matched mechanics) | −0.16 / −0.01 | +0.09 / +0.14 | +0.14 | **2.6%** |

**Reading 1 — H2 confirmed in causal form.** The clean-output protection
flips the median item from clearly-deleted (−0.55/−1.56) to untouched
(+0.02/+0.01): the bulk of part-1's live-ablation damage was literally
deleting the token being emitted. The paper's safeguard is not a detail;
it is most of the phenomenon on this model.

**Reading 2 — a J-specific internal-content tail survives protection.**
The protected mean (−0.52) hides a heavy tail: 16/60 twohop items lose
>1 nat (mean −2.37 on that cohort) while the protected random control on
the SAME items sits at +0.34. Tail items are the harder ones (baseline lp
−3.33 vs −1.17) — precisely the items whose answers were NOT already
imminent output (and hence not protected/not output-occupied). This is
the first protected-protocol evidence of internally-held J-content
deletion, and it reconciles the part-1 story in one picture: **the
workspace effect is conditional on content not yet being in the output
stream** — items where the answer is already surfacing are untouched by
construction, items holding it internally lose it, and the median just
reflects the battery's easy/hard mix.

**Specificity bonus:** dyn-J removes ~0.12% of raw activation energy vs
dyn-random's 2.6% (energy-matched-by-mechanics, 20× more) — yet only
dyn-J produces the tail. Selection content, not removal size.

**Clean-rank mechanization test (banked 00:3x, evidence
`r7-cleanrank-think-v1`):** 94% of battery items carry the answer inside
the clean top-10 (protection covers nearly the whole battery — C3's hard
items will widen the unprotected cohort). Items with rank>10 are crushed
as the mechanical account predicts (twohop −2.68, onehop −4.75). **But
13/16 twohop tail items have the answer at clean rank 1–3 — explicitly
protected — and still lose 1–4.5 nats.** The protected-protocol damage on
those items is therefore INDIRECT: deletion of other J-content the answer
depends on (bridge-entity reading) or geometric leakage through
answer-similar rows; protected-random's +0.34 on the same items excludes
generic-removal accounts. Discriminators queued for the confirmatory
grid: per-item selected-id sets (was the bridge token selected?) and
removed-energy-along-answer-direction logging.

Other follow-ups: occupancy-conditional split; dose/persistence + C1 load
battery targeting the tail cohort; Instruct mirror of this grid. Caveats:
pilot tier, single seed, prose guard shows a real +0.24 NLL/tok fluency
cost for dynJ (protected included), audit generations coherent.

## R7-Qwen — the protected protocol transfers, and the DISSOCIATION appears (2026-07-28 06:1x) — tier: PILOT

Evidence `r7-protected-dynamic-pilot-qwen-v1`; same runner/config family
as the Think grid (their n=1000 lens; chunked dictionary build after a
248k-vocab OOM, fix committed).

| Δ answer-seq lp (mean/median) | dynJ protected | dynJ unprotected | dynR protected |
|---|---|---|---|
| two-hop (n=60) | **−0.91 / −0.40** | −1.24 / −0.80 | −0.23 / −0.21 |
| one-hop (n=30) | **−0.06 / +0.03** | −0.05 / −0.00 | +0.02 / +0.01 |
| prose ΔNLL/tok (n=20) | +0.35 | +0.35 | +0.11 |

**Reading (paired family-clustered bootstrap CIs, `r7_paired_ci.json`
both models).** The cross-model contrast is now CI-clean at pilot tier:

| protected dyn-J Δlp [95% CI] | two-hop | one-hop | dissociation? |
|---|---|---|---|
| **Qwen3.6-27B** | **−0.91 [−1.49,−0.60]** | −0.06 [−0.37,+0.23] | **YES — paper's shape** |
| OLMo-3-32B-Think | −0.52 [−0.85,−0.13] | −0.59 [−1.14,−0.13] | NO — equal-depth damage (content channel) |

Same instruments, same items, same protocol: the paper's causal signature
appears on the fatter-workspace model and not on the thin one, matching
the capacity axis (occ 3–4 vs 2). Caveats: single seed; Qwen's random
twohop control is small-but-nonzero (−0.23 [−0.34,−0.11]) so the J-vs-
random contrast needs its own paired test in the confirmatory grid;
Qwen one-hop = near-ceiling capitals (C3 hard set must confirm the spared
side); fluency only partially survives (+0.35 nats/token vs random's
+0.11); OLMo's protected effect includes the indirect hard-item tail
documented above.

## A2a cell 2 — chat-mode grids expose a register confound (2026-07-28) — tier: PILOT, METHODS FINDING

Evidence `a2a-mode-grid-qwen-v1`. Protected dyn-J and dyn-random grids on
chat-rendered prompts (official thinking toggle) with teacher-forced
bare-answer scoring produced LARGE POSITIVE deltas (dynJ think-on: +4.0
twohop / +4.4 onehop nats; even dyn-random +1.7/+3.1). Reading: after the
template — especially an open `<think>` — the baseline distribution is
dominated by the reasoning-register opener, not answer content; ablating
live J-directions (which are register-occupied in this context, per the
output-alignment picture) un-anchors the register and relatively inflates
any appended continuation. **Two consequences:** (1) direct evidence that
in chat contexts the strongest live J-content is the act-of-responding
register itself; (2) a binding design rule — chat-mode causal cells must
use generation-based endpoints (budgeted think + post-`</think>`
final-answer grading), never teacher-forced bare-answer lp. The raw-
completion R7 grids are unaffected (their baselines are natural
continuations). H1b's causal contrast moves to the confirmatory design
with the corrected endpoint.

## R5/G4 — swap positive control PASSES (2026-07-28 08:1x) — tier: PILOT

Evidence `r5-swap-positive-control-think-v1` (released probe-swap items;
remove-bridge + inject-swap J-directions at band layers, prefill-only;
dose calibrated on 10 recorded items, measured on the remaining 50).

| condition | two-way flip rate to swap answer | mean lp(swap ans) | mean lp(orig ans) |
|---|---|---|---|
| none | 0.04 | −9.10 | −1.48 |
| **swap-J (predicted direction)** | **0.76** | −38.07 | −48.24 |
| swap-random (matched perturbation) | 0.18 | −13.13 | −12.37 |

**G4 passes**: the instrument moves answers in a PREDICTED direction 4×
above matched random perturbation — the anchor that makes the campaign's
nulls evidence rather than insensitivity. Wrinkle recorded: at the
calibrated α=0.2 the relative preference flips while absolute calibration
craters (both answers' lp collapse); α=0.05 showed clean positive steering
(+0.93 lp toward swap, 0.30 flips) — the confirmatory swap cell dose-maps
0.05–0.10 for calibration-preserving steering. First steering-INTO-J
causal mode this replication has exercised.

## A1 Instruct mirror + independent-lens reproduction (2026-07-28) — tier: PILOT

Evidence `r7-protected-dynamic-pilot-olmo31instruct-v1` (own lens A) and
`…-lensB-v1` (independent disjoint-corpus lens B).

Instruct under the protected protocol: two-hop **−0.80 [−1.25, −0.36]**
(CI-clean), one-hop **−0.45 [−1.03, +0.04]** (straddles zero) — a WEAK
forward dissociation, intermediate between Think's equal-depth damage and
Qwen's clean shape. **This is the first clean H1a causal datum**: Instruct
and Think have identical measured capacity (occupancy 2, excess ≤0.9%),
so the causal shape cannot be tracking occupancy here — it tracks the
post-training regime. Prose cost +0.62 (Think +0.24).

Reproduction with lens B (independent draw, disjoint fitting corpus):
per-item protected deltas correlate **0.989** with lens A; the 17-item
tail stays the tail (15/17 below −1 nat, mean −3.37); grid means within
0.09 nats. **The protected tail is a model property, not a fit artifact.**

## A1 base leg — the ladder's bottom rung, twice relabeled (2026-07-28) — tier: PILOT

Evidence `a1-base-leg-gate-and-grid-v1`, then `a1-base-hard-onehop-v1`.

(a) **Transfer**: the Think lens reads the base model (`Olmo-3-1125-32B`)
at DONOR-IDENTICAL resolution — pass@1 0.283, advantage +0.083, probes
17/21. Since Instruct reads at 0.217, instruct-tuning DEGRADED bridge
resolution; think-training PRESERVED what the base already had. The
rank-1 bridge deficit is a property of the post-training branch, not
something think-training created.

(b) **Grid**: base shows a REVERSE dissociation — one-hop **−0.81
[−1.29, −0.40]** CI-clean, two-hop −0.14 straddling zero; random controls
clean. Combined with Think/Instruct/Qwen this yields the four-model
ladder: **REVERSE → EQUAL → WEAK-FORWARD → CLEAN-FORWARD**. The paper's
signature reorganizes non-monotonically under post-training.

(c) **Ceiling check (the relabel)**: the base one-hop battery is
near-ceiling capitals. Rerunning the protected arm on the 41-item HARD
one-hop dev set (difficulty-matched to two-hop on Think) gives **+0.14
mean / +0.21 median — a NULL** (matched random −0.01). So the −0.81 was
an EASY-FACT effect: **rehearsed facts occupy the deletable
output-adjacent channel; hard facts do not.** Consequence for P0: ladder
hypotheses must be stated in fact-accessibility terms, not task depth
alone. (This is also the C3 dev set's first validating use.)

## The method's linear-transport premise: holds on OLMo, fails on Gemma-4 (2026-07-28 20:5x) — tier: PILOT

Evidence `local-linearity-v3-olmo3think`, `local-linearity-v3-gemma4-31b`.

A Jacobian lens assumes the source→target map is well described by a
linear operator. That assumption is testable on the network alone, with
**no fitted J involved**: perturb the residual by the same δ at every
valid position and check superposition, `r(2δ) = 2·r(δ)` and
`r(a+b) = r(a)+r(b)`. Every cell reported here has input fidelity 1.000
(bf16 delivered the perturbation exactly); cells failing that check are
marked unmeasurable rather than nonlinear.

| ε | OLMo L24 | OLMo L32 | OLMo L40 | Gemma L22 | Gemma L30 | Gemma L37 |
|---|---|---|---|---|---|---|
| 0.02 | 1.66 | 1.87 | 1.93 | 1.54 | 2.64 | 1.27 |
| 0.05 | 1.91 | 1.97 | 1.98 | 1.37 | 1.83 | 1.58 |
| 0.10 | **1.98** | **1.99** | **2.01** | 1.59 | 1.17 | 0.81 |
| 0.20 | **1.99** | **2.01** | **2.02** | 1.79 | 1.34 | 0.87 |

*(scale ratio; 2.00 = perfectly linear)*

**OLMo is linear across the paper's band** (L16–L60; only the very
shallow L4 fails). **Gemma is nonlinear at every layer and every scale**,
and critically does not improve as ε grows — L37 runs 1.27 → 0.87 with
additivity error climbing 0.76 → 1.30. That divergence is what
distinguishes real nonlinearity from a measurement floor.

**Consequences, which resolve the A3 thread.** On OLMo the premise is
sound, so the fitted J's mere ~50% direction accuracy in-band is an
**estimation** failure: jlens averages the Jacobian over positions and
fitting prompts, discarding most of the achievable accuracy. That is the
campaign's H7 "mean-J mismatch", now measured — and it is fixable, with a
sharp prediction that a per-position Jacobian should be markedly more
faithful. On Gemma the premise **fails**, so no Jacobian models the
transport however well estimated. This explains mechanistically why
Gemma's fitted lens is identified-but-useless (every corpus recovers the
same *bad linear approximation* of a nonlinear map) and why it reads
worse than the plain logit lens (J actively mis-models transport; the
logit lens makes no transport claim).

**The A3 statement can therefore be sharpened**, though still not into a
claim about Gemma's cognition: Gemma-4 violates the linear-transport
premise the Jacobian lens is built on. It remains untrue to say "Gemma
has no workspace" — a nonlinear architecture could host one that this
method cannot see.

*(A v1 of this test was withdrawn: sweeping ε from 0.005 it reported
in-band nonlinearity on both models, but the trend ran backwards —
linearity improved with larger ε, the signature of a precision floor
rather than nonlinearity. In bf16 a perturbation 200× smaller than the
activation loses direction on addition, and the response side is worse.
v3 moves ε up into the faithfully-delivered range — also where
ablation-scale interventions operate — and gates each cell on input
fidelity.)*

## How good a first-order model is the Jacobian? (2026-07-28 20:1x) — tier: PILOT

Evidence `linearization-faithfulness-olmo3think-v2`,
`linearization-faithfulness-gemma4-31b-v2`.

The A3 verdict left "identified but useless" unexplained, so we asked
whether the fitted J is a faithful first-order model of what it claims to
model. jlens builds `J[i,j] = mean_s [ sum_t ∂h_tgt[t,i]/∂h_src[s,j] ]`
— position-averaged, target-summed — so the test perturbs h_src by the
same δ at every valid position and compares the measured change in the
summed target activation against `P·(J@δ)`. The bf16 noise floor measured
exactly 0.000 (deterministic forwards), so nothing below is numerical.

**On OLMo, where the method works and J beats the logit lens:**

| layer | 4 | 16 | 24 | 32 | 40 | 48 | 56 | 60 |
|---|---|---|---|---|---|---|---|---|
| cosine | 0.11 | 0.23 | **0.49** | **0.69** | **0.77** | 0.85 | 0.93 | 0.98 |
| ‖pred‖/‖actual‖ | 0.13 | 0.25 | **0.41** | **0.57** | **0.67** | 0.77 | 0.89 | 0.96 |

Only the final two layers are faithful (cos ≥ 0.9). **Across the paper's
own band (L24/32/40 ≈ 37–62% depth) the linearization is merely
partial** — roughly half-right in direction, capturing 41–67% of the
response magnitude. Part of the depth trend is trivial (layers nearer
the target are nearer identity), but the in-band values stand on their
own.

**Gemma at matched relative depth is qualitatively different:**

| relative depth | Gemma cos | Gemma ratio | OLMo cos | OLMo ratio |
|---|---|---|---|---|
| ~37% | **−0.18** | 0.88 | 0.49 | 0.41 |
| ~50% | 0.47 | 1.04 | 0.69 | 0.57 |
| ~62% | 0.27 | 1.90 | 0.77 | 0.67 |
| deepest | 0.10 | **2.5–4.3** | 0.98 | 0.96 |

OLMo's linearization degrades gracefully and under-predicts; Gemma's
never becomes faithful at any depth (peak 0.47, anti-correlated at the
band's shallow edge) and over-predicts by 2–4× at depth, consistent with
its 8× larger ‖J‖/√d. That is a mechanism for "identified but useless":
independent corpora agree on a linear map that is not a good model of the
transport.

**A dissociation worth carrying forward.** At Gemma L52 the J-lens reads
the answer at rank 2 while its transport model is unfaithful (cos 0.10,
over-predicting 2.5×). Readout quality and transport fidelity are
separable, so deep-layer readout success does **not** license causal use
of the same J.

**Scope.** This does not touch the protected-ablation results, which are
behavioral, and does not invalidate J-readout, since ranking can survive
a poor magnitude model. It does mean that J-direction causal claims rest
on a transport model that is about half direction-accurate in the studied
band *even on the model where the method works* — the campaign's H7
"mean-J mismatch" hypothesis, now measured rather than asserted.

*(A v1 of this test was withdrawn: it compared a single-position
derivative against the position-averaged object jlens fits. It was caught
by an internal inconsistency — Gemma L52 reads at rank 2 while v1
reported cos 0.045. Standing lesson: when a new instrument contradicts an
established result, suspect the instrument first.)*

## A3 Gemma — the leg is COMPLETE, with a verdict and a caveat (2026-07-28 20:0x) — tier: PILOT

Evidence `a3-gemma-fullfit-v1` → `a3-gemma-identification-v1` →
`a3-gemma-readout-verdict-v1`.

**The fit.** 120 prompts, 4 slices, OLMo-commensurable recipe, band
22/30/37/40/42/44/48/52 chosen from the depth sweep below.

**The identification gate — which is why the verdict is trustworthy.**
The merged L22 Jacobian norm (0.042) came in *below* every slice that
built it (0.124/0.098/0.029): averaging shrank it, meaning the slice
Jacobians disagreed in direction and cancelled. Per-layer cross-slice
cosine confirms it:

| layer | 22 | 30 | 37 | 40 | 42 | 44 | 48 | 52 |
|---|---|---|---|---|---|---|---|---|
| mean pairwise cos | **0.057** | 0.967 | 0.994 | 0.994 | 0.995 | 0.996 | 0.996 | 0.997 |

L22 — Gemma's 37% relative depth, the shallow edge of the paper's band —
is **NOT IDENTIFIED**: four independent 30-prompt fits recover mutually
near-orthogonal maps. A readout claim there would be vacuous in either
direction, since a bad rank measures the FIT, not the model. The staged
decision rule had L22 inside its band, so the in-flight sweep was killed
before writing anything, the gate was added, and the verdict re-derived
on the identified paper-band layers L30 (50% depth) and L37 (62%).

**The verdict: `GEMMA_NO_RESCUE`** (n=50 probes, 40 known-answer).
Median answer-token rank:

| layer | 22 *(no J)* | 30 | 37 | 40 | 42 | 44 | 48 | 52 |
|---|---|---|---|---|---|---|---|---|
| J-lens | *(41892)* | 61022 | 44640 | 8054 | 3459 | 880 | 98 | 2 |
| logit lens | *(86234)* | 31354 | 17746 | 3531 | 92 | 23 | 9 | 1 |
| gain logit/J | *(2.06)* | 0.51 | 0.40 | 0.44 | **0.027** | **0.026** | 0.09 | 0.67 |

Two findings, the second unplanned and stronger than the first: (1) a
fully fitted, verifiably identified J-lens does **not** read the answer
anywhere in the paper's band; (2) the J-lens is **worse than the plain
logit lens at every identified layer** — ~40× worse at L42–44. On OLMo
the Jacobian *beat* the logit lens, and that advantage is the method's
whole premise.

**Caveat, stated deliberately: identification ≠ validity.** Cosine
0.97–1.00 proves independent corpora recover the *same* map; it does not
prove it is the *right* map. A recipe mis-specification for this
architecture would reproduce just as cleanly, and finding (2) — J
systematically worse than the trivial baseline — is exactly what such a
mis-specification looks like. **The honest statement is therefore "the
paper's method as we implement it does not transfer to Gemma-4," NOT
"Gemma-4 has no workspace."** Separating those requires the queued
follow-up (pre-cap Jacobian target, target-layer sweep, norm-placement
re-check). No Gemma family verdict beyond this wording.

**Related latent bug, caught before it produced a result**
(`norm-gain-convention-fix-v1`). While investigating recipe validity:
`build_j_dictionaries` read `norm.weight` directly. That is correct for
Llama/OLMo/Qwen RMSNorm (`x_normed * w`) but **wrong for Gemma**, which
applies `x_normed * (1 + w)` (transformers PR #29402). Because Gemma
stores weights near zero, the error is *silent* — it yields a near-zero,
partly sign-flipped dictionary with no exception raised. Verified from
the transformers sources that Olmo3 and Qwen3 use `x * w`, so **every
existing OLMo and Qwen result is unaffected**, and no Gemma dictionary
had been built yet (the A3 work used `lens.apply`/`model.unembed`, which
call the real norm module). Fixed by measuring the gain instead of
assuming a convention — probe the norm with a ones-vector, since
`rms(ones) = 1` implies `norm(ones) = gain` — with a five-check
conformance test including sign-flip detection.

## A3 Gemma — the depth sweep that set the band (2026-07-28) — tier: PILOT

Evidence `a3-gemma-gate-v1`, then `a3-gemma-deepband-logit-v1` (VM7).

The adaptation gate passed on **infrastructure 6/6** — jlens handles the
`Gemma4ForConditionalGeneration` wrapper and the 30.0 logit softcap,
hooks fire on both sliding and full attention layers, and a micro-fit
fits in 73 GB — but failed the readout check: at L24/30 of 60 the known
answer ranks ~52k/69k of 262k. Crucially the **vanilla logit lens fails
equally**, so this is not a Jacobian-lens defect.

The VM7 depth sweep (n=40 probes, median rank of the answer's first token
by layer, reference `unembed` = final norm + tied head + softcap) maps
it: the output basis is opaque across the entire mid-band and then
resolves abruptly.

| layer | 24 | 30 | 36 | 40 | **42** | **44** | 48 | 52 | 59 |
|---|---|---|---|---|---|---|---|---|---|
| median answer rank | 73177 | 57570 | 22684 | 10695 | **352** | **23** | 8 | 1 | 1 |

First layer with median rank ≤100 is **L44 (73% depth)**; ≤10 is L48. The
paper's relative depths (37/50/62% → L22/30/37) all sit inside the opaque
zone. **Reading: Gemma-4 does not write to an output-token-aligned basis
until very late** — a genuine architecture/family difference, not an
instrument failure. Whether the Jacobian transport rescues mid-band
readability where the logit lens cannot is exactly what the queued full
120-prompt fit (band 22/30/37/40/42/44/48/52 — spanning paper depths, the
transition, and the readable zone) will answer; the 2-prompt micro-fit is
not that test. No Gemma-family verdict is stated until it lands.

## G6 power simulation — a design consequence, not a formality (2026-07-28) — tier: PILOT

Evidence `g6-power-sim-v2` (supersedes v1), calibrated entirely from the
pilot parquets (which stay dev-tier), Holm-worst α=0.01, target 90%.

Variance decomposition of the protected paired deltas: within-family
σ 1.2–1.5 nats, ICC ≈ 0.40 — the deltas are a **zero-mode + heavy-tail
mixture** (median ~0, a minority of items losing several nats), which is
precisely the pilot's qualitative finding expressed as a variance
structure. Consequences:

- 0.5-nat **mean-delta** primaries reach only ~24–42% power even at
  n=180 across 60 families; **no affordable n reaches 90%.**
- TOST equivalence at the same margin needs **n ≈ 300**.
- The **tail-RATE** endpoint (per-item >1-nat protected deletion, J vs
  matched random, paired and family-clustered) reaches 90%+ within the
  simulated grid at a 10pp rate SESOI.
- Cross-model per-item delta correlations are lineage-structured
  (Think–Instruct 0.687, base–Think 0.482, OLMo–Qwen ≤ 0.14), so HP1's
  paired-item advantage exists only WITHIN the OLMo lineage.

**This is a decision the preregistration cannot dodge**: either restate
the tail-carried primaries on the rate endpoint (recommended), accept
n≈150–200 families per task, or revise the margins. The draft records all
three options; the choice is the user's at freeze.

## Stage-3 item bank v2 (2026-07-28) — tier: DEV

The v1 candidate pool was too easy (68/113 at ceiling on Think → only
n=41 survived, the frozen dev set). The expansion (`jspace_part2/c3_pool.py`)
is authored **family-first**: 212 candidates across **45 relation
templates**, one template per family, so the analysis clusters on the true
generation unit and no template is mistaken for independent draws — the
pseudo-replication debt the addendum found in the SQL battery. Items
target genuinely known-but-unrehearsed facts (second-order superlatives,
less-canonical exemplars, mid-frequency proper nouns). Scoring on the
anchor model with full-answer-sequence lp (the R4 rule; the v1 dev set
used first-token lp — a recorded change, both emitted) yields the
difficulty distribution and the family readiness count against the prereg
floor (n≥90 across ≥30 families). **The pool is deliberately NOT
partitioned** — dev/confirmatory/replication partitioning is a freeze
action, hashed before any outcome is viewed.

## Instrument added: selected-id logging (2026-07-28, VM7) — tier: PILOT

The protected-dynamic ablator now optionally records, per (layer,
position), which dictionary rows it deflated (with scores) and which the
protection mask blocked. The capture is opt-in and behavior-preserving
(conformance tests assert the ablation is bit-identical with and without
it). The grids over Think's two-hop, one-hop, and hard-one-hop batteries
feed `r7_tail_mechanism`, which decides between three readings of the
protected tail: **H-content** (the deflated rows carry the two-hop bridge
entity — workspace content), **H-output** (the answer sat at clean rank
> 10 and was never protectable), or **H-nothing** (selection is
item-generic). This converts the campaign's most interesting behavioral
observation into a mechanistic claim with a falsifier.

*(sections append here per banked phase)*
