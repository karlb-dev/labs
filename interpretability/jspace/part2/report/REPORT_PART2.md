# J-space Part 2 — living report (ADDENDUM-GOVERNED campaign)

**Status: assay-repair era begun 2026-07-27 evening.** The forensic review
(`jspace_part2_plan1_addendum.md`) was adopted in full: Part 1 is
reclassified as the exploratory campaign (claim corrections in
`REPORT_v2_ERRATA.md` beside REPORT_v2.md), and no cross-model
confirmatory cell runs until Workstream R (R0 provenance → R1
output-protected dynamic ablation → R2 paper occupancy → R3 rank-safe
frozen control family → R4 phase/scoring/paired-stats → R5 paper tasks →
R6 golden tests) passes its gates. **Every row below carries an evidence
tier; everything banked so far is `exploratory`.** Original pre-data
prereg `27c6d3c` (kept as record) is superseded by
`REPAIR_PREREGISTRATION.md`.

## Campaign question (revised)

Does the paper's workspace signature exist in open models when measured
with the paper's own intervention (output-protected dynamic ablation —
missing from Part 1), the paper's own capacity estimand (marginal-gain
occupancy + excess variance — not what Part 1 computed), and controls
matched for energy, effective rank, and geometry? The hypothesis ladder is
H0 instruments-first, H1a/b/c externalization variants, H2 output
alignment, H3 bounded residual, H4 scale (descriptive), H6 task demand,
H7 mean-J mismatch, H8 sparse-frame geometry (see part2 README).

## Matrix scoreboard (grows per banked cell)

| cell | olmo3-think | olmo31-instruct | qwen36-27b | gemma4-31b |
|---|---|---|---|---|
| lens | part-1 (120p) | **transfer FAIL → own fit running** | Neuronpedia 1000p (part 1) | gate pending |
| sanity probes @20 | 17/21 | **17/21 (transferred)** | pass (part 1) | — |
| multihop J pass@1 | 0.283 | **0.217 transferred** (own-lens: pending) | 0.350 | — |
| static grid (energy-matched) | null ≤k40 (part 1) | pending | null k20 (part 1) | — |
| frozen-J | −2.89 nats (part 1) | pending | −2.42 (part 1) | — |
| frozen-logit (B3) | **running** | pending | pending | — |
| occupancy (D) | pending | pending | pending | — |

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

*(sections append here per banked phase)*
