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

*(sections append here per banked phase)*
