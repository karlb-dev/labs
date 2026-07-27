# J-space Part 2 — living report (model-matrix campaign)

**Status: OLMo set 1 in progress.** Preregistration committed 2026-07-27
(`27c6d3c`) before any data. Plan: `PLAN_PART2.md` §8 order with model-set
disk grouping. Part-1 baseline: `2026-07-26_v2/report/REPORT_v2.md` (static
causal null on both models at matched energy; frozen per-item J-ablation
−2.9/−2.4 nats cross-model; capacity 10× thinner on OLMo; foil-calibrated
46-step CoT lead).

## Campaign question

Why is the paper's static causal dissociation missing on open models — H1
externalization, H2 output-occupancy, H3 training-lab specifics, H5 residual
instruments — or is it findable after all (Instruct sibling, load-demanding
tasks, wider/persistence-selected doses)?

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

## A0 — does the J-dictionary survive post-training? (2026-07-27)

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
`metrics/olmo31-instruct/a0_transfer_gate.json`. Ledger: SL2-C1 drafted at
next reporting boundary.

## D — output-occupancy index, phase 1: Think saved traces (2026-07-27)

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

## B3 — frozen-logit control (queued behind the Instruct fit)

Same per-item frozen mechanism as the part-1 marquee instrument, dictionary
= plain (W_U⊙g) rows (no Jacobian). Pool sizes inherently matched
(vocab-sized both). Decides whether the −2.9-nat deletion needed the
Jacobian pullback or just readable output-aligned directions.

*(sections append here per banked phase)*
