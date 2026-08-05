# A2 — Phase 2/3 causal-channel core (recheck from raw parquets)

Inputs: `tables/recon_phase2.csv` (byte-identical rerun of the frozen
confirmatory pipeline), `tables/recon_phase3.csv` (39/39 byte-identical
from item rows), `tables/recon_paper_draft.csv`, and the registered
span-audit/prose/mediation summaries. Descriptive juxtaposition of
registered estimates only; no new inferential object.

## The core, restated with verified numbers

On Qwen3.6-27B, removing the selected J span — output-protected, and in
Phase 3 span-safe by construction — produces a **minority-tail** effect
that an exact per-site rank-and-energy-matched control does not produce:

| Estimand | Confirmatory | Held-out replication |
|---|---|---|
| P-HP3 protected tail-rate excess (Phase 2) | +0.2788 [0.2048, 0.3608], Holm p=5e-4 | +0.2966 [0.2071, 0.3824] |
| P3-P2 span-safe tail excess at −1 nat (Phase 3) | +0.095833 (188 items / 26 fams), p=1/100001 | +0.102083 (190/28), p=1/100001 |

Both rows reproduced bit-for-bit from raw item parquets this phase.

## SQ2 answers

**1. Tail, not shifted bulk.** The span-safe J arm's tail rates are
0.13–0.32 across models while every matched-control arm sits at
0.00–0.02 (registered cross-model span audit); mean shifts are small
relative to tail mass. The claim object is "a minority of items lose
>1 nat", not "all items degrade".

**2. Tail membership is stable where measured.** Across partitions the
*rate* replicates (above). Across lenses, the independent-lens clause
gives per-item damage correlation r = 0.988/0.990 and tail-Jaccard
0.878/0.883 (OLMo pair, replication partition,
`n6-repl-lens-independence-v2`). Phase 4's Q-L4 caveat applies to newly
fitted sparse Qwen lenses, not to these frozen instruments.

**3. Ruled-out control families** (each ≈ 0 on the same items):
exact instant rank+energy matched (primary); overlap-matched;
protected-energy-matched; persistent-matched; span-safe own control;
mechanics random. Output-stream deletion is excluded by construction
(label protection + span-safety); the label-vs-span decomposition shows
the Phase 2 label-protected effect was ~3× the span-safe effect on the
OLMo pair (r = 0.195/0.292 — damage *reallocates*) but mostly content on
Qwen (r = 0.745).

**4. Still-open alternatives:** protection-geometry scope (one
protection convention tested); the named secondary arms
(`dynJ_label_shuffled`, `dynJ_rotated`, `dyn_spectrum_matched_nonJ`)
remain robustness options, not run as primaries; external generalization
beyond the frozen banks; the thin two-hop replication leg (P-HP1's
replication ran on the disclosed fallback population and did not
reject — the interaction is confirmatory-unreplicated).

**5. Bridge content explains more than dose — but is not yet separated
from answer-direction content.** True-bridge protection beats the frozen
chosen distractor by +0.431367 (p=0.009180; family bootstrap CI from the
geometry-v2 event [0.132018, 0.763437]); measured rank/energy/geometry
covariates do not absorb it (+0.403816 after residualization,
p=0.01854; cross-fit R² = −0.0947). The development factorial adds:
bridge-only lesion −0.887; counterfactual substitution −4.053
(catastrophically worse than deletion — content is read, not just
capacity); unrelated-content lesion −0.037. But counterfactual-bridge
minus counterfactual-answer-direction is +1.342 [−1.593, +4.482],
p=0.419: the abstract-bridge-vs-answer-direction split is unresolved
(development tier), and P3-P3 has no held-out replication by protocol.

**6. Prose preservation fails everywhere; the cost is content, not
dose.** Prose damage exceeds task damage in standardized units on all
three models (std effects −2.11/−2.14/−1.51 vs −0.55/−0.64/−0.26); the
exact control's prose cost is nil (+0.0018–+0.0206/token vs label-arm
+0.1754–+0.9452). Span-safety removes 49.3%/71.6%/77.8% of the label
prose cost (Think/Instruct/Qwen) — **the draft's "72–78%" must be
rescoped**. No model earns "selective" wording; the licensed noun
remains *knowledge-access channel*.

## Wording boundaries carried into P8

- P3-P1 is descriptive, negative, seed-sensitive: −0.271183 (exact
  p=0.057892) at the sha256-v1 realization; the randomization CI
  [−0.535135, +0.014565] and percentile-t CI [−0.537109, +0.015201]
  cross zero; the normal-approximation interval [−0.515271, −0.006637]
  may not be called a bootstrap/wild-cluster interval, and **no
  "near-miss" wording is permitted** (release-audit correction #1).
- The P3-P3 headline mixes sources: estimate+p from
  `p3-inference-audit-v1`, CI from the geometry-v2 100000-draw
  bootstrap; the audit's own 20000-draw CI is [0.127795, 0.762762].
  Papers must cite the pair they use consistently (reconstruction
  reproduced both).
- Phase 2's replication P-HP1 population is the disclosed fallback (all
  partition items, 322 rows / 32 families), not the confirmatory
  intersection cohort (172 / 24): never describe the two as
  same-population.
