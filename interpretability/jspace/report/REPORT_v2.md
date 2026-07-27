# special-lab-1 v2 — Instrument Audit + Cross-Model Verdict

Delta run over v1 (`2026-07-25_1726`, frozen). v1 ended with three verdict-
blocking instrument gaps: controls that were not energy-matched, a live
per-token ablation with no matched-live control, and a lens band that
stopped at 70% depth. v2 closed all three, calibrated the lead claim's
false-positive floor, added CoT-rescue and a second seed, and ran the
decisive cross-model leg: the same clean instruments on Qwen3.6-27B with
the community's published lens. Model: `allenai/Olmo-3-32B-Think` (+
`Qwen/Qwen3.6-27B`); lens engine `anthropics/jacobian-lens @ 581d3986`.

## Verdict

**The disagreement with the paper's causal story was instruments all
along — and the residue that survives clean instruments is real and
transfers.** At energy-matched removal, static J-span ablation is
indistinguishable from matched-random and matched-non-J on *both* models
(every CI ⊃ baseline). Per-item frozen top-10 J-ablation deletes retrieved
factual content on *both* models (−2.9 nats OLMo, −2.4 nats Qwen; random-
dictionary twins on baseline; fluency intact) — the first control-clean
causal handle either replication produced. Meanwhile the capacity anomaly
v1 flagged is a genuine model difference, not harness: measured by the
same code path, Qwen's workspace holds 4.3–6.8% of residual variance with
paper-range active-concept counts, OLMo holds 0.61–0.72% with ~6. The
pre-CoT anticipation null and the 46–49-step workspace-ahead-of-text lead
both survive their dedicated audits (late-band lens, foil floor).
And the deletion is bypassable when the model can reason out loud:
think-mode recovers the frozen-deleted two-hop answer in 0.80 of traces
(vs 0.23 silent; control 0.93) and fully rescues one-hop — the paper's
rescue prediction holds in content-channel form. All of it replicates
under a second seed on fresh two-hop items: frozen-J accuracy 0.233 under
both seeds on disjoint item sets (Δlp −2.82 vs −2.89), controls and
statics on baseline.

## v1 claims: survived / flipped / sharpened

| v1 claim | v2 instrument | outcome |
|---|---|---|
| non-J PCs cause real damage where J-span doesn't (apparent selectivity) | P1 energy match | **FLIPPED** — 100% energy artifact; at matched energy all static conditions = baseline (SQL 0.667 = base at all doses; twohop_lp deltas −0.01/−0.17/−0.15, CIs ⊃ 0) |
| static J-span ablation does nothing | P1 | survived (now clean: matched-rand and matched-nonJ also nothing) |
| live dyn-10 lobotomy is uninterpretable live-computation deletion | P2 matched-live control | **SHARPENED** — live_rand10 = baseline, so the lobotomy is J-specific (pool-size caveat recorded) |
| no content-specific causal effect found | P2 frozen selection | **FLIPPED** — frozen_j10 deletes the retrieved fact: twohop_lp −1.74→−4.63 (Δ−2.9, CI [−5.5,−3.8]), recall 0.58→0.23, single- AND multi-hop; frozen_rand10 = baseline everywhere |
| capacity ~10× thinner than paper (var share ≤0.67%) | P3 late lens + Q same-harness Qwen | **SHARPENED** — not late-shifted (flat 0.61–0.72% through L58; L62 1.6% is unembed-adjacent, lowest persistence) and not harness (Qwen: 4.3–6.8%, act@0.01 32–42) |
| pre-CoT anticipation null | P3 late lens | survived (think-mode late-band median rank 3916–7291; 0% ≤20) |
| suppressed-CoT answer-time loading | P3 | sharpened — best-late-layer median rank 3 (76% ≤20); logit lens equal late → J-advantage is mid-band only |
| workspace leads CoT by 46 steps | P4 foils + P3 late lens | **SHARPENED** — foil floor 0.06 (freq-matched) vs 0.92 answers; family foils fire concurrent with text (lead 3); late-band replicate: 49.5 steps, 86% ws-first, det 99%, n=88 |
| suppression asymmetry predicts CoT rescue | P5 rescue | **CONFIRMED** — frozen-J think-mode any-rate 0.80 twohop / 1.00 onehop (vs 0.23 silent recall; frozen-rand 0.93); frozen-J halves the `</think>`-closure rate (0.20 vs 0.40) |
| (single seed) | P6 seed-1 + fresh items | **REPLICATED** — fresh probe-swap items [60:90], redrawn pools: frozen-J 0.233 acc (= seed 0 exactly), Δlp −2.82 (seed 0: −2.89); frozen-rand Δ−0.11; statics all CIs ⊃ baseline |
| null is OLMo-specific vs harness? (open) | Q Qwen leg | **RESOLVED** — static null transfers to Qwen; frozen deletion transfers to Qwen; capacity does NOT transfer (real model difference) |

## Headline numbers (v2)

| quantity | OLMo-3-32B-Think | Qwen3.6-27B (their lens, our harness) | paper (Claude) |
|---|---|---|---|
| multihop readout, J-lens vs logit pass@1 | 0.283 vs 0.200 (n=60) | 0.350 vs 0.317 (n=60) | (J-lens ahead) |
| peak variance share | 0.67% (0.61–0.72 flat; L62 1.6%) | **6.8%** (4.3→6.8 rising) | 6–10% |
| active concepts @0.01 / @0.02 | ~6 / ~3 | 32–42 / 14–21 | 10–25 (k≈25) |
| top-1 persistence | 0.19–0.20 mid-band | 0.03–0.09 | — |
| static J-span k20 (energy-matched), Δ twohop_lp | −0.17 [CI ⊃ 0] | −0.08 [CI ⊃ 0] | paper: multi-step collapses |
| matched-rand / matched-nonJ k20 | −0.01 / −0.15 [CIs ⊃ 0] | +0.05 / −0.03 [CIs ⊃ 0] | (controls) |
| frozen per-item J top-10, Δ twohop_lp | **−2.89** (recall 0.58→0.23) | **−2.42** (0.87→0.37) | (no direct analog) |
| frozen random twin | −0.04 | −0.09 | — |
| frozen 1-hop vs 2-hop | collapses equally (0.23 / 0.23) | **asymmetric** (0.83 vs 0.37) | shallow survives |
| CoT lead (foil-calibrated) | 46 steps mid-band / 49.5 late-band | not run (gated) | (workspace precedes text) |
| pre-CoT anticipation | null (all bands) | not run | — |

## Phase detail

### P1 — Energy-matched controls (s11 → f10, f11)

v1's rank-matched controls removed ~10× mismatched raw-h energy in
opposite directions. v2 matched removed energy to ratios 0.97–1.01 per
layer/dose (prefix or windowed rank selection, method recorded per layer).
Result: the entire static grid collapses onto baseline — v1's "non-J does
real damage" was pure energy; v1's "J-span does nothing" was conservative
(J carries 2–5× more energy per rank than random, so v1 under-dosed the
controls, not the treatment). The causal null is now clean: **no static
subspace of the J-space, nor any energy-matched control, moves any battery
task at k ≤ 40 dims/layer.**

### P2 — Frozen per-item selection (s12 → f12)

The confound-free variant of the paper's live protocol: rank dictionary
directions once per item over the clean prompt, freeze top-10 per band
layer, QR, generate under the static per-item projector. frozen_j10
deletes the retrieved fact (−2.9 nats; recall 0.58→0.23) with the
identical mechanism on a random dictionary at baseline and generations
staying coherent (verbatim samples audited). Not the paper's dissociation:
1-hop collapses identically to 2-hop → a **factual-recall content
channel**, not a demonstrated reasoning scratchpad. live_rand10 = baseline
establishes v1's dyn-10 lobotomy as J-specific (pool sizes differ; caveat
recorded).

### P3 — Late-band lens (s14, s15 → f13, f14)

Dedicated lens on L{46,50,54,58,62} (same recipe/corpus/seeds; 45 s/prompt).
No late-shifted workspace: variance share flat 0.61–0.72% through L58
(L62 1.6% = unembed-adjacent, fastest-turnover), persistence declines
monotonically. Pre-CoT null survives the late lens; suppressed-CoT
answer-rank bottoms at median 3 (76% ≤20); the logit lens reads the late
band equally well — the J-over-logit advantage is a mid-band phenomenon.
The 46-step lead replicates at 49.5 steps on this independent lens over
disjoint layers (86.4% ws-first, det 98.9%, n=88).

### P4 — Foil calibration of the lead (s13 → f9)

Identical detector on (a) family foils, (b) frequency-decile-matched
corpus words: answers det 0.92 / med lead 46; freq-matched floor det 0.06;
family foils det 0.32 but lead ≈3 (concurrent with text = legitimate
candidate enumeration, not hidden foreknowledge). Answer beats all its
foils in 66% of items; median detection rank 1.

### P5 — CoT rescue (s16, capped grid per PLAN_v3 → f16)

Same per-item frozen projectors as P2, open `<think>`, ≤400 reasoning
tokens; frozen_j10 vs frozen_rand10 on twohop(30)/onehop(15)/arith(15).
Scoring decision fixed before reading results: OLMo rarely closes
`</think>` in 400 tokens on multi-step tasks (closure 0.20/0.40 twohop,
0.07 arith, 0.87–1.00 onehop), so answer-anywhere-in-trace and
answer-in-think are the informative metrics. Result: **externalization
largely bypasses the deletion** — twohop any-rate 0.80 under frozen-J
(3.4× the 0.23 silent-recall rate; control 0.93), onehop fully rescued
(1.00), arithmetic untouched (0.93 = both conditions' scorer floor at the
token cap). Residual gap to control = the bypass is incomplete at 400
tokens; re-retrieval vs re-derivation-via-the-other-hop is not
distinguished. Secondary observable: frozen-J halves the closure rate
(0.20 vs 0.40) — damaged recall lengthens deliberation, as expected if
the CoT is doing compensatory work. VM4's 4 banked 4-condition items
(incl. `none`/`jspace_k20` think cells) retained in the JSON.

### P6 — Seed-1 robustness on fresh items (s17, trimmed per PLAN_v3 → f17)

The deliberately-harder replicate: fresh two-hop items (probe-swap
[60:90], untouched by any fitting or prior eval), seed-1 random
dictionary and matched-random pools, per-item frozen selection re-run
from scratch; twohop + twohop_lp readouts across all six causal
conditions. Fresh items are harder (baseline lp −1.74→−2.81; acc
0.58→0.50), so effect sizes carry the comparison: frozen-J accuracy
**0.233 under both seeds** on disjoint item sets, Δlp −2.82 vs −2.89;
frozen-rand Δ−0.11 (baseline); statics jspace_k20/vmatch_rand/vmatch_nonJ
at Δ−0.39/+0.13/−0.24 with every CI overlapping baseline's. The frozen
handle survives item-set, control-pool, and seed replacement; the static
null generalizes to never-seen items.

### Q — Qwen3.6-27B under our instruments (s18 → f15)

Their model, their published lens (neuronpedia/jacobian-lens, 1000-prompt
WikiText, all-layer), our harness end-to-end; band = same depth fractions
(= same indices L20–44 on 64 layers); no-think completion prompts.
Sanity gate passed (readout advantage replicates). Descriptive: variance
share 4.3–6.8% rising through the band, act@0.01 32–42, act@0.02 14–21 —
paper-range capacity measured by the same code that read OLMo at 0.67%/6/3.
Causal: frozen_j10 −2.42 nats, 0.87→0.37 twohop, controls clean; static
energy-matched trio all within ±0.08 nats of baseline (and insensitive to
a broken-selection pass: −0.85/−0.77/−0.87). One genuine asymmetry: Qwen's
1-hop barely moves (0.90→0.83) where OLMo's collapsed equally — the
composed task is hit harder on Qwen, the closest thing to the paper's
dissociation signature either model produced (1-hop ceiling + relative-
dose caveats recorded; sharpest follow-up target).

Qwen run notes: 248k-vocab dictionaries forced an fp16 staging path for
build_dicts (fp32 recipe peaks ~99 GB > card); Qwen's outlier-heavy
activations forced two pursuit fixes (bf16 correlations; 1/(k+2) refit
step — fixed lr=0.25 diverges when near-duplicate atoms push the active-set
Gram past the stability bound). Stable layers moved ≤0.2pp under the fix;
all OLMo results predate and are unaffected.

## Synthesis

Static span = nothing (both models, matched energy). Frozen per-item =
fact deletion (both models, controls clean). Live per-token = computation
deletion (OLMo; J-specific under matched-live control). Late band = no
hidden workspace (OLMo). Lead = foil-calibrated, two independent lenses.
Capacity = real 10× cross-model difference under one harness.
Rescue = the deleted content channel is largely bypassable by
externalized reasoning (0.23→0.80), i.e. the J-space holds *retrieved
content*, and the CoT can regenerate what it held — the cleanest
functional statement of the workspace's role this replication supports.

## Limitations (v2-specific; v1's ten carry over)

1. Qwen 1-hop items are near-ceiling capital probes — the OLMo/Qwen
   asymmetry needs harder single-hop items before it upgrades to a
   dissociation claim.
2. Frozen-rand control uses a 5120-row dictionary vs frozen-J's
   vocab-sized pool (OLMo 100k, Qwen 248k) — alignment-depth caveat from
   v1 still applies to both models.
3. Qwen leg is single-seed, n=30/15/20 per cell, one lens (theirs); no
   Qwen CoT-lead or rescue (gated by the 3 h window).
4. Pursuit hyperparameters differ across models by necessity (bf16 + stable
   step on Qwen); the OLMo-vs-Qwen capacity comparison uses the same final
   code path, but OLMo's banked numbers ran the earlier (finite-on-OLMo)
   path.
5. SQL remains the flaky 3-schema cell; dropped from v2 P5/P6 grids.
6. The static-null grid tops out at k=40 dims/layer (OLMo) / k=20 (Qwen);
   "no static effect at any dose" is bounded by those doses.

## What 3 more GPU-hours would have bought

- Q3: Qwen think-mode cot-lead spot check with the foil floor (n=15) —
  skipped under PLAN_v3's drop order; the cross-model lead comparison is
  the most interesting absent number.
- Qwen CoT-rescue mirror of P5, and a Qwen seed-1 replicate.
- Harder 1-hop items on both models to settle the asymmetry (the sharpest
  single experiment left).
- n→60 on arithmetic/SQL; Dolma-corpus lens (v1 §7.5); implied-cue
  eval-awareness variants (v1 §7.8); 'famously' divergence forensics
  (v1 §7.6).

## Run notes (VM5, for reproducibility)

Single 3 h window, T+0 22:47 UTC. Phase Q ran first out of disk necessity
(PLAN_v3 priority rule): Colab's DriveFS stages streamed reads in a local
SQLite chunks.db, so Qwen-local (55 GB) + OLMo-from-Drive (64 GB stream)
cannot coexist on the 113 GB disk. File-level cache eviction is not
possible (deleting chunks.db wedges the mount into serving empty content;
recovery = kill the `--single_process` drive worker and let the supervisor
respawn it — reads recover, server content unaffected). Final layout:
metrics mirrored locally each boundary; logs tee locally and rsync at
boundaries; watchdog logs (never deletes). s18 pass economics: Qwen loads
in 7 s from local NVMe vs 750 s for OLMo through DriveFS.

## Read next

`report/summary_v2.json` (regenerated by s19 from metrics),
`report/handout/olmo32b_jspace_handout.pdf` (the primary reading format,
§"v2 delta run"), `labs/lab37_jspace_workspace.md` (claim ledger
SL1-C1..C7 + promotion checklist), v1 `report/REPORT.md` (baseline run).
