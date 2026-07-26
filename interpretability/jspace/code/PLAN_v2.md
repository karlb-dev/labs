# special-lab-1 v2 — Resolve the Open Instruments (delta run)

v1 (`2026-07-25_1726`) is complete; REPORT.md §7 lists the open items. v2
resolves the ones that block the verdict, in priority order, reusing v1's
lens, layer_state, caches, and any grid cells whose protocol is unchanged.
v2 outputs → `2026-07-26_v2/` (`RUN_DIR_V2` in sl1_common); v1 read-only.
Same guardrails: no commits, incremental Drive checkpoints, resumable
scripts, seeds recorded, bootstrap CIs, hard honesty.

## The framing question (user): is the causal null OLMo or our harness?

Published Qwen 3.6 27B results (Nanda) are *narrower* than "Qwen has a
J-space": readout/descriptive evals replicated; poetry/arithmetic failed;
causal interventions flagged by the author as possibly confounded. Our v1
matches the descriptive/readout pattern and nulls only on causal — with
three known instrument gaps, now P1–P3. Decision tree:

1. P1–P3 flip the causal result → harness, verdict revised, done.
2. P1–P3 confirm the null cleanly → run OUR instruments on Qwen 3.6 27B
   (phase Q below). Same harness, two models:
   - Qwen shows dissociation, OLMo doesn't → real model difference
     (reportable; OLMo's full openness makes the contrast studyable).
   - Qwen also nulls under our instruments → the published causal story
     doesn't survive clean instruments on Qwen either; the disagreement
     was instruments all along.

## Pins from re-fetching the paper (2026-07-26)

- Active concepts: **no absolute threshold exists** — sparse nonnegative
  gradient pursuit, k=25 "typical". Our θ-sweep + pursuit-k is the right
  sensitivity treatment; v1/paper counts are convention-adjacent, not
  identical. (P6c resolved: nothing further to adopt.)
- Ablation protocol: paper selects the top-10 most-active directions **live
  per token position** — v1's `jspace_dyn10` was protocol-faithful. On
  Claude this spares fluency; on OLMo it lobotomizes (NLL 2.71→24.34).
  "Same protocol, opposite side effect" is itself a model-difference datum.
  P2 (frozen prompt-selection) is the *confound-free variant*, not the
  paper's protocol; both get reported.
- Paper controls are **matched-norm**; v1's dyn condition had no matched
  control at all → P2 adds `live_rand10`.
- Paper band: ~38–92% of depth (≈L24–L59 here). v1 fitted 33–70%; P3 fits
  the missing back half {46,50,54,58,62}.
- Variance share: pursuit-reconstruction share of activation variance —
  matches v1's s5 definition (commensurable as-is).

## Priorities → scripts

| P | Script | Design | Status source of truth |
|---|---|---|---|
| 1 | `s11_energy_match.py` | Streaming raw-`h` energy pass over the s5 prompt set at band layers → per-layer E_J(k), per-direction energies of the non-J PC pool (≤64, v1 construction) and a 512-dim random pool. Matched rank m per layer per k by prefix cumsum (windowed fallback if prefix ratio ∉[0.8,1.25]; method recorded). Verification per layer in `metrics/energy_match.json`. Then s7 battery on conditions `vmatch_rand_k{10,20,40}`, `vmatch_nonJ_k{10,20,40}`; `none`/`jspace_k*` **reused from v1** (identical protocol+seed; one integrity cell re-run and compared). → `metrics/ablation_v2.json` | v2 metrics/ |
| 2 | `s12_frozen_ablation.py` | Per item: clean prompt pass → per band layer, rank dictionary dirs by summed \|corr\| over prompt positions (skip first 4), freeze top-10, QR → static per-item projector; generate. Conditions: `frozen_j10`, `frozen_rand10` (same mechanism, 5120-row random dictionary, seed 0), `live_rand10` (dyn mechanism on the random dictionary — the matched-live control v1 lacked). `live_j10` = v1 dyn10 (reused). Same battery. → `metrics/frozen_ablation.json` | v2 metrics/ |
| 3 | `s14_lateband_fit.py` + `s15_lateband_readouts.py` | Fit {46,50,54,58,62}, 120 WikiText prompts, s3 settings, chunked+merged, → `lens/olmo32bthink_late.pt`. Then: s5-style descriptive on late layers (variance share keep rising or peak?); s8 answer-time loading + cot-lead re-read with late dicts; late-band column added to P1/P2 grids (band2 = {46,50,54,58,62}∩ablatable). | v2 lens/ + metrics/ |
| 4 | `s13_cot_foils.py` (CPU) | For each traced v1 item: identical top-8 detector run for (a) family foils (other items' answers / other join columns / nearby numbers), (b) 5 frequency-decile-matched content words from the fitting corpus (excluded: words in item text). Report foil detection rate, foil median "lead", answer-vs-foil separation (detection-rate gap + lead distributions). → `metrics/cot_foils.json` | v2 metrics/ |
| 5 | `s16_cot_rescue.py` | Winning instrument(s) from P1/P2 under think-mode prompting (≤400 think tokens) on twohop/arith/sql, vs v1 no-think ablation. Prediction from v1 suppression asymmetry: externalization rescues. | v2 metrics/ |
| 6 | `s17_robustness.py` | Seed=1 replicate of P1+P2 grids; SQL/arith n 30→60 (v1 generators, extended range); both seeds side by side. | v2 metrics/ |
| Q | `s18_qwen_instruments.py` (design gated) | Qwen 3.6 27B leg: prefer Neuronpedia's published WikiText lens for Qwen (their lens + our instruments isolates harness); weights (~54GB bf16) do NOT fit Drive cache (54GB free) → stream/download to local disk (56GB free, tight — clear HF locks first) or accept 8-bit tier (record). Run: sanity multihop, s5 descriptive, P1/P2 causal grids. Only after P1–P5 land and if GPU window allows. | v2 metrics/ |

Optional if GPU remains: Dolma lens (v1 §7.5), confound-broken
eval-awareness (§7.8), `famously` forensics (§7.6).

## Deliverables

`report/REPORT_v2.md` (revised verdict slug; survived/flipped/sharpened
table; v1-vs-v2 causal grids side by side), `report/summary_v2.json`,
claim ledger `[SL1-C1..C5]` + lab-header block in lab36 template style,
LOG.md updated throughout, figures regenerable from metrics.

## Handout-first reporting (user directive, 2026-07-26)

The LaTeX handout is the PRIMARY reading format — tex prose + figures over
raw JSON. **After every result-bearing phase, run `bash refresh_handout.sh`**:
it regenerates all v2 figures from whatever metrics exist
(`s19_figures_v2.py`, guarded per metric), recompiles the living handout
(`handout/olmo32b_jspace_handout.tex` — §"v2 delta run" grows a subsection
per phase, with analysis prose, not just numbers), ships tex+pdf+figures to
`2026-07-26_v2/report/handout/`, and mirrors code. v1's archived handout in
`2026-07-25_1726/report/handout/` is frozen — never overwrite it. Figure
conventions: entity->hue fixed across all figures (J=blue #2a78d6,
random=orange #eb6834, non-J=aqua #1baf7a, baseline=ink); v2 figures are
f9+ in the v2 figures/ dir.

## Budget notes (measured v1 rates)

Battery block rates: readout tasks seconds; generation tasks ~100–160s per
30-item cell. P1 ≈ 10 min energy pass + ~70 min grid. P2 ≈ 45 min.
P3 fit ≈ 1.5–2.5 h + ~1.5 h re-reads. P4 CPU-only (run alongside).
P5 ≈ 2 h. P6 ≈ 2 h. Order fixed; if the VM dies, the tables in
inprogress.md say exactly where.
