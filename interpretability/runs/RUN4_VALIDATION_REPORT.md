# Run 4 — full-course validation report (continuous engine + expanded datasets)

Date: 2026-06-12 · Machine: Colab A100-80GB · Branch: `lab1_colab`
Tree under test: the continuous-batching push (`d24a8ea`) + the engine
rewrite and fixes from this pass + the dataset expansion (`964bb8d`,
including the new `misconceptions` truth family). Protocol identical to
runs 2 and 3. Dashboard: `runs/course_dashboard_run4.{md,png}`.

## Verdict

**24/24 runs green** (after two real bugs were fixed mid-pass, below), all
26 Tier B self-checks pass, and the generation engine ran every
generation-heavy lab end-to-end with zero lockstep fallbacks.

## The engine, proven in production

| Lab 10 Tier B (36 items × 6 conditions) | wall |
|---|---|
| run 3: lockstep, batch 12 | 2275s |
| old engine as pushed (after the cache-API fix), 16 rows | 3536s |
| lockstep, batch 32 | 1474s |
| **persistent-cache engine, 32 rows (this run)** | **1391s** (the Exp-1 generation block is ~2.5× faster; Exp-2's short-job tail and model load dominate the remainder) |

Micro-benchmarks (`bench_inference.py`, identical greedy tokens vs lockstep
verified): 7B 433 tok/s @32 rows (31.6 GiB); **32B-Think 116 tok/s @16 rows
(64.5 GiB)** — ~300 600-token probes on a 32B in ~26 min, which was the
acceptance bar. Lab 11's CoT audit also ran on the engine (854s).

## Bugs found and fixed in this pass

1. **The pushed engine had never actually run.** transformers 5 made
   `DynamicCache` non-subscriptable; the engine crashed on first contact
   and silently fell back to lockstep. Fixed with a version-tolerant
   accessor — then the real problem surfaced: the design extracted the full
   KV into pairs every step and rebuilt the cache, holding 2× the cache
   through every forward (OOM at 32 rows; *slower than lockstep* at 16).
   Rewritten around one persistent in-place cache; retire/admit do
   event-time surgery. Identical outputs, 2.5× throughput, ~45 GiB freed.
2. **Lab 4 crashed on the new dataset** (both tiers): the expansion added a
   `misconceptions` family but `FAMILY_COLORS` didn't know it —
   plot KeyError. Fixed + made family colors fallback-safe (a new CSV
   family must never crash a plot).

## Science notes

- **Lab 4 + misconceptions:** peak mass-mean accuracy 0.860 (was 0.963 on
  3 families) and worst cross-family transfer 0.56 — the new family is
  doing exactly its job: misconception statements anti-correlate truth
  with surface plausibility, and the probe feels it. Honest hardening, not
  regression.
- **Labs 5/6 (code untouched):** drift only from the expanded data
  (recovery 0.6693→0.6721; minimality 0.043→0.056) — sane.
- **Lab 10 flips** on dataset v2 at full budget: 0.227/0.273/0.5
  (syco/authority/metadata) — within the established 0.14–0.5 across-draw
  range; quote ranges, not points.
- **Lab 11 hint-presence probe flipped positive** on this slice (held-out
  AUC 0.765 vs shuffled 0.48) after two nulls on earlier slices. Three
  measurements, two verdicts: the probe's conclusion is *slice-dependent*
  at n≈12–16 items, which is now the documented finding — the conditional
  claim machinery labeled each run correctly, and the larger item counts
  the engine now makes cheap are the obvious next step.

## Archive

Drive `interpret/run4/`: `lab1..lab11` (both tiers, incl. the engine
timing runs), `code/` (exact tree), this report + dashboard. Engine
benchmark log: `runs/bench_inference_log.jsonl` (committed runs in
`run4/code`).
