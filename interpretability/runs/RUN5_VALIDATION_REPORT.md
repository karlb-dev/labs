# Run 5 — full-course validation report (static-cache engine + run-4 follow-ups)

Date: 2026-06-12/13 · Machine: Colab A100-80GB · Branch: `lab1_colab_followup`
(squash-merged from `lab1_colab`). Tree under test: the static windowed KV
cache + admit batching, the two engine memory fixes found this pass
(reference-cycle leak, allocator fragmentation), the depth-aware
component-anatomy gate, and the lab follow-ups — Lab 7 on the engine, Lab 10
strong add-mistake + error bars, Lab 11 third domain (sentiment_negation) +
probe SEs, 32B Tier C. Dashboard: `runs/course_dashboard_run5.{md,png}`.

## Verdict

**Green across the board, with four real bugs found and fixed mid-pass**
(all engine/scale robustness, none in lab science). Tier A 13/13; Tier B
labs 1–9 + full-140 Lab 10 + all three Lab 11 domains; Tier C 32B
spot-checks for labs 1/2/3 and the Lab 11 factual audit. One run deferred
on purpose: Lab 6 at 32B (circuit discovery) is compute-bound in eager
attention and was cut after it had passed every self-check and both
screening phases — a cost profile, not a correctness gap (below).

The two headline science wins the user asked for both landed:

- **The Lab 11 hint-presence probe wobble is resolved.** Run 4 saw it go
  null/null/positive across three n≈12–16 slices. At the full 35-item fresh
  slice (105 probe jobs) it is cleanly positive: **held-out AUC 0.725 ±
  0.069** (Hanley–McNeil) **vs shuffled 0.498**, selectivity 0.227 — ~3 SE
  above chance. The earlier flicker was small-n noise; the larger slice the
  engine now makes cheap settles it as a real, decodable answer-emission
  signal.
- **Lab 10 flip rates are now condition-separated with tight error bars.**
  The run-4 "0.14–0.5 across-draw range" was condition mixing. At the full
  140 items (n=104 baseline-correct): sycophancy 0.135 ± 0.033, authority
  0.269 ± 0.043, metadata 0.356 ± 0.047. Three distinct, well-separated
  rates — the course's weakest numbers are now among its most defensible.

## The engine, re-proven (static windowed KV cache)

The run-4 engine kept one persistent DynamicCache, which still
reallocate-and-copied the whole cache every decode step (~2.4 GB/step at
32B × 16 rows). Run 5's engine replaces it with preallocated per-layer
buffers and a sliding column window: per-step writes are one in-place
`copy_` per layer, left-pad trims are free, retire/admit stay event-rate,
admits are batched (`admit_block = max_concurrent // 4`), and capacity grows
in ~512-column increments.

`bench_inference.py`, A100-80GB, greedy bf16, identical heavy-tailed
workload as run 4:

| config | run 4 (persistent DynamicCache) | run 5 (static cache) |
|---|---|---|
| 7B, 48j @ 16 rows | 248.7 tok/s · 22.6 GiB · p95 ITL 50.2 ms | 249.5 tok/s · 22.7 GiB · p95 **47.7** ms |
| 7B, 96j @ 32 rows | 433.2 tok/s · 31.6 GiB · p95 60.7 ms | **489.7** tok/s · 31.7 GiB · p95 **46.4** ms |
| 7B, 96j @ 48 rows | 451.4 tok/s · 36.2 GiB · p95 80.2 ms | **648.8** tok/s · 40.7 GiB · p95 **47.2** ms |
| 32B, 48j @ 16 rows | 116.2 tok/s · 64.5 GiB · p95 116 ms | 117.4 tok/s · 64.6 GiB · p95 110 ms |
| 32B, 72j @ 24 rows | — | 81.1 tok/s · 66.8 GiB · p95 140 ms |

The headline is the p95 column: per-step latency stays flat ~47 ms as 7B
rows triple, where the old engine paid 50→80 ms — nothing per-step copies
the cache and prefill stalls are batched. Lockstep reference on the 7B
workload: 182.8 tok/s, 31.4 GiB.

**Operational note for 31B runs:** on 80 GB, 32B throughput peaks near 16
rows (117 tok/s); 24 rows regresses to 81 tok/s — longer contexts push
per-step latency (110→140 ms p95) and prefill-stall/TTFT (p95 144→174 s)
without an amortization gain. 16 rows is the sweet spot at this scale.

### The determinism contract, made honest

Run 4 verified "identical greedy tokens vs lockstep" on a 4-job sample.
Run 5's strided buffer views select different attention kernels than
contiguous tensors, surfacing what was always true: bf16 greedy decoding is
only bitwise stable per kernel path. The check is now a two-part contract
the harness enforces:

1. **Self-determinism (bitwise):** two identical engine calls must produce
   identical tokens for every job. **48/48.**
2. **Cross-engine near-tie verification:** every token where continuous and
   lockstep disagree is teacher-forced through a clean single-row forward;
   both engines' choices must sit within 0.25 logits (≈2 bf16 ulps) of the
   top-1. **45 divergences, all verified benign, 0 real** (each an n-way
   near-tie 170–300 tokens deep; a real cache bug diverges at a large gap).

Per-job steering (new, for Lab 7) was verified fp32-exact against
`generate_text`: 4/4 jobs bitwise identical on gpt2, including a 0-dose row
sharing the schedule with steered rows.

## Bugs found and fixed in this pass

1. **Engine reference-cycle memory leak (the real find).** `StaticKVLayer`
   held a back-reference to its owning cache, closing a cycle
   (`cache.layers → layer → cache`). Cyclic garbage is only reclaimed by the
   generational GC, so multi-GiB CUDA buffers outlived the engine call: a
   1-job call leaked ~27 GiB on the gpt2 probe (1230 MB retained vs 10 MB
   after the fix). In production this made Lab 10 Tier B silently fall back
   to lockstep and OOM'd Lab 11-CoT outright. Fixed: layers share a plain
   window dict (no cycle), the engine explicitly `release()`s buffers on
   exit, and buffers size to `min(max_concurrent, n_jobs)` (no ghost rows
   for small calls).
2. **Long-prompt prefill OOM in the rescue path (initially misdiagnosed).**
   After the leak fix, Lab 10 *still* fell back at the unparseable-rescue
   step. The first hypothesis — allocator fragmentation — was wrong, and a
   `torch.cuda.empty_cache()` in the engine's finally (committed, kept as a
   harmless general improvement) did not help: the OOM signature was 78 GiB
   of *live* allocation with only 63 MiB reserved-but-unallocated, so there
   was nothing cached to reclaim. The true cause: the rescue re-feeds every
   unparseable job its full prompt **plus up to 2048 generated think
   tokens**, so a `max_concurrent=32` admit prefilled ~70k tokens in one
   batch and that peak blew past 80 GiB on the 7B. Fixed with a
   `max_prefill_tokens` budget (default 16384): an admit fills greedily up
   to the free slots but stops before a prefill whose width
   (rows × padded length) exceeds the budget, so long-prompt batches admit
   in narrower waves and decode normally; ≥1 row is always admitted. Short
   prompts and the throughput benchmarks never approach the budget and are
   unchanged. Verified on gpt2: output bitwise-identical with/without the
   budget (admit timing only), and a tight budget splits a long-prompt batch
   from 2 wide admits into 12 narrow waves. Confirmed at 7B scale: the full
   140-item Lab 10 reran end-to-end with zero engine fallbacks (1.68M tokens
   on-engine), where the two prior attempts both fell back at the rescue.
   This is the general robustness guard for large-model / long-context runs.
3. **Component-anatomy gate too tight at depth.** The decomposition
   reconstruction gate (0.02, calibrated on 32-layer models) is a *max over
   n_layers* blocks; at 64 layers the correct hook pair lands at 0.021–0.024
   (a wrong pair fails by 10×). Gate now widens as √(n_layers/32), both
   values recorded in the diagnostic. Unblocked labs 2/3/6 at Tier C.

The fallback itself behaved correctly throughout: every time the engine hit
an allocation it could not satisfy, it degraded to lockstep `model.generate`
and the lab still finished green with valid numbers. Robustness here means
*both* that the engine now handles these cases natively and that its
failure mode was always graceful.

## Labs: what run 5 measured

- **Lab 10 (full 140 items, Tier B Olmo-3-7B-Think).** Baseline acc 0.743 ±
  0.037. Flip rates above. Strong add-mistake variant: even the
  self-correction phrasing leaves Olmo near-immune (0.042 ± 0.041 follow vs
  0.083 ± 0.056 for the bare assertion) — the run-4 immunity finding holds
  under a much harder injection, now with a worked SE. Necessity curve
  0.417 → 0.708 over the truncation grid, filler stuck at the k0 floor
  (0.417 ± 0.101): the visible CoT carries behavioral load. Engine carried
  the 840-generation main block at 600 tok/s; numbers reproduce the
  full-lockstep run within SE (authority 0.263 vs 0.269, metadata 0.343 vs
  0.356, sycophancy 0.111 vs 0.135). After the prefill-budget fix the rerun
  completed **fully on the engine with zero fallbacks** — 14 calls, 1142
  jobs, 1.68M tokens at 574 tok/s, including the previously-OOMing rescue and
  experiment 2.
- **Lab 11 cot_faithfulness (full 35-item fresh slice, Olmo-3-7B-Think).**
  Probe resolved (above): held-out AUC 0.725 ± 0.069 vs shuffled 0.498 at
  depth 22. Baseline acc 0.714, max flip 0.36.
- **Lab 11 sentiment_negation (new domain, Tier B Olmo-3-7B).** Behavioral
  negation gap is real: plain acc 1.0 (margin 3.67) → negated acc 0.688
  (margin 1.82), a 0.312 drop. The plain-trained valence probe transfers
  *weakly* to the negated family at 7B (AUC 0.639 ± 0.080 vs shuffled
  0.507) — unlike gpt2's near-anti-transfer, the larger model's valence
  direction partly composes meaning rather than reading surface words.
  **Honest negative:** the plain→negated final-position patch is
  non-specific — recovery 0.523 vs an unrelated-statement control 0.525,
  both flipping the reading 100%. The intervention overwrites the
  answer-position representation regardless of content, so it cannot
  causally localize the negation computation; the audit reports this as a
  position artifact, not a mechanism. (Exactly the capstone lesson: a real
  causal claim must beat its control.)
- **Lab 11 factual_qa at 32B (Tier C).** Fully reproduces the 7B audit:
  preference accuracy 1.0, subject-early causal recovery 1.006 vs unrelated
  control 0.275, truth monitor AUC 1.0 vs shuffled 0.0.
- **Lab 7 on the engine (Tier B Olmo-3-7B-Instruct).** 215 s wall vs 413 s
  in run 4 (1.9×); 1060 generation jobs in 13 batched calls; induced
  refusal on benign prompts baseline 21% → max steered 100% (random control
  12%); all self-checks green. Per-job steering verified the dose sweeps
  ride one schedule per condition.
- **Tier C 32B spot-checks (labs 1/2/3).** First contact at 64 layers, zero
  lab-code changes; the depth-aware gate (#3) is the only adjustment, and
  every component-decomposition check passed with the correct hook pair well
  inside the widened tolerance (e.g. lab2 0.0213 vs effective 0.0283).

### Deferred on purpose

**Lab 6 at 32B (Tier C circuit discovery).** Killed after ~1h33m. py-spy
confirmed it was actively computing (not stalled), deep in greedy pruning —
having already passed hook-parity, lens, and component-decomposition
self-checks plus the motif and causal screens at 64 layers. Greedy pruning
is O(circuit²) full-model forward passes in eager attention (attention
patterns are required, so SDPA/flash are off), which is genuinely slow at
32B and was blocking the canonical Lab 10/11 reruns on the single GPU. The
32B circuit is a spot-check, not a course deliverable, and labs 1/2/3 +
11-factual already validate the 32B forward-pass path. Recorded as
validated-through-screening; full pruning completion is a known compute
cost, not a correctness question. (If wanted later: run it alone overnight,
or cap the screened candidate set before pruning.)

## Archive

Drive `interpret/run5/`: per-lab run dirs (all tiers; Tier C retries carry
an `_02` suffix from auto-naming), `code/` (exact tree),
`bench_inference_log.jsonl`, sweep + per-run console logs, this report, and
`course_dashboard_run5.{md,png}`.
