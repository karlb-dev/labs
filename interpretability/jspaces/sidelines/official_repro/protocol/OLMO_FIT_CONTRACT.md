# OLMo fit contract (frozen pre-data)

One prospective official-estimator fit on
`allenai/Olmo-3.1-32B-Instruct` @ `ac0587e4`. Addendum §2.1/§2.4/§2.5
folded in. No outcome-selected refit; no third lens; no transferred lens
as primary.

## 1. Estimator (upstream-exact)

`jlens.fitting.jacobian_for_prompt` semantics at `581d3986`: one forward
with the prompt replicated `dim_batch` times, `ceil(5120/dim_batch)`
backwards; one-hot cotangents at every valid target position; mean over
valid source positions (`skip_first=16` … `seq_len−2`). Wrapper may
harden checkpointing/logging/recovery (Phase-4 patterns) but must be
proven **algebraically identical** to upstream `jlens.fit` +
`JacobianLens.merge` on a tiny model before OLMo runs.

```
target_layer 63 · source_layers = the frozen 32-layer union
max_seq_len 128 · skip_first 16 · fp32 accumulator ·
saved lens fp16 + exact fp32 running-state checkpoints
```

## 2. Fit population (addendum §2.1 — upstream criterion verbatim)

`Salesforce/wikitext` config `wikitext-103-raw-v1` split `train`,
resolved to an exact Hub revision + fingerprint before fitting. Accept a
record iff `len(record["text"].strip()) >= 600` (leading/trailing strip
only — internal whitespace counts). Materialize the **raw, unstripped**
`text` of the first 1,000 qualifying records to
`data/fit/wikitext_first1000_min600.jsonl` with per-row: stream index,
raw bytes sha256, raw length, stripped length, per-model token count.
Halves: even-index rows = **A**, odd = **B**. Rows after the first 1,000
feed timing/runtime sentinels only.

## 3. Timing gate and precommitted route (addendum §2.4 break-evens)

Benchmark `dim_batch ∈ {1,2,4,8}` (+ {16} if 8 fits with ≥15 GiB head-
room, recorded before any timing is read) on 3 excluded prompts; two
repeated runs at the selected value must agree within the runtime
sentinel tolerance (max relative per-layer Frobenius diff ≤ 0.5, the
campaign's prospective contract value). Backward count per prompt is
`ceil(5120/dim_batch)` (640 at dim_batch=8) **independent of layer
count**; measured time far above the ≈157 s/prompt campaign prior
indicates a runtime problem, not a layer tax.

Ceiling 18 GPU-hours for the complete two-half fit. Break-evens
(64,800 s ÷ n_prompts): **n=1000 needs ≤ 64.8 s/prompt · n=500 ≤ 129.6 ·
n=250 ≤ 259.2.** Route = largest tier whose projection fits; chosen from
timing only, before any lens-eval or causal outcome; recorded
prominently; never extended after outcomes. If 2×125 still exceeds the
ceiling → OLMo official-fit instrument blocked (no substitute primary).

## 4. Checkpoint and durability (addendum §2.5)

- Local atomic checkpoint every 3 accepted prompts (upstream
  `_atomic_save` pattern), checkpoint-first, header (JSON sidecar with
  counts + runtime sentinel + content hash) last.
- Drive recovery copy every 15 accepted prompts; retain newest **two**
  per half + every registered milestone; delete an older copy only after
  the newer is rematerialized and rehashed. (Each fp32 checkpoint ≈
  32·5120²·4 B ≈ 3.4 GB.)
- Milestone lenses at equal half counts 125/250/500 when the route
  includes them; the final merged lens is prompt-count-weighted
  `JacobianLens.merge([A, B])`.
- Skipped prompts logged with source index and reason; `next_idx` /
  `n_done` reconciled at every checkpoint (§15 stop rule).
- Resume forbidden when the runtime sentinel or fit header differs.

## 5. Per-prompt diagnostics

seq_len, valid positions, per-prompt max ||J||/√d, running-mean relative
movement, finite status, wall time, peak VRAM, checkpoint hash. No
heavy-tail row is trimmed post hoc.

## 6. Split-half audit (before merge use; plan §8.6)

Operator layer (raw / minus-identity / minus-scaled-identity cosine,
symmetric relative Frobenius, identity fraction, streamed principal
subspaces) · readout layer (six evals on both halves; target-row cosines
on every battery token; top-10 overlap on fixed activations; both bands)
· sparse/intervention layer on the frozen 20-cell calibration subset
(selected before either half exists; hash in the preregistration).
Stability margins: half-vs-half eval pass@20 gap ≤ 0.10 per set and
calibration-subset effect-direction agreement; outside → OLMo causal
results classified `FIT-SENSITIVE`, stop after calibration/core boundary.
