# Lab 11: Bandwidth-Optimal Ring All-Reduce

Goal: stop paying the whole-token tax. Implement all-reduce as reduce-scatter
plus all-gather over `B/N` shards so the ring finally moves the same bytes as
`lax.psum` — then measure how close correct-by-construction byte optimality
gets you to the compiler's latency, and read the residual gap out of the trace.

Labs 7 and 8 both end with the same confession. Lab 7 composed all-reduce from
whole-token ring phases; Lab 8 fused the whole-token ring into one overlapped
Pallas program with real async RDMA — and *still* couldn't touch `lax.psum` at
large payloads, because no amount of overlap forgives moving `N/2` times the
necessary bytes. Lab 8's handout promised that "a bandwidth-optimal
chunk-per-hop ring is a natural follow-on lab." This is that lab: the toll is
finally charged in shards, not whole tokens. 🧾

```text
Lab 7: composed, whole-token   ->  correct, wasteful
Lab 8: fused, whole-token      ->  overlapped, still wasteful
Lab 11: composed, sharded      ->  the bytes are finally right
```

The algorithm is the classic two-phase ring (Patarasuk & Yuan's
bandwidth-optimal all-reduce; the same schedule inside NCCL's ring and, when
XLA picks a ring strategy, inside `psum` itself):

```text
reduce-scatter   N-1 steps: forward your partial shard, fold in the local one.
                 Ends with each device owning ONE fully reduced shard.
all-gather       N-1 steps: circulate the finished shards verbatim.
                 Ends with every device holding the complete reduced payload.
```

Every one of the `2(N-1)` steps sends a `B/N` shard per device. Total reported
by the harness: `2(N-1)/N * B`, the standard full-duplex bandwidth term for
ring all-reduce and the same convention used by the existing `pmap_psum` byte
model. Each device receives the same amount; endpoint ingress+egress counters
would be twice the displayed `wire_bytes`. For the custom ring family in this
lab, **wire == logical** for the first time: the implementation's actual ring
traffic matches the optimal model instead of carrying a whole-token tax.

This is still a *systems-skills* lab, not a "beat the compiler" contest — but
the contest terms have changed. In Lab 8 losing to XLA was structural; here the
byte volumes are identical, so the remaining gap is scheduling. A strong v5e-4
run should be close to `xla_all_reduce` at multi-MiB payloads, often within a
few percent to a few tens of percent depending on trace-visible scheduling. The
job is to explain the gap, not hide it. If you're 2x off, you probably have a
serialization or topology problem you can find. That's a better class of
problem.

## Implemented Happy Path

Run (a single sweep compares the foil, the headline, the variant, and the
roofline):

```bash
python collective_bench.py --lab lab11
```

Implemented operations:

- `pmap_psum`: built-in `lax.psum` baseline from Lab 0 (logical-byte model
  already optimal; useful for cross-lab continuity).
- `pmap_token_ring`: the whole-token dependency-chain ring from Lab 2, riding
  along as the `N/2`-penalty foil. Watch its `useful/dev` column: it reports
  `(N-1) * B` while the optimal all-reduce rows report `2(N-1)/N * B`. It uses
  the harness's normal device-ID ring; use `--lab11-ring-order ids` when you
  want the cleanest byte-only comparison to the new shard ring.
- `pmap_rs_ag_all_reduce`: **the headline**. Unidirectional shard ring —
  reduce-scatter then all-gather — built from `lax.ppermute` plus dynamic shard
  indexing inside `jax.shard_map`.
- `pmap_rs_ag_all_reduce_bidir`: the same volume split into two
  counter-rotating half-rings so both ICI directions carry traffic at once.
  Same total bytes, `4(N-1)` half-size messages instead of `2(N-1)`.
- `xla_all_reduce`: `lax.psum` on the **same case in the same wire dtype** —
  the roofline. (Different from Lab 8's `xla_token_ring`, which cast to float32
  *before* the psum; see "Correctness Contract" for why that matters.)
- `lab11_optimal_all_reduce_spec`: course artifact with the shard schedules,
  byte model, α–β crossover, ring-order preview, and trace evidence rules.

All Lab 11 ops are portable JAX — no Pallas, no TPU gating. The whole lab runs
on CPU with forced host devices, which makes the schedule debuggable at your
desk before it touches a slice:

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  python collective_bench.py --lab lab11 --sizes 16KiB --iters 5 --no-plots
```

The module also self-tests standalone (schedule simulator at N=2..5, all
kernel modes × float32/bfloat16 × both directions, wire-byte invariants, and
the ring-order constructor):

```bash
python labs/lab11_optimal_all_reduce.py
```

## The Byte Model, Finally Optimal

Let `B` be the per-device payload and `N` the ring size. Lab 11 pins the shard
count to `N` (one shard per ring position) and rounds the *shard* tile up to
`rows x cols` elements, so `B_actual = N * shard_bytes` exactly — no remainder
shard, no asterisk on the model.

Per device, per full all-reduce, using the benchmark's send-side/full-duplex
bandwidth convention:

```text
whole-token ring (Labs 2/7/8):   send (N-1) * B
reduce-scatter + all-gather:     send 2 * (N-1) * (B/N)  =  2(N-1)/N * B
```

Receive volume is equal by symmetry. Hardware endpoint counters that add bytes
sent and bytes received would show twice the table values; the harness does not
double them because its GB/s denominator matches the earlier `pmap_psum` model.

The penalty ratio is the headline number of the course:

```text
naive / optimal = (N-1)*B / (2(N-1)/N * B) = N/2
```

| N   | naive `(N-1)·B` | optimal `2(N-1)/N·B` | naive/optimal |
|-----|-----------------|----------------------|---------------|
| 2   | `1.000 B`       | `1.000 B`            | 1x (identical) |
| 4   | `3.000 B`       | `1.500 B`            | **2x**        |
| 8   | `7.000 B`       | `1.750 B`            | 4x            |
| 16  | `15.000 B`      | `1.875 B`            | 8x            |
| 256 | `255.000 B`     | `1.992 B`            | 128x          |

Two readings worth saying out loud. First, the optimal volume *saturates* near
`2B` as the ring grows — all-reduce cost per device is nearly independent of
ring size in the bandwidth regime, which is the entire reason data-parallel
training scales. Second, the naive penalty grows *linearly* with `N`: a 2x
embarrassment on our v5e-4 becomes a 128x catastrophe on a full 256-chip v5e
slice. Lab 8's fused kernel was a fine engine bolted to the wrong gearbox.

At `N = 2` the two algorithms move identical bytes (the table's polite way of
saying this lab needs at least 3 devices to be interesting; the code still
runs and notes it).

## Why 2(N-1)/N·B Is a Lower Bound, Not Just a Good Idea

Sketch under the standard balanced, full-duplex, partial-sum message model:
fix any correct all-reduce and any device `d`.

- **Reduce ownership.** Partition the `B` bytes of result into the `1/N`
  fraction whose reduction `d` could finalize locally and the `(N-1)/N`
  fraction finalized elsewhere. Every byte of `d`'s contribution to that
  second fraction must leave `d` at least once — partial sums do not compress
  the contribution away. This costs at least `(N-1)/N * B` of send-side
  bandwidth from `d`.
- **Result dissemination.** Symmetrically, after reductions are finalized, `d`
  cannot be the final owner of `(N-1)/N * B` of the output but must still end
  with those bytes. In a balanced all-reduce bandwidth model, disseminating
  finalized shards costs another `(N-1)/N * B` send-side term per device.

Together that is the reported `2(N-1)/N * B` per-device bandwidth term. Each
device also receives the same amount, but the harness follows the usual
full-duplex collective convention and does not add receive bytes a second time
to the GB/s denominator. Reduce-scatter + all-gather meets the bound with
equality, which is what "optimal" means here — not "fast on Tuesdays" but
"no balanced correct algorithm in this cost model moves fewer reported bytes."
A student checkpoint question: where exactly does this argument use the fact
that messages may carry partial sums, and what changes if they may not?

## How Fast Should It Be, Really?

No measured Lab 11 numbers are printed here — this handout was written before
your run, and the lab should teach students to keep receipts instead of
prophecies. Run the sweep and fill in your own table. What the model
*predicts*, so you know whether to believe your receipts:

1. **`pmap_rs_ag_all_reduce` vs `xla_all_reduce`:** identical byte volume, so
   the gap is scheduling. Expect the student ring to trail by a few percent to
   a few tens of percent at multi-MiB payloads rather than by a structural byte
   multiple. The tax has three named
   collectors:
   - dynamic-update-slice copies between steps (the gathered shard is written
     into the output buffer instead of arriving in place),
   - strict step boundaries (each `ppermute` completes before the next add
     starts; XLA's fused collective pipelines internally),
   - XLA's `psum` may already drive both ICI directions, in which case a
     unidirectional ring's ceiling is *half* the link budget — which is
     precisely the bet `pmap_rs_ag_all_reduce_bidir` exists to test.
2. **vs `pmap_token_ring`:** at 4 MiB and up, the shard ring should win by
   roughly the byte ratio (2x at N=4), minus its extra per-step latencies.
   For calibration, Lab 8's measured table on v5e-4 at 4 MiB bf16 had
   `pmap_token_ring` at ~1375 µs against `xla_token_ring` at ~344 µs — a 4x
   gap of which 2x was bytes. Lab 11 removes the byte half of that excuse.
3. **A free prediction about the roofline itself:** Lab 8's `xla_token_ring`
   cast to float32 *before* its `psum`, so at bf16 it moved twice the wire
   bytes of Lab 11's `xla_all_reduce` on the same nominal payload. If the
   large-payload regime is truly bandwidth-bound, `xla_all_reduce` should beat
   Lab 8's 344 µs noticeably at the same 4 MiB. If it doesn't, your "bandwidth
   regime" starts later than you thought. Either outcome is a lesson.
4. **Small payloads:** the optimal ring should *lose* to the naive ring below
   the crossover. That is not a bug; that is the next section.

Profile before bragging. 🔬

## The α–β Crossover: Where the Naive Ring Wins

Model each step as latency `α` plus bytes over bandwidth `β`:

```text
T_naive   = (N-1) * (α + B/β)
T_optimal = 2(N-1) * (α + B/(N·β))
```

The optimal ring halves the bytes per step but doubles the step count. Setting
the two equal:

```text
B* = α·β·N / (N-2)        (undefined at N=2, where the byte volumes tie)
```

For `N = 4`: `B* = 2αβ`. Illustration only (fit your own constants): with
`α ≈ 2 µs` and `β ≈ 45 GB/s` per link, `B* ≈ 180 KB` — so the default sweep's
16 KiB and 256 KiB sizes are deliberately placed on either side of a plausible
crossover. The experiment:

1. Sweep small sizes: `--sizes 4KiB,16KiB,64KiB,256KiB,1MiB`.
2. Find the payload where `pmap_rs_ag_all_reduce` overtakes `pmap_token_ring`.
3. Fit `α` from the small-payload plateau and `β` from the large-payload slope
   of either op; check the measured crossover against `2αβ`.

If your measured crossover and your fitted `2αβ` disagree wildly, one of your
fits is keeping bad books — usually `α`, which on real hardware also includes
per-step dispatch and the dynamic-slice tax, not just the wire.

## Mental Model

Input is the Lab 8 tensor with the chunk axis re-badged as the **shard** axis
and its length pinned to `N`:

```text
x[device, shard, row, col]    shape [N, N, rows, cols], default dtype bfloat16
x[s, c, 0, 0] = 10*s + c                      (the hand-checkable marker)
x[s, c, i, j] = 10*s + c + 0.25*(i%8) + 0.03125*(j%16)
```

The mesh partitions the first axis. All-reduce sums over the *device* axis, so
every device must end with the same `[N, rows, cols]` total. The marker makes
the schedule auditable by eye:

```text
expected marker at [*, c, 0, 0] = Σ_s (10s + c) = 10·N(N-1)/2 + N·c
N = 4:   shard 0 -> 60,  shard 1 -> 64,  shard 2 -> 68,  shard 3 -> 72
```

Those four integers are exactly representable even in bfloat16, so the marker
table is *exact* regardless of wire dtype — only the fractional row/col
pattern feels rounding. A wrong marker is always a schedule bug, never noise.
`observed_shard_markers(jax, y)` prints the `[:, :, 0, 0]` table for exactly
this purpose.

## The Two Schedules

Let `sgn = +1` for a right ring (source `i` sends to `(i+1) % N`) and `-1` for
left. Device `d`'s local view is `chunks[N, rows, cols]`; one `[rows, cols]`
partial `v` circulates via `ppermute`.

**Reduce-scatter.** Start `v = chunks[d]` (your own shard index `d`). Then for
step `s = 1 .. N-1`: forward `v`, and fold in local shard `(d - sgn·s) % N`.
The invariant — provable by induction and checked mechanically by
`simulate_rs_ag` — is:

```text
after step s, device d holds   Σ_{j=0..s} C[(d - sgn·j) % N][(d - sgn·s) % N]
```

a partial of shard `(d - sgn·s) % N` covering `s+1` sources. At `s = N-1` the
partial covers everyone and `(d - sgn·(N-1)) % N = (d + sgn) % N`:

```text
device d ends reduce-scatter owning fully reduced shard (d + sgn) % N
```

Worked table, N=4, right ring (`sgn=+1`) — which shard each device's `v` is a
partial of:

| after step | device 0 | device 1 | device 2 | device 3 | sources covered |
|-----------|----------|----------|----------|----------|-----------------|
| start     | 0        | 1        | 2        | 3        | 1               |
| 1         | 3        | 0        | 1        | 2        | 2               |
| 2         | 2        | 3        | 0        | 1        | 3               |
| 3         | 1        | 2        | 3        | 0        | 4 — done        |

**All-gather.** Seed the output with the shard you just finished:
`out[(d + sgn) % N] = v`. Then for `s = 0 .. N-2`: forward `v`, and store the
arriving *finished* shard at index `(d - sgn·s) % N`. No arithmetic happens in
this phase — finished shards are copied verbatim, which is what makes the
bitwise-replica invariant below possible.

Worked table, N=4, right ring — which shard index each device stores:

| event     | device 0 | device 1 | device 2 | device 3 |
|-----------|----------|----------|----------|----------|
| seed      | 1        | 2        | 3        | 0        |
| step 1    | 0        | 1        | 2        | 3        |
| step 2    | 3        | 0        | 1        | 2        |
| step 3    | 2        | 3        | 0        | 1        |

Each column covers `{0,1,2,3}`: every device assembles the full payload in
`N-1` gather steps. Total ppermutes: `2(N-1)` (6 at N=4), each one shard.

Two index identities students reliably trip on:

- The RS loop body at iteration `s_loop ∈ [0, N-2]` is executing *step*
  `s = s_loop + 1`, so the local shard index is `(d - sgn·(s_loop+1)) % N`.
  Off by one here and your sums silently cover the wrong source sets.
- `%` on a traced int32 in JAX lowers to `jnp.remainder`, which is
  non-negative for a positive modulus — so `(d - 3) % 4` is safe. `lax.rem`
  follows C semantics and is **not** safe. Use `%`.

The spec artifact (`lab11_optimal_all_reduce_spec`) emits both schedules as
explicit per-device tables for your `N` and direction, and
`simulate_rs_ag(chunks_per_device, direction)` replays the exact index
schedule in pure NumPy with explicit message passing — when a device run
disagrees with the simulator, the bug is in the collective plumbing; when both
agree and are wrong, it's your expectation.

## Direction Convention

Same as Labs 1–8: `right` means source `i` sends to `(i+1) % N`, so receiver
`r` hears from `r-1` and sees source history `[r, r-1, r-2, ...] mod N`;
`left` is the mirror. `--neighbor-direction` flips it globally. Direction does
not change bytes or step count — it changes *which* shard each device ends up
owning after reduce-scatter (`(d ± 1) % N`) and which physical ICI direction
carries the traffic. The bidirectional mode runs the configured direction on
the top row-half of every shard and the opposite direction on the bottom half.

## Ring Order on Real Hardware

`jax.devices()` order is an ID order, not a topology order. On a 2x2 v5e
slice with the common coordinate assignment, the ID ring `0→1→2→3→0` contains
two edges of Manhattan distance 2 — two of your four "neighbor" hops actually
traverse two physical links, paying double latency and sharing segments with
other steps.

`--lab11-ring-order auto` (the default) instead orders the mesh along a
**unit-step Hamiltonian cycle** over device `.coords` when one exists, so
every `ppermute` hop crosses exactly one physical ICI link. On 2x2 that is the
cycle `(0,0)→(0,1)→(1,1)→(1,0)→(0,0)`. The constructor handles any `X×Y` grid
with an even cell count (comb construction; transposed when needed) and
**falls back to ID order with a recorded reason** when no such cycle exists —
odd×odd grids, lines longer than a pair, missing coords (CPU/GPU), or sparse
slices. `--lab11-ring-order ids` forces the naive order so you can measure
the difference.

This reordering is safe for an all-reduce: the result is the same full sum on
every device, and inputs, outputs, and the correctness check all address
devices by logical mesh rank. The only thing that changes is which physical
links carry each step — which is exactly the experiment. The spec artifact's
`ring_order_preview` shows the chosen cycle (and the fallback reason, if any)
for your live devices before you commit to a sweep; the per-run `note` column
records the actual ring as `ring=[...]`.

Coordinate assignments vary across runtimes and slice shapes — trust the
preview, not folklore.

Fairness note: `--lab11-ring-order auto` is a topology-aware optimization for
the new shard-ring cases. The Lab 2 `pmap_token_ring` foil still uses the
harness's ordinary device-ID ring, so the purest algorithmic byte comparison is
`--lab11-ring-order ids`. Then run `auto` separately to ask the topology
question: same bytes, better physical neighbors? The two experiments teach
different things; keep their receipts in separate pockets.

## Correctness Contract

Checked by `check_result(jax, jnp, y, expected, dtype=...)`, in order of
diagnostic value:

1. **Bitwise replica identity.** All device replicas of the output must be
   bitwise equal: every reduced shard is computed exactly once during
   reduce-scatter and copied *verbatim* during all-gather, and `psum` is a
   single deterministic collective. This check fires before any tolerance is
   consulted, because a replica mismatch is never rounding — it is a stale
   shard slot or a wrong-owner index, and it localizes the bug to the
   schedule. Note what is **not** promised: `rs-ag` and `xla-psum` need not
   match *each other* bitwise. Their reduction orders differ; both must pass
   the contract independently.
2. **Full-tile numeric check.** Every element of every shard, on every
   device, against float32 sums of the **dtype-quantized** input — not just
   the `[0,0]` markers. Expected values are built by casting the input to the
   wire dtype first (so input quantization is modeled) and summing in
   float32.
3. **Dtype-aware tolerances.** The kernel accumulates partials in the *wire
   dtype* — that is the whole point of the byte model; circulating float32
   partials of a bfloat16 payload would double the wire bytes and put us
   right back in Lab 8. Wire-dtype accumulation costs precision, so the
   permitted error is a property of the dtype, not the algorithm:

   | wire dtype | rtol | atol | note |
   |-----------|------|------|------|
   | float32/64 | 1e-5 | 1e-4 | accumulation ≈ reference |
   | bfloat16   | 2e-2 | 2.0  | 8 mantissa bits, tile values reach ~75 |
   | float16    | 1e-2 | 0.5  | |
   | integers   | 0    | 0    | exact or wrong, no third option |

   The scalar markers (60/64/68/72 at N=4) are exact in every supported float
   dtype regardless — small integers don't round. A marker error is a
   schedule bug; a fractional-pattern drift inside tolerance is the cost of
   honest bytes.
4. **Shape and rounding.** The shard tile rounds up to `rows × cols`
   elements, so `actual_payload_bytes = N · shard_bytes` exactly and the
   reported byte model carries no remainder term. The bidirectional mode
   needs an even row count and bumps an odd `tile_rows`, recording the
   adjustment in the case note.

One deliberate contrast with Lab 8, worth teaching explicitly: Lab 8's
`xla_token_ring` cast to float32 *before* its `psum`, which both inflated bf16
wire bytes 2x and bought it float32 accumulation accuracy. Lab 11's
`xla_all_reduce` reduces in the wire dtype and casts after — same bytes, same
rounding exposure as the student kernel. Apples to apples, including the
bruises.

## Run Commands

The default lab sweep (all six ops, sizes spanning the crossover through the
bandwidth regime, bf16 wire dtype):

```bash
python collective_bench.py --lab lab11
```

Find the α–β crossover against the naive ring:

```bash
python collective_bench.py --lab lab11 \
  --ops pmap_token_ring,pmap_rs_ag_all_reduce,xla_all_reduce \
  --sizes 4KiB,16KiB,64KiB,256KiB,1MiB
```

Ring order: physical cycle vs ID order (run both, compare p50 at a fixed
payload — bytes are identical by construction, so any difference is topology):

```bash
python collective_bench.py --lab lab11 --sizes 4MiB --lab11-ring-order auto
python collective_bench.py --lab lab11 --sizes 4MiB --lab11-ring-order ids
```

The bidirectional bet, against the unidirectional ring and the roofline:

```bash
python collective_bench.py --lab lab11 \
  --ops pmap_rs_ag_all_reduce,pmap_rs_ag_all_reduce_bidir,xla_all_reduce \
  --sizes 1MiB,4MiB,16MiB
```

Wire dtype sweep — bf16 halves the bytes of f32 at the same element count;
does it halve the time?

```bash
python collective_bench.py --lab lab11 --sizes 4MiB --dtype bfloat16
python collective_bench.py --lab lab11 --sizes 4MiB --dtype float32
```

Capture a trace of the headline kernel at a bandwidth-regime payload:

```bash
python collective_bench.py --lab lab11 --profile \
  --trace-op pmap_rs_ag_all_reduce --trace-size 4MiB
```

Direction flip (owned shards move from `(d+1)%N` to `(d-1)%N`; bytes and
latency should not care):

```bash
python collective_bench.py --lab lab11 --sizes 1MiB --neighbor-direction left
```

Everything also runs on CPU with forced host devices for schedule debugging —
prepend `XLA_FLAGS="--xla_force_host_platform_device_count=4"` and add
`--iters 3 --no-plots`. CPU timings are noise; CPU *correctness* is the same
schedule you'll ship to the slice.

## What To Inspect

Run artifacts, same layout as every lab:

- `results.jsonl` / `csvs/results.csv`: per-op rows. For Lab 11 check that
  `wire_bytes == logical_bytes == 2(N-1) · shard_bytes` for the three new ops
  under the harness's send-side/full-duplex convention, and that
  `pmap_token_ring`'s model is `N/2` times larger.
- `lab_artifacts/*lab11_optimal_all_reduce_spec*`: schedules, byte model,
  crossover formula, ring-order preview, checkpoint questions.
- `plots/latency_by_payload.png` and `plots/bandwidth_by_payload.png`: the
  crossover is visible as the payload where the rs-ag curve crosses under the
  token-ring curve.
- `runs/<run>/traces/...`: the receipts.

Reading the trace (XProf or Perfetto on the captured `.json.gz`):

1. **Count the permutes.** `pmap_rs_ag_all_reduce` at N=4 shows **6**
   `collective-permute-start`/`-done` pairs, each moving ~`shard_bytes`
   (`B/4`). The whole-token ring shows **3** pairs at ~`B` each. Same wall,
   different bricks.
2. **Name the tax.** Between the permutes sit dynamic-(update-)slice copies
   and fused elementwise adds — the step-boundary work that XLA's single
   fused all-reduce region doesn't pay. Sum those gaps; that is most of your
   delta to `xla_all_reduce`.
3. **Per-step achieved bandwidth.** `shard_bytes / step_time` per permute,
   against the link roofline. If early steps are slower than late ones,
   you're watching warm-up and dispatch, not the wire.
4. **The bidir verdict.** `pmap_rs_ag_all_reduce_bidir` should show two
   counter-rotating permute chains *overlapping in time*. If they serialize,
   XLA scheduled your two independent dataflow chains back-to-back and the
   bidirectional bet bought you nothing but smaller messages — that result is
   just as reportable as a win.
5. **Ring-order forensics.** With `--lab11-ring-order ids` on a 2x2 slice,
   the two 2-hop edges should show as slower permute steps; with `auto` the
   six steps should be near-uniform.
6. **One caveat about the built-in summary.** The harness's trace summarizer
   (`_classify_comm_event`) was written for Pallas kernels — it recognizes
   `copy*`, `barrier-cores`, and semaphore events, so Lab 11's XLA
   `collective-permute` events won't appear in `trace_summaries/`. Read the
   raw trace in the viewer, or apply the optional two-line classifier
   extension in Appendix B.

## Common Failure Modes

Symptom:

```text
replicas are bitwise identical, markers are wrong (e.g. shard 0 shows 64, not 60)
```

Likely explanation: reduce-scatter local-shard index is off by one — the loop
body at iteration `s_loop` is step `s_loop + 1`, so the index is
`(d - sgn*(s_loop+1)) % N`. The all-gather faithfully replicated your wrong
sums everywhere, which is exactly why the replica check and the value check
are separate diagnostics.

Fix: replay the case in `simulate_rs_ag` with unit tiles and compare the
marker table step by step against the schedule tables above.

Symptom:

```text
asked for a right ring, but device d ends reduce-scatter owning shard (d-1) % N
```

Likely explanation: `ppermute`'s permutation entries are `(source,
destination)` pairs. A right ring is `[(i, (i+1) % N) ...]` — writing
"who do I receive from" instead of "who do I send to" silently builds the
mirror ring, and every index formula is then signed wrong.

Fix: one convention, stated once: `perm[i] = (i, (i + sgn) % N)`; receiver `d`
hears from `(d - sgn) % N`.

Symptom:

```text
TracerIntegerConversionError / ConcretizationTypeError on chunks[local_idx]
```

Likely explanation: `local_idx` depends on `lax.axis_index`, which is a traced
value — Python `[]` indexing needs a concrete integer.

Fix: `lax.dynamic_index_in_dim(chunks, local_idx, axis=0, keepdims=False)` and
`lax.dynamic_update_index_in_dim(out, v, idx, axis=0)`. The case builder
already does this; this bites people writing their own variant.

Symptom:

```text
negative or wrapped-wrong shard indices when hand-rolling with lax.rem
```

Likely explanation: `lax.rem` follows C semantics (sign of the dividend), so
`lax.rem(-3, 4) == -3`. Python `%` on a traced int lowers to `jnp.remainder`,
which is non-negative for a positive modulus.

Fix: use `%`. This is the one place in the course where the *less* explicit
spelling is the correct one.

Symptom:

```text
rs-ag-bidir fails to build, or concat shapes mismatch
```

Likely explanation: the bidirectional split halves the row axis, which needs
an even row count. The shipped builder bumps an odd `tile_rows` and records
`bidir needs even rows; tile_rows R->R+1` in the note; a hand-rolled variant
that slices `[:half]` / `[half:]` with odd rows builds unequal halves.

Fix: keep `tile_rows` even (the default 4 is), or let the builder bump it.

Symptom:

```text
bfloat16 run fails the full-tile check with float32-grade tolerances
```

Likely explanation: the kernel accumulates in the wire dtype — by design —
so comparing against the float32 reference at `rtol=1e-5` is asking bf16 to
be something it is not.

Fix: pass `dtype=` to `check_result` (the harness runner does) and accept the
dtype table. If the *markers* are wrong, that is a real bug; the markers are
exact in bf16.

Symptom:

```text
rs-ag and xla_all_reduce outputs differ bitwise
```

Likely explanation: nothing. Different reduction orders, different rounding;
both pass the contract against the float32 reference independently.

Fix: none. Stop diffing them and read the trace instead.

Symptom:

```text
note says "no dense 2D device coords; using id order"
```

Likely explanation: CPU/GPU devices (or an unusual slice) expose no usable
`.coords`, so the auto ring constructor fell back, as documented.

Fix: nothing to fix on CPU. On a TPU slice where you expected a cycle, check
the spec's `ring_order_preview` — odd×odd grids and lines have no unit-step
Hamiltonian cycle, and the fallback reason will say so.

## Pass Condition

- `pmap_rs_ag_all_reduce` and `pmap_rs_ag_all_reduce_bidir` pass the full
  correctness contract (replica identity + full-tile values) across the
  default size sweep, both directions, in bfloat16 and float32.
- Reported `wire_bytes == logical_bytes == 2(N-1) · shard_bytes` for all
  three new ops; `pmap_token_ring`'s model is `N/2` times larger.
- The spec artifact emits the shard schedules, owner map, byte model with
  lower-bound note, α–β crossover, and ring-order preview.
- At ≥4 MiB, `pmap_rs_ag_all_reduce` is in the same performance neighborhood
  as `xla_all_reduce`, **or** the trace names the specific gap collectors
  (slice copies, step stalls, unidirectional ceiling, topology) that account
  for the difference. An explained 15% beats an unexplained 8%.
- The crossover experiment produces a fitted `α`, `β`, and a measured
  crossover payload consistent with `B* = αβN/(N-2)` to within honest error
  bars.

## What the Kernel Actually Does

For readers who want the code path without opening the file:
`build_case` sizes the shard tile (`shard_bytes = rows·cols·itemsize`,
`B_actual = N · shard_bytes`), orders the mesh devices along the ring policy,
builds the Lab 8-style input, and jits one of three functions. The rs-ag path
is `jax.shard_map` over a local function that strips the leading device axis,
runs the two loops above with `lax.ppermute` + dynamic shard indexing in the
wire dtype, and casts the assembled `[N, rows, cols]` result to float32 once
at the end. The bidir path calls the same inner function twice on row-halves
with opposite signs and concatenates. The psum path is `lax.psum` on the local
block in the wire dtype, cast after. ~80 lines of algorithm; the rest of the
module is schedules, validation, and receipts — the usual ratio for code you
intend to trust.

## Deferred Work

- **Fuse it.** This lab deliberately composes XLA collectives so the schedule
  is legible. The Lab 8 payoff move — one Pallas program, `k·N` sub-shards,
  async remote DMA overlapped with the local adds, buffer slots and capacity
  semaphores — applies here and is the real follow-on: a *pipelined* optimal
  ring that hides the step-boundary tax this lab measures.
- **Latency regime.** Recursive halving / tree all-reduce beats every ring
  below the crossover (`O(log N)` steps); implementing it would complete the
  payload-size story.
- **General topologies.** The unit-step cycle constructor handles dense 2D
  grids; Lab 9's meshes want Hamiltonian cycles (or multi-ring decompositions)
  on wrapped tori and 3D slices.
- **Partial hops.** A `--token-hops`-style partial schedule for straggler and
  fault experiments.

## Bridge

Lab 8 built the engine: a fused kernel with real overlap, bolted to a
whole-token schedule that wasted `N/2` of its effort. Lab 11 built the
gearbox: the optimal schedule, run through legible composed collectives that
leave a few percent on the table at step boundaries. The course's remaining
arc is to put them together — and Lab 9's topology work tells you which
physical links the fused, sharded, correctly-geared ring should ride.

## Appendix A: Wiring Lab 11 Into `collective_bench.py`

Two options. **Mechanical:** run the bundled patcher from the course
directory — it validates every anchor appears exactly once before touching
anything, writes a `.bak`, preserves CRLF/LF, and refuses to double-patch:

```bash
python apply_lab11_patch.py            # patches ./collective_bench.py in place
python apply_lab11_patch.py src.py dst.py   # or explicit paths
```

**By hand:** eight edits, listed in file order. Drop
`lab11_optimal_all_reduce.py` into `labs/` first.

**A1. Op table** — after the `LAB10_OPS = (...)` tuple:

```python
# Lab 11 closes the byte gap: the same all-reduce as reduce-scatter plus
# all-gather over B/N shards, matching lax.psum's 2*(N-1)/N*B volume. The
# whole-token pmap_token_ring rides along as the N/2-penalty foil.
LAB11_OPS = (
    "pmap_psum",
    "pmap_token_ring",
    "pmap_rs_ag_all_reduce",
    "pmap_rs_ag_all_reduce_bidir",
    "xla_all_reduce",
    "lab11_optimal_all_reduce_spec",
)
```

**A2. Spec routing** — add to the `LAB_SPEC_OPS` dict, after the lab10 entry:

```python
    "lab11_optimal_all_reduce_spec": "labs.lab11_optimal_all_reduce",
```

**A3. Op validation** — in `ALL_OPS`, after `"pmap_2d_staged_all_gather",`
and before `*LAB_SPEC_OPS,`:

```python
    "pmap_rs_ag_all_reduce",
    "pmap_rs_ag_all_reduce_bidir",
    "xla_all_reduce",
```

**A4. Lab defaults** — in `apply_lab_defaults`, immediately before the final
`raise ValueError(f"unknown lab {args.lab!r}")`:

```python
    if args.lab == "lab11":
        # Lab 11 is the bandwidth-optimal shard ring; the default sweep spans
        # the alpha-beta crossover (small sizes, where the naive ring wins on
        # latency) through the bandwidth regime where matching psum's volume
        # pays off.
        if args.ops is None:
            args.ops = ",".join(LAB11_OPS)
        if args.sizes is None:
            args.sizes = "16KiB,256KiB,1MiB,4MiB,16MiB"
        if args.run_name is None:
            args.run_name = f"lab11_optimal_all_reduce-{now_slug()}"
        return
```

**A5. CLI choices** — in `build_parser`, add `"lab11",` after `"lab10",` in
the `--lab` choices tuple, and (optional) extend the help string's
`lab10 is multi-host run-control smoke` to
`lab10 is multi-host run-control smoke, lab11 is the bandwidth-optimal
shard-ring all-reduce`.

**A6. Ring-order flag** — after the `--lab9-axis-order` argument:

```python
    parser.add_argument(
        "--lab11-ring-order",
        choices=("auto", "ids"),
        default="auto",
        help=(
            "Lab 11 ring layout: auto walks a unit-step Hamiltonian cycle over "
            "device coords when one exists; ids uses jax.devices() order and "
            "is the cleanest byte-only comparison with pmap_token_ring"
        ),
    )
```

**A7. Dispatch** — in `dispatch_case`, after the `semaphore_bug_zoo` branch
and before `if op in LAB_SPEC_OPS:`:

```python
        if op == "pmap_rs_ag_all_reduce":
            return run_pmap_rs_ag_all_reduce(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pmap_rs_ag_all_reduce_bidir":
            return run_pmap_rs_ag_all_reduce_bidir(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "xla_all_reduce":
            return run_xla_all_reduce(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
```

**A8. Runners** — after `run_xla_token_ring` and before
`run_pallas_2d_staged_all_gather` (mirrors `_run_lab8_ring`'s structure: lazy
import, case build, correctness, trace-wrapped timing, BenchResult with the
case's byte model):

```python
def _run_lab11_ring(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
    *,
    op: str,
    kernel_mode: str,
) -> BenchResult:
    """Run one Lab 11 all-reduce implementation in a given kernel mode.

    ``kernel_mode`` pins the implementation per benchmark op: ``rs-ag`` (the
    bandwidth-optimal shard ring), ``rs-ag-bidir`` (two counter-rotating
    half-rings), or ``xla-psum`` (lax.psum on the same case in the same wire
    dtype). All three move the optimal byte volume, so wire == logical and the
    remaining differences are pure scheduling.
    """
    layer = "shard_map/psum" if kernel_mode == "xla-psum" else "shard_map/ppermute"

    try:
        from labs import lab11_optimal_all_reduce
    except Exception as exc:
        return BenchResult(
            op=op, layer=layer, payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False, note=f"import failed: {exc}",
        )

    try:
        # Lab 11's case builder owns shard sizing, ring layout, and expected
        # sums. The harness records the byte model exposed by the case.
        case = lab11_optimal_all_reduce.build_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            direction=args.neighbor_direction,
            kernel_mode=kernel_mode,
            tile_rows=args.pallas_tile_rows,
            min_cols=args.pallas_min_cols,
            ring_order=args.lab11_ring_order,
        )
    except Exception as exc:
        return BenchResult(
            op=op, layer=layer, payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False,
            note=f"case setup failed: {exc}",
        )

    try:
        y = block_until_ready(jax, case.fn(case.x))
        if args.skip_correctness:
            ok = True
        else:
            ok = lab11_optimal_all_reduce.check_result(
                jax, jnp, y, case.expected_sums, dtype=dtype
            )
        with maybe_trace(jax, args, run, op, payload_bytes) as trace_artifact:
            timing = time_jax_call(
                jax, case.fn, case.x, warmup=args.warmup, iters=args.iters
            )
        memory_profile = maybe_write_memory_profile(jax, args, run, op, payload_bytes)
    except Exception as exc:
        return BenchResult(
            op=op, layer=layer, payload_bytes=case.actual_payload_bytes,
            logical_bytes=case.optimal_bytes_per_device, seconds=None, ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op=op,
        layer=layer,
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=case.optimal_bytes_per_device,
        wire_bytes=case.wire_bytes,
        byte_model="optimal",
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=case.note,
        **result_timing_kwargs(timing),
    )


def run_pmap_rs_ag_all_reduce(
    jax: Any, jnp: Any, args: argparse.Namespace, run: RunContext,
    payload_bytes: int, dtype: Any, n_devices: int,
) -> BenchResult:
    """Lab 11 bandwidth-optimal shard ring (reduce-scatter + all-gather)."""
    return _run_lab11_ring(
        jax, jnp, args, run, payload_bytes, dtype, n_devices,
        op="pmap_rs_ag_all_reduce", kernel_mode="rs-ag",
    )


def run_pmap_rs_ag_all_reduce_bidir(
    jax: Any, jnp: Any, args: argparse.Namespace, run: RunContext,
    payload_bytes: int, dtype: Any, n_devices: int,
) -> BenchResult:
    """Lab 11 bidirectional variant: two counter-rotating half-rings."""
    return _run_lab11_ring(
        jax, jnp, args, run, payload_bytes, dtype, n_devices,
        op="pmap_rs_ag_all_reduce_bidir", kernel_mode="rs-ag-bidir",
    )


def run_xla_all_reduce(
    jax: Any, jnp: Any, args: argparse.Namespace, run: RunContext,
    payload_bytes: int, dtype: Any, n_devices: int,
) -> BenchResult:
    """Lab 11 roofline: lax.psum on the same case in the same wire dtype."""
    return _run_lab11_ring(
        jax, jnp, args, run, payload_bytes, dtype, n_devices,
        op="xla_all_reduce", kernel_mode="xla-psum",
    )
```

Smoke-test the wiring without a TPU:

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  python collective_bench.py --lab lab11 --sizes 16KiB --iters 3 --warmup 1 --no-plots
```

All six rows should print `ok=True`; the new ops report `useful/dev` of
1.5x the payload, the token ring 3x, and the spec op writes
`lab_artifacts/*lab11_optimal_all_reduce_spec.{json,md}`.

## Appendix B (Optional): Teach the Trace Summarizer About XLA Collectives

The built-in `trace_summaries/` only counts Pallas-style events. To get Lab 11
permutes into the summary, add two checks at the top of
`_classify_comm_event`:

```python
    if name.startswith("collective-permute"):
        return "xla_collective_permute"
    if name.startswith(("all-reduce", "all-gather", "reduce-scatter")):
        return "xla_collective"
```

Event naming varies across XLA versions (fusion wrappers sometimes prefix
names), so treat the summary as a convenience and the raw trace as the
arbiter — which was true in Lab 8 too, the trace just didn't have a lab
number on it.
