# Lab 6: Ring Reduce-Scatter

## Course Placement

Lab 5 built an **all-gather**: each device started with one shard and ended up
seeing every shard. Lab 6 flips the story. Each device starts with one chunk for
every possible output owner, all devices cooperate to reduce those chunks, and
then each device keeps only the chunk it owns.

This is the ownership-changing half of serious all-reduce:

```text
all-reduce = reduce-scatter + all-gather
```

That identity is the bridge to Lab 7.

## Goal

Learn reduce-scatter as an ownership transform, not just as an API call.

By the end of this lab, students should be able to explain:

- what each input chunk means;
- which device owns each output chunk;
- which source ranks contributed to each owner chunk;
- why the teaching implementation sends too many bytes on purpose;
- why reduce-scatter is useful in ML systems even before all-gather runs.

## Run

```bash
python collective_bench.py --lab lab6
```

Short smoke run:

```bash
python collective_bench.py \
  --lab lab6 \
  --sizes 1KiB,64KiB \
  --iters 10 \
  --warmup 2
```

Profile the custom Pallas path:

```bash
python collective_bench.py \
  --lab lab6 \
  --sizes 1MiB,4MiB,16MiB \
  --profile \
  --profile-cases 1 \
  --trace-op pallas_ring_reduce_scatter \
  --trace-size 4MiB
```

## Implemented Happy Path

- `pmap_psum_scatter`: built-in reduce-scatter executable specification.
- `pallas_ring_reduce_scatter`: whole-token ring built from repeated Lab 1
  Pallas remote-DMA hops.
- `lab6_reduce_scatter_spec`: course artifact that records ownership, byte
  model, and the optimized ring plan still to build.

The custom Pallas path is a teaching implementation, not the final bandwidth
model. It sends the full per-device chunk vector on every hop and accumulates
only the chunk owned by the receiving device. The optimized ring version sends
one chunk per hop. That optimization is intentionally deferred so the ownership
transform stays inspectable first.

## Mental Model

For `N` devices, each device starts with `N` chunks:

```text
source device s owns input token:

[chunk_for_owner_0, chunk_for_owner_1, ..., chunk_for_owner_N-1]
```

After reduce-scatter:

```text
device j owns reduced chunk j
```

That means:

```text
output[j] = sum over all source devices s of input[s, chunk j]
```

The teaching input uses a deliberately simple value pattern:

```text
x[source, owner_chunk, row, col] = 10 * source + owner_chunk
```

For four devices, the input scalars look like this:

| Source device | Chunk 0 | Chunk 1 | Chunk 2 | Chunk 3 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 2 | 3 |
| 1 | 10 | 11 | 12 | 13 |
| 2 | 20 | 21 | 22 | 23 |
| 3 | 30 | 31 | 32 | 33 |

The full reduce-scatter result should therefore be:

| Output owner | Reduced value | Explanation |
|---:|---:|---|
| 0 | 60 | `0 + 10 + 20 + 30` |
| 1 | 64 | `1 + 11 + 21 + 31` |
| 2 | 68 | `2 + 12 + 22 + 32` |
| 3 | 72 | `3 + 13 + 23 + 33` |

Every element of device `j`'s output tile should equal the reduced value for
chunk `j`.

## Ring Direction

Direction is defined from the sender's point of view, matching Labs 1, 2, and
5.

```text
right: device i sends to i + 1 mod N
left:  device i sends to i - 1 mod N
```

For a right-moving ring, receiver `r` accumulates sources in this order:

```text
[r, r-1, r-2, ...] mod N
```

For a left-moving ring, receiver `r` accumulates sources in this order:

```text
[r, r+1, r+2, ...] mod N
```

For a full run with `hops = N - 1`, both directions should produce the same
final reduced values because every source appears exactly once. For partial-hop
experiments, direction changes the partial sum.

## What The Pallas Teaching Path Does

The custom function is intentionally built from the Lab 1 primitive:

```text
token = local full chunk vector
accum = token[my_owner_chunk]

for hop in range(hops):
    token = neighbor_copy(token)
    accum += token[my_owner_chunk]

return accum
```

So each hop has three visible ideas:

1. move the whole token one ring step;
2. select the chunk owned by the receiver;
3. add it into that receiver's accumulator.

This is not how the optimized version should move bytes. It is how students can
see the ownership spell before optimizing it into a blur.

## Byte Model

Let:

```text
N = number of devices
B = full input payload bytes per device
C = output chunk bytes per device = B / N
H = number of ring hops
```

For the Lab 6 whole-token teaching implementation:

```text
send bytes per device    = H * B
receive bytes per device = H * B
output bytes per device  = C
```

For the planned one-chunk-per-hop optimized ring:

```text
send bytes per device    = H * C
receive bytes per device = H * C
output bytes per device  = C
```

For the normal full reduce-scatter setting, `H = N - 1`, so:

```text
optimized send bytes per device = (N - 1) / N * B
teaching send bytes per device  = (N - 1) * B
```

The teaching implementation therefore sends about `N` times more data than the
one-chunk-per-hop plan. That is acceptable for this lab because the goal is
ownership clarity. Lab 8 is where performance gets its fangs.

**Reading GB/s across ops:** the `GB/s` column divides each op's own logical
byte model by its time. `pallas_ring_reduce_scatter` is credited the inflated
whole-token traffic (`H * B`) while the built-in `pmap_psum_scatter` is credited
the optimal `2 * B * (N-1) / N`, so the custom path's GB/s looks several times
larger even when it is not moving useful data faster. Do **not** read the GB/s
gap as the teaching kernel beating the built-in — compare the `us` (latency)
column for that, and use GB/s only to watch one op scale across payload sizes.

## Correctness Contract

The default full run should satisfy:

```text
hops = device_count - 1
chunk_count = device_count
output shape = [device_count, tile_rows, tile_cols]
```

And for every device `j`:

```text
output[j, :, :] == sum_s input[s, j, :, :]
```

The updated checker validates the full output tile, not only `output[:, 0, 0]`.
That matters because reduce-scatter copies and reduces whole chunks. A scalar
rank table is a useful debugging window, but it is not a correctness proof.

## Useful Flags

```bash
--neighbor-direction right
--neighbor-direction left
--token-hops 0
--token-hops 1
--token-hops 3
--pallas-collective-id 10
--pallas-memory-space HBM
--pallas-tile-rows 4
--pallas-min-cols 128
```

`--token-hops` is the generic ring-hop flag (shared with Lab 2) and it controls
the custom `pallas_ring_reduce_scatter` op: each value runs as its own case and
`hops=k` reduces `k + 1` sources per chunk. Omitting it (or `None`, `-1`,
`full`, `all`, `n-1`) means `device_count - 1`, the full reduce-scatter. The
built-in `pmap_psum_scatter` reference ignores the flag — it lowers to a single
atomic `lax.psum_scatter` and can only do the full reduction.

## Artifacts To Inspect

Look in the run directory for:

```text
results.jsonl
csvs/results.csv
results_summary.md
lab_artifacts/*lab6_reduce_scatter_spec*
plots/latency_by_payload.png
plots/bandwidth_by_payload.png
logs/console.log
errors/*.txt
traces/* when profiling is enabled
```

The spec artifact is especially important in this lab. It should record:

```text
chunk_count
output_fraction
ring_hops
neighbor_direction
source_history
expected_owner_values_for_teaching_input
whole-token teaching bytes
one-chunk optimized bytes
custom collective status
```

## Suggested Experiments

### 1. Run partial hops

Sweep the hop count against the custom path:

```bash
python collective_bench.py --lab lab6 \
  --ops pallas_ring_reduce_scatter --sizes 64KiB --token-hops 0,1,2,3
```

Each hop value runs as its own case, and correctness is checked against the
*partial* reduce-scatter for that hop count, so every case passes.

Expected lesson:

```text
hops = 0: only local source contributes
hops = 1: local source plus one neighbor contribute
hops = N - 1: every source contributes exactly once
```

### 2. Flip direction

Run the same partial-hop experiment with `--neighbor-direction left` and
`--neighbor-direction right`.

Expected lesson:

```text
full reduce-scatter values are direction-independent
partial reduce-scatter values are direction-dependent
```

### 3. Compare teaching bytes to optimized bytes

Read the spec artifact and compute:

```text
teaching_overhead_factor = whole_token_bytes / one_chunk_bytes
```

Expected lesson:

```text
clarity-first code can be intentionally wasteful
optimization begins by removing unnecessary bytes
```

### 4. Profile the custom path

Capture one XProf trace for `pallas_ring_reduce_scatter` and look for repeated
single-hop phases. The trace should make the teaching composition visible.

## Common Failure Modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Correct for `hops=0`, wrong for `hops=1` | Direction or neighbor map mismatch | Source history table |
| Only `[:, 0, 0]` looks right | Partial tile copy or stale data | Full-tile checker, output dump |
| Full run is wrong but partial source table is right | Wrong owner chunk axis | `owned_chunk(...)` |
| Hangs or crashes | Semaphore or collective-ID issue | Lab 4 rules, `errors/*.txt` |
| Pallas GB/s looks far higher than the built-in | Different byte models, not a real speedup | Byte model note; compare `us`, not GB/s |
| VMEM allocation failure | Whole-token payload too large for VMEM | Use HBM or reduce size |

## Pass Condition

```text
built-in pmap_psum_scatter passes correctness
composed Pallas reduce-scatter matches owner-chunk sums on TPU
full tile contents match expected owner values
the spec artifact records chunk ownership, source history, output fraction, and byte model
no hang-prone broken semaphore variants are introduced
```

## Deferred Work

```text
send one chunk per hop instead of the whole token
fuse local reduce, chunk selection, and remote movement into one Pallas kernel
add per-hop chunk ownership tables for the optimized ring
add integer exactness tests before floating-point tolerance tests
add bidirectional reduce-scatter
record teaching bytes and optimized bytes in every benchmark row
reuse the optimized phase as Lab 7 all-reduce phase 1
```

## Bridge To Lab 7

Lab 7 should reuse the Lab 6 output as phase 1 of all-reduce:

```text
phase 1: reduce-scatter
phase 2: all-gather
```

Once students can explain why device `j` owns reduced chunk `j`, all-reduce is
just the second act: distribute those reduced chunks back to everyone.
