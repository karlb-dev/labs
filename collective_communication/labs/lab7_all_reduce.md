# Lab 7: All-Reduce From Reduce-Scatter Plus All-Gather

## Goal

Assemble a full all-reduce from the two ownership transformations students just
built:

```text
reduce-scatter -> all-gather
```

This is the first lab where the course reaches a familiar ML systems primitive:
gradient synchronization. The point is not to beat XLA's built-in collective.
The point is to make all-reduce visible as a schedule with phase boundaries,
ownership changes, synchronization state, layout choices, and a byte model.

## Course Placement

Lab 5 taught this pattern:

```text
one shard per device -> all shards on every device
```

Lab 6 taught this pattern:

```text
one chunk for every owner -> one reduced owner chunk per device
```

Lab 7 snaps those two pieces together:

```text
full input per device
  -> reduce-scatter: each device owns one reduced chunk
  -> all-gather: every device receives every reduced chunk
  -> canonical chunk order: every device has the same final layout
```

Lab 8 will turn this correct composed collective into a performance lab with
chunking, pipelining, buffering, and overlap. Lab 7 is the bridge, so clarity
wins over cleverness.

## Run

```bash
python collective_bench.py --lab lab7
```

Short smoke run:

```bash
python collective_bench.py \
  --lab lab7 \
  --sizes 1KiB,64KiB \
  --iters 10 \
  --warmup 2
```

Profile the composed Pallas path:

```bash
python collective_bench.py \
  --lab lab7 \
  --sizes 4MiB \
  --xprof \
  --profile-cases 1 \
  --trace-op pallas_ring_all_reduce \
  --trace-size 4MiB
```

`--profile`, `--collect-profiles`, and `--xprof` are equivalent profile flags in
this benchmark harness.

## Implemented Happy Path

- `pmap_psum`: built-in all-reduce executable specification.
- `pallas_ring_all_reduce`: Lab 6 reduce-scatter plus Lab 5 all-gather, followed
  by a canonical chunk-order restore.
- `lab7_all_reduce_spec`: course artifact describing phase boundaries, byte
  accounting, and remaining optimized-kernel work.

The custom Pallas path is still a teaching implementation. It intentionally
composes earlier lab primitives instead of hiding the whole collective inside a
single fused kernel.

## Mental Model

All-reduce means:

```text
every device contributes a tensor
sum those tensors across devices
give the full reduced tensor back to every device
```

A bandwidth-aware ring all-reduce usually decomposes that into two phases:

```text
phase 1: reduce-scatter
phase 2: all-gather
```

Phase 1 changes ownership. Phase 2 restores replication.

```text
Before phase 1:
  device i owns its full input tensor split into N owner chunks

After phase 1:
  device j owns reduced chunk j

After phase 2:
  every device owns all reduced chunks
```

Lab 7 uses the Lab 6 whole-token reduce-scatter for phase 1. That means it sends
more bytes than the optimized ring algorithm. This is deliberate: the ownership
transform stays inspectable before Lab 8 brings in the little performance
cauldron.

## Worked Four-Device Example

The teaching input follows the Lab 6 pattern:

```text
x[source_device, owner_chunk, row, col] = 10 * source_device + owner_chunk
```

For four devices, the local source tensors are conceptually:

```text
source 0: [ 0,  1,  2,  3]
source 1: [10, 11, 12, 13]
source 2: [20, 21, 22, 23]
source 3: [30, 31, 32, 33]
```

After reduce-scatter, each device owns one reduced chunk:

```text
owner chunk 0 -> 0  + 10 + 20 + 30 = 60
owner chunk 1 -> 1  + 11 + 21 + 31 = 64
owner chunk 2 -> 2  + 12 + 22 + 32 = 68
owner chunk 3 -> 3  + 13 + 23 + 33 = 72
```

After all-gather and canonical ordering, every device should hold:

```text
[60, 64, 68, 72]
```

That final repeated tensor is the all-reduce result.

## Phase Ledger

For `N` devices and default full phases, both phases use `N - 1` ring hops.

| Phase | Input ownership | Output ownership | Communication primitive | Collective IDs |
| --- | --- | --- | --- | --- |
| Reduce-scatter | each device has all owner chunks for its source | device `j` owns reduced chunk `j` | Lab 6 whole-token ring | `base + 0 ... base + rs_hops - 1` |
| All-gather | each device has one reduced owner chunk | every device has all reduced chunks in arrival order | Lab 5 chunk ring | `base + rs_hops ... base + rs_hops + ag_hops - 1` |
| Canonical reorder | arrival-order chunk slots | canonical source/chunk order | local `shard_map` reorder | no new collective ID |

The reorder is not a network phase. It is a layout normalization step so every
device sees the same chunk order.

## Arrival Order Versus Canonical Order

Lab 5 deliberately returned arrival order because it made the ring schedule easy
to see:

```text
right-moving ring on device r: [r, r-1, r-2, ...] mod N
left-moving ring on device r:  [r, r+1, r+2, ...] mod N
```

That is good for debugging, but all-reduce should return the same final layout
on every device. Lab 7 therefore converts the arrival-order all-gather output
into canonical owner-chunk order:

```text
[chunk 0, chunk 1, ..., chunk N-1]
```

This is a small local operation, but it is a huge conceptual milestone: a
collective algorithm has a data movement schedule and an output layout contract.
Those are not the same thing.

## Byte Model

Let:

```text
N = number of devices
B = full input payload bytes per device
C = one chunk bytes = B / N
H = N - 1 hops per full phase
```

Optimized one-direction ring all-reduce send bytes per device:

```text
reduce-scatter send bytes = H * C = (N - 1) / N * B
all-gather send bytes     = H * C = (N - 1) / N * B
total send bytes          = 2 * (N - 1) / N * B
```

The Lab 7 teaching implementation uses the Lab 6 whole-token reduce-scatter:

```text
teaching reduce-scatter send bytes = H * B
teaching all-gather send bytes     = H * C
teaching total send bytes          = H * B + H * C
```

For `N = 4`, the optimized ring sends `1.5 * B` per device, while the teaching
path sends `3.75 * B` per device. That overhead is not a bug. It is the price
of keeping the phase boundary easy to inspect before optimization.

## Correctness Contract

For a full Lab 7 all-reduce:

```text
output shape: [num_devices, num_devices, tile_rows, tile_cols]
output[receiver, owner_chunk, :, :] == reduced_value_for_owner_chunk
```

For the four-device teaching input, every tile on every receiver should contain:

```text
[60, 64, 68, 72]
```

The updated checker validates the full tile, not just `[:, :, 0, 0]`. The scalar
view is still useful for debugging, but it is not the whole correctness story.
A collective moves tensors, not just rank labels.

## What To Inspect

Look in the run directory:

```text
results_summary.md
results.jsonl
csvs/results.csv
lab_artifacts/*lab7_all_reduce_spec*
plots/latency_by_payload.png
plots/bandwidth_by_payload.png
traces/* when profiling is enabled
errors/*.txt if a case failed
```

Useful questions for the CSV and trace:

```text
Does pallas_ring_all_reduce pass for small and medium payloads?
How much slower is the composed teaching path than pmap_psum?
Does the p50-to-p99 spread grow for larger payloads?
Can you identify the two phases in the trace?
Does the trace show separate hop kernels rather than one fused all-reduce?
```

## Suggested Experiments

1. Run `1KiB`, `64KiB`, `1MiB`, and `4MiB`. Find the size where bandwidth starts
to matter more than latency.
2. Flip `--neighbor-direction left` and verify that the final full all-reduce
result is unchanged after canonical ordering.
3. Compare `pmap_psum` and `pallas_ring_all_reduce`. Explain why the built-in
collective is expected to win.
4. Use the spec artifact's byte model to predict the teaching overhead factor.
Then compare that with measured latency. The ratio will not match perfectly,
and explaining why is the fun part.
5. Temporarily inspect the arrival-order output before canonical reorder. Make a
rank table for a right-moving and left-moving ring.

## Common Failure Modes

| Symptom | Likely cause | Debug move |
| --- | --- | --- |
| Output chunks are right values but wrong order | missing or wrong canonical reorder | print `y[:, :, 0, 0]` before and after reorder |
| Built-in passes but Pallas path fails | phase composition bug | test Lab 5 and Lab 6 independently first |
| Correct for right ring but wrong for left ring | arrival-slot formula assumes one direction | inspect the arrival-order table |
| Hangs or TPU runtime error | repeated collective ID or semaphore mismatch | revisit Lab 4's collective-ID ledger |
| Small payloads dominated by overhead | composed kernels launch many hop phases | compare against `pmap_psum` and inspect trace |
| Large VMEM failure | whole-token teaching path stages too much at once | use HBM for whole-tile remote-copy labs |

## Pass Condition

```text
pmap_psum passes correctness
pallas_ring_all_reduce matches canonical reduced chunks on TPU
full tile contents match, not only the first scalar
lab7_all_reduce_spec records phase boundaries, collective IDs, and byte model
profile run can identify reduce-scatter, all-gather, and canonical reorder work
```

## Deferred To Later Labs

```text
measure reduce-scatter and all-gather phases separately
replace whole-token reduce-scatter with one-chunk-per-hop movement
fuse phase transitions and canonical layout restore
add chunking, double buffering, and overlap
compare one-direction and bidirectional all-reduce
compare flat ring with 2D mesh staging
```

## Bridge To Lab 8

Lab 7 gives you the correct two-phase skeleton. Lab 8 asks the performance
question:

```text
Can we keep the same ownership contract while reducing bubbles and overlapping movement?
```

That means chunks, buffer slots, capacity semaphores, and profiler-guided tuning.
The all-reduce wizard hat is still warm. 🧙‍♂️
