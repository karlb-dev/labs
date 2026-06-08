# Lab 5: Ring All-Gather

Goal: build the first full custom collective from repeated ring movement.

Lab 5 is the first lab where the word **collective** really earns its robe.
Labs 1 and 2 moved one tile or one token around the ring. Lab 5 uses the same
single-hop primitive to make every device collect every device's shard.

The teaching version is intentionally simple:

```text
each device starts with one shard
for each ring hop:
  receive the next shard from one neighbor
  append that shard to my local arrival list
```

This is not yet the optimized fused Pallas all-gather. That is deliberate. The
course goal here is to make the schedule, ownership, and correctness model so
visible that students can debug it with a pencil before they reach for XProf.

## Course Placement

Previous labs gave students the ingredients:

```text
Lab 1: one custom remote-DMA neighbor copy
Lab 2: repeated neighbor copies as a dependency chain
Lab 3: local HBM/VMEM movement
Lab 4: semaphore invariants and failure modes
```

Lab 5 combines those ideas into a full one-direction ring all-gather. Later
labs reuse the same ownership model for reduce-scatter, all-reduce, pipelining,
and topology-aware collectives.

## Mental Model

Each device owns one input shard:

```text
device 0 owns S0
device 1 owns S1
device 2 owns S2
device 3 owns S3
```

For a right-moving ring, every device sends its current shard to the device on
its right. That means device `r` receives from `r - 1 mod N` on every hop.

After `N - 1` hops, device `r` has seen:

```text
[r, r-1, r-2, ..., r-(N-1)] mod N
```

That is **arrival order**, not canonical rank order. Arrival order is the right
layout for this teaching lab because it shows the path through the ring. A
later lab can reorder the result into canonical rank order `[0, 1, 2, ...]` or
fuse the schedule into one lower-level kernel.

## Four-Device Worked Example

Right-moving ring, `N = 4`, default `hops = N - 1 = 3`:

| receiver device | slot 0 | slot 1 | slot 2 | slot 3 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 3 | 2 | 1 |
| 1 | 1 | 0 | 3 | 2 |
| 2 | 2 | 1 | 0 | 3 |
| 3 | 3 | 2 | 1 | 0 |

Left-moving ring, `N = 4`:

| receiver device | slot 0 | slot 1 | slot 2 | slot 3 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 1 | 2 | 3 |
| 1 | 1 | 2 | 3 | 0 |
| 2 | 2 | 3 | 0 | 1 |
| 3 | 3 | 0 | 1 | 2 |

The output tensor shape for the ring path is:

```text
[num_devices, hops + 1, tile_rows, tile_cols]
```

Slot `0` is always the local shard. Slots `1..hops` are shards that arrived
after successive ring hops.

## Implemented Operations

The lab currently exposes these operations through the benchmark harness:

| Operation | Role | What students should compare |
| --- | --- | --- |
| `pmap_ring_all_gather` | Reference ring schedule using repeated `lax.ppermute` | Same arrival-order layout as the custom path |
| `pmap_all_gather` | Built-in executable specification | Canonical all-gather behavior and compiler-managed performance |
| `pallas_ring_all_gather` | Repeated Lab 1 Pallas remote-DMA hop | Custom path with explicit per-hop collective IDs |
| `pallas_all_gather` | Installed JAX Pallas all-gather bridge when available on TPU | A more fused/custom implementation to compare against |
| `lab5_ring_all_gather_spec` | Course artifact | Documents what remains deferred |

The custom Pallas path composes `N - 1` single-hop kernels from Lab 1. Each hop
uses a distinct collective ID:

```text
base_collective_id + hop
```

That keeps semaphore ownership clear and avoids accidentally reusing one
communication phase's barrier/semaphore state for another phase.

## Run

```bash
cd ~/labs/collective_communication
python collective_bench.py --lab lab5
```

Short smoke test:

```bash
python collective_bench.py \
  --lab lab5 \
  --sizes 1KiB,64KiB \
  --iters 10 \
  --warmup 2
```

Profile the custom path:

```bash
python collective_bench.py \
  --lab lab5 \
  --sizes 1MiB,4MiB,16MiB \
  --trace-op pallas_ring_all_gather \
  --trace-size 4MiB \
  --profile
```

Run only the spec artifact if you want the teaching notes without a TPU-heavy
sweep:

```bash
python collective_bench.py \
  --lab lab5 \
  --ops lab5_ring_all_gather_spec \
  --no-plots
```

## Useful Flags

```bash
--neighbor-direction right
--neighbor-direction left
--token-hops 1
--token-hops 3
--pallas-collective-id 50
--pallas-memory-space HBM
--pallas-tile-rows 4
--pallas-min-cols 128
```

The CLI may still call the hop count flag `--token-hops` because Lab 2
introduced the repeated-hop machinery. In Lab 5, read it as **ring hops**.
Default all-gather behavior should use:

```text
hops = num_devices - 1
```

## Correctness Contract

For the default full all-gather case:

```text
every receiver has exactly N arrival slots
every receiver sees every source rank exactly once
every tile in an arrival slot contains the expected source rank
the Pallas ring schedule matches the pmap ring schedule
the built-in all-gather succeeds for the same payload sizes
```

The check intentionally validates the whole output tile, not just
`output[:, :, 0, 0]`. Lab 5 is not a single scalar postcard; the entire shard
must arrive.

## Byte Model

Let:

```text
N = number of devices
H = number of ring hops
B = shard payload bytes per device
```

For the arrival-order ring:

```text
send bytes per device     = H * B
receive bytes per device  = H * B
logical result per device = (H + 1) * B
total ring send bytes     = N * H * B
```

For the normal full all-gather case, `H = N - 1`:

```text
send bytes per device     = (N - 1) * B
receive bytes per device  = (N - 1) * B
logical result per device = N * B
```

This distinction matters in benchmark tables:

- **Payload bytes** describe one local shard.
- **Logical result bytes** describe how much data each device owns after the
  all-gather.
- **Estimated ring bytes** describe traffic induced by this particular
  one-direction algorithm.

The built-in collective may not use exactly this byte schedule. That is one
reason it is a reference and a performance baseline, not a line-by-line copy of
this teaching implementation.

**Reading the GB/s columns:** the headline `GB/s` is *useful* throughput
(optimal-model bytes / time). For a full all-gather the ring is already
bandwidth-optimal — it moves `H * B = (N-1) * B`, exactly the optimal model — so
the custom ring and `pmap_all_gather` are credited the **same** `useful/dev`
bytes and their `GB/s` is directly comparable. Because the ring wastes nothing,
its `wireGB/s` equals its `GB/s` (`byte_model=optimal`); contrast that with Lab 6
and Lab 7, whose whole-token kernels show a `wireGB/s` well above their useful
`GB/s`. The `us` (latency) column is always comparable across ops if you want a
byte-model-free ranking.

## What To Inspect

Start with:

```text
results_summary.md
csvs/results.csv
plots/latency_by_payload.png
plots/bandwidth_by_payload.png
logs/console.log
```

For Lab 5 specifically, look for:

```text
lab_artifacts/*lab5_ring_all_gather_spec*.json
lab_artifacts/*lab5_ring_all_gather_spec*.md
```

With profiling enabled, inspect:

```text
traces/
artifact_index.json
```

A good student report should answer:

```text
At what payload size does latency stop dominating?
How much slower is composed Pallas than built-in all_gather?
Does direction change anything on this topology?
Does the Pallas bridge all-gather behave more like the built-in or the composed ring?
What visible profiler evidence supports the explanation?
```

## Code Tour

Important helpers in `lab5_ring_all_gather.py`:

| Helper | Purpose |
| --- | --- |
| `expected_arrival_ranks(...)` | Pure Python ownership table for arrival order |
| `expected_arrivals(...)` | JAX array version of the expected rank table |
| `ring_byte_model(...)` | Per-device and total byte accounting |
| `ring_all_gather(...)` | Repeated Lab 1 neighbor-copy schedule |
| `canonicalize_arrival_order(...)` | Optional teaching helper to reorder arrival slots by source rank |
| `check_result(...)` | Full-tile correctness check |
| `build_spec(...)` | Writes a lab artifact explaining implemented and deferred work |

## Suggested Experiments

1. **Partial gather:** sweep the hop count and explain the output shape and
   ownership after each value:

   ```bash
   python collective_bench.py --lab lab5 \
     --ops pmap_ring_all_gather,pallas_ring_all_gather \
     --sizes 64KiB --token-hops 0,1,2,3
   ```

   Each hop value runs as its own case; `hops=k` stacks `k + 1` arrivals per
   device, and correctness is checked against that partial gather, so `hops=0`
   (local only) through `hops=N-1` (full gather) all pass. The atomic built-in
   `pmap_all_gather` ignores the hop count by design (it is a single XLA op),
   which is itself worth noting.
2. **Direction flip:** compare `--neighbor-direction right` and
   `--neighbor-direction left`. The rank tables should differ; performance may
   or may not.
3. **Built-in comparison:** compare `pmap_all_gather` against
   `pallas_ring_all_gather`. For a full all-gather both move the optimal bytes,
   so their `GB/s` (useful throughput) and `us` (latency) are both directly
   comparable. On this harness the composed Pallas path usually *wins* at these
   sizes because the `pmap` baseline pays per-call dispatch overhead; explain
   where you would expect the built-in to pull ahead instead (larger payloads).
4. **Payload sweep:** identify the latency-dominated and bandwidth-dominated
   regimes.
5. **Canonical reorder:** use `canonicalize_arrival_order(...)` in a notebook or
   small test to transform arrival order into rank order.

## Common Failure Modes

| Symptom | Likely cause | First place to look |
| --- | --- | --- |
| Output rank table is reversed | Direction convention mismatch | `expected_arrival_ranks(...)` and Lab 1 neighbor direction |
| Device sees duplicate ranks | Too many or too few hops, or wrong neighbor map | `hops`, `direction`, mesh axis |
| Pallas path hangs | Mismatched semaphore/collective ID or broken Lab 1 hop | Lab 4 invariants and `errors/*.txt` |
| Built-in passes but ring fails | Arrival-order schedule bug | Compare against `pmap_ring_all_gather`, not canonical `all_gather` |
| Custom path is much slower | Composed kernels pay per-hop overhead and may hide less from XLA | XProf trace and latency curve |
| VMEM case fails at large sizes | Whole-tile VMEM staging is not the right model yet | Use HBM or reduce payload; chunked VMEM belongs later |

## Pass Condition

```text
every device sees every rank exactly once for H = N - 1
arrival-order pmap ring and Pallas ring agree
the full tile contents match the expected source rank
built-in all-gather passes on the same payloads
the spec artifact records the remaining fused-kernel work
no hang-prone broken variant is added to the default sweep
```

## Deferred Work

These are intentionally left for later labs:

```text
fuse the whole ring schedule into one Pallas kernel
record estimated wire bytes separately from logical bytes in every CSV row
add canonical rank-order output layout as a normal operation
chunk payloads so VMEM staging can overlap local work and remote DMA
add double buffering and capacity semaphores
compare one-direction and bidirectional all-gather/reduce-scatter designs
```

## Bridge To Lab 6

Ring all-gather copies shards until everyone has everything. Reduce-scatter is
the mirror-image skill: move partial sums around the ring until each device owns
only the reduced shard it is responsible for. Lab 5 teaches the arrival and
ownership tables; Lab 6 adds arithmetic to the moving shards.
