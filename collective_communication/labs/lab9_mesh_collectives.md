# Lab 9: 2D Mesh Collectives

Goal: stop treating every TPU slice as a flat ring. Use a logical `x` by `y`
mesh, stage an all-gather over one axis and then the other, and explain the
result using topology rather than folklore.

This lab is the topology-awareness pivot in the course. Labs 1-8 taught a
single logical ring: one-hop copy, token passing, all-gather, reduce-scatter,
all-reduce, and chunked pipeline planning. Lab 9 asks a new question:

```text
What changes when the devices are not just ranks 0, 1, 2, 3, ...,
but coordinates like (x, y)?
```

The answer is not only performance. The answer is also ownership, layout,
axis order, phase size, and which links the algorithm asks to work at the same
time.

---

## Run

```bash
python collective_bench.py --lab lab9
```

Short smoke run:

```bash
python collective_bench.py \
  --lab lab9 \
  --sizes 1KiB,64KiB \
  --iters 10 \
  --warmup 2
```

Run explicit 2D mesh experiments:

```bash
python collective_bench.py \
  --lab lab9 \
  --lab9-mesh-shape 2x2 \
  --lab9-axis-order x_then_y

python collective_bench.py \
  --lab lab9 \
  --lab9-mesh-shape 2x2 \
  --lab9-axis-order y_then_x
```

Profile the custom Pallas path:

```bash
python collective_bench.py \
  --lab lab9 \
  --sizes 4MiB \
  --lab9-mesh-shape 2x2 \
  --lab9-axis-order x_then_y \
  --trace-op pallas_2d_staged_all_gather \
  --profile
```

---

## Implemented Happy Path

- `pmap_all_gather`: built-in baseline and executable specification.
- `pmap_2d_staged_all_gather`: CPU-runnable staged-mesh reference using
  repeated `lax.ppermute`.
- `pallas_2d_staged_all_gather`: TPU logical-neighbor remote-DMA path.
- `lab9_mesh_collectives_spec`: course artifact that records mesh shape,
  axis-order, byte model, and schedule tables.

The custom Pallas path keeps the physical `shard_map` mesh flat, then computes
logical 2D coordinates inside the kernel. That means the global device list is
still one-dimensional to JAX, but Lab 9 overlays this coordinate system:

```text
rank = x_coord * y_size + y_coord
```

For four local devices with `--lab9-mesh-shape 2x2`:

```text
rank 0 -> (x=0, y=0)
rank 1 -> (x=0, y=1)
rank 2 -> (x=1, y=0)
rank 3 -> (x=1, y=1)
```

That coordinate map is the compass. Without it, staged collectives are just
rank soup.

---

## Mental Model

A flat one-direction all-gather says:

```text
every device sends one shard around one ring until every device has all shards
```

A 2D staged all-gather says:

```text
stage 1: gather along one mesh axis
stage 2: move those partial gathers along the other mesh axis
stage 3: restore canonical rank order
```

For `x_then_y`:

```text
first gather along x
then gather the x-partials along y
```

For `y_then_x`:

```text
first gather along y
then gather the y-partials along x
```

Both produce the same logical result:

```text
output[receiver_rank, source_rank, ...] = input[source_rank, ...]
```

The difference is the schedule. The two axis orders can use different link
groups, place large partial payloads on different axes, and expose different
contention patterns. That is the dragon under the floorboards.

---

## A Four-Device Worked Example

Use a `2x2` logical mesh:

```text
      y=0     y=1
x=0   r0      r1
x=1   r2      r3
```

For `x_then_y`, `direction=right`:

### Stage 1: gather along x

Each fixed-`y` column gathers between its two `x` coordinates.

```text
y=0 column: r0 <-> r2
y=1 column: r1 <-> r3
```

After this stage:

```text
r0 has sources [0, 2]
r1 has sources [1, 3]
r2 has sources [2, 0]
r3 has sources [3, 1]
```

The order is arrival order, not canonical rank order.

### Stage 2: move the partial gathers along y

Each fixed-`x` row now exchanges the partial bundles.

```text
x=0 row: r0 <-> r1
x=1 row: r2 <-> r3
```

After this stage, every device has all four sources, but still in staged arrival
layout.

### Stage 3: canonical restore

The lab reorders the staged arrival tensor so the final result is:

```text
source slot 0 -> rank 0
source slot 1 -> rank 1
source slot 2 -> rank 2
source slot 3 -> rank 3
```

This final restore is important. Lab 5 intentionally exposed ring arrival order
because that was the useful debugging view. Lab 9 teaches a production-shaped
view: the schedule may be staged, but the API-facing result should be canonical.

---

## Direction Convention

Lab 9 reuses the course convention:

```text
right = +1 along the active logical mesh coordinate, with wraparound
left  = -1 along the active logical mesh coordinate, with wraparound
```

So if the active axis is `x`, `right` means:

```text
(x, y) -> ((x + 1) mod x_size, y)
```

If the active axis is `y`, `right` means:

```text
(x, y) -> (x, (y + 1) mod y_size)
```

This is a logical convention for the lab. The physical TPU torus may not match
this layout perfectly. That mismatch is exactly why the spec artifact records
both logical schedules and runtime topology evidence.

---

## Shape And Correctness Contract

Each device starts with a tile whose values encode both source rank and position
inside the tile:

```text
x[source, row, col] = source + 0.25 * row + (1 / 32) * (col mod 32)
```

So `x[source, 0, 0]` is exactly the source rank, and every other element is
slightly different. The exact constants are not sacred. They are a debugging
trick. Rank-only tiles
can hide partial-copy bugs because every element in a tile is identical. A
rank-plus-position pattern makes stale rows, wrong columns, accidental
broadcasts, and partial remote copies easier to catch.

The final result should satisfy:

```text
y[receiver, source, row, col] == x[source, row, col]
```

for every receiver, every source, and every tile element.

Pass condition:

```text
the built-in baseline passes
pmap staged all-gather matches canonical rank order
Pallas staged all-gather matches canonical rank order on TPU
the full tile is validated, not only y[:, :, 0, 0]
the spec artifact records candidate mesh shapes and axis-order experiments
```

---

## Byte Model

Let:

```text
N  = x_size * y_size
B  = input payload bytes per device
A1 = size of the first staged axis
A2 = size of the second staged axis
```

The logical output per device is:

```text
logical result bytes per device = N * B
```

The staged one-direction all-gather sends:

```text
stage 1 send bytes per device = (A1 - 1) * B
stage 2 send bytes per device = (A2 - 1) * A1 * B
staged send bytes per device  = (N - 1) * B
```

The idealized per-device send-byte count matches a flat one-direction ring:

```text
flat ring send bytes per device = (N - 1) * B
```

So why stage at all?

Because bytes are not the whole story. Axis staging changes:

```text
which logical links are active
which phase sends small shards versus partial bundles
which axis carries the larger second-stage payload
how much contention the physical topology sees
how easily the algorithm extends to hierarchy later
```

This lab is not claiming that staged always wins. It is teaching students how to
ask better questions of the profiler.

---

## Artifacts To Inspect

Look for:

```text
results.jsonl
csvs/results.csv
plots/latency_by_payload.png
plots/bandwidth_by_payload.png
run_metadata.json
diagnostics/runtime.json
lab_artifacts/*lab9_mesh_collectives_spec*
traces/*, when profiling is enabled
```

In the spec artifact, inspect:

```text
configured_mesh_shape
configured_axis_order
candidate_2d_mesh_shapes
rank_layout
axis_groups
configured_stage_plan
canonical_lookup_preview
configured_byte_model
byte_model_by_axis_order
student_checkpoint_questions
```

The important profiler question is not only “which one is faster?” Ask:

```text
Does x_then_y move the large partial bundle over a different physical axis than y_then_x?
Are the trace durations dominated by stage 1, stage 2, or layout restore?
Does the built-in all_gather appear to use a strategy that your staged model did not try?
Does the Pallas path lose to built-in because launch overhead dominates?
```

---

## Suggested Experiments

### 1. Axis-order sweep

```bash
python collective_bench.py \
  --lab lab9 \
  --sizes 1MiB,4MiB,16MiB \
  --lab9-mesh-shape 2x2 \
  --lab9-axis-order x_then_y

python collective_bench.py \
  --lab lab9 \
  --sizes 1MiB,4MiB,16MiB \
  --lab9-mesh-shape 2x2 \
  --lab9-axis-order y_then_x
```

Question:

```text
Do the two schedules behave the same on your slice? If not, what evidence says why?
```

### 2. Direction sweep

```bash
python collective_bench.py \
  --lab lab9 \
  --lab9-mesh-shape 2x2 \
  --neighbor-direction right

python collective_bench.py \
  --lab lab9 \
  --lab9-mesh-shape 2x2 \
  --neighbor-direction left
```

Question:

```text
Does reversing logical direction matter, or does the physical topology make it symmetric?
```

### 3. Non-square logical mesh

On eight local devices, try:

```bash
python collective_bench.py \
  --lab lab9 \
  --lab9-mesh-shape 2x4

python collective_bench.py \
  --lab lab9 \
  --lab9-mesh-shape 4x2
```

Question:

```text
Does the larger second-stage partial move along the axis you expected?
```

### 4. Compare against Lab 5

Run Lab 5 and Lab 9 with the same payload sizes.

Question:

```text
Is a custom staged Pallas path faster than the composed Lab 5 ring, or did it only produce a nicer story?
```

A disappointing measurement is still a good measurement if it teaches the right
thing. Tiny silicon goblin, honest clipboard.

---

## Common Failure Modes

### Wrong logical coordinate map

Symptom:

```text
correctness fails; output contains all sources but canonical source slots are wrong
```

Fix:

```text
check rank = x * y_size + y and verify the coordinate table in the spec artifact
```

### Axis-order mismatch

Symptom:

```text
x_then_y pmap reference passes but Pallas path fails, or vice versa
```

Fix:

```text
compare axis_sequence(), mesh_axis_perm(), and logical_mesh_neighbor_rank()
```

### Direction mismatch

Symptom:

```text
right-moving schedule is checked as if it were left-moving
```

Fix:

```text
use the canonical index table; do not reason from memory after the third cup of coffee
```

### Partial-copy bug hidden by rank-only input

Symptom:

```text
y[:, :, 0, 0] looks correct but full tensor differs
```

Fix:

```text
use the rank-plus-position payload and full-tile check_result()
```

### VMEM whole-tile pressure

Symptom:

```text
Pallas compile failure or low-level memory-space error for large payloads
```

Fix:

```text
use HBM/ANY memory space for whole-tile teaching kernels, or reduce --sizes
```

### Built-in baseline wins

Symptom:

```text
custom Pallas staged path is slower than pmap_all_gather
```

Fix:

```text
this is not automatically a bug; inspect launch overhead, fusion loss, and whether XLA chose a better collective schedule
```

---

## Student Checkpoints

Students should be able to answer:

```text
What logical coordinate does rank 3 have in a 2x2 mesh?
For x_then_y, which devices communicate in stage 1?
For y_then_x, which devices communicate in stage 1?
Why can staged and flat-ring all-gather have the same ideal send-byte count?
Why might they still perform differently?
Why does the Pallas implementation use a flat shard_map mesh but logical 2D math inside the kernel?
What does canonical layout restore do?
Which artifact would you inspect first when x_then_y and y_then_x differ?
```

---

## Deferred Work

```text
add staged reduce-scatter over the same logical mesh
add staged all-reduce over the same logical mesh
compare flat-ring, x-then-y, and y-then-x traces against physical topology evidence
add true 3D staged collectives for larger v4 slices
extend staged algorithms across host boundaries in Lab 10
```

Lab 10 will move from local topology to multi-host run control and hierarchy:
process-local device groups, global device groups, process-index-aware logs, and
intra-host versus inter-host phases.
