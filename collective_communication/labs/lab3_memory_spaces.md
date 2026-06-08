# Lab 3: Pallas Memory Spaces and Local DMA

Goal: understand the local memory path inside a TPU Pallas kernel before adding
more complex remote communication.

Lab 1 gave us one custom remote hop. Lab 2 repeated that hop into a dependency
chain. Lab 3 turns the network off. Every device keeps its own local tile and
studies what happens inside one TPU device when a Pallas kernel stages data
through local scratch memory.

The custom kernel performs:

```text
HBM input -> VMEM input scratch -> arithmetic -> VMEM output scratch -> HBM output
```

The arithmetic is intentionally boring:

```text
y = x.astype(float32) * scale + bias
```

The useful part is the path. Later labs combine this same HBM/VMEM staging
pattern with remote DMA, chunking, double buffering, and semaphore flow control.
This is the local-memory lantern before the collective tunnel.

## Course Placement

This lab is the memory-hierarchy checkpoint in the 10-lab sequence.

```text
Lab 1: one remote hop
Lab 2: repeated remote hops with a dependency chain
Lab 3: local memory spaces, local async copy, and VMEM scratch
Lab 4: semaphore bug zoo
Lab 5+: full collectives and pipelined communication
```

After this lab, students should stop treating a Pallas kernel as just "JAX but
lower." They should know where a value lives, how it moves, and why copying a
block into VMEM is different from ordinary array indexing.

## Mental Model

Each device owns one local tile:

```text
x_i = tile owned by device i
```

For the simplest rank-valued pattern on four devices, the tiles would look like:

```text
device 0: all 0s
device 1: all 1s
device 2: all 2s
device 3: all 3s
```

The updated Python uses a slightly richer rank-plus-position pattern so the full
tile checker can catch partial-copy mistakes:

```text
x[i, row, col] = i + row * 0.25 + (col mod 8) * 0.03125
```

The first scalar on each device still equals the rank, so the lab remains easy
to inspect by eye.

The Pallas kernel runs independently on each device. There is no all-reduce, no
all-gather, no ppermute, no neighbor copy, no entry barrier, and no cross-device
semaphore. (If you do see an `entry_barrier` / `barrier-cores` entry in
`traces/trace_comm_summary.json`, that is generic XLA SPMD scaffolding emitted by
`shard_map`, not a barrier authored by this lab.) The only movement is local:

```text
local HBM on device i
  -> local VMEM input scratch on device i
  -> local VMEM output scratch on device i
  -> local HBM output on device i
```

The expected output is:

```text
y[i, row, col] = float32(x[i, row, col]) * scale + bias
```

## Memory Vocabulary

Use these terms precisely during the lab:

| Term | Meaning in this lab |
|---|---|
| HBM | Device memory backing the input and output arrays. Large, global to the device, higher latency than local scratch. |
| VMEM | Vector memory used as local scratch inside the Pallas kernel. The lab allocates one input scratch tile and one output scratch tile. |
| SMEM | Scalar memory. This lab does not allocate SMEM, but later labs use the idea for scalar control and prefetch metadata. |
| DMA semaphore | Completion tracker for an async copy. Lab 3 uses one for HBM -> VMEM and one for VMEM -> HBM. |
| VREG | Vector registers used by compute. Students do not allocate these directly, but reading from VMEM into expressions feeds vector compute. |

The key teaching distinction:

```text
HBM/ANY refs are not ordinary fast local arrays.
VMEM refs are scratch arrays the kernel can index and compute from directly.
```

## Run

```bash
cd ~/labs/collective_communication
python collective_bench.py --lab lab3
```

Short smoke run:

```bash
python collective_bench.py \
  --lab lab3 \
  --sizes 1KiB,64KiB \
  --iters 10 \
  --warmup 2
```

Profile the Pallas VMEM path:

```bash
python collective_bench.py \
  --lab lab3 \
  --sizes 4MiB \
  --profile \
  --profile-cases 1 \
  --trace-op pallas_vmem_arith \
  --trace-size 4MiB
```

Optional HLO/XLA dumps:

```bash
python collective_bench.py \
  --lab lab3 \
  --sizes 64KiB \
  --xla-dump-to run
```

## What This Lab Teaches

- How to name the local memory spaces used by TPU Pallas kernels.
- How an HBM-backed input is staged into a VMEM scratch buffer.
- How a local async copy uses a DMA semaphore, even without remote DMA.
- Why output dtype matters for byte accounting.
- Why tile shape is part of the kernel contract.
- Why local memory bandwidth and remote collective bandwidth should not be
  mixed in the same mental bucket.
- How named scopes in a teaching kernel become breadcrumbs in profiler traces.

## Implementations

The benchmark harness should compare two operations:

| Operation | Purpose |
|---|---|
| `pmap_local_arith` | Reference path using normal JAX arithmetic under `pmap`. |
| `pallas_vmem_arith` | Teaching kernel that explicitly stages through VMEM with local async copies. |

The Pallas implementation has no cross-device communication. That is deliberate.
Students should be able to inspect a profile and separate local memory movement
from the remote DMA work introduced in Labs 1 and 2.

## Code Walkthrough

The core file has four conceptual layers.

### 1. Case construction

`build_case(...)` creates a patterned tile on the host, shards it across the
device mesh, creates the expected float32 result, and returns a small case
object for the benchmark harness.

```text
input dtype  = benchmark dtype, often bf16 or float32
output dtype = float32
shape        = [num_devices, rows, cols]
```

### 2. Reference computation

`local_arith_reference(...)` computes:

```text
value.astype(float32) * scale + bias
```

This is the correctness oracle. It is not the performance target.

### 3. Pallas kernel

`vmem_arith_kernel(...)` does three named steps:

```text
lab3_hbm_to_vmem:  make_async_copy(...).start(); wait()
lab3_vmem_compute: output_scratch_ref[...] = input_scratch_ref[...] * scale + bias
lab3_vmem_to_hbm:  make_async_copy(...).start(); wait()
```

The waits are part of the lesson. They create the local ordering:

```text
copy input before reading scratch
finish compute before copying output
finish output copy before leaving the kernel
```

### 4. `shard_map` wrapper

`vmem_arith(...)` uses `shard_map` so each device runs the same local Pallas
program on its own shard. `shard_map` is present because the course is about
multi-device programs, but this lab uses it only for per-device placement, not
for communication.

## Byte Model

Let:

```text
R = tile_rows
C = tile_cols
I = input dtype itemsize
O = output dtype itemsize = 4 bytes for float32
```

Then per device:

```text
input_payload_bytes      = R * C * I
output_payload_bytes     = R * C * O
logical_bytes_per_device = input_payload_bytes + output_payload_bytes
local_dma_bytes          = input_payload_bytes + output_payload_bytes
remote_bytes             = 0
vmem_scratch_bytes       = input_payload_bytes + output_payload_bytes
```

For `bf16 -> float32`, output bytes are twice the input bytes. A 4 MiB bf16
input tile produces an 8 MiB float32 output tile, so the two VMEM scratch tiles
hold about 12 MiB total per device.

This lab's bandwidth row should be read as local memory movement plus local
compute, not as network or collective bandwidth.

## Tile Shape Notes

The lab uses a whole local tile. The default shape is controlled by:

```bash
--pallas-tile-rows 4
--pallas-min-cols 128
```

For a requested payload size, the code chooses enough columns to hold at least
that many input bytes:

```text
cols = max(min_cols, ceil(requested_payload_bytes / (dtype_size * rows)))
```

The benchmark should report both requested bytes and actual bytes because tile
rounding is part of low-level kernel work. The rounded shape is not bookkeeping
fluff, it is the kernel contract.

TPU Pallas kernels have block-shape and layout constraints, especially around
the last two dimensions. If a shape fails to compile, do not treat it as random
compiler grumbling. Inspect the tile shape first.

## Useful Flags

```bash
--pallas-tile-rows 4
--pallas-min-cols 128
--lab3-scale 1.5
--lab3-bias 0.25
```

Good experiments:

```bash
# Tiny payloads: fixed overhead dominates.
python collective_bench.py \
  --lab lab3 \
  --sizes 1KiB,4KiB,16KiB \
  --iters 20 \
  --warmup 5

# Medium payloads: local bandwidth starts to matter.
python collective_bench.py \
  --lab lab3 \
  --sizes 64KiB,256KiB,1MiB,4MiB

# Shape experiment: keep bytes similar but alter rows and columns.
python collective_bench.py \
  --lab lab3 \
  --sizes 1MiB \
  --pallas-tile-rows 1

python collective_bench.py \
  --lab lab3 \
  --sizes 1MiB \
  --pallas-tile-rows 8
```

## Artifacts To Inspect

Each run creates a directory under `runs/`.

Start with:

- `results_summary.md`
- `csvs/results.csv`
- `plots/latency_by_payload.png`
- `plots/bandwidth_by_payload.png`
- `logs/console.log`

For profiles, inspect:

- `traces/`
- the `lab3_hbm_to_vmem`, `lab3_vmem_compute`, and `lab3_vmem_to_hbm` named
  scopes (see the note below on where they actually surface)

Note: a Pallas kernel lowers to a single fused custom-call, so these
`jax.named_scope` labels do **not** appear as separate bars on the device
timeline. They live in the kernel's HLO metadata and in XProf's source view
(which maps device ops back to lines in `lab3_memory_spaces.py`), not as distinct
timeline regions. Treat them as source-level breadcrumbs, not timeline events.

For compile/runtime failures, inspect:

- `errors/*.txt`
- `logs/stderr.log`
- `diagnostics/runtime.json`

## Correctness Invariant

For every device and every element:

```text
y = x.astype(float32) * scale + bias
```

The checker validates the full tile, not just `y[:, 0, 0]`. That matters because
Lab 3 is a memory movement lab. A partial copy that preserves the first scalar
is still wrong.

## Common Failure Modes

| Symptom | Likely cause | What to inspect |
|---|---|---|
| Pallas compile error about memory or scratch | VMEM scratch tile is too large | requested size, actual input bytes, output bytes |
| Shape or layout compile error | Tile shape violates TPU Pallas constraints or is inefficient | `tile_rows`, `tile_cols`, `pallas_min_cols` |
| Correct first scalar, wrong rest of tile | Partial write or wrong scratch shape | full-tile checker, `output_scratch_ref[...]` |
| Output dtype mismatch | Reference and kernel disagree about float32 output | `out_shape`, expected dtype, CSV dtype columns |
| Pallas slower than JAX reference | Tiny arithmetic and explicit staging overhead | payload-size curve and profile trace |

## Student Questions

1. Why does this lab use `float32` output even when the input is `bf16`?
2. What bytes are counted as input payload bytes? What bytes are counted as
   output payload bytes?
3. Why is `pallas_vmem_arith` not a collective communication benchmark?
4. What would break if the input copy did not wait before the compute phase?
5. Why does a local async copy need a DMA semaphore?
6. Which profile region looks most expensive for tiny payloads? Which region
   dominates for large payloads?
7. How would this kernel need to change to handle a 64 MiB tile without trying
   to place the whole input and output scratch in VMEM at once?

## Pass Criteria

Students should be able to:

- Run `--lab lab3` successfully for small and medium sizes.
- Explain HBM, VMEM, SMEM, VREG, and DMA semaphore roles.
- Explain why the Pallas kernel has no cross-device traffic.
- Read a benchmark row and compute input bytes, output bytes, and total local
  logical bytes.
- Explain why full-tile correctness is stronger than checking one scalar.
- Locate the `lab3_hbm_to_vmem`, `lab3_vmem_compute`, and `lab3_vmem_to_hbm`
  scopes in the kernel source / XProf source view, and explain why they do not
  appear as separate device-timeline bars (the fused custom-call folds them in).

## Bridge To Lab 4

Lab 3 uses semaphores only for local DMA completion. Lab 4 should intentionally
break synchronization in tiny examples: missing wait, over-wait, over-signal,
wrong entry barrier, wrong `collective_id`, and buffer-slot races. After Lab 3,
students know what a correct local wait looks like before they start meeting the
semaphore goblins in the basement.
