# Lab 1: Custom Single-Hop Neighbor Copy

Lab 1 is the course's first "hello, wire" lab. The goal is to write, run, validate, and profile the smallest useful custom TPU communication primitive: every device sends one local tile to one logical neighbor.

This lab is intentionally narrow. There is no reduction, no multi-hop chain, no chunking, and no overlap yet. Those show up later. Here we want the communication skeleton to be visible:

```text
topology + ownership + data movement + synchronization + measurement
```

Concept source:

```text
labs/lab1_single_hop.py
```

The benchmark harness, logging, plotting, CSV/JSONL output, diagnostics, and profile capture live in:

```text
collective_bench.py
```

The lab file should remain small enough to read while learning the primitive, even if it contains generous comments.

---

## Where This Fits In The 10-Lab Course

This course is about moving from "I can call a collective" to "I can reason about the data movement schedule, synchronization protocol, memory hierarchy, topology, and profiler trace."

Lab 1 is the first custom communication brick. It prepares the ground for:

| Later lab | How Lab 1 feeds it |
|---|---|
| Lab 2: token-passing ring | Repeat this one-hop copy several times and introduce dependency chains. |
| Lab 3: Pallas memory spaces | Study HBM, VMEM, SMEM, and local async copies in isolation. |
| Lab 4: semaphore bug zoo | Break the synchronization rules deliberately and learn the failure modes. |
| Lab 5: ring all-gather | Use repeated neighbor copies to build a full collective. |
| Lab 6: reduce-scatter | Add ownership of chunks and local reduction. |
| Lab 7: all-reduce | Compose reduce-scatter plus all-gather. |
| Lab 8: pipelined ring | Split whole-tile copies into chunks and overlap transfers. |
| Lab 9: 2D mesh collectives | Replace a flat ring with topology-aware staged movement. |
| Lab 10: multi-host hierarchy | Apply the same invariants across process and host boundaries. |

The important habit starts here: every communication lab must state who owns each byte before and after the operation.

---

## The Communication Pattern

Each device owns one local tile. For the default right-neighbor case, each device pushes its tile to the next device in a logical ring:

```text
device 0 -> device 1
device 1 -> device 2
device 2 -> device 3
device 3 -> device 0
```

For `N` devices:

```text
source i sends to destination (i + 1) mod N
```

That means output device `i` should contain the rank value from:

```text
(i - 1) mod N
```

For four devices, the expected result is:

| Output device | Receives from | Expected first value |
|---:|---:|---:|
| 0 | 3 | 3 |
| 1 | 0 | 0 |
| 2 | 1 | 1 |
| 3 | 2 | 2 |

With `--neighbor-direction left`, the arrows reverse:

```text
source i sends to destination (i - 1) mod N
```

So output device `i` receives from:

```text
(i + 1) mod N
```

---

## Reference Operation

The reference operation is `lax.ppermute`, the XLA-managed collective-permute style baseline. Students should treat the built-in collective as an executable specification:

```text
custom Pallas remote DMA result == lax.ppermute result
```

This is an important course pattern. Before building a custom collective, write down or run the built-in operation that defines the desired ownership transformation.

---

## Learning Objectives

By the end of Lab 1, students should be able to:

1. Explain the logical ring and the difference between "send right" and "receive from left."
2. Describe how a `jax.sharding.Mesh` and `PartitionSpec(axis_name)` create one local shard per device.
3. Explain why the global array shape is `(num_devices, tile_rows, tile_cols)` but the Pallas kernel sees the local shard, normally `(1, tile_rows, tile_cols)` in this representation.
4. Identify the source ref, destination ref, send semaphore, and receive semaphore in a remote DMA.
5. Explain why the entry barrier is present before the remote write.
6. Explain why every send has exactly one matching receive.
7. Explain why all semaphores must drain before kernel completion.
8. Compare `pallas_neighbor_copy` against `pmap_ppermute` across payload sizes.
9. Read the run artifacts and connect one CSV row to one plotted point and one profiler trace.
10. Say, without embarrassment, when the custom kernel is slower than the built-in collective.

That last point matters. A custom communication kernel is a microscope, not automatically a race car.

---

## Mental Model

A single-hop custom copy has five moving parts.

### 1. Topology

The lab builds a one-dimensional logical mesh:

```python
mesh = jax.sharding.Mesh(np.array(devices), (axis_name,))
```

The logical ring order comes from the order of `devices` in that mesh. This is not yet a topology optimizer. Later labs will ask whether this logical order matches physical TPU coordinates.

### 2. Ownership

Before the copy:

```text
output device i does not yet own neighbor data
input shard i contains rank i
```

After a right-neighbor copy:

```text
output shard i contains rank (i - 1) mod N
```

The input tiles are filled with rank constants. A one-element check is enough to see the neighbor mapping, but the updated lab checks the full output tile to catch partial-copy bugs.

### 3. Movement

The Pallas kernel issues one remote DMA:

```text
local x_ref on source device -> remote o_ref on destination device
```

This is a push model. The source device writes to the destination. The destination is not issuing a remote read.

### 4. Synchronization

The lab uses:

```text
barrier semaphore: entry ordering before remote writes
send DMA semaphore: source-side DMA progress
recv DMA semaphore: destination-side DMA progress
```

This lab waits immediately after starting the DMA. Later labs will split start and wait so they can overlap communication with compute or with other communication.

### 5. Measurement

The harness should separate compile time from run time, warm up the operation, use `block_until_ready()` during timing, and report distributions rather than a single lonely average.

---

## Run Commands

Run the default Lab 1 sweep:

```bash
cd ~/labs/collective_communication
python collective_bench.py --lab lab1
```

Run a short smoke test:

```bash
python collective_bench.py \
  --lab lab1 \
  --sizes 1KiB,64KiB \
  --iters 10 \
  --warmup 2
```

Run a payload sweep that makes latency and bandwidth regimes more visible:

```bash
python collective_bench.py \
  --lab lab1 \
  --sizes 1KiB,4KiB,16KiB,64KiB,256KiB,1MiB,4MiB,16MiB \
  --iters 50 \
  --warmup 5
```

Reverse the ring direction:

```bash
python collective_bench.py \
  --lab lab1 \
  --sizes 1KiB,64KiB,4MiB \
  --neighbor-direction left
```

Run a tiny VMEM experiment:

```bash
python collective_bench.py \
  --lab lab1 \
  --sizes 1KiB,16KiB,64KiB \
  --pallas-memory-space VMEM
```

Profile one custom Pallas case:

```bash
python collective_bench.py \
  --lab lab1 \
  --sizes 4MiB \
  --profile \
  --profile-cases 1 \
  --trace-op pallas_neighbor_copy \
  --trace-size 4MiB
```

Equivalent profiling flags:

```text
--profile
--collect-profiles
--xprof
```

Dump HLO/XLA artifacts for a tiny case:

```bash
python collective_bench.py \
  --lab lab1 \
  --sizes 1KiB \
  --iters 5 \
  --xla-dump-to run
```

---

## What To Inspect

Each run creates a directory under:

```text
runs/
```

Start here:

| Artifact | Why it matters |
|---|---|
| `results_summary.md` | Human-readable summary of cases, status, and timing. |
| `csvs/results.csv` | Main spreadsheet-friendly benchmark table. |
| `results.jsonl` | One machine-readable row per case. |
| `plots/latency_by_payload.png` | Shows latency floor and payload-size regimes. |
| `plots/bandwidth_by_payload.png` | Shows bandwidth scaling as payload grows. |
| `logs/console.log` | Combined stdout/stderr, including TPU/XLA messages. |
| `diagnostics/runtime.json` | JAX device report and runtime metadata. |
| `artifact_index.json` | Map of all generated artifacts. |

If compilation or execution fails, inspect:

```text
errors/*.txt
logs/stderr.log
diagnostics/runtime.json
```

If profiling is enabled, inspect:

```text
traces/
traces/trace_comm_summary.json
plots/trace_comm_time.png
```

Open the trace in XProf (`tensorboard --logdir <run>/traces`, then Trace Viewer)
or drag the `traces/.../*.trace.json.gz` into https://ui.perfetto.dev. On each
`/device:TPU:*` row, the single hop shows up as these runtime events:

```text
copy.<n>            the remote DMA (the actual neighbor hop)
barrier-cores       the entry barrier
Acquire semaphore   DMA / barrier semaphore wait
Release semaphore   DMA / barrier semaphore signal
```

Note: the kernel's `jax.named_scope` labels (`lab1_entry_barrier`,
`lab1_remote_dma`) do not appear as separate timeline bars. A Pallas kernel
lowers to one fused custom-call, so those names live in the kernel's HLO
metadata and XProf's source view (which maps the device ops back to lines in
`lab1_single_hop.py`), not on the device timeline. The harness also distills the
trace into `traces/trace_comm_summary.json` and `plots/trace_comm_time.png`,
which show mean per-device DMA vs barrier vs semaphore time — for a tiny payload
you should see the entry barrier dominate the actual copy.

---

## Columns Worth Reading In The CSV

Depending on the current harness version, the exact column names may vary, but students should look for these concepts:

| Concept | Meaning |
|---|---|
| `op` | Built-in baseline or custom Pallas implementation. |
| `requested_payload_bytes` | Payload requested on the CLI. |
| `payload_bytes` or `actual_payload_bytes` | Actual whole-tile bytes copied per device. |
| `latency_p50_us` | Median measured execution time. |
| `latency_p90_us` | Tail-ish latency, useful for seeing noise. |
| `effective_logical_GBps` | Throughput computed from logical payload bytes. |
| `ok` or `status` | Correctness and execution status. |
| `note` | Direction, tile shape, and other case notes. |

The most common surprise is that `requested_payload_bytes` can differ from actual copied bytes. The tile is rounded to a shape that the Pallas kernel can copy as one whole local tile.

---

## Memory Space Notes

Lab 1 defaults to HBM-like whole-tile copies because broad payload sweeps include sizes too large for scoped VMEM.

Useful flags:

```bash
--pallas-memory-space HBM
--pallas-memory-space VMEM
--pallas-tile-rows 4
--pallas-min-cols 128
```

Use `VMEM` only for small payload experiments in this lab. If a VMEM whole-tile copy exceeds the conservative Lab 1 limit, the lab should fail early with an actionable error. That is intentional. Chunked VMEM staging belongs in a later lab where students can reason about local copies, buffer counts, and overlap.

---

## Correctness Invariants

For `direction='right'`:

```text
observed_rank_at_output_device_i == (i - 1) mod N
```

For `direction='left'`:

```text
observed_rank_at_output_device_i == (i + 1) mod N
```

Every run should satisfy:

```text
every source sends exactly once
every destination receives exactly once
every DMA has one send semaphore and one receive semaphore
every barrier wait has a matching signal
every semaphore drains before kernel completion
the custom result matches the built-in reference
```

These are the little iron laws. Later labs will break them on purpose so students can learn how failures look.

---

## Student Exercises

### Exercise 1: Draw The Ownership Table

For `N = 4` and `direction='right'`, fill in:

```text
source -> destination
output device -> expected source
```

Then repeat for `direction='left'`.

### Exercise 2: Reverse The Ring

Run:

```bash
python collective_bench.py \
  --lab lab1 \
  --sizes 1KiB,64KiB,4MiB \
  --neighbor-direction left
```

Answer:

```text
Did correctness still pass?
Which expected-rank formula changed?
Did timing change meaningfully?
```

### Exercise 3: Built-In Versus Custom

Compare `pmap_ppermute` and `pallas_neighbor_copy` for:

```text
1KiB
64KiB
4MiB
16MiB
```

Answer:

```text
Which operation wins at small payloads?
Which operation wins at large payloads?
Does the custom kernel ever win?
What fixed costs seem visible?
```

A perfectly valid result is that the built-in collective wins. The lesson is not "custom is faster." The lesson is "now I can see the machinery."

### Exercise 4: VMEM Boundary

Run a small VMEM case, then try a too-large VMEM case.

Answer:

```text
What error did the lab produce?
Why is early failure better than an opaque TPU runtime failure?
Which later lab should handle large staged VMEM transfers?
```

### Exercise 5: Inspect The Trace

Capture a trace for the custom case:

```bash
python collective_bench.py \
  --lab lab1 \
  --sizes 4MiB \
  --profile \
  --profile-cases 1 \
  --trace-op pallas_neighbor_copy \
  --trace-size 4MiB
```

On the TPU device timeline, find:

```text
copy.<n>         the remote DMA hop
barrier-cores    the entry barrier
Acquire/Release semaphore
```

Then open `plots/trace_comm_time.png` and `traces/trace_comm_summary.json`.

Answer:

```text
Can you identify the communication phase (which event is the actual copy)?
For a tiny payload, is the entry barrier or the copy more expensive?
Is the barrier wait the same on every device, or asymmetric? Why?
Does profiling perturb the timing?
Why should normal timing and profile capture be separate?
```

### Exercise 6: Explain The Shape

Given:

```text
num_devices = 4
tile_rows = 4
tile_cols = 128
dtype = bf16
```

Answer:

```text
What is the global array shape?
What is the local shard shape?
How many bytes does each device copy?
Why does the local shard have a leading singleton dimension?
```

---

## Instructor Review Notes

This lab is well-positioned as Lab 1 because it starts with custom communication rather than another all-reduce review. The original lab already had the right core: a single-hop Pallas remote-DMA neighbor copy compared against `lax.ppermute`, plus explicit notes about barriers, DMA semaphores, payload rounding, and HBM versus VMEM defaults.

The main teaching improvements are:

1. Make ownership explicit before and after the operation.
2. Make the global-shape versus local-shard-shape distinction impossible to miss.
3. Explain that this is a push model, not a remote read.
4. Name the synchronization objects and their purpose.
5. Show that the barrier is an entry-ordering device, not decorative syntax.
6. Make measurement discipline part of the lab, not an appendix.
7. State that a built-in collective may be faster and that this is not failure.

The Python file should be heavily commented because this is the first time students see Pallas TPU remote DMA in this course. Later labs can become terser once the spellbook has names.

---

## Troubleshooting

### No TPU devices are visible

Check that the TPU runtime is available and that JAX sees devices:

```python
import jax
print(jax.devices())
print(jax.local_devices())
```

### The Pallas case fails but `pmap_ppermute` works

Start with:

```text
errors/*.txt
logs/stderr.log
diagnostics/runtime.json
```

Common causes:

```text
unsupported JAX/Pallas API version
memory space too large for VMEM
incorrect collective ID reuse in a larger experiment
shape that violates a TPU/Pallas tiling constraint
```

### The benchmark seems too fast

JAX dispatch is asynchronous. Timing code must block on the result. The harness should use `block_until_ready()` for measured iterations.

### The first iteration is slow

That is usually compilation. Compare warmup behavior against measured iterations and do not mix compile time into steady-state latency.

### The actual payload is larger than requested

The requested byte count is rounded into a whole local tile. Read the actual payload field in the CSV before computing throughput.

### `pallas_neighbor_copy` is slower than `pmap_ppermute`

That is allowed. Built-in collectives are compiler-managed and often highly optimized. Lab 1 is a visibility lab. Performance work begins in earnest when students add chunking, buffering, bidirectional movement, and topology-aware schedules.

---

## Pass Criteria

A student passes Lab 1 when they can produce:

```text
one successful default run
one short smoke run
one reverse-direction run
one profiler trace for pallas_neighbor_copy
one short written explanation of ownership and synchronization
```

The explanation should answer:

```text
What does each device send?
What does each device receive?
What does the barrier protect?
What do send_sem and recv_sem track?
Why compare against lax.ppermute?
Why might the custom kernel be slower?
```

---

## Next Lab Preview

Lab 2 turns this one-hop copy into a token-passing ring. Instead of doing one independent hop, each device passes data around the ring for multiple hops and accumulates what it has seen.

That introduces the next monster in the maze: dependency chains. 🧭
