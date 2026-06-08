# Collective Communications for TPU

## Course Cover Page

This course is a systems lab sequence for people who already know modern ML
systems and want to understand collective communication on TPUs from the ground
up. The target student can read a JAX benchmark, understands sharding and
model-parallel training at a high level, and has probably used collectives such
as all-reduce or all-gather without seeing the machinery underneath.

The course goal is to move from:

```text
I can call a collective.
```

to:

```text
I can reason about the data movement schedule, synchronization protocol,
memory hierarchy, topology, benchmark methodology, and profiler trace.
```

By the end, students should be able to explain why a collective is fast or
slow, build small custom communication kernels, integrate communication into
ML-shaped computations, and know when not to do any of that because the
built-in compiler collective is better.

## Course Thesis

A collective is not magic. It is:

- a set of devices and a topology
- a tensor ownership layout
- a data movement schedule
- synchronization state
- optional reduction math
- a memory hierarchy
- a correctness invariant
- a performance model
- a profiler trace that confirms or challenges the model

The labs make those parts visible one at a time. Early labs are deliberately
small. Later labs assemble the pieces into all-gather, reduce-scatter,
all-reduce, chunked rings, staged mesh collectives, and multi-host run-control
checks.

The course is a lantern, not a leaderboard. A custom Pallas kernel that is
slower than `lax.psum` can still be the perfect lab if it exposes where the
bytes move, which semaphore proves they arrived, and why the benchmark curve
looks the way it does.

## Audience And Prerequisites

Students should already be comfortable with:

- Python and JAX basics
- `jax.jit`, arrays, shapes, dtypes, and asynchronous dispatch
- basic TPU or accelerator execution
- sharding and model-parallel training at a high level
- standard collectives such as all-reduce, all-gather, all-to-all, and permute
- reading benchmark output without treating one average as a law of nature

Students do not need to know Pallas before the course. Pallas, remote DMA,
local TPU memory spaces, and semaphores are introduced as the course substrate.

## Why Learn This

Large ML workloads are increasingly communication-shaped. Gradient sync,
tensor-parallel matmul, MoE token exchange, pipeline activation movement,
optimizer sharding, and multi-host training all spend real time moving data
between chips. If you only know the API call, performance debugging becomes
mapless cave exploration.

This course treats collective communication as a concrete system:

- where the bytes start
- where the bytes land
- which chip pushes them
- which semaphore proves they arrived
- which buffer slot is safe to read
- which topology axis is being used
- which benchmark row proves the claim
- which profiler event explains the surprise

## Technical Boundary

The practical low-level boundary for this course is not host networking
sockets. Host sockets matter for job orchestration and multi-host launch
control; they are not how TPU chips move tensors across the interconnect inside
a collective.

For this course, the useful substrate is:

- JAX arrays and `lax` collectives as executable specifications
- `pmap` for simple replicated-axis references
- `shard_map` for writing per-device programs over a logical mesh
- Pallas TPU kernels for explicit memory-space and kernel-level control
- Pallas TPU remote DMA for push-style device-to-device copies
- DMA, regular, and barrier semaphores for synchronization and flow control

Custom Pallas kernels are not automatically faster. They are useful when the
lab objective is to expose the mechanism, when the algorithm requires custom
layout or synchronization, or when communication must be fused with custom
compute. Built-in XLA/JAX collectives are the production baseline until a
profile proves otherwise.

## Primary Hardware Target And Naming

The primary target for this 10-lab sequence is a **4-chip TPU v5e slice**, often
created on Google Cloud as `v5litepod-4`. In repo prose, `v5e-4` means that
4-chip v5e target unless a command is showing the exact Cloud TPU accelerator
type. A 4-chip v5e slice is enough for the main course because it gives
students:

- four local TPU devices
- a natural logical `2x2` mesh
- ring schedules with nontrivial wraparound
- enough devices for all-gather, reduce-scatter, and all-reduce ownership
  transformations
- a small, cheap-ish place to iterate on Pallas remote-DMA kernels

The labs should also remain conceptually portable to other 4-chip TPU slices,
including a 4-chip TPU v4 slice. However, the default documentation and command
examples should assume **TPU v5e-4 first**.

Hardware vocabulary for this course:

```text
4-chip v5e slice       primary target, usually v5litepod-4
logical 2x2 mesh       default mesh used by Labs 1-9
single-process run     default local-device execution mode
v5e-16 / v5litepod-16  future multi-host target for Lab 10 follow-ups
multi-host run         multiple JAX processes that must enter the same program
```

Avoid ambiguous phrases such as "TPU-4" or "TPU-16" unless the meaning is
explicitly defined. Prefer physical descriptions:

```text
4-chip v5e slice
16-chip v5e slice
logical 2x2 mesh
logical 4x4 mesh
single-process local slice
multi-host slice
```

Lab 10 is intentionally useful on a 4-chip v5e slice as a smoke test, even
though its multi-host checks only become interesting on a larger multi-host
slice such as a 16-chip v5e slice.

## What Students Will Learn

Students will learn to:

- dump TPU topology and explain `coords`, `process_index`, local devices,
  global devices, and mesh axes
- use built-in JAX collectives as executable specifications
- measure collectives without lying to themselves
- separate compilation, warmup, measurement, profiling, and correctness
- read latency distributions rather than one average
- capture XProf traces and connect them to benchmark rows
- write Pallas TPU kernels with explicit memory spaces
- reason about HBM, VMEM, SMEM, scratch buffers, and semaphore storage
- move data with Pallas TPU remote DMA
- use DMA, regular, and barrier semaphores correctly
- build single-hop communication, token rings, all-gather, reduce-scatter, and
  all-reduce
- reason about payload bytes, logical bytes, and estimated wire bytes
- compare flat rings with staged logical-mesh algorithms
- identify run-ahead races, buffer slot hazards, and collective ID mistakes
- separate local memory movement from remote communication
- decide when built-in XLA collectives beat custom Pallas kernels
- use multi-host smoke tests before attempting multi-host custom collectives

Some ML-shaped integrations, such as distributed matmul, MoE token exchange,
and bidirectional optimized collectives, are now best treated as capstone or
post-course extensions. The 10 labs focus on the durable communication grammar.

## Current Project Layout

```text
collective_communication/
  collective_bench.py         # CLI, measurement, logging, profiles, plots
  COURSE.md                   # this course cover page and lab sequence
  README.md                   # quick start and operation reference
  labs/
    roadmap.md                # detailed roadmap and invariants
    lab_spec_utils.py         # shared spec/artifact helpers
    lab1_single_hop.py        # Lab 1 concept code
    lab1_single_hop.md        # Lab 1 teaching notes
    lab2_token_ring.py        # Lab 2 concept code
    lab2_token_ring.md        # Lab 2 teaching notes
    lab3_memory_spaces.py     # Lab 3 concept code
    lab3_memory_spaces.md     # Lab 3 teaching notes
    lab4_semaphore_bug_zoo.py # Lab 4 concept code
    lab4_semaphore_bug_zoo.md # Lab 4 teaching notes
    lab5_ring_all_gather.py   # Lab 5 ring all-gather concept code
    lab5_ring_all_gather.md
    lab6_reduce_scatter.py    # Lab 6 reduce-scatter concept code
    lab6_reduce_scatter.md
    lab7_all_reduce.py        # Lab 7 all-reduce concept code
    lab7_all_reduce.md
    lab8_chunked_pipeline.py  # Lab 8 chunked-ring concept code
    lab8_chunked_pipeline.md
    lab9_mesh_collectives.py  # Lab 9 staged mesh concept code
    lab9_mesh_collectives.md
    lab10_multihost_smoke.py  # Lab 10 multi-host run-control code
    lab10_multihost_smoke.md
  runs/                       # generated artifacts, ignored by git
```

The design principle is separation:

- lab files teach one communication idea at a time
- `collective_bench.py` owns logging, timing, profiling, CSV/JSONL output,
  diagnostics, plots, and artifact indexes
- run artifacts should be rich enough to debug after the TPU VM is gone

## Runner Contract

Basic command:

```bash
cd ~/labs/collective_communication
python collective_bench.py --lab lab1
```

Every invocation creates a run directory:

```text
runs/<run-name>/
```

Each benchmark case should:

1. build input tensors and expected results
2. compile on first call, outside the measured steady-state loop
3. run warmup iterations
4. record measured iterations with `block_until_ready()`
5. write one JSONL/CSV row
6. optionally capture an XProf trace or memory profile
7. include correctness status and failure artifacts
8. emit enough metadata to reproduce the case later

The console table reports p50 latency. Machine-readable rows should include:

- raw timing samples, or enough summary statistics to reconstruct the story
- mean
- p10
- p50
- p90
- p99
- min
- max
- payload size
- dtype
- operation name
- layer or implementation family
- device count
- mesh axis or mesh shape, when applicable
- correctness status
- error artifact path, when applicable

## Measurement Contract

The benchmark harness treats measurement as a lab object, not a side effect.

Required measurement habits:

- compile before measuring
- run warmups before recording
- use `block_until_ready()` in measured paths
- report distributions, not just a single average
- keep profiler-enabled runs separate from clean timing runs
- record package versions and relevant environment variables
- record the topology observed by the program
- preserve both successes and failures in machine-readable output

Byte vocabulary:

- `payload_bytes`: bytes in the local tensor payload for one device
- `logical_bytes`: bytes implied by the collective API or algorithmic result
- `estimated_wire_bytes`: a hand-derived estimate of bytes sent over device
  links by the chosen schedule
- `effective_logical_bandwidth`: logical bytes divided by measured time
- `effective_wire_bandwidth`: estimated wire bytes divided by measured time,
  when the wire-byte model is meaningful

First-order byte models are useful but not final proof. For a one-direction
ring all-gather over `n` devices where each device starts with a shard of size
`S`:

```text
send_bytes_per_device ~= (n - 1) * S
recv_bytes_per_device ~= (n - 1) * S
```

For a ring all-reduce over `n` devices with full per-device payload size `B`,
implemented as reduce-scatter plus all-gather:

```text
send_bytes_per_device ~= 2 * (n - 1) / n * B
recv_bytes_per_device ~= 2 * (n - 1) / n * B
```

The teaching implementations in Labs 5-8 sometimes move more bytes than the
optimal schedule. That is intentional. The byte model should say which version
is being measured: composed teaching bytes or optimized target bytes.

## Artifacts Captured

Each run directory contains:

- `logs/console.log`: combined stdout/stderr, including XLA and TPU runtime diagnostics
- `logs/stdout.log`
- `logs/stderr.log`
- `results.jsonl`: one row per benchmark case
- `csvs/results.csv`: spreadsheet-friendly results
- `csvs/results_ok.csv`
- `csvs/results_failed.csv`, when needed
- `plots/latency_by_payload.png` (line = p50, shaded band = p10-p90)
- `plots/bandwidth_by_payload.png`
- `plots/speedup_by_payload.png`
- `plots/case_status.png`
- `plots/trace_comm_*.png` and `traces/trace_comm_summary.json`, only when a profiling run captured traces
- `diagnostics/preimport.json`
- `diagnostics/runtime.json`
- `diagnostics/memory_start.json`
- `diagnostics/memory_end.json`
- `diagnostics/postrun.json`
- `errors/*.txt`, for failed cases
- `run_config.json`
- `run_metadata.json`
- `run_summary.json`
- `results_summary.md`
- `artifact_index.json`

Diagnostics include Python version, package versions, relevant JAX/XLA/TPU
environment variables, transparent hugepage state, JAX device report, git state
when available, and memory snapshots.

## Profiling

Profiles are off by default because they perturb timing and can produce noisy
runtime messages. Enable them explicitly:

```bash
python collective_bench.py \
  --lab lab2 \
  --sizes 4MiB \
  --profile \
  --profile-cases 1 \
  --trace-op pallas_token_ring \
  --trace-size 4MiB
```

Profiler traces are written under `traces/` in the run directory and listed in
`artifact_index.json`. Memory profiles can be captured with:

```bash
--memory-profiles
```

XLA/HLO dumps can be captured with:

```bash
--xla-dump-to run
```

A good profile note should answer:

```text
What operation did I trace?
What payload size did I trace?
Was the trace captured from a clean single case?
Do the measured timing rows and profiler events describe the same phenomenon?
```

## Lab Sequence

The course now has 10 concrete labs. The default path is designed to run on a
4-chip TPU v5e slice. Lab 10 also runs locally as a smoke test, then becomes a
multi-host validation lab on a larger slice.

### Pre-Lab: Topology And Baseline Orientation

This is not a numbered lab in the current runner. It is the orientation step
students should perform before Lab 1.

Students inspect:

- `jax.devices()`
- `jax.local_devices()`
- `jax.process_index()`
- `jax.process_count()`
- device kind
- device IDs
- coordinates, when exposed
- local versus global device count
- candidate logical ring orders
- candidate logical `2x2` mesh shapes on v5e-4

Students should be able to answer:

```text
Which devices are local to this process?
Which logical mesh axes are available?
What ring order am I about to use?
Will this lab cross a process or host boundary?
What topology facts were recorded in the run directory?
```

### Lab 1: Custom Single-Hop Neighbor Copy

First contact with custom TPU communication. Each device owns a local tile and
pushes it to its logical right neighbor using Pallas TPU remote DMA and
semaphores. The reference implementation is `lax.ppermute`.

Whole-tile remote-copy refs default to HBM for this lab, because the default
payload sweep includes sizes that can exceed scoped VMEM. VMEM appears
explicitly in Lab 3 and later chunked-buffering labs.

Students learn:

- logical ring neighbors
- `shard_map`
- Pallas TPU remote DMA
- DMA send and receive semaphores
- entry barriers
- correctness against `lax.ppermute`
- how tiny-payload latency differs from large-payload bandwidth

Run:

```bash
python collective_bench.py --lab lab1
```

Pass condition:

```text
output[rank] == input[left_neighbor]
custom output matches lax.ppermute
all tested payload sizes either pass or produce useful failure artifacts
```

### Lab 2: Token-Passing Ring

The single-hop primitive becomes a coordination computation. Each device starts
with a rank-valued tile, passes it around the ring for several hops, and
accumulates the ranks it has seen.

Students learn:

- latency amplification through dependencies
- one-hop schedules as building blocks
- per-hop correctness invariants
- collective ID discipline
- timing distributions for communication chains
- why dependency depth can dominate payload size for small messages

Run:

```bash
python collective_bench.py --lab lab2
```

Pass condition:

```text
after n - 1 hops, every device has seen every rank exactly once
custom token ring matches the ppermute token-ring reference
hop count, device count, and final ownership are recorded
```

Suggested experiment:

```bash
python collective_bench.py \
  --lab lab2 \
  --sizes 1KiB,64KiB,4MiB \
  --token-hops 0,1,2,3
```

### Lab 3: Pallas Memory Spaces

Before deeper remote communication, students study the local memory path inside
a Pallas TPU kernel. The custom path copies each local HBM tile into VMEM
scratch with `pltpu.make_async_copy`, computes a simple arithmetic transform
into VMEM, and copies a `float32` result back to HBM.

Students learn:

- what lives in HBM versus VMEM
- how local async copies differ from normal array indexing
- why TPU tiling constraints shape kernel design
- how local memory movement appears in profiles
- why remote communication kernels often need explicit staging buffers

Run:

```bash
python collective_bench.py --lab lab3
```

Core exercise:

```text
HBM input -> VMEM input scratch -> compute -> VMEM output scratch -> HBM output
```

Pass condition:

```text
students can explain each memory space used by the kernel
results match a pure JAX reference
profile artifacts show the local movement and compute region
```

### Lab 4: Semaphore Bug Zoo

Implemented as a safe bug catalog plus a correct semaphore probe. Small,
intentionally broken kernels are documented by mutation, expected symptom,
diagnostic, and recovery rule, but hang-prone variants are not executed by
default.

Students study:

- missing wait
- over-wait hang
- over-signal nonzero semaphore state
- missing entry barrier
- wrong `collective_id` reuse
- buffer slot races
- two senders writing the same destination
- insufficient bytes sent to satisfy a receive wait

Run:

```bash
python collective_bench.py --lab lab4
```

Core exercise:

```text
run the correct semaphore probe
inspect the bug catalog artifacts
for each mutation, identify the violated invariant
record the failure mode
write the debugging rule that would have caught it
```

Pass condition:

```text
every failure mode has a minimal reproducer, expected symptom, and recovery note
the correct semaphore probe still passes
no hang-prone broken kernel runs by default
```

### Lab 5: Ring All-Gather

Build the first full custom collective from repeated ring movement. The happy
path runs a pmap ring schedule, compares against `lax.all_gather`, and composes
the Lab 1 Pallas remote-DMA hop into an arrival-order all-gather.

Students learn:

- ownership layout
- ring schedules
- shard placement
- exact byte models for one-direction all-gather
- why layout choices affect later computation
- why a composed custom collective is useful before a fused kernel

Run:

```bash
python collective_bench.py --lab lab5
```

Core exercise:

```text
each device starts with shard S_i
after n - 1 hops, each device owns every shard in arrival order
```

Pass condition:

```text
arrival-order ring all-gather sees every rank exactly once
composed Pallas ring matches the pmap arrival-order schedule on TPU
built-in lax.all_gather passes for the same payloads
the spec artifact records the fused-kernel and canonical-layout follow-ups
```

### Lab 6: Ring Reduce-Scatter

Learn the useful half of serious all-reduce. Each device starts with chunks,
accumulates reductions, and finishes owning only one reduced shard. The custom
path is a whole-token teaching implementation built from Lab 1 remote-DMA hops;
the optimized one-chunk-per-hop ring comes later.

Students learn:

- chunk ownership
- local reduce plus remote movement
- correctness against `lax.psum_scatter`
- why sharded reduced results matter in ML systems
- how reduce-scatter differs from "all-reduce then slice"
- why the first correct custom path can be intentionally bandwidth-inefficient

Run:

```bash
python collective_bench.py --lab lab6
```

Core exercise:

```text
each device starts with chunks [x_i0, x_i1, ..., x_i,n-1]
device j finishes with sum_i x_ij
```

Pass condition:

```text
custom reduce-scatter matches lax.psum_scatter
students can identify which device owns each reduced chunk after every phase
the report distinguishes whole-token teaching bytes from optimal ring bytes
```

### Lab 7: All-Reduce From Reduce-Scatter Plus All-Gather

Assemble gradient synchronization from the two primitive phases and compare
against `lax.psum`. The custom path composes Lab 6 reduce-scatter with Lab 5
all-gather, then restores canonical chunk order so every device sees the same
full reduced tensor.

Students learn:

- all-reduce byte models
- numerical correctness
- why built-ins often win
- when custom kernels are useful anyway
- how all-reduce hides two distinct ownership transformations
- where layout normalization belongs between communication phases

Run:

```bash
python collective_bench.py --lab lab7
```

Core exercise:

```text
phase 1: whole-token reduce-scatter
phase 2: chunk all-gather
phase 3: canonical chunk-order restore
result: every device owns the fully reduced tensor
```

Pass condition:

```text
custom all-reduce matches lax.psum
students can explain both teaching-byte and optimal ring-byte models
report includes a byte model and explains any gap from measured performance
```

### Lab 8: Chunked And Pipelined Ring

Turn a correct collective into a performance experiment with chunk size,
multiple in-flight copies, and buffer count. The implemented custom path is
chunked but serialized; the fused double-buffered kernel is the next
optimization target.

Students learn:

- latency versus bandwidth regimes
- flow control
- double buffering as a design contract
- overlap limits
- profile-guided tuning
- run-ahead hazards
- the difference between chunking and true overlap

Core experiment:

```bash
python collective_bench.py \
  --lab lab8 \
  --sizes 16KiB,64KiB,256KiB,1MiB,4MiB,16MiB \
  --lab8-chunks 8 \
  --lab8-buffer-count 2
```

Pass condition:

```text
serialized chunked Pallas ring matches per-chunk token sums on TPU
students can identify chunk overhead before claiming overlap
profiles show whether future fused versions have real overlap
```

### Lab 9: 2D Mesh Collectives

Stop treating every local slice as a flat ring. On the 4-chip v5e target, use a
logical `2x2` mesh and compare staged all-gather algorithms against flat-ring
thinking. The custom Pallas path keeps a flat physical mesh and computes
logical 2D neighbors inside the kernel.

Students learn:

- mesh axis choice
- x-then-y versus y-then-x staging
- hop distance and contention hypotheses
- topology-aware explanations
- why a 2D algorithm may be clearer even when n is only 4
- how axis order changes phases without changing logical result bytes

Run:

```bash
python collective_bench.py \
  --lab lab9 \
  --lab9-mesh-shape 2x2 \
  --lab9-axis-order x_then_y
```

Core exercise:

```text
all-gather along x, then y
compare against flat ring
repeat y, then x
record topology evidence and profile traces
```

Pass condition:

```text
pmap and Pallas staged all-gather match canonical rank order
students can explain axis-order timing using recorded topology evidence
```

### Lab 10: Multi-Host Smoke And Hierarchy

Move from one process to multi-host run control and hierarchical collectives.
On the primary 4-chip v5e target, this lab is a single-process topology and
process-collective smoke. On a future 16-chip v5e slice or larger, it becomes a
real multi-host launch validation.

Students learn:

- `process_index`
- `process_count`
- global versus local devices
- per-process logs
- process-0 summaries
- local versus cross-process phases
- why all processes must enter distributed JAX consistently
- how to fail early when process count or global device count is wrong

Run:

```bash
python collective_bench.py --lab lab10
```

Optional launch-shape checks for a future multi-host run:

```bash
python collective_bench.py \
  --lab lab10 \
  --lab10-expected-process-count 2 \
  --lab10-expected-global-devices 16
```

Core exercise:

```text
inspect process and device topology
run a tiny process collective across all processes
write process-local launch facts
emit a hierarchy plan for future custom collectives
```

Pass condition:

```text
all processes agree on global device count and mesh shape
local logs identify local devices and process index
process all-gather returns one payload per process
hierarchical prototype separates local and cross-process phases
```

## Invariants For Custom Communication

Every custom communication lab should preserve these rules:

- every send has exactly one matching receive
- every DMA semaphore wait matches the bytes sent
- every regular or barrier semaphore drains before kernel completion
- every cross-device kernel has a clear entry synchronization story
- every collective ID has a documented communication pattern
- every buffer slot has one writer at a time
- no device reads a buffer slot while another device can write it
- every benchmark has a correctness check
- every failure emits an artifact
- every run has enough artifacts to debug after the TPU VM is gone

## Skill Milestones

### Apprentice

Can run built-in collectives, explain rank and mesh axes, read payload-size
curves, and interpret the run artifacts.

### Adept

Can implement single-hop remote DMA, debug neighbor maps and semaphore waits,
build a token ring, and compare custom kernels against `lax` references.

### Expert

Can build all-gather, reduce-scatter, and all-reduce; reason about byte
models; use XProf evidence; and explain when custom Pallas is worse than the
built-in compiler collective.

### Collective Communication Wizard

Can design topology-aware schedules, plan hierarchical collectives, reason
about buffer ownership and overlap, and write a reproducible benchmark report
that explains a surprising performance result.

## Capstone Direction

A final project should choose one:

- high-performance all-reduce for a fixed topology
- MoE token exchange with skew handling
- distributed matmul with overlapped tile exchange
- pipeline-parallel activation send/recv schedule
- hierarchical multi-host reduce-scatter
- topology optimizer that chooses a ring or mesh schedule from `jax.devices()`

The implementation must be correct, the benchmark must be reproducible, and
the report must connect performance claims to measured artifacts.

Capstone report outline:

```text
1. Problem statement
2. Hardware and topology
3. Tensor ownership layout
4. Communication schedule
5. Synchronization protocol
6. Correctness tests
7. Byte model
8. Benchmark methodology
9. Results
10. XProf evidence
11. One surprising result
12. What would change on a larger slice
```

## References

- Cloud TPU v5e documentation: https://docs.cloud.google.com/tpu/docs/v5e
- Cloud TPU system architecture: https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm
- JAX distributed Pallas for TPUs: https://docs.jax.dev/en/latest/pallas/tpu/distributed.html
- JAX profiling guide: https://docs.jax.dev/en/latest/profiling.html
