# Lab 10: Multi-Host Smoke And Hierarchy

Goal: move from local-device algorithms to process topology, launch validation,
and hierarchical collective planning.

Lab 10 is the course finale. Labs 1 through 9 taught device-level movement:
single-hop remote DMA, token rings, all-gather, reduce-scatter, all-reduce,
chunking, and logical 2D mesh staging. This lab steps one level up. Before a
multi-host collective can be trusted, the run itself must prove that every
Python process sees the same world.

This lab is not a new Pallas kernel. It is the launch tower for the next one.
It asks:

```text
How many processes are participating?
Which process am I?
Which devices are local to this process?
Which devices exist globally?
Can every process enter the same synchronization and all-gather?
How should a future hierarchical collective split local and cross-process work?
```

## Run

```bash
python collective_bench.py --lab lab10
```

Short single-process smoke:

```bash
python collective_bench.py \
  --lab lab10 \
  --sizes 1KiB \
  --iters 1 \
  --warmup 0 \
  --no-plots
```

Validation knobs for a real multi-host launch:

```bash
python collective_bench.py \
  --lab lab10 \
  --lab10-expected-process-count 2 \
  --lab10-expected-global-devices 8
```

Payload-size sweep for the process collective smoke:

```bash
python collective_bench.py \
  --lab lab10 \
  --sizes 16B,1KiB,64KiB \
  --lab10-expected-process-count 2 \
  --lab10-expected-global-devices 8
```

The `--sizes` value in this lab controls the tiny process-level all-gather
payload. It is control-plane validation, not a TPU interconnect bandwidth
benchmark.

## Implemented Happy Path

- `lab10_topology_smoke`: records process count, process index, host identity,
  local devices, global devices, device grouping by process, selected launch
  environment variables, and expected-count checks.
- `lab10_process_collective_smoke`: calls `multihost_utils.assert_equal` when
  available, then `sync_global_devices` and `process_allgather` with a small
  int32 payload.
- `lab10_multihost_spec`: writes a course artifact for hierarchical collective
  planning and capstone follow-up work.

Single-process runs still pass and are useful. They verify the artifact
pipeline, topology report, render path, and process-collective code path. On a
real multi-host launch, the same operations become a launch validation: every
process must enter the same program, use the same payload shape, pass the same
contract, and produce one gathered payload segment per process.

## Mental Model

A multi-host JAX job is many Python processes cooperating over one logical JAX
program.

Vocabulary:

```text
process
  One Python controller. Usually one process per TPU host for these labs.

process_index
  This process's global rank among the participating controllers.

a process 0 summary
  A global human-readable summary written by only process 0. Every process may
  write local logs, but only one process should write merged summaries.

local_devices
  Devices addressable by this process.

global devices
  All devices participating in the JAX distributed computation.

host-local phase
  A collective phase that can stay within one process's local devices.

cross-process phase
  A collective phase that moves data across process boundaries.
```

The critical distinction:

```text
jax.local_devices() tells this process what it can address directly.
jax.devices() tells the program what devices exist globally.
```

In a single-process run those lists often describe the same set. In a
multi-host run they should differ: each process has a local slice of the global
machine.

## Why This Lab Exists

Multi-host failures rarely look like elegant math mistakes. They look like tiny
launch gremlins wearing boring hats:

```text
one host did not start
two hosts disagree about process_count
one process saw a different command-line flag
one process used a different sync name
one process saw fewer devices
all processes wrote the same global summary file
```

Lab 10 gives students a debugging artifact before they run a multi-host Pallas
kernel. The collective wizard does not chant at semaphores until the launch
contract is clean.

## Process Collective Smoke

The process collective smoke builds one int32 vector per process:

```text
payload[p] = p * 1_000_003 + arange(elems)
```

Then it gathers all process payloads with `process_allgather(..., tiled=True)`.
For two processes and four elements per process, the expected flattened result
is:

```text
process 0 contributes: [0, 1, 2, 3]
process 1 contributes: [1000003, 1000004, 1000005, 1000006]

gathered flat result:
[0, 1, 2, 3, 1000003, 1000004, 1000005, 1000006]
```

The large stride makes process segments easy to recognize in artifact previews.
It is not a numerical trick. It is a flashlight.

The smoke checks:

```text
contract_assert_equal_reached
gather_shape
gather_values
one_segment_per_process
```

`assert_equal`, when available in the installed JAX version, is a useful
preflight step because it catches mismatched payload shape or global-device
visibility before the later gather comparison.

## Topology Smoke Artifact

The topology artifact records:

```text
hostname
pid
process_index
process_count
is_process_0
local_device_count
global_device_count
local_devices
global_devices
devices_by_process
process_group_summaries
distributed_env
checks
failed_checks
hierarchy_plan
hierarchy_byte_model
launch_plan
```

Start by reading:

```text
lab_artifacts/*lab10_topology_smoke*.json
lab_artifacts/*lab10_topology_smoke*.md
```

Useful questions:

```text
Does every process agree on process_count?
Does process_index satisfy 0 <= process_index < process_count?
Does global_device_count match the expected slice size?
Are device records grouped under the process indices you expected?
Do coords or device_kind fields reveal physical topology clues?
Are all launch environment variables identical where they should be?
```

## Hierarchical Collective Plan

A hierarchical collective separates local movement from cross-process movement.
The point is not just to move fewer bytes. The point is to use the right fabric
for each phase and keep ownership clear.

General plan:

```text
phase 1: intra-process
  Use devices local to each process.
  Reduce, gather, or prepare process-local blocks.

phase 2: inter-process
  Exchange host blocks, representatives, or owner chunks across process
  boundaries.

phase 3: intra-process fanout or layout restore
  Replicate, scatter, or restore the local layout needed by the next operation.
```

All-gather example:

```text
local gather on each process
cross-process exchange of process-local blocks
local layout restore or fanout
```

All-reduce example:

```text
local reduce-scatter or local reduce inside each process
cross-process reduction over host blocks or representatives
local all-gather, broadcast, or fanout inside each process
```

Reduce-scatter example:

```text
local partial reductions into owner chunks
cross-process exchange/reduce for matching owner chunks
optional local scatter to final owner devices
```

The spec artifact deliberately separates the plan from the measurement. The
process smoke tells you whether processes can coordinate. It does not tell you
how many TPU interconnect bytes a future Pallas kernel will send.

## Byte Model

Let:

```text
B = payload bytes per device
P = process count
L = local device count for this process
N = global device count
```

For a hierarchical all-gather planning model:

```text
host block size for this process = L * B
cross-process exchange per process, rough upper model = (P - 1) * L * B
full all-gather result per device = N * B
```

For a hierarchical all-reduce planning model:

```text
local phase:
  combine local contributions first

cross-process phase:
  exchange reduced host blocks or owner chunks

fanout phase:
  restore the result layout required by the next computation
```

These are planning ledgers, not measured bandwidth rows. Lab 10 is about run
symmetry and hierarchy design. Actual device-level byte accounting returns in
the capstone or the next custom kernel.

## Correctness Contract

Pass condition:

```text
all processes agree on process count and global device count
local logs identify process index, host identity, and local devices
process all-gather returns exactly one payload segment per process
the spec artifact separates intra-process, inter-process, and fanout phases
```

A stronger multi-host pass condition:

```text
every process writes a local artifact
every process reports the same process_count
every process reports the same global_device_count
process indices are unique and cover 0..process_count-1
process_allgather preview contains one recognizable segment per process
only process 0 writes the merged global summary
```

## What To Inspect

Artifacts:

```text
lab_artifacts/*lab10_topology_smoke*.json
lab_artifacts/*lab10_topology_smoke*.md
lab_artifacts/*lab10_process_collective_smoke*.json
lab_artifacts/*lab10_process_collective_smoke*.md
lab_artifacts/*lab10_multihost_spec*.json
lab_artifacts/*lab10_multihost_spec*.md
run_metadata.json
diagnostics/runtime.json
logs/console.log
errors/*.txt
```

In the JSON artifact, inspect:

```text
checks
failed_checks
devices_by_process
process_group_summaries
process_collective_checks
process_allgather_preview
expected_preview
hierarchy_plan
hierarchy_byte_model
launch_plan
```

## Common Failure Modes

### Expected process count mismatch

Symptom:

```text
expected_process_count fails
```

Likely cause:

```text
You ran the lab on one process but asked for two, or the launcher started fewer
processes than expected.
```

Recovery:

```text
Fix the launcher, or remove the expected-count knob for a single-process smoke.
```

### Global device count mismatch

Symptom:

```text
expected_global_devices fails
```

Likely cause:

```text
The process sees a different TPU slice than you think, or distributed JAX was
not initialized before device discovery.
```

Recovery:

```text
Check the launch command, runtime version, coordinator settings, and whether
JAX distributed initialization happened before device access.
```

### Process all-gather hangs

Symptom:

```text
the run stops at sync_global_devices or process_allgather
```

Likely cause:

```text
Not every process entered the same operation in the same order.
```

Recovery:

```text
Compare per-process logs. Confirm every process has the same command-line args,
payload size, process count, and sync name.
```

### Gather values mismatch

Symptom:

```text
gather_shape passes but gather_values fails
```

Likely cause:

```text
A process contributed an unexpected payload, or the tiled/stacked interpretation
changed.
```

Recovery:

```text
Inspect process_allgather_preview and expected_preview. Confirm tiled=True and
matching payload shape across processes.
```

### Duplicate global summaries

Symptom:

```text
multiple processes overwrite or append to the same global artifact
```

Likely cause:

```text
Every process wrote the merged summary instead of only process 0.
```

Recovery:

```text
Gate merged writes on process_index == 0. Keep process-local logs for everyone.
```

## Student Exercises

1. Run Lab 10 with no expected-count knobs on a single process. Explain why it
   still passes and which fields prove it is not a multi-host launch.
2. Run Lab 10 with `--lab10-expected-process-count 2` on one process. Identify
   the failed check and explain why it is a good failure.
3. On a real multi-host slice, compare `local_device_count` and
   `global_device_count` on every process.
4. Inspect `devices_by_process` and draw the process-to-device map.
5. For an all-gather with payload `B`, write separate byte ledgers for
   intra-process gather, cross-process exchange, and local fanout.
6. Decide which process should write the merged run summary and why.
7. Pick one earlier lab, such as Lab 7 all-reduce or Lab 9 staged all-gather,
   and sketch the first hierarchical version.

## Deferred Work

This lab intentionally stops before the next kernel. Future work:

```text
call jax.distributed.initialize in a launcher wrapper when auto-init is absent
write per-process logs and a process-0 merged summary
run Lab 9 staged all-gather over a process-spanning mesh axis
run Lab 7 all-reduce with host-local and cross-host phases separated
measure real cross-process device collective bytes and profile traces
compare flat multi-host collectives with hierarchical schedules
```

## Bridge To The Capstone

After Lab 10, students have the full spine of the course:

```text
single-hop copy
token ring
local memory spaces
semaphore failure modes
ring all-gather
reduce-scatter
all-reduce
chunking and pipeline planning
2D mesh staging
multi-host run control
```

The natural capstone is a hierarchical collective:

```text
local reduce-scatter
cross-process reduce-scatter or all-gather
local fanout or layout restore
```

The capstone report should connect four ledgers:

```text
correctness invariant
byte model
process/device topology
profiler evidence
```

That is the difference between "the collective ran" and "I understand why the
collective behaved the way it did." Tiny difference, enormous hat. 🎩
