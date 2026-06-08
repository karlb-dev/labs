# TPU Collective Communication Benchmarks

This project measures communication across TPU devices at several levels of
abstraction:

- `pmap`/`lax` collectives: XLA-managed baseline operations such as all-gather,
  all-reduce, all-to-all, and ring permute.
- Pallas TPU kernels: custom kernels using documented Pallas TPU primitives
  such as remote DMA and semaphores.
- Lab/spec operations: teaching artifacts that record byte models, ownership
  layouts, deferred optimizations, and multi-host plans.
- External runners: standalone binaries or future shared-library bindings for
  lower-level experiments.

The project is both a benchmark harness and a course scaffold. The benchmark
rows are the evidence; the labs teach what each row means.

## Primary Target

The primary target is a **4-chip TPU v5e slice**, commonly created as
`v5litepod-4` on Google Cloud. In this repo, `v5e-4` means that 4-chip local
TPU target unless an exact Cloud TPU accelerator type is shown.

The 10 labs are designed so the main path runs on this 4-chip v5e target:

- Labs 1-2: local ring communication over four devices
- Lab 3: local HBM/VMEM movement
- Lab 4: semaphore correctness and failure-mode catalog
- Labs 5-8: composed collectives and chunking over the local ring
- Lab 9: logical `2x2` staged mesh all-gather
- Lab 10: single-process topology and process-collective smoke

Future multi-host work should start with Lab 10 on a larger v5e slice, such as
`v5litepod-16`, then extend the custom collectives only after topology and
process launch are trustworthy.

## Current Status

Implemented or active:

- built-in `pmap`/`lax` collective baselines
- Lab 1: custom single-hop neighbor copy
- Lab 2: token-passing ring
- Lab 3: Pallas memory spaces
- Lab 4: semaphore bug zoo catalog and correct semaphore probe
- Lab 5: ring all-gather reference plus composed Pallas custom path
- Lab 6: reduce-scatter reference plus composed Pallas custom path
- Lab 7: all-reduce reference plus composed Pallas custom path
- Lab 8: chunked ring reference plus serialized Pallas custom path
- Lab 9: 2D mesh all-gather reference plus staged Pallas custom path
- Lab 10: multi-host topology and process-collective smoke
- Pallas all-gather bridge using the installed JAX example implementation
- run directories, diagnostics, CSV/JSONL output, plots, optional profiles, and
  external runner hook

Planned next:

- Validate all 10 updated labs on the primary v5e-4 target
- Add estimated wire-byte columns for the remaining custom schedules
- Replace composed teaching paths with fused custom Pallas kernels one lab at a
  time, only when the teaching path is stable
- Add timeout-controlled repros for dangerous synchronization bugs
- Use Lab 10 as the gate before any multi-host custom Pallas collective work

## Source Layout

```text
collective_communication/
  collective_bench.py         # CLI, run dirs, logging, measurement, profiling, plots
  COURSE.md                   # course cover page and lab sequence
  README.md                   # this operation reference
  labs/
    lab_spec_utils.py         # shared spec/artifact rendering helpers
    lab1_single_hop.py        # Lab 1 Pallas communication code
    lab1_single_hop.md        # Lab 1 teaching notes
    lab2_token_ring.py        # Lab 2 token-passing ring code
    lab2_token_ring.md        # Lab 2 teaching notes
    lab3_memory_spaces.py     # Lab 3 local memory-space code
    lab3_memory_spaces.md     # Lab 3 teaching notes
    lab4_semaphore_bug_zoo.py # Lab 4 semaphore failure catalog
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
  runs/                       # generated benchmark artifacts, ignored by git
```

## Quick Start

```bash
cd ~/labs/collective_communication
python collective_bench.py
```

Every invocation creates a run directory by default:

```text
runs/collective_bench-YYYYMMDD_HHMMSS-xxxxxx/
```

Run a small smoke test:

```bash
python collective_bench.py \
  --sizes 1KiB,64KiB \
  --iters 10 \
  --warmup 2
```

Run the whole 10-lab arc one lab at a time:

```bash
python collective_bench.py --lab lab1
python collective_bench.py --lab lab2
python collective_bench.py --lab lab3
python collective_bench.py --lab lab4
python collective_bench.py --lab lab5
python collective_bench.py --lab lab6
python collective_bench.py --lab lab7
python collective_bench.py --lab lab8
python collective_bench.py --lab lab9
python collective_bench.py --lab lab10
```

Useful v5e-4 smoke defaults:

```bash
python collective_bench.py \
  --lab lab5 \
  --sizes 1KiB,64KiB \
  --iters 10 \
  --warmup 2
```

Whole-tile Lab 1 and Lab 2 Pallas refs default to HBM so large payload sweeps
do not exceed scoped VMEM. VMEM is still available for small explicit
experiments with `--pallas-memory-space VMEM`.

## Lab Guide

| Lab | Main command | What it teaches | Primary custom path |
| --- | --- | --- | --- |
| Lab 1 | `--lab lab1` | single-hop device-to-device copy | `pallas_neighbor_copy` |
| Lab 2 | `--lab lab2` | dependency-chain token ring | `pallas_token_ring` |
| Lab 3 | `--lab lab3` | HBM to VMEM to HBM local memory movement | `pallas_vmem_arith` |
| Lab 4 | `--lab lab4` | semaphore invariants and safe bug catalog | `pallas_semaphore_correct` |
| Lab 5 | `--lab lab5` | arrival-order ring all-gather | `pallas_ring_all_gather` |
| Lab 6 | `--lab lab6` | reduce-scatter ownership | `pallas_ring_reduce_scatter` |
| Lab 7 | `--lab lab7` | all-reduce as reduce-scatter plus all-gather | `pallas_ring_all_reduce` |
| Lab 8 | `--lab lab8` | chunk size, buffer planning, serialized chunking | `pallas_chunked_token_ring` |
| Lab 9 | `--lab lab9` | logical `2x2` staged mesh collectives | `pallas_2d_staged_all_gather` |
| Lab 10 | `--lab lab10` | process topology and multi-host launch smoke | `lab10_process_collective_smoke` |

The custom paths in Labs 5-8 are teaching implementations. They compose earlier
correct primitives so the communication schedule is visible. Fused kernels,
double-buffering, and optimal one-chunk-per-hop reduce-scatter are deliberate
follow-ups, not hidden promises.

## Common Runs

Run Lab 1 with a profiler trace for the custom Pallas case:

```bash
python collective_bench.py \
  --lab lab1 \
  --sizes 4MiB \
  --profile \
  --profile-cases 1 \
  --trace-op pallas_neighbor_copy \
  --trace-size 4MiB
```

Run Lab 2 with hop-depth sweeps:

```bash
python collective_bench.py \
  --lab lab2 \
  --sizes 1KiB,64KiB,4MiB \
  --token-hops 0,1,2,3
```

Run Lab 8 chunking experiments:

```bash
python collective_bench.py \
  --lab lab8 \
  --sizes 16KiB,64KiB,256KiB,1MiB,4MiB,16MiB \
  --lab8-chunks 8 \
  --lab8-buffer-count 2
```

Run Lab 9 on the default logical `2x2` mesh:

```bash
python collective_bench.py \
  --lab lab9 \
  --lab9-mesh-shape 2x2 \
  --lab9-axis-order x_then_y
```

Run Lab 10 with explicit future multi-host expectations:

```bash
python collective_bench.py \
  --lab lab10 \
  --lab10-expected-process-count 2 \
  --lab10-expected-global-devices 16
```

Run only a few baseline and bridge operations:

```bash
python collective_bench.py \
  --ops pmap_ppermute,pmap_all_gather,pallas_all_gather
```

## Run Directories And Artifacts

Important files:

- `logs/console.log`: combined stdout/stderr, including TPU runtime and XLA diagnostics
- `logs/stdout.log`
- `logs/stderr.log`
- `results.jsonl`: one machine-readable row per benchmark case
- `csvs/results.csv`: spreadsheet-friendly results
- `csvs/results_ok.csv`
- `csvs/results_failed.csv`, when needed
- `plots/latency_by_payload.png` (line = p50, shaded band = p10-p90)
- `plots/bandwidth_by_payload.png`
- `plots/speedup_by_payload.png`: relative speedup vs the slowest op at each payload
- `plots/case_status.png`
- `plots/trace_comm_*.png`: per-device on-device communication time, only when a profiling run captured traces
- `traces/trace_comm_summary.json`: machine-readable per-device DMA/barrier/semaphore time, only when profiling captured traces
- `diagnostics/`: environment, JAX device, THP, and memory snapshots
- `lab_artifacts/`: lab-specific specs, bug catalogs, topology reports, and hierarchy plans
- `errors/`: full diagnostic text for failed cases
- `run_config.json`
- `run_metadata.json`
- `run_summary.json`
- `results_summary.md`
- `artifact_index.json`

Control run names and locations:

```bash
python collective_bench.py \
  --run-root /tmp/tpu-collective-runs \
  --run-name v5e4_ring_baseline
```

If a named run already exists, the script creates a suffixed directory such as
`v5e4_ring_baseline_02` rather than appending into the previous run. Use
`--run-dir` when you intentionally want an exact directory.

## Measurement Contract

Each benchmark case should separate setup from measurement:

1. build inputs and expected results
2. compile on first call
3. run warmup iterations
4. measure steady-state iterations
5. call `block_until_ready()` in the measured path
6. check correctness
7. write a JSONL/CSV row
8. capture failure details when something goes wrong

The console table reports p50 latency. Machine-readable rows preserve more of
the timing distribution:

- raw samples, when available
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
- implementation layer
- device count
- mesh shape or mesh axis, when applicable
- correctness status

Byte vocabulary:

- `payload_bytes`: local payload bytes for one device
- `logical_bytes`: bytes implied by the collective API or algorithmic result
- `estimated_wire_bytes`: hand-derived estimate of bytes sent by a schedule,
  when available
- `effective_logical_bandwidth`: logical bytes divided by measured time
- `effective_wire_bandwidth`: estimated wire bytes divided by measured time,
  when meaningful

## Profiling

Capture a TPU profiler trace for one benchmark case:

```bash
python collective_bench.py \
  --ops pmap_all_gather \
  --sizes 4MiB \
  --profile \
  --trace-op pmap_all_gather \
  --trace-size 4MiB
```

Profiler traces are written under `traces/` in the run directory by default.
You can override the trace root with `--trace-dir`. Normal timing does not call
the JAX profiler, so TensorFlow profiler-hook warnings should not appear unless
profiling is enabled.

The examples use `--profile`; `--collect-profiles` and `--xprof` are accepted as
aliases for the same flag.

Optional memory profile capture:

```bash
python collective_bench.py \
  --profile \
  --profile-cases 2 \
  --memory-profiles
```

Optional XLA/HLO dumps:

```bash
python collective_bench.py \
  --xla-dump-to run \
  --sizes 1KiB \
  --iters 5
```

`--xla-dump-to run` writes dumps under `<run_dir>/xla_dumps`.

Good profiling hygiene:

- profile one or a small number of cases at a time
- choose a payload size intentionally
- use clean timing runs for latency and bandwidth tables
- use profile runs to explain the timing, not to replace it
- keep the trace operation and trace payload size in the run metadata

## External Runner Hook

The `external` operation shells out to a command template. This is meant for
future C++ console binaries or other low-level experiments.

```bash
python collective_bench.py \
  --ops external \
  --sizes 1MiB \
  --external './build/ring_bench --bytes {bytes} --iters {iters} --devices {devices}'
```

The external program should print JSON on stdout with at least:

```json
{"seconds": 0.000123, "ok": true}
```

Optional fields such as `logical_bytes`, `note`, and `layer` will be used if
present.

## Current Operation Names

| Operation | Layer | Purpose |
| --- | --- | --- |
| `pmap_psum` | `pmap`/`lax` | all-reduce baseline |
| `pmap_all_gather` | `pmap`/`lax` | all-gather baseline |
| `pmap_all_to_all` | `pmap`/`lax` | all-to-all baseline |
| `pmap_ppermute` | `pmap`/`lax` | ring send/receive reference |
| `pmap_token_ring` | `pmap`/`lax` | repeated-permute token-ring reference |
| `pmap_local_arith` | `pmap`/local | Lab 3 local arithmetic reference |
| `pmap_ring_all_gather` | `pmap`/`lax` | arrival-order ring all-gather reference |
| `pmap_psum_scatter` | `pmap`/`lax` | reduce-scatter reference |
| `pmap_2d_staged_all_gather` | `pmap`/`lax` | staged 2D mesh all-gather reference |
| `pallas_neighbor_copy` | Pallas TPU | custom single-hop remote-DMA copy |
| `pallas_token_ring` | Pallas TPU | repeated custom remote-DMA hop |
| `pallas_ring_all_gather` | Pallas TPU | composed ring all-gather built from Lab 1 hops |
| `pallas_ring_reduce_scatter` | Pallas TPU | whole-token reduce-scatter built from Lab 1 hops |
| `pallas_ring_all_reduce` | Pallas TPU | composed reduce-scatter plus all-gather all-reduce |
| `pallas_chunked_token_ring` | Pallas TPU | serialized chunked token ring built from Lab 1 hops |
| `pallas_2d_staged_all_gather` | Pallas TPU | logical 2D mesh staged all-gather |
| `pallas_vmem_arith` | Pallas TPU | HBM/VMEM local memory-space exercise |
| `pallas_semaphore_correct` | Pallas TPU | correct barrier and DMA semaphore probe |
| `semaphore_bug_zoo` | lab/spec | safe catalog of semaphore failure modes |
| `lab5_ring_all_gather_spec` | lab/spec | fused ring all-gather follow-up plan |
| `lab6_reduce_scatter_spec` | lab/spec | optimized reduce-scatter follow-up plan |
| `lab7_all_reduce_spec` | lab/spec | optimized two-phase all-reduce follow-up plan |
| `lab8_chunked_pipeline_spec` | lab/spec | fused pipelined ring follow-up plan |
| `lab9_mesh_collectives_spec` | lab/spec | topology-aware mesh follow-up plan |
| `lab10_topology_smoke` | lab/topology | process and device topology artifact |
| `lab10_process_collective_smoke` | lab/multihost | process sync/all-gather validation |
| `lab10_multihost_spec` | lab/spec | multi-host hierarchy plan |
| `pallas_all_gather` | Pallas TPU example | Pallas all-gather bridge implementation |
| `external` | subprocess | hook for standalone or lower-level experiments |

The Pallas all-gather path uses the installed JAX example implementation from
`jax.experimental.pallas.ops.tpu.all_gather`. That code exercises TPU remote
copies and semaphores, so it is a useful bridge from XLA collectives to custom
communication kernels.

## Reading A Run

Start here:

```text
results_summary.md
plots/latency_by_payload.png
plots/bandwidth_by_payload.png
csvs/results.csv
logs/console.log
artifact_index.json
```

Suggested reading order:

1. check `results_summary.md` for pass/fail counts
2. inspect `plots/case_status.png` for failures
3. compare latency curves before bandwidth curves
4. open `csvs/results.csv` for exact rows
5. inspect `errors/*.txt` for failed cases
6. open `lab_artifacts/` for teaching specs and topology reports
7. open traces only for intentionally profiled runs
8. use `artifact_index.json` as the map when the run directory gets crowded

## Troubleshooting

### No TPU devices are visible

Check the device report in `diagnostics/runtime.json` and the console log. Make
sure the script is running on the TPU VM or in the intended TPU environment.
For the default course path, expect four local TPU devices on the primary v5e-4
target.

### A Pallas communication kernel hangs

Suspect a synchronization mismatch first:

- a device waited for bytes that no sender produced
- sender and receiver disagree on payload size
- the wrong neighbor map was used
- a collective ID was reused for an incompatible pattern
- a device entered a different control path than its peers

Keep hang-prone bug demos isolated in Lab 4-style repros.

### A Pallas kernel crashes at completion

A likely cause is a nonzero semaphore state at program end, often from an
over-signal, missing wait, or mismatched send/receive byte count. Check the
error artifact and console log.

### A Pallas kernel runs out of VMEM

Whole-tile Lab 1 and Lab 2 remote-copy refs default to HBM because large
payload sweeps can exceed scoped VMEM. Use `--pallas-memory-space VMEM` only
for small payload experiments until chunked VMEM staging is introduced.

### Results are correct but slower than `lax`

That is normal and often the point. Built-in collectives can benefit from XLA
optimizations and runtime scheduling that custom Pallas kernels may obscure.
Use custom kernels to learn, to test algorithms, and to fuse specialized
communication with specialized compute. Do not assume custom means faster.

### Profiler output is noisy

Use `--profile-cases 1`, select one `--trace-op`, and choose one `--trace-size`.
Use non-profiled runs for clean timing tables.

### Lab 10 passes on v5e-4 but shows one process

That is expected for the primary single-process target. Lab 10 still validates
topology reporting, artifact writing, and the process-collective code path. The
multi-host part becomes meaningful on a larger slice where `jax.process_count()`
is greater than one.

## Adding A New Lab

When adding `labN`, update these places together:

```text
labs/labN_name.py
labs/labN_name.md
labs/roadmap.md
COURSE.md
README.md
collective_bench.py lab/profile registration
```

Each lab should define:

- concept being taught
- correctness reference
- input shape and dtype policy
- expected ownership layout
- benchmark sizes
- pass/fail criteria
- artifacts worth inspecting
- one known failure mode

A good lab is narrow enough to debug and rich enough to teach one durable
communication idea.

## Multi-Host Notes

Multi-host support is now represented by Lab 10 rather than being assumed for
every lab. When enabling it, the runner should record:

- `process_index`
- `process_count`
- local devices
- global devices
- process-local versus cross-process mesh phases
- per-process log paths
- one process-0 summary

All processes should enter distributed JAX consistently before device discovery
or device computation. Multi-host experiments should be treated as systems
experiments: logs and run metadata matter as much as the timing row.

A good first future target is a 16-chip v5e slice. Start by running Lab 10 with
explicit expectations, then port one simple reference collective before trying
a custom Pallas multi-host schedule.

## References

- Cloud TPU v5e documentation: https://docs.cloud.google.com/tpu/docs/v5e
- Cloud TPU system architecture: https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm
- JAX distributed Pallas for TPUs: https://docs.jax.dev/en/latest/pallas/tpu/distributed.html
- JAX profiling guide: https://docs.jax.dev/en/latest/profiling.html
