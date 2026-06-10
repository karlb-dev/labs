# TPU v5e Validation Report

This report records what the course is expected to do on 4-chip, 8-chip, and
16-chip TPU v5e slices. It is meant to help readers understand the shape of a
successful run without provisioning every target themselves.

The timings below are illustrative evidence, not grading thresholds. TPU timing
varies with runtime version, slice health, placement, warmup, and profiler
settings. The important result is the combination of zero correctness failures,
the expected process/device topology, and the relative performance lessons
shown by the rows.

## Run Context

- Date: 2026-06-09
- Runtime install used on fresh TPU-4 and TPU-8 runs: `jax[tpu]` with
  `jax==0.6.2`, `jaxlib==0.6.2`, and `libtpu==0.0.17`
- Hardware checked:
  - `v5litepod-4`: one process, 4 local/global TPU devices
  - `v5litepod-8`: one process, 8 local/global TPU devices
  - `v5litepod-16`: four processes, 4 local devices each, 16 global TPU devices
- Command shape:

```bash
python collective_bench.py --lab lab0 --no-plots
python collective_bench.py --lab lab1 --no-plots
# ...
python collective_bench.py --lab lab11 --no-plots
```

The results below are presented in the current Lab 0 through Lab 11 order.

## Pass Summary

All three targets ran the full course arc with zero failed rows.

| Target | Coverage | Cases | Failures | Shape observed |
| --- | --- | ---: | ---: | --- |
| `v5litepod-4` | Labs 0-11 | 147 | 0 | `process_count=1`, `global_device_count=4` |
| `v5litepod-8` | Labs 0-11 | 147 | 0 | `process_count=1`, `global_device_count=8` |
| `v5litepod-16` | Labs 0-11, all workers | 588 | 0 | `process_count=4`, `global_device_count=16` |

No stale lab-name references, sequence mismatches, or README/course-doc
mismatches were found after the lab reorder.

## How To Read The Targets

The 4-chip slice is the primary course target. It is small enough for fast
iteration and still exposes real ring, all-gather, reduce-scatter, all-reduce,
semaphore, and `2x2` mesh behavior.

The 8-chip slice is a scaling sanity check. The same lab sequence should still
pass, but flat-ring rows usually get longer because there are more hops. Rows
that reduce wire volume or use staged communication should preserve the same
qualitative ordering as the 4-chip run.

The 16-chip slice is the multi-process check. Each worker sees 4 local devices,
while JAX sees 16 global devices across 4 processes. This is the shape Lab 11
is designed to validate before any future multi-host custom Pallas collective
work.

## Representative Rows

These rows are useful sanity checks for the expected shape of the results. They
should not be read as exact required numbers. `n/a` means the row is a
correctness/topology smoke test rather than a bandwidth benchmark.

| Lab | Metric | TPU-4 us / GB/s | TPU-8 us / GB/s | TPU-16 worker0 us / GB/s |
| --- | --- | ---: | ---: | ---: |
| Lab 0 | baseline all-reduce, 32 MiB | 5705.6 / 8.82 | 14307.5 / 4.10 | 5971.8 / 10.54 |
| Lab 1 | single-hop Pallas copy, 32 MiB | 955.1 / 35.13 | 1072.0 / 31.30 | 960.4 / 34.94 |
| Lab 8 | double-buffer token ring, 1 MiB | 262.1 / 12.00 | 504.4 / 14.55 | 639.2 / 24.61 |
| Lab 9 | whole-token all-reduce foil, 16 MiB | 3756.2 / 13.40 | 9608.4 / 12.22 | 8411.8 / 29.92 |
| Lab 9 | reduce-scatter plus all-gather, 16 MiB | 826.8 / 30.44 | 1049.8 / 27.97 | 1023.1 / 30.75 |
| Lab 9 | bidirectional rs-ag, 16 MiB | 619.4 / 40.63 | 779.6 / 37.66 | 726.2 / 43.32 |
| Lab 9 | XLA roofline, 16 MiB | 522.6 / 48.16 | 668.9 / 43.89 | 640.3 / 49.13 |
| Lab 10 | staged 2D all-gather, 1 MiB | 258.5 / 12.17 | 525.5 / 13.97 | 677.7 / 23.21 |
| Lab 11 | process collective smoke, 1 KiB | 3671.7 / n/a | 3792.3 / n/a | 228101.2 / n/a |

## Course-Level Takeaways

- The 11-lab sequence is runnable end to end on the primary 4-chip v5e target.
- The same current code path also passes on an 8-chip v5e slice.
- The 16-chip v5e evidence covers the larger multi-process shape needed by
  Lab 11 and matches the semantic lab sequence after renumbering.
- Lab 9 shows the intended lesson: the whole-token ring is a byte-volume foil,
  reduce-scatter plus all-gather removes that penalty, bidirectional scheduling
  improves it further, and XLA remains the production roofline.
- Lab 10 follows topology as expected: 4-chip runs use a `2x2` mesh, 8-chip
  runs use a `2x4` mesh, and staged mesh behavior remains consistent.
- Lab 11 is useful both as a local smoke test and as the larger-slice gate for
  process count and global device count before future multi-host custom kernels.

## Lab-Specific Reading Guide

Labs 0-4 establish the measurement and correctness substrate. The results show
that built-in collectives, single-hop Pallas DMA, token-ring dependency chains,
local HBM/VMEM movement, and semaphore sanity checks all run correctly on the
tested slice sizes.

Labs 5-8 are teaching collectives. They are intentionally composed from visible
steps rather than presented as final production kernels. The important evidence
is that the rows pass consistently and that larger rings show the expected
extra hop cost.

Lab 9 is the main bandwidth-model lesson. The whole-token all-reduce row is
the deliberate foil: it moves too many bytes as the device count grows. The
reduce-scatter plus all-gather rows fix the byte model, the bidirectional
variant improves scheduling, and the XLA row remains the production roofline.

Lab 10 is the topology lesson. On 4 chips, the automatic mesh is `2x2`. On 8
chips, it becomes `2x4`. The staged mesh row should be read against that mesh
shape, not as a universal claim that one schedule wins on every topology.

Lab 11 is the launch-shape lesson. On 4-chip and 8-chip single-process slices,
it is mostly a smoke test. On the 16-chip slice, it validates the important
larger shape: 4 JAX processes, 4 local devices per process, and 16 global
devices entering the same collective.

## What This Does Not Prove

This report does not claim that every custom Pallas row is production-faster
than XLA. Several custom rows are intentionally slower because their job is to
make the communication schedule visible. It also does not prove future
multi-host custom Pallas kernels; it proves that the course sequence and the
multi-process gate are ready for that next step.
