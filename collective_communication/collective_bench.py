#!/usr/bin/env python3
"""Teaching benchmark harness for collective communication experiments.

This file is the executable "lab bench" for the collective-communication
course. The individual files under ``labs/`` own the concept implementations:
one-hop remote DMA, token rings, all-gather, reduce-scatter, all-reduce, mesh
staging, and multi-host smoke tests. This file owns the surrounding experiment
machinery:

* parse a CLI into a sweep of ``(operation, payload size, hop count)`` cases;
* set process environment that must exist before JAX is imported;
* build and time reference ``pmap``/``lax`` collectives;
* call the custom Pallas teaching kernels in ``labs/*.py``;
* validate small correctness signals before timing;
* capture logs, JSONL rows, CSVs, summaries, plots, traces, and memory profiles;
* keep failed cases as rows instead of aborting the whole sweep.

Why the harness is deliberately verbose
=======================================

Benchmark scripts often hide the boring parts: run directories, payload
rounding, warmup, ``block_until_ready()``, profiler capture, and row schemas.
Those details matter here because students are comparing algorithmic ideas with
runtime artifacts. A result row should explain:

* which operation ran;
* how many bytes the local device owned;
* what "logical bytes" means for that teaching implementation;
* whether the result passed a data-level check;
* where to find the trace, memory profile, or error artifact.

Quick start
===========

Run from the repository root:

    python3 collective_communication/collective_bench.py --help

Small CPU or single-process smoke test. This exercises the harness and the JAX
``pmap`` reference path, while Pallas TPU-only operations are left out:

    python3 collective_communication/collective_bench.py \\
      --ops pmap_ppermute,pmap_all_gather,pmap_psum \\
      --sizes 1KiB,16KiB \\
      --warmup 1 \\
      --iters 3 \\
      --no-plots

Typical Lab 1 TPU run. On a Cloud TPU VM, the TPU platform is usually selected
by the installed JAX/libtpu stack; setting ``JAX_PLATFORMS=tpu`` makes the
expectation explicit:

    JAX_PLATFORMS=tpu python3 collective_communication/collective_bench.py \\
      --lab lab1 \\
      --sizes 1KiB,16KiB,256KiB \\
      --warmup 5 \\
      --iters 50

Token-ring hop sweep for Lab 2:

    JAX_PLATFORMS=tpu python3 collective_communication/collective_bench.py \\
      --lab lab2 \\
      --sizes 1KiB,64KiB \\
      --token-hops 0,1,2,3

Capture one profiler trace and put XLA dumps inside the run directory:

    JAX_PLATFORMS=tpu python3 collective_communication/collective_bench.py \\
      --lab lab5 \\
      --sizes 64KiB \\
      --profile \\
      --profile-cases 1 \\
      --trace-op pallas_ring_all_gather \\
      --xla-dump-to run

Multi-host launch notes
=======================

The Lab 10 smoke operations record process topology and process collectives, but
this script intentionally does not launch other hosts by itself. Use the cluster
launcher for your environment, then run this same Python file on each process.
Common environment variables worth checking before launch are:

* ``JAX_COORDINATOR_ADDRESS`` and ``JAX_COORDINATOR_PORT``;
* ``JAX_PROCESS_COUNT`` and ``JAX_PROCESS_INDEX``;
* ``JAX_LOCAL_DEVICE_IDS`` when pinning local devices;
* ``TPU_*`` and ``PJRT_*`` variables supplied by the TPU runtime or launcher.

The harness records these variables in ``diagnostics/*.json`` and
``run_metadata.json`` so a run can be debugged after the VM is gone.

Important environment behavior
==============================

Some environment must be configured before ``import jax``. For that reason
``main()`` parses arguments first, creates the run directory, calls
``configure_env_pre_import()``, and only then imports JAX. The most important
settings are:

* ``TF_CPP_MIN_LOG_LEVEL`` defaults to ``2`` to keep console noise manageable.
* ``MPLCONFIGDIR`` is placed under the run directory so matplotlib can write in
  restricted or ephemeral environments.
* ``--xla-dump-to run`` appends XLA dump flags to ``XLA_FLAGS`` before JAX
  loads.
* User-provided ``JAX_*``, ``XLA_*``, ``TPU_*``, ``LIBTPU_*``, ``TF_*``, and
  ``PJRT_*`` variables are preserved and copied into the run metadata.

Outputs
=======

Every run creates a directory under ``collective_communication/runs`` unless
``--run-dir`` or ``--run-root`` says otherwise. The main artifacts are:

* ``logs/console.log``, ``stdout.log``, and ``stderr.log``;
* ``results.jsonl`` with one JSON row per benchmark case;
* ``csvs/results.csv`` plus success/failure subsets;
* ``run_config.json`` for the parsed CLI;
* ``run_metadata.json`` for package, device, git, and environment state;
* ``run_summary.json`` and ``results_summary.md``;
* ``plots/*.png`` unless ``--no-plots`` is passed;
* ``traces/*`` and ``memory/*.prof`` when profiling is enabled;
* ``errors/*.txt`` for failed cases with notes/stdout/stderr.

Operation map
=============

``pmap_*`` operations are reference implementations built from JAX ``lax``
collectives. ``pallas_*`` operations are custom TPU teaching kernels, usually
delegating to a specific lab module. ``*_spec`` operations write JSON/Markdown
teaching artifacts instead of timing a kernel. ``external`` lets another
executable plug into the same result-row contract by printing JSON to stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime
import importlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


# ---------------------------------------------------------------------------
# Operation groups
# ---------------------------------------------------------------------------
#
# The CLI lets the user request operations by name, while the lab shortcuts
# choose a curated subset for a teaching sequence. Keeping these tuples near the
# top makes the course map visible before any implementation details appear.
#
# Naming convention:
#   * pmap_*: reference path implemented in this harness with jax.pmap/lax.
#   * pallas_*: custom TPU teaching path, usually delegated to labs/*.py.
#   * lab*_spec: artifact-producing operation, not a timed kernel.
#   * external: subprocess hook that emits the same JSON row shape.
DEFAULT_OPS = (
    "pmap_ppermute",
    "pmap_all_gather",
    "pmap_psum",
    "pmap_all_to_all",
    "pallas_all_gather",
)

# Lab 0 is the baseline collection students can run before seeing custom Pallas
# kernels. It answers: "what do the built-in collectives do on this slice?"
LAB0_OPS = (
    "pmap_ppermute",
    "pmap_all_gather",
    "pmap_psum",
    "pmap_all_to_all",
)

# Lab 1 compares a built-in one-hop permutation with the first explicit Pallas
# remote-DMA neighbor copy.
LAB1_OPS = (
    "pmap_ppermute",
    "pallas_neighbor_copy",
)

# Lab 2 repeats Lab 1's hop into a dependent token ring. Hop count is sweepable
# for these ops, which is why they also appear in HOP_SWEEP_OPS below.
LAB2_OPS = (
    "pmap_token_ring",
    "pallas_token_ring",
)

# Lab 3 is intentionally local. It isolates memory-space and tiling behavior
# before the labs return to network movement.
LAB3_OPS = (
    "pmap_local_arith",
    "pallas_vmem_arith",
)

# Lab 4 runs the safe correct probe, one safe runnable bug demo (a correctness
# failure that completes cleanly), and the catalog/spec artifact. Hang/crash/race
# bugs are catalog-only by default and only run through the guarded subprocess
# path; see run_pallas_semaphore_bug and run_guarded_subprocess.
LAB4_OPS = (
    "pallas_semaphore_correct",
    "pallas_semaphore_bug",
    "semaphore_bug_zoo",
)

# Labs 5-9 compare a built-in/reference path, a custom Pallas path, and a spec
# artifact where useful. The spec artifacts make run directories double as lab
# handouts.
LAB5_OPS = (
    "pmap_ring_all_gather",
    "pmap_all_gather",
    "pallas_ring_all_gather",
    "pallas_all_gather",
    "lab5_ring_all_gather_spec",
)

LAB6_OPS = (
    "pmap_psum_scatter",
    "pallas_ring_reduce_scatter",
    "lab6_reduce_scatter_spec",
)

LAB7_OPS = (
    "pmap_psum",
    "pallas_ring_all_reduce",
    "lab7_all_reduce_spec",
)

# Lab 8 compares three implementations of the same token-ring reduction plus the
# pmap baseline: the serialized teaching path, the fused custom double-buffered
# kernel (the "make it fast" payoff), and the XLA tuned-collective roofline.
LAB8_OPS = (
    "pmap_token_ring",
    "pallas_chunked_token_ring",
    "pallas_db_token_ring",
    "xla_token_ring",
    "lab8_chunked_pipeline_spec",
)

LAB9_OPS = (
    "pmap_all_gather",
    "pmap_2d_staged_all_gather",
    "pallas_2d_staged_all_gather",
    "lab9_mesh_collectives_spec",
)

# Lab 10 is about launch topology and process collectives. It is grouped here
# with the timed ops so the same sweep/report machinery can record it.
LAB10_OPS = (
    "lab10_topology_smoke",
    "lab10_process_collective_smoke",
    "lab10_multihost_spec",
)

# Lab 11 closes the byte gap: the same all-reduce as reduce-scatter plus
# all-gather over B/N shards, matching lax.psum's 2*(N-1)/N*B volume. The
# whole-token pmap_token_ring rides along as the N/2-penalty foil.
LAB11_OPS = (
    "pmap_psum",
    "pmap_token_ring",
    "pmap_rs_ag_all_reduce",
    "pmap_rs_ag_all_reduce_bidir",
    "xla_all_reduce",
    "lab11_optimal_all_reduce_spec",
)

# Fast membership tests for dispatch. The names in these sets are not arbitrary:
# PMAP_OPS are safe to build through run_pmap_op(), while PALLAS_OPS are TPU-only
# custom-kernel paths with their own case builders.
PMAP_OPS = {
    "pmap_psum",
    "pmap_all_gather",
    "pmap_all_to_all",
    "pmap_ppermute",
    "pmap_token_ring",
    "pmap_local_arith",
    "pmap_ring_all_gather",
    "pmap_psum_scatter",
    "pmap_2d_staged_all_gather",
}

# PALLAS_OPS is currently mostly documentation plus future-proofing. Dispatch
# below uses explicit if/elif branches because each custom op has different lab
# metadata, correctness checks, and byte models.
PALLAS_OPS = {
    "pallas_all_gather",
    "pallas_neighbor_copy",
    "pallas_token_ring",
    "pallas_ring_all_gather",
    "pallas_ring_reduce_scatter",
    "pallas_ring_all_reduce",
    "pallas_chunked_token_ring",
    "pallas_db_token_ring",
    "pallas_2d_staged_all_gather",
    "pallas_vmem_arith",
    "pallas_semaphore_correct",
    "pallas_semaphore_bug",
}

# Ops whose behavior depends on the ring hop count, so a --token-hops sweep
# expands into one case per hop value. Every other op runs once regardless of
# the hop sweep, since the hop count does not change what they do.
#
# The Lab 5/6 ring collectives are built from a neighbor-copy hop loop, so a
# partial hop count produces a partial all-gather / reduce-scatter whose
# expected output the lab modules compute exactly. That makes the documented
# "compare hops=0,1,N-1" experiments runnable straight from the CLI. The
# built-in reduce-scatter reference (pmap_psum_scatter) is deliberately absent:
# it lowers to a single atomic lax.psum_scatter with no hop loop to shorten, so
# it can only ever do the full reduction.
HOP_SWEEP_OPS = {
    "pmap_token_ring",
    "pallas_token_ring",
    "pallas_chunked_token_ring",
    "pallas_db_token_ring",
    "xla_token_ring",
    "pmap_ring_all_gather",
    "pallas_ring_all_gather",
    "pallas_ring_reduce_scatter",
}

LAB_SPEC_OPS = {
    "lab5_ring_all_gather_spec": "labs.lab5_ring_all_gather",
    "lab6_reduce_scatter_spec": "labs.lab6_reduce_scatter",
    "lab7_all_reduce_spec": "labs.lab7_all_reduce",
    "lab8_chunked_pipeline_spec": "labs.lab8_chunked_pipeline",
    "lab9_mesh_collectives_spec": "labs.lab9_mesh_collectives",
    "lab10_multihost_spec": "labs.lab10_multihost_smoke",
    "lab11_optimal_all_reduce_spec": "labs.lab11_optimal_all_reduce",
}

# ALL_OPS is the validation list for --ops. If a new operation should be
# user-selectable, it belongs here and in dispatch_case().
ALL_OPS = (
    *DEFAULT_OPS,
    "pallas_neighbor_copy",
    "pmap_token_ring",
    "pallas_token_ring",
    "pallas_ring_all_gather",
    "pallas_ring_reduce_scatter",
    "pallas_ring_all_reduce",
    "pallas_chunked_token_ring",
    "pallas_db_token_ring",
    "xla_token_ring",
    "pallas_2d_staged_all_gather",
    "pmap_local_arith",
    "pallas_vmem_arith",
    "pallas_semaphore_correct",
    "pallas_semaphore_bug",
    "semaphore_bug_zoo",
    "pmap_ring_all_gather",
    "pmap_psum_scatter",
    "pmap_2d_staged_all_gather",
    "pmap_rs_ag_all_reduce",
    "pmap_rs_ag_all_reduce_bidir",
    "xla_all_reduce",
    *LAB_SPEC_OPS,
    "lab10_topology_smoke",
    "lab10_process_collective_smoke",
    "external",
)

# Run directories are intentionally inside the course folder by default so a
# student can inspect code, result rows, generated lab specs, and plots together.
DEFAULT_RUN_ROOT = pathlib.Path(__file__).resolve().parent / "runs"

# Human-friendly byte suffixes accepted by --sizes and --trace-size. Decimal and
# binary suffixes are both supported because benchmark writeups often mix them.
SIZE_SUFFIXES = {
    "b": 1,
    "k": 1_000,
    "kb": 1_000,
    "m": 1_000_000,
    "mb": 1_000_000,
    "g": 1_000_000_000,
    "gb": 1_000_000_000,
    "ki": 1024,
    "kib": 1024,
    "mi": 1024**2,
    "mib": 1024**2,
    "gi": 1024**3,
    "gib": 1024**3,
}


@dataclasses.dataclass(frozen=True)
class BenchResult:
    """In-memory result returned by every operation runner.

    The harness has many different kinds of cases: JAX reference collectives,
    Pallas kernels, spec artifact generators, multi-host smoke checks, and
    external subprocesses. This dataclass is the common contract between those
    case-specific runners and the generic reporting pipeline.

    ``payload_bytes`` is the actual per-local-device payload after shape and
    dtype rounding.

    Two byte models keep the GB/s columns honest:

    * ``logical_bytes`` is the *useful* (comparable) byte model: the bytes an
      optimal implementation of this collective would have to move to deliver the
      same result. It is defined the same way for the built-in baseline and the
      custom kernel of a given collective, so the headline ``gbps`` is an
      apples-to-apples "effective throughput toward the result" number within a
      collective family.
    * ``wire_bytes`` is the *actual* per-device traffic this particular
      implementation pushes. For efficient paths it equals ``logical_bytes``; for
      the deliberately wasteful teaching kernels (e.g. Lab 6/7 whole-token
      reduce-scatter) it is larger, and ``wire_gbps`` together with the
      ``wire_bytes / logical_bytes`` ratio exposes that overhead.

    ``byte_model`` is a short label for the implementation's wire model
    ("optimal", "whole-token", "serialized", ...). Cross-op comparisons should
    use the ``us`` latency column, which needs no byte model at all. Failed
    setup/run cases keep ``seconds=None`` and explain themselves through ``note``
    and artifacts.
    """

    op: str
    layer: str
    payload_bytes: int
    logical_bytes: int
    seconds: float | None
    ok: bool
    note: str = ""
    # Actual per-device wire traffic for this implementation. Defaults to
    # logical_bytes in __post_init__ when a runner does not set it (i.e. the
    # implementation is already optimal, so useful == wire).
    wire_bytes: int | None = None
    byte_model: str = "optimal"
    trace_artifact: str | None = None
    memory_profile_artifact: str | None = None
    error_artifact: str | None = None
    external_stdout: str | None = None
    external_stderr: str | None = None
    timing_mean_s: float | None = None
    timing_p10_s: float | None = None
    timing_p50_s: float | None = None
    timing_p90_s: float | None = None
    timing_p99_s: float | None = None
    timing_min_s: float | None = None
    timing_max_s: float | None = None
    timing_samples_s: tuple[float, ...] = ()
    # Per-device received-rank evidence for single-hop / ring ops. For a right
    # neighbor copy on 4 devices this is [3, 0, 1, 2]: each device shows the rank
    # it received from its left neighbor. None for ops where it does not apply.
    observed_ranks: tuple[int, ...] | None = None
    expected_ranks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        # Frozen dataclass: default wire_bytes to the useful model when a runner
        # did not provide a separate wire figure (efficient path: useful==wire).
        if self.wire_bytes is None:
            object.__setattr__(self, "wire_bytes", self.logical_bytes)

    @property
    def us(self) -> float | None:
        """Median/p50 time in microseconds, matching the terminal table."""
        if self.seconds is None:
            return None
        return self.seconds * 1e6

    @property
    def gbps(self) -> float | None:
        """Useful GB/s: comparable optimal-model bytes over p50 time."""
        if self.seconds is None or self.seconds == 0:
            return None
        return self.logical_bytes / self.seconds / 1e9

    @property
    def wire_gbps(self) -> float | None:
        """Wire GB/s: this implementation's actual traffic over p50 time."""
        if self.seconds is None or self.seconds == 0 or self.wire_bytes is None:
            return None
        return self.wire_bytes / self.seconds / 1e9


@dataclasses.dataclass(frozen=True)
class TimingStats:
    """Raw timing samples plus percentile helpers.

    The benchmark reports p50 as the headline latency, but keeping p10/p90/p99
    and the raw samples makes the run more useful when TPU startup, compilation
    cache behavior, or device contention introduces noise.
    """

    samples_s: tuple[float, ...]

    def percentile(self, pct: float) -> float | None:
        """Linear-interpolated percentile over the collected seconds samples."""
        vals = sorted(self.samples_s)
        if not vals:
            return None
        if len(vals) == 1:
            return vals[0]
        rank = (len(vals) - 1) * pct / 100.0
        lo = math.floor(rank)
        hi = math.ceil(rank)
        if lo == hi:
            return vals[int(rank)]
        return vals[lo] * (hi - rank) + vals[hi] * (rank - lo)

    @property
    def mean(self) -> float | None:
        if not self.samples_s:
            return None
        return sum(self.samples_s) / len(self.samples_s)

    @property
    def p10(self) -> float | None:
        return self.percentile(10)

    @property
    def p50(self) -> float | None:
        return self.percentile(50)

    @property
    def p90(self) -> float | None:
        return self.percentile(90)

    @property
    def p99(self) -> float | None:
        return self.percentile(99)

    @property
    def min(self) -> float | None:
        return min(self.samples_s) if self.samples_s else None

    @property
    def max(self) -> float | None:
        return max(self.samples_s) if self.samples_s else None


@dataclasses.dataclass
class RunContext:
    """Mutable state shared by the sweep and artifact writers.

    ``args`` is the parsed argparse namespace after lab defaults are applied.
    ``results`` accumulates already-recorded row dictionaries, not BenchResult
    objects, because downstream writers need the fully flattened row schema.
    ``case_index`` is incremented by the main loop before each dispatch and is
    used to name trace/error/spec artifacts.
    """

    run_dir: pathlib.Path
    args: argparse.Namespace
    started_at_unix: float = dataclasses.field(default_factory=time.time)
    results: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    profile_count: int = 0
    memory_profile_count: int = 0
    case_index: int = 0

    @property
    def results_jsonl(self) -> pathlib.Path:
        """Append-only row log, safe to inspect while a long sweep runs."""
        return self.run_dir / "results.jsonl"

    @property
    def csv_dir(self) -> pathlib.Path:
        """Directory for post-run CSV exports derived from results.jsonl."""
        return self.run_dir / "csvs"

    @property
    def plot_dir(self) -> pathlib.Path:
        """Directory for matplotlib plots and trace-derived communication plots."""
        return self.run_dir / "plots"

    @property
    def errors_dir(self) -> pathlib.Path:
        """Directory for expanded failure notes, tracebacks, and external output."""
        return self.run_dir / "errors"


class FdTee:
    """Capture stdout/stderr at the file-descriptor level while echoing them.

    Python-level ``redirect_stdout`` is not enough for this harness because JAX,
    XLA, libtpu, subprocesses, and native extensions can write directly to file
    descriptors 1 and 2. ``FdTee`` redirects those descriptors through pipes,
    mirrors bytes back to the original terminal, and writes separate plus
    combined log files under the run directory.
    """

    def __init__(self, log_dir: pathlib.Path) -> None:
        self.log_dir = log_dir
        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self._stdout_read_fd: int | None = None
        self._stderr_read_fd: int | None = None
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._combined_file: Any = None
        self._stdout_file: Any = None
        self._stderr_file: Any = None

    def __enter__(self) -> "FdTee":
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Flush Python buffers before swapping descriptors so no bytes are left
        # pointing at the old terminal-only streams.
        sys.stdout.flush()
        sys.stderr.flush()
        self._combined_file = (self.log_dir / "console.log").open("ab", buffering=0)
        self._stdout_file = (self.log_dir / "stdout.log").open("ab", buffering=0)
        self._stderr_file = (self.log_dir / "stderr.log").open("ab", buffering=0)
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        # Each stream gets its own pipe and pump thread. The write side becomes
        # the process stdout/stderr; the read side is consumed by _pump().
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        self._stdout_read_fd = stdout_r
        self._stderr_read_fd = stderr_r
        os.dup2(stdout_w, 1)
        os.dup2(stderr_w, 2)
        os.close(stdout_w)
        os.close(stderr_w)
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(line_buffering=True)
        with contextlib.suppress(Exception):
            sys.stderr.reconfigure(line_buffering=True)
        # Daemon threads are acceptable here because __exit__ joins them after
        # restoring the original descriptors. They should not outlive the run.
        self._threads = [
            threading.Thread(
                target=self._pump,
                args=(stdout_r, self._saved_stdout_fd, self._stdout_file),
                daemon=True,
            ),
            threading.Thread(
                target=self._pump,
                args=(stderr_r, self._saved_stderr_fd, self._stderr_file),
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()
        return self

    def _pump(self, read_fd: int, mirror_fd: int, stream_file: Any) -> None:
        """Copy pipe bytes to the terminal mirror, stream log, and combined log."""
        try:
            while True:
                data = os.read(read_fd, 8192)
                if not data:
                    break
                os.write(mirror_fd, data)
                stream_file.write(data)
                with self._lock:
                    self._combined_file.write(data)
        finally:
            with contextlib.suppress(OSError):
                os.close(read_fd)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Restore descriptors first. Closing/restoring the write end lets pump
        # threads observe EOF and finish flushing the log files.
        sys.stdout.flush()
        sys.stderr.flush()
        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)
        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)
        for thread in self._threads:
            thread.join(timeout=2.0)
        for fd in (self._saved_stdout_fd, self._saved_stderr_fd):
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
        for f in (self._stdout_file, self._stderr_file, self._combined_file):
            if f is not None:
                with contextlib.suppress(Exception):
                    f.close()


def now_slug() -> str:
    """Timestamp fragment used in auto-generated run names."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_tag(text: Any) -> str:
    """Return a filesystem-friendly tag for artifact names.

    Long notes and operation names can contain spaces, punctuation, or traceback
    text. Artifact paths should stay readable but not exceed common filename
    limits, so this keeps a conservative character set and truncates.
    """
    tag = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(text)).strip("_")
    return tag[:180] or "untitled"


def json_default(obj: Any) -> Any:
    """JSON fallback for dataclasses, paths, arrays, dtypes, and odd objects."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        with contextlib.suppress(Exception):
            return obj.tolist()
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {
            "shape": list(getattr(obj, "shape", [])),
            "dtype": str(getattr(obj, "dtype", "")),
        }
    return repr(obj)


def write_json(path: pathlib.Path, payload: Any) -> None:
    """Write a deterministic, human-readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """Append one result row so progress survives even if a later case fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(payload), sort_keys=True, default=json_default) + "\n")


def flatten_for_csv(value: Any) -> Any:
    """Turn nested row values into one CSV cell without losing structure."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=json_default)
    return value


def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows with the union of keys in first-seen order.

    Different operation families produce different optional fields. Building the
    header from all rows keeps CSV exports complete without requiring every row
    to know every possible column.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten_for_csv(row.get(key)) for key in keys})


def package_version(name: str) -> str | None:
    """Return an installed package version, or None when unavailable."""
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def env_subset() -> dict[str, str]:
    """Capture environment variables that commonly affect JAX/TPU behavior."""
    prefixes = ("JAX_", "XLA_", "TPU_", "LIBTPU_", "TF_", "PJRT_")
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes)}


def read_text_if_exists(path: pathlib.Path) -> str | None:
    """Best-effort read for Linux diagnostic files that may not exist."""
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"
    return None


def git_state(cwd: pathlib.Path) -> dict[str, Any]:
    """Record repository identity without making git a hard dependency."""
    def run(cmd: list[str]) -> str | None:
        """Run a small git command and return stdout when it succeeds."""
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    status = run(["git", "status", "--porcelain"])
    return {
        "sha": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def make_run_dir(args: argparse.Namespace) -> pathlib.Path:
    """Resolve the run directory, avoiding accidental overwrite by default."""
    if args.run_dir:
        return pathlib.Path(args.run_dir).expanduser().resolve()
    run_root = pathlib.Path(args.run_root).expanduser().resolve()
    run_name = args.run_name or f"collective_bench-{now_slug()}-{uuid.uuid4().hex[:6]}"
    base = run_root / sanitize_tag(run_name)
    if not base.exists():
        return base
    for i in range(2, 10_000):
        candidate = base.with_name(f"{base.name}_{i:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find an unused run directory for {base}")


def configure_env_pre_import(args: argparse.Namespace, run_dir: pathlib.Path) -> None:
    """Set environment variables that must exist before importing JAX.

    JAX reads many XLA and backend settings at import time. This helper runs
    before ``import jax`` in ``main()`` so flags like ``--xla-dump-to`` affect
    compilation in the current process.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    mpl_config = run_dir / "matplotlib_config"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    if args.xla_dump_to:
        dump_to = pathlib.Path(args.xla_dump_to).expanduser()
        if str(args.xla_dump_to).lower() == "run":
            dump_to = run_dir / "xla_dumps"
        dump_to.mkdir(parents=True, exist_ok=True)
        existing = os.environ.get("XLA_FLAGS", "")
        # Append only missing flags so a caller can still provide additional XLA
        # settings through the environment.
        for flag in (
            f"--xla_dump_to={dump_to}",
            "--xla_dump_hlo_as_text",
            "--xla_dump_hlo_as_proto",
        ):
            if flag not in existing:
                existing = (existing + " " + flag).strip()
        os.environ["XLA_FLAGS"] = existing


def device_report(jax: Any) -> dict[str, Any]:
    """Serialize the visible JAX device topology for metadata artifacts."""
    devices = []
    for device in jax.devices():
        devices.append(
            {
                "id": getattr(device, "id", None),
                "process_index": getattr(device, "process_index", None),
                "coords": getattr(device, "coords", None),
                "core_on_chip": getattr(device, "core_on_chip", None),
                "platform": getattr(device, "platform", None),
                "device_kind": getattr(device, "device_kind", None),
                "local_hardware_id": getattr(device, "local_hardware_id", None),
                "repr": repr(device),
            }
        )
    return {
        "backend": jax.default_backend(),
        "device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "process_index": int(getattr(jax, "process_index", lambda: 0)()),
        "process_count": int(getattr(jax, "process_count", lambda: 1)()),
        "devices": devices,
    }


def collect_diagnostics(jax: Any | None = None) -> dict[str, Any]:
    """Collect host, package, git, environment, and optional JAX diagnostics."""
    diag = {
        "time_unix": time.time(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "env": env_subset(),
        "transparent_hugepage_enabled": read_text_if_exists(
            pathlib.Path("/sys/kernel/mm/transparent_hugepage/enabled")
        ),
        "transparent_hugepage_defrag": read_text_if_exists(
            pathlib.Path("/sys/kernel/mm/transparent_hugepage/defrag")
        ),
        "packages": {
            "jax": package_version("jax"),
            "jaxlib": package_version("jaxlib"),
            "libtpu": package_version("libtpu"),
            "matplotlib": package_version("matplotlib"),
            "numpy": package_version("numpy"),
        },
        "git": git_state(pathlib.Path.cwd()),
    }
    if jax is not None:
        with contextlib.suppress(Exception):
            diag["device_report"] = device_report(jax)
    return diag


def sample_device_memory(jax: Any, label: str) -> dict[str, Any]:
    """Best-effort snapshot of per-local-device memory stats."""
    per_device: list[dict[str, Any]] = []
    max_in_use = 0
    max_limit = 0
    for device in jax.local_devices():
        try:
            stats = dict(device.memory_stats() or {})
        except Exception as exc:
            stats = {"unavailable": True, "error": repr(exc)}
        bytes_in_use = int(stats.get("bytes_in_use") or stats.get("peak_bytes_in_use") or 0)
        bytes_limit = int(stats.get("bytes_limit") or stats.get("bytes_reserved") or 0)
        max_in_use = max(max_in_use, bytes_in_use)
        max_limit = max(max_limit, bytes_limit)
        per_device.append(
            {
                "id": getattr(device, "id", None),
                "process_index": getattr(device, "process_index", None),
                "bytes_in_use": bytes_in_use,
                "bytes_limit": bytes_limit,
                "raw": stats,
            }
        )
    return {
        "label": label,
        "time_unix": time.time(),
        "bytes_in_use_max": max_in_use,
        "bytes_limit_max": max_limit,
        "per_device": per_device,
    }


def parse_size(text: str) -> int:
    """Parse human-friendly byte strings such as ``1KiB`` or ``4MiB``."""
    raw = text.strip()
    if not raw:
        raise argparse.ArgumentTypeError("empty size")
    split_at = len(raw)
    for i, char in enumerate(raw):
        if not (char.isdigit() or char == "."):
            split_at = i
            break
    number = raw[:split_at]
    suffix = raw[split_at:].strip().lower()
    if not number:
        raise argparse.ArgumentTypeError(f"invalid size: {text!r}")
    if suffix == "":
        suffix = "b"
    if suffix not in SIZE_SUFFIXES:
        raise argparse.ArgumentTypeError(f"unknown size suffix in {text!r}")
    return int(float(number) * SIZE_SUFFIXES[suffix])


def parse_csv(value: str, item_parser: Callable[[str], Any] = str) -> list[Any]:
    """Parse comma-separated CLI values, dropping empty items after stripping."""
    return [item_parser(item) for item in value.split(",") if item.strip()]


def parse_token_hops(value: str | None) -> list[int | None]:
    """Parse the ``--token-hops`` sweep into a list of hop values.

    Accepts a single value or a comma-separated sweep, for example ``3`` or
    ``0,1,2,3``. ``None`` (flag omitted) yields ``[None]`` so hop-dependent ops
    fall back to their default of ``n_devices - 1`` hops. A negative value such
    as ``-1`` is kept verbatim and later resolved to the full ring, matching the
    ``lab2_token_ring`` convention.
    """
    if value is None:
        return [None]
    hops = [int(item) for item in value.split(",") if item.strip()]
    return hops or [None]


def apply_lab_defaults(args: argparse.Namespace) -> None:
    """Fill in ops/sizes/run-name defaults for the selected lab profile.

    The lab shortcuts are convenience presets, not hidden modes. After this
    function runs, the rest of the harness sees ordinary ``--ops`` and
    ``--sizes`` values. A user can override either field explicitly.
    """
    if args.lab is None:
        if args.ops is None:
            args.ops = "all"
        if args.sizes is None:
            args.sizes = "1KiB,16KiB,256KiB,4MiB,32MiB"
        return

    if args.lab == "lab0":
        # Lab 0 is the built-in collective baseline. Larger sizes are useful
        # because no custom Pallas tile constraints apply.
        if args.ops is None:
            args.ops = ",".join(LAB0_OPS)
        if args.sizes is None:
            args.sizes = "1KiB,16KiB,256KiB,4MiB,32MiB"
        if args.run_name is None:
            args.run_name = f"lab0_baselines-{now_slug()}"
        return

    if args.lab == "lab1":
        # Lab 1 focuses on the one-hop pattern, so it compares ppermute with
        # the first Pallas neighbor-copy kernel over a broad payload sweep.
        if args.ops is None:
            args.ops = ",".join(LAB1_OPS)
        if args.sizes is None:
            args.sizes = "1KiB,16KiB,256KiB,4MiB,32MiB"
        if args.run_name is None:
            args.run_name = f"lab1_single_hop-{now_slug()}"
        return

    if args.lab == "lab2":
        # Lab 2 repeats one-hop movement. The default size ceiling is slightly
        # lower to keep token-ring experiments approachable.
        if args.ops is None:
            args.ops = ",".join(LAB2_OPS)
        if args.sizes is None:
            args.sizes = "1KiB,16KiB,256KiB,4MiB"
        if args.run_name is None:
            args.run_name = f"lab2_token_ring-{now_slug()}"
        return

    if args.lab == "lab3":
        # Lab 3 isolates local memory-space behavior from network movement.
        if args.ops is None:
            args.ops = ",".join(LAB3_OPS)
        if args.sizes is None:
            args.sizes = "1KiB,16KiB,256KiB,4MiB"
        if args.run_name is None:
            args.run_name = f"lab3_memory_spaces-{now_slug()}"
        return

    if args.lab == "lab4":
        # Lab 4's bug zoo is about correctness and synchronization patterns, so
        # one small payload is enough for the default run.
        if args.ops is None:
            args.ops = ",".join(LAB4_OPS)
        if args.sizes is None:
            args.sizes = "1KiB"
        if args.run_name is None:
            args.run_name = f"lab4_semaphore_bug_zoo-{now_slug()}"
        return

    if args.lab == "lab5":
        # Lab 5 introduces all-gather. The default includes both reference and
        # Pallas versions plus an explanatory spec artifact.
        if args.ops is None:
            args.ops = ",".join(LAB5_OPS)
        if args.sizes is None:
            args.sizes = "1KiB,64KiB,1MiB"
        if args.run_name is None:
            args.run_name = f"lab5_ring_all_gather-{now_slug()}"
        return

    if args.lab == "lab6":
        # Lab 6 adds reduction ownership and a reduce-scatter byte model.
        if args.ops is None:
            args.ops = ",".join(LAB6_OPS)
        if args.sizes is None:
            args.sizes = "1KiB,64KiB,1MiB"
        if args.run_name is None:
            args.run_name = f"lab6_reduce_scatter-{now_slug()}"
        return

    if args.lab == "lab7":
        # Lab 7 composes reduce-scatter plus all-gather into all-reduce.
        if args.ops is None:
            args.ops = ",".join(LAB7_OPS)
        if args.sizes is None:
            args.sizes = "1KiB,64KiB,1MiB"
        if args.run_name is None:
            args.run_name = f"lab7_all_reduce-{now_slug()}"
        return

    if args.lab == "lab8":
        # Lab 8 studies chunking/pipeline structure, so the default starts at a
        # larger size where chunks are meaningful.
        if args.ops is None:
            args.ops = ",".join(LAB8_OPS)
        if args.sizes is None:
            args.sizes = "16KiB,64KiB,256KiB,1MiB"
        if args.run_name is None:
            args.run_name = f"lab8_chunked_pipeline-{now_slug()}"
        return

    if args.lab == "lab9":
        # Lab 9 compares flat all-gather with staged logical-2D mesh movement.
        if args.ops is None:
            args.ops = ",".join(LAB9_OPS)
        if args.sizes is None:
            args.sizes = "1KiB,64KiB,1MiB"
        if args.run_name is None:
            args.run_name = f"lab9_mesh_collectives-{now_slug()}"
        return

    if args.lab == "lab10":
        # Lab 10 is topology/control-plane validation; payload size is mostly a
        # marker carried through the common row schema.
        if args.ops is None:
            args.ops = ",".join(LAB10_OPS)
        if args.sizes is None:
            args.sizes = "1KiB"
        if args.run_name is None:
            args.run_name = f"lab10_multihost_smoke-{now_slug()}"
        return

    if args.lab == "lab11":
        # Lab 11 is the bandwidth-optimal shard ring; the default sweep spans
        # the alpha-beta crossover (small sizes, where the naive ring wins on
        # latency) through the bandwidth regime where matching psum's volume
        # pays off.
        if args.ops is None:
            args.ops = ",".join(LAB11_OPS)
        if args.sizes is None:
            args.sizes = "16KiB,256KiB,1MiB,4MiB,16MiB"
        if args.run_name is None:
            args.run_name = f"lab11_optimal_all_reduce-{now_slug()}"
        return

    raise ValueError(f"unknown lab {args.lab!r}")


def format_bytes(num_bytes: int) -> str:
    """Format bytes for compact terminal and markdown tables."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ("KiB", "MiB", "GiB")
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} TiB"


def dtype_from_name(jnp: Any, name: str) -> Any:
    """Map CLI dtype names to JAX dtype objects."""
    aliases = {
        "bf16": jnp.bfloat16,
        "bfloat16": jnp.bfloat16,
        "f32": jnp.float32,
        "float32": jnp.float32,
        "i32": jnp.int32,
        "int32": jnp.int32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def dtype_itemsize(jnp: Any, dtype: Any) -> int:
    """Return dtype item size through JAX so aliases resolve consistently."""
    return int(jnp.dtype(dtype).itemsize)


def elems_for_payload(payload_bytes: int, itemsize: int, divisor: int = 1) -> int:
    """Round a byte request to an element count.

    ``divisor`` is used for shapes that split the payload into equal chunks
    across devices. The result is always at least one element so tiny smoke tests
    still produce a valid array.
    """
    denom = max(1, itemsize * divisor)
    return max(1, math.ceil(payload_bytes / denom))


def block_until_ready(jax: Any, value: Any) -> Any:
    """Synchronize a JAX result tree so timings measure completed work.

    JAX dispatch is asynchronous. Without this helper the timer would often
    measure enqueue time rather than device execution time.
    """
    def block(x: Any) -> Any:
        """Block a leaf when it is a JAX array-like object."""
        if hasattr(x, "block_until_ready"):
            return x.block_until_ready()
        return x

    return jax.tree_util.tree_map(block, value)


@contextlib.contextmanager
def maybe_trace(
    jax: Any,
    args: argparse.Namespace,
    run: RunContext,
    op: str,
    payload_bytes: int,
):
    """Conditionally capture a JAX profiler trace for one benchmark case.

    The context manager yields a run-relative artifact path when tracing starts,
    or ``None`` when profiling is disabled or this case is outside the requested
    trace filter. The caller records that yielded value in the result row.
    """
    enabled = bool(args.profile or args.trace_dir)
    if not enabled:
        yield None
        return
    if args.trace_op is not None and args.trace_op != op:
        yield None
        return
    if args.trace_size is not None and args.trace_size != payload_bytes:
        yield None
        return
    if run.profile_count >= max(0, int(args.profile_cases)):
        yield None
        return

    trace_root = pathlib.Path(args.trace_dir).expanduser() if args.trace_dir else run.run_dir / "traces"
    tag = f"{run.case_index:04d}_{sanitize_tag(op)}_{payload_bytes}"
    trace_dir = trace_root / tag
    trace_dir.mkdir(parents=True, exist_ok=True)
    run.profile_count += 1
    try:
        # JAX writes a trace directory containing Chrome/Perfetto-style files.
        # The post-run trace summarizer below knows how to inspect the resulting
        # *.trace.json.gz files for TPU communication events.
        jax.profiler.start_trace(
            str(trace_dir),
            create_perfetto_link=False,
            create_perfetto_trace=args.perfetto,
        )
        yield relative_artifact(run.run_dir, trace_dir)
    finally:
        with contextlib.suppress(Exception):
            jax.profiler.stop_trace()


def relative_artifact(run_dir: pathlib.Path, path: pathlib.Path) -> str:
    """Prefer run-relative artifact paths in result rows and summaries."""
    with contextlib.suppress(ValueError):
        return str(path.relative_to(run_dir))
    return str(path)


def maybe_write_memory_profile(
    jax: Any,
    args: argparse.Namespace,
    run: RunContext,
    op: str,
    payload_bytes: int,
) -> str | None:
    """Optionally save one JAX device-memory profile artifact for this case."""
    if not args.memory_profiles:
        return None
    if run.memory_profile_count >= max(0, int(args.profile_cases)):
        return None
    tag = f"{run.case_index:04d}_{sanitize_tag(op)}_{payload_bytes}"
    path = run.run_dir / "memory" / f"{tag}.prof"
    path.parent.mkdir(parents=True, exist_ok=True)
    run.memory_profile_count += 1
    try:
        jax.profiler.save_device_memory_profile(str(path))
        return relative_artifact(run.run_dir, path)
    except Exception as exc:
        err = path.with_suffix(".error.txt")
        err.write_text(repr(exc), encoding="utf-8")
        return relative_artifact(run.run_dir, err)


def time_jax_call(
    jax: Any,
    fn: Callable[[Any], Any],
    x: Any,
    *,
    warmup: int,
    iters: int,
) -> TimingStats:
    """Warm up and time a single JAX callable.

    The warmup loop compiles and primes caches before collecting samples. Each
    measured iteration blocks on the output so the returned samples represent
    completed device work.
    """
    for _ in range(warmup):
        block_until_ready(jax, fn(x))

    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        y = fn(x)
        block_until_ready(jax, y)
        end = time.perf_counter()
        samples.append(end - start)
    return TimingStats(tuple(samples))


def result_timing_kwargs(timing: TimingStats) -> dict[str, Any]:
    """Convert TimingStats into BenchResult keyword arguments."""
    return {
        "seconds": timing.p50,
        "timing_mean_s": timing.mean,
        "timing_p10_s": timing.p10,
        "timing_p50_s": timing.p50,
        "timing_p90_s": timing.p90,
        "timing_p99_s": timing.p99,
        "timing_min_s": timing.min,
        "timing_max_s": timing.max,
        "timing_samples_s": timing.samples_s,
    }


def rank_vector(values: Any) -> tuple[int, ...]:
    """Flatten a per-device scalar array into a rounded int tuple.

    Used to surface the observed received-rank pattern (the data-level evidence
    of a single-hop copy or ring) in result rows.
    """
    import numpy as np

    flat = np.asarray(values).reshape(-1)
    return tuple(int(round(float(v))) for v in flat.tolist())


def process_rank_start(jax: Any, local_device_count: int) -> int:
    """Return the pmap/shard global rank of this process's first local device."""
    return int(jax.process_index()) * int(local_device_count)


def device_get_addressable(jax: Any, value: Any) -> Any:
    """Fetch only this process's addressable portion of a possibly global array."""
    try:
        return jax.device_get(value)
    except RuntimeError as exc:
        if "non-addressable" not in str(exc):
            raise

    shards = getattr(value, "addressable_shards", None)
    if shards is None:
        raise

    import numpy as np

    parts = []
    value_ndim = len(getattr(value, "shape", ()))
    for shard in sorted(shards, key=shard_leading_start):
        part = np.asarray(jax.device_get(shard.data))
        if value_ndim and part.ndim == value_ndim - 1:
            part = np.expand_dims(part, axis=0)
        parts.append(part)
    return np.concatenate(parts, axis=0) if parts else jax.device_get(value)


def shard_leading_start(shard: Any) -> int:
    """Return a stable sort key for an addressable shard's leading slice."""
    index = getattr(shard, "index", ())
    first = index[0] if index else slice(0)
    return int(first.start or 0) if isinstance(first, slice) else int(first)


def check_addressable_shards_against_host(
    jax: Any,
    value: Any,
    expected_host: Any,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> bool:
    """Compare each addressable shard with the matching slice of a host array."""
    import numpy as np

    expected_host = np.asarray(expected_host)
    try:
        got = np.asarray(jax.device_get(value))
        return bool(
            np.allclose(
                got.astype(np.float32),
                expected_host.astype(np.float32),
                rtol=rtol,
                atol=atol,
            )
        )
    except RuntimeError as exc:
        if "non-addressable" not in str(exc):
            raise

    shards = getattr(value, "addressable_shards", None)
    if shards is None:
        raise

    for shard in sorted(shards, key=shard_leading_start):
        got = np.asarray(jax.device_get(shard.data)).astype(np.float32)
        expected = np.asarray(expected_host[shard.index]).astype(np.float32)
        if got.shape != expected.shape or not np.allclose(
            got, expected, rtol=rtol, atol=atol
        ):
            return False
    return True


def expected_for_addressable(
    jax: Any,
    expected: Any,
    local_device_count: int,
) -> Any:
    """Slice a global expected array down to this process's local rank window."""
    import numpy as np

    expected_host = np.asarray(jax.device_get(expected))
    axis_size = int(jax.device_count())
    if expected_host.ndim and expected_host.shape[0] == axis_size:
        start = process_rank_start(jax, local_device_count)
        expected_host = expected_host[start : start + local_device_count]
    return expected_host


def check_addressable_result(
    jax: Any,
    jnp: Any,
    y: Any,
    expected: Any,
    local_device_count: int,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> bool:
    """Compare this process's addressable shards against a global expected array."""
    del jnp

    import numpy as np

    got = np.asarray(device_get_addressable(jax, y))
    expected_host = np.asarray(
        expected_for_addressable(jax, expected, local_device_count),
        dtype=np.float32,
    )

    # Lab 9 pmap returns [receiver, owner_rank, col] while its full Pallas-style
    # expected payload may include a singleton row dimension.
    if expected_host.ndim == got.ndim + 1 and expected_host.shape[2] == 1:
        expected_host = expected_host[:, :, 0, :]

    if expected_host.shape != got.shape and expected_host.ndim < got.ndim:
        expected_shape = expected_host.shape + (1,) * (got.ndim - expected_host.ndim)
        expected_host = expected_host.reshape(expected_shape) + np.zeros_like(
            got, dtype=np.float32
        )

    if expected_host.shape != got.shape:
        return False
    return bool(np.allclose(got.astype(np.float32), expected_host, rtol=rtol, atol=atol))


def make_rank_payload(
    jnp: Any,
    n_devices: int,
    elems: int,
    dtype: Any,
    *,
    rank_start: int = 0,
) -> Any:
    """Create a per-device payload filled with that device's rank.

    Rank-filled payloads make communication correctness easy to see. If device
    2 receives a tile whose first element is 1, the data itself identifies the
    sender.
    """
    ranks = jnp.arange(rank_start, rank_start + n_devices, dtype=jnp.float32).reshape(
        n_devices, 1
    )
    x = jnp.broadcast_to(ranks, (n_devices, elems))
    return x.astype(dtype)


def make_all_to_all_payload(
    jnp: Any,
    n_devices: int,
    chunk_elems: int,
    dtype: Any,
    *,
    axis_size: int | None = None,
    rank_start: int = 0,
) -> Any:
    """Create a payload whose values identify both source and destination.

    The value ``src * 10 + dst`` lets the all-to-all correctness check verify
    that each destination received the right chunk from every source.
    """
    axis_size = n_devices if axis_size is None else int(axis_size)
    src = jnp.arange(rank_start, rank_start + n_devices, dtype=jnp.float32).reshape(
        n_devices, 1, 1
    )
    dst = jnp.arange(axis_size, dtype=jnp.float32).reshape(1, axis_size, 1)
    x = src * 10.0 + dst
    x = jnp.broadcast_to(x, (n_devices, axis_size, chunk_elems))
    return x.astype(dtype)


def make_pmap_fn(
    jax: Any,
    lax: Any,
    op: str,
    axis_name: str,
    n_devices: int,
    direction: str = "right",
):
    """Build the simple pmap/lax reference function for a baseline op.

    These references are intentionally compact. They are not trying to outsmart
    XLA or match a specific hardware schedule; they provide a known-correct
    collective semantics against which custom teaching kernels can be compared.

    ``direction`` only affects ``pmap_ppermute`` (the Lab 1 single-hop baseline):
    ``"right"`` sends source ``i`` to ``i + 1`` and ``"left"`` sends it to
    ``i - 1``, so the baseline matches ``--neighbor-direction`` instead of being
    hardwired to a right-moving ring.
    """
    import functools

    if op == "pmap_psum":

        @functools.partial(jax.pmap, axis_name=axis_name)
        def fn(x):
            return lax.psum(x, axis_name)

        return fn

    if op == "pmap_all_gather":

        @functools.partial(jax.pmap, axis_name=axis_name)
        def fn(x):
            return lax.all_gather(x, axis_name, axis=0)

        return fn

    if op == "pmap_all_to_all":

        @functools.partial(jax.pmap, axis_name=axis_name)
        def fn(x):
            return lax.all_to_all(x, axis_name, split_axis=0, concat_axis=0)

        return fn

    if op == "pmap_ppermute":
        # Ring send in the requested direction: "right" writes source i to
        # destination i + 1 (receiver i observes i - 1); "left" writes source i
        # to destination i - 1 (receiver i observes i + 1). This keeps the
        # built-in baseline aligned with the custom kernel under
        # --neighbor-direction instead of being hardwired to a right ring.
        if direction == "left":
            perm = tuple((i, (i - 1) % n_devices) for i in range(n_devices))
        else:
            perm = tuple((i, (i + 1) % n_devices) for i in range(n_devices))

        @functools.partial(jax.pmap, axis_name=axis_name)
        def fn(x):
            return lax.ppermute(x, axis_name, perm=perm)

        return fn

    raise ValueError(f"unsupported pmap op {op!r}")


def pmap_logical_bytes(op: str, payload_bytes: int, n_devices: int) -> int:
    """Teaching byte model for built-in pmap reference collectives.

    The values are per-device logical communication work, not hardware counter
    readings. They give result rows enough context for rough GB/s comparisons.
    """
    if op == "pmap_ppermute":
        return payload_bytes
    if op == "pmap_all_gather":
        return payload_bytes * max(0, n_devices - 1)
    if op == "pmap_psum":
        return int(2 * payload_bytes * max(0, n_devices - 1) / max(1, n_devices))
    if op == "pmap_all_to_all":
        return int(payload_bytes * max(0, n_devices - 1) / max(1, n_devices))
    raise ValueError(f"unsupported pmap op {op!r}")


def check_pmap_result(
    jax: Any,
    jnp: Any,
    op: str,
    y: Any,
    n_devices: int,
    dtype: Any,
    direction: str = "right",
    *,
    local_device_count: int | None = None,
    rank_start: int = 0,
) -> bool:
    """Small correctness checks for pmap reference collectives.

    The payload builders encode ranks into data. Each check reads the minimum
    signal needed to confirm collective semantics before timing the function.
    """
    del dtype
    y_host = jax.device_get(y)
    local_device_count = (
        int(local_device_count) if local_device_count is not None else y_host.shape[0]
    )
    if op == "pmap_psum":
        expected = n_devices * (n_devices - 1) / 2
        return bool(jnp.allclose(y_host[..., 0], expected))
    if op == "pmap_all_gather":
        expected = jnp.arange(n_devices, dtype=jnp.float32)
        got = y_host[:, :, 0]
        return bool(jnp.allclose(got, expected.reshape(1, n_devices)))
    if op == "pmap_ppermute":
        # Receiver i observes (i - 1) for a right send and (i + 1) for a left
        # send, matching the perm chosen in make_pmap_fn.
        ranks = range(rank_start, rank_start + local_device_count)
        if direction == "left":
            expected = jnp.array([(i + 1) % n_devices for i in ranks])
        else:
            expected = jnp.array([(i - 1) % n_devices for i in ranks])
        return bool(jnp.allclose(y_host[:, 0], expected))
    if op == "pmap_all_to_all":
        expected_src = jnp.arange(n_devices, dtype=jnp.float32).reshape(1, n_devices)
        expected_dst = jnp.arange(
            rank_start,
            rank_start + local_device_count,
            dtype=jnp.float32,
        ).reshape(local_device_count, 1)
        expected = expected_src * 10.0 + expected_dst
        return bool(jnp.allclose(y_host[:, :, 0], expected))
    return False


def token_ring_hops(args: argparse.Namespace, n_devices: int) -> int:
    """Resolve the hop count for the token-ring case currently being run.

    ``args.token_hops`` carries the scalar hop value for this case. When the user
    passes a sweep such as ``--token-hops 0,1,2,3`` the main loop sets this to one
    value per case. ``None`` (flag omitted) or a negative value means "use the
    full ring", matching the ``lab2_token_ring`` convention of ``n_devices - 1``
    hops.
    """
    hops = args.token_hops
    if hops is None or int(hops) < 0:
        return max(0, n_devices - 1)
    return int(hops)


def expected_token_ring_sums(jnp: Any, n_devices: int, hops: int, direction: str) -> Any:
    """Expected running rank sums for a token after a given number of hops."""
    values = []
    for rank in range(n_devices):
        if direction == "right":
            seen = [(rank - hop) % n_devices for hop in range(hops + 1)]
        else:
            seen = [(rank + hop) % n_devices for hop in range(hops + 1)]
        values.append(sum(seen))
    return jnp.array(values, dtype=jnp.float32)


def run_pmap_ring_all_gather(
    jax: Any,
    jnp: Any,
    lax: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run a pmap/lax reference ring all-gather built from ppermute hops.

    This is a teaching reference for Lab 5. It preserves arrival order, meaning
    each device stacks the token it starts with, then the token observed after
    each hop around the ring.
    """
    import functools

    itemsize = dtype_itemsize(jnp, dtype)
    elems = elems_for_payload(payload_bytes, itemsize)
    actual_payload = elems * itemsize
    axis_size = int(jax.device_count())
    rank_start = int(jax.process_index()) * int(n_devices)
    x = make_rank_payload(jnp, n_devices, elems, dtype, rank_start=rank_start)
    # A full all-gather is n_devices - 1 hops; --token-hops lets the Lab 5
    # partial-gather experiment ask for fewer (or more) hops, in which case the
    # output stacks only hops + 1 arrivals per device.
    hops = token_ring_hops(args, axis_size)
    if args.neighbor_direction == "right":
        # Source i sends to i + 1. Device r therefore receives ranks
        # r, r - 1, r - 2, ... over successive hops.
        perm = tuple((i, (i + 1) % axis_size) for i in range(axis_size))
    else:
        # Source i sends to i - 1. Device r receives r, r + 1, r + 2, ...
        perm = tuple((i, (i - 1) % axis_size) for i in range(axis_size))

    @functools.partial(jax.pmap, axis_name=args.axis_name)
    def fn(value):
        """Return tokens in the order this device observes them."""
        token = value
        pieces = [token]
        for _ in range(hops):
            token = lax.ppermute(token, args.axis_name, perm=perm)
            pieces.append(token)
        return jnp.stack(pieces, axis=0)

    y = block_until_ready(jax, fn(x))
    if args.skip_correctness:
        ok = True
    else:
        got = jax.device_get(y)[:, :, 0]
        expected_rows = []
        for rank in range(rank_start, rank_start + n_devices):
            if args.neighbor_direction == "right":
                expected_rows.append([(rank - hop) % axis_size for hop in range(hops + 1)])
            else:
                expected_rows.append([(rank + hop) % axis_size for hop in range(hops + 1)])
        expected = jnp.array(expected_rows, dtype=jnp.float32)
        ok = bool(jnp.allclose(got, expected))
    with maybe_trace(jax, args, run, "pmap_ring_all_gather", payload_bytes) as trace_artifact:
        timing = time_jax_call(
            jax,
            fn,
            x,
            warmup=args.warmup,
            iters=args.iters,
        )
    memory_profile = maybe_write_memory_profile(
        jax, args, run, "pmap_ring_all_gather", payload_bytes
    )
    return BenchResult(
        op="pmap_ring_all_gather",
        layer="pmap/lax",
        payload_bytes=actual_payload,
        logical_bytes=actual_payload * hops,
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=(
            f"arrival-order ring all-gather direction={args.neighbor_direction} "
            f"hops={hops}"
        ),
        **result_timing_kwargs(timing),
    )


def run_pmap_psum_scatter(
    jax: Any,
    jnp: Any,
    lax: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run the pmap/lax reduce-scatter reference for Lab 6."""
    import functools

    itemsize = dtype_itemsize(jnp, dtype)
    axis_size = int(jax.device_count())
    rank_start = int(jax.process_index()) * int(n_devices)
    chunk_elems = elems_for_payload(payload_bytes, itemsize, divisor=axis_size)
    actual_payload = axis_size * chunk_elems * itemsize
    # Encode both source rank and chunk index so the reduced owner chunk has a
    # predictable first element after psum_scatter.
    src = jnp.arange(rank_start, rank_start + n_devices, dtype=jnp.float32).reshape(
        n_devices, 1, 1
    )
    chunk = jnp.arange(axis_size, dtype=jnp.float32).reshape(1, axis_size, 1)
    x = jnp.broadcast_to(src * 10.0 + chunk, (n_devices, axis_size, chunk_elems))
    x = x.astype(dtype)

    @functools.partial(jax.pmap, axis_name=args.axis_name)
    def fn(value):
        """Built-in reduce-scatter over the pmap axis."""
        return lax.psum_scatter(
            value,
            args.axis_name,
            scatter_dimension=0,
            tiled=False,
        )

    y = block_until_ready(jax, fn(x))
    if args.skip_correctness:
        ok = True
    else:
        reduced = 10.0 * axis_size * (axis_size - 1) / 2
        local_ranks = jnp.arange(
            rank_start,
            rank_start + n_devices,
            dtype=jnp.float32,
        )
        expected = reduced + axis_size * local_ranks
        ok = bool(jnp.allclose(jax.device_get(y)[:, 0], expected))
    with maybe_trace(jax, args, run, "pmap_psum_scatter", payload_bytes) as trace_artifact:
        timing = time_jax_call(
            jax,
            fn,
            x,
            warmup=args.warmup,
            iters=args.iters,
        )
    memory_profile = maybe_write_memory_profile(
        jax, args, run, "pmap_psum_scatter", payload_bytes
    )
    return BenchResult(
        op="pmap_psum_scatter",
        layer="pmap/lax",
        payload_bytes=actual_payload,
        logical_bytes=int(2 * actual_payload * max(0, axis_size - 1) / max(1, axis_size)),
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=f"reduce-scatter reference chunks={axis_size} chunk_elems={chunk_elems}",
        **result_timing_kwargs(timing),
    )


def run_pmap_2d_staged_all_gather(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run the pmap/lax reference for Lab 9's staged 2D all-gather."""
    try:
        from labs import lab9_mesh_collectives
    except Exception as exc:
        return BenchResult(
            op="pmap_2d_staged_all_gather",
            layer="pmap/lax",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    try:
        # The Lab 9 module owns the mesh-shape and axis-order teaching logic so
        # the pmap and Pallas variants stay comparable.
        case = lab9_mesh_collectives.build_pmap_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            mesh_shape_name=args.lab9_mesh_shape,
            axis_order=args.lab9_axis_order,
            direction=args.neighbor_direction,
        )
    except Exception as exc:
        return BenchResult(
            op="pmap_2d_staged_all_gather",
            layer="pmap/lax",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"case setup failed: {exc}",
        )

    try:
        rank_start = process_rank_start(jax, n_devices)
        local_x = case.x[rank_start : rank_start + n_devices]
        y = block_until_ready(jax, case.fn(local_x))
        if args.skip_correctness:
            ok = True
        else:
            ok = check_addressable_result(jax, jnp, y, case.expected_ranks, n_devices)
        with maybe_trace(
            jax, args, run, "pmap_2d_staged_all_gather", payload_bytes
        ) as trace_artifact:
            timing = time_jax_call(
                jax,
                case.fn,
                local_x,
                warmup=args.warmup,
                iters=args.iters,
            )
        memory_profile = maybe_write_memory_profile(
            jax, args, run, "pmap_2d_staged_all_gather", payload_bytes
        )
    except Exception as exc:
        return BenchResult(
            op="pmap_2d_staged_all_gather",
            layer="pmap/lax",
            payload_bytes=case.actual_payload_bytes,
            logical_bytes=case.wire_bytes,
            seconds=None,
            ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op="pmap_2d_staged_all_gather",
        layer="pmap/lax",
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=case.wire_bytes,
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=case.note,
        **result_timing_kwargs(timing),
    )


def run_pmap_local_arith(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run the pmap/local arithmetic reference for Lab 3."""
    import functools
    from labs import lab3_memory_spaces

    itemsize = dtype_itemsize(jnp, dtype)
    rows, cols, actual_input = lab3_memory_spaces.tile_shape_for_payload(
        payload_bytes=payload_bytes,
        itemsize=itemsize,
        tile_rows=args.pallas_tile_rows,
        min_cols=args.pallas_min_cols,
    )
    x = lab3_memory_spaces.make_rank_tile(jnp, n_devices, rows, cols, dtype)
    expected = lab3_memory_spaces.local_arith_reference(
        x, scale=args.lab3_scale, bias=args.lab3_bias
    )
    output_bytes = rows * cols * int(jnp.dtype(jnp.float32).itemsize)

    @functools.partial(jax.pmap, axis_name=args.axis_name)
    def fn(value):
        """Apply the same elementwise formula that the Pallas VMEM kernel uses."""
        return lab3_memory_spaces.local_arith_reference(
            value, scale=args.lab3_scale, bias=args.lab3_bias
        )

    y = block_until_ready(jax, fn(x))
    ok = True if args.skip_correctness else bool(jnp.allclose(jax.device_get(y), expected))
    with maybe_trace(jax, args, run, "pmap_local_arith", payload_bytes) as trace_artifact:
        timing = time_jax_call(
            jax,
            fn,
            x,
            warmup=args.warmup,
            iters=args.iters,
        )
    memory_profile = maybe_write_memory_profile(
        jax, args, run, "pmap_local_arith", payload_bytes
    )
    return BenchResult(
        op="pmap_local_arith",
        layer="pmap/local",
        payload_bytes=actual_input,
        logical_bytes=actual_input + output_bytes,
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=(
            f"local read+write tile={rows}x{cols} "
            f"{jnp.dtype(dtype)}->float32 y=x*{args.lab3_scale:g}+{args.lab3_bias:g}"
        ),
        **result_timing_kwargs(timing),
    )


def run_pmap_token_ring(
    jax: Any,
    jnp: Any,
    lax: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run the pmap/lax token ring reference for Lab 2."""
    import functools

    itemsize = dtype_itemsize(jnp, dtype)
    elems = elems_for_payload(payload_bytes, itemsize)
    actual_payload = elems * itemsize
    axis_size = int(jax.device_count())
    rank_start = int(jax.process_index()) * int(n_devices)
    x = make_rank_payload(jnp, n_devices, elems, dtype, rank_start=rank_start)
    hops = token_ring_hops(args, axis_size)
    if args.neighbor_direction == "right":
        # Match Lab 1/Lab 2 convention: direction is from the sender's point of
        # view, not the receiver's point of view.
        perm = tuple((i, (i + 1) % axis_size) for i in range(axis_size))
    else:
        perm = tuple((i, (i - 1) % axis_size) for i in range(axis_size))

    @functools.partial(jax.pmap, axis_name=args.axis_name)
    def fn(value):
        """Forward one token per hop and accumulate every rank observed."""
        token = value
        seen_sum = token.astype(jnp.float32)
        for _ in range(hops):
            token = lax.ppermute(token, args.axis_name, perm=perm)
            seen_sum = seen_sum + token.astype(jnp.float32)
        return seen_sum

    y = block_until_ready(jax, fn(x))
    if args.skip_correctness:
        ok = True
    else:
        expected_all = expected_token_ring_sums(
            jnp, axis_size, hops, args.neighbor_direction
        )
        expected = expected_all[rank_start : rank_start + n_devices]
        ok = bool(jnp.allclose(jax.device_get(y)[:, 0], expected))
    with maybe_trace(jax, args, run, "pmap_token_ring", payload_bytes) as trace_artifact:
        timing = time_jax_call(
            jax,
            fn,
            x,
            warmup=args.warmup,
            iters=args.iters,
        )
    memory_profile = maybe_write_memory_profile(jax, args, run, "pmap_token_ring", payload_bytes)
    return BenchResult(
        op="pmap_token_ring",
        layer="pmap/lax",
        payload_bytes=actual_payload,
        logical_bytes=actual_payload * hops,
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=f"direction={args.neighbor_direction} hops={hops}",
        **result_timing_kwargs(timing),
    )


def run_pmap_op(
    jax: Any,
    jnp: Any,
    lax: Any,
    args: argparse.Namespace,
    run: RunContext,
    op: str,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Dispatch and run one pmap-family operation.

    Most pmap ops share payload construction, timing, and result packaging. The
    lab-specific references above have custom shape or correctness needs and
    exit early from this function.
    """
    if op == "pmap_ring_all_gather":
        return run_pmap_ring_all_gather(
            jax, jnp, lax, args, run, payload_bytes, dtype, n_devices
        )
    if op == "pmap_psum_scatter":
        return run_pmap_psum_scatter(
            jax, jnp, lax, args, run, payload_bytes, dtype, n_devices
        )
    if op == "pmap_2d_staged_all_gather":
        return run_pmap_2d_staged_all_gather(
            jax, jnp, args, run, payload_bytes, dtype, n_devices
        )
    if op == "pmap_local_arith":
        return run_pmap_local_arith(jax, jnp, args, run, payload_bytes, dtype, n_devices)
    if op == "pmap_token_ring":
        return run_pmap_token_ring(jax, jnp, lax, args, run, payload_bytes, dtype, n_devices)

    itemsize = dtype_itemsize(jnp, dtype)
    axis_size = int(jax.device_count())
    rank_start = int(jax.process_index()) * int(n_devices)
    if op == "pmap_all_to_all":
        # all_to_all splits each device's payload into one chunk per destination,
        # so round by n_devices to keep chunks equal-sized.
        chunk_elems = elems_for_payload(payload_bytes, itemsize, divisor=axis_size)
        x = make_all_to_all_payload(
            jnp,
            n_devices,
            chunk_elems,
            dtype,
            axis_size=axis_size,
            rank_start=rank_start,
        )
        actual_payload = axis_size * chunk_elems * itemsize
    else:
        # The rank payload is the common one-dimensional teaching input for
        # psum, all_gather, and ppermute.
        elems = elems_for_payload(payload_bytes, itemsize)
        x = make_rank_payload(jnp, n_devices, elems, dtype, rank_start=rank_start)
        actual_payload = elems * itemsize

    fn = make_pmap_fn(jax, lax, op, args.axis_name, axis_size, args.neighbor_direction)
    y = block_until_ready(jax, fn(x))
    ok = (
        True
        if args.skip_correctness
        else check_pmap_result(
            jax,
            jnp,
            op,
            y,
            axis_size,
            dtype,
            args.neighbor_direction,
            local_device_count=n_devices,
            rank_start=rank_start,
        )
    )
    # ppermute is the Lab 1 reference hop: surface its received-rank map too so it
    # lines up column-for-column with pallas_neighbor_copy. The expected map flips
    # with --neighbor-direction so the baseline and the custom kernel agree.
    observed_ranks = expected_ranks = None
    if op == "pmap_ppermute":
        observed_ranks = rank_vector(jax.device_get(y)[:, 0])
        if args.neighbor_direction == "left":
            expected_ranks = tuple(
                (i + 1) % axis_size for i in range(rank_start, rank_start + n_devices)
            )
        else:
            expected_ranks = tuple(
                (i - 1) % axis_size for i in range(rank_start, rank_start + n_devices)
            )
    with maybe_trace(jax, args, run, op, payload_bytes) as trace_artifact:
        timing = time_jax_call(
            jax,
            fn,
            x,
            warmup=args.warmup,
            iters=args.iters,
        )
    memory_profile = maybe_write_memory_profile(jax, args, run, op, payload_bytes)
    return BenchResult(
        op=op,
        layer="pmap/lax",
        payload_bytes=actual_payload,
        logical_bytes=pmap_logical_bytes(op, actual_payload, axis_size),
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        observed_ranks=observed_ranks,
        expected_ranks=expected_ranks,
        **result_timing_kwargs(timing),
    )


def run_pallas_all_gather(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run JAX's packaged Pallas TPU all-gather example.

    This is different from the course's custom Pallas kernels: it comes from
    ``jax.experimental.pallas.ops`` and acts as an additional TPU-side reference
    point when the installed JAX version provides it.
    """
    del n_devices
    if jax.default_backend() != "tpu":
        return BenchResult(
            op="pallas_all_gather",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note="requires TPU backend",
        )
    try:
        import numpy as np
        from jax.experimental.pallas.ops.tpu.all_gather import (
            all_gather as pallas_all_gather,
        )
    except Exception as exc:  # pragma: no cover - depends on installed JAX.
        return BenchResult(
            op="pallas_all_gather",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices), (args.axis_name,))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(args.axis_name))
    itemsize = dtype_itemsize(jnp, dtype)
    # The installed JAX Pallas all-gather example splits each device's shard in
    # half along its leading axis and rings the halves around. That needs a 2D,
    # tiling-aligned shard: a flat 1D shard makes the gathered output a 2D
    # (axis_size, N) buffer whose leading axis is sublane-tiled, so slicing one
    # device-row out of it fails Mosaic's tiling check. We build a (rows, 128)
    # shard and round rows-per-device up to a multiple of 16 so each half stays
    # sublane-aligned for bf16, float32, and int32.
    cols = 128
    row_align = 16
    rows_needed = (elems_for_payload(payload_bytes, itemsize) + cols - 1) // cols
    rows_per_device = max(row_align, (rows_needed + row_align - 1) // row_align * row_align)
    actual_payload = rows_per_device * cols * itemsize
    global_rows = len(devices) * rows_per_device
    # Unlike rank-marker payloads, this payload uses unique increasing values so
    # the all-gather check can compare the complete reconstructed host tensor.
    x_host = np.arange(global_rows * cols, dtype=np.float32).reshape(global_rows, cols)
    x_host_jax = jnp.asarray(x_host, dtype=dtype)
    x_expected_host = np.asarray(jax.device_get(x_host_jax), dtype=np.float32)
    x = jax.device_put(x_host_jax, sharding)

    def fn(value):
        """Invoke the packaged Pallas all_gather over the one-axis mesh."""
        return pallas_all_gather(value, mesh=mesh, axis_name=args.axis_name)

    try:
        y = block_until_ready(jax, fn(x))
        if args.skip_correctness:
            ok = True
        else:
            ok = check_addressable_shards_against_host(jax, y, x_expected_host)
        with maybe_trace(jax, args, run, "pallas_all_gather", payload_bytes) as trace_artifact:
            timing = time_jax_call(
                jax,
                fn,
                x,
                warmup=args.warmup,
                iters=args.iters,
            )
        memory_profile = maybe_write_memory_profile(
            jax, args, run, "pallas_all_gather", payload_bytes
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_all_gather",
            layer="pallas/tpu",
            payload_bytes=actual_payload,
            logical_bytes=actual_payload * max(0, len(devices) - 1),
            seconds=None,
            ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op="pallas_all_gather",
        layer="pallas/tpu",
        payload_bytes=actual_payload,
        logical_bytes=actual_payload * max(0, len(devices) - 1),
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        **result_timing_kwargs(timing),
    )


def run_pallas_neighbor_copy(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run Lab 1's custom Pallas one-hop neighbor copy."""
    if jax.default_backend() != "tpu":
        return BenchResult(
            op="pallas_neighbor_copy",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note="requires TPU backend",
        )
    try:
        from labs import lab1_single_hop
    except Exception as exc:
        return BenchResult(
            op="pallas_neighbor_copy",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    try:
        # Lab 1 builds the sharded input, the jitted Pallas function, and the
        # expected receiver-rank pattern. The harness only times and records it.
        case = lab1_single_hop.build_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            direction=args.neighbor_direction,
            tile_rows=args.pallas_tile_rows,
            min_cols=args.pallas_min_cols,
            memory_space_name=args.pallas_memory_space,
            collective_id=args.pallas_collective_id,
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_neighbor_copy",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"case setup failed: {exc}",
        )

    try:
        y = block_until_ready(jax, case.fn(case.x))
        # Surface the rank evidence as explicit row columns so Lab 1 output can
        # be read without opening the full tensor.
        local_y = device_get_addressable(jax, y)
        rank_start = process_rank_start(jax, n_devices)
        expected_all = jax.device_get(case.expected_ranks)
        expected_local = expected_all[rank_start : rank_start + n_devices]
        observed_ranks = rank_vector(local_y[:, 0, 0])
        expected_ranks = rank_vector(expected_local)
        if args.skip_correctness:
            ok = True
        else:
            ok = bool(jnp.allclose(local_y[:, 0, 0], expected_local))
        with maybe_trace(jax, args, run, "pallas_neighbor_copy", payload_bytes) as trace_artifact:
            timing = time_jax_call(
                jax,
                case.fn,
                case.x,
                warmup=args.warmup,
                iters=args.iters,
            )
        memory_profile = maybe_write_memory_profile(
            jax, args, run, "pallas_neighbor_copy", payload_bytes
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_neighbor_copy",
            layer="pallas/tpu",
            payload_bytes=case.actual_payload_bytes,
            logical_bytes=case.actual_payload_bytes,
            seconds=None,
            ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op="pallas_neighbor_copy",
        layer="pallas/tpu",
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=case.actual_payload_bytes,
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=case.note,
        observed_ranks=observed_ranks,
        expected_ranks=expected_ranks,
        **result_timing_kwargs(timing),
    )


def run_pallas_token_ring(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run Lab 2's custom Pallas token ring."""
    if jax.default_backend() != "tpu":
        return BenchResult(
            op="pallas_token_ring",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note="requires TPU backend",
        )
    try:
        from labs import lab2_token_ring
    except Exception as exc:
        return BenchResult(
            op="pallas_token_ring",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    axis_size = int(jax.device_count())
    hops = token_ring_hops(args, axis_size)
    try:
        # The scalar hop count has already been selected by the main sweep. Lab
        # 2 resolves tile shapes, remote-DMA memory space, and expected sums.
        case = lab2_token_ring.build_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            direction=args.neighbor_direction,
            hops=hops,
            tile_rows=args.pallas_tile_rows,
            min_cols=args.pallas_min_cols,
            memory_space_name=args.pallas_memory_space,
            collective_id=args.pallas_collective_id,
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_token_ring",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"case setup failed: {exc}",
        )

    try:
        y = block_until_ready(jax, case.fn(case.x))
        if args.skip_correctness:
            ok = True
        else:
            ok = check_addressable_result(jax, jnp, y, case.expected_sums, n_devices)
        with maybe_trace(jax, args, run, "pallas_token_ring", payload_bytes) as trace_artifact:
            timing = time_jax_call(
                jax,
                case.fn,
                case.x,
                warmup=args.warmup,
                iters=args.iters,
            )
        memory_profile = maybe_write_memory_profile(
            jax, args, run, "pallas_token_ring", payload_bytes
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_token_ring",
            layer="pallas/tpu",
            payload_bytes=case.actual_payload_bytes,
            logical_bytes=case.actual_payload_bytes * case.hops,
            seconds=None,
            ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op="pallas_token_ring",
        layer="pallas/tpu",
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=case.actual_payload_bytes * case.hops,
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=case.note,
        **result_timing_kwargs(timing),
    )


def run_pallas_ring_all_gather(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run Lab 5's custom Pallas ring all-gather."""
    if jax.default_backend() != "tpu":
        return BenchResult(
            op="pallas_ring_all_gather",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note="requires TPU backend",
        )
    try:
        from labs import lab5_ring_all_gather
    except Exception as exc:
        return BenchResult(
            op="pallas_ring_all_gather",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    # A full all-gather needs n_devices - 1 neighbor-copy phases; --token-hops
    # lets the Lab 5 partial-gather experiment run fewer (or more) hops. Lab 5
    # computes arrival-order expectations for whatever hop count is requested, so
    # the correctness check stays exact for partial gathers.
    axis_size = int(jax.device_count())
    hops = token_ring_hops(args, axis_size)
    try:
        case = lab5_ring_all_gather.build_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            direction=args.neighbor_direction,
            hops=hops,
            tile_rows=args.pallas_tile_rows,
            min_cols=args.pallas_min_cols,
            memory_space_name=args.pallas_memory_space,
            collective_id=args.pallas_collective_id,
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_ring_all_gather",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"case setup failed: {exc}",
        )

    try:
        y = block_until_ready(jax, case.fn(case.x))
        if args.skip_correctness:
            ok = True
        else:
            ok = check_addressable_result(
                jax, jnp, y, case.expected_arrivals, n_devices
            )
        with maybe_trace(jax, args, run, "pallas_ring_all_gather", payload_bytes) as trace_artifact:
            timing = time_jax_call(
                jax,
                case.fn,
                case.x,
                warmup=args.warmup,
                iters=args.iters,
            )
        memory_profile = maybe_write_memory_profile(
            jax, args, run, "pallas_ring_all_gather", payload_bytes
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_ring_all_gather",
            layer="pallas/tpu",
            payload_bytes=case.actual_payload_bytes,
            logical_bytes=case.actual_payload_bytes * case.hops,
            seconds=None,
            ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op="pallas_ring_all_gather",
        layer="pallas/tpu",
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=case.actual_payload_bytes * case.hops,
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=case.note,
        **result_timing_kwargs(timing),
    )


def run_pallas_ring_reduce_scatter(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run Lab 6's custom Pallas ring reduce-scatter teaching kernel."""
    if jax.default_backend() != "tpu":
        return BenchResult(
            op="pallas_ring_reduce_scatter",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note="requires TPU backend",
        )
    try:
        from labs import lab6_reduce_scatter
    except Exception as exc:
        return BenchResult(
            op="pallas_ring_reduce_scatter",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    # A full reduce-scatter needs n_devices - 1 hops; --token-hops lets the Lab 6
    # partial-scatter experiment run fewer (or more). Lab 6 computes the expected
    # partial sums for whatever hop count is requested, so correctness stays
    # exact. Lab 6's implementation is intentionally whole-token teaching code:
    # its case object exposes teaching_wire_bytes rather than pretending to be
    # the optimized one-chunk-per-hop ring.
    axis_size = int(jax.device_count())
    hops = token_ring_hops(args, axis_size)
    try:
        case = lab6_reduce_scatter.build_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            direction=args.neighbor_direction,
            hops=hops,
            tile_rows=args.pallas_tile_rows,
            min_cols=args.pallas_min_cols,
            memory_space_name=args.pallas_memory_space,
            collective_id=args.pallas_collective_id,
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_ring_reduce_scatter",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"case setup failed: {exc}",
        )

    # Headline GB/s uses the optimal reduce-scatter byte model (matching the
    # pmap_psum_scatter baseline) so the two are directly comparable; wire_bytes
    # carries the inflated whole-token traffic this teaching kernel actually
    # moves, surfaced as wire GB/s and the overhead ratio.
    useful_bytes = int(2 * case.actual_payload_bytes * max(0, axis_size - 1) / max(1, axis_size))
    try:
        y = block_until_ready(jax, case.fn(case.x))
        if args.skip_correctness:
            ok = True
        else:
            ok = check_addressable_result(
                jax, jnp, y, case.expected_chunks, n_devices
            )
        with maybe_trace(
            jax, args, run, "pallas_ring_reduce_scatter", payload_bytes
        ) as trace_artifact:
            timing = time_jax_call(
                jax,
                case.fn,
                case.x,
                warmup=args.warmup,
                iters=args.iters,
            )
        memory_profile = maybe_write_memory_profile(
            jax, args, run, "pallas_ring_reduce_scatter", payload_bytes
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_ring_reduce_scatter",
            layer="pallas/tpu",
            payload_bytes=case.actual_payload_bytes,
            logical_bytes=useful_bytes,
            wire_bytes=case.teaching_wire_bytes,
            byte_model="whole-token",
            seconds=None,
            ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op="pallas_ring_reduce_scatter",
        layer="pallas/tpu",
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=useful_bytes,
        wire_bytes=case.teaching_wire_bytes,
        byte_model="whole-token",
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=case.note,
        **result_timing_kwargs(timing),
    )


def run_pallas_ring_all_reduce(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run Lab 7's composed Pallas all-reduce teaching implementation."""
    if jax.default_backend() != "tpu":
        return BenchResult(
            op="pallas_ring_all_reduce",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note="requires TPU backend",
        )
    try:
        from labs import lab7_all_reduce
    except Exception as exc:
        return BenchResult(
            op="pallas_ring_all_reduce",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    axis_size = int(jax.device_count())
    hops = max(0, axis_size - 1)
    try:
        # Lab 7 composes Lab 6 reduce-scatter and Lab 5 all-gather. Passing the
        # same full-ring hop count to both phases makes the composition explicit.
        case = lab7_all_reduce.build_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            direction=args.neighbor_direction,
            reduce_scatter_hops=hops,
            all_gather_hops=hops,
            tile_rows=args.pallas_tile_rows,
            min_cols=args.pallas_min_cols,
            memory_space_name=args.pallas_memory_space,
            collective_id=args.pallas_collective_id,
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_ring_all_reduce",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"case setup failed: {exc}",
        )

    # Headline GB/s uses the optimal all-reduce byte model (matching the pmap_psum
    # baseline) so the two are directly comparable; wire_bytes carries the
    # inflated composed whole-token traffic this teaching kernel actually moves.
    useful_bytes = int(2 * case.actual_payload_bytes * max(0, axis_size - 1) / max(1, axis_size))
    try:
        y = block_until_ready(jax, case.fn(case.x))
        if args.skip_correctness:
            ok = True
        else:
            ok = check_addressable_result(jax, jnp, y, case.expected_full, n_devices)
        with maybe_trace(
            jax, args, run, "pallas_ring_all_reduce", payload_bytes
        ) as trace_artifact:
            timing = time_jax_call(
                jax,
                case.fn,
                case.x,
                warmup=args.warmup,
                iters=args.iters,
            )
        memory_profile = maybe_write_memory_profile(
            jax, args, run, "pallas_ring_all_reduce", payload_bytes
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_ring_all_reduce",
            layer="pallas/tpu",
            payload_bytes=case.actual_payload_bytes,
            logical_bytes=useful_bytes,
            wire_bytes=case.teaching_wire_bytes,
            byte_model="whole-token",
            seconds=None,
            ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op="pallas_ring_all_reduce",
        layer="pallas/tpu",
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=useful_bytes,
        wire_bytes=case.teaching_wire_bytes,
        byte_model="whole-token",
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=case.note,
        **result_timing_kwargs(timing),
    )


def _run_lab8_ring(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
    *,
    op: str,
    kernel_mode: str,
    byte_model: str,
    layer: str = "pallas/tpu",
) -> BenchResult:
    """Run one Lab 8 token-ring implementation in a given kernel mode.

    ``kernel_mode`` selects the implementation explicitly so each benchmark op is
    pinned: ``serialized`` (the clarity-first teaching baseline), ``pallas-db``
    (the fused custom double-buffered ring), or ``xla-psum`` (the XLA tuned
    collective roofline). A single Lab 8 sweep can then compare all three.
    """
    if jax.default_backend() != "tpu" and kernel_mode == "pallas-db":
        return BenchResult(
            op=op, layer=layer, payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False,
            note="pallas-db requires a TPU backend",
        )
    try:
        from labs import lab8_chunked_pipeline
    except Exception as exc:
        return BenchResult(
            op=op, layer=layer, payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False, note=f"import failed: {exc}",
        )

    axis_size = int(jax.device_count())
    hops = token_ring_hops(args, axis_size)
    try:
        # Lab 8's case builder owns chunk count, modeled buffer slots, and the
        # expected sums. The harness records the byte model exposed by the case.
        case = lab8_chunked_pipeline.build_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            direction=args.neighbor_direction,
            hops=hops,
            n_chunks=args.lab8_chunks,
            buffer_count=args.lab8_buffer_count,
            tile_rows=args.pallas_tile_rows,
            min_cols=args.pallas_min_cols,
            memory_space_name=args.pallas_memory_space,
            collective_id=args.pallas_collective_id,
            kernel_mode=kernel_mode,
            inner_pipeline_cols=args.lab8_inner_cols,
        )
    except Exception as exc:
        return BenchResult(
            op=op, layer=layer, payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False,
            note=f"case setup failed: {exc}",
        )

    try:
        y = block_until_ready(jax, case.fn(case.x))
        if args.skip_correctness:
            ok = True
        else:
            ok = check_addressable_result(jax, jnp, y, case.expected_sums, n_devices)
        with maybe_trace(jax, args, run, op, payload_bytes) as trace_artifact:
            timing = time_jax_call(
                jax, case.fn, case.x, warmup=args.warmup, iters=args.iters
            )
        memory_profile = maybe_write_memory_profile(jax, args, run, op, payload_bytes)
    except Exception as exc:
        return BenchResult(
            op=op, layer=layer, payload_bytes=case.actual_payload_bytes,
            logical_bytes=case.wire_bytes, seconds=None, ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op=op,
        layer=layer,
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=case.wire_bytes,
        byte_model=byte_model,
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=case.note,
        **result_timing_kwargs(timing),
    )


def run_pallas_chunked_token_ring(
    jax: Any, jnp: Any, args: argparse.Namespace, run: RunContext,
    payload_bytes: int, dtype: Any, n_devices: int,
) -> BenchResult:
    """Lab 8 clarity-first serialized chunked token-ring (teaching baseline)."""
    return _run_lab8_ring(
        jax, jnp, args, run, payload_bytes, dtype, n_devices,
        op="pallas_chunked_token_ring", kernel_mode="serialized",
        byte_model="serialized",
    )


def run_pallas_db_token_ring(
    jax: Any, jnp: Any, args: argparse.Namespace, run: RunContext,
    payload_bytes: int, dtype: Any, n_devices: int,
) -> BenchResult:
    """Lab 8 fused custom double-buffered ring with overlapped remote DMA."""
    return _run_lab8_ring(
        jax, jnp, args, run, payload_bytes, dtype, n_devices,
        op="pallas_db_token_ring", kernel_mode="pallas-db",
        byte_model="double-buffered",
    )


def run_xla_token_ring(
    jax: Any, jnp: Any, args: argparse.Namespace, run: RunContext,
    payload_bytes: int, dtype: Any, n_devices: int,
) -> BenchResult:
    """Lab 8 XLA tuned-collective roofline reference (lax.psum / ppermute)."""
    return _run_lab8_ring(
        jax, jnp, args, run, payload_bytes, dtype, n_devices,
        op="xla_token_ring", kernel_mode="xla-psum",
        byte_model="optimal", layer="pmap/lax",
    )


def _run_lab11_ring(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
    *,
    op: str,
    kernel_mode: str,
) -> BenchResult:
    """Run one Lab 11 all-reduce implementation in a given kernel mode.

    ``kernel_mode`` pins the implementation per benchmark op: ``rs-ag`` (the
    bandwidth-optimal shard ring), ``rs-ag-bidir`` (two counter-rotating
    half-rings), or ``xla-psum`` (lax.psum on the same case in the same wire
    dtype). All three move the optimal byte volume, so wire == logical and the
    remaining differences are pure scheduling.
    """
    layer = "shard_map/psum" if kernel_mode == "xla-psum" else "shard_map/ppermute"

    try:
        from labs import lab11_optimal_all_reduce
    except Exception as exc:
        return BenchResult(
            op=op, layer=layer, payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False, note=f"import failed: {exc}",
        )

    try:
        # Lab 11's case builder owns shard sizing, ring layout, and expected
        # sums. The harness records the byte model exposed by the case.
        case = lab11_optimal_all_reduce.build_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            direction=args.neighbor_direction,
            kernel_mode=kernel_mode,
            tile_rows=args.pallas_tile_rows,
            min_cols=args.pallas_min_cols,
            ring_order=args.lab11_ring_order,
        )
    except Exception as exc:
        return BenchResult(
            op=op, layer=layer, payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False,
            note=f"case setup failed: {exc}",
        )

    try:
        y = block_until_ready(jax, case.fn(case.x))
        if args.skip_correctness:
            ok = True
        else:
            ok = check_addressable_result(
                jax,
                jnp,
                y,
                case.expected_sums,
                n_devices,
                rtol=case.check_rtol,
                atol=case.check_atol,
            )
        with maybe_trace(jax, args, run, op, payload_bytes) as trace_artifact:
            timing = time_jax_call(
                jax, case.fn, case.x, warmup=args.warmup, iters=args.iters
            )
        memory_profile = maybe_write_memory_profile(jax, args, run, op, payload_bytes)
    except Exception as exc:
        return BenchResult(
            op=op, layer=layer, payload_bytes=case.actual_payload_bytes,
            logical_bytes=case.optimal_bytes_per_device, seconds=None, ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op=op,
        layer=layer,
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=case.optimal_bytes_per_device,
        wire_bytes=case.wire_bytes,
        byte_model="optimal",
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=case.note,
        **result_timing_kwargs(timing),
    )


def run_pmap_rs_ag_all_reduce(
    jax: Any, jnp: Any, args: argparse.Namespace, run: RunContext,
    payload_bytes: int, dtype: Any, n_devices: int,
) -> BenchResult:
    """Lab 11 bandwidth-optimal shard ring (reduce-scatter + all-gather)."""
    return _run_lab11_ring(
        jax, jnp, args, run, payload_bytes, dtype, n_devices,
        op="pmap_rs_ag_all_reduce", kernel_mode="rs-ag",
    )


def run_pmap_rs_ag_all_reduce_bidir(
    jax: Any, jnp: Any, args: argparse.Namespace, run: RunContext,
    payload_bytes: int, dtype: Any, n_devices: int,
) -> BenchResult:
    """Lab 11 bidirectional variant: two counter-rotating half-rings."""
    return _run_lab11_ring(
        jax, jnp, args, run, payload_bytes, dtype, n_devices,
        op="pmap_rs_ag_all_reduce_bidir", kernel_mode="rs-ag-bidir",
    )


def run_xla_all_reduce(
    jax: Any, jnp: Any, args: argparse.Namespace, run: RunContext,
    payload_bytes: int, dtype: Any, n_devices: int,
) -> BenchResult:
    """Lab 11 roofline: lax.psum on the same case in the same wire dtype."""
    return _run_lab11_ring(
        jax, jnp, args, run, payload_bytes, dtype, n_devices,
        op="xla_all_reduce", kernel_mode="xla-psum",
    )


def run_pallas_2d_staged_all_gather(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run Lab 9's custom Pallas staged all-gather over a logical 2D mesh."""
    if jax.default_backend() != "tpu":
        return BenchResult(
            op="pallas_2d_staged_all_gather",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note="requires TPU backend",
        )
    try:
        from labs import lab9_mesh_collectives
    except Exception as exc:
        return BenchResult(
            op="pallas_2d_staged_all_gather",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    try:
        # The JAX device mesh remains flat; Lab 9 maps flat ranks into a logical
        # 2D coordinate system so students can study staged topology effects.
        case = lab9_mesh_collectives.build_pallas_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            mesh_shape_name=args.lab9_mesh_shape,
            axis_order=args.lab9_axis_order,
            direction=args.neighbor_direction,
            tile_rows=args.pallas_tile_rows,
            min_cols=args.pallas_min_cols,
            memory_space_name=args.pallas_memory_space,
            collective_id=args.pallas_collective_id,
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_2d_staged_all_gather",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"case setup failed: {exc}",
        )

    try:
        y = block_until_ready(jax, case.fn(case.x))
        if args.skip_correctness:
            ok = True
        else:
            ok = check_addressable_result(jax, jnp, y, case.expected_ranks, n_devices)
        with maybe_trace(
            jax, args, run, "pallas_2d_staged_all_gather", payload_bytes
        ) as trace_artifact:
            timing = time_jax_call(
                jax,
                case.fn,
                case.x,
                warmup=args.warmup,
                iters=args.iters,
            )
        memory_profile = maybe_write_memory_profile(
            jax, args, run, "pallas_2d_staged_all_gather", payload_bytes
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_2d_staged_all_gather",
            layer="pallas/tpu",
            payload_bytes=case.actual_payload_bytes,
            logical_bytes=case.wire_bytes,
            seconds=None,
            ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op="pallas_2d_staged_all_gather",
        layer="pallas/tpu",
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=case.wire_bytes,
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=case.note,
        **result_timing_kwargs(timing),
    )


def run_pallas_vmem_arith(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run Lab 3's local Pallas VMEM arithmetic kernel."""
    if jax.default_backend() != "tpu":
        return BenchResult(
            op="pallas_vmem_arith",
            layer="pallas/tpu-local",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note="requires TPU backend",
        )
    try:
        from labs import lab3_memory_spaces
    except Exception as exc:
        return BenchResult(
            op="pallas_vmem_arith",
            layer="pallas/tpu-local",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    try:
        # This is intentionally local compute/memory movement, not a collective.
        # It shares the harness so local and network timings land in the same
        # artifact format.
        case = lab3_memory_spaces.build_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            tile_rows=args.pallas_tile_rows,
            min_cols=args.pallas_min_cols,
            scale=args.lab3_scale,
            bias=args.lab3_bias,
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_vmem_arith",
            layer="pallas/tpu-local",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"case setup failed: {exc}",
        )

    logical_bytes = case.actual_input_bytes + case.actual_output_bytes
    try:
        y = block_until_ready(jax, case.fn(case.x))
        if args.skip_correctness:
            ok = True
        else:
            ok = check_addressable_result(jax, jnp, y, case.expected, n_devices)
        with maybe_trace(jax, args, run, "pallas_vmem_arith", payload_bytes) as trace_artifact:
            timing = time_jax_call(
                jax,
                case.fn,
                case.x,
                warmup=args.warmup,
                iters=args.iters,
            )
        memory_profile = maybe_write_memory_profile(
            jax, args, run, "pallas_vmem_arith", payload_bytes
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_vmem_arith",
            layer="pallas/tpu-local",
            payload_bytes=case.actual_input_bytes,
            logical_bytes=logical_bytes,
            seconds=None,
            ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op="pallas_vmem_arith",
        layer="pallas/tpu-local",
        payload_bytes=case.actual_input_bytes,
        logical_bytes=logical_bytes,
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=case.note,
        **result_timing_kwargs(timing),
    )


def run_pallas_semaphore_correct(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run Lab 4's safe semaphore/barrier probe."""
    if jax.default_backend() != "tpu":
        return BenchResult(
            op="pallas_semaphore_correct",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note="requires TPU backend",
        )
    try:
        from labs import lab4_semaphore_bug_zoo
    except Exception as exc:
        return BenchResult(
            op="pallas_semaphore_correct",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    try:
        # Lab 4 owns the "correct probe" concept; it delegates to the Lab 1
        # single-hop copy so the synchronization structure students study here is
        # exactly the kernel they already know.
        case = lab4_semaphore_bug_zoo.build_correct_probe_case(
            jax=jax,
            jnp=jnp,
            devices=jax.devices(),
            axis_name=args.axis_name,
            payload_bytes=payload_bytes,
            dtype=dtype,
            direction=args.neighbor_direction,
            tile_rows=args.pallas_tile_rows,
            min_cols=args.pallas_min_cols,
            memory_space_name=args.pallas_memory_space,
            collective_id=args.pallas_collective_id,
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_semaphore_correct",
            layer="pallas/tpu",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"case setup failed: {exc}",
        )

    try:
        y = block_until_ready(jax, case.fn(case.x))
        if args.skip_correctness:
            ok = True
        else:
            ok = check_addressable_result(
                jax, jnp, y, case.expected_ranks, n_devices
            )
        with maybe_trace(
            jax, args, run, "pallas_semaphore_correct", payload_bytes
        ) as trace_artifact:
            timing = time_jax_call(
                jax,
                case.fn,
                case.x,
                warmup=args.warmup,
                iters=args.iters,
            )
        memory_profile = maybe_write_memory_profile(
            jax, args, run, "pallas_semaphore_correct", payload_bytes
        )
    except Exception as exc:
        return BenchResult(
            op="pallas_semaphore_correct",
            layer="pallas/tpu",
            payload_bytes=case.actual_payload_bytes,
            logical_bytes=case.actual_payload_bytes,
            seconds=None,
            ok=False,
            note=f"failed: {exc}",
        )

    return BenchResult(
        op="pallas_semaphore_correct",
        layer="pallas/tpu",
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=case.actual_payload_bytes,
        ok=ok,
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        note=f"correct barrier+dma semaphore probe; {case.note}",
        **result_timing_kwargs(timing),
    )


def run_guarded_subprocess(
    cmd: Sequence[str], *, timeout: float, label: str = ""
) -> dict[str, Any]:
    """Run ``cmd`` in an isolated process group under a hard wall-clock timeout.

    Returns ``{"timed_out", "returncode", "stdout", "stderr"}``. On timeout the
    entire process group is SIGKILLed, so a hung TPU kernel cannot keep the device
    wedged after the child is reaped. This is the sanctioned way to execute Lab 4's
    hang/crash/race bug repros: even if the child deadlocks, the parent recovers.

    The child is launched with ``start_new_session=True`` so signalling its group
    does not touch the parent benchmark process.
    """
    import signal

    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return {
            "timed_out": False,
            "returncode": proc.returncode,
            "stdout": out,
            "stderr": err,
        }
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return {
            "timed_out": True,
            "returncode": None,
            "stdout": out,
            "stderr": err,
        }


# Hidden entrypoint flag: when present, the process does nothing but hang. It
# exists only so run_guarded_subprocess can be validated (timeout + teardown)
# without importing JAX or touching the TPU. main() handles it before any heavy
# imports.
SELFTEST_HANG_FLAG = "--__lab4-selftest-hang"


def run_pallas_semaphore_bug(
    jax: Any,
    jnp: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run a SAFE Lab 4 bug demo: reproduce a documented synchronization bug and
    show the correctness oracle catching it.

    Only correctness-class bugs run here (they complete cleanly and merely fail
    validation). ``ok`` means "the documented symptom was reproduced", so a
    healthy Lab 4 run stays all-green even though this row deliberately produces
    wrong data — the ``observed_ranks`` vs ``expected_ranks`` columns show the
    mismatch and the note explains it. Hang/crash/race bugs are not runnable here;
    they only run through the guarded subprocess path (see run_guarded_subprocess
    and --lab4-allow-dangerous).
    """
    op = "pallas_semaphore_bug"
    bug_id = getattr(args, "lab4_run_bug", None) or "wrong_neighbor_map"
    if jax.default_backend() != "tpu":
        return BenchResult(
            op=op, layer="pallas/tpu", payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False, note="requires TPU backend",
        )
    try:
        from labs import lab4_semaphore_bug_zoo
    except Exception as exc:
        return BenchResult(
            op=op, layer="pallas/tpu", payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False, note=f"import failed: {exc}",
        )

    if bug_id not in lab4_semaphore_bug_zoo.SAFE_RUNNABLE_BUGS:
        safe = ", ".join(lab4_semaphore_bug_zoo.SAFE_RUNNABLE_BUGS)
        return BenchResult(
            op=op, layer="lab/spec", payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False,
            note=(
                f"bug '{bug_id}' has no safe in-process demo (safe ids: {safe}). "
                "Hang/crash/race bugs run only via the guarded subprocess path; "
                "see lab4_semaphore_bug_zoo.md and --lab4-allow-dangerous."
            ),
        )

    if n_devices < 3:
        # On 2 devices the left and right neighbor maps coincide, so the wrong-map
        # mutation cannot change the output. Report honestly instead of pretending.
        return BenchResult(
            op=op, layer="pallas/tpu", payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False,
            note=f"wrong_neighbor_map demo needs >=3 devices to differ; have {n_devices}",
        )

    try:
        case, intended_expected, buggy_direction = (
            lab4_semaphore_bug_zoo.build_wrong_neighbor_map_case(
                jax=jax,
                jnp=jnp,
                devices=jax.devices(),
                axis_name=args.axis_name,
                payload_bytes=payload_bytes,
                dtype=dtype,
                intended_direction=args.neighbor_direction,
                tile_rows=args.pallas_tile_rows,
                min_cols=args.pallas_min_cols,
                memory_space_name=args.pallas_memory_space,
                collective_id=args.pallas_collective_id,
            )
        )
    except Exception as exc:
        return BenchResult(
            op=op, layer="pallas/tpu", payload_bytes=payload_bytes,
            logical_bytes=0, seconds=None, ok=False,
            note=f"case setup failed: {exc}",
        )

    try:
        y = block_until_ready(jax, case.fn(case.x))
        observed = rank_vector(device_get_addressable(jax, y)[:, 0, 0])
        intended = rank_vector(
            expected_for_addressable(jax, intended_expected, n_devices)
        )
        # The bug is "reproduced" when the buggy kernel's output does not match
        # the ownership map the caller intended.
        reproduced = observed != intended
        ok = True if args.skip_correctness else reproduced
        with maybe_trace(jax, args, run, op, payload_bytes) as trace_artifact:
            timing = time_jax_call(
                jax, case.fn, case.x, warmup=args.warmup, iters=args.iters
            )
        memory_profile = maybe_write_memory_profile(jax, args, run, op, payload_bytes)
    except Exception as exc:
        return BenchResult(
            op=op, layer="pallas/tpu", payload_bytes=case.actual_payload_bytes,
            logical_bytes=case.actual_payload_bytes, seconds=None, ok=False,
            note=f"failed: {exc}",
        )

    if reproduced:
        note = (
            f"BUG DEMO wrong_neighbor_map: kernel sent '{buggy_direction}' while "
            f"caller intended '{args.neighbor_direction}', so correctness "
            f"intentionally FAILED (observed={list(observed)} "
            f"intended={list(intended)}). Documented symptom reproduced; ok=True "
            "means the demo behaved as designed."
        )
    else:
        note = (
            "BUG DEMO wrong_neighbor_map did NOT reproduce (observed==intended); "
            "unexpected on this topology."
        )
    return BenchResult(
        op=op,
        layer="pallas/tpu",
        payload_bytes=case.actual_payload_bytes,
        logical_bytes=case.actual_payload_bytes,
        ok=ok,
        byte_model="bug-demo",
        trace_artifact=trace_artifact,
        memory_profile_artifact=memory_profile,
        observed_ranks=observed,
        expected_ranks=intended,
        note=note,
        **result_timing_kwargs(timing),
    )


def run_semaphore_bug_zoo(
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
) -> BenchResult:
    """Write Lab 4's semaphore bug-zoo teaching artifacts."""
    del args
    try:
        from labs import lab4_semaphore_bug_zoo
    except Exception as exc:
        return BenchResult(
            op="semaphore_bug_zoo",
            layer="lab/spec",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    artifact_dir = run.run_dir / "lab_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run.case_index:04d}_semaphore_bug_zoo"
    json_path = artifact_dir / f"{stem}.json"
    md_path = artifact_dir / f"{stem}.md"
    # The lab module owns the canonical artifact shape (catalog version, category
    # and safety summaries, the semaphore-ledger template, and scenario cards), so
    # render straight from it instead of hand-rolling a thinner JSON here.
    json_path.write_text(lab4_semaphore_bug_zoo.render_json(), encoding="utf-8")
    md_path.write_text(lab4_semaphore_bug_zoo.render_markdown(), encoding="utf-8")
    summary = lab4_semaphore_bug_zoo.catalog_summary()
    safe = int(summary["safe_to_run_by_default"])
    dangerous = int(summary["catalog_only_by_default"])
    return BenchResult(
        op="semaphore_bug_zoo",
        layer="lab/spec",
        payload_bytes=payload_bytes,
        logical_bytes=0,
        seconds=None,
        ok=True,
        note=(
            f"wrote {relative_artifact(run.run_dir, json_path)} and "
            f"{relative_artifact(run.run_dir, md_path)}; "
            f"{safe} safe scenario(s), {dangerous} dangerous scenario(s)"
        ),
    )


def run_lab_spec_op(
    jax: Any,
    args: argparse.Namespace,
    run: RunContext,
    op: str,
    payload_bytes: int,
    n_devices: int,
) -> BenchResult:
    """Render a lab module's spec artifact into the current run directory."""
    module_name = LAB_SPEC_OPS[op]
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return BenchResult(
            op=op,
            layer="lab/spec",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    try:
        # Newer lab modules can build a spec from the active run context. Older
        # ones can expose a static LAB_SPEC dictionary. Both render to the same
        # JSON/Markdown artifact pair.
        if hasattr(module, "build_spec"):
            spec = module.build_spec(
                jax=jax,
                args=args,
                payload_bytes=payload_bytes,
                n_devices=n_devices,
            )
        else:
            spec = dict(module.LAB_SPEC)
        if hasattr(module, "render_markdown"):
            markdown = module.render_markdown(spec)
        else:
            markdown = render_lab_spec_markdown(spec)
    except Exception as exc:
        return BenchResult(
            op=op,
            layer="lab/spec",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"spec build failed: {exc}",
        )

    artifact_dir = run.run_dir / "lab_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run.case_index:04d}_{sanitize_tag(op)}"
    json_path = artifact_dir / f"{stem}.json"
    md_path = artifact_dir / f"{stem}.md"
    write_json(json_path, spec)
    md_path.write_text(markdown, encoding="utf-8")
    return BenchResult(
        op=op,
        layer="lab/spec",
        payload_bytes=payload_bytes,
        logical_bytes=0,
        seconds=None,
        ok=True,
        note=(
            f"wrote {relative_artifact(run.run_dir, json_path)} and "
            f"{relative_artifact(run.run_dir, md_path)}"
        ),
    )


def run_lab10_topology_smoke(
    jax: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    n_devices: int,
) -> BenchResult:
    """Record Lab 10 process/device topology facts."""
    try:
        from labs import lab10_multihost_smoke
    except Exception as exc:
        return BenchResult(
            op="lab10_topology_smoke",
            layer="lab/topology",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    try:
        spec = lab10_multihost_smoke.build_topology_smoke(
            jax=jax,
            args=args,
            payload_bytes=payload_bytes,
            n_devices=n_devices,
        )
        markdown = lab10_multihost_smoke.render_markdown(spec)
    except Exception as exc:
        return BenchResult(
            op="lab10_topology_smoke",
            layer="lab/topology",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"topology smoke failed: {exc}",
        )

    artifact_dir = run.run_dir / "lab_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run.case_index:04d}_lab10_topology_smoke"
    json_path = artifact_dir / f"{stem}.json"
    md_path = artifact_dir / f"{stem}.md"
    write_json(json_path, spec)
    md_path.write_text(markdown, encoding="utf-8")
    ok = bool(spec.get("ok", True))
    return BenchResult(
        op="lab10_topology_smoke",
        layer="lab/topology",
        payload_bytes=payload_bytes,
        logical_bytes=0,
        seconds=None,
        ok=ok,
        note=(
            f"process={spec.get('process_index')}/{spec.get('process_count')} "
            f"local_devices={n_devices}; wrote "
            f"{relative_artifact(run.run_dir, json_path)} and "
            f"{relative_artifact(run.run_dir, md_path)}"
        ),
    )


def run_lab10_process_collective_smoke(
    jax: Any,
    args: argparse.Namespace,
    run: RunContext,
    payload_bytes: int,
    n_devices: int,
) -> BenchResult:
    """Run Lab 10's process sync/all-gather smoke check."""
    try:
        from labs import lab10_multihost_smoke
    except Exception as exc:
        return BenchResult(
            op="lab10_process_collective_smoke",
            layer="lab/multihost",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"import failed: {exc}",
        )

    try:
        spec = lab10_multihost_smoke.build_process_collective_smoke(
            jax=jax,
            args=args,
            payload_bytes=payload_bytes,
            n_devices=n_devices,
        )
        markdown = lab10_multihost_smoke.render_markdown(spec)
    except Exception as exc:
        return BenchResult(
            op="lab10_process_collective_smoke",
            layer="lab/multihost",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"process collective smoke failed: {exc}",
        )

    artifact_dir = run.run_dir / "lab_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run.case_index:04d}_lab10_process_collective_smoke"
    json_path = artifact_dir / f"{stem}.json"
    md_path = artifact_dir / f"{stem}.md"
    write_json(json_path, spec)
    md_path.write_text(markdown, encoding="utf-8")
    process_count = int(spec.get("process_count") or 1)
    logical_bytes = payload_bytes * process_count
    return BenchResult(
        op="lab10_process_collective_smoke",
        layer="lab/multihost",
        payload_bytes=payload_bytes,
        logical_bytes=logical_bytes,
        seconds=float(spec.get("seconds") or 0.0),
        ok=bool(spec.get("ok", True)),
        note=(
            f"process={spec.get('process_index')}/{spec.get('process_count')} "
            f"allgather_shape={spec.get('process_allgather_shape')}; wrote "
            f"{relative_artifact(run.run_dir, json_path)} and "
            f"{relative_artifact(run.run_dir, md_path)}"
        ),
    )


def render_lab_spec_markdown(spec: Mapping[str, Any]) -> str:
    """Fallback Markdown renderer for simple LAB_SPEC dictionaries."""
    lines = [
        f"# {spec.get('title', spec.get('lab', 'Lab Spec'))}",
        "",
        str(spec.get("goal", "")).strip(),
        "",
    ]
    for key in ("happy_path", "implemented_ops", "deferred_ops", "pass_condition"):
        values = spec.get(key) or []
        if not values:
            continue
        title = key.replace("_", " ").title()
        lines.extend([f"## {title}", ""])
        for item in values:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_external(
    args: argparse.Namespace,
    payload_bytes: int,
    dtype_name: str,
    n_devices: int,
) -> BenchResult:
    """Run a user-provided executable behind the BenchResult contract.

    The external command receives formatted placeholders and must print JSON to
    stdout. This keeps non-Python or lower-level experiments comparable with the
    built-in result rows.
    """
    if not args.external:
        return BenchResult(
            op="external",
            layer="external",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note="--external command template is required",
        )

    command = args.external.format(
        bytes=payload_bytes,
        dtype=dtype_name,
        iters=args.iters,
        warmup=args.warmup,
        devices=n_devices,
    )
    # shlex.split keeps the CLI template familiar while avoiding shell=True.
    # If a future external runner needs shell features, wrap them in a script and
    # call that script here.
    proc = subprocess.run(
        shlex.split(command),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        note = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        return BenchResult(
            op="external",
            layer="external",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=note,
            external_stdout=proc.stdout,
            external_stderr=proc.stderr,
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return BenchResult(
            op="external",
            layer="external",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note=f"expected JSON stdout, got: {proc.stdout.strip()[:160]}",
            external_stdout=proc.stdout,
            external_stderr=proc.stderr,
        )

    seconds = data.get("seconds")
    logical_bytes = int(data.get("logical_bytes", payload_bytes))
    ok = bool(data.get("ok", True))
    return BenchResult(
        op="external",
        layer=str(data.get("layer", "external")),
        payload_bytes=int(data.get("payload_bytes", payload_bytes)),
        logical_bytes=logical_bytes,
        seconds=float(seconds) if seconds is not None else None,
        ok=ok,
        note=str(data.get("note", "")),
        external_stdout=proc.stdout,
        external_stderr=proc.stderr,
    )


def concise_note(note: str, max_chars: int = 120) -> str:
    """Return a one-line note for terminal tables and failure summaries."""
    if not note:
        return ""
    first = note.strip().splitlines()[0] if note.strip() else ""
    if len(first) <= max_chars:
        return first
    return first[: max_chars - 3] + "..."


def result_to_row(
    run: RunContext,
    args: argparse.Namespace,
    result: BenchResult,
    *,
    requested_payload_bytes: int,
    n_devices: int,
    dtype_name: str,
    token_hops: int | None = None,
) -> dict[str, Any]:
    """Flatten BenchResult plus run context into the persisted row schema."""
    return {
        "case_index": run.case_index,
        "time_unix": time.time(),
        "elapsed_since_run_start_s": time.time() - run.started_at_unix,
        "op": result.op,
        "layer": result.layer,
        "lab": args.lab,
        "ok": result.ok,
        "requested_payload_bytes": requested_payload_bytes,
        "payload_bytes": result.payload_bytes,
        "logical_bytes": result.logical_bytes,
        "wire_bytes": result.wire_bytes,
        "byte_model": result.byte_model,
        "seconds": result.seconds,
        "us": result.us,
        "gbps": result.gbps,
        "wire_gbps": result.wire_gbps,
        "timing_mean_s": result.timing_mean_s,
        "timing_p10_s": result.timing_p10_s,
        "timing_p50_s": result.timing_p50_s,
        "timing_p90_s": result.timing_p90_s,
        "timing_p99_s": result.timing_p99_s,
        "timing_min_s": result.timing_min_s,
        "timing_max_s": result.timing_max_s,
        "timing_samples_s": result.timing_samples_s,
        "dtype": dtype_name,
        "n_devices": n_devices,
        "warmup": args.warmup,
        "iters": args.iters,
        "axis_name": args.axis_name,
        "neighbor_direction": args.neighbor_direction,
        "token_hops": token_hops,
        "observed_ranks": list(result.observed_ranks) if result.observed_ranks is not None else None,
        "expected_ranks": list(result.expected_ranks) if result.expected_ranks is not None else None,
        "note": result.note,
        "note_first_line": concise_note(result.note, 400),
        "trace_artifact": result.trace_artifact,
        "memory_profile_artifact": result.memory_profile_artifact,
        "error_artifact": result.error_artifact,
        "external_stdout": result.external_stdout,
        "external_stderr": result.external_stderr,
    }


def write_error_artifact(run: RunContext, row: dict[str, Any]) -> None:
    """Write an expanded text artifact for failed rows.

    The JSONL row stays compact, while the error artifact can include traceback
    text and external stdout/stderr without making the terminal table unreadable.
    """
    if row.get("ok"):
        return
    if not (row.get("note") or row.get("external_stdout") or row.get("external_stderr")):
        return
    tag = f"{int(row['case_index']):04d}_{sanitize_tag(row['op'])}_{row['requested_payload_bytes']}"
    path = run.errors_dir / f"{tag}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    pieces = [
        "Result row:",
        json.dumps(row, indent=2, sort_keys=True, default=json_default),
        "",
    ]
    if row.get("note"):
        pieces.extend(["Note:", str(row["note"]), ""])
    if row.get("external_stdout"):
        pieces.extend(["External stdout:", str(row["external_stdout"]), ""])
    if row.get("external_stderr"):
        pieces.extend(["External stderr:", str(row["external_stderr"]), ""])
    path.write_text("\n".join(pieces), encoding="utf-8")
    row["error_artifact"] = relative_artifact(run.run_dir, path)


def record_result(
    run: RunContext,
    args: argparse.Namespace,
    result: BenchResult,
    *,
    requested_payload_bytes: int,
    n_devices: int,
    dtype_name: str,
    token_hops: int | None = None,
) -> dict[str, Any]:
    """Persist one case result everywhere needed for an in-progress run."""
    row = result_to_row(
        run,
        args,
        result,
        requested_payload_bytes=requested_payload_bytes,
        n_devices=n_devices,
        dtype_name=dtype_name,
        token_hops=token_hops,
    )
    write_error_artifact(run, row)
    run.results.append(row)
    # JSONL append happens before printing so tailing the file is a reliable
    # progress monitor during long TPU sweeps.
    append_jsonl(run.results_jsonl, row)
    print_result(row)
    sys.stdout.flush()
    return row


def print_header() -> None:
    """Print the fixed-width table header for interactive runs."""
    # GB/s is "useful" throughput (optimal-model bytes / time), comparable within
    # a collective family. wire GB/s uses the implementation's actual traffic; a
    # gap between them is wasted bandwidth. For cross-op comparison prefer p50
    # time, which needs no byte model.
    print(
        f"{'op':34s} {'layer':12s} {'payload':>12s} {'useful/dev':>12s} "
        f"{'p50 time':>12s} {'GB/s':>10s} {'wireGB/s':>10s} {'ok':>4s}  note"
    )
    print(
        "  (GB/s = useful optimal-model throughput; wireGB/s = actual traffic; "
        "compare p50 time across ops)"
    )


def print_result(row: Mapping[str, Any]) -> None:
    """Print one compact terminal row for a benchmark case."""
    us = row.get("us")
    gbps = row.get("gbps")
    wire_gbps = row.get("wire_gbps")
    time_text = "n/a" if us is None else f"{float(us):9.2f} us"
    gbps_text = "n/a" if gbps is None else f"{float(gbps):8.2f}"
    wire_text = "n/a" if wire_gbps is None else f"{float(wire_gbps):8.2f}"
    note = concise_note(str(row.get("note") or ""))
    if row.get("error_artifact") and note:
        note = f"{note} [{row['error_artifact']}]"
    elif row.get("error_artifact"):
        note = f"[{row['error_artifact']}]"
    print(
        f"{str(row['op']):34s} {str(row['layer']):12s} "
        f"{format_bytes(int(row['payload_bytes'])):>12s} "
        f"{format_bytes(int(row['logical_bytes'])):>12s} "
        f"{time_text:>12s} {gbps_text:>10s} {wire_text:>10s} "
        f"{str(row['ok']):>4s}  {note}"
    )


def normalize_ops(ops: Iterable[str]) -> list[str]:
    """Expand aliases and validate operation names from --ops."""
    normalized: list[str] = []
    for op in ops:
        if op == "all":
            normalized.extend(DEFAULT_OPS)
        elif op in ALL_OPS:
            normalized.append(op)
        else:
            allowed = ", ".join(("all", *ALL_OPS))
            raise ValueError(f"unknown op {op!r}; allowed: {allowed}")
    return list(dict.fromkeys(normalized))


def numeric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Filter rows down to successful finite numeric timing rows for plots."""
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            seconds = float(row["seconds"])
            gbps = float(row["gbps"])
            payload = int(row["payload_bytes"])
        except Exception:
            continue
        if not (row.get("ok") and math.isfinite(seconds) and math.isfinite(gbps)):
            continue
        out.append({**dict(row), "seconds": seconds, "gbps": gbps, "payload_bytes": payload})
    return out


def maybe_write_plots(rows: Sequence[Mapping[str, Any]], run: RunContext) -> list[str]:
    """Create summary plots when matplotlib is available and plotting is enabled."""
    if run.args.no_plots:
        return []
    data = numeric_rows(rows)
    if not data:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        note = run.run_dir / "plots.error.txt"
        note.write_text(f"plot import failed: {exc!r}\n", encoding="utf-8")
        return [relative_artifact(run.run_dir, note)]

    run.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []
    ops = sorted({str(row["op"]) for row in data})

    def save(fig: Any, name: str) -> None:
        """Save one matplotlib figure and remember its run-relative path."""
        path = run.plot_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths.append(relative_artifact(run.run_dir, path))

    def finite(value: Any) -> bool:
        """Small guard for optional percentile columns."""
        return isinstance(value, (int, float)) and math.isfinite(value)

    # Plot 1: latency curve. This is the first plot students usually inspect
    # because it shows fixed overheads and payload-scaling behavior directly.
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for op in ops:
        group = sorted((row for row in data if row["op"] == op), key=lambda r: r["payload_bytes"])
        line, = ax.plot(
            [row["payload_bytes"] for row in group],
            [row["seconds"] * 1e6 for row in group],
            marker="o",
            linewidth=1.8,
            label=op,
        )
        # Shade the p10-p90 spread so run-to-run noise is visible, not hidden
        # behind a single p50 point.
        bx, lo, hi = [], [], []
        for row in group:
            p10, p90 = row.get("timing_p10_s"), row.get("timing_p90_s")
            if finite(p10) and finite(p90):
                bx.append(row["payload_bytes"])
                lo.append(p10 * 1e6)
                hi.append(p90 * 1e6)
        if len(bx) >= 2:
            ax.fill_between(bx, lo, hi, color=line.get_color(), alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("payload bytes per local device")
    ax.set_ylabel("latency (us)")
    ax.set_title("Collective latency by payload (line = p50, band = p10-p90)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "latency_by_payload.png")

    # Plot 2: useful bandwidth (optimal-model bytes / time). This is comparable
    # within a collective family; the wire_gbps column shows actual traffic, and
    # the byte_model column flags which implementations move more than the
    # optimal bytes. Read alongside operation notes and lab specs.
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for op in ops:
        group = sorted((row for row in data if row["op"] == op), key=lambda r: r["payload_bytes"])
        ax.plot(
            [row["payload_bytes"] for row in group],
            [row["gbps"] for row in group],
            marker="o",
            linewidth=1.8,
            label=op,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("payload bytes per local device")
    ax.set_ylabel("useful GB/s per device (optimal model)")
    ax.set_title("Collective bandwidth by payload")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "bandwidth_by_payload.png")

    # Relative speedup vs the slowest op at each payload. The slowest op sits at
    # 1.0; a custom kernel that beats the reference rises above it. This makes the
    # pallas-vs-reference comparison readable without eyeballing two latency lines.
    slowest = {}
    for payload in {row["payload_bytes"] for row in data}:
        lat = [row["seconds"] for row in data if row["payload_bytes"] == payload and row["seconds"] > 0]
        if lat:
            slowest[payload] = max(lat)
    if len(ops) >= 2 and slowest:
        # Plot 3: relative speedup at each payload. This avoids comparing two
        # different x-axis positions by eye.
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        for op in ops:
            group = sorted(
                (row for row in data if row["op"] == op and row["seconds"] > 0),
                key=lambda r: r["payload_bytes"],
            )
            xs = [row["payload_bytes"] for row in group]
            ys = [slowest[row["payload_bytes"]] / row["seconds"] for row in group]
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=op)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("payload bytes per local device")
        ax.set_ylabel("speedup vs slowest op (x)")
        ax.set_title("Relative speedup by payload (higher = faster)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        save(fig, "speedup_by_payload.png")

    # Plot 4: status counts. This is intentionally included even in benchmark
    # runs because a "fast" failed op should never disappear into averages.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ok_counts = {op: 0 for op in sorted({str(row["op"]) for row in rows})}
    fail_counts = {op: 0 for op in ok_counts}
    for row in rows:
        op = str(row["op"])
        if row.get("ok"):
            ok_counts[op] = ok_counts.get(op, 0) + 1
        else:
            fail_counts[op] = fail_counts.get(op, 0) + 1
    labels = list(ok_counts)
    positions = list(range(len(labels)))
    ax.bar(positions, [ok_counts[label] for label in labels], label="ok")
    ax.bar(
        positions,
        [fail_counts[label] for label in labels],
        bottom=[ok_counts[label] for label in labels],
        label="failed",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("cases")
    ax.set_title("Benchmark case status")
    ax.legend()
    save(fig, "case_status.png")

    return plot_paths


# Device-timeline event names that represent real communication work in a
# Pallas remote-DMA kernel. These are the runtime names that actually appear in
# the trace (the Python named_scope labels are folded into the fused custom-call
# and do not show up as separate timeline events).
def _classify_comm_event(name: str) -> str | None:
    """Map low-level trace event names into teaching communication categories."""
    if name.startswith("copy"):
        return "remote_dma_copy"
    if name.startswith("barrier-cores"):
        return "entry_barrier"
    if name.startswith("Acquire semaphore"):
        return "semaphore_acquire"
    if name.startswith("Release semaphore"):
        return "semaphore_release"
    return None


def _summarize_one_trace(trace: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """Return {device: {event_category: {count, total_us, mean_us}}} for a trace."""
    events = trace.get("traceEvents", [])
    # Trace events identify devices by process id. Metadata events provide the
    # human-readable process name, which is where TPU device labels live.
    pid_name = {
        e["pid"]: e.get("args", {}).get("name", "")
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    tpu_pids = {pid: name for pid, name in pid_name.items() if "TPU" in name}
    per_device: dict[str, dict[str, dict[str, float]]] = {}
    for e in events:
        # Only complete-duration events ("X") on TPU device timelines contribute
        # to this communication summary.
        if e.get("ph") != "X" or e.get("pid") not in tpu_pids:
            continue
        category = _classify_comm_event(e.get("name", ""))
        if category is None:
            continue
        device = tpu_pids[e["pid"]]
        bucket = per_device.setdefault(device, {}).setdefault(
            category, {"count": 0.0, "total_us": 0.0}
        )
        bucket["count"] += 1.0
        bucket["total_us"] += float(e.get("dur", 0.0))
    for cats in per_device.values():
        for bucket in cats.values():
            bucket["mean_us"] = bucket["total_us"] / bucket["count"] if bucket["count"] else 0.0
    return per_device


def _plot_trace_comm(
    summary: Mapping[str, dict[str, dict[str, float]]], run: RunContext, stem: str
) -> str | None:
    """Grouped bar chart of mean per-iteration comm time per device, by event."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    devices = sorted(summary)
    categories = sorted({cat for cats in summary.values() for cat in cats})
    if not devices or not categories:
        return None

    x = np.arange(len(devices))
    width = 0.8 / len(categories)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for i, category in enumerate(categories):
        values = [summary[d].get(category, {}).get("mean_us", 0.0) for d in devices]
        ax.bar(x + i * width, values, width, label=category)
    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels([d.split(":")[-1] for d in devices])
    ax.set_xlabel("TPU device")
    ax.set_ylabel("mean time per iteration (us)")
    ax.set_title("On-device communication time per device (from trace)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    run.plot_dir.mkdir(parents=True, exist_ok=True)
    path = run.plot_dir / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return relative_artifact(run.run_dir, path)


def maybe_write_trace_summary(run: RunContext) -> dict[str, list[str]]:
    """Turn captured Perfetto traces into communication-time data and plots.

    This only does work when a profiling run captured traces. For each
    ``*.trace.json.gz`` it extracts the on-device communication events (remote
    DMA copy, entry barrier, semaphore acquire/release), writes a JSON summary,
    and plots mean per-device communication time. Non-profiled runs return {}.
    """
    traces_root = run.run_dir / "traces"
    if not traces_root.exists():
        return {}
    gz_files = sorted(traces_root.rglob("*.trace.json.gz"))
    if not gz_files:
        return {}

    import gzip

    summaries: list[dict[str, Any]] = []
    plots: list[str] = []
    for gz in gz_files:
        try:
            with gzip.open(gz, "rt", encoding="utf-8") as f:
                trace = json.load(f)
        except Exception:
            continue
        per_device = _summarize_one_trace(trace)
        if not per_device:
            continue
        rel = relative_artifact(run.run_dir, gz)
        summaries.append({"trace": rel, "per_device": per_device})
        # Name the plot after the trace case directory, e.g.
        # 0002_pallas_neighbor_copy_1048576 (the first path part under traces/).
        rel_parts = gz.relative_to(traces_root).parts
        case_name = rel_parts[0] if rel_parts else gz.stem
        # The JSON summary is data and always written; the plot honors --no-plots.
        plot = None if run.args.no_plots else _plot_trace_comm(
            per_device, run, f"trace_comm_{case_name}"
        )
        if plot:
            plots.append(plot)

    if not summaries:
        return {}
    summary_path = traces_root / "trace_comm_summary.json"
    write_json(summary_path, {"traces": summaries})
    return {
        "trace_summaries": [relative_artifact(run.run_dir, summary_path)],
        "trace_plots": plots,
    }


def build_run_summary(rows: Sequence[Mapping[str, Any]], run: RunContext) -> dict[str, Any]:
    """Build the machine-readable post-run summary."""
    successes = [row for row in rows if row.get("ok")]
    failures = [row for row in rows if not row.get("ok")]
    best_by_op: dict[str, dict[str, Any]] = {}
    for row in numeric_rows(rows):
        op = str(row["op"])
        # Best bandwidth is a compact headline for each operation. Detailed
        # latency/percentile data remains in results.jsonl and CSVs.
        current = best_by_op.get(op)
        if current is None or float(row.get("gbps") or 0) > float(current.get("gbps") or 0):
            best_by_op[op] = {
                "payload_bytes": row["payload_bytes"],
                "payload": format_bytes(int(row["payload_bytes"])),
                "us": row["seconds"] * 1e6,
                "gbps": row["gbps"],
                "case_index": row["case_index"],
            }
    return {
        "run_dir": str(run.run_dir),
        "started_at_unix": run.started_at_unix,
        "finished_at_unix": time.time(),
        "elapsed_s": time.time() - run.started_at_unix,
        "case_count": len(rows),
        "success_count": len(successes),
        "failure_count": len(failures),
        "profile_count": run.profile_count,
        "memory_profile_count": run.memory_profile_count,
        "ops": sorted({str(row.get("op")) for row in rows}),
        "payload_bytes": sorted({int(row.get("payload_bytes") or 0) for row in rows}),
        "best_by_op": best_by_op,
        "failures": [
            {
                "case_index": row.get("case_index"),
                "op": row.get("op"),
                "payload_bytes": row.get("payload_bytes"),
                "note_first_line": row.get("note_first_line"),
                "error_artifact": row.get("error_artifact"),
            }
            for row in failures
        ],
    }


def write_markdown_summary(
    run: RunContext,
    summary: Mapping[str, Any],
    plot_paths: Sequence[str],
    csv_paths: Sequence[str],
) -> str:
    """Write a human-readable run summary next to the raw artifacts."""
    lines = [
        "# Collective Communication Benchmark Run",
        "",
        f"Run directory: `{run.run_dir}`",
        f"Cases: {summary.get('case_count')} total, {summary.get('success_count')} ok, {summary.get('failure_count')} failed",
        f"Elapsed: {float(summary.get('elapsed_s') or 0):.2f} s",
        "",
        "## Artifacts",
        "",
        "- `logs/console.log`",
        "- `results.jsonl`",
    ]
    for path in csv_paths:
        lines.append(f"- `{path}`")
    for path in plot_paths:
        lines.append(f"- `{path}`")

    best_by_op = summary.get("best_by_op") or {}
    if best_by_op:
        # Keep the markdown summary short: one best row per op, with full data in
        # CSV/JSONL for deeper analysis.
        lines.extend(["", "## Best Bandwidth By Op", ""])
        lines.append(
            "GB/s below is *useful* throughput (optimal-model bytes / time), "
            "comparable within a collective family. See the `wire_gbps` and "
            "`byte_model` CSV columns for actual traffic and overhead; use the "
            "`us` latency column to compare across ops."
        )
        lines.append("")
        lines.append("| op | payload | latency us | useful GB/s |")
        lines.append("| --- | ---: | ---: | ---: |")
        for op, row in sorted(best_by_op.items()):
            lines.append(
                f"| `{op}` | {row.get('payload')} | "
                f"{float(row.get('us') or 0):.2f} | {float(row.get('gbps') or 0):.2f} |"
            )

    failures = summary.get("failures") or []
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            artifact = row.get("error_artifact") or "results.jsonl"
            lines.append(
                f"- case {row.get('case_index')} `{row.get('op')}` "
                f"{format_bytes(int(row.get('payload_bytes') or 0))}: "
                f"{row.get('note_first_line') or 'failed'} (`{artifact}`)"
            )

    path = run.run_dir / "results_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return relative_artifact(run.run_dir, path)


def write_postrun_artifacts(run: RunContext) -> dict[str, Any]:
    """Write CSVs, plots, summaries, and an artifact index after the sweep."""
    csv_paths: list[str] = []
    results_csv = run.csv_dir / "results.csv"
    write_csv(results_csv, run.results)
    csv_paths.append(relative_artifact(run.run_dir, results_csv))
    ok_rows = [row for row in run.results if row.get("ok")]
    if ok_rows:
        ok_csv = run.csv_dir / "results_ok.csv"
        write_csv(ok_csv, ok_rows)
        csv_paths.append(relative_artifact(run.run_dir, ok_csv))
    failed_rows = [row for row in run.results if not row.get("ok")]
    if failed_rows:
        failed_csv = run.csv_dir / "results_failed.csv"
        write_csv(failed_csv, failed_rows)
        csv_paths.append(relative_artifact(run.run_dir, failed_csv))

    plot_paths = maybe_write_plots(run.results, run)
    # Standalone trace analysis: only produces output when a profiling run
    # captured traces. Adds a comm-time JSON summary and a per-device plot.
    trace_outputs = maybe_write_trace_summary(run)
    plot_paths.extend(trace_outputs.get("trace_plots", []))
    summary = build_run_summary(run.results, run)
    write_json(run.run_dir / "run_summary.json", summary)
    report = write_markdown_summary(run, summary, plot_paths, csv_paths)
    lab_artifacts_root = run.run_dir / "lab_artifacts"
    # Lab spec operations can create nested JSON/Markdown files. Indexing them
    # here makes a run directory browsable without knowing which labs were run.
    lab_artifacts = (
        sorted(
            relative_artifact(run.run_dir, path)
            for path in lab_artifacts_root.rglob("*")
            if path.is_file()
        )
        if lab_artifacts_root.exists()
        else []
    )
    artifact_index = {
        # This index is intentionally redundant with run_summary.json. It is the
        # quickest way for another tool or a student to discover where outputs
        # were written.
        "logs": [
            "logs/console.log",
            "logs/stdout.log",
            "logs/stderr.log",
        ],
        "results_jsonl": "results.jsonl",
        "csvs": csv_paths,
        "plots": plot_paths,
        "traces": sorted(
            {
                str(row["trace_artifact"])
                for row in run.results
                if row.get("trace_artifact")
            }
        ),
        "memory_profiles": sorted(
            {
                str(row["memory_profile_artifact"])
                for row in run.results
                if row.get("memory_profile_artifact")
            }
        ),
        "errors": sorted(
            {
                str(row["error_artifact"])
                for row in run.results
                if row.get("error_artifact")
            }
        ),
        "lab_artifacts": lab_artifacts,
        "trace_summaries": trace_outputs.get("trace_summaries", []),
        "report": report,
        "run_summary": "run_summary.json",
        "run_config": "run_config.json",
        "run_metadata": "run_metadata.json",
    }
    write_json(run.run_dir / "artifact_index.json", artifact_index)
    return artifact_index


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Defaults here are intentionally ordinary and conservative. Lab-specific
    defaults are applied later by ``apply_lab_defaults()`` so the help text can
    show the raw knobs without hiding the lab profiles.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lab",
        choices=(
            "lab0",
            "lab1",
            "lab2",
            "lab3",
            "lab4",
            "lab5",
            "lab6",
            "lab7",
            "lab8",
            "lab9",
            "lab10",
            "lab11",
        ),
        default=None,
        help=(
            "lab profile; lab1 is single-hop communication, "
            "lab2 is token ring, lab3 is Pallas memory spaces, "
            "lab4 is semaphore bug zoo, lab5-lab9 are composed custom "
            "collectives, lab10 is multi-host run-control smoke, "
            "lab11 is the bandwidth-optimal shard-ring all-reduce"
        ),
    )
    parser.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT),
        help="root directory for auto-named runs",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="optional run name; defaults to collective_bench-<timestamp>-<id>",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="exact run directory; overrides --run-root and --run-name",
    )
    parser.add_argument(
        "--ops",
        default=None,
        help="comma-separated ops, or 'all'",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="comma-separated payload bytes per device, e.g. 1KiB,4MiB",
    )
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16, float32, or int32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--axis-name", default="x")
    parser.add_argument(
        "--neighbor-direction",
        choices=("right", "left"),
        default="right",
        help="direction for pallas_neighbor_copy; right means device i writes to i+1",
    )
    parser.add_argument(
        "--pallas-tile-rows",
        type=int,
        default=4,
        help="local row count for Lab 1 Pallas tiles",
    )
    parser.add_argument(
        "--pallas-min-cols",
        type=int,
        default=128,
        help="minimum local column count for Lab 1 Pallas tiles",
    )
    parser.add_argument(
        "--pallas-memory-space",
        choices=("VMEM", "HBM"),
        default="HBM",
        help=(
            "Pallas TPU memory space for custom whole-tile refs; "
            "use VMEM only for small payloads or explicit memory-space experiments"
        ),
    )
    parser.add_argument(
        "--pallas-collective-id",
        type=int,
        default=1,
        help="Pallas collective_id for semaphore/barrier allocation",
    )
    parser.add_argument(
        "--token-hops",
        type=str,
        default=None,
        help=(
            "ring hop count(s) for token-ring labs; accepts a single value or a "
            "comma-separated sweep such as 0,1,2,3; defaults to "
            "local_device_count - 1"
        ),
    )
    parser.add_argument(
        "--lab8-chunks",
        type=int,
        default=4,
        help="Lab 8 chunk count for serialized chunked ring experiments",
    )
    parser.add_argument(
        "--lab8-buffer-count",
        type=int,
        default=2,
        help=(
            "Lab 8 buffer-slot count. Modeled for the serialized path; the fused "
            "pallas_db_token_ring kernel uses it as real HBM double-buffer slots "
            "(must be >= 2)."
        ),
    )
    parser.add_argument(
        "--lab8-inner-cols",
        type=int,
        default=0,
        help=(
            "Lab 8 inner VMEM pipeline width (columns per block) for the fused "
            "pallas_db_token_ring kernel. 0 (default) auto-sizes the largest "
            "VMEM-safe block; a small positive value shows the micro-transfer "
            "penalty of an undersized block."
        ),
    )
    parser.add_argument(
        "--lab9-mesh-shape",
        default="auto",
        help="Lab 9 logical 2D mesh shape, for example 2x2, 2x4, or auto",
    )
    parser.add_argument(
        "--lab9-axis-order",
        choices=("x_then_y", "y_then_x"),
        default="x_then_y",
        help="Lab 9 staged all-gather axis order",
    )
    parser.add_argument(
        "--lab11-ring-order",
        choices=("auto", "ids"),
        default="auto",
        help=(
            "Lab 11 ring layout: auto walks a unit-step Hamiltonian cycle over "
            "device coords when one exists; ids uses jax.devices() order and "
            "is the cleanest byte-only comparison with pmap_token_ring"
        ),
    )
    parser.add_argument(
        "--lab10-expected-process-count",
        type=int,
        default=None,
        help="Lab 10 validation: expected jax.process_count()",
    )
    parser.add_argument(
        "--lab10-expected-global-devices",
        type=int,
        default=None,
        help="Lab 10 validation: expected len(jax.devices())",
    )
    parser.add_argument(
        "--lab3-scale",
        type=float,
        default=1.5,
        help="Lab 3 local arithmetic scale",
    )
    parser.add_argument(
        "--lab3-bias",
        type=float,
        default=0.25,
        help="Lab 3 local arithmetic bias",
    )
    parser.add_argument(
        "--lab4-run-bug",
        default="wrong_neighbor_map",
        help=(
            "Lab 4: which catalog bug the pallas_semaphore_bug op reproduces. "
            "Safe (correctness-only) bugs run in-process; hang/crash/race bugs "
            "run only via the guarded subprocess path with --lab4-allow-dangerous."
        ),
    )
    parser.add_argument(
        "--lab4-allow-dangerous",
        action="store_true",
        help=(
            "Lab 4: permit hang/crash/race bug repros, which run ONLY in an "
            "isolated subprocess under --lab4-bug-timeout. Use a disposable VM; a "
            "real over-wait can force a TPU runtime restart."
        ),
    )
    parser.add_argument(
        "--lab4-bug-timeout",
        type=float,
        default=20.0,
        help="Lab 4: wall-clock seconds before a guarded dangerous bug repro is killed",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="skip small correctness checks before timing",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="optional profiler trace root; implies profiling",
    )
    parser.add_argument(
        "--profile",
        "--collect-profiles",
        "--xprof",
        dest="profile",
        action="store_true",
        help="capture JAX/XProf profiler traces",
    )
    parser.add_argument("--profile-cases", type=int, default=1, help="max traced/profiled cases")
    parser.add_argument("--profile-server-port", type=int, default=0)
    parser.add_argument(
        "--memory-profiles",
        action="store_true",
        help="write JAX device memory profiles for the first --profile-cases cases",
    )
    parser.add_argument(
        "--trace-op",
        default=None,
        help="only trace this op when profiling is enabled",
    )
    parser.add_argument(
        "--trace-size",
        type=parse_size,
        default=None,
        help="only trace this payload size when profiling is enabled",
    )
    parser.add_argument(
        "--perfetto",
        action="store_true",
        help="ask JAX to emit Perfetto traces when tracing",
    )
    parser.add_argument(
        "--xla-dump-to",
        default=None,
        help="optional XLA dump directory; use 'run' for <run_dir>/xla_dumps",
    )
    parser.add_argument("--no-plots", action="store_true", help="skip matplotlib plots")
    parser.add_argument("--no-diagnostics", action="store_true", help="skip diagnostics JSON files")
    parser.add_argument(
        "--external",
        default=None,
        help=(
            "command template for --ops external; placeholders: "
            "{bytes}, {dtype}, {iters}, {warmup}, {devices}"
        ),
    )
    return parser


def dispatch_case(
    jax: Any,
    jnp: Any,
    lax: Any,
    args: argparse.Namespace,
    run: RunContext,
    op: str,
    payload_bytes: int,
    dtype: Any,
    n_devices: int,
) -> BenchResult:
    """Run a single (op, payload) case and return its BenchResult.

    Any exception raised while building or running the case is captured and
    turned into a failed BenchResult so the sweep keeps going and the traceback
    is recorded in the run artifacts.
    """
    try:
        # Dispatch stays explicit instead of clever so each operation family can
        # be found quickly by name during teaching or debugging.
        if op in PMAP_OPS:
            return run_pmap_op(
                jax, jnp, lax, args, run, op, payload_bytes, dtype, n_devices
            )
        if op == "pallas_all_gather":
            return run_pallas_all_gather(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pallas_neighbor_copy":
            return run_pallas_neighbor_copy(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pallas_token_ring":
            return run_pallas_token_ring(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pallas_ring_all_gather":
            return run_pallas_ring_all_gather(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pallas_ring_reduce_scatter":
            return run_pallas_ring_reduce_scatter(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pallas_ring_all_reduce":
            return run_pallas_ring_all_reduce(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pallas_chunked_token_ring":
            return run_pallas_chunked_token_ring(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pallas_db_token_ring":
            return run_pallas_db_token_ring(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "xla_token_ring":
            return run_xla_token_ring(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pallas_2d_staged_all_gather":
            return run_pallas_2d_staged_all_gather(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pallas_vmem_arith":
            return run_pallas_vmem_arith(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pallas_semaphore_correct":
            return run_pallas_semaphore_correct(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pallas_semaphore_bug":
            return run_pallas_semaphore_bug(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "semaphore_bug_zoo":
            return run_semaphore_bug_zoo(args, run, payload_bytes)
        if op == "pmap_rs_ag_all_reduce":
            return run_pmap_rs_ag_all_reduce(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "pmap_rs_ag_all_reduce_bidir":
            return run_pmap_rs_ag_all_reduce_bidir(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op == "xla_all_reduce":
            return run_xla_all_reduce(
                jax, jnp, args, run, payload_bytes, dtype, n_devices
            )
        if op in LAB_SPEC_OPS:
            return run_lab_spec_op(jax, args, run, op, payload_bytes, n_devices)
        if op == "lab10_topology_smoke":
            return run_lab10_topology_smoke(jax, args, run, payload_bytes, n_devices)
        if op == "lab10_process_collective_smoke":
            return run_lab10_process_collective_smoke(
                jax, args, run, payload_bytes, n_devices
            )
        if op == "external":
            return run_external(args, payload_bytes, args.dtype, n_devices)
        raise AssertionError(op)
    except Exception as exc:
        return BenchResult(
            op=op,
            layer="unknown",
            payload_bytes=payload_bytes,
            logical_bytes=0,
            seconds=None,
            ok=False,
            note="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: parse, configure, run the sweep, and write artifacts."""
    # Self-test entrypoint for run_guarded_subprocess: hang forever with no JAX
    # import and no TPU access, so the guard's timeout + group-kill teardown can be
    # validated safely. The parent kills this child's process group on timeout.
    if SELFTEST_HANG_FLAG in (sys.argv[1:] if argv is None else argv):
        print("selftest-hang: sleeping until killed", flush=True)
        while True:
            time.sleep(3600)
    args = build_parser().parse_args(argv)
    apply_lab_defaults(args)
    ops = normalize_ops(parse_csv(args.ops))
    sizes = parse_csv(args.sizes, parse_size)
    hop_sweep = parse_token_hops(args.token_hops)
    run_dir = make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_env_pre_import(args, run_dir)
    run = RunContext(run_dir=run_dir, args=args)
    # run_config is written before JAX import and before the long sweep starts,
    # so even an early import/backend failure leaves useful provenance.
    write_json(
        run_dir / "run_config.json",
        {
            "argv": sys.argv[:] if argv is None else [sys.argv[0], *argv],
            "args": vars(args),
            "ops": ops,
            "sizes": sizes,
            "hop_sweep": hop_sweep,
            "created_at_unix": run.started_at_unix,
            "created_at_local": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    )

    with FdTee(run_dir / "logs"):
        print(f"[run] dir={run_dir}", flush=True)
        if not args.no_diagnostics:
            write_json(run_dir / "diagnostics" / "preimport.json", collect_diagnostics())

        # Keep JAX import after argument parsing so --help stays fast and quiet.
        # More importantly, this comes after configure_env_pre_import() so XLA
        # dump flags and matplotlib config are visible before backend setup.
        import jax
        import jax.numpy as jnp
        from jax import lax

        dtype = dtype_from_name(jnp, args.dtype)
        n_devices = jax.local_device_count()
        if args.profile_server_port:
            # Optional live profiler server. Trace capture through maybe_trace()
            # is separate and writes artifacts into the run directory.
            jax.profiler.start_server(int(args.profile_server_port))
            print(f"[profile] server listening on {args.profile_server_port}")

        metadata = {
            "argv": sys.argv[:] if argv is None else [sys.argv[0], *argv],
            "args": vars(args),
            "ops": ops,
            "sizes": sizes,
            "hop_sweep": hop_sweep,
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "env": env_subset(),
            "packages": {
                "jax": getattr(jax, "__version__", package_version("jax")),
                "jaxlib": package_version("jaxlib"),
                "libtpu": package_version("libtpu"),
                "matplotlib": package_version("matplotlib"),
                "numpy": package_version("numpy"),
            },
            "device_report": device_report(jax),
            "git": git_state(pathlib.Path.cwd()),
        }
        write_json(run_dir / "run_metadata.json", metadata)
        if not args.no_diagnostics:
            write_json(run_dir / "diagnostics" / "runtime.json", collect_diagnostics(jax))
            with contextlib.suppress(Exception):
                write_json(
                    run_dir / "diagnostics" / "memory_start.json",
                    sample_device_memory(jax, "start"),
                )

        print(f"jax={jax.__version__} backend={jax.default_backend()} local_devices={n_devices}")
        print(f"devices={jax.devices()}")
        print_header()

        try:
            for payload_bytes in sizes:
                for op in ops:
                    # Only token-ring ops vary with the hop count, so they expand
                    # across the --token-hops sweep. Everything else runs once.
                    case_hops = hop_sweep if op in HOP_SWEEP_OPS else [None]
                    for hops_value in case_hops:
                        run.case_index += 1
                        # token_ring_hops() reads args.token_hops, so set the
                        # scalar value for the case currently being run.
                        args.token_hops = hops_value
                        result = dispatch_case(
                            jax,
                            jnp,
                            lax,
                            args,
                            run,
                            op,
                            payload_bytes,
                            dtype,
                            n_devices,
                        )
                        record_result(
                            run,
                            args,
                            result,
                            requested_payload_bytes=payload_bytes,
                            n_devices=n_devices,
                            dtype_name=args.dtype,
                            token_hops=hops_value if op in HOP_SWEEP_OPS else None,
                        )
        finally:
            # Always attempt post-run diagnostics and artifact writing. This is
            # useful when a late case fails or the user interrupts after partial
            # progress: earlier rows still become CSVs/summaries.
            if not args.no_diagnostics:
                with contextlib.suppress(Exception):
                    write_json(
                        run_dir / "diagnostics" / "memory_end.json",
                        sample_device_memory(jax, "end"),
                    )
                with contextlib.suppress(Exception):
                    write_json(run_dir / "diagnostics" / "postrun.json", collect_diagnostics(jax))
            artifacts = write_postrun_artifacts(run)
            print(f"[done] artifacts={json.dumps(artifacts, sort_keys=True)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
