"""Lab 8: chunked and pipelined ring movement.

This is a systems-skills lab about building a real fused collective kernel and
characterizing it honestly. It ships three implementations of the same token-ring
reduction so students can read a schedule, build the fused version that overlaps
communication with local work, and measure where that overlap helps.

    pallas_chunked_token_ring
        The clarity-first teaching path.  It calls the Lab 1 one-hop primitive
        once per chunk/hop pair, so it is intentionally serialized: chunking
        without pipelining.  Excellent for explaining chunk boundaries,
        collective IDs, and source history.

    pallas_double_buffered_token_ring
        The fused ring.  A single Pallas program carries the ring across
        ``hops + 1`` steps.  Each device alternates HBM buffer slots, starts an
        async remote DMA into the neighbor's next slot, accumulates the current
        slot through an inner HBM<->VMEM pipeline while the remote copy is in
        flight, and waits only when correctness requires it.  ``buffer_count``
        controls real HBM double-buffer slots here, and a capacity semaphore
        guards slot reuse.  This is where the overlap is real.

    xla_fast_token_ring
        An XLA reference point.  For a full ring it lowers to ``lax.psum``; for
        partial-hop experiments it uses ``lax.ppermute``.

Matching XLA is not the goal: this whole-token ring moves about twice the bytes
of an optimal all-reduce, so a tuned collective wins on large transfers by moving
fewer bytes, not by overlapping better.  The lesson is to build the fused kernel
correctly and use the trace and byte model to explain what the overlap buys (it
hides fixed per-phase overhead, biggest at small/medium payloads) and what it
does not.  A bandwidth-optimal chunk-per-hop ring is a natural follow-on lab.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

try:  # Normal repository layout: collective_comm_bench/labs/*.py
    from labs import lab1_single_hop  # type: ignore
except Exception:  # pragma: no cover - useful for standalone review in /mnt/data.
    import sys
    from pathlib import Path

    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    import lab1_single_hop  # type: ignore

try:
    from labs import lab_spec_utils  # type: ignore
except Exception:  # pragma: no cover - fallback keeps this file importable alone.
    try:
        import sys
        from pathlib import Path

        _THIS_DIR = Path(__file__).resolve().parent
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import lab_spec_utils  # type: ignore[no-redef]
    except Exception:
        lab_spec_utils = None  # type: ignore[assignment]


SUPPORTED_DIRECTIONS = frozenset({"right", "left"})

# Small menu used by the spec artifact. These are not magic values. They simply
# give students sensible chunk sizes to sweep across latency-dominated and more
# bandwidth-dominated regimes.
SUGGESTED_CHUNK_BYTES = (
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
)

SUPPORTED_KERNEL_MODES = frozenset({"auto", "serialized", "pallas-db", "xla-psum"})
# 0 means "auto": choose_inner_pipeline_cols picks the largest VMEM-safe column
# block. A positive value pins the inner block (useful for teaching the
# micro-transfer penalty of a too-small block).
DEFAULT_INNER_PIPELINE_COLS = 0



@dataclasses.dataclass(frozen=True)
class ChunkedTokenRingCase:
    """Container returned to the benchmark harness.

    The harness needs a sharded input ``x``, a jitted function ``fn``, and an
    expected result for correctness. The remaining fields are here to make CSV
    rows, notes, and lab artifacts self-explanatory.

    Shape convention
    ----------------

    ``x`` has global shape::

        [device, chunk, row, col]

    The mesh partitions the first dimension across devices. Each device owns all
    chunks for its local token.

    Byte vocabulary
    ---------------

    ``actual_payload_bytes`` is the full per-device payload across all chunks.
    ``chunk_payload_bytes`` is the size of one chunk moved by a single Lab 1 hop.
    The serialized teaching path sends ``actual_payload_bytes * hops`` bytes per
    device, because every chunk is sent for every hop.
    """

    x: Any
    fn: Callable[[Any], Any]
    expected_sums: Any
    actual_payload_bytes: int
    chunk_payload_bytes: int
    tile_rows: int
    tile_cols: int
    n_chunks: int
    buffer_count: int
    direction: str
    hops: int
    n_devices: int
    kernel_mode: str = "serialized"
    fast_path_reason: str = ""

    @property
    def note(self) -> str:
        """Short human-readable note for benchmark tables."""
        reason = f" reason={self.fast_path_reason}" if self.fast_path_reason else ""
        return (
            f"lab8-kernel={self.kernel_mode} direction={self.direction} "
            f"chunks={self.n_chunks} buffers={self.buffer_count} "
            f"hops={self.hops} devices={self.n_devices} "
            f"chunk={self.chunk_payload_bytes}B "
            f"tile={self.tile_rows}x{self.tile_cols} "
            f"serialized_phases={self.serialized_remote_copy_phases}"
            f"{reason}"
        )

    @property
    def implemented_overlap(self) -> bool:
        """Whether the selected custom path contains actual async overlap."""
        return self.kernel_mode == "pallas-db"

    @property
    def is_roofline_reference(self) -> bool:
        """Whether the selected path is XLA's tuned collective reference."""
        return self.kernel_mode == "xla-psum"

    @property
    def num_devices(self) -> int:
        """Alias used by some benchmark/reporting code."""
        return self.n_devices

    @property
    def input_bytes_per_device(self) -> int:
        """Bytes each device starts with before chunked ring movement."""
        return self.actual_payload_bytes

    @property
    def output_bytes_per_device(self) -> int:
        """Bytes each device writes after accumulating all chunks."""
        # The output dtype is float32 in this lab because token sums accumulate in
        # float32. The benchmark's payload field still describes input bytes.
        return self.n_chunks * self.tile_rows * self.tile_cols * 4

    @property
    def wire_bytes(self) -> int:
        """Backward-compatible per-device send-byte estimate."""
        return self.serialized_send_bytes_per_device

    @property
    def serialized_send_bytes_per_device(self) -> int:
        """Per-device send bytes for the implemented teaching path."""
        return self.actual_payload_bytes * self.hops

    @property
    def serialized_recv_bytes_per_device(self) -> int:
        """Per-device receive bytes for the implemented teaching path."""
        return self.actual_payload_bytes * self.hops

    @property
    def serialized_remote_copy_phases(self) -> int:
        """Number of Lab 1 neighbor-copy phases emitted by the teaching path."""
        return self.n_chunks * self.hops

    @property
    def full_ring_hops(self) -> int:
        """Number of hops needed for every receiver to see every source once."""
        return max(0, self.n_devices - 1)

    @property
    def is_full_ring(self) -> bool:
        """Whether this case traverses a complete N-device ring."""
        return self.hops == self.full_ring_hops

    @property
    def bytes_per_remote_copy_phase(self) -> int:
        """Bytes moved by one per-chunk Lab 1 hop on one device."""
        return self.chunk_payload_bytes

    def byte_model(self) -> dict[str, int | float | bool | str]:
        """Return benchmark-friendly byte accounting for this case."""
        return chunked_ring_byte_model(
            n_devices=self.n_devices,
            full_payload_bytes=self.actual_payload_bytes,
            chunk_payload_bytes=self.chunk_payload_bytes,
            n_chunks=self.n_chunks,
            hops=self.hops,
            direction=self.direction,
            buffer_count=self.buffer_count,
            kernel_mode=self.kernel_mode,
        )

    def buffer_schedule(self, *, max_chunks: int = 16) -> list[dict[str, Any]]:
        """Return the planned buffer-slot schedule for this case."""
        return buffer_slot_schedule(
            n_chunks=self.n_chunks,
            buffer_count=self.buffer_count,
            max_chunks=max_chunks,
        )


LAB_SPEC: dict[str, Any] = {
    "lab": "lab8",
    "title": "Lab 8: Chunked And Pipelined Ring",
    "goal": (
        "Build a real fused collective kernel and characterize it: a Pallas "
        "double-buffered ring with async remote DMA, capacity synchronization, "
        "and inner HBM<->VMEM accumulation overlapped with communication. The "
        "point is the machinery and an honest read of where overlap helps, not "
        "beating a tuned XLA collective."
    ),
    "implemented_ops": [
        "`pmap_token_ring`: dependency-chain reference from Lab 2",
        "`pallas_chunked_token_ring`: serialized chunked ring built from Lab 1 hops",
        "`pallas_double_buffered_token_ring`: fused custom Pallas ring with real double/N-buffering",
        "`xla_fast_token_ring`: tuned XLA collective reference (`lax.psum` for full rings)",
        "`lab8_chunked_pipeline_spec`: source-history, byte-model, collective-ID, buffer-slot, and overlap artifact",
    ],
    "deferred_ops": [
        "Add a bidirectional reduce-scatter/all-gather variant to use both ICI directions",
        "Relax capacity synchronization to allow controlled multi-slot run-ahead",
        "Teach TPU interpret-mode race checks as a required pre-profile debugging step",
        "Carry the fused schedule into optimized reduce-scatter and all-reduce",
    ],
    "byte_model": [
        "Chunking does not reduce total bytes: ring send bytes are `hops * full_payload_bytes` per device",
        "The serialized path emits `chunks * hops` composed Lab 1 phases",
        "The fused Pallas path emits one ring program with `hops + 1` steps and HBM buffer slots",
        "Real double-buffering changes overlap and latency hiding, not required payload bytes",
    ],
    "pass_condition": [
        "reference token-ring correctness remains stable across payload sizes",
        "serialized Pallas chunked ring matches per-chunk token sums on TPU",
        "full output tiles are checked, not only scalar rank markers",
        "spec artifact defines source history, byte model, collective IDs, buffer ownership, and overlap evidence rules",
        "pallas-db path matches full-tile expected sums and profiles with RDMA start/accumulate/wait structure",
    ],
    "artifacts": [
        "results.jsonl",
        "csvs/results.csv",
        "plots/latency_by_payload.png",
        "plots/bandwidth_by_payload.png",
        "lab_artifacts/*lab8_chunked_pipeline_spec*",
    ],
    "next_steps": [
        "Use XProf traces to decide whether overlap exists or is only hoped for",
        "Carry chunked schedules into topology-aware Lab 10",
    ],
}


def normalize_direction(direction: str | None) -> str:
    """Validate and normalize the logical ring direction."""
    direction = "right" if direction is None else str(direction).strip().lower()
    if direction not in SUPPORTED_DIRECTIONS:
        raise ValueError(
            f"unknown direction {direction!r}; expected one of "
            f"{sorted(SUPPORTED_DIRECTIONS)}"
        )
    return direction


def normalize_hops(hops: int | str | None, *, n_devices: int | None = None) -> int:
    """Normalize a hop count.

    ``None`` means a full ring when ``n_devices`` is known. Friendly string
    values such as ``"full"``, ``"all"``, and ``"n-1"`` also mean a full ring.
    This mirrors the default token-ring teaching pattern used earlier in the
    course.
    """
    if hops is None:
        if n_devices is None:
            return 0
        return max(0, int(n_devices) - 1)
    if isinstance(hops, str):
        cleaned = hops.strip().lower()
        if cleaned in {"", "default", "full", "all", "ring", "n-1"}:
            if n_devices is None:
                raise ValueError(f"hops={hops!r} requires n_devices")
            return max(0, int(n_devices) - 1)
        hops_int = int(cleaned)
    else:
        hops_int = int(hops)
    if hops_int == -1:
        if n_devices is None:
            raise ValueError("hops=-1 requires n_devices")
        return max(0, int(n_devices) - 1)
    if hops_int < 0:
        raise ValueError(f"hops must be non-negative or -1, got {hops_int}")
    return hops_int


def normalize_positive_int(value: int | str | None, *, name: str, default: int) -> int:
    """Normalize integer CLI-style options such as chunk and buffer counts."""
    value_int = int(default if value is None else value)
    if value_int <= 0:
        raise ValueError(f"{name} must be positive, got {value_int}")
    return value_int


def ceil_div(numer: int, denom: int) -> int:
    """Integer ceiling division with a clearer error for bad denominators."""
    numer = int(numer)
    denom = int(denom)
    if denom <= 0:
        raise ValueError(f"denominator must be positive, got {denom}")
    return -(-numer // denom)


def source_ranks_seen(
    *,
    receiver_rank: int,
    n_devices: int,
    hops: int,
    direction: str,
) -> list[int]:
    """Return the source ranks a receiver has accumulated after ``hops``.

    Hop 0 is always the local token. A right-moving ring means device ``i`` sends
    to ``i + 1``; therefore receiver ``r`` gets ``r - hop`` at hop ``hop``. A
    left-moving ring is the mirror image.
    """
    direction = normalize_direction(direction)
    n_devices = int(n_devices)
    hops = normalize_hops(hops)
    if n_devices <= 0:
        raise ValueError(f"n_devices must be positive, got {n_devices}")

    if direction == "right":
        return [(int(receiver_rank) - hop) % n_devices for hop in range(hops + 1)]
    return [(int(receiver_rank) + hop) % n_devices for hop in range(hops + 1)]


def source_history_table(
    *,
    n_devices: int,
    hops: int,
    direction: str,
) -> list[list[int]]:
    """Return ``[receiver_rank][arrival_index] -> source_rank``."""
    return [
        source_ranks_seen(
            receiver_rank=rank,
            n_devices=n_devices,
            hops=hops,
            direction=direction,
        )
        for rank in range(int(n_devices))
    ]


def expected_chunk_sum_values(
    *,
    n_devices: int,
    n_chunks: int,
    hops: int,
    direction: str,
) -> list[list[float]]:
    """Return the expected scalar value for every receiver/chunk.

    This helper models the hand-checkable scalar marker at ``row=0, col=0``.
    The full teaching input also contains row/column offsets so full-tile
    correctness can catch partial-copy and stale-slot bugs.

    For chunk ``c`` at the scalar marker, a receiver should see:

        sum(10 * source + c for source in sources_seen)

    The function supports partial-hop experiments as well as the full ``N - 1``
    ring traversal.
    """
    direction = normalize_direction(direction)
    n_devices = int(n_devices)
    n_chunks = normalize_positive_int(n_chunks, name="n_chunks", default=1)
    hops = normalize_hops(hops)

    values: list[list[float]] = []
    for receiver in range(n_devices):
        seen = source_ranks_seen(
            receiver_rank=receiver,
            n_devices=n_devices,
            hops=hops,
            direction=direction,
        )
        values.append(
            [float(sum(src * 10.0 + chunk for src in seen)) for chunk in range(n_chunks)]
        )
    return values


def expected_chunk_sums(
    jnp: Any,
    *,
    n_devices: int,
    n_chunks: int,
    hops: int,
    direction: str,
) -> Any:
    """JAX-array wrapper around ``expected_chunk_sum_values``."""
    return jnp.array(
        expected_chunk_sum_values(
            n_devices=n_devices,
            n_chunks=n_chunks,
            hops=hops,
            direction=direction,
        ),
        dtype=jnp.float32,
    )


def expected_chunk_scalar_values(
    *,
    n_devices: int,
    n_chunks: int,
    hops: int,
    direction: str,
) -> list[list[float]]:
    """Alias with a more explicit teaching name.

    ``expected_chunk_sum_values`` is the compact name; this alias reads better in
    notebooks and worksheets.
    """
    return expected_chunk_sum_values(
        n_devices=n_devices,
        n_chunks=n_chunks,
        hops=hops,
        direction=direction,
    )


def chunk_tile_shape_for_payload(
    *,
    payload_bytes: int,
    itemsize: int,
    tile_rows: int,
    min_cols: int,
    n_chunks: int,
) -> tuple[int, int, int, int]:
    """Choose a per-chunk tile shape for the requested full payload.

    ``payload_bytes`` is the requested full per-device input payload. Lab 8 splits
    that full payload across ``n_chunks`` chunks, then rounds the per-chunk tile
    up to ``tile_rows * tile_cols`` elements.

    Returns:

        ``(rows, cols, chunk_payload_bytes, actual_payload_bytes)``
    """
    payload_bytes = max(1, int(payload_bytes))
    itemsize = int(itemsize)
    rows = max(1, int(tile_rows))
    min_cols = max(1, int(min_cols))
    n_chunks = normalize_positive_int(n_chunks, name="n_chunks", default=1)
    if itemsize <= 0:
        raise ValueError(f"itemsize must be positive, got {itemsize}")

    # Round up so the total actual payload is at least the requested payload.
    cols = max(min_cols, ceil_div(payload_bytes, itemsize * rows * n_chunks))
    chunk_payload_bytes = rows * cols * itemsize
    actual_payload_bytes = n_chunks * chunk_payload_bytes
    return rows, cols, chunk_payload_bytes, actual_payload_bytes


def make_chunked_rank_input(
    jnp: Any,
    *,
    n_devices: int,
    n_chunks: int,
    rows: int,
    cols: int,
    dtype: Any,
) -> Any:
    """Create the teaching input tensor for Lab 8.

    The scalar at ``[row=0, col=0]`` is the hand-checkable pattern:

        x[source, chunk, 0, 0] = 10 * source + chunk

    The rest of the tile includes tiny row/column patterns. That makes the full
    correctness check meaningful: a partial-copy or stale-slot bug cannot hide
    behind a correct rank marker at ``[0, 0]``.
    """
    src = jnp.arange(n_devices, dtype=jnp.float32).reshape(n_devices, 1, 1, 1)
    chunk = jnp.arange(n_chunks, dtype=jnp.float32).reshape(1, n_chunks, 1, 1)
    row_pattern = (jnp.arange(rows, dtype=jnp.float32) % 8).reshape(1, 1, rows, 1) * 0.25
    col_pattern = (jnp.arange(cols, dtype=jnp.float32) % 16).reshape(1, 1, 1, cols) * 0.03125
    x_host = src * 10.0 + chunk + row_pattern + col_pattern
    return x_host.astype(dtype)


def expected_chunk_tiles(
    jnp: Any,
    *,
    n_devices: int,
    n_chunks: int,
    rows: int,
    cols: int,
    dtype: Any,
    hops: int,
    direction: str,
) -> Any:
    """Return the full expected output tile for every receiver and chunk.

    The calculation starts from ``make_chunked_rank_input(...).astype(float32)``
    so low-precision input dtypes are modeled the same way as the real kernel.
    """
    base = make_chunked_rank_input(
        jnp,
        n_devices=n_devices,
        n_chunks=n_chunks,
        rows=rows,
        cols=cols,
        dtype=dtype,
    ).astype(jnp.float32)
    outputs = []
    for receiver in range(n_devices):
        sources = jnp.array(
            source_ranks_seen(
                receiver_rank=receiver,
                n_devices=n_devices,
                hops=hops,
                direction=direction,
            ),
            dtype=jnp.int32,
        )
        outputs.append(jnp.sum(base[sources, :, :, :], axis=0))
    return jnp.stack(outputs, axis=0)


def chunked_ring_byte_model(
    *,
    n_devices: int,
    full_payload_bytes: int,
    chunk_payload_bytes: int,
    n_chunks: int,
    hops: int,
    direction: str,
    buffer_count: int,
    kernel_mode: str = "serialized",
) -> dict[str, int | float | bool | str]:
    """Return byte and phase accounting for Lab 8.

    These numbers are algorithmic accounting for the selected Lab 8 path.
    They are not a claim about how a built-in XLA collective is implemented.
    """
    direction = normalize_direction(direction)
    kernel_mode = normalize_kernel_mode(kernel_mode)
    n_devices = int(n_devices)
    full_payload_bytes = int(full_payload_bytes)
    chunk_payload_bytes = int(chunk_payload_bytes)
    n_chunks = normalize_positive_int(n_chunks, name="n_chunks", default=1)
    buffer_count = normalize_positive_int(buffer_count, name="buffer_count", default=2)
    hops = normalize_hops(hops)

    if n_devices <= 0:
        raise ValueError(f"n_devices must be positive, got {n_devices}")
    if full_payload_bytes < 0:
        raise ValueError("full_payload_bytes must be non-negative")
    if chunk_payload_bytes < 0:
        raise ValueError("chunk_payload_bytes must be non-negative")

    serialized_phases = n_chunks * hops
    send_bytes = full_payload_bytes * hops
    recv_bytes = full_payload_bytes * hops
    chunk_send_bytes = chunk_payload_bytes * hops
    phase_count_per_chunk = hops

    return {
        "direction": direction,
        "n_devices": n_devices,
        "hops": hops,
        "is_full_ring": hops == max(0, n_devices - 1),
        "n_chunks": n_chunks,
        "buffer_count": buffer_count,
        "full_payload_bytes_per_device": full_payload_bytes,
        "chunk_payload_bytes_per_device": chunk_payload_bytes,
        "serialized_send_bytes_per_device": send_bytes,
        "serialized_recv_bytes_per_device": recv_bytes,
        "total_serialized_ring_bytes_all_devices": n_devices * send_bytes,
        "send_bytes_per_chunk_per_device": chunk_send_bytes,
        "remote_copy_bytes_per_phase_per_device": chunk_payload_bytes,
        "serialized_remote_copy_phases": serialized_phases,
        "remote_copy_phases_per_chunk": phase_count_per_chunk,
        "bytes_do_not_change_with_chunking": True,
        "kernel_mode": kernel_mode,
        "implemented_overlap": kernel_mode == "pallas-db",
        "uses_xla_tuned_collective": kernel_mode == "xla-psum",
        "fused_pallas_ring_steps": hops + 1,
        "required_buffer_slots_for_overlap": 2,
        "note": (
            "Serialized mode changes only phase granularity. pallas-db mode "
            "uses async remote DMA plus HBM<->VMEM accumulation so traces can "
            "show real overlap. xla-psum is an XLA reference; it moves fewer "
            "bytes (optimal all-reduce), so it is not an apples-to-apples target."
        ),
    }


def buffer_slot_schedule(
    *,
    n_chunks: int,
    buffer_count: int,
    max_chunks: int = 16,
) -> list[dict[str, Any]]:
    """Return the planned chunk -> buffer slot assignment.

    This schedule is an invariant ledger, not an implementation of overlap. A
    fused double-buffered kernel can use it to decide when each slot is safe to
    reuse.
    """
    n_chunks = normalize_positive_int(n_chunks, name="n_chunks", default=1)
    buffer_count = normalize_positive_int(buffer_count, name="buffer_count", default=2)
    max_chunks = max(0, int(max_chunks))

    return [
        {
            "chunk": chunk,
            "buffer_slot": chunk % buffer_count,
            "slot_epoch": chunk // buffer_count,
            "hazard_rule": (
                "slot may be reused only after all send waits, receive waits, "
                "local reads, and local writes for the previous occupant drain"
            ),
        }
        for chunk in range(min(n_chunks, max_chunks))
    ]


def planned_wavefront_events(
    *,
    n_chunks: int,
    hops: int,
    buffer_count: int,
    max_events: int = 64,
) -> list[dict[str, Any]]:
    """Preview the wavefront schedule a fused pipeline would try to implement.

    This table is not executed by ``pallas_chunked_token_ring``. It shows the
    shape of the future optimization: at pipeline epoch ``e``, chunk ``k`` can
    be on hop ``e - k`` when that hop is valid. Multiple rows with the same
    epoch are potential overlap opportunities.
    """
    n_chunks = normalize_positive_int(n_chunks, name="n_chunks", default=1)
    hops = normalize_hops(hops)
    buffer_count = normalize_positive_int(buffer_count, name="buffer_count", default=2)
    max_events = max(0, int(max_events))

    if hops == 0 or max_events == 0:
        return []

    rows: list[dict[str, Any]] = []
    final_epoch = (n_chunks - 1) + (hops - 1)
    for epoch in range(final_epoch + 1):
        for chunk in range(n_chunks):
            hop = epoch - chunk
            if 0 <= hop < hops:
                rows.append(
                    {
                        "epoch": epoch,
                        "chunk": chunk,
                        "hop": hop,
                        "buffer_slot": chunk % buffer_count,
                        "status": "planned only, not implemented by current serialized path",
                        "overlap_question": (
                            "Can this event run while another chunk is in a different hop? "
                            "Only a fused kernel and trace can answer yes."
                        ),
                    }
                )
                if len(rows) >= max_events:
                    return rows
    return rows


def collective_id_schedule(
    *,
    base_collective_id: int,
    n_chunks: int,
    hops: int,
    max_entries: int = 64,
) -> list[dict[str, int]]:
    """Return the chunk/hop -> collective ID mapping used by this teaching path.

    Lab 1 uses one collective ID for one communication pattern. Lab 8 emits many
    Lab 1-style phases, so every chunk/hop pair receives a unique ID.
    """
    base = int(base_collective_id)
    n_chunks = normalize_positive_int(n_chunks, name="n_chunks", default=1)
    hops = normalize_hops(hops)
    max_entries = max(0, int(max_entries))
    id_stride = max(1, hops)

    rows: list[dict[str, int]] = []
    for chunk in range(n_chunks):
        for hop in range(hops):
            if len(rows) >= max_entries:
                return rows
            rows.append(
                {
                    "chunk": chunk,
                    "hop": hop,
                    "collective_id": base + chunk * id_stride + hop,
                }
            )
    return rows


def serialized_phase_schedule(
    *,
    base_collective_id: int,
    n_chunks: int,
    hops: int,
    buffer_count: int,
    max_entries: int = 64,
) -> list[dict[str, int | str]]:
    """Return the actual sequential phase order for the implemented path.

    The current custom implementation processes chunk 0 through every hop,
    then chunk 1 through every hop, and so on. This table is useful when a
    profiler trace looks confusing: it tells students the order the composed
    Python/JAX program asked for.
    """
    base = int(base_collective_id)
    n_chunks = normalize_positive_int(n_chunks, name="n_chunks", default=1)
    buffer_count = normalize_positive_int(buffer_count, name="buffer_count", default=2)
    hops = normalize_hops(hops)
    max_entries = max(0, int(max_entries))
    id_stride = max(1, hops)

    rows: list[dict[str, int | str]] = []
    phase = 0
    for chunk in range(n_chunks):
        for hop in range(hops):
            if len(rows) >= max_entries:
                return rows
            rows.append(
                {
                    "phase": phase,
                    "chunk": chunk,
                    "hop": hop,
                    "buffer_slot": chunk % buffer_count,
                    "collective_id": base + chunk * id_stride + hop,
                    "current_status": "serialized: the next phase starts only after this hop completes",
                }
            )
            phase += 1
    return rows


def planned_overlapped_wavefront(
    *,
    n_chunks: int,
    hops: int,
    buffer_count: int,
    max_entries: int = 64,
) -> list[dict[str, int | str]]:
    """Sketch the wavefront a future fused pipelined kernel might target.

    This is a design artifact only. A real implementation needs per-slot
    capacity semaphores and proof in XProf that work is actually interleaved.
    """
    n_chunks = normalize_positive_int(n_chunks, name="n_chunks", default=1)
    buffer_count = normalize_positive_int(buffer_count, name="buffer_count", default=2)
    hops = normalize_hops(hops)
    max_entries = max(0, int(max_entries))

    rows: list[dict[str, int | str]] = []
    for time_step in range(n_chunks + hops - 1):
        for chunk in range(n_chunks):
            hop = time_step - chunk
            if hop < 0 or hop >= hops:
                continue
            if len(rows) >= max_entries:
                return rows
            rows.append(
                {
                    "time_step": time_step,
                    "chunk": chunk,
                    "hop": hop,
                    "buffer_slot": chunk % buffer_count,
                    "future_status": "planned overlap: legal only with per-slot capacity synchronization",
                }
            )
    return rows


def pipeline_planning_notes(*, buffer_count: int) -> list[str]:
    """Return the conceptual schedule for the future fused pipelined kernel."""
    buffer_count = normalize_positive_int(buffer_count, name="buffer_count", default=2)
    if buffer_count == 1:
        return [
            "one buffer slot forces conservative serialization",
            "the producer must not overwrite the only slot until all consumers are done",
            "use this setting to demonstrate why buffering is a flow-control tool",
        ]
    return [
        f"fill up to {buffer_count} slots before steady state",
        "steady state alternates enqueue, local work, and waits",
        "reuse slot s for chunk k only after chunk k-buffer_count has fully drained",
        "trace evidence, not configuration flags, decides whether overlap exists",
    ]




def normalize_kernel_mode(mode: str | None) -> str:
    """Normalize the Lab 8 implementation selector.

    ``auto`` chooses the fused Pallas double-buffered kernel when the platform
    and configuration can support it, otherwise it falls back to the XLA fast
    reference.  ``serialized`` selects the composed teaching path.
    """
    cleaned = "auto" if mode is None else str(mode).strip().lower()
    aliases = {
        "": "auto",
        "fast": "auto",
        "db": "pallas-db",
        "double-buffered": "pallas-db",
        "double_buffered": "pallas-db",
        "pallas_db": "pallas-db",
        "pallas-db": "pallas-db",
        "pallas": "pallas-db",
        "teaching": "serialized",
        "slow": "serialized",
        "chunked": "serialized",
        "serialized": "serialized",
        "xla": "xla-psum",
        "psum": "xla-psum",
        "xla-psum": "xla-psum",
        "roofline": "xla-psum",
    }
    cleaned = aliases.get(cleaned, cleaned)
    if cleaned not in SUPPORTED_KERNEL_MODES:
        raise ValueError(
            f"unknown Lab 8 kernel mode {mode!r}; expected one of "
            f"{sorted(SUPPORTED_KERNEL_MODES)}"
        )
    return cleaned


def _device_kind(devices: list[Any]) -> str:
    """Return a stable-ish device kind string without importing TPU-only code."""
    if not devices:
        return "unknown"
    return str(getattr(devices[0], "device_kind", getattr(devices[0], "platform", "unknown")))


def _looks_like_tpu(devices: list[Any]) -> bool:
    """Best-effort TPU detector used only for auto mode."""
    return "TPU" in _device_kind(devices).upper()


def choose_inner_pipeline_cols(
    *, cols: int, requested: int, rows: int = 1, budget_elems: int = 262_144
) -> int:
    """Choose an inner HBM<->VMEM pipeline column block that divides ``cols``.

    The inner block is what ``emit_pipeline`` streams through VMEM. Picking it too
    small is a performance trap: a 4 MiB tile with a 128-column block becomes
    thousands of micro-transfers and the "fast" kernel ends up slower than the
    serialized baseline. So unless the caller pins a value, this targets the
    LARGEST column block that (a) divides ``cols`` for full blocks and (b) keeps
    one VMEM tile (``rows * inner_cols`` elements) within ``budget_elems`` so it
    stays VMEM-safe as payloads grow.

    ``requested <= 0`` means "auto" (the default). A positive ``requested`` is
    honored as an upper bound (largest divisor of ``cols`` no larger than it), so
    students can still force a tiny block to *see* the micro-transfer penalty.
    """
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    if int(requested) <= 0:
        target = max(1, int(budget_elems) // rows)
    else:
        target = int(requested)
    target = max(1, min(target, cols))
    for candidate in range(target, 0, -1):
        if cols % candidate == 0:
            return candidate
    return 1


def choose_lab8_kernel_mode(
    *,
    requested_mode: str | None,
    devices: list[Any],
    buffer_count: int,
    hops: int,
    n_devices: int,
    tile_cols: int,
    inner_cols: int,
) -> tuple[str, str]:
    """Return ``(selected_mode, reason)`` for benchmark notes and artifacts."""
    mode = normalize_kernel_mode(requested_mode)
    buffer_count = normalize_positive_int(buffer_count, name="buffer_count", default=2)
    hops = normalize_hops(hops, n_devices=n_devices)
    inner_cols = choose_inner_pipeline_cols(cols=tile_cols, requested=inner_cols)

    if mode == "serialized":
        return "serialized", "requested serialized teaching path"
    if mode == "xla-psum":
        return "xla-psum", "requested XLA tuned collective reference"
    if mode == "pallas-db":
        if buffer_count < 2:
            raise ValueError("pallas-db requires lab8_buffer_count >= 2")
        if not _looks_like_tpu(devices):
            raise ValueError(
                "pallas-db requires TPU devices because it uses Pallas TPU RDMA; "
                f"found {_device_kind(devices)!r}"
            )
        if hops < 0:
            raise ValueError("pallas-db requires non-negative hops")
        return "pallas-db", f"requested fused Pallas DB kernel inner_cols={inner_cols}"

    # auto mode: use the real custom kernel on TPU when buffering is legal.  On
    # CPU/GPU smoke tests, use XLA so correctness still runs without TPU RDMA.
    if _looks_like_tpu(devices) and buffer_count >= 2:
        return "pallas-db", f"auto selected fused Pallas DB kernel inner_cols={inner_cols}"
    return "xla-psum", "auto fallback: non-TPU or buffer_count < 2"


def xla_fast_token_ring(
    value: Any,
    *,
    mesh: Any,
    axis_name: str,
    direction: str,
    hops: int,
    n_devices: int,
) -> Any:
    """Fast portable reference using XLA collectives.

    For the full-ring all-reduce case this lowers to ``lax.psum``.  For partial
    hop experiments it uses a compact ``lax.ppermute`` chain so the semantics
    still match the teaching source-history tables.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    direction = normalize_direction(direction)
    hops = normalize_hops(hops, n_devices=n_devices)
    n_devices = int(n_devices)
    partition = jax.sharding.PartitionSpec(axis_name, None, None, None)

    if direction == "right":
        perm = [(rank, (rank + 1) % n_devices) for rank in range(n_devices)]
    else:
        perm = [(rank, (rank - 1) % n_devices) for rank in range(n_devices)]

    def _local(local_value):
        if hops == max(0, n_devices - 1):
            return lax.psum(local_value.astype(jnp.float32), axis_name)
        token = local_value
        seen_sum = token.astype(jnp.float32)
        for _ in range(hops):
            token = lax.ppermute(token, axis_name, perm)
            seen_sum = seen_sum + token.astype(jnp.float32)
        return seen_sum

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_vma=False,
    )(value)


def _pallas_local_barrier(pl: Any, pltpu: Any, first_neighbor: Any, second_neighbor: Any) -> None:
    """Synchronize with two ring neighbors at kernel entry.

    The second barrier prevents reuse races when the same collective id is used
    for repeated benchmark iterations.
    """
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor in [first_neighbor, second_neighbor]:
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, 2)

    @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
    def _(second_barrier):
        for neighbor in [first_neighbor, second_neighbor]:
            pl.semaphore_signal(
                second_barrier,
                inc=1,
                device_id=(neighbor,),
                device_id_type=pl.DeviceIdType.MESH,
            )
        pl.semaphore_wait(second_barrier, 2)


def pallas_double_buffered_token_ring(
    value: Any,
    *,
    mesh: Any,
    axis_name: str,
    direction: str,
    hops: int,
    collective_id: int,
    buffer_count: int,
    n_devices: int,
    inner_cols: int = DEFAULT_INNER_PIPELINE_COLS,
) -> Any:
    """Fused custom Pallas ring with real remote-DMA double buffering.

    Shape convention
    ----------------
    ``value`` has global shape ``[device, chunk, row, col]`` and is sharded on
    the leading device axis.  Inside ``shard_map`` each device sees a local shard
    of shape ``[1, chunk, row, col]``.  The Pallas kernel operates on the local
    ``[chunk, row, col]`` tile.

    Pipeline structure
    ------------------
    For step ``s``:

    1. ``working_slot = s % buffer_count`` contains the token to accumulate.
    2. ``receiving_slot = (s + 1) % buffer_count`` is the safe destination for
       the neighbor's next RDMA write.
    3. A capacity semaphore tells the incoming neighbor the receiving slot is
       safe.
    4. The device starts an async remote HBM->HBM copy to the outgoing neighbor.
    5. While that copy is in flight, ``emit_pipeline`` streams the working slot
       through VMEM-sized blocks and accumulates into HBM output.
    6. The device waits for the remote copy before the slot can be reused.

    The remote copy and the local accumulation overlap, so this hides
    communication latency behind useful work.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    direction = normalize_direction(direction)
    hops = normalize_hops(hops, n_devices=n_devices)
    buffer_count = normalize_positive_int(buffer_count, name="buffer_count", default=2)
    if buffer_count < 2:
        raise ValueError("pallas_double_buffered_token_ring requires buffer_count >= 2")
    n_devices = int(n_devices)
    if n_devices < 2:
        raise ValueError("pallas_double_buffered_token_ring requires at least two devices")

    if len(value.shape) != 4:
        raise ValueError(f"expected [device, chunk, row, col], got shape {value.shape}")
    _, n_chunks, rows, cols = map(int, value.shape)
    inner_cols = choose_inner_pipeline_cols(cols=cols, requested=inner_cols, rows=rows)
    local_tile_shape = (n_chunks, rows, cols)
    steps = hops + 1
    send_delta = 1 if direction == "right" else -1
    recv_delta = -send_delta

    inner_grid = (n_chunks, cols // inner_cols)
    # emit_pipeline windows the HBM refs and stages each block through VMEM
    # itself, so the inner BlockSpec must NOT pin memory_space=VMEM: doing so
    # tells Pallas the block already lives in VMEM and skips the staging copy,
    # leaving the body with a raw HBM (ANY) ref that Mosaic refuses to load.
    inner_block_spec = pl.BlockSpec(
        index_map=lambda chunk_i, col_i: (chunk_i, 0, col_i),
        block_shape=(1, rows, inner_cols),
    )

    def _ring_kernel(
        x_ref,
        o_ref,
        hbm_scratch,
        remote_recv_sem,
        remote_send_sem,
        capacity_sem,
    ):
        outer_step = pl.program_id(0)
        last_iteration = outer_step == pl.num_programs(0) - 1
        working_slot = lax.rem(outer_step, buffer_count)
        receiving_slot = lax.rem(outer_step + 1, buffer_count)

        my_id = lax.axis_index(axis_name)
        send_to_neighbor = lax.rem(my_id + send_delta + n_devices, n_devices)
        recv_from_neighbor = lax.rem(my_id + recv_delta + n_devices, n_devices)

        def _init_inner(src_ref, accum_ref):
            # Initialize the float32 accumulator from the first token (with cast).
            # This has an input spec, so emit_pipeline stages both refs through
            # VMEM; an output-only "zero" pipeline would try to load the HBM
            # accumulator directly, which Mosaic forbids.
            accum_ref[...] = src_ref[...].astype(jnp.float32)

        init_pipeline = pltpu.emit_pipeline(
            _init_inner,
            in_specs=[inner_block_spec],
            out_specs=inner_block_spec,
            grid=inner_grid,
        )

        def _copy_inner(src_ref, dst_ref):
            dst_ref[...] = src_ref[...]

        copy_pipeline = pltpu.emit_pipeline(
            _copy_inner,
            in_specs=[inner_block_spec],
            out_specs=inner_block_spec,
            grid=inner_grid,
        )

        def _accum_inner(src_ref, accum_ref):
            # With should_accumulate_out=True the pipeline performs the
            # read-modify-write into the HBM accumulator itself (o_ref += body).
            # The body therefore writes only this step's contribution. Doing the
            # "+= accum_ref[...]" by hand instead reads a write-only output block
            # that emit_pipeline never loads from HBM, which silently corrupts the
            # sum whenever the inner grid has more than one block.
            accum_ref[...] = src_ref[...].astype(jnp.float32)

        accum_pipeline = pltpu.emit_pipeline(
            _accum_inner,
            in_specs=[inner_block_spec],
            out_specs=inner_block_spec,
            grid=inner_grid,
            should_accumulate_out=True,
        )

        @pl.when(outer_step == 0)
        def _():
            _pallas_local_barrier(pl, pltpu, send_to_neighbor, recv_from_neighbor)
            copy_pipeline(x_ref, hbm_scratch.at[0])

        remote_copy = pltpu.make_async_remote_copy(
            src_ref=hbm_scratch.at[working_slot],
            dst_ref=hbm_scratch.at[receiving_slot],
            send_sem=remote_send_sem,
            recv_sem=remote_recv_sem,
            device_id=(send_to_neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )

        @pl.when(~last_iteration)
        def _():
            # Tell the neighbor that sends into this device that our receiving
            # slot for the next token is free.  Then wait until our outgoing
            # neighbor says the same thing to us before issuing the RDMA.
            pl.semaphore_signal(
                capacity_sem,
                inc=1,
                device_id=(recv_from_neighbor,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            pl.semaphore_wait(capacity_sem, 1)
            remote_copy.start()

        # This is the overlap: local accumulation streams through VMEM while the
        # HBM->HBM remote copy is in flight. Step 0 initializes the accumulator
        # from the first token; later steps add each arriving token.
        @pl.when(outer_step == 0)
        def _():
            init_pipeline(hbm_scratch.at[working_slot], o_ref)

        @pl.when(outer_step != 0)
        def _():
            accum_pipeline(hbm_scratch.at[working_slot], o_ref)

        @pl.when(~last_iteration)
        def _():
            remote_copy.wait()

    out_shape = (
        jax.ShapeDtypeStruct(local_tile_shape, jnp.float32),
        jax.ShapeDtypeStruct((buffer_count, *local_tile_shape), value.dtype),
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=[pl.BlockSpec(memory_space=pl.ANY), pl.BlockSpec(memory_space=pl.ANY)],
        grid=(steps,),
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,      # remote_recv_sem
            pltpu.SemaphoreType.DMA,      # remote_send_sem
            pltpu.SemaphoreType.REGULAR,  # capacity_sem
        ),
    )
    kernel = pl.pallas_call(
        _ring_kernel,
        out_shape=out_shape,
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=int(collective_id)),
    )

    partition = jax.sharding.PartitionSpec(axis_name, None, None, None)

    def _per_device(local_value):
        local_tile = local_value[0, :, :, :]
        output_tile, _scratch = kernel(local_tile)
        return output_tile[jnp.newaxis, :, :, :]

    return jax.shard_map(
        _per_device,
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_vma=False,
    )(value)

def chunked_token_ring(
    value: Any,
    *,
    mesh: Any,
    axis_name: str,
    direction: str,
    hops: int,
    collective_id: int,
    memory_space: Any,
    n_chunks: int,
) -> Any:
    """Run a serialized token ring independently for each chunk.

    ``value`` has global shape ``[device, chunk, row, col]``. For each chunk we
    take a ``[device, row, col]`` token and pass it around the logical ring using
    the Lab 1 single-hop remote-copy helper.

    Why the loop is outside the Pallas kernel
    -----------------------------------------

    This is the clarity-first version. Every chunk/hop pair remains a visible
    communication phase with its own collective ID. A later optimization can move
    this schedule inside one fused Pallas kernel and use buffer slots for actual
    double buffering.
    """
    import jax.numpy as jnp

    direction = normalize_direction(direction)
    hops = normalize_hops(hops)
    n_chunks = normalize_positive_int(n_chunks, name="n_chunks", default=1)

    outputs = []
    id_stride = max(1, hops)
    for chunk_idx in range(n_chunks):
        # Select one chunk from the chunk axis. The selected token is shaped like
        # the earlier Lab 2 token-ring tile: [device, row, col].
        token = value[:, chunk_idx, :, :]
        seen_sum = token.astype(jnp.float32)

        for hop in range(hops):
            # Each call is one Lab 1-style Pallas remote-DMA communication phase.
            # The unique collective ID prevents semaphore state from aliasing
            # across different chunks or hops.
            token = lab1_single_hop.neighbor_copy(
                token,
                mesh=mesh,
                axis_name=axis_name,
                direction=direction,
                collective_id=collective_id + chunk_idx * id_stride + hop,
                memory_space=memory_space,
            )
            seen_sum = seen_sum + token.astype(jnp.float32)

        outputs.append(seen_sum)

    # Restore the chunk axis so the final result is [device, chunk, row, col].
    return jnp.stack(outputs, axis=1)


def _resolve_memory_space(memory_space_name: str) -> Any:
    """Resolve a Pallas TPU memory-space name with a friendly error."""
    helper = getattr(lab1_single_hop, "resolve_memory_space", None)
    if helper is not None:
        return helper(memory_space_name)

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    normalized = str(memory_space_name).strip().upper()
    if normalized == "HBM":
        # HBM must map to the non-windowed MemorySpace.ANY for the remote-DMA
        # hop; the explicit MemorySpace.HBM makes Mosaic reject the operand.
        # Mirrors lab1_single_hop.resolve_memory_space.
        return pl.ANY
    if normalized == "VMEM":
        return pltpu.VMEM
    if normalized == "ANY":
        return pl.ANY
    raise ValueError(
        f"unknown Pallas TPU memory space {memory_space_name!r}; expected "
        "a name such as 'HBM', 'VMEM', or 'ANY'"
    )


def build_case(
    *,
    jax: Any,
    jnp: Any,
    devices: list[Any],
    axis_name: str,
    payload_bytes: int,
    dtype: Any,
    direction: str,
    hops: int,
    n_chunks: int,
    buffer_count: int,
    tile_rows: int,
    min_cols: int,
    memory_space_name: str,
    collective_id: int,
    kernel_mode: str | None = None,
    inner_pipeline_cols: int = DEFAULT_INNER_PIPELINE_COLS,
) -> ChunkedTokenRingCase:
    """Build input, expected output, and jitted function for Lab 8."""
    import numpy as np

    n_devices = len(devices)
    if n_devices < 2:
        raise ValueError("Lab 8 requires at least two devices to form a ring")

    direction = normalize_direction(direction)
    n_chunks = normalize_positive_int(n_chunks, name="n_chunks", default=4)
    buffer_count = normalize_positive_int(buffer_count, name="buffer_count", default=2)
    hops = normalize_hops(hops, n_devices=n_devices)

    itemsize = int(jnp.dtype(dtype).itemsize)
    rows, cols, chunk_payload_bytes, actual_payload_bytes = chunk_tile_shape_for_payload(
        payload_bytes=payload_bytes,
        itemsize=itemsize,
        tile_rows=tile_rows,
        min_cols=min_cols,
        n_chunks=n_chunks,
    )

    # The implemented remote copy moves one chunk per Lab 1 hop, so VMEM guardrails
    # apply to the per-chunk tile rather than to the entire full payload.
    lab1_single_hop.validate_whole_tile_memory_space(
        memory_space_name=memory_space_name,
        actual_payload_bytes=chunk_payload_bytes,
    )

    mesh = jax.sharding.Mesh(np.array(devices), (axis_name,))
    sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(axis_name),
    )

    x_host = make_chunked_rank_input(
        jnp,
        n_devices=n_devices,
        n_chunks=n_chunks,
        rows=rows,
        cols=cols,
        dtype=dtype,
    )
    x = jax.device_put(x_host, sharding)
    memory_space = _resolve_memory_space(memory_space_name)

    selected_kernel_mode, fast_path_reason = choose_lab8_kernel_mode(
        requested_mode=kernel_mode,
        devices=devices,
        buffer_count=buffer_count,
        hops=hops,
        n_devices=n_devices,
        tile_cols=cols,
        inner_cols=inner_pipeline_cols,
    )
    selected_inner_cols = choose_inner_pipeline_cols(
        cols=cols,
        requested=inner_pipeline_cols,
        rows=rows,
    )

    def ring_fn(value):
        if selected_kernel_mode == "serialized":
            return chunked_token_ring(
                value,
                mesh=mesh,
                axis_name=axis_name,
                direction=direction,
                hops=hops,
                collective_id=collective_id,
                memory_space=memory_space,
                n_chunks=n_chunks,
            )
        if selected_kernel_mode == "pallas-db":
            return pallas_double_buffered_token_ring(
                value,
                mesh=mesh,
                axis_name=axis_name,
                direction=direction,
                hops=hops,
                collective_id=collective_id,
                buffer_count=buffer_count,
                n_devices=n_devices,
                inner_cols=selected_inner_cols,
            )
        return xla_fast_token_ring(
            value,
            mesh=mesh,
            axis_name=axis_name,
            direction=direction,
            hops=hops,
            n_devices=n_devices,
        )

    return ChunkedTokenRingCase(
        x=x,
        fn=jax.jit(ring_fn),
        expected_sums=expected_chunk_tiles(
            jnp,
            n_devices=n_devices,
            n_chunks=n_chunks,
            rows=rows,
            cols=cols,
            dtype=dtype,
            hops=hops,
            direction=direction,
        ),
        actual_payload_bytes=actual_payload_bytes,
        chunk_payload_bytes=chunk_payload_bytes,
        tile_rows=rows,
        tile_cols=cols,
        n_chunks=n_chunks,
        buffer_count=buffer_count,
        direction=direction,
        hops=hops,
        n_devices=n_devices,
        kernel_mode=selected_kernel_mode,
        fast_path_reason=fast_path_reason,
    )


def observed_chunk_scalars(jax: Any, y: Any) -> Any:
    """Return ``y[:, :, 0, 0]`` on the host for quick schedule debugging."""
    return jax.device_get(y)[:, :, 0, 0]


def check_result(jax: Any, jnp: Any, y: Any, expected: Any) -> bool:
    """Validate the full chunked token-ring output.

    Inspecting only ``y[:, :, 0, 0]`` gives a quick scalar table, but Lab 8 moves
    whole chunks, so this checker verifies that every element of every output
    chunk equals the expected per-chunk sum.
    """
    del jnp  # Host-side NumPy makes this a pure correctness check.

    import numpy as np

    y_host = np.asarray(jax.device_get(y))
    expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)

    if y_host.ndim != 4:
        return False

    if expected_host.ndim == 2:
        if y_host.shape[:2] != expected_host.shape:
            return False
        expected_tiles = expected_host[:, :, None, None]
    elif expected_host.ndim == 4:
        if y_host.shape != expected_host.shape:
            return False
        expected_tiles = expected_host
    else:
        return False

    y_f32 = y_host.astype(np.float32)
    return bool(
        np.allclose(
            y_f32,
            expected_tiles + np.zeros_like(y_f32),
            rtol=1e-3,
            atol=1e-3,
        )
    )


def _fallback_build_spec(
    template: dict[str, Any],
    *,
    args: Any,
    payload_bytes: int,
    n_devices: int,
) -> dict[str, Any]:
    """Small local fallback for ``lab_spec_utils.build_spec``.

    The benchmark repository's real utility should be used during normal runs.
    This fallback only keeps the file convenient to import and smoke-test on its
    own.
    """
    return {
        **template,
        "payload_bytes": int(payload_bytes),
        "n_devices": int(n_devices),
        "args": {
            "neighbor_direction": getattr(args, "neighbor_direction", None),
            "token_hops": getattr(args, "token_hops", None),
            "ring_hops": getattr(args, "ring_hops", None),
            "lab8_chunks": getattr(args, "lab8_chunks", None),
            "lab8_buffer_count": getattr(args, "lab8_buffer_count", None),
            "pallas_collective_id": getattr(args, "pallas_collective_id", None),
            "pallas_memory_space": getattr(args, "pallas_memory_space", None),
        },
    }


def _fallback_render_markdown(spec: dict[str, Any]) -> str:
    """Small local fallback for ``lab_spec_utils.render_markdown``."""
    lines = [f"# {spec.get('title', 'Lab 8 Spec')}", ""]
    for key in sorted(spec):
        if key == "title":
            continue
        lines.append(f"## {key}")
        value = spec[key]
        if isinstance(value, (dict, list, tuple)):
            import json

            lines.extend(["", "```json", json.dumps(value, indent=2, sort_keys=True), "```", ""])
        else:
            lines.extend(["", str(value), ""])
    return "\n".join(lines).rstrip() + "\n"


def _extract_lab8_chunks(args: Any, *, default: int = 4) -> int:
    """Return the configured chunk count from CLI args."""
    return normalize_positive_int(
        getattr(args, "lab8_chunks", None),
        name="lab8_chunks",
        default=default,
    )


def _extract_lab8_buffer_count(args: Any, *, default: int = 2) -> int:
    """Return the configured buffer count from CLI args."""
    return normalize_positive_int(
        getattr(args, "lab8_buffer_count", None),
        name="lab8_buffer_count",
        default=default,
    )


def _extract_hops(args: Any, *, n_devices: int) -> int:
    """Return Lab 8 hop count from common course CLI names."""
    raw = (
        getattr(args, "ring_hops", None)
        if getattr(args, "ring_hops", None) is not None
        else getattr(args, "token_hops", None)
    )
    return normalize_hops(raw, n_devices=n_devices)


def build_spec(*, jax: Any, args: Any, payload_bytes: int, n_devices: int) -> dict[str, Any]:
    """Build the Lab 8 course artifact consumed by the benchmark harness."""
    del jax

    if lab_spec_utils is not None:
        spec = lab_spec_utils.build_spec(
            LAB_SPEC,
            args=args,
            payload_bytes=payload_bytes,
            n_devices=n_devices,
        )
    else:  # pragma: no cover - standalone fallback.
        spec = _fallback_build_spec(
            LAB_SPEC,
            args=args,
            payload_bytes=payload_bytes,
            n_devices=n_devices,
        )

    direction = normalize_direction(getattr(args, "neighbor_direction", None) or "right")
    n_chunks = _extract_lab8_chunks(args)
    buffer_count = _extract_lab8_buffer_count(args)
    hops = _extract_hops(args, n_devices=n_devices)
    requested_chunk_payload = ceil_div(max(1, int(payload_bytes)), n_chunks)
    base_collective_id = int(getattr(args, "pallas_collective_id", 0) or 0)

    requested_kernel_mode = normalize_kernel_mode(
        getattr(args, "lab8_kernel", None)
        or getattr(args, "lab8_mode", None)
        or getattr(args, "lab8_kernel_mode", None)
        or "auto"
    )
    # 0 / None means "auto" (choose_inner_pipeline_cols sizes the block), so this
    # is a plain int read rather than a positive-only check.
    _raw_inner_cols = getattr(args, "lab8_inner_cols", None)
    try:
        configured_inner_cols = int(_raw_inner_cols)
    except (TypeError, ValueError):
        configured_inner_cols = DEFAULT_INNER_PIPELINE_COLS
    if configured_inner_cols < 0:
        configured_inner_cols = 0

    spec["configured_kernel_mode"] = requested_kernel_mode
    spec["configured_inner_pipeline_cols"] = configured_inner_cols
    spec["configured_chunks"] = n_chunks
    spec["configured_buffer_count"] = buffer_count
    spec["requested_chunk_payload_bytes"] = requested_chunk_payload
    spec["ring_hops"] = hops
    spec["neighbor_direction"] = direction
    spec["suggested_chunk_bytes"] = list(SUGGESTED_CHUNK_BYTES)
    spec["minimum_buffer_count_for_double_buffering"] = 2
    spec["serialized_remote_copy_phases"] = n_chunks * hops
    spec["source_history"] = {
        "right": "device r sees [r, r-1, r-2, ...] mod N",
        "left": "device r sees [r, r+1, r+2, ...] mod N",
        "active": source_history_table(
            n_devices=n_devices,
            hops=hops,
            direction=direction,
        ),
    }
    spec["expected_scalar_chunk_sums_for_unit_tiles"] = expected_chunk_sum_values(
        n_devices=n_devices,
        n_chunks=n_chunks,
        hops=hops,
        direction=direction,
    )
    spec["byte_model"] = chunked_ring_byte_model(
        n_devices=n_devices,
        full_payload_bytes=int(payload_bytes),
        chunk_payload_bytes=requested_chunk_payload,
        n_chunks=n_chunks,
        hops=hops,
        direction=direction,
        buffer_count=buffer_count,
        kernel_mode=requested_kernel_mode,
    )
    spec["buffer_slot_schedule"] = buffer_slot_schedule(
        n_chunks=n_chunks,
        buffer_count=buffer_count,
        max_chunks=16,
    )
    spec["planned_wavefront_events"] = planned_wavefront_events(
        n_chunks=n_chunks,
        hops=hops,
        buffer_count=buffer_count,
        max_events=64,
    )
    spec["collective_id_schedule"] = collective_id_schedule(
        base_collective_id=base_collective_id,
        n_chunks=n_chunks,
        hops=hops,
        max_entries=64,
    )
    spec["serialized_phase_schedule_preview"] = serialized_phase_schedule(
        base_collective_id=base_collective_id,
        n_chunks=n_chunks,
        hops=hops,
        buffer_count=buffer_count,
        max_entries=64,
    )
    spec["planned_overlapped_wavefront_preview"] = planned_overlapped_wavefront(
        n_chunks=n_chunks,
        hops=hops,
        buffer_count=buffer_count,
        max_entries=64,
    )
    spec["pipeline_planning_notes"] = pipeline_planning_notes(
        buffer_count=buffer_count,
    )
    spec["custom_collective_status"] = (
        "serialized mode is the clarity-first teaching path; pallas-db mode is a "
        "fused custom Pallas kernel with real async remote-DMA double/N-buffering; "
        "xla-psum is an XLA reference that moves fewer bytes, not a target to match"
    )
    spec["student_checkpoint_questions"] = [
        "Does chunking change total bytes or only scheduling granularity?",
        "How many communication phases does chunks * hops create?",
        "Which collective IDs are used by chunk 0 and chunk 1?",
        "When is buffer slot 0 safe to reuse?",
        "What evidence in XProf would prove overlap?",
        "Does pallas-db start the remote DMA before the inner accumulation pipeline?",
        "How much faster is pallas-db than serialized at the same payload and chunk count?",
    ]
    return spec


def render_markdown(spec: dict[str, Any]) -> str:
    """Render a spec artifact as Markdown."""
    if lab_spec_utils is not None:
        return lab_spec_utils.render_markdown(spec)
    return _fallback_render_markdown(spec)
