"""Lab 8: chunked ring movement and pipeline planning.

This file contains the concept code for the Lab 8 teaching module. The benchmark
harness owns run directories, logging, CSV/JSONL output, plots, and profiler
capture. Keeping those pieces out of the lab file lets students read one idea at
a time.

What this lab implements
========================

The implemented custom path is **chunked but serialized**:

    for each chunk:
        run the Lab 1 one-hop remote-DMA primitive for H hops
        accumulate the token values seen by this device

That is deliberately not a fully fused, double-buffered Pallas pipeline yet.
Chunking is the act of splitting a payload. Pipelining is the act of overlapping
stages so useful work continues while copies are in flight. Lab 8 teaches the
scheduling vocabulary and records the buffer-slot plan before claiming overlap.

What this lab does not claim
============================

Changing ``buffer_count`` in this file does not magically create overlap. The
current Pallas path calls the Lab 1 remote-copy helper repeatedly. The buffer
count is recorded in the lab spec so the next fused kernel can implement the
right slot ownership discipline.

This distinction is important enough to repeat: a trace must prove overlap.
Prose cannot.
"""

from __future__ import annotations

import dataclasses
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

    @property
    def note(self) -> str:
        """Short human-readable note for benchmark tables."""
        return (
            f"chunked-serialized direction={self.direction} "
            f"chunks={self.n_chunks} buffers={self.buffer_count} "
            f"hops={self.hops} devices={self.n_devices} "
            f"chunk={self.chunk_payload_bytes}B "
            f"tile={self.tile_rows}x{self.tile_cols} "
            f"phases={self.serialized_remote_copy_phases}"
        )

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
        "Turn a correct ring into a performance experiment with chunk size, "
        "serialized remote-copy phases, buffer-slot ownership, and profile-based "
        "claims about overlap. The implemented custom path is chunked but "
        "serialized; the spec records the pipeline schedule a fused kernel should "
        "implement next."
    ),
    "implemented_ops": [
        "`pmap_token_ring`: dependency-chain reference from Lab 2",
        "`pallas_chunked_token_ring`: serialized chunked ring built from Lab 1 hops",
        "`lab8_chunked_pipeline_spec`: source-history, byte-model, collective-ID, and buffer-slot artifact",
    ],
    "deferred_ops": [
        "Fuse the chunk schedule into one Pallas kernel",
        "Use buffer slots for real double buffering rather than spec-only planning",
        "Record run-ahead hazards as explicit correctness checks",
        "Measure overlap by separating enqueue, wait, and local-compute profile regions",
        "Carry the chunked schedule into optimized reduce-scatter and all-reduce",
    ],
    "byte_model": [
        "Chunking does not reduce total bytes: serialized send bytes are `hops * full_payload_bytes` per device",
        "Each per-chunk remote-copy phase sends `chunk_payload_bytes` per device",
        "Serialized chunking increases phase count to `chunks * hops` before true overlap is implemented",
        "True pipelining should change overlap and latency hiding, not required payload bytes",
    ],
    "pass_condition": [
        "reference token-ring correctness remains stable across payload sizes",
        "serialized Pallas chunked ring matches per-chunk token sums on TPU",
        "full output tiles are checked, not only scalar rank markers",
        "spec artifact defines source history, byte model, collective IDs, and buffer ownership rules",
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
        "Carry chunked schedules into topology-aware Lab 9",
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

    ``expected_chunk_sum_values`` is kept for compatibility with the compact
    original lab. This alias reads better in notebooks and worksheets.
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

    The scalar at ``[row=0, col=0]`` keeps the original hand-checkable pattern:

        x[source, chunk, 0, 0] = 10 * source + chunk

    The rest of the tile includes tiny row/column patterns. That makes the full
    correctness check meaningful: a partial-copy or stale-slot bug can no longer
    hide behind a correct rank marker at ``[0, 0]``.
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
) -> dict[str, int | float | bool | str]:
    """Return byte and phase accounting for Lab 8.

    These numbers are algorithmic accounting for the implemented teaching path.
    They are not a claim about how a built-in XLA collective is implemented.
    """
    direction = normalize_direction(direction)
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
        "implemented_overlap": False,
        "note": (
            "Chunking changes phase granularity. The implemented path remains "
            "serialized; overlap must be proven by a later fused kernel trace."
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

    def ring_fn(value):
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
    )


def observed_chunk_scalars(jax: Any, y: Any) -> Any:
    """Return ``y[:, :, 0, 0]`` on the host for quick schedule debugging."""
    return jax.device_get(y)[:, :, 0, 0]


def check_result(jax: Any, jnp: Any, y: Any, expected: Any) -> bool:
    """Validate the full chunked token-ring output.

    The compact original check inspected only ``y[:, :, 0, 0]``. That is useful
    for a quick scalar table, but Lab 8 moves whole chunks. This checker verifies
    that every element of every output chunk equals the expected per-chunk sum.
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
    """Small local replacement for ``lab_spec_utils.build_spec``.

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
    """Small local replacement for ``lab_spec_utils.render_markdown``."""
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
        "implemented as serialized per-chunk Lab 1 hop kernels; true "
        "double-buffered overlap remains deferred to a fused Pallas kernel"
    )
    spec["student_checkpoint_questions"] = [
        "Does chunking change total bytes or only scheduling granularity?",
        "How many communication phases does chunks * hops create?",
        "Which collective IDs are used by chunk 0 and chunk 1?",
        "When is buffer slot 0 safe to reuse?",
        "What evidence in XProf would prove overlap?",
    ]
    return spec


def render_markdown(spec: dict[str, Any]) -> str:
    """Render a spec artifact as Markdown."""
    if lab_spec_utils is not None:
        return lab_spec_utils.render_markdown(spec)
    return _fallback_render_markdown(spec)
