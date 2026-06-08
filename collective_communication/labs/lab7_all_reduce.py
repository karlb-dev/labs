"""Lab 7: all-reduce from reduce-scatter plus all-gather.

This file contains the *concept code* for Lab 7. The benchmark harness owns run
folders, CSV/JSONL rows, plotting, profile capture, and CLI parsing. This file
owns one idea:

    all-reduce = reduce-scatter + all-gather + final layout contract

The custom path composes two earlier teaching implementations:

  * Lab 6: whole-token ring reduce-scatter
  * Lab 5: arrival-order ring all-gather

Then Lab 7 adds a canonical chunk-order restore so every device sees the same
final tensor layout. That final reorder is not a network phase. It is the
answer to a subtle but important question:

    After the data moves, what layout did the algorithm promise to return?

Implementation status
---------------------
This is intentionally a *composed teaching implementation*, not the final
high-performance all-reduce. In particular, the reduce-scatter phase reuses Lab
6's whole-token implementation, so it sends more bytes than an optimized
one-chunk-per-hop ring. Lab 8 can replace this with chunked and pipelined
movement once students can explain the phase boundary without fog machines.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

try:  # Normal path inside the repository package.
    from labs import lab5_ring_all_gather
    from labs import lab6_reduce_scatter
except Exception:  # pragma: no cover - convenience for standalone file review.
    # Direct importlib loading does not always put the file's directory on
    # sys.path. Add it so adjacent lab files can be imported during standalone
    # teaching/review sessions.
    import sys
    from pathlib import Path

    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    import lab5_ring_all_gather  # type: ignore[no-redef]
    import lab6_reduce_scatter  # type: ignore[no-redef]

try:  # The benchmark repo provides this for course artifact rendering.
    from labs import lab_spec_utils
except Exception:  # pragma: no cover - fallback keeps this module importable alone.
    try:
        import lab_spec_utils  # type: ignore[no-redef]
    except Exception:
        lab_spec_utils = None  # type: ignore[assignment]


SUPPORTED_DIRECTIONS = frozenset({"right", "left"})


@dataclasses.dataclass(frozen=True)
class RingAllReduceCase:
    """Container returned to the benchmark harness.

    The harness needs a sharded input ``x``, a jitted function ``fn``, and an
    expected result for correctness. The remaining fields make result rows and
    lab artifacts more explanatory.

    ``actual_payload_bytes`` is the full per-device input payload before the
    reduce-scatter phase. ``chunk_payload_bytes`` is the reduced owner chunk
    size after reduce-scatter and the shard size moved by all-gather.
    """

    x: Any
    fn: Callable[[Any], Any]
    expected_full: Any
    actual_payload_bytes: int
    chunk_payload_bytes: int
    tile_rows: int
    tile_cols: int
    n_chunks: int
    direction: str
    reduce_scatter_hops: int
    all_gather_hops: int
    n_devices: int

    @property
    def note(self) -> str:
        """Short human-readable note for benchmark tables."""
        return (
            f"rs=whole-token ag=chunk direction={self.direction} "
            f"rs_hops={self.reduce_scatter_hops} "
            f"ag_hops={self.all_gather_hops} "
            f"devices={self.n_devices} tile={self.tile_rows}x{self.tile_cols} "
            f"teaching_wire/dev={self.teaching_wire_bytes} "
            f"optimal_wire/dev={self.optimal_ring_wire_bytes}"
        )

    @property
    def num_devices(self) -> int:
        """Alias used by some benchmark/reporting code."""
        return self.n_devices

    @property
    def input_bytes_per_device(self) -> int:
        """Bytes each device starts with before all-reduce."""
        return self.actual_payload_bytes

    @property
    def output_bytes_per_device(self) -> int:
        """Bytes each device owns after a full all-reduce."""
        return self.n_chunks * self.chunk_payload_bytes

    @property
    def reduced_chunk_bytes_per_device(self) -> int:
        """Bytes each device owns immediately after reduce-scatter."""
        return self.chunk_payload_bytes

    @property
    def teaching_reduce_scatter_send_bytes(self) -> int:
        """Per-device send bytes for Lab 6's whole-token teaching phase."""
        return self.actual_payload_bytes * self.reduce_scatter_hops

    @property
    def teaching_all_gather_send_bytes(self) -> int:
        """Per-device send bytes for Lab 5's chunk all-gather phase."""
        return self.chunk_payload_bytes * self.all_gather_hops

    @property
    def teaching_wire_bytes(self) -> int:
        """Per-device send bytes for this composed teaching all-reduce."""
        return self.teaching_reduce_scatter_send_bytes + self.teaching_all_gather_send_bytes

    @property
    def teaching_recv_bytes(self) -> int:
        """Per-device receive bytes for this composed teaching all-reduce."""
        # The one-direction ring is symmetric: every send has one matching
        # receive. Lab 4's ledger lives inside this one-line fact.
        return self.teaching_wire_bytes

    @property
    def optimal_reduce_scatter_send_bytes(self) -> int:
        """Per-device send bytes for an optimized one-chunk reduce-scatter."""
        return self.chunk_payload_bytes * self.reduce_scatter_hops

    @property
    def optimal_all_gather_send_bytes(self) -> int:
        """Per-device send bytes for an optimized one-chunk all-gather."""
        return self.chunk_payload_bytes * self.all_gather_hops

    @property
    def optimal_ring_wire_bytes(self) -> int:
        """Per-device send bytes for the planned one-chunk-per-hop ring."""
        return self.optimal_reduce_scatter_send_bytes + self.optimal_all_gather_send_bytes

    @property
    def optimal_ring_recv_bytes(self) -> int:
        """Per-device receive bytes for the planned optimized ring."""
        return self.optimal_ring_wire_bytes

    @property
    def full_phase_hops(self) -> int:
        """Number of hops needed for each full ring phase."""
        return max(0, self.n_devices - 1)

    @property
    def is_full_all_reduce(self) -> bool:
        """Whether both phases are complete N-device phases."""
        return (
            self.reduce_scatter_hops == self.full_phase_hops
            and self.all_gather_hops == self.full_phase_hops
        )

    @property
    def teaching_overhead_factor(self) -> float:
        """How much more the teaching path sends than the optimized ring.

        For a full N-device run with N chunks, the Lab 6 whole-token phase makes
        this composed path intentionally heavier. Returning a float keeps the
        property safe for zero-byte smoke tests.
        """
        if self.optimal_ring_wire_bytes == 0:
            return 0.0
        return self.teaching_wire_bytes / self.optimal_ring_wire_bytes

    def byte_model(self) -> dict[str, int | float | bool | str]:
        """Return benchmark-friendly byte accounting for this case."""
        return all_reduce_byte_model(
            n_devices=self.n_devices,
            full_payload_bytes=self.actual_payload_bytes,
            chunk_payload_bytes=self.chunk_payload_bytes,
            reduce_scatter_hops=self.reduce_scatter_hops,
            all_gather_hops=self.all_gather_hops,
            direction=self.direction,
        )


def normalize_direction(direction: str) -> str:
    """Validate and normalize the ring direction.

    Direction is from the sender's point of view, matching Labs 1, 5, and 6:
      * ``right`` means device i sends to i + 1 mod N;
      * ``left`` means device i sends to i - 1 mod N.
    """
    # Reuse an earlier lab helper when present, but keep this file standalone.
    for module in (lab5_ring_all_gather, lab6_reduce_scatter):
        helper = getattr(module, "normalize_direction", None)
        if helper is not None:
            return helper(direction)

    normalized = str(direction).strip().lower()
    if normalized not in SUPPORTED_DIRECTIONS:
        allowed = ", ".join(sorted(SUPPORTED_DIRECTIONS))
        raise ValueError(f"unknown ring direction {direction!r}; use one of: {allowed}")
    return normalized


def normalize_hops(hops: int | str | None, n_devices: int | None = None) -> int:
    """Return a non-negative hop count.

    For all-reduce, a complete phase normally uses ``num_devices - 1`` hops.
    Friendly values such as ``None``, ``-1``, ``"full"``, ``"all"``, and
    ``"n-1"`` mean ``num_devices - 1`` when ``n_devices`` is supplied.
    """
    for module in (lab6_reduce_scatter, lab5_ring_all_gather):
        helper = getattr(module, "normalize_hops", None)
        if helper is not None:
            return helper(hops, n_devices=n_devices)

    if n_devices is not None and int(n_devices) < 2:
        raise ValueError("Lab 7 needs at least two devices to form a ring")

    if hops is None:
        if n_devices is None:
            raise ValueError("hops=None requires n_devices")
        return max(0, int(n_devices) - 1)

    if isinstance(hops, str):
        cleaned = hops.strip().lower()
        if cleaned in {"", "default", "full", "all", "ring", "n-1"}:
            if n_devices is None:
                raise ValueError(f"hops={hops!r} requires n_devices")
            return max(0, int(n_devices) - 1)
        value = int(cleaned)
    else:
        try:
            value = int(hops)
        except Exception as exc:  # pragma: no cover - defensive CLI guardrail.
            raise ValueError(f"hops must be an integer, got {hops!r}") from exc

    if value == -1:
        if n_devices is None:
            raise ValueError("hops=-1 requires n_devices")
        return max(0, int(n_devices) - 1)
    if value < 0:
        raise ValueError(f"hops must be non-negative, or -1 for full ring; got {hops!r}")
    return value


def _validate_devices(devices: Sequence[Any]) -> int:
    """Return device count after checking the lab has a real ring."""
    n_devices = len(devices)
    if n_devices < 2:
        raise ValueError("Lab 7 needs at least two TPU devices to form a ring")
    return n_devices


def _axis_size_from_mesh(mesh: Any, axis_name: str) -> int | None:
    """Best-effort extraction of a mesh axis size.

    This runs at trace time on the Python side, so using the Mesh object is fine.
    It keeps ``ring_all_reduce`` able to decide whether the all-gather phase is
    complete enough to perform canonical ordering.
    """
    shape = getattr(mesh, "shape", None)
    if shape is None:
        return None
    try:
        return int(shape[axis_name])
    except Exception:
        try:
            axis_names = list(getattr(mesh, "axis_names"))
            return int(tuple(shape.values())[axis_names.index(axis_name)])
        except Exception:
            return None


def _resolve_memory_space(memory_space_name: str) -> Any:
    """Resolve a Pallas memory-space name, reusing Lab 1 when possible.

    Lab 7 reaches memory-space resolution through Lab 6, but resolving here too
    keeps the composed call robust across compact and expanded earlier lab files.
    """
    # Lab 6's expanded file exposes its own resolver. Use it if available.
    resolver = getattr(lab6_reduce_scatter, "_resolve_memory_space", None)
    if resolver is not None:
        return resolver(memory_space_name)

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
    raise ValueError(f"unknown Pallas memory space {memory_space_name!r}")


def source_ranks_seen(
    *,
    receiver_rank: int,
    n_devices: int,
    hops: int,
    direction: str,
) -> list[int]:
    """Return source or owner ranks that arrive at ``receiver_rank``.

    The same ring table is useful twice in Lab 7:
      * during reduce-scatter it describes which source tokens an owner reduced;
      * during all-gather it describes which reduced owner chunks a receiver saw.
    """
    helper = getattr(lab6_reduce_scatter, "source_ranks_seen", None)
    if helper is not None:
        return helper(
            receiver_rank=receiver_rank,
            n_devices=n_devices,
            hops=hops,
            direction=direction,
        )

    direction = normalize_direction(direction)
    hops = normalize_hops(hops)
    if n_devices <= 0:
        raise ValueError("n_devices must be positive")
    if receiver_rank < 0 or receiver_rank >= n_devices:
        raise ValueError(f"receiver_rank must be in [0, {n_devices}), got {receiver_rank}")

    if direction == "right":
        return [(receiver_rank - hop) % n_devices for hop in range(hops + 1)]
    return [(receiver_rank + hop) % n_devices for hop in range(hops + 1)]


def expected_reduced_values(
    *,
    n_devices: int,
    reduce_scatter_hops: int,
    direction: str,
) -> list[float]:
    """Return reduced chunk values for the Lab 6 teaching input.

    The Lab 6 input pattern is:

        x[source, owner_chunk, row, col] = 10 * source + owner_chunk

    Device ``owner`` accumulates owner chunk ``owner`` from each source token it
    sees during reduce-scatter. With a full run, the result for owner j is:

        10 * N * (N - 1) / 2 + N * j

    This helper also handles partial reduce-scatter hop counts for debugging.
    """
    helper = getattr(lab6_reduce_scatter, "expected_reduced_values", None)
    if helper is not None:
        return helper(
            n_devices=n_devices,
            hops=reduce_scatter_hops,
            direction=direction,
        )

    direction = normalize_direction(direction)
    reduce_scatter_hops = normalize_hops(reduce_scatter_hops)
    values: list[float] = []
    for owner in range(n_devices):
        seen_sources = source_ranks_seen(
            receiver_rank=owner,
            n_devices=n_devices,
            hops=reduce_scatter_hops,
            direction=direction,
        )
        values.append(float(sum(10.0 * src + owner for src in seen_sources)))
    return values


def all_gather_owner_table(
    *,
    n_devices: int,
    all_gather_hops: int,
    direction: str,
) -> list[list[int]]:
    """Return ``[receiver][arrival_slot] -> owner_chunk`` for phase 2."""
    return [
        source_ranks_seen(
            receiver_rank=receiver,
            n_devices=n_devices,
            hops=all_gather_hops,
            direction=direction,
        )
        for receiver in range(n_devices)
    ]


def canonical_arrival_slot_for_owner(
    *,
    receiver_rank: int,
    owner_rank: int,
    n_devices: int,
    direction: str,
) -> int:
    """Return which arrival slot contains ``owner_rank`` on ``receiver_rank``.

    This is a pure-Python teaching mirror of the JAX indexing formula in
    ``canonical_rank_order``. It is only valid for a full all-gather phase,
    where every owner chunk is present on every receiver.
    """
    direction = normalize_direction(direction)
    if direction == "right":
        return (receiver_rank - owner_rank) % n_devices
    return (owner_rank - receiver_rank) % n_devices


def canonical_arrival_slot_table(*, n_devices: int, direction: str) -> list[list[int]]:
    """Return ``[receiver][owner] -> arrival_slot`` for a full phase 2."""
    return [
        [
            canonical_arrival_slot_for_owner(
                receiver_rank=receiver,
                owner_rank=owner,
                n_devices=n_devices,
                direction=direction,
            )
            for owner in range(n_devices)
        ]
        for receiver in range(n_devices)
    ]


def expected_all_reduce_table(
    *,
    n_devices: int,
    reduce_scatter_hops: int,
    all_gather_hops: int,
    direction: str,
) -> list[list[float]]:
    """Return the expected scalar table for Lab 7 output tiles.

    For a full all-gather phase, ``ring_all_reduce`` canonicalizes chunk order,
    so every receiver gets the same row:

        [reduced_chunk_0, reduced_chunk_1, ..., reduced_chunk_N-1]

    For partial all-gather debugging, the function returns the arrival-order
    subset that actually moved. A partial result is not a complete all-reduce,
    but it can be useful when teaching phase 2 one hop at a time.
    """
    direction = normalize_direction(direction)
    reduce_scatter_hops = normalize_hops(reduce_scatter_hops, n_devices=n_devices)
    all_gather_hops = normalize_hops(all_gather_hops, n_devices=n_devices)

    reduced = expected_reduced_values(
        n_devices=n_devices,
        reduce_scatter_hops=reduce_scatter_hops,
        direction=direction,
    )

    if all_gather_hops == n_devices - 1:
        canonical = [float(reduced[owner]) for owner in range(n_devices)]
        return [canonical[:] for _ in range(n_devices)]

    rows: list[list[float]] = []
    for owners_seen in all_gather_owner_table(
        n_devices=n_devices,
        all_gather_hops=all_gather_hops,
        direction=direction,
    ):
        rows.append([float(reduced[owner]) for owner in owners_seen])
    return rows


def expected_full_result(
    jnp: Any,
    n_devices: int,
    reduce_scatter_hops: int | str | None = None,
    all_gather_hops: int | str | None = None,
    direction: str = "right",
) -> Any:
    """JAX-array wrapper for the expected Lab 7 scalar table.

    The compact original helper accepted only ``(jnp, n_devices)`` and returned
    the full all-reduce answer. The expanded version keeps that default while
    also supporting partial-hop teaching experiments.
    """
    reduce_scatter_hops = normalize_hops(reduce_scatter_hops, n_devices=n_devices)
    all_gather_hops = normalize_hops(all_gather_hops, n_devices=n_devices)
    return jnp.array(
        expected_all_reduce_table(
            n_devices=n_devices,
            reduce_scatter_hops=reduce_scatter_hops,
            all_gather_hops=all_gather_hops,
            direction=direction,
        ),
        dtype=jnp.float32,
    )


def phase_collective_ids(
    *,
    base_collective_id: int,
    reduce_scatter_hops: int,
    all_gather_hops: int,
) -> dict[str, list[int]]:
    """Return the collective IDs used by each network phase.

    The canonical reorder is local, so it has no collective IDs.
    """
    base = int(base_collective_id)
    rs_hops = max(0, int(reduce_scatter_hops))
    ag_hops = max(0, int(all_gather_hops))
    return {
        "reduce_scatter": [base + hop for hop in range(rs_hops)],
        "all_gather": [base + rs_hops + hop for hop in range(ag_hops)],
        "canonical_reorder": [],
    }


def all_reduce_byte_model(
    *,
    n_devices: int,
    full_payload_bytes: int,
    chunk_payload_bytes: int,
    reduce_scatter_hops: int,
    all_gather_hops: int,
    direction: str,
) -> dict[str, int | float | bool | str]:
    """Return simple byte accounting for Lab 7.

    The numbers are algorithmic accounting for the teaching ring and the planned
    optimized ring. They are not a claim about how XLA implements ``lax.psum``.
    """
    direction = normalize_direction(direction)
    n_devices = int(n_devices)
    reduce_scatter_hops = normalize_hops(reduce_scatter_hops)
    all_gather_hops = normalize_hops(all_gather_hops)
    full_payload_bytes = int(full_payload_bytes)
    chunk_payload_bytes = int(chunk_payload_bytes)

    if n_devices <= 0:
        raise ValueError("n_devices must be positive")
    if full_payload_bytes < 0:
        raise ValueError("full_payload_bytes must be non-negative")
    if chunk_payload_bytes < 0:
        raise ValueError("chunk_payload_bytes must be non-negative")

    teaching_rs_send = reduce_scatter_hops * full_payload_bytes
    teaching_ag_send = all_gather_hops * chunk_payload_bytes
    teaching_send = teaching_rs_send + teaching_ag_send

    optimal_rs_send = reduce_scatter_hops * chunk_payload_bytes
    optimal_ag_send = all_gather_hops * chunk_payload_bytes
    optimal_send = optimal_rs_send + optimal_ag_send

    full_phase_hops = max(0, n_devices - 1)
    overhead = 0.0 if optimal_send == 0 else teaching_send / optimal_send

    return {
        "direction": direction,
        "n_devices": n_devices,
        "reduce_scatter_hops": reduce_scatter_hops,
        "all_gather_hops": all_gather_hops,
        "full_phase_hops": full_phase_hops,
        "full_input_payload_bytes_per_device": full_payload_bytes,
        "reduced_chunk_bytes_per_device": chunk_payload_bytes,
        "logical_final_result_bytes_per_device": n_devices * chunk_payload_bytes,
        "teaching_reduce_scatter_send_bytes_per_device": teaching_rs_send,
        "teaching_all_gather_send_bytes_per_device": teaching_ag_send,
        "teaching_total_send_bytes_per_device": teaching_send,
        "teaching_total_recv_bytes_per_device": teaching_send,
        "teaching_total_send_bytes_all_devices": n_devices * teaching_send,
        "optimal_reduce_scatter_send_bytes_per_device": optimal_rs_send,
        "optimal_all_gather_send_bytes_per_device": optimal_ag_send,
        "optimal_total_send_bytes_per_device": optimal_send,
        "optimal_total_recv_bytes_per_device": optimal_send,
        "optimal_total_send_bytes_all_devices": n_devices * optimal_send,
        "formula_for_full_optimal_ring_send_bytes_per_device": "2 * (n - 1) / n * full_payload_bytes",
        "teaching_overhead_factor_vs_optimal_ring": overhead,
        "is_full_all_reduce": (
            reduce_scatter_hops == full_phase_hops
            and all_gather_hops == full_phase_hops
        ),
    }


LAB_SPEC: dict[str, Any] = {
    "lab": "lab7",
    "title": "Lab 7: All-Reduce From Reduce-Scatter Plus All-Gather",
    "goal": (
        "Assemble full all-reduce from two ownership transformations: "
        "reduce-scatter followed by all-gather. The custom path composes the "
        "Lab 6 whole-token reduce-scatter with the Lab 5 ring all-gather, then "
        "restores canonical chunk order."
    ),
    "implemented_ops": [
        "`pmap_psum`: built-in all-reduce executable specification",
        "`pallas_ring_all_reduce`: Lab 6 reduce-scatter plus Lab 5 all-gather",
        "`lab7_all_reduce_spec`: course artifact for the optimized two-phase plan",
    ],
    "deferred_ops": [
        "Measure reduce-scatter and all-gather phases separately before timing the combined collective",
        "Compare custom two-phase byte model with built-in `lax.psum`",
        "Replace Lab 6 whole-token reduce-scatter with optimized one-chunk-per-hop movement",
        "Fuse phase transitions and layout normalization into lower-overhead kernels",
        "Add chunking, double buffering, and overlap in Lab 8",
        "Compare one-direction and bidirectional all-reduce variants",
    ],
    "byte_model": [
        "optimal ring all-reduce sends `2 * (n - 1) / n * full_payload_bytes` per device for a full run",
        "Lab 7 teaching version sends Lab 6 whole-token reduce-scatter bytes plus Lab 5 chunk all-gather bytes",
        "phase 1 changes ownership; phase 2 restores full tensor replication",
        "canonical layout restore is local reindexing, not a third network phase",
    ],
    "pass_condition": [
        "built-in `pmap_psum` passes correctness",
        "composed Pallas all-reduce matches canonical reduced chunks on TPU",
        "full tile contents match expected owner values, not only one scalar",
        "spec artifact identifies reduce-scatter and all-gather phase boundaries",
        "spec artifact records collective IDs and byte accounting for the chosen payload",
    ],
    "artifacts": [
        "results.jsonl",
        "csvs/results.csv",
        "lab_artifacts/*lab7_all_reduce_spec*",
        "plots/latency_by_payload.png",
        "plots/bandwidth_by_payload.png",
        "traces/* when profiling is enabled",
    ],
    "suggested_experiments": [
        "Compare `pmap_psum` with `pallas_ring_all_reduce` for small and large payloads",
        "Flip `--neighbor-direction` and verify that canonical output order still matches",
        "Use the byte model to estimate the overhead of the whole-token teaching reduce-scatter",
        "Inspect the trace and identify the reduce-scatter hops, all-gather hops, and reorder",
        "Temporarily inspect arrival order before canonical reorder to debug layout mistakes",
    ],
    "next_steps": [
        "Introduce chunking, double buffering, and overlap in Lab 8",
        "Use the same phase ledger when comparing flat rings and mesh-staged collectives",
    ],
}




def collective_id_plan(
    *,
    base_collective_id: int,
    reduce_scatter_hops: int,
    all_gather_hops: int,
) -> dict[str, int | list[int]]:
    """Return the collective IDs consumed by each remote-DMA phase.

    Lab 6 consumes one collective ID per reduce-scatter hop. Lab 5 consumes one
    additional ID per all-gather hop. The canonical reorder is local and consumes
    no collective ID. Recording the plan in the spec artifact helps students
    catch aliasing mistakes before they become semaphore goblins.
    """
    base = int(base_collective_id)
    rs_hops = normalize_hops(reduce_scatter_hops)
    ag_hops = normalize_hops(all_gather_hops)
    rs_ids = list(range(base, base + rs_hops))
    ag_start = base + rs_hops
    ag_ids = list(range(ag_start, ag_start + ag_hops))
    return {
        "base_collective_id": base,
        "reduce_scatter_ids": rs_ids,
        "all_gather_ids": ag_ids,
        "canonical_reorder_ids": [],
        "next_free_collective_id": ag_start + ag_hops,
    }

def canonical_rank_order(
    value: Any,
    *,
    mesh: Any,
    axis_name: str,
    direction: str,
) -> Any:
    """Convert Lab 5 arrival-order all-gather output to canonical chunk order.

    ``value`` is globally shaped like:

        [receiver_device, arrival_slot, rows, cols]

    For a right-moving ring, receiver r sees owner chunks in arrival order:

        [r, r-1, r-2, ...] mod N

    The canonical all-reduce contract is instead:

        [0, 1, 2, ..., N-1]

    This function performs that local reindexing under ``shard_map``. It assumes
    phase 2 was a full all-gather, so every owner chunk is present on every
    receiver. Partial all-gather debugging results are intentionally left in
    arrival order by ``ring_all_reduce``.
    """
    import jax
    from jax import lax
    from jax._src import shard_map
    import jax.numpy as jnp

    direction = normalize_direction(direction)
    partition = jax.sharding.PartitionSpec

    def local_reorder(x_shard):
        # Expected local shape is [1, n_chunks, rows, cols]. The leading size-1
        # dimension is the local slice of the globally sharded receiver axis.
        receiver_rank = lax.axis_index(axis_name)
        n_chunks = x_shard.shape[1]
        owners = jnp.arange(n_chunks)

        # Which arrival slot contains canonical owner chunk k?
        # Right ring: arrival order is [r, r-1, r-2, ...].
        # Left ring:  arrival order is [r, r+1, r+2, ...].
        if direction == "right":
            arrival_idx = jnp.mod(receiver_rank - owners, n_chunks)
        else:
            arrival_idx = jnp.mod(owners - receiver_rank, n_chunks)

        return jnp.take(x_shard, arrival_idx, axis=1)

    return shard_map.shard_map(
        local_reorder,
        mesh=mesh,
        in_specs=partition(axis_name),
        out_specs=partition(axis_name),
        check_vma=False,
    )(value)


def ring_all_reduce(
    value: Any,
    *,
    mesh: Any,
    axis_name: str,
    direction: str,
    reduce_scatter_hops: int,
    all_gather_hops: int,
    collective_id: int,
    memory_space: Any,
) -> Any:
    """Compose reduce-scatter, all-gather, and canonical layout restore.

    The phase structure is deliberately written as straight-line Python so
    students can point to the ownership boundary:

      1. ``reduced_owned`` has one reduced owner chunk per device.
      2. ``gathered_arrival`` has all reduced chunks, but in ring arrival order.
      3. ``canonical_rank_order`` returns the conventional all-reduce layout.

    A unique collective-ID range is reserved for each network phase. The local
    canonical reorder does not use a collective ID.
    """
    direction = normalize_direction(direction)
    reduce_scatter_hops = normalize_hops(reduce_scatter_hops)
    all_gather_hops = normalize_hops(all_gather_hops)

    # Phase 1: ownership-changing reduction. Shape changes conceptually from
    # [N, N, rows, cols] to [N, rows, cols], where device j owns reduced chunk j.
    reduced_owned = lab6_reduce_scatter.ring_reduce_scatter(
        value,
        mesh=mesh,
        axis_name=axis_name,
        direction=direction,
        hops=reduce_scatter_hops,
        collective_id=int(collective_id),
        memory_space=memory_space,
    )

    # Phase 2: move reduced owner chunks around the ring. The base collective ID
    # starts after the IDs consumed by phase 1.
    gathered_arrival = lab5_ring_all_gather.ring_all_gather(
        reduced_owned,
        mesh=mesh,
        axis_name=axis_name,
        direction=direction,
        hops=all_gather_hops,
        collective_id=int(collective_id) + reduce_scatter_hops,
        memory_space=memory_space,
    )

    # If phase 2 is partial, not every owner chunk is present. Leave the partial
    # debugging result in arrival order. A complete all-reduce performs the
    # canonical layout restore.
    axis_size = _axis_size_from_mesh(mesh, axis_name)
    if axis_size is not None and all_gather_hops != axis_size - 1:
        return gathered_arrival

    return canonical_rank_order(
        gathered_arrival,
        mesh=mesh,
        axis_name=axis_name,
        direction=direction,
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
    reduce_scatter_hops: int | str | None,
    all_gather_hops: int | str | None,
    tile_rows: int,
    min_cols: int,
    memory_space_name: str,
    collective_id: int,
) -> RingAllReduceCase:
    """Build input, expected output, and jitted function for Lab 7.

    ``payload_bytes`` means the full input payload per device before
    reduce-scatter. Lab 6 splits that payload into ``num_devices`` owner chunks,
    so the reduced chunk moved by all-gather is roughly ``payload_bytes / N``
    after shape rounding.
    """
    import numpy as np

    direction = normalize_direction(direction)
    n_devices = _validate_devices(devices)
    reduce_scatter_hops = normalize_hops(reduce_scatter_hops, n_devices=n_devices)
    all_gather_hops = normalize_hops(all_gather_hops, n_devices=n_devices)

    base = lab6_reduce_scatter.build_case(
        jax=jax,
        jnp=jnp,
        devices=devices,
        axis_name=axis_name,
        payload_bytes=payload_bytes,
        dtype=dtype,
        direction=direction,
        hops=reduce_scatter_hops,
        tile_rows=tile_rows,
        min_cols=min_cols,
        memory_space_name=memory_space_name,
        collective_id=collective_id,
    )

    mesh = jax.sharding.Mesh(np.array(devices), (axis_name,))
    memory_space = _resolve_memory_space(memory_space_name)

    def all_reduce_fn(value):
        return ring_all_reduce(
            value,
            mesh=mesh,
            axis_name=axis_name,
            direction=direction,
            reduce_scatter_hops=reduce_scatter_hops,
            all_gather_hops=all_gather_hops,
            collective_id=collective_id,
            memory_space=memory_space,
        )

    return RingAllReduceCase(
        x=base.x,
        fn=jax.jit(all_reduce_fn),
        expected_full=expected_full_result(
            jnp,
            n_devices,
            reduce_scatter_hops=reduce_scatter_hops,
            all_gather_hops=all_gather_hops,
            direction=direction,
        ),
        actual_payload_bytes=int(base.actual_payload_bytes),
        chunk_payload_bytes=int(base.chunk_payload_bytes),
        tile_rows=int(base.tile_rows),
        tile_cols=int(base.tile_cols),
        n_chunks=int(base.n_chunks),
        direction=direction,
        reduce_scatter_hops=int(reduce_scatter_hops),
        all_gather_hops=int(all_gather_hops),
        n_devices=int(n_devices),
    )


def observed_chunk_scalars(jax: Any, y: Any) -> Any:
    """Return ``y[:, :, 0, 0]`` on the host for quick table debugging."""
    return jax.device_get(y)[:, :, 0, 0]


def check_result(jax: Any, jnp: Any, y: Any, expected: Any) -> bool:
    """Validate the full all-reduce output tiles.

    Older minimal checks looked only at ``y[:, :, 0, 0]``. That scalar table is
    useful for a quick glance, but all-reduce returns a whole tensor. This
    checker verifies that every element of every output tile equals the expected
    reduced value for that owner chunk.
    """
    del jnp  # Host-side NumPy makes this a pure correctness check, not more TPU work.

    import numpy as np

    y_host = np.asarray(jax.device_get(y))
    expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)

    if y_host.ndim != 4:
        return False
    if expected_host.ndim != 2:
        return False
    if y_host.shape[:2] != expected_host.shape:
        return False

    y_f32 = y_host.astype(np.float32)
    expected_tiles = expected_host[:, :, None, None]
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

    This fallback is only for standalone review of this file. The benchmark
    repository's real utility should be used during normal runs.
    """
    spec = dict(template)
    spec["payload_bytes"] = int(payload_bytes)
    spec["n_devices"] = int(n_devices)
    spec["args"] = {
        "neighbor_direction": getattr(args, "neighbor_direction", None),
        "token_hops": getattr(args, "token_hops", None),
        "ring_hops": getattr(args, "ring_hops", None),
        "reduce_scatter_hops": getattr(args, "reduce_scatter_hops", None),
        "all_gather_hops": getattr(args, "all_gather_hops", None),
        "pallas_collective_id": getattr(args, "pallas_collective_id", None),
        "pallas_memory_space": getattr(args, "pallas_memory_space", None),
    }
    return spec


def _first_present_attr(args: Any, names: tuple[str, ...]) -> Any:
    """Return the first non-None CLI attribute from ``args``."""
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            return value
    return None


def build_spec(*, jax: Any, args: Any, payload_bytes: int, n_devices: int) -> dict[str, Any]:
    """Build the Lab 7 course artifact consumed by the benchmark harness."""
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

    # Prefer explicit phase flags if the harness adds them later. Fall back to
    # the ring/token hop flags used by earlier labs, then default to full phases.
    raw_rs_hops = _first_present_attr(
        args,
        ("reduce_scatter_hops", "ring_hops", "token_hops"),
    )
    raw_ag_hops = _first_present_attr(
        args,
        ("all_gather_hops", "ring_hops", "token_hops"),
    )
    reduce_scatter_hops = normalize_hops(raw_rs_hops, n_devices=n_devices)
    all_gather_hops = normalize_hops(raw_ag_hops, n_devices=n_devices)

    # The spec works with requested payload bytes, while build_case may round to
    # a whole tile. Treat this as an explanatory model, not the exact row value.
    approx_chunk_bytes = int(payload_bytes) // max(1, int(n_devices))
    base_collective_id = int(getattr(args, "pallas_collective_id", 0) or 0)

    spec["phase_count"] = 2
    spec["phases"] = [
        {
            "name": "reduce_scatter",
            "implementation": "Lab 6 whole-token ring reduce-scatter",
            "input_ownership": "each device owns all owner chunks for its source",
            "output_ownership": "device j owns reduced chunk j",
            "collective_ids": phase_collective_ids(
                base_collective_id=base_collective_id,
                reduce_scatter_hops=reduce_scatter_hops,
                all_gather_hops=all_gather_hops,
            )["reduce_scatter"],
        },
        {
            "name": "all_gather",
            "implementation": "Lab 5 ring all-gather over reduced chunks",
            "input_ownership": "each device owns one reduced chunk",
            "output_ownership": "every device has all reduced chunks in arrival order",
            "collective_ids": phase_collective_ids(
                base_collective_id=base_collective_id,
                reduce_scatter_hops=reduce_scatter_hops,
                all_gather_hops=all_gather_hops,
            )["all_gather"],
        },
    ]
    spec["local_layout_phase"] = {
        "name": "canonical_reorder",
        "implementation": "local shard_map take over arrival slots",
        "collective_ids": [],
        "purpose": "convert ring arrival order to canonical owner-chunk order",
    }
    spec["neighbor_direction"] = direction
    spec["chunk_count"] = int(n_devices)
    spec["reduce_scatter_hops"] = int(reduce_scatter_hops)
    spec["all_gather_hops"] = int(all_gather_hops)
    spec["phase_collective_ids"] = phase_collective_ids(
        base_collective_id=base_collective_id,
        reduce_scatter_hops=reduce_scatter_hops,
        all_gather_hops=all_gather_hops,
    )
    spec["all_gather_arrival_owner_table"] = all_gather_owner_table(
        n_devices=n_devices,
        all_gather_hops=all_gather_hops,
        direction=direction,
    )
    spec["canonical_arrival_slot_table_for_full_all_gather"] = canonical_arrival_slot_table(
        n_devices=n_devices,
        direction=direction,
    )
    spec["expected_values_for_teaching_input"] = expected_all_reduce_table(
        n_devices=n_devices,
        reduce_scatter_hops=reduce_scatter_hops,
        all_gather_hops=all_gather_hops,
        direction=direction,
    )
    spec["byte_model_for_this_payload"] = all_reduce_byte_model(
        n_devices=n_devices,
        full_payload_bytes=payload_bytes,
        chunk_payload_bytes=approx_chunk_bytes,
        reduce_scatter_hops=reduce_scatter_hops,
        all_gather_hops=all_gather_hops,
        direction=direction,
    )
    spec["custom_collective_status"] = (
        "implemented as Lab 6 whole-token reduce-scatter plus Lab 5 chunk "
        "all-gather with canonical chunk-order restore; optimized one-chunk "
        "reduce-scatter and fused phase transition remain deferred"
    )
    spec["correctness_contract"] = (
        "For a full all-reduce, the output shape is "
        "[num_devices, num_devices, tile_rows, tile_cols]. Every receiver must "
        "hold the same canonical owner-chunk table. Full tile contents must "
        "match, not only y[:, :, 0, 0]."
    )
    return spec


def _fallback_render_markdown(spec: dict[str, Any]) -> str:
    """Simple Markdown renderer used only when lab_spec_utils is unavailable."""
    lines = [f"# {spec.get('title', 'Lab 7 All-Reduce')}", ""]
    for key in (
        "goal",
        "custom_collective_status",
        "correctness_contract",
    ):
        value = spec.get(key)
        if value:
            lines.extend([f"## {key.replace('_', ' ').title()}", "", str(value), ""])
    for key in (
        "implemented_ops",
        "deferred_ops",
        "pass_condition",
        "suggested_experiments",
    ):
        values = spec.get(key)
        if values:
            lines.extend([f"## {key.replace('_', ' ').title()}", ""])
            for item in values:
                lines.append(f"- {item}")
            lines.append("")
    if "phases" in spec:
        lines.extend(["## Phases", ""])
        for phase in spec["phases"]:
            lines.append(f"- `{phase.get('name')}`: {phase.get('implementation')}")
        lines.append("")
    if "phase_collective_ids" in spec:
        lines.extend(["## Collective IDs", "", "```text"])
        for key, value in spec["phase_collective_ids"].items():
            lines.append(f"{key}: {value}")
        lines.extend(["```", ""])
    if "collective_id_plan" in spec:
        lines.extend(["## Collective ID Plan", ""])
        for key, value in spec["collective_id_plan"].items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    if "byte_model_for_this_payload" in spec:
        lines.extend(["## Byte Model For This Payload", ""])
        for key, value in spec["byte_model_for_this_payload"].items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    if "expected_values_for_teaching_input" in spec:
        lines.extend(["## Expected Teaching Values", "", "```text"])
        for row in spec["expected_values_for_teaching_input"]:
            lines.append(str(row))
        lines.extend(["```", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(spec: dict[str, Any]) -> str:
    """Render a Lab 7 spec artifact as Markdown."""
    if lab_spec_utils is not None:
        return lab_spec_utils.render_markdown(spec)
    return _fallback_render_markdown(spec)
