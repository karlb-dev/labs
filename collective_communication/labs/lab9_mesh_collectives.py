"""Lab 9: topology-aware staged all-gather over a logical 2D mesh.

This file contains the concept code for the Lab 9 teaching module. The benchmark
harness owns run directories, logging, CSV/JSONL output, plots, and profiler
capture. Keeping those pieces out of the lab file lets students read one idea at
a time.

What this lab implements
========================

The implemented custom collective is a **two-stage all-gather**:

    1. gather along one logical mesh axis;
    2. gather those partial results along the other logical mesh axis;
    3. reorder the staged arrival grid into canonical flat-rank order.

The ``pmap`` version uses ``lax.ppermute`` so it can run as a reference. The
Pallas version uses explicit remote DMA to a logical 2D neighbor. Both versions
use the same schedule helpers, so students can compare the executable code to
the tables in the spec artifact.

Important design choice
=======================

The Pallas path keeps the *JAX* mesh flat and computes a logical 2D coordinate
from the one-dimensional device rank. That keeps Lab 9 compatible with the Lab 1
remote-DMA substrate while teaching 2D topology. Later labs can replace this
logical reshape with physical-coordinate-aware placement and multi-host axes.
"""

from __future__ import annotations

import dataclasses
import functools
import math
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
SUPPORTED_AXIS_ORDERS = frozenset({"x_then_y", "y_then_x"})
MESH_AXES = frozenset({"x", "y"})

# Position offsets for teaching payloads. Element [source, row, col] keeps
# [source, 0, 0] equal to the source rank while making every other element
# slightly different. That turns full-tile correctness into a real test instead
# of a rank-marker costume party.
ROW_PATTERN_SCALE = 0.25
COL_PATTERN_SCALE = 1.0 / 32.0


@dataclasses.dataclass(frozen=True)
class MeshAllGatherCase:
    """Container returned to the benchmark harness.

    The harness needs a sharded input ``x``, a jitted function ``fn``, and an
    expected result for correctness. The remaining fields make CSV rows, notes,
    and lab artifacts explain themselves.

    Shape convention
    ----------------

    ``pmap_2d_staged_all_gather`` uses a simple input shape::

        [device, col]

    and returns::

        [device, owner_rank, col]

    ``pallas_2d_staged_all_gather`` uses a tiled input shape::

        [device, row, col]

    and returns::

        [device, owner_rank, row, col]

    The first dimension is partitioned across devices. The ``expected_ranks``
    field name is kept for harness compatibility, but the updated teaching
    cases store full expected payload values so ``check_result`` can validate
    every element, not just the first rank marker.
    """

    x: Any
    fn: Callable[[Any], Any]
    expected_ranks: Any
    actual_payload_bytes: int
    mesh_shape: tuple[int, int]
    axis_order: str
    direction: str
    tile_rows: int
    tile_cols: int

    @property
    def n_devices(self) -> int:
        """Number of devices represented by the logical 2D mesh."""
        return self.mesh_shape[0] * self.mesh_shape[1]

    @property
    def num_devices(self) -> int:
        """Alias used by some benchmark/reporting code."""
        return self.n_devices

    @property
    def first_axis(self) -> str:
        """The mesh axis gathered during stage 1."""
        return axis_sequence(self.axis_order)[0]

    @property
    def second_axis(self) -> str:
        """The mesh axis gathered during stage 2."""
        return axis_sequence(self.axis_order)[1]

    @property
    def first_axis_size(self) -> int:
        return axis_size(self.mesh_shape, self.first_axis)

    @property
    def second_axis_size(self) -> int:
        return axis_size(self.mesh_shape, self.second_axis)

    @property
    def first_stage_hops(self) -> int:
        return max(0, self.first_axis_size - 1)

    @property
    def second_stage_hops(self) -> int:
        return max(0, self.second_axis_size - 1)

    @property
    def staged_remote_copy_phases(self) -> int:
        """Number of remote-copy phases in the staged teaching path."""
        return self.first_stage_hops + self.second_stage_hops

    @property
    def first_stage_send_bytes_per_device(self) -> int:
        """Per-device send bytes during stage 1."""
        return self.actual_payload_bytes * self.first_stage_hops

    @property
    def second_stage_send_bytes_per_device(self) -> int:
        """Per-device send bytes during stage 2.

        The second stage sends partial gathers. After stage 1, each partial has
        ``first_axis_size`` shards, so one second-axis hop moves
        ``first_axis_size * actual_payload_bytes``.
        """
        return self.actual_payload_bytes * self.first_axis_size * self.second_stage_hops

    @property
    def wire_bytes(self) -> int:
        """Backward-compatible per-device send-byte estimate."""
        return self.staged_send_bytes_per_device

    @property
    def staged_send_bytes_per_device(self) -> int:
        """Total per-device send bytes for the implemented staged all-gather."""
        return self.first_stage_send_bytes_per_device + self.second_stage_send_bytes_per_device

    @property
    def staged_recv_bytes_per_device(self) -> int:
        """Total per-device receive bytes for the implemented staged all-gather."""
        return self.staged_send_bytes_per_device

    @property
    def flat_ring_send_bytes_per_device(self) -> int:
        """Per-device send bytes for a one-direction flat ring all-gather."""
        return self.actual_payload_bytes * max(0, self.n_devices - 1)

    @property
    def result_bytes_per_device(self) -> int:
        """Logical all-gather result size per device."""
        return self.actual_payload_bytes * self.n_devices

    @property
    def note(self) -> str:
        """Short human-readable note for benchmark tables."""
        return (
            f"mesh={format_mesh_shape(self.mesh_shape)} "
            f"order={self.axis_order} direction={self.direction} "
            f"tile={self.tile_rows}x{self.tile_cols} "
            f"phases={self.staged_remote_copy_phases} "
            f"wire/dev={self.staged_send_bytes_per_device}B"
        )

    def byte_model(self) -> dict[str, Any]:
        """Return benchmark-friendly byte accounting for this case."""
        return mesh_all_gather_byte_model(
            mesh_shape=self.mesh_shape,
            axis_order=self.axis_order,
            payload_bytes=self.actual_payload_bytes,
        )

    def stage_plan(self) -> list[dict[str, Any]]:
        """Return a compact stage plan for artifacts or debugging."""
        return staged_all_gather_plan(
            mesh_shape=self.mesh_shape,
            axis_order=self.axis_order,
            direction=self.direction,
            base_collective_id=0,
            payload_bytes=self.actual_payload_bytes,
        )


LAB_SPEC: dict[str, Any] = {
    "lab": "lab9",
    "title": "Lab 9: 2D Mesh Collectives",
    "goal": (
        "Stop treating every slice as a flat ring. Compare flat ring movement "
        "with staged all-gather over logical mesh axes, and learn to connect "
        "axis order, message size, hop groups, and profiler evidence."
    ),
    "implemented_ops": [
        "`pmap_all_gather`: built-in all-gather baseline",
        "`pmap_2d_staged_all_gather`: CPU-runnable x/y staged reference",
        "`pallas_2d_staged_all_gather`: TPU remote-DMA staged all-gather",
        "`lab9_mesh_collectives_spec`: mesh-shape, byte-model, and schedule artifact",
    ],
    "deferred_ops": [
        "Add reduce-scatter and all-reduce variants over the same staged mesh",
        "Compare x-then-y and y-then-x traces against physical topology evidence",
        "Record TPU coordinates alongside every staged schedule",
        "Choose logical mesh layout from physical coordinates instead of local-device order",
        "Extend staged algorithms across host boundaries in Lab 10",
    ],
    "byte_model": [
        "stage 1 sends `(first_axis_size - 1) * shard_bytes` per device",
        "stage 2 sends `(second_axis_size - 1) * first_axis_size * shard_bytes` per device",
        "total staged send bytes equal `(n_devices - 1) * shard_bytes` per device",
        "same logical bytes can have different phase boundaries and contention",
    ],
    "pass_condition": [
        "built-in baseline passes",
        "pmap staged all-gather matches canonical rank order",
        "Pallas staged all-gather matches canonical rank order on TPU",
        "full payload tiles are checked, not only rank marker scalars",
        "expected tiles model the wire-dtype input quantization, so the check is "
        "exact on 4-, 8-, and 16-device meshes instead of tripping bfloat16 "
        "rounding once ranks push tile values past a precision boundary",
        "spec artifact records candidate 2D mesh shapes and axis-order experiments",
    ],
    "artifacts": [
        "results.jsonl",
        "csvs/results.csv",
        "plots/latency_by_payload.png",
        "plots/bandwidth_by_payload.png",
        "run_metadata.json device report",
        "diagnostics/runtime.json",
        "lab_artifacts/*lab9_mesh_collectives_spec*",
    ],
    "next_steps": [
        "Use Lab 10 process topology before extending staged algorithms multi-host",
    ],
}


def ceil_div(a: int, b: int) -> int:
    """Return ``ceil(a / b)`` for positive integers."""
    if b <= 0:
        raise ValueError(f"ceil_div divisor must be positive, got {b}")
    return -(-int(a) // int(b))


def normalize_direction(direction: str | None) -> str:
    """Normalize and validate a sender-direction string."""
    normalized = (direction or "right").strip().lower()
    if normalized not in SUPPORTED_DIRECTIONS:
        raise ValueError(
            f"Lab 9 direction must be one of {sorted(SUPPORTED_DIRECTIONS)}, "
            f"got {direction!r}"
        )
    return normalized


def normalize_axis_order(axis_order: str | None) -> str:
    """Normalize and validate a two-stage axis-order string."""
    normalized = (axis_order or "x_then_y").strip().lower().replace("-", "_")
    aliases = {
        "xy": "x_then_y",
        "x_y": "x_then_y",
        "x_then_y": "x_then_y",
        "yx": "y_then_x",
        "y_x": "y_then_x",
        "y_then_x": "y_then_x",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_AXIS_ORDERS:
        raise ValueError(
            f"Lab 9 axis order must be one of {sorted(SUPPORTED_AXIS_ORDERS)}, "
            f"got {axis_order!r}"
        )
    return normalized


def normalize_mesh_axis(mesh_axis: str) -> str:
    """Normalize and validate a logical mesh-axis name."""
    normalized = mesh_axis.strip().lower()
    if normalized not in MESH_AXES:
        raise ValueError(f"mesh axis must be one of {sorted(MESH_AXES)}, got {mesh_axis!r}")
    return normalized


def all_2d_factor_shapes(n_devices: int) -> list[tuple[int, int]]:
    """Return all 2D factorizations of ``n_devices``, near-square first.

    Both orientations are included. For eight devices this returns ``2x4`` and
    ``4x2`` before the flat-ish ``1x8`` and ``8x1`` shapes. Lab 9 is explicitly
    a 2D lab, so this function does not return physical 3D shapes such as
    ``2x2x2``.
    """
    n = max(1, int(n_devices))
    shapes = [(x, n // x) for x in range(1, n + 1) if n % x == 0]
    shapes.sort(key=lambda pair: (abs(pair[0] - pair[1]), pair[0]))
    return shapes


def parse_mesh_shape(mesh_shape: str | None, n_devices: int) -> tuple[int, int]:
    """Parse a 2D logical mesh shape.

    ``auto`` chooses a near-square factorization. Explicit values use ``XxY`` or
    ``X,Y`` syntax. The product must match the number of devices in the local
    run because this lab stages over the local device list.
    """
    n = max(1, int(n_devices))
    requested = (mesh_shape or "auto").strip().lower()
    if requested in {"", "auto", "near_square", "square"}:
        return all_2d_factor_shapes(n)[0]
    if requested in {"flat", "flat_ring", "1d"}:
        return (1, n)

    normalized = requested.replace(" ", "").replace(",", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Lab 9 expects a 2D mesh shape like '2x2', '2x4', or 'auto'; "
            f"got {mesh_shape!r}"
        )
    try:
        x_size, y_size = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"mesh dimensions must be integers, got {mesh_shape!r}") from exc
    if x_size <= 0 or y_size <= 0:
        raise ValueError(f"mesh dimensions must be positive, got {mesh_shape!r}")
    if x_size * y_size != n:
        raise ValueError(
            f"mesh shape {x_size}x{y_size} needs {x_size * y_size} devices, "
            f"but this run has {n}"
        )
    return (x_size, y_size)


def format_mesh_shape(mesh_shape: tuple[int, int]) -> str:
    """Format ``(x_size, y_size)`` as ``'XxY'``."""
    return f"{mesh_shape[0]}x{mesh_shape[1]}"


def axis_sequence(axis_order: str) -> tuple[str, str]:
    """Return ``(first_axis, second_axis)`` for a two-stage schedule."""
    normalized = normalize_axis_order(axis_order)
    if normalized == "x_then_y":
        return ("x", "y")
    if normalized == "y_then_x":
        return ("y", "x")
    # ``normalize_axis_order`` should have rejected everything else.
    raise ValueError(f"unknown Lab 9 axis order {axis_order!r}")


def axis_size(mesh_shape: tuple[int, int], mesh_axis: str) -> int:
    """Return the size of one logical 2D mesh axis."""
    x_size, y_size = mesh_shape
    axis = normalize_mesh_axis(mesh_axis)
    return x_size if axis == "x" else y_size


def logical_rank(x_coord: int, y_coord: int, y_size: int) -> int:
    """Return row-major rank for logical coordinate ``(x_coord, y_coord)``."""
    return int(x_coord) * int(y_size) + int(y_coord)


def logical_coords(rank: int, mesh_shape: tuple[int, int]) -> tuple[int, int]:
    """Return ``(x_coord, y_coord)`` for a row-major logical rank."""
    _, y_size = mesh_shape
    r = int(rank)
    return (r // y_size, r % y_size)


def coord_on_axis(coords: tuple[int, int], mesh_axis: str) -> int:
    """Return the coordinate value along ``mesh_axis``."""
    axis = normalize_mesh_axis(mesh_axis)
    return coords[0] if axis == "x" else coords[1]


def replace_axis_coord(
    coords: tuple[int, int],
    *,
    mesh_axis: str,
    new_coord: int,
) -> tuple[int, int]:
    """Return coordinates with one logical axis coordinate replaced."""
    axis = normalize_mesh_axis(mesh_axis)
    if axis == "x":
        return (int(new_coord), coords[1])
    return (coords[0], int(new_coord))


def source_coord_at_arrival(
    receiver_coord: int,
    arrival_index: int,
    size: int,
    direction: str,
) -> int:
    """Return the source coordinate seen at a receiver arrival index.

    Direction is from the sender's point of view. For a right-moving ring,
    receiver ``r`` sees coordinates ``[r, r - 1, r - 2, ...]``. For a left-moving
    ring, receiver ``r`` sees ``[r, r + 1, r + 2, ...]``.
    """
    direction = normalize_direction(direction)
    if size <= 0:
        raise ValueError(f"axis size must be positive, got {size}")
    if direction == "right":
        return (int(receiver_coord) - int(arrival_index)) % int(size)
    return (int(receiver_coord) + int(arrival_index)) % int(size)


def mesh_axis_perm(
    mesh_shape: tuple[int, int],
    mesh_axis: str,
    direction: str,
) -> tuple[tuple[int, int], ...]:
    """Return a ``lax.ppermute`` table for one logical mesh-axis hop.

    The source/target pairs are expressed in the flat rank space used by pmap.
    """
    direction = normalize_direction(direction)
    axis = normalize_mesh_axis(mesh_axis)
    x_size, y_size = mesh_shape
    step = 1 if direction == "right" else -1
    perm: list[tuple[int, int]] = []
    for x_coord in range(x_size):
        for y_coord in range(y_size):
            if axis == "x":
                dst = logical_rank((x_coord + step) % x_size, y_coord, y_size)
            else:
                dst = logical_rank(x_coord, (y_coord + step) % y_size, y_size)
            src = logical_rank(x_coord, y_coord, y_size)
            perm.append((src, dst))
    return tuple(perm)


def expected_rank_table(jnp: Any, n_devices: int) -> Any:
    """Return the canonical all-gather owner-rank table.

    Shape: ``[receiver, owner_rank]``. Every receiver should end with the same
    canonical owner sequence ``[0, 1, ..., N - 1]``.
    """
    ranks = jnp.arange(int(n_devices), dtype=jnp.float32)
    return jnp.broadcast_to(ranks.reshape(1, int(n_devices)), (int(n_devices), int(n_devices)))


def rank_layout(mesh_shape: tuple[int, int]) -> list[list[int]]:
    """Return a Python table mapping logical coordinates to ranks."""
    x_size, y_size = mesh_shape
    return [[logical_rank(x, y, y_size) for y in range(y_size)] for x in range(x_size)]


def mesh_axis_groups(mesh_shape: tuple[int, int], mesh_axis: str) -> list[list[int]]:
    """Return the independent rank groups used by one staged axis.

    Gathering along ``x`` keeps ``y`` fixed and varies ``x``, so the groups are
    the columns of the logical rank layout. Gathering along ``y`` keeps ``x``
    fixed and varies ``y``, so the groups are the rows.
    """
    axis = normalize_mesh_axis(mesh_axis)
    x_size, y_size = mesh_shape
    if axis == "x":
        return [
            [logical_rank(x, y, y_size) for x in range(x_size)]
            for y in range(y_size)
        ]
    return [
        [logical_rank(x, y, y_size) for y in range(y_size)]
        for x in range(x_size)
    ]


def all_axis_groups(mesh_shape: tuple[int, int]) -> dict[str, list[list[int]]]:
    """Return staged-communication groups for both logical axes."""
    return {"x": mesh_axis_groups(mesh_shape, "x"), "y": mesh_axis_groups(mesh_shape, "y")}


def make_rank_position_tile(
    jnp: Any,
    *,
    n_devices: int,
    rows: int,
    cols: int,
    dtype: Any,
) -> Any:
    """Return a per-source payload with rank and position encoded.

    Shape: ``[source_rank, row, col]``. Element ``[s, r, c]`` equals::

        s + ROW_PATTERN_SCALE * r + COL_PATTERN_SCALE * (c mod 32)

    The first element of each source tile remains exactly the source rank, so
    old rank-marker debugging still works. The nonzero row/column offsets make
    full-tile correctness checks catch partial copies and accidental broadcasts.
    """
    ranks = jnp.arange(int(n_devices), dtype=jnp.float32).reshape(int(n_devices), 1, 1)
    row_offsets = ROW_PATTERN_SCALE * jnp.arange(int(rows), dtype=jnp.float32).reshape(1, int(rows), 1)
    col_offsets = COL_PATTERN_SCALE * jnp.mod(
        jnp.arange(int(cols), dtype=jnp.float32), 32
    ).reshape(1, 1, int(cols))
    return (ranks + row_offsets + col_offsets).astype(dtype)


def expected_tile_from_ranks(
    rank_table: Any,
    *,
    rows: int,
    cols: int,
    use_position_offsets: bool,
    dtype: Any = None,
) -> Any:
    """Expand ``[receiver, source]`` rank markers into a full expected tile.

    The new Lab 9 builders set ``use_position_offsets=True`` so the checker
    validates every tile element. The flag is useful for comparing with older
    rank-only inputs during standalone smoke tests.

    Modeling input quantization
    ---------------------------

    Lab 9 is a pure all-gather: the kernel *moves* shards, it never does
    arithmetic, so the output equals the input bit for bit. The input itself is
    ``make_rank_position_tile(...).astype(dtype)`` -- the rank/row/col sum
    rounded to the wire dtype. When ``dtype`` is given, this function rounds the
    expected tile the same way, so the comparison is against the values the
    hardware actually carries. This matters once tile values cross a dtype's
    precision boundary: with ``N`` devices, ranks reach ``N - 1``, and for
    ``N >= 8`` a bfloat16 tile value can exceed 8.0, where one ULP (0.0625) is
    larger than a naive ``atol`` -- so a float32-only reference would flag a
    perfectly correct gather as wrong on an 8- or 16-device slice while passing
    on a 4-device one. Leaving ``dtype`` as ``None`` keeps the old float32
    behaviour for legacy callers.
    """
    import jax.numpy as jnp

    base = rank_table.astype(jnp.float32)[:, :, None, None]
    if not use_position_offsets:
        tile = jnp.broadcast_to(
            base, (rank_table.shape[0], rank_table.shape[1], int(rows), int(cols))
        )
    else:
        row_offsets = ROW_PATTERN_SCALE * jnp.arange(int(rows), dtype=jnp.float32).reshape(1, 1, int(rows), 1)
        col_offsets = COL_PATTERN_SCALE * jnp.mod(
            jnp.arange(int(cols), dtype=jnp.float32), 32
        ).reshape(1, 1, 1, int(cols))
        tile = base + row_offsets + col_offsets
    if dtype is not None:
        # Round through the wire dtype to model the input quantization, then view
        # the result back in float32 for a clean, dtype-aware comparison.
        tile = tile.astype(jnp.dtype(dtype)).astype(jnp.float32)
    return tile


def staged_arrival_rank_grid_for_receiver(
    *,
    receiver_rank: int,
    mesh_shape: tuple[int, int],
    axis_order: str,
    direction: str,
) -> list[list[int]]:
    """Return staged arrival owner ranks for one receiver.

    The outer list indexes second-stage arrival. The inner list indexes
    first-stage arrival. This is the table students should draw by hand.
    """
    direction = normalize_direction(direction)
    first_axis, second_axis = axis_sequence(axis_order)
    receiver_coords = logical_coords(receiver_rank, mesh_shape)
    first_size = axis_size(mesh_shape, first_axis)
    second_size = axis_size(mesh_shape, second_axis)
    receiver_first = coord_on_axis(receiver_coords, first_axis)
    receiver_second = coord_on_axis(receiver_coords, second_axis)

    rows: list[list[int]] = []
    for second_idx in range(second_size):
        source_second = source_coord_at_arrival(
            receiver_second,
            second_idx,
            second_size,
            direction,
        )
        row: list[int] = []
        for first_idx in range(first_size):
            source_first = source_coord_at_arrival(
                receiver_first,
                first_idx,
                first_size,
                direction,
            )
            # Build coordinates by assigning the two axis coordinates in the
            # order chosen by the staged algorithm.
            coords = receiver_coords
            coords = replace_axis_coord(coords, mesh_axis=first_axis, new_coord=source_first)
            coords = replace_axis_coord(coords, mesh_axis=second_axis, new_coord=source_second)
            row.append(logical_rank(coords[0], coords[1], mesh_shape[1]))
        rows.append(row)
    return rows


def staged_arrival_rank_tables(
    *,
    mesh_shape: tuple[int, int],
    axis_order: str,
    direction: str,
    max_receivers: int | None = None,
) -> dict[str, list[list[int]]]:
    """Return staged arrival grids for several receivers."""
    n_devices = mesh_shape[0] * mesh_shape[1]
    limit = n_devices if max_receivers is None else min(n_devices, int(max_receivers))
    return {
        f"receiver_{rank}": staged_arrival_rank_grid_for_receiver(
            receiver_rank=rank,
            mesh_shape=mesh_shape,
            axis_order=axis_order,
            direction=direction,
        )
        for rank in range(limit)
    }


def canonical_lookup_for_receiver(
    *,
    receiver_rank: int,
    mesh_shape: tuple[int, int],
    axis_order: str,
    direction: str,
) -> list[dict[str, int]]:
    """Return canonical owner -> staged arrival-index lookup for one receiver."""
    grid = staged_arrival_rank_grid_for_receiver(
        receiver_rank=receiver_rank,
        mesh_shape=mesh_shape,
        axis_order=axis_order,
        direction=direction,
    )
    lookup: dict[int, tuple[int, int]] = {}
    for second_idx, row in enumerate(grid):
        for first_idx, owner in enumerate(row):
            lookup[int(owner)] = (second_idx, first_idx)
    return [
        {
            "owner_rank": owner,
            "second_stage_arrival_index": lookup[owner][0],
            "first_stage_arrival_index": lookup[owner][1],
        }
        for owner in range(mesh_shape[0] * mesh_shape[1])
    ]


def _arrival_index(coord: Any, target: int, size: int, direction: str) -> Any:
    """JAX version of canonical owner -> arrival-index math."""
    import jax.numpy as jnp

    direction = normalize_direction(direction)
    if direction == "right":
        return jnp.mod(coord - target, size)
    return jnp.mod(target - coord, size)


def canonical_from_2d_arrival(
    arrival: Any,
    *,
    rank: Any,
    mesh_shape: tuple[int, int],
    axis_order: str,
    direction: str,
) -> Any:
    """Convert staged arrival order to canonical flat-rank order.

    Parameters
    ----------
    arrival:
        Local staged-arrival tensor with shape ``[second_arrival, first_arrival,
        ...payload...]``.
    rank:
        Receiver rank along the flat pmap/shard_map axis.
    mesh_shape:
        Logical 2D mesh shape ``(x_size, y_size)``.
    axis_order:
        ``x_then_y`` or ``y_then_x``.
    direction:
        Sender direction, ``right`` or ``left``.

    Returns
    -------
    Tensor with shape ``[owner_rank, ...payload...]``.
    """
    import jax.numpy as jnp

    direction = normalize_direction(direction)
    first_axis, second_axis = axis_sequence(axis_order)
    x_size, y_size = mesh_shape
    x_coord = rank // y_size
    y_coord = jnp.mod(rank, y_size)
    rows = []
    for owner in range(x_size * y_size):
        target_x = owner // y_size
        target_y = owner % y_size
        current_first = x_coord if first_axis == "x" else y_coord
        target_first = target_x if first_axis == "x" else target_y
        current_second = x_coord if second_axis == "x" else y_coord
        target_second = target_x if second_axis == "x" else target_y
        first_idx = _arrival_index(
            current_first,
            target_first,
            axis_size(mesh_shape, first_axis),
            direction,
        )
        second_idx = _arrival_index(
            current_second,
            target_second,
            axis_size(mesh_shape, second_axis),
            direction,
        )
        block = jnp.take(arrival, second_idx, axis=0)
        rows.append(jnp.take(block, first_idx, axis=0))
    return jnp.stack(rows, axis=0)


def staged_2d_all_gather_pmap(
    value: Any,
    *,
    axis_name: str,
    mesh_shape: tuple[int, int],
    axis_order: str,
    direction: str,
) -> Any:
    """Reference staged all-gather using repeated ``lax.ppermute``.

    This is the executable spec for the Pallas path. It is intentionally small:
    first collect payloads along one logical axis, then collect those partial
    results along the other axis, then canonicalize the output.
    """
    from jax import lax
    import jax.numpy as jnp

    direction = normalize_direction(direction)
    first_axis, second_axis = axis_sequence(axis_order)

    # Stage 1: move the original shard around the first logical axis.
    token = value
    first_pieces = [token]
    for _ in range(max(0, axis_size(mesh_shape, first_axis) - 1)):
        token = lax.ppermute(
            token,
            axis_name,
            perm=mesh_axis_perm(mesh_shape, first_axis, direction),
        )
        first_pieces.append(token)
    partial = jnp.stack(first_pieces, axis=0)

    # Stage 2: move the stage-1 partial gather around the second logical axis.
    token = partial
    second_blocks = [token]
    for _ in range(max(0, axis_size(mesh_shape, second_axis) - 1)):
        token = lax.ppermute(
            token,
            axis_name,
            perm=mesh_axis_perm(mesh_shape, second_axis, direction),
        )
        second_blocks.append(token)
    arrival = jnp.stack(second_blocks, axis=0)

    return canonical_from_2d_arrival(
        arrival,
        rank=lax.axis_index(axis_name),
        mesh_shape=mesh_shape,
        axis_order=axis_order,
        direction=direction,
    )


def logical_mesh_neighbor_rank(
    idx: Any,
    *,
    x_size: int,
    y_size: int,
    mesh_axis: str,
    direction: str,
) -> Any:
    """Return the flat rank of a logical 2D neighbor.

    This function is used inside a Pallas kernel, so the implementation uses JAX
    array operations rather than pure Python integer arithmetic.
    """
    import jax.numpy as jnp

    direction = normalize_direction(direction)
    axis = normalize_mesh_axis(mesh_axis)
    x_coord = idx // y_size
    y_coord = jnp.mod(idx, y_size)
    step = 1 if direction == "right" else -1
    if axis == "x":
        x_coord = jnp.mod(x_coord + step, x_size)
    else:
        y_coord = jnp.mod(y_coord + step, y_size)
    return x_coord * y_size + y_coord


def logical_neighbor_copy_kernel(
    x_ref,
    o_ref,
    send_sem,
    recv_sem,
    *,
    axis_name: str,
    mesh_axis: str,
    x_size: int,
    y_size: int,
    direction: str,
) -> None:
    """Pallas TPU kernel: remote-DMA to a logical 2D-mesh neighbor.

    The surrounding JAX mesh is still one-dimensional. ``dst_mesh_index`` is a
    one-element tuple because Pallas remote DMA is addressing that flat mesh.
    The logical 2D interpretation lives in ``logical_mesh_neighbor_rank``.
    """
    import jax
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    direction = normalize_direction(direction)
    axis = normalize_mesh_axis(mesh_axis)
    my_id = lax.axis_index(axis_name)
    dst_rank = logical_mesh_neighbor_rank(
        my_id,
        x_size=x_size,
        y_size=y_size,
        mesh_axis=axis,
        direction=direction,
    )
    dst_mesh_index = (dst_rank,)

    # Entry barrier: everyone reaches the phase before anyone writes into a peer
    # output buffer. This is the same synchronization discipline introduced in
    # Lab 1, now repeated once per staged phase.
    with jax.named_scope("lab9_entry_barrier"):
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=dst_mesh_index,
            device_id_type=pl.DeviceIdType.MESH,
        )
        pltpu.semaphore_wait(barrier_sem, dec=1)

    # Remote DMA: explicit descriptor lifecycle. Later labs can place work
    # between ``start`` and the waits. Lab 9 waits immediately so the schedule is
    # easy to reason about.
    with jax.named_scope("lab9_remote_dma"):
        dma = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=o_ref,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=dst_mesh_index,
            device_id_type=pl.DeviceIdType.MESH,
        )
        dma.start()
        dma.wait_send()
        dma.wait_recv()


def _get_shard_map(jax: Any) -> Callable[..., Any]:
    """Return public ``jax.shard_map`` when available, else compatibility path."""
    public_shard_map = getattr(jax, "shard_map", None)
    if public_shard_map is not None:
        return public_shard_map
    try:
        from jax.experimental import shard_map as shard_map_module
    except Exception:  # pragma: no cover - compatibility fallback only.
        from jax._src import shard_map as shard_map_module  # type: ignore[import-not-found]
    return shard_map_module.shard_map


def logical_neighbor_copy(
    x: Any,
    *,
    mesh: Any,
    axis_name: str,
    mesh_axis: str,
    mesh_shape: tuple[int, int],
    direction: str,
    collective_id: int,
    memory_space: Any,
) -> Any:
    """Shard-mapped Pallas copy to a logical 2D-mesh neighbor.

    This is the Lab 9 equivalent of the Lab 1 single-hop primitive. The only new
    ingredient is that the destination rank is computed from a logical 2D axis.
    """
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    direction = normalize_direction(direction)
    axis = normalize_mesh_axis(mesh_axis)
    x_size, y_size = mesh_shape
    partition = jax.sharding.PartitionSpec
    shard_map_fn = _get_shard_map(jax)

    def local_copy(x_shard):
        out_shape = jax.ShapeDtypeStruct(x_shard.shape, x_shard.dtype)
        return pl.pallas_call(
            functools.partial(
                logical_neighbor_copy_kernel,
                axis_name=axis_name,
                mesh_axis=axis,
                x_size=x_size,
                y_size=y_size,
                direction=direction,
            ),
            out_shape=out_shape,
            compiler_params=pltpu.CompilerParams(collective_id=int(collective_id)),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                scratch_shapes=(
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
                in_specs=[pl.BlockSpec(memory_space=memory_space)],
                out_specs=pl.BlockSpec(memory_space=memory_space),
            ),
        )(x_shard)

    return shard_map_fn(
        local_copy,
        mesh=mesh,
        in_specs=partition(axis_name),
        out_specs=partition(axis_name),
        check_vma=False,
    )(x)


def canonical_from_2d_arrival_sharded(
    value: Any,
    *,
    mesh: Any,
    axis_name: str,
    mesh_shape: tuple[int, int],
    axis_order: str,
    direction: str,
) -> Any:
    """Shard-mapped canonicalization for Pallas staged-arrival output."""
    import jax
    from jax import lax

    partition = jax.sharding.PartitionSpec
    shard_map_fn = _get_shard_map(jax)

    def local_reorder(arrival):
        # shard_map keeps the sharded device axis as a leading size-1 dimension,
        # so the local shard is [1, second_arrival, first_arrival, ...payload].
        # canonical_from_2d_arrival expects the pmap-style layout without that
        # axis ([second_arrival, first_arrival, ...payload]), so drop it before
        # reordering and restore it afterwards. (The pmap path calls
        # canonical_from_2d_arrival directly because pmap removes the device axis.)
        canonical = canonical_from_2d_arrival(
            arrival[0],
            rank=lax.axis_index(axis_name),
            mesh_shape=mesh_shape,
            axis_order=axis_order,
            direction=direction,
        )
        return canonical[None]

    return shard_map_fn(
        local_reorder,
        mesh=mesh,
        in_specs=partition(axis_name),
        out_specs=partition(axis_name),
        check_vma=False,
    )(value)


def staged_2d_all_gather_pallas(
    value: Any,
    *,
    mesh: Any,
    axis_name: str,
    mesh_shape: tuple[int, int],
    axis_order: str,
    direction: str,
    collective_id: int,
    memory_space: Any,
) -> Any:
    """Compose logical-neighbor remote DMA into a staged 2D all-gather."""
    import jax.numpy as jnp

    direction = normalize_direction(direction)
    first_axis, second_axis = axis_sequence(axis_order)

    # Stage 1: move the original shard around the first logical axis. Each hop
    # uses a distinct collective ID so semaphore state is not accidentally shared.
    token = value
    first_pieces = [token]
    id_offset = 0
    first_hops = max(0, axis_size(mesh_shape, first_axis) - 1)
    for hop in range(first_hops):
        token = logical_neighbor_copy(
            token,
            mesh=mesh,
            axis_name=axis_name,
            mesh_axis=first_axis,
            mesh_shape=mesh_shape,
            direction=direction,
            collective_id=int(collective_id) + id_offset + hop,
            memory_space=memory_space,
        )
        first_pieces.append(token)
    id_offset += first_hops
    # Stack after the global device axis. The array is sharded over axis 0, so
    # global shape becomes [device, first_arrival, ...payload] and each local
    # shard sees [first_arrival, ...payload]. That local shard layout matches
    # the pmap reference even though the global stack axis differs.
    partial = jnp.stack(first_pieces, axis=1)

    # Stage 2: move partial gathers around the second logical axis. The payload
    # of each second-stage hop is larger by a factor of first_axis_size.
    token = partial
    second_blocks = [token]
    second_hops = max(0, axis_size(mesh_shape, second_axis) - 1)
    for hop in range(second_hops):
        token = logical_neighbor_copy(
            token,
            mesh=mesh,
            axis_name=axis_name,
            mesh_axis=second_axis,
            mesh_shape=mesh_shape,
            direction=direction,
            collective_id=int(collective_id) + id_offset + hop,
            memory_space=memory_space,
        )
        second_blocks.append(token)
    # Again, stack after the global device axis. Global shape is
    # [device, second_arrival, first_arrival, ...payload]; each local shard sees
    # [second_arrival, first_arrival, ...payload].
    arrival = jnp.stack(second_blocks, axis=1)

    # The staged arrival grid is excellent for debugging but awkward for APIs.
    # Return canonical rank order so this matches ordinary all-gather semantics.
    return canonical_from_2d_arrival_sharded(
        arrival,
        mesh=mesh,
        axis_name=axis_name,
        mesh_shape=mesh_shape,
        axis_order=axis_order,
        direction=direction,
    )


def resolve_memory_space(pltpu: Any, memory_space_name: str) -> Any:
    """Return a Pallas TPU memory-space object with a useful error message.

    Lab 1's updated helper maps the course-facing name ``HBM`` to the Pallas
    memory space normally used for HBM-backed refs. Keep that mapping when the
    helper is available, and fall back to local handling for standalone review.
    """
    name = (memory_space_name or "HBM").strip().upper()
    lab1_resolver = getattr(lab1_single_hop, "resolve_memory_space", None)
    if lab1_resolver is not None:
        return lab1_resolver(name)

    if name == "HBM":
        # HBM must map to the non-windowed MemorySpace.ANY for the remote-DMA
        # hop; the explicit MemorySpace.HBM makes Mosaic reject the operand.
        # Mirrors lab1_single_hop.resolve_memory_space.
        from jax.experimental import pallas as pl

        return pl.ANY
    if hasattr(pltpu, name):
        return getattr(pltpu, name)
    valid = ["HBM", "VMEM", "SMEM"]
    raise ValueError(f"unknown Pallas TPU memory space {memory_space_name!r}; try one of {valid}")


def tile_shape_for_payload(
    *,
    payload_bytes: int,
    itemsize: int,
    tile_rows: int,
    min_cols: int,
) -> tuple[int, int, int]:
    """Return ``(rows, cols, actual_payload_bytes)`` for a whole-tile payload."""
    rows = max(1, int(tile_rows))
    cols = max(max(1, int(min_cols)), max(1, ceil_div(int(payload_bytes), itemsize * rows)))
    actual_payload_bytes = rows * cols * int(itemsize)
    return rows, cols, actual_payload_bytes


def _validate_stage_memory_space(
    *,
    memory_space_name: str,
    actual_payload_bytes: int,
    mesh_shape: tuple[int, int],
    axis_order: str,
) -> None:
    """Guard whole-tile VMEM experiments against the largest staged payload."""
    first_axis, _ = axis_sequence(axis_order)
    max_remote_bytes = actual_payload_bytes * axis_size(mesh_shape, first_axis)
    validate = getattr(lab1_single_hop, "validate_whole_tile_memory_space", None)
    if validate is not None:
        validate(
            memory_space_name=(memory_space_name or "HBM").strip().upper(),
            actual_payload_bytes=max_remote_bytes,
        )


def build_pmap_case(
    *,
    jax: Any,
    jnp: Any,
    devices: list[Any],
    axis_name: str,
    payload_bytes: int,
    dtype: Any,
    mesh_shape_name: str,
    axis_order: str,
    direction: str,
) -> MeshAllGatherCase:
    """Build input, expected output, and pmap function for staged 2D gather."""
    mesh_shape = parse_mesh_shape(mesh_shape_name, len(devices))
    axis_order = normalize_axis_order(axis_order)
    direction = normalize_direction(direction)
    itemsize = int(jnp.dtype(dtype).itemsize)
    elems = max(1, ceil_div(int(payload_bytes), itemsize))
    actual_payload_bytes = elems * itemsize

    # Rank-plus-position payloads keep the rank marker obvious at element 0,
    # while making every column meaningful for full-payload correctness checks.
    x = make_rank_position_tile(
        jnp,
        n_devices=len(devices),
        rows=1,
        cols=elems,
        dtype=dtype,
    )[:, 0, :]

    @functools.partial(jax.pmap, axis_name=axis_name)
    def fn(value):
        return staged_2d_all_gather_pmap(
            value,
            axis_name=axis_name,
            mesh_shape=mesh_shape,
            axis_order=axis_order,
            direction=direction,
        )

    return MeshAllGatherCase(
        x=x,
        fn=fn,
        expected_ranks=jnp.asarray(
            expected_tile_from_ranks(
                expected_rank_table(jnp, len(devices)),
                rows=1,
                cols=elems,
                use_position_offsets=True,
                dtype=dtype,
            ),
            dtype=jnp.float32,
        ),
        actual_payload_bytes=actual_payload_bytes,
        mesh_shape=mesh_shape,
        axis_order=axis_order,
        direction=direction,
        tile_rows=1,
        tile_cols=elems,
    )


def build_pallas_case(
    *,
    jax: Any,
    jnp: Any,
    devices: list[Any],
    axis_name: str,
    payload_bytes: int,
    dtype: Any,
    mesh_shape_name: str,
    axis_order: str,
    direction: str,
    tile_rows: int,
    min_cols: int,
    memory_space_name: str,
    collective_id: int,
) -> MeshAllGatherCase:
    """Build input, expected output, and Pallas function for staged 2D gather."""
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu

    mesh_shape = parse_mesh_shape(mesh_shape_name, len(devices))
    axis_order = normalize_axis_order(axis_order)
    direction = normalize_direction(direction)
    itemsize = int(jnp.dtype(dtype).itemsize)
    rows, cols, actual_payload_bytes = tile_shape_for_payload(
        payload_bytes=int(payload_bytes),
        itemsize=itemsize,
        tile_rows=tile_rows,
        min_cols=min_cols,
    )
    _validate_stage_memory_space(
        memory_space_name=memory_space_name,
        actual_payload_bytes=actual_payload_bytes,
        mesh_shape=mesh_shape,
        axis_order=axis_order,
    )

    # The JAX mesh is flat. Logical x/y coordinates are derived from the rank
    # position inside Pallas kernels.
    mesh = jax.sharding.Mesh(np.array(devices), (axis_name,))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(axis_name))

    x_host = make_rank_position_tile(
        jnp,
        n_devices=len(devices),
        rows=rows,
        cols=cols,
        dtype=dtype,
    )
    x = jax.device_put(x_host, sharding)
    memory_space = resolve_memory_space(pltpu, memory_space_name)

    def fn(value):
        return staged_2d_all_gather_pallas(
            value,
            mesh=mesh,
            axis_name=axis_name,
            mesh_shape=mesh_shape,
            axis_order=axis_order,
            direction=direction,
            collective_id=int(collective_id),
            memory_space=memory_space,
        )

    return MeshAllGatherCase(
        x=x,
        fn=jax.jit(fn),
        expected_ranks=jnp.asarray(
            expected_tile_from_ranks(
                expected_rank_table(jnp, len(devices)),
                rows=rows,
                cols=cols,
                use_position_offsets=True,
                dtype=dtype,
            ),
            dtype=jnp.float32,
        ),
        actual_payload_bytes=actual_payload_bytes,
        mesh_shape=mesh_shape,
        axis_order=axis_order,
        direction=direction,
        tile_rows=rows,
        tile_cols=cols,
    )


def check_result(jax: Any, jnp: Any, y: Any, expected: Any) -> bool:
    """Validate canonical rank order across the full output tile.

    New Lab 9 cases pass a full expected payload with shape
    ``[receiver, owner_rank, row, col]``. The pmap case returns
    ``[receiver, owner_rank, col]`` and is compared against the singleton-row
    slice of that expected payload.

    The fallback path still accepts the old compact rank table
    ``[receiver, owner_rank]`` so older notebooks and smoke tests fail gently.
    """
    y_host = jax.device_get(y)
    expected_host = jax.device_get(expected)

    if getattr(expected_host, "ndim", 0) == 4 and getattr(y_host, "ndim", 0) == 3:
        if expected_host.shape[2] != 1:
            return False
        expected_host = expected_host[:, :, 0, :]

    if getattr(expected_host, "ndim", 0) == getattr(y_host, "ndim", -1):
        if getattr(expected_host, "shape", None) != getattr(y_host, "shape", None):
            return False
        return bool(jnp.allclose(y_host, expected_host, rtol=1e-3, atol=1e-3))

    # Legacy compatibility: rank table only. This catches rank-order mistakes but
    # not partial-tile mistakes, so new Lab 9 cases should not use it.
    if getattr(expected_host, "ndim", 0) == 2:
        if getattr(y_host, "ndim", 0) == 3:
            got = y_host[:, :, 0]
        elif getattr(y_host, "ndim", 0) == 4:
            got = y_host[:, :, 0, 0]
        else:
            return False
        return bool(jnp.allclose(got, expected_host, rtol=1e-3, atol=1e-3))

    return False


def candidate_mesh_shapes(n_devices: int) -> list[str]:
    """Return candidate 2D logical mesh shapes for the current device count."""
    return [format_mesh_shape(shape) for shape in all_2d_factor_shapes(int(n_devices))]


def candidate_2d_mesh_shapes(n_devices: int) -> list[str]:
    """Alias with the explicit 2D name used by newer spec code."""
    return candidate_mesh_shapes(n_devices)


def physical_topology_hint(n_devices: int) -> str:
    """Return a short hardware-scope hint for common course device counts."""
    if n_devices == 4:
        return "4 local devices are commonly studied as a 2x2 logical mesh"
    if n_devices == 8:
        return (
            "8 devices can be viewed as 2x4 or 4x2 for this 2D lab; a physical "
            "v4-16-style slice is often discussed as 2x2x2 and belongs in the "
            "3D/multi-host follow-up"
        )
    return "choose logical 2D factors, then compare them against actual device coordinates"


def mesh_all_gather_byte_model(
    *,
    mesh_shape: tuple[int, int],
    axis_order: str,
    payload_bytes: int,
) -> dict[str, Any]:
    """Return byte accounting for a staged 2D all-gather."""
    first_axis, second_axis = axis_sequence(axis_order)
    first_size = axis_size(mesh_shape, first_axis)
    second_size = axis_size(mesh_shape, second_axis)
    n_devices = mesh_shape[0] * mesh_shape[1]
    payload = int(payload_bytes)
    first_stage = max(0, first_size - 1) * payload
    second_stage = max(0, second_size - 1) * first_size * payload
    staged_total = first_stage + second_stage
    flat_total = max(0, n_devices - 1) * payload
    return {
        "mesh_shape": format_mesh_shape(mesh_shape),
        "axis_order": normalize_axis_order(axis_order),
        "first_axis": first_axis,
        "second_axis": second_axis,
        "first_axis_size": first_size,
        "second_axis_size": second_size,
        "n_devices": n_devices,
        "payload_bytes_per_device": payload,
        "logical_result_bytes_per_device": n_devices * payload,
        "first_stage_hops": max(0, first_size - 1),
        "second_stage_hops": max(0, second_size - 1),
        "remote_copy_phases_per_device": max(0, first_size - 1) + max(0, second_size - 1),
        "first_stage_send_bytes_per_device": first_stage,
        "second_stage_send_bytes_per_device": second_stage,
        "staged_send_bytes_per_device": staged_total,
        "staged_recv_bytes_per_device": staged_total,
        "flat_ring_send_bytes_per_device": flat_total,
        "same_total_send_bytes_as_flat_ring": staged_total == flat_total,
    }


def staged_all_gather_plan(
    *,
    mesh_shape: tuple[int, int],
    axis_order: str,
    direction: str,
    base_collective_id: int,
    payload_bytes: int,
) -> list[dict[str, Any]]:
    """Return the ordered remote-copy phases used by the staged Pallas path."""
    direction = normalize_direction(direction)
    first_axis, second_axis = axis_sequence(axis_order)
    plan: list[dict[str, Any]] = []
    collective_id = int(base_collective_id)
    first_size = axis_size(mesh_shape, first_axis)
    second_size = axis_size(mesh_shape, second_axis)
    for hop in range(max(0, first_size - 1)):
        plan.append(
            {
                "stage": 1,
                "stage_name": f"gather_along_{first_axis}",
                "mesh_axis": first_axis,
                "hop": hop,
                "collective_id": collective_id,
                "message_bytes_per_device": int(payload_bytes),
                "direction": direction,
                "axis_groups": mesh_axis_groups(mesh_shape, first_axis),
            }
        )
        collective_id += 1
    for hop in range(max(0, second_size - 1)):
        plan.append(
            {
                "stage": 2,
                "stage_name": f"move_partial_gather_along_{second_axis}",
                "mesh_axis": second_axis,
                "hop": hop,
                "collective_id": collective_id,
                "message_bytes_per_device": int(payload_bytes) * first_size,
                "direction": direction,
                "axis_groups": mesh_axis_groups(mesh_shape, second_axis),
            }
        )
        collective_id += 1
    return plan


def _fallback_build_spec(
    template: dict[str, Any],
    *,
    args: Any,
    payload_bytes: int,
    n_devices: int,
) -> dict[str, Any]:
    """Small local replacement for ``lab_spec_utils.build_spec``."""
    return {
        **template,
        "payload_bytes": int(payload_bytes),
        "n_devices": int(n_devices),
        "args": {
            "lab9_mesh_shape": getattr(args, "lab9_mesh_shape", None),
            "lab9_axis_order": getattr(args, "lab9_axis_order", None),
            "neighbor_direction": getattr(args, "neighbor_direction", None),
            "pallas_collective_id": getattr(args, "pallas_collective_id", None),
            "pallas_memory_space": getattr(args, "pallas_memory_space", None),
        },
    }


def _fallback_render_markdown(spec: dict[str, Any]) -> str:
    """Small local replacement for ``lab_spec_utils.render_markdown``."""
    lines = [f"# {spec.get('title', 'Lab 9 Spec')}", ""]
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


def build_spec(*, jax: Any, args: Any, payload_bytes: int, n_devices: int) -> dict[str, Any]:
    """Build the Lab 9 course artifact consumed by the benchmark harness."""
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

    mesh_shape = parse_mesh_shape(getattr(args, "lab9_mesh_shape", "auto"), n_devices)
    axis_order = normalize_axis_order(getattr(args, "lab9_axis_order", "x_then_y"))
    direction = normalize_direction(getattr(args, "neighbor_direction", None) or "right")
    base_collective_id = int(getattr(args, "pallas_collective_id", 0) or 0)

    spec["candidate_2d_mesh_shapes"] = candidate_mesh_shapes(n_devices)
    spec["physical_topology_hint"] = physical_topology_hint(n_devices)
    spec["configured_mesh_shape"] = format_mesh_shape(mesh_shape)
    spec["configured_axis_order"] = axis_order
    spec["neighbor_direction"] = direction
    spec["rank_layout"] = rank_layout(mesh_shape)
    spec["axis_groups"] = all_axis_groups(mesh_shape)
    spec["axis_orders"] = ["x_then_y", "y_then_x"]
    spec["configured_stage_plan"] = staged_all_gather_plan(
        mesh_shape=mesh_shape,
        axis_order=axis_order,
        direction=direction,
        base_collective_id=base_collective_id,
        payload_bytes=int(payload_bytes),
    )
    spec["byte_model_by_axis_order"] = {
        order: mesh_all_gather_byte_model(
            mesh_shape=mesh_shape,
            axis_order=order,
            payload_bytes=int(payload_bytes),
        )
        for order in sorted(SUPPORTED_AXIS_ORDERS)
    }
    spec["configured_byte_model"] = mesh_all_gather_byte_model(
        mesh_shape=mesh_shape,
        axis_order=axis_order,
        payload_bytes=int(payload_bytes),
    )
    spec["arrival_rank_grid_preview"] = staged_arrival_rank_tables(
        mesh_shape=mesh_shape,
        axis_order=axis_order,
        direction=direction,
        max_receivers=min(int(n_devices), 8),
    )
    spec["canonical_lookup_preview"] = {
        f"receiver_{rank}": canonical_lookup_for_receiver(
            receiver_rank=rank,
            mesh_shape=mesh_shape,
            axis_order=axis_order,
            direction=direction,
        )
        for rank in range(min(int(n_devices), 4))
    }
    spec["stage_explanation"] = (
        "stage 1 gathers original shards along the first logical axis; stage 2 "
        "moves those partial gathers along the second axis; final output is "
        "reordered from arrival-grid layout into canonical flat-rank order"
    )
    spec["custom_collective_status"] = (
        "implemented as staged logical-neighbor Pallas remote-DMA hops over a "
        "flat JAX mesh; physical-coordinate-aware placement is deferred"
    )
    spec["student_checkpoint_questions"] = [
        "What rank layout does the configured mesh shape create?",
        "Which first-axis groups communicate during stage 1?",
        "Why does stage 2 send larger messages than stage 1?",
        "Why can x_then_y and y_then_x have the same bytes but different traces?",
        "What evidence would justify claiming one axis order matches topology better?",
    ]
    return spec


def render_markdown(spec: dict[str, Any]) -> str:
    """Render a spec artifact as Markdown."""
    if lab_spec_utils is not None:
        return lab_spec_utils.render_markdown(spec)
    return _fallback_render_markdown(spec)
