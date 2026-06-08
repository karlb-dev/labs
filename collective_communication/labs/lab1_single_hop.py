"""Lab 1: custom single-hop TPU neighbor copy.

This file is intentionally a *teaching kernel*, not a full benchmark harness.
The surrounding benchmark runner owns CLI parsing, run directories, logging,
CSV/JSONL output, plots, and profiler capture. This file owns one idea:

    every device pushes one local tile to one logical neighbor.

The communication pattern is a one-hop ring. For the default ``direction`` of
``"right"``:

    device 0 -> device 1
    device 1 -> device 2
    ...
    device N-1 -> device 0

Because every device sends to its right neighbor, device ``i`` receives the tile
from device ``i - 1 mod N``. We fill every input tile with the sender's rank so
correctness can be checked by looking at the rank value in each output tile.

Why this is Lab 1:
  * It is the smallest custom collective-communication primitive worth knowing.
  * It exposes the difference between an XLA-managed collective and a Pallas
    kernel that explicitly issues remote DMA.
  * It introduces the three concepts every later lab reuses: neighbor mapping,
    DMA semaphores, and an entry barrier.

Important course boundary:
  This is still a high-level, documented TPU programming substrate. We are not
  opening sockets or bypassing the TPU runtime. The useful low-level boundary
  for this course is JAX + shard_map + Pallas TPU remote DMA + semaphores.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any


# This is a deliberately conservative guardrail for the *whole-tile* version of
# the lab. Later labs should teach chunked VMEM staging. Lab 1 should not let a
# student accidentally ask for a 32 MiB VMEM tile and then receive a cryptic
# compiler/runtime failure from the silicon goblin under the floorboards.
WHOLE_TILE_VMEM_LIMIT_BYTES = 16 * 1024 * 1024

SUPPORTED_DIRECTIONS = frozenset({"right", "left"})
SUPPORTED_MEMORY_SPACE_NAMES = ("HBM", "VMEM", "ANY")


@dataclasses.dataclass(frozen=True)
class NeighborCopyCase:
    """Container returned to the benchmark harness.

    The harness expects a small object with:
      * ``x``: the input JAX array, already sharded across devices
      * ``fn``: a jitted function to benchmark
      * ``expected_ranks``: host-checkable rank pattern for correctness
      * ``actual_payload_bytes``: per-device bytes after tile-shape rounding

    The rest of the fields are metadata that make CSV rows and markdown
    summaries easier to interpret.
    """

    x: Any
    fn: Callable[[Any], Any]
    expected_ranks: Any
    actual_payload_bytes: int
    tile_rows: int
    tile_cols: int
    direction: str

    @property
    def note(self) -> str:
        """Short note suitable for benchmark tables."""
        return (
            f"direction={self.direction} "
            f"whole_tile={self.tile_rows}x{self.tile_cols}"
        )

    @property
    def elements_per_tile(self) -> int:
        """Number of elements copied by each device in this lab case."""
        return self.tile_rows * self.tile_cols


def _ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division with an explicit name for teaching."""
    return -(-numerator // denominator)


def normalize_direction(direction: str) -> str:
    """Validate and normalize the sender's direction around the ring.

    Direction is always from the sender's point of view:
      * ``right`` means device i writes to device i + 1 mod N.
      * ``left`` means device i writes to device i - 1 mod N.
    """
    normalized = str(direction).strip().lower()
    if normalized not in SUPPORTED_DIRECTIONS:
        allowed = ", ".join(sorted(SUPPORTED_DIRECTIONS))
        raise ValueError(f"unknown neighbor direction {direction!r}; use one of: {allowed}")
    return normalized


def normalize_memory_space_name(memory_space_name: str) -> str:
    """Validate the memory-space name used for the whole tile.

    ``HBM`` is the teaching default for Lab 1 because payload sweeps can be much
    larger than scoped VMEM. ``VMEM`` is useful for small experiments. ``ANY`` is
    accepted mostly for compatibility with older examples, where it usually
    means HBM for ordinary array refs.
    """
    normalized = str(memory_space_name).strip().upper()
    if normalized not in SUPPORTED_MEMORY_SPACE_NAMES:
        allowed = ", ".join(SUPPORTED_MEMORY_SPACE_NAMES)
        raise ValueError(
            f"unknown Pallas memory space {memory_space_name!r}; use one of: {allowed}"
        )
    return normalized


def resolve_memory_space(memory_space_name: str) -> Any:
    """Return the Pallas memory-space object for a validated name.

    This helper keeps deprecation and naming details away from the lab's main
    kernel. The benchmark CLI can pass a string, while ``pl.BlockSpec`` wants a
    memory-space object.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    normalized = normalize_memory_space_name(memory_space_name)
    if normalized == "HBM":
        # This kernel performs its own remote DMA between operand refs, so the
        # operands must stay HBM-resident and *non-windowed*. The right memory
        # space for that is ``MemorySpace.ANY`` (a.k.a. ``pl.ANY``): Pallas hands
        # the kernel the HBM ref directly instead of trying to copy a window into
        # VMEM. Passing the explicit ``MemorySpace.HBM`` here makes Mosaic treat
        # the operand as a window and fail with "Operand windows can only be
        # requested in VMEM or SMEM". The JAX distributed Pallas tutorial uses
        # ``pltpu.ANY`` for the same reason.
        return pl.ANY
    if normalized == "VMEM":
        return pltpu.VMEM
    if normalized == "ANY":
        return pl.ANY
    # ``normalize_memory_space_name`` should make this unreachable. Keep the
    # explicit raise so future edits fail loudly instead of wandering into a
    # confusing compiler error.
    raise AssertionError(f"unhandled memory space {normalized!r}")


def expected_neighbor_ranks(num_devices: int, direction: str) -> list[int]:
    """Expected output rank on each device after the one-hop copy.

    For ``right`` sends, device i receives from i - 1.
    For ``left`` sends, device i receives from i + 1.
    """
    direction = normalize_direction(direction)
    if num_devices <= 0:
        raise ValueError("num_devices must be positive")
    if direction == "right":
        return [(i - 1) % num_devices for i in range(num_devices)]
    return [(i + 1) % num_devices for i in range(num_devices)]


def mesh_neighbor(
    idx: Any,
    mesh: Any,
    axis_name: str,
    *,
    direction: str,
) -> tuple[Any, ...]:
    """Return the destination mesh index for a left/right neighbor.

    ``idx`` is usually ``lax.axis_index(axis_name)`` inside a ``shard_map``. It
    may be a traced scalar rather than a Python integer, so the arithmetic uses
    JAX primitives.

    The return value is a tuple because Pallas remote DMA can address devices by
    an N-dimensional mesh coordinate. In Lab 1 our mesh is normally 1D, so the
    tuple is just ``(neighbor_index,)``. Keeping the tuple form now prepares the
    same mental model for later 2D and 3D mesh labs.
    """
    from jax import lax

    direction = normalize_direction(direction)
    axis_names = tuple(mesh.axis_names)
    if axis_name not in axis_names:
        raise ValueError(f"axis_name {axis_name!r} not found in mesh axes {axis_names!r}")

    which_axis = axis_names.index(axis_name)
    axis_size = lax.axis_size(axis_name)

    # Start with this device's full coordinate in the logical mesh. On the
    # communication axis, use the provided ``idx``. On every other axis, keep the
    # current device's coordinate. Lab 1 uses a 1D mesh, but this function is
    # written so the idea survives into the 2D mesh lab.
    mesh_index = [
        idx if i == which_axis else lax.axis_index(other_axis_name)
        for i, other_axis_name in enumerate(axis_names)
    ]

    if direction == "right":
        neighbor_idx = lax.rem(idx + 1, axis_size)
    else:
        # Avoid Python's ``%`` here because ``idx`` can be a tracer. Adding
        # ``axis_size`` before subtracting one keeps the value non-negative.
        neighbor_idx = lax.rem(idx + axis_size - 1, axis_size)

    mesh_index[which_axis] = neighbor_idx
    return tuple(mesh_index)


def validate_whole_tile_memory_space(
    *,
    memory_space_name: str,
    actual_payload_bytes: int,
) -> None:
    """Reject VMEM whole-tile cases that exceed the Lab 1 guardrail.

    Lab 1 copies the whole local tile in one remote DMA. That is excellent for
    teaching the mechanism but it is not a good way to teach scoped VMEM at large
    sizes. Later labs should introduce chunking and HBM -> VMEM staging.
    """
    normalized = normalize_memory_space_name(memory_space_name)
    if normalized != "VMEM":
        return
    if actual_payload_bytes <= WHOLE_TILE_VMEM_LIMIT_BYTES:
        return

    mib = actual_payload_bytes / 1024 / 1024
    limit_mib = WHOLE_TILE_VMEM_LIMIT_BYTES / 1024 / 1024
    raise ValueError(
        "VMEM whole-tile remote copy would exceed the Lab 1 VMEM guardrail: "
        f"payload={mib:.1f} MiB, guardrail={limit_mib:.1f} MiB. "
        "Use --pallas-memory-space HBM for Lab 1 whole-tile payload sweeps, "
        "or reduce --sizes when intentionally studying small VMEM copies. "
        "Chunked VMEM staging belongs in a later lab."
    )


def neighbor_copy_kernel(
    x_ref,
    o_ref,
    send_sem,
    recv_sem,
    *,
    axis_name: str,
    mesh: Any,
    direction: str,
) -> None:
    """Pallas TPU kernel: remotely DMA this device's tile to one neighbor.

    Parameters are Pallas ``Ref`` objects, not ordinary JAX arrays:
      * ``x_ref`` is this device's local input tile.
      * ``o_ref`` is this device's local output tile, but when used as the
        destination of a remote copy it names the corresponding output tile on
        the target device.
      * ``send_sem`` tracks this device's send progress.
      * ``recv_sem`` tracks this device's receive progress.

    Every device runs this same program. That SPMD symmetry is the tiny ember
    that later grows into all-gather, reduce-scatter, and all-reduce.
    """
    import jax
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    direction = normalize_direction(direction)

    # ``lax.axis_index`` is the rank of this shard along the logical mesh axis
    # supplied by ``shard_map``. It is not necessarily the same thing as a host
    # process index or a physical chip coordinate. In Lab 1 it is simply our ring
    # position.
    my_axis_index = lax.axis_index(axis_name)
    dst_mesh_index = mesh_neighbor(
        my_axis_index,
        mesh,
        axis_name,
        direction=direction,
    )

    # Barrier semaphore: everyone announces that they have entered the kernel
    # before any device begins writing into another device's output buffer.
    #
    # Why signal ``dst_mesh_index`` and then wait locally?
    #   * This device will write into ``dst_mesh_index``.
    #   * The signal increments the barrier semaphore on that target device.
    #   * Since every device does this once, every device receives exactly one
    #     signal from its opposite neighbor and can then proceed.
    #
    # This is intentionally explicit even though Pallas may insert an automatic
    # entry barrier in some cases. The course goal is to make synchronization
    # visible rather than letting it vanish behind velvet compiler curtains.
    with jax.named_scope("lab1_entry_barrier"):
        barrier_sem = pltpu.get_barrier_semaphore()
        pltpu.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=dst_mesh_index,
            device_id_type=pl.DeviceIdType.MESH,
        )
        pltpu.semaphore_wait(barrier_sem, dec=1)

    # Remote DMA: create a descriptor, start it, and wait for both halves.
    #
    # The important model is push-only:
    #   * this device pushes ``x_ref`` to ``o_ref`` on ``dst_mesh_index``;
    #   * another device is simultaneously pushing its tile into *our* ``o_ref``;
    #   * all devices wait for their send and receive semaphores to drain.
    #
    # Lab 1 uses a single whole-tile transfer and no overlap. Later labs will
    # split the tile into chunks, add double buffers, and place computation
    # between ``start`` and the waits.
    with jax.named_scope("lab1_remote_dma"):
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
    """Return the public ``jax.shard_map`` when available, else a fallback.

    Older versions of JAX exposed ``shard_map`` through a module path. Keeping a
    tiny compatibility helper lets the teaching file stay useful across JAX
    versions without sprinkling private imports through the main code.
    """
    public_shard_map = getattr(jax, "shard_map", None)
    if public_shard_map is not None:
        return public_shard_map

    # Fallback for older environments. Try the public experimental module before
    # reaching for the private path. This is intentionally isolated so it is easy
    # to delete once all course environments have a public jax.shard_map.
    try:
        from jax.experimental import shard_map as shard_map_module
    except Exception:  # pragma: no cover - compatibility fallback only.
        from jax._src import shard_map as shard_map_module  # type: ignore[import-not-found]

    return shard_map_module.shard_map


def neighbor_copy(
    x: Any,
    *,
    mesh: Any,
    axis_name: str,
    direction: str,
    collective_id: int,
    memory_space: Any,
) -> Any:
    """Shard-mapped Pallas call for the single-hop neighbor copy.

    The global array ``x`` is sharded along ``axis_name``. Inside ``shard_map``,
    each device sees only its local tile. Inside ``pallas_call``, that tile is
    presented as a ``Ref`` so the kernel can issue remote DMA into another
    device's matching output ``Ref``.
    """
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    direction = normalize_direction(direction)
    shard_map = _get_shard_map(jax)
    partition = jax.sharding.PartitionSpec

    def local_copy(x_shard):
        """The per-shard function that ``shard_map`` runs on every device."""

        # ``x_shard`` is the local shard shape, not the global shape. In this
        # lab the global input is [num_devices, rows, cols] and is sharded on
        # the first dimension, so each device normally sees [1, rows, cols].
        # The leading singleton is the shard slot for this device; the course
        # metadata calls rows x cols the payload tile.
        out_shape = jax.ShapeDtypeStruct(x_shard.shape, x_shard.dtype)

        # ``pallas_call`` lowers the Python kernel above into a TPU kernel. The
        # grid spec says:
        #   * no scalar prefetch values are used;
        #   * allocate two DMA semaphores in scratch memory;
        #   * place the input and output refs in the requested memory space.
        #
        # ``collective_id`` names the barrier semaphore used by this communication
        # pattern. Reusing IDs for different patterns can create delightful little
        # race gremlins, so the benchmark runner should assign stable IDs per lab
        # and pattern.
        return pl.pallas_call(
            functools.partial(
                neighbor_copy_kernel,
                axis_name=axis_name,
                mesh=mesh,
                direction=direction,
            ),
            out_shape=out_shape,
            compiler_params=pltpu.CompilerParams(collective_id=collective_id),
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

    return shard_map(
        local_copy,
        mesh=mesh,
        in_specs=partition(axis_name),
        out_specs=partition(axis_name),
        check_vma=False,
    )(x)


def build_case(
    *,
    jax: Any,
    jnp: Any,
    devices: list[Any],
    axis_name: str,
    payload_bytes: int,
    dtype: Any,
    direction: str,
    tile_rows: int,
    min_cols: int,
    memory_space_name: str,
    collective_id: int,
) -> NeighborCopyCase:
    """Build input, expected output, and jitted function for Lab 1.

    This is the one function the benchmark harness needs to import. It converts
    CLI-ish parameters into a concrete sharded input and a jitted function.
    """
    import numpy as np

    direction = normalize_direction(direction)
    memory_space_name = normalize_memory_space_name(memory_space_name)

    num_devices = len(devices)
    if num_devices < 2:
        raise ValueError("Lab 1 needs at least two TPU devices to form a ring")

    # Build a 1D logical mesh from the provided devices. Later labs can replace
    # this with a 2D/3D mesh, but the teaching point here is one named axis and
    # one ring schedule.
    mesh = jax.sharding.Mesh(np.array(devices), (axis_name,))
    sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(axis_name),
    )

    # Payload handling:
    #   requested_payload_bytes is what the CLI asks for;
    #   actual_payload_bytes is the per-device whole-tile size we really copy.
    #
    # We use at least ``tile_rows`` rows and at least ``min_cols`` columns so TPU
    # tiling constraints do not surprise students immediately. The benchmark CSV
    # should record the actual payload so performance math stays honest.
    itemsize = int(jnp.dtype(dtype).itemsize)
    rows = max(1, int(tile_rows))
    cols_from_payload = _ceil_div(max(1, int(payload_bytes)), itemsize * rows)
    cols = max(max(1, int(min_cols)), cols_from_payload)
    actual_payload_bytes = rows * cols * itemsize

    validate_whole_tile_memory_space(
        memory_space_name=memory_space_name,
        actual_payload_bytes=actual_payload_bytes,
    )

    # Each device's tile is filled with that device's rank. This makes the copy
    # pattern visible in the output: after a right send, output tile i should be
    # filled with rank i - 1 mod N.
    ranks = jnp.arange(num_devices, dtype=jnp.float32).reshape(num_devices, 1, 1)
    x_host = jnp.broadcast_to(ranks, (num_devices, rows, cols)).astype(dtype)
    x = jax.device_put(x_host, sharding)

    memory_space = resolve_memory_space(memory_space_name)

    def fn(value):
        return neighbor_copy(
            value,
            mesh=mesh,
            axis_name=axis_name,
            direction=direction,
            collective_id=collective_id,
            memory_space=memory_space,
        )

    expected = expected_neighbor_ranks(num_devices, direction)

    return NeighborCopyCase(
        x=x,
        fn=jax.jit(fn),
        expected_ranks=jnp.array(expected, dtype=jnp.float32),
        actual_payload_bytes=actual_payload_bytes,
        tile_rows=rows,
        tile_cols=cols,
        direction=direction,
    )


def check_result(jax: Any, jnp: Any, y: Any, expected_ranks: Any) -> bool:
    """Validate that every output tile contains the expected neighbor rank.

    The old, minimal check looked only at ``y[:, 0, 0]``. That catches neighbor
    mapping mistakes, but it can miss partial-copy bugs. Lab 1 is small enough
    that we should check the full tile and make correctness non-negotiable.

    ``jnp`` is accepted for compatibility with the benchmark harness. The check
    uses NumPy after ``device_get`` so it does not accidentally schedule more TPU
    work while the harness is trying to measure communication.
    """
    del jnp  # The harness passes this in, but host-side NumPy is clearer here.

    import numpy as np

    y_host = np.asarray(jax.device_get(y))
    expected = np.asarray(jax.device_get(expected_ranks), dtype=np.float32)

    if y_host.ndim != 3:
        return False
    if y_host.shape[0] != expected.shape[0]:
        return False

    y_f32 = y_host.astype(np.float32)
    expected_tiles = expected.reshape((-1, 1, 1))

    # Remote DMA is a copy, not a floating-point reduction, so exact equality is
    # the right expectation for the rank-valued inputs used in this lab.
    return bool(np.array_equal(y_f32, expected_tiles + np.zeros_like(y_f32)))
