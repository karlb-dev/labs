"""Lab 6: reduce-scatter built from local ownership plus ring movement.

This file contains the *concept code* for Lab 6. The benchmark harness owns run
folders, CSV/JSONL rows, plotting, profile capture, and CLI parsing. This file
owns one idea:

    every device starts with one chunk for every possible output owner, then
    a ring schedule accumulates exactly the chunk owned by the receiver.

Why reduce-scatter?
-------------------
All-gather from Lab 5 increased ownership: every device started with one shard
and ended with all shards. Reduce-scatter is the opposite-looking move that ML
systems love: every device starts with a full vector of chunks, all devices
cooperate to reduce those chunks, and each device keeps only the chunk it owns.

For N devices, each input token is shaped like this on every source device:

    [chunk_0, chunk_1, ..., chunk_{N-1}]

After a full reduce-scatter, device j owns:

    sum_over_sources(source.chunk_j)

Implementation status
---------------------
The custom path here is intentionally a *whole-token teaching implementation*.
Each hop sends the full per-device chunk vector, while the receiver selects and
accumulates only its owner chunk. That is not the bandwidth-optimal algorithm.
It is a didactic stepping stone:

  * the ownership transform is easy to inspect;
  * every hop reuses the Lab 1 remote-DMA primitive;
  * every hop gets a distinct collective_id;
  * Lab 7 can reuse the output as phase 1 of all-reduce;
  * Lab 8 can replace whole-token movement with chunked and pipelined movement.

The bandwidth-optimal one-direction ring sends one chunk per hop, not the whole
chunk vector. This lab keeps that dragon in the next cave.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

try:  # Normal path inside the repository package.
    from labs import lab1_single_hop
except Exception:  # pragma: no cover - convenience for standalone file review.
    # Direct importlib loading does not always put the file's directory on
    # sys.path. Add it so adjacent lab files can be imported during standalone
    # teaching/review sessions.
    import sys
    from pathlib import Path

    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    import lab1_single_hop  # type: ignore[no-redef]

try:  # The benchmark repo provides this for course artifact rendering.
    from labs import lab_spec_utils
except Exception:  # pragma: no cover - fallback keeps this module importable alone.
    try:
        import lab_spec_utils  # type: ignore[no-redef]
    except Exception:
        lab_spec_utils = None  # type: ignore[assignment]


SUPPORTED_DIRECTIONS = frozenset({"right", "left"})


@dataclasses.dataclass(frozen=True)
class RingReduceScatterCase:
    """Container returned to the benchmark harness.

    The harness needs a sharded input ``x``, a jitted function ``fn``, and an
    expected result for correctness. The remaining fields make result rows and
    lab artifacts more explanatory.

    ``actual_payload_bytes`` is the full per-device input payload: all chunks
    owned by that device before the reduce-scatter. ``chunk_payload_bytes`` is
    the per-device output size after reduce-scatter.
    """

    x: Any
    fn: Callable[[Any], Any]
    expected_chunks: Any
    actual_payload_bytes: int
    chunk_payload_bytes: int
    tile_rows: int
    tile_cols: int
    n_chunks: int
    direction: str
    hops: int
    n_devices: int

    @property
    def note(self) -> str:
        """Short human-readable note for benchmark tables."""
        return (
            f"whole-token direction={self.direction} chunks={self.n_chunks} "
            f"hops={self.hops} devices={self.n_devices} "
            f"tile={self.tile_rows}x{self.tile_cols} "
            f"teaching_wire/dev={self.teaching_wire_bytes} "
            f"optimal_wire/dev={self.optimal_ring_wire_bytes}"
        )

    @property
    def num_devices(self) -> int:
        """Alias used by some benchmark/reporting code."""
        return self.n_devices

    @property
    def input_bytes_per_device(self) -> int:
        """Bytes each device starts with before reduce-scatter."""
        return self.actual_payload_bytes

    @property
    def output_bytes_per_device(self) -> int:
        """Bytes each device owns after reduce-scatter."""
        return self.chunk_payload_bytes

    @property
    def teaching_wire_bytes(self) -> int:
        """Per-device send bytes for this composed whole-token teaching path."""
        return self.actual_payload_bytes * self.hops

    @property
    def teaching_recv_bytes(self) -> int:
        """Per-device receive bytes for this composed whole-token teaching path."""
        return self.actual_payload_bytes * self.hops

    @property
    def optimal_ring_wire_bytes(self) -> int:
        """Per-device send bytes for the later one-chunk-per-hop ring plan."""
        return self.chunk_payload_bytes * self.hops

    @property
    def optimal_ring_recv_bytes(self) -> int:
        """Per-device receive bytes for the later one-chunk-per-hop ring plan."""
        return self.chunk_payload_bytes * self.hops

    @property
    def full_reduce_scatter_hops(self) -> int:
        """Number of hops needed to see every source exactly once."""
        return max(0, self.n_devices - 1)

    @property
    def is_full_reduce_scatter(self) -> bool:
        """Whether ``hops`` is exactly enough to reduce across all devices."""
        return self.hops == self.full_reduce_scatter_hops

    @property
    def teaching_overhead_factor(self) -> float:
        """How much more this teaching path sends than the one-chunk ring.

        For the default N chunks, the whole-token path sends N times as many
        bytes per hop as the optimized one-chunk-per-hop path. Returning a
        float keeps the property safe if someone experiments with zero-byte
        chunks in a smoke test.
        """
        if self.optimal_ring_wire_bytes == 0:
            return 0.0
        return self.teaching_wire_bytes / self.optimal_ring_wire_bytes

    def byte_model(self) -> dict[str, int | float | bool | str]:
        """Return benchmark-friendly byte accounting for this case."""
        return reduce_scatter_byte_model(
            n_devices=self.n_devices,
            full_payload_bytes=self.actual_payload_bytes,
            chunk_payload_bytes=self.chunk_payload_bytes,
            hops=self.hops,
            direction=self.direction,
        )


def normalize_direction(direction: str) -> str:
    """Validate and normalize the ring direction.

    Direction is from the sender's point of view, matching Lab 1:
      * ``right`` means device i sends to i + 1 mod N;
      * ``left`` means device i sends to i - 1 mod N.

    The receiver's source history is the inverse. For a right-moving ring,
    receiver r sees sources r, r-1, r-2, and so on.
    """
    normalized = str(direction).strip().lower()
    if normalized not in SUPPORTED_DIRECTIONS:
        allowed = ", ".join(sorted(SUPPORTED_DIRECTIONS))
        raise ValueError(f"unknown ring direction {direction!r}; use one of: {allowed}")
    return normalized


def normalize_hops(hops: int | str | None, n_devices: int | None = None) -> int:
    """Return a non-negative hop count.

    The full reduce-scatter setting is normally ``num_devices - 1``. Keeping
    this helper permissive is useful for experiments: ``hops=0`` should reduce
    only the local source, ``hops=1`` should show the first neighbor, and
    ``hops>N-1`` can demonstrate repeated sources and why production algorithms
    stop at exactly N-1 hops.

    Friendly values such as ``None``, ``-1``, ``"full"``, ``"all"``, and
    ``"n-1"`` mean ``num_devices - 1`` when ``n_devices`` is supplied.
    """
    if n_devices is not None and int(n_devices) < 2:
        raise ValueError("Lab 6 needs at least two devices to form a ring")

    if hops is None:
        if n_devices is None:
            raise ValueError("hops=None requires n_devices so the full ring depth is known")
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


def source_ranks_seen(
    *,
    receiver_rank: int,
    n_devices: int,
    hops: int,
    direction: str,
) -> list[int]:
    """Return the source ranks whose tokens receiver_rank has accumulated.

    Returned order is the temporal order of accumulation. Slot 0 is local.
    For a full run, this list contains every source rank exactly once. For a
    partial run, it is the partial reduce set. For an over-complete debug run,
    ranks may repeat.
    """
    direction = normalize_direction(direction)
    hops = normalize_hops(hops)
    if n_devices <= 0:
        raise ValueError("n_devices must be positive")
    if receiver_rank < 0 or receiver_rank >= n_devices:
        raise ValueError(
            f"receiver_rank must be in [0, {n_devices}); got {receiver_rank}"
        )

    if direction == "right":
        return [(receiver_rank - hop) % n_devices for hop in range(hops + 1)]
    return [(receiver_rank + hop) % n_devices for hop in range(hops + 1)]


def source_history_table(
    *,
    n_devices: int,
    hops: int,
    direction: str,
) -> list[list[int]]:
    """Return ``[receiver_rank][hop] -> source_rank`` for the ring schedule."""
    return [
        source_ranks_seen(
            receiver_rank=receiver_rank,
            n_devices=n_devices,
            hops=hops,
            direction=direction,
        )
        for receiver_rank in range(n_devices)
    ]


def expected_reduced_values(
    *,
    n_devices: int,
    hops: int,
    direction: str,
) -> list[float]:
    """Return expected scalar tile values for this reduce-scatter schedule.

    The input construction below uses this easy-to-read pattern:

        x[source, owner_chunk, row, col] = 10 * source + owner_chunk

    Device ``owner`` selects chunk ``owner`` from every token it sees. Therefore
    its expected output is:

        sum(10 * source + owner for source in source_ranks_seen(owner))

    For a full N-device reduce-scatter, this simplifies to:

        10 * N * (N - 1) / 2 + N * owner

    but this function intentionally handles partial hop counts too.
    """
    direction = normalize_direction(direction)
    hops = normalize_hops(hops)
    values: list[float] = []
    for owner in range(n_devices):
        seen_sources = source_ranks_seen(
            receiver_rank=owner,
            n_devices=n_devices,
            hops=hops,
            direction=direction,
        )
        values.append(float(sum(10.0 * src + owner for src in seen_sources)))
    return values


def expected_reduced_chunks(
    jnp: Any,
    n_devices: int,
    hops: int | str | None = None,
    direction: str = "right",
) -> Any:
    """JAX-array wrapper around ``expected_reduced_values``.

    The historical compact Lab 6 helper accepted only ``(jnp, n_devices)`` and
    always returned the full reduce-scatter answer. This expanded version keeps
    that default behavior while also supporting partial-hop experiments.
    """
    hops = normalize_hops(hops, n_devices=n_devices)
    return jnp.array(
        expected_reduced_values(
            n_devices=n_devices,
            hops=hops,
            direction=direction,
        ),
        dtype=jnp.float32,
    )


def reduce_scatter_byte_model(
    *,
    n_devices: int,
    full_payload_bytes: int,
    chunk_payload_bytes: int,
    hops: int,
    direction: str,
) -> dict[str, int | float | bool | str]:
    """Return simple byte accounting for Lab 6.

    This is algorithmic accounting for the teaching ring and the planned
    optimized ring. It is not a claim about how a built-in XLA collective is
    implemented internally.
    """
    direction = normalize_direction(direction)
    hops = normalize_hops(hops)
    n_devices = int(n_devices)
    full_payload_bytes = int(full_payload_bytes)
    chunk_payload_bytes = int(chunk_payload_bytes)
    if n_devices <= 0:
        raise ValueError("n_devices must be positive")
    if full_payload_bytes < 0:
        raise ValueError("full_payload_bytes must be non-negative")
    if chunk_payload_bytes < 0:
        raise ValueError("chunk_payload_bytes must be non-negative")

    teaching_send = hops * full_payload_bytes
    teaching_recv = hops * full_payload_bytes
    optimal_send = hops * chunk_payload_bytes
    optimal_recv = hops * chunk_payload_bytes
    overhead = 0.0 if optimal_send == 0 else teaching_send / optimal_send

    return {
        "direction": direction,
        "n_devices": n_devices,
        "hops": hops,
        "full_input_payload_bytes_per_device": full_payload_bytes,
        "output_chunk_bytes_per_device": chunk_payload_bytes,
        "output_fraction": f"1/{max(1, n_devices)}",
        "whole_token_teaching_send_bytes_per_device": teaching_send,
        "whole_token_teaching_recv_bytes_per_device": teaching_recv,
        "whole_token_teaching_total_send_bytes": n_devices * teaching_send,
        "optimal_one_chunk_send_bytes_per_device": optimal_send,
        "optimal_one_chunk_recv_bytes_per_device": optimal_recv,
        "optimal_one_chunk_total_send_bytes": n_devices * optimal_send,
        "teaching_overhead_factor_vs_one_chunk_ring": overhead,
        "full_reduce_scatter_hops": max(0, n_devices - 1),
        "is_full_reduce_scatter": hops == n_devices - 1,
    }


def _resolve_memory_space(memory_space_name: str) -> Any:
    """Resolve a Pallas memory-space name, reusing Lab 1 when possible."""
    resolver = getattr(lab1_single_hop, "resolve_memory_space", None)
    if resolver is not None:
        return resolver(memory_space_name)

    # Compatibility fallback for the earlier compact Lab 1 file.
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


def _validate_devices(devices: Sequence[Any]) -> int:
    """Return device count after checking the lab has a real ring."""
    n_devices = len(devices)
    if n_devices < 2:
        raise ValueError("Lab 6 needs at least two TPU devices to form a ring")
    return n_devices


LAB_SPEC: dict[str, Any] = {
    "lab": "lab6",
    "title": "Lab 6: Ring Reduce-Scatter",
    "goal": (
        "Learn the ownership-changing half of serious all-reduce. Each device "
        "starts with one chunk for every output owner and finishes owning one "
        "reduced chunk. The custom teaching path composes the Lab 1 Pallas "
        "remote-DMA hop with local owner-chunk selection and accumulation."
    ),
    "implemented_ops": [
        "`pmap_psum_scatter`: built-in reduce-scatter executable specification",
        "`pallas_ring_reduce_scatter`: whole-token ring built from Lab 1 hops",
        "`lab6_reduce_scatter_spec`: course artifact for the optimized ring plan",
    ],
    "deferred_ops": [
        "Replace whole-token movement with bandwidth-optimal one-chunk-per-hop movement",
        "Fuse chunk selection, local reduce, and remote movement into one Pallas kernel",
        "Track chunk ownership after every hop inside a single kernel",
        "Add integer exactness tests before floating-point tolerance tests",
        "Add bidirectional reduce-scatter after students understand ownership",
        "Reuse the optimized phase as Lab 7 all-reduce phase 1",
    ],
    "byte_model": [
        "optimal one-direction ring reduce-scatter sends `(n - 1) / n * full_payload_bytes` per device",
        "Lab 6 composed teaching version sends `(n - 1) * full_payload_bytes` per device for a full run",
        "output shard size is `full_payload_bytes / n` per device",
        "the teaching path is intentionally about ownership clarity, not bandwidth optimality",
    ],
    "pass_condition": [
        "built-in `pmap_psum_scatter` passes correctness",
        "composed Pallas reduce-scatter matches owner-chunk sums on TPU",
        "full tile contents match expected owner values, not only one scalar",
        "spec artifact defines chunk ownership and planned optimized phases",
    ],
    "artifacts": [
        "results.jsonl",
        "csvs/results.csv",
        "lab_artifacts/*lab6_reduce_scatter_spec*",
        "plots/latency_by_payload.png",
        "plots/bandwidth_by_payload.png",
        "traces/* when profiling is enabled",
    ],
    "suggested_experiments": [
        "Compare `hops=0`, `hops=1`, and `hops=num_devices-1`",
        "Flip `--neighbor-direction` and explain why full results match but partial sums change",
        "Compare whole-token teaching bytes with one-chunk-per-hop optimal bytes",
        "Use the source history table to debug any wrong owner result",
    ],
    "next_steps": [
        "Reuse this ownership model as phase 1 of Lab 7 all-reduce",
        "Follow Lab 8 by moving from whole-token copies to chunked and pipelined copies",
    ],
}


def owned_chunk(value: Any, *, mesh: Any, axis_name: str) -> Any:
    """Return the chunk whose owner is this device's axis index.

    Global conceptual input shape is:

        [source_device, owner_chunk, rows, cols]

    The first dimension is sharded over ``axis_name``. Inside ``shard_map``, the
    local shard keeps a leading dimension of size 1, so the chunk dimension is
    axis 1. Selecting ``owner = lax.axis_index(axis_name)`` means device j keeps
    chunk j from every token it receives.
    """
    import jax
    from jax import lax
    from jax._src import shard_map
    import jax.numpy as jnp

    partition = jax.sharding.PartitionSpec

    def local_owned_chunk(x_shard):
        # Expected local shape is [1, n_chunks, rows, cols]. The leading size-1
        # dimension is the local piece of the globally sharded source-device
        # axis. Axis 1 is the chunk owner axis.
        owner = lax.axis_index(axis_name)
        return jnp.take(x_shard, owner, axis=1).astype(jnp.float32)

    return shard_map.shard_map(
        local_owned_chunk,
        mesh=mesh,
        in_specs=partition(axis_name),
        out_specs=partition(axis_name),
        check_vma=False,
    )(value)


def ring_reduce_scatter(
    value: Any,
    *,
    mesh: Any,
    axis_name: str,
    direction: str,
    hops: int,
    collective_id: int,
    memory_space: Any,
) -> Any:
    """Compose Lab 1 remote-DMA hops into a whole-token reduce-scatter.

    ``value`` is the global sharded array with N owner chunks per device. The
    loop keeps a moving ``token``. On each hop, every device sends its current
    token to its neighbor using the Lab 1 remote-DMA primitive. After receiving
    the next token, the device selects exactly the chunk it owns and adds it to
    ``accum``.

    This is intentionally not a fused kernel. The point of Lab 6 is the owner
    transform:

        all source devices have chunk j  ->  device j owns reduced chunk j

    A unique ``collective_id`` per hop keeps the semaphore/barrier state for
    each remote-DMA phase separate. Lab 4 explains what goes sideways when that
    discipline slips on a banana peel.
    """
    direction = normalize_direction(direction)
    hops = normalize_hops(hops)

    token = value
    accum = owned_chunk(token, mesh=mesh, axis_name=axis_name)

    for hop in range(hops):
        token = lab1_single_hop.neighbor_copy(
            token,
            mesh=mesh,
            axis_name=axis_name,
            direction=direction,
            collective_id=int(collective_id) + hop,
            memory_space=memory_space,
        )
        accum = accum + owned_chunk(token, mesh=mesh, axis_name=axis_name)

    # Global shape: [num_devices, tile_rows, tile_cols]
    return accum


def make_reduce_scatter_input(
    *,
    jnp: Any,
    n_devices: int,
    n_chunks: int,
    rows: int,
    cols: int,
    dtype: Any,
) -> Any:
    """Construct the host-side input tensor used by Lab 6.

    The values are deliberately simple:

        x[source, owner_chunk, row, col] = 10 * source + owner_chunk

    That means the correct reduced output for owner chunk j can be computed on
    a napkin, which is exactly what a teaching lab wants.
    """
    src = jnp.arange(n_devices, dtype=jnp.float32).reshape(n_devices, 1, 1, 1)
    chunk = jnp.arange(n_chunks, dtype=jnp.float32).reshape(1, n_chunks, 1, 1)
    x_host = jnp.broadcast_to(
        src * 10.0 + chunk,
        (n_devices, n_chunks, rows, cols),
    )
    return x_host.astype(dtype)


def build_case(
    *,
    jax: Any,
    jnp: Any,
    devices: list[Any],
    axis_name: str,
    payload_bytes: int,
    dtype: Any,
    direction: str,
    hops: int | str | None,
    tile_rows: int,
    min_cols: int,
    memory_space_name: str,
    collective_id: int,
) -> RingReduceScatterCase:
    """Build input, expected output, and jitted function for Lab 6.

    ``payload_bytes`` means the full input payload per device: the whole vector
    of owner chunks. Lab 6 chooses ``n_chunks = n_devices`` so each output owner
    has exactly one chunk. The resulting per-device output is therefore roughly
    ``payload_bytes / n_devices`` after shape rounding.
    """
    import numpy as np

    direction = normalize_direction(direction)
    n_devices = _validate_devices(devices)
    hops = normalize_hops(hops, n_devices=n_devices)

    itemsize = int(jnp.dtype(dtype).itemsize)
    rows = max(1, int(tile_rows))
    n_chunks = n_devices

    # Round requested payload bytes up to a whole number of columns for a tensor
    # shaped [n_chunks, rows, cols] on each source device.
    cols = max(
        max(1, int(min_cols)),
        max(1, -(-int(payload_bytes) // (itemsize * rows * n_chunks))),
    )
    actual_payload_bytes = n_chunks * rows * cols * itemsize
    chunk_payload_bytes = rows * cols * itemsize

    # This teaching implementation moves the whole per-device token on each hop,
    # so the same whole-tile VMEM guardrail from Lab 1 applies.
    lab1_single_hop.validate_whole_tile_memory_space(
        memory_space_name=memory_space_name,
        actual_payload_bytes=actual_payload_bytes,
    )

    mesh = jax.sharding.Mesh(np.array(devices), (axis_name,))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(axis_name)
    )

    x_host = make_reduce_scatter_input(
        jnp=jnp,
        n_devices=n_devices,
        n_chunks=n_chunks,
        rows=rows,
        cols=cols,
        dtype=dtype,
    )
    x = jax.device_put(x_host, sharding)
    memory_space = _resolve_memory_space(memory_space_name)

    def ring_fn(value):
        return ring_reduce_scatter(
            value,
            mesh=mesh,
            axis_name=axis_name,
            direction=direction,
            hops=hops,
            collective_id=collective_id,
            memory_space=memory_space,
        )

    return RingReduceScatterCase(
        x=x,
        fn=jax.jit(ring_fn),
        expected_chunks=expected_reduced_chunks(
            jnp,
            n_devices,
            hops=hops,
            direction=direction,
        ),
        actual_payload_bytes=int(actual_payload_bytes),
        chunk_payload_bytes=int(chunk_payload_bytes),
        tile_rows=int(rows),
        tile_cols=int(cols),
        n_chunks=int(n_chunks),
        direction=direction,
        hops=int(hops),
        n_devices=int(n_devices),
    )


def observed_owner_scalars(jax: Any, y: Any) -> Any:
    """Return ``y[:, 0, 0]`` on the host for quick owner-table debugging."""
    return jax.device_get(y)[:, 0, 0]


def check_result(jax: Any, jnp: Any, y: Any, expected: Any) -> bool:
    """Validate the full reduce-scatter output.

    Older minimal checks looked only at ``y[:, 0, 0]``. That is useful for a
    quick owner-value glance, but reduce-scatter produces a whole output chunk.
    This checker verifies that every element of device j's output tile equals
    the expected reduced value for owner chunk j.
    """
    del jnp  # Host-side NumPy makes this a pure correctness check, not more TPU work.

    import numpy as np

    y_host = np.asarray(jax.device_get(y))
    expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)

    if y_host.ndim != 3:
        return False

    if expected_host.ndim == 1:
        if y_host.shape[0] != expected_host.shape[0]:
            return False
        expected_tiles = expected_host[:, None, None]
    elif expected_host.ndim == 3:
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
        "pallas_collective_id": getattr(args, "pallas_collective_id", None),
        "pallas_memory_space": getattr(args, "pallas_memory_space", None),
    }
    return spec


def build_spec(*, jax: Any, args: Any, payload_bytes: int, n_devices: int) -> dict[str, Any]:
    """Build the Lab 6 course artifact consumed by the benchmark harness."""
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
    raw_hops = (
        getattr(args, "ring_hops", None)
        if getattr(args, "ring_hops", None) is not None
        else getattr(args, "token_hops", None)
    )
    hops = normalize_hops(raw_hops, n_devices=n_devices)
    chunk_bytes = int(payload_bytes) // max(1, n_devices)

    spec["chunk_count"] = int(n_devices)
    spec["ring_hops"] = int(hops)
    spec["output_fraction"] = f"1/{max(1, n_devices)}"
    spec["neighbor_direction"] = direction
    spec["source_history"] = {
        "right": "device r accumulates sources [r, r-1, r-2, ...] mod N",
        "left": "device r accumulates sources [r, r+1, r+2, ...] mod N",
        "active": source_history_table(
            n_devices=n_devices,
            hops=hops,
            direction=direction,
        ),
    }
    spec["expected_owner_values_for_teaching_input"] = expected_reduced_values(
        n_devices=n_devices,
        hops=hops,
        direction=direction,
    )
    spec["byte_model_for_this_payload"] = reduce_scatter_byte_model(
        n_devices=n_devices,
        full_payload_bytes=payload_bytes,
        chunk_payload_bytes=chunk_bytes,
        hops=hops,
        direction=direction,
    )
    spec["custom_collective_status"] = (
        "implemented as whole-token local selection plus repeated Lab 1 Pallas "
        "hop kernels; optimized one-chunk-per-hop ring remains deferred"
    )
    spec["correctness_contract"] = (
        "The output shape is [num_devices, tile_rows, tile_cols]. Device j must "
        "own reduced chunk j: the sum of source chunk j over the source ranks "
        "seen by the ring schedule."
    )
    return spec


def _fallback_render_markdown(spec: dict[str, Any]) -> str:
    """Simple Markdown renderer used only when lab_spec_utils is unavailable."""
    lines = [f"# {spec.get('title', 'Lab 6 Ring Reduce-Scatter')}", ""]
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
    if "source_history" in spec:
        lines.extend(["## Source History", "", "```text"])
        for row in spec["source_history"].get("active", []):
            lines.append(str(row))
        lines.extend(["```", ""])
    if "byte_model_for_this_payload" in spec:
        lines.extend(["## Byte Model For This Payload", ""])
        for key, value in spec["byte_model_for_this_payload"].items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(spec: dict[str, Any]) -> str:
    """Render a Lab 6 spec artifact as Markdown."""
    if lab_spec_utils is not None:
        return lab_spec_utils.render_markdown(spec)
    return _fallback_render_markdown(spec)
