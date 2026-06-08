"""Lab 2: token-passing ring built from the Lab 1 remote-DMA hop.

This file intentionally contains only the concept code for Lab 2. The benchmark
harness owns command-line parsing, run directories, timing, profiling, CSV /
JSONL output, plots, and diagnostics.

The idea is deliberately simple:

  1. Lab 1 taught one custom communication hop:

       device i sends its local tile to device i + 1 mod N

  2. Lab 2 repeats that hop. Each device keeps a running sum of every token it
     has observed.

For the default case, ``hops = n_devices - 1``. That means each token moves far
enough around the ring that every device has seen every rank exactly once. On a
4-device slice, the expected sum on every device is therefore:

    0 + 1 + 2 + 3 = 6

This is still not a full collective like all-gather or all-reduce. It is the
smallest useful coordination computation: later hops depend on earlier hops, so
latency becomes visible in a way Lab 1's independent one-hop copy cannot show.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

# In the repo, this file lives in collective_comm_bench/labs/ and is imported as
# ``from labs import lab2_token_ring`` by the benchmark harness. The primary
# import below is therefore the repo import. The fallback makes the file easier
# to open in isolation during teaching or quick notebook experiments.
try:  # pragma: no cover - import style depends on how the lab is launched.
    from labs import lab1_single_hop
except ImportError:  # pragma: no cover
    import lab1_single_hop  # type: ignore[no-redef]


VALID_DIRECTIONS = frozenset({"left", "right"})


@dataclasses.dataclass(frozen=True)
class TokenRingCase:
    """A fully specified benchmark case for Lab 2.

    The runner expects each lab's ``build_case`` function to return a small
    object containing:

    * ``x``: the sharded input array placed on devices.
    * ``fn``: a jitted function to benchmark.
    * ``expected_sums``: one expected scalar per device.
    * payload and tile metadata for logging.

    The lab file should stay concept-sized. The harness is responsible for
    iteration counts, warmups, ``block_until_ready()``, profiles, and artifacts.
    """

    x: Any
    fn: Callable[[Any], Any]
    expected_sums: Any
    actual_payload_bytes: int
    tile_rows: int
    tile_cols: int
    direction: str
    hops: int

    @property
    def note(self) -> str:
        """Human-readable metadata that appears in benchmark output."""
        return (
            f"direction={self.direction} hops={self.hops} "
            f"tile={self.tile_rows}x{self.tile_cols}"
        )

    @property
    def remote_copies_per_device(self) -> int:
        """Number of one-hop remote copies issued by each device."""
        return self.hops

    @property
    def logical_send_bytes_per_device(self) -> int:
        """Simple byte model for this teaching implementation.

        Every hop sends the entire local token tile once. Since this lab is a
        dependency chain, the important first-order model is:

            bytes sent per device = hops * payload_bytes

        This is not a topology-level wire-byte model. It is a clear logical-byte
        model for the code students are reading.
        """
        return self.hops * self.actual_payload_bytes

    @property
    def logical_recv_bytes_per_device(self) -> int:
        """Symmetric receive-side byte model for the ring."""
        return self.hops * self.actual_payload_bytes


def _validate_direction(direction: str) -> str:
    """Return a normalized direction or raise an actionable error.

    Direction is from the sender's point of view:
      * ``right`` means device i sends to i + 1 mod N.
      * ``left`` means device i sends to i - 1 mod N.

    The receiver sees the opposite neighbor. A right send means device i
    receives from i - 1. That little convention is where many Lab 2 bugs nest.
    """
    normalized = str(direction).strip().lower()
    if normalized not in VALID_DIRECTIONS:
        allowed = ", ".join(sorted(VALID_DIRECTIONS))
        raise ValueError(f"direction must be one of {{{allowed}}}; got {direction!r}")
    return normalized


def normalize_hops(hops: int | str | None, n_devices: int) -> int:
    """Normalize the requested hop count.

    The benchmark runner normally passes ``n_devices - 1`` as the default. This
    helper also accepts ``None``, ``-1``, or friendly strings such as ``full`` so
    the lab is easy to call by hand.

    ``hops`` counts remote-copy phases, not ranks observed. A device observes
    its own initial rank before any communication, so it observes ``hops + 1``
    token values in total.
    """
    if n_devices < 2:
        raise ValueError("Lab 2 needs at least two devices to form a ring")

    if hops is None:
        return n_devices - 1

    if isinstance(hops, str):
        cleaned = hops.strip().lower()
        if cleaned in {"", "default", "full", "ring", "all"}:
            return n_devices - 1
        hops_int = int(cleaned)
    else:
        hops_int = int(hops)

    if hops_int == -1:
        return n_devices - 1
    if hops_int < 0:
        raise ValueError(
            f"token hops must be non-negative, or -1 for n_devices - 1; got {hops!r}"
        )
    return hops_int


def collective_ids_for_hops(base_collective_id: int, hops: int) -> tuple[int, ...]:
    """Return the collective IDs used by the Pallas hop sequence.

    Lab 2 intentionally uses a fresh ID for each hop:

        base_collective_id + hop

    This is pedagogical. Each Lab 1 neighbor copy contains an entry barrier, so
    assigning consecutive IDs makes the synchronization ownership obvious in the
    code and in traces. Later labs can discuss safe reuse and more compact
    protocols.
    """
    base = int(base_collective_id)
    if base < 0:
        raise ValueError(f"collective_id must be non-negative; got {base}")
    if hops < 0:
        raise ValueError(f"hops must be non-negative after normalization; got {hops}")
    return tuple(base + hop for hop in range(hops))


def ranks_seen_by_device(
    *,
    rank: int,
    n_devices: int,
    hops: int,
    direction: str,
) -> tuple[int, ...]:
    """Return the rank tokens observed by one device.

    Direction is the *send* direction used by Lab 1:

    * ``right`` means device ``i`` pushes its token to ``i + 1``. Therefore,
      after one hop, device ``i`` receives the token that used to live on
      ``i - 1``.
    * ``left`` means device ``i`` pushes its token to ``i - 1``. Therefore,
      after one hop, device ``i`` receives the token that used to live on
      ``i + 1``.

    Keeping this function in ordinary Python is useful for teaching because the
    correctness invariant is visible without reading JAX tracing machinery.
    """
    direction = _validate_direction(direction)
    if n_devices <= 0:
        raise ValueError(f"n_devices must be positive; got {n_devices}")
    if rank < 0 or rank >= n_devices:
        raise ValueError(f"rank {rank} is outside [0, {n_devices})")
    if hops < 0:
        raise ValueError(f"hops must be non-negative after normalization; got {hops}")

    if direction == "right":
        return tuple((rank - hop) % n_devices for hop in range(hops + 1))
    return tuple((rank + hop) % n_devices for hop in range(hops + 1))


def expected_seen_rank_paths(
    *,
    n_devices: int,
    hops: int,
    direction: str,
) -> tuple[tuple[int, ...], ...]:
    """Return the full expected token path for every device.

    This helper is not required by the benchmark harness. It exists because the
    path table is the easiest way to debug Lab 2 by eye.
    """
    hops = normalize_hops(hops, n_devices)
    return tuple(
        ranks_seen_by_device(
            rank=rank,
            n_devices=n_devices,
            hops=hops,
            direction=direction,
        )
        for rank in range(n_devices)
    )


def expected_sums(jnp: Any, n_devices: int, hops: int, direction: str) -> Any:
    """Expected scalar sum on each device after ``hops`` ring phases.

    For ``hops = n_devices - 1``, every device sees every rank once, so every
    entry is ``sum(range(n_devices))``. For larger hop counts, ranks repeat after
    wraparound. That case is still useful because it tests the schedule more
    than once around the ring.
    """
    direction = _validate_direction(direction)
    hops = normalize_hops(hops, n_devices)
    values = [sum(path) for path in expected_seen_rank_paths(
        n_devices=n_devices,
        hops=hops,
        direction=direction,
    )]
    return jnp.array(values, dtype=jnp.float32)


def resolve_memory_space(memory_space_name: str) -> Any:
    """Resolve a CLI memory-space name to the object expected by Pallas.

    Prefer Lab 1's resolver when available so Lab 1 and Lab 2 agree about names
    such as HBM, VMEM, and ANY. Fall back to a small local resolver for
    compatibility with the original minimal Lab 1 file.
    """
    lab1_resolver = getattr(lab1_single_hop, "resolve_memory_space", None)
    if lab1_resolver is not None:
        return lab1_resolver(memory_space_name)

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    normalized = str(memory_space_name).strip().upper()
    if normalized == "HBM":
        # The neighbor-copy kernel does its own remote DMA on these refs, so HBM
        # must map to the non-windowed MemorySpace.ANY. Using the explicit
        # MemorySpace.HBM makes Mosaic fail with "Operand windows can only be
        # requested in VMEM or SMEM". This mirrors lab1_single_hop.resolve_memory_space.
        return pl.ANY
    if normalized == "VMEM":
        return pltpu.VMEM
    if normalized == "ANY":
        return pl.ANY
    raise ValueError(f"unknown Pallas memory space {memory_space_name!r}")


def token_ring(
    value: Any,
    *,
    jnp: Any,
    mesh: Any,
    axis_name: str,
    direction: str,
    hops: int,
    collective_id: int,
    memory_space: Any,
) -> Any:
    """Run the token-passing ring using repeated Lab 1 neighbor copies.

    This function is intentionally written as a Python ``for`` loop over a static
    hop count. When ``ring_fn`` is jitted in ``build_case``, JAX traces and
    unrolls the loop. Each hop calls the Lab 1 Pallas neighbor-copy primitive.

    Pedagogical choice: keep one Pallas call per hop here. That makes the
    dependency chain and per-hop ``collective_id`` discipline obvious. Later
    labs can move the loop inside a larger Pallas kernel, introduce buffers, and
    overlap communication with local computation.
    """
    import jax

    direction = _validate_direction(direction)
    hops = int(hops)
    hop_collective_ids = collective_ids_for_hops(collective_id, hops)

    # The token itself keeps moving around the ring. It has the same dtype as
    # the input tile, usually bf16 or float32 depending on the payload sweep.
    token = value

    # The running sum is the actual Lab 2 computation. Accumulating in float32
    # makes the expected value simple and avoids dtype-specific surprises if the
    # input tile is bf16.
    seen_sum = token.astype(jnp.float32)

    for hop, hop_collective_id in enumerate(hop_collective_ids):
        # Named scopes are not required for correctness. They make traces easier
        # to read, which matters in a teaching lab.
        with jax.named_scope(f"lab2_token_hop_{hop:02d}"):
            token = lab1_single_hop.neighbor_copy(
                token,
                mesh=mesh,
                axis_name=axis_name,
                direction=direction,
                collective_id=hop_collective_id,
                memory_space=memory_space,
            )

        # Local computation after communication. This is tiny, but it turns the
        # data movement into a dependency chain: hop k + 1 cannot start from the
        # right value unless hop k produced the right token.
        seen_sum = seen_sum + token.astype(jnp.float32)

    return seen_sum


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
    tile_rows: int,
    min_cols: int,
    memory_space_name: str,
    collective_id: int,
) -> TokenRingCase:
    """Build input, expected output, and a jitted function for Lab 2.

    The benchmark harness calls this once per payload size and configuration.
    The shape and sharding construction are delegated to Lab 1 so the two labs
    use the same tile layout, memory-space guardrails, and rank-valued inputs.
    """
    import numpy as np

    n_devices = len(devices)
    if n_devices < 2:
        raise ValueError(
            "Lab 2 is a ring communication lab and needs at least two devices. "
            f"Got {n_devices}."
        )

    direction = _validate_direction(direction)
    hops = normalize_hops(hops, n_devices)
    collective_id = int(collective_id)
    if collective_id < 0:
        raise ValueError(f"collective_id must be non-negative; got {collective_id}")

    # Accept either HBM/VMEM or hbm/vmem on the command line. Lab 1 performs the
    # VMEM whole-tile capacity check, so this lab inherits the same safety rail.
    memory_space_name = str(memory_space_name).strip().upper()

    # Reuse Lab 1 to construct the input. We do not use ``base.fn`` here. We only
    # want its sharded rank-valued array and tile metadata.
    base = lab1_single_hop.build_case(
        jax=jax,
        jnp=jnp,
        devices=devices,
        axis_name=axis_name,
        payload_bytes=payload_bytes,
        dtype=dtype,
        direction=direction,
        tile_rows=tile_rows,
        min_cols=min_cols,
        memory_space_name=memory_space_name,
        collective_id=collective_id,
    )

    mesh = jax.sharding.Mesh(np.array(devices), (axis_name,))
    memory_space = resolve_memory_space(memory_space_name)

    def ring_fn(value):
        return token_ring(
            value,
            jnp=jnp,
            mesh=mesh,
            axis_name=axis_name,
            direction=direction,
            hops=hops,
            collective_id=collective_id,
            memory_space=memory_space,
        )

    return TokenRingCase(
        x=base.x,
        fn=jax.jit(ring_fn),
        expected_sums=expected_sums(jnp, n_devices, hops, direction),
        actual_payload_bytes=base.actual_payload_bytes,
        tile_rows=base.tile_rows,
        tile_cols=base.tile_cols,
        direction=direction,
        hops=hops,
    )


def check_result(jax: Any, jnp: Any, y: Any, expected: Any) -> bool:
    """Validate that every element of every output tile has the expected sum.

    The original Lab 2 checker only inspected ``y[:, 0, 0]``. That is enough to
    catch many neighbor-map errors, but the teaching invariant is stronger: this
    lab sends and accumulates whole tiles, so the whole tile should be correct.

    ``jnp`` is accepted for compatibility with the benchmark harness. The check
    uses NumPy after ``device_get`` so it does not accidentally schedule more TPU
    work while the harness is measuring communication.
    """
    del jnp

    import numpy as np

    y_host = np.asarray(jax.device_get(y))
    expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)

    if y_host.ndim < 1:
        return False
    if y_host.shape[0] != expected_host.shape[0]:
        return False

    expected_shape = (expected_host.shape[0],) + (1,) * (y_host.ndim - 1)
    expected_tiles = expected_host.reshape(expected_shape) + np.zeros_like(
        y_host, dtype=np.float32
    )

    # Values are sums of small integer ranks accumulated in float32. Exact
    # equality is the right invariant for this lab. Future labs that reduce
    # arbitrary floating-point payloads should switch to a tolerance.
    return bool(np.array_equal(y_host.astype(np.float32), expected_tiles))


def observed_tile_scalars(jax: Any, y: Any) -> Any:
    """Return the first scalar from each device's tile for compact debugging.

    This helper is not required by the harness. It is useful in a notebook or
    REPL when a student wants to print the observed per-device scalar pattern
    without dumping every element of every tile.
    """
    y_host = jax.device_get(y)
    if y_host.ndim == 0:
        return y_host
    index = (slice(None),) + (0,) * (y_host.ndim - 1)
    return y_host[index]
