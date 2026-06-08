"""Lab 5: ring all-gather built from repeated single-hop movement.

This file contains the *concept code* for Lab 5. The benchmark harness owns run
folders, CSV/JSONL rows, plotting, profile capture, and CLI parsing. This file
owns one idea:

    build an arrival-order all-gather by repeatedly applying the Lab 1
    single-hop neighbor-copy primitive.

Why arrival order?
------------------
For a right-moving ring, device r sees shards in this order:

    [r, r - 1, r - 2, ...] mod N

That is not the same layout as a canonical all-gather sorted by source rank:

    [0, 1, 2, ...]

Arrival order is excellent for teaching because it exposes the communication
schedule. Later labs can add a canonical reorder, chunking, and a fused Pallas
kernel. First, students should be able to look at a rank table and say exactly
which shard traveled through which hop.

Implementation status
---------------------
The custom path here is intentionally *not* one fused Pallas kernel. It composes
``hops`` calls to ``lab1_single_hop.neighbor_copy``. That means:

  * every hop has a clear input token and output token;
  * every hop gets its own ``collective_id``;
  * profiler traces expose the cost of the composed teaching implementation;
  * the lab remains debuggable before optimization goblinry enters the room.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
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
class RingAllGatherCase:
    """Container returned to the benchmark harness.

    The harness needs a sharded input ``x``, a jitted function ``fn``, and an
    expected result for correctness. The remaining fields make result rows and
    lab artifacts more explanatory.
    """

    x: Any
    fn: Callable[[Any], Any]
    expected_arrivals: Any
    actual_payload_bytes: int
    tile_rows: int
    tile_cols: int
    direction: str
    hops: int
    n_devices: int

    @property
    def note(self) -> str:
        """Short human-readable note for benchmark tables."""
        return (
            f"arrival-order direction={self.direction} hops={self.hops} "
            f"devices={self.n_devices} tile={self.tile_rows}x{self.tile_cols}"
        )

    @property
    def arrivals_per_device(self) -> int:
        """Number of shard slots each receiver owns after this ring schedule."""
        return self.hops + 1

    @property
    def num_devices(self) -> int:
        """Alias used by some benchmark/reporting code."""
        return self.n_devices

    @property
    def logical_result_bytes_per_device(self) -> int:
        """Bytes of output data held by each device after the ring schedule."""
        return self.arrivals_per_device * self.actual_payload_bytes

    @property
    def logical_send_bytes_per_device(self) -> int:
        """Alias for the per-device logical ring send byte count."""
        return self.ring_send_bytes_per_device

    @property
    def logical_recv_bytes_per_device(self) -> int:
        """Alias for the per-device logical ring receive byte count."""
        return self.ring_recv_bytes_per_device

    @property
    def total_ring_send_bytes(self) -> int:
        """Total logical send bytes across all devices in this schedule."""
        return self.n_devices * self.ring_send_bytes_per_device

    @property
    def ring_send_bytes_per_device(self) -> int:
        """Estimated one-direction ring traffic sent by each device."""
        return self.hops * self.actual_payload_bytes

    @property
    def ring_recv_bytes_per_device(self) -> int:
        """Estimated one-direction ring traffic received by each device."""
        return self.hops * self.actual_payload_bytes

    @property
    def is_full_all_gather(self) -> bool:
        """Whether ``hops`` is exactly enough for every device to see all ranks."""
        return self.hops == self.n_devices - 1

    def byte_model(self) -> dict[str, int | bool | str]:
        """Return benchmark-friendly byte accounting for this case."""
        return ring_byte_model(
            n_devices=self.n_devices,
            shard_bytes=self.actual_payload_bytes,
            hops=self.hops,
            direction=self.direction,
        )


def normalize_direction(direction: str) -> str:
    """Validate and normalize the ring direction.

    Direction is from the sender's point of view, matching Lab 1:
      * ``right`` means device i sends to i + 1 mod N;
      * ``left`` means device i sends to i - 1 mod N.

    A receiver's arrival table is the inverse of the sender direction. For a
    right-moving ring, receiver r sees r, then r-1, then r-2, and so on.
    """
    normalized = str(direction).strip().lower()
    if normalized not in SUPPORTED_DIRECTIONS:
        allowed = ", ".join(sorted(SUPPORTED_DIRECTIONS))
        raise ValueError(f"unknown ring direction {direction!r}; use one of: {allowed}")
    return normalized


def normalize_hops(hops: int | str | None, n_devices: int | None = None) -> int:
    """Return a non-negative hop count.

    The full all-gather setting is normally ``num_devices - 1``. Keeping this
    helper permissive is useful for experiments: ``hops=0`` should return only
    the local shard, ``hops=1`` should show the first neighbor, and ``hops>N-1``
    can be used to demonstrate duplicate arrivals.

    Friendly values such as ``None``, ``-1``, ``"full"``, ``"all"``, and
    ``"n-1"`` mean ``num_devices - 1`` when ``n_devices`` is supplied.
    """
    if n_devices is not None and int(n_devices) < 2:
        raise ValueError("Lab 5 needs at least two devices to form a ring")

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


def expected_arrival_ranks(
    *,
    n_devices: int,
    hops: int,
    direction: str,
) -> list[list[int]]:
    """Pure Python rank table for the arrival-order ring schedule.

    Returned shape is ``[receiver_rank][arrival_slot]``.

    For a right-moving ring, sender i writes to i+1, so receiver r sees sources
    r, r-1, r-2, ... . For a left-moving ring, receiver r sees r, r+1, r+2, ... .
    """
    direction = normalize_direction(direction)
    hops = normalize_hops(hops)
    if n_devices <= 0:
        raise ValueError("n_devices must be positive")

    rows: list[list[int]] = []
    for receiver_rank in range(n_devices):
        if direction == "right":
            rows.append(
                [(receiver_rank - hop) % n_devices for hop in range(hops + 1)]
            )
        else:
            rows.append(
                [(receiver_rank + hop) % n_devices for hop in range(hops + 1)]
            )
    return rows


def expected_arrivals(jnp: Any, n_devices: int, hops: int, direction: str) -> Any:
    """JAX-array wrapper around ``expected_arrival_ranks``.

    The benchmark harness stores this object on the case and passes it back to
    ``check_result``. The table contains source ranks, not full expected tiles.
    ``check_result`` broadcasts the source ranks across the tile dimensions.
    """
    return jnp.array(
        expected_arrival_ranks(
            n_devices=n_devices,
            hops=hops,
            direction=direction,
        ),
        dtype=jnp.float32,
    )


def arrival_slot_for_source(
    *,
    receiver_rank: int,
    source_rank: int,
    n_devices: int,
    direction: str,
) -> int:
    """Return which arrival slot contains ``source_rank`` on ``receiver_rank``.

    This is an optional teaching helper used to explain how arrival order can be
    transformed into canonical source-rank order.
    """
    direction = normalize_direction(direction)
    if direction == "right":
        return (receiver_rank - source_rank) % n_devices
    return (source_rank - receiver_rank) % n_devices


def canonicalize_arrival_order(arrivals: Any, *, direction: str) -> Any:
    """Reorder arrival slots into canonical source-rank order.

    Parameters
    ----------
    arrivals:
        A JAX array shaped ``[num_devices, arrivals_per_device, ...]``. The
        second axis is arrival order: slot 0 is local, slot 1 is one hop away,
        and so on.
    direction:
        The ring direction used to produce ``arrivals``.

    Returns
    -------
    A JAX array shaped ``[num_devices, num_devices, ...]`` where the second axis
    is source rank order: slot 0 contains source 0, slot 1 contains source 1,
    etc. This requires at least ``num_devices`` arrival slots, so it is only
    valid for a full all-gather or for an over-complete debug run.

    This function is not used by the default benchmark path. It is here so
    students can experiment with the difference between "the order data arrived"
    and "the canonical order an API usually returns".
    """
    import jax.numpy as jnp

    direction = normalize_direction(direction)
    num_devices = int(arrivals.shape[0])
    arrivals_per_device = int(arrivals.shape[1])
    if arrivals_per_device < num_devices:
        raise ValueError(
            "canonicalize_arrival_order needs at least num_devices arrival slots; "
            f"got arrivals_per_device={arrivals_per_device}, num_devices={num_devices}"
        )

    receiver_rows = []
    for receiver_rank in range(num_devices):
        source_slots = []
        for source_rank in range(num_devices):
            slot = arrival_slot_for_source(
                receiver_rank=receiver_rank,
                source_rank=source_rank,
                n_devices=num_devices,
                direction=direction,
            )
            source_slots.append(arrivals[receiver_rank, slot, ...])
        receiver_rows.append(jnp.stack(source_slots, axis=0))
    return jnp.stack(receiver_rows, axis=0)


def ring_byte_model(
    *,
    n_devices: int,
    shard_bytes: int,
    hops: int,
    direction: str,
) -> dict[str, int | bool | str]:
    """Return simple byte accounting for a one-direction ring all-gather.

    This is algorithmic accounting for the teaching ring, not a claim about how
    a built-in XLA collective is implemented under the hood.
    """
    direction = normalize_direction(direction)
    hops = normalize_hops(hops)
    n_devices = int(n_devices)
    shard_bytes = int(shard_bytes)
    if n_devices <= 0:
        raise ValueError("n_devices must be positive")
    if shard_bytes < 0:
        raise ValueError("shard_bytes must be non-negative")

    return {
        "direction": direction,
        "n_devices": n_devices,
        "hops": hops,
        "shard_bytes_per_device": shard_bytes,
        "arrivals_per_device": hops + 1,
        "logical_result_bytes_per_device": (hops + 1) * shard_bytes,
        "ring_send_bytes_per_device": hops * shard_bytes,
        "ring_recv_bytes_per_device": hops * shard_bytes,
        "total_ring_send_bytes": n_devices * hops * shard_bytes,
        "full_all_gather_hops": max(0, n_devices - 1),
        "is_full_all_gather": hops == n_devices - 1,
    }


def observed_arrival_scalars(jax: Any, y: Any) -> Any:
    """Return ``y[:, :, 0, 0]`` on the host for quick rank-table debugging."""
    return jax.device_get(y)[:, :, 0, 0]


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


LAB_SPEC: dict[str, Any] = {
    "lab": "lab5",
    "title": "Lab 5: Ring All-Gather",
    "goal": (
        "Build the first full custom collective from repeated ring movement. "
        "The happy path starts with a pmap ring schedule, then composes the "
        "Lab 1 Pallas remote-DMA hop into an arrival-order all-gather."
    ),
    "implemented_ops": [
        "`pmap_ring_all_gather`: repeated `lax.ppermute` in arrival order",
        "`pmap_all_gather`: built-in executable specification",
        "`pallas_ring_all_gather`: repeated Lab 1 Pallas remote-DMA hop",
        "`pallas_all_gather`: installed JAX Pallas TPU bridge when on TPU",
    ],
    "deferred_ops": [
        "Fuse the full ring schedule into one custom Pallas kernel",
        "Record estimated wire bytes separately from logical bytes in every row",
        "Add canonical rank-order output layout as a normal benchmark operation",
        "Chunk payloads so VMEM staging can overlap copy and local work",
        "Add double buffering and capacity semaphores",
    ],
    "byte_model": [
        "one-direction ring all-gather sends `(n - 1) * shard_bytes` per device",
        "one-direction ring all-gather receives `(n - 1) * shard_bytes` per device",
        "logical result size is `n * shard_bytes` per device for the full gather",
    ],
    "pass_condition": [
        "arrival-order pmap ring contains every rank exactly once on every device",
        "arrival-order Pallas ring matches the same rank schedule on TPU",
        "full tile contents match the expected source rank, not only one scalar",
        "built-in `pmap_all_gather` passes for the same payload sizes",
        "spec artifact records the remaining fused-kernel work",
    ],
    "artifacts": [
        "results.jsonl",
        "csvs/results.csv",
        "lab_artifacts/*lab5_ring_all_gather_spec*",
        "plots/latency_by_payload.png",
        "plots/bandwidth_by_payload.png",
        "traces/* when profiling is enabled",
    ],
    "suggested_experiments": [
        "Compare `hops=0`, `hops=1`, and `hops=num_devices-1`",
        "Flip `--neighbor-direction` and explain the rank table",
        "Compare composed Pallas hop latency with built-in all-gather",
        "Use `canonicalize_arrival_order` to transform arrival order into rank order",
    ],
    "next_steps": [
        "Introduce chunk ownership before reduce-scatter in Lab 6",
        "Reuse the Lab 5 arrival table as the ownership table for moving partial sums",
    ],
}


def ring_all_gather(
    value: Any,
    *,
    mesh: Any,
    axis_name: str,
    direction: str,
    hops: int,
    collective_id: int,
    memory_space: Any,
) -> Any:
    """Compose the Lab 1 single-hop remote DMA into an all-gather ring.

    ``value`` is the global sharded array. Each call to
    ``lab1_single_hop.neighbor_copy`` performs one synchronized remote-DMA hop
    around the logical ring and returns the newly received shard.

    We append each token to ``pieces`` before forwarding it again. After H hops,
    every device owns H+1 pieces in arrival order:

        [local, one-hop-away, two-hops-away, ...]

    A unique ``collective_id`` per hop is a simple, visible way to avoid
    accidental semaphore/barrier aliasing between phases. Lab 4 is the bug zoo;
    Lab 5 should be the tidy greenhouse.
    """
    import jax.numpy as jnp

    direction = normalize_direction(direction)
    hops = normalize_hops(hops)

    token = value
    pieces = [token]

    for hop in range(hops):
        token = lab1_single_hop.neighbor_copy(
            token,
            mesh=mesh,
            axis_name=axis_name,
            direction=direction,
            collective_id=int(collective_id) + hop,
            memory_space=memory_space,
        )
        pieces.append(token)

    # Global shape: [num_devices, hops + 1, tile_rows, tile_cols]
    return jnp.stack(pieces, axis=1)


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
) -> RingAllGatherCase:
    """Build input, expected output, and jitted function for Lab 5.

    The input construction is delegated to Lab 1 so the rank-valued tile pattern
    and the whole-tile memory-space guardrail stay consistent across labs. Lab 5
    then wraps the Lab 1 hop in a repeated ring schedule.
    """
    import numpy as np

    direction = normalize_direction(direction)
    n_devices = len(devices)
    if n_devices < 2:
        raise ValueError("Lab 5 needs at least two TPU devices to form a ring")
    # A None hop count means the real all-gather depth: N - 1 phases.
    # Smaller explicit values remain useful for partial-ring debugging.
    hops = normalize_hops(hops, n_devices=n_devices)

    # Reuse Lab 1's sharded input and payload-shape logic. That keeps the first
    # five labs aligned: a "payload" is always one local shard, rounded to a
    # TPU-friendly whole-tile shape by the same helper.
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
    memory_space = _resolve_memory_space(memory_space_name)

    def ring_fn(value):
        return ring_all_gather(
            value,
            mesh=mesh,
            axis_name=axis_name,
            direction=direction,
            hops=hops,
            collective_id=collective_id,
            memory_space=memory_space,
        )

    return RingAllGatherCase(
        x=base.x,
        fn=jax.jit(ring_fn),
        expected_arrivals=expected_arrivals(jnp, n_devices, hops, direction),
        actual_payload_bytes=int(base.actual_payload_bytes),
        tile_rows=int(base.tile_rows),
        tile_cols=int(base.tile_cols),
        direction=direction,
        hops=hops,
        n_devices=n_devices,
    )


def check_result(jax: Any, jnp: Any, y: Any, expected: Any) -> bool:
    """Validate the full arrival-order all-gather output.

    Older minimal checks looked only at ``y[:, :, 0, 0]``. That is useful for a
    quick rank-table glance, but an all-gather copies whole shards. This checker
    verifies that every element of every received tile equals the expected
    source rank.

    The input tiles for this lab are rank-filled, so broadcasting the expected
    source-rank table over the tile dimensions produces the full expected output.
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
    return bool(np.allclose(y_f32, expected_tiles + np.zeros_like(y_f32), rtol=1e-3, atol=1e-3))


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
        "pallas_collective_id": getattr(args, "pallas_collective_id", None),
        "pallas_memory_space": getattr(args, "pallas_memory_space", None),
    }
    return spec


def build_spec(*, jax: Any, args: Any, payload_bytes: int, n_devices: int) -> dict[str, Any]:
    """Build the Lab 5 course artifact consumed by the benchmark harness."""
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
    # The harness may pass a hop flag from Lab 2. If absent, the all-gather
    # default is N-1 hops.
    raw_hops = (
        getattr(args, "ring_hops", None)
        if getattr(args, "ring_hops", None) is not None
        else getattr(args, "token_hops", None)
    )
    ring_hops = normalize_hops(raw_hops, n_devices=n_devices)

    spec["ring_hops"] = ring_hops
    spec["neighbor_direction"] = direction
    spec["arrival_order"] = {
        "right": "device r sees [r, r-1, r-2, ...] mod N",
        "left": "device r sees [r, r+1, r+2, ...] mod N",
        "active": expected_arrival_ranks(
            n_devices=n_devices,
            hops=ring_hops,
            direction=direction,
        ),
    }
    spec["byte_model_for_this_payload"] = ring_byte_model(
        n_devices=n_devices,
        shard_bytes=payload_bytes,
        hops=ring_hops,
        direction=direction,
    )
    spec["custom_collective_status"] = (
        "implemented as a composed sequence of Lab 1 Pallas hop kernels; "
        "a single fused Pallas kernel remains a later optimization"
    )
    spec["correctness_contract"] = (
        "The output shape is [num_devices, hops + 1, tile_rows, tile_cols]. "
        "For the default full gather, each receiver should see every source "
        "rank exactly once in arrival order."
    )
    return spec


def _fallback_render_markdown(spec: dict[str, Any]) -> str:
    """Simple markdown renderer used only when lab_spec_utils is unavailable."""
    lines = [f"# {spec.get('title', 'Lab 5 Ring All-Gather')}", ""]
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
    if "byte_model_for_this_payload" in spec:
        lines.extend(["## Byte Model For This Payload", ""])
        for key, value in spec["byte_model_for_this_payload"].items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(spec: dict[str, Any]) -> str:
    """Render a Lab 5 spec artifact as Markdown."""
    if lab_spec_utils is not None:
        return lab_spec_utils.render_markdown(spec)
    return _fallback_render_markdown(spec)
