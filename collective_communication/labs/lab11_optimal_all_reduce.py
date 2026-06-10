"""Lab 11: bandwidth-optimal ring all-reduce (reduce-scatter + all-gather).

Lab 7 composed an all-reduce out of whole-token ring phases and Lab 8 fused the
whole-token ring into one overlapped Pallas program. Both labs end with the same
honest confession: the whole-token ring moves about ``N/2`` times the bytes of an
optimal all-reduce, so no amount of overlap lets it beat ``lax.psum`` at large
payloads. Lab 8's docstring promises that "a bandwidth-optimal chunk-per-hop
ring is a natural follow-on lab". This is that lab.

The trick is to stop circulating full tokens and circulate shards instead:

    reduce-scatter
        Split each device's payload into ``N`` shards. For ``N - 1`` steps, send
        the partial sum you hold to your ring neighbor and add the matching local
        shard to the partial you just received. After the last step every device
        holds exactly one *fully reduced* shard.

    all-gather
        For another ``N - 1`` steps, circulate the finished shards so every
        device assembles the complete reduced payload.

Each of the ``2 * (N - 1)`` steps sends one ``B / N`` shard per device, so the
standard full-duplex bandwidth term is ``2 * (N - 1) / N * B`` -- the same
volume used by the harness for ``lax.psum``. Each device receives the same
amount; the benchmark reports the send-side/full-duplex term rather than adding
ingress and egress a second time. The naive whole-token ring sends
``(N - 1) * B``, which is ``N / 2`` times more.

Implemented ops (all portable, all runnable on CPU with forced host devices):

    pmap_rs_ag_all_reduce
        The headline kernel: unidirectional shard ring built from
        ``lax.ppermute`` plus dynamic shard indexing inside ``jax.shard_map``.

    pmap_rs_ag_all_reduce_bidir
        The same algorithm split into two opposite-direction half-rings so both
        ICI directions carry traffic at once. Same total bytes, twice the
        ppermute count at half the shard size.

    xla_all_reduce
        ``lax.psum`` on the same case, in the same wire dtype, as the roofline.

Matching ``psum``'s *byte volume* is the goal and is achieved by construction.
Matching its *latency* is not promised: the compiler's collective gets fused
scheduling, while this kernel pays for dynamic-slice updates between steps and
strict step boundaries. Landing within a few percent of ``psum`` at multi-MiB
payloads -- after being ~``N/2`` times worse in Lab 8 -- is the win, and the
remaining gap is the lesson, readable directly in the trace.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

try:  # Normal repository layout: collective_comm_bench/labs/*.py
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

# "rs-ag" is the headline unidirectional shard ring; "rs-ag-bidir" splits the
# payload into two counter-rotating half-rings; "xla-psum" is the tuned XLA
# collective run on the *same case in the same wire dtype* so the comparison is
# apples-to-apples on bytes. "auto" resolves to "rs-ag".
SUPPORTED_KERNEL_MODES = frozenset({"auto", "rs-ag", "rs-ag-bidir", "xla-psum"})

# Ring order policies. "ids" walks devices in jax.devices() order. "auto" tries
# to build a unit-step Hamiltonian cycle from device .coords on 2D TPU slices
# (so every ring hop is one physical ICI link) and falls back to "ids" with a
# recorded reason when it cannot.
SUPPORTED_RING_ORDERS = frozenset({"auto", "ids"})


@dataclasses.dataclass(frozen=True)
class OptimalAllReduceCase:
    """Container returned to the benchmark harness.

    The harness needs a sharded input ``x``, a jitted function ``fn``, and an
    expected result for correctness. The remaining fields make CSV rows, notes,
    and lab artifacts self-explanatory.

    Shape convention
    ----------------

    ``x`` has global shape::

        [device, shard, row, col]

    The mesh partitions the first dimension across devices. The shard dimension
    has length ``N`` (one shard per ring step owner), mirroring Lab 8's
    ``[device, chunk, row, col]`` layout with ``n_chunks`` pinned to ``N``.

    Byte vocabulary
    ---------------

    ``actual_payload_bytes`` is the full per-device payload across all shards.
    ``shard_payload_bytes`` is the size of the one shard a device sends per ring
    step. The implemented paths send ``2 * (N - 1) * shard_payload_bytes`` per
    device -- the standard optimal full-duplex bandwidth term used by the
    harness -- so for this lab's custom all-reduce ``wire == logical``.
    """

    x: Any
    fn: Callable[[Any], Any]
    expected_sums: Any
    actual_payload_bytes: int
    shard_payload_bytes: int
    tile_rows: int
    tile_cols: int
    n_shards: int
    direction: str
    n_devices: int
    kernel_mode: str = "rs-ag"
    ring_order: str = "auto"
    ring_device_ids: tuple[int, ...] = ()
    ring_order_reason: str = ""
    check_rtol: float = 1e-3
    check_atol: float = 1e-3
    fast_path_reason: str = ""

    @property
    def note(self) -> str:
        """Short human-readable note for benchmark tables."""
        reason = f" reason={self.fast_path_reason}" if self.fast_path_reason else ""
        ring = ",".join(str(i) for i in self.ring_device_ids)
        if self.is_roofline_reference:
            schedule = f"modeled_steps={self.ring_steps}"
        else:
            schedule = (
                f"permutes={self.ppermute_calls_per_device}"
                f"x{self.bytes_per_ppermute_message}B"
            )
        return (
            f"lab11-kernel={self.kernel_mode} direction={self.direction} "
            f"shards={self.n_shards} devices={self.n_devices} "
            f"shard={self.shard_payload_bytes}B "
            f"tile={self.tile_rows}x{self.tile_cols} "
            f"steps={self.ring_steps} {schedule} "
            f"naive/opt={self.naive_over_optimal_ratio:g}x "
            f"ring=[{ring}]"
            f"{reason}"
        )

    @property
    def num_devices(self) -> int:
        """Alias used by some benchmark/reporting code."""
        return self.n_devices

    @property
    def input_bytes_per_device(self) -> int:
        """Bytes each device starts with before the shard ring."""
        return self.actual_payload_bytes

    @property
    def output_bytes_per_device(self) -> int:
        """Bytes each device writes after assembling the reduced payload."""
        # Output dtype is float32 because reduced sums are reported in float32.
        return self.n_shards * self.tile_rows * self.tile_cols * 4

    @property
    def ring_steps(self) -> int:
        """Total neighbor-exchange steps: (N-1) reduce-scatter + (N-1) gather."""
        return 2 * max(0, self.n_devices - 1)

    @property
    def ppermute_chains(self) -> int:
        """Independent ppermute chains in the hand-written schedule."""
        if self.is_roofline_reference:
            return 0
        return 2 if self.is_bidirectional else 1

    @property
    def ppermute_calls_per_device(self) -> int:
        """Number of explicit ppermute messages sent by one device."""
        return self.ring_steps * self.ppermute_chains

    @property
    def bytes_per_ppermute_message(self) -> int:
        """Bytes in one explicit ppermute message for this case."""
        if self.is_roofline_reference:
            return 0
        divisor = 2 if self.is_bidirectional else 1
        return self.shard_payload_bytes // divisor

    @property
    def optimal_bytes_per_device(self) -> int:
        """Per-device send bytes for a bandwidth-optimal all-reduce.

        ``2 * (N - 1) * shard_bytes`` equals ``2 * (N - 1) / N`` of the full
        payload because ``shard_bytes == actual_payload_bytes / N`` exactly
        (the tile builder rounds the shard, not the total).
        """
        return self.ring_steps * self.shard_payload_bytes

    @property
    def naive_ring_bytes_per_device(self) -> int:
        """Per-device send bytes for the Lab 7/8 whole-token full ring."""
        return max(0, self.n_devices - 1) * self.actual_payload_bytes

    @property
    def naive_over_optimal_ratio(self) -> float:
        """The headline penalty ratio; equals ``N / 2`` for full rings."""
        if self.optimal_bytes_per_device == 0:
            return 1.0
        return self.naive_ring_bytes_per_device / self.optimal_bytes_per_device

    @property
    def wire_bytes(self) -> int:
        """Actual per-device send bytes for the implemented path.

        Every Lab 11 mode -- including the ``lax.psum`` reference under the
        standard ring model -- moves the optimal volume, so wire == logical.
        """
        return self.optimal_bytes_per_device

    @property
    def is_bidirectional(self) -> bool:
        """Whether the selected path drives both ICI directions at once."""
        return self.kernel_mode == "rs-ag-bidir"

    @property
    def is_roofline_reference(self) -> bool:
        """Whether the selected path is XLA's tuned collective reference."""
        return self.kernel_mode == "xla-psum"

    def byte_model(self) -> dict[str, int | float | str]:
        """Return benchmark-friendly byte accounting for this case."""
        return optimal_all_reduce_byte_model(
            n_devices=self.n_devices,
            full_payload_bytes=self.actual_payload_bytes,
            shard_payload_bytes=self.shard_payload_bytes,
            kernel_mode=self.kernel_mode,
        )


LAB_SPEC: dict[str, Any] = {
    "lab": "lab11",
    "title": "Lab 11: Bandwidth-Optimal Ring All-Reduce",
    "goal": (
        "Close the byte gap confessed in Labs 7 and 8: implement all-reduce as "
        "reduce-scatter plus all-gather so each of the 2*(N-1) ring steps moves "
        "only a B/N shard, matching lax.psum's 2*(N-1)/N*B volume instead of "
        "the whole-token ring's (N-1)*B. The point is to earn the optimal byte "
        "model with an explicit shard schedule, then read the remaining few-"
        "percent latency gap to the compiler's collective out of the trace."
    ),
    "implemented_ops": [
        "`pmap_psum`: built-in lax.psum baseline from Lab 0",
        "`pmap_token_ring`: whole-token dependency-chain ring from Lab 2 (the N/2-penalty foil)",
        "`pmap_rs_ag_all_reduce`: bandwidth-optimal shard ring (reduce-scatter + all-gather via lax.ppermute)",
        "`pmap_rs_ag_all_reduce_bidir`: same volume split across two counter-rotating half-rings",
        "`xla_all_reduce`: lax.psum on the same case in the same wire dtype (roofline)",
        "`lab11_optimal_all_reduce_spec`: shard schedule, byte-model, alpha-beta crossover, and trace-evidence artifact",
    ],
    "deferred_ops": [
        "Fuse the shard ring into a Lab 8-style Pallas program with k*N sub-shards and overlapped remote DMA",
        "Recursive-halving / tree all-reduce for the latency-dominated small-payload regime",
        "Topology-general Hamiltonian ring construction carried into Lab 9's 2D meshes",
        "Partial-hop shard rings for fault and stragglers experiments",
    ],
    "byte_model": [
        "Whole-token full ring: `(N - 1) * B` send bytes per device",
        "Reduce-scatter + all-gather: `2 * (N - 1) * (B / N)` send bytes per device",
        "Penalty ratio naive/optimal = `N / 2`: it grows with the ring, which is why Lab 8 could never win big",
        "`2 * (N - 1) / N * B` is the standard full-duplex bandwidth term; for the custom all-reduce wire == logical",
        "Latency caveat: optimal pays `2 * (N - 1)` per-step latencies vs `(N - 1)`, so the naive ring wins below the alpha-beta crossover",
    ],
    "pass_condition": [
        "rs-ag output matches float32 sums of the dtype-quantized input within dtype-aware tolerance",
        "all device replicas of the rs-ag output are bitwise identical (each reduced shard is computed once and copied)",
        "full output tiles are checked, not only scalar rank markers",
        "rs-ag and rs-ag-bidir wire bytes equal the optimal model exactly: 2*(N-1)*shard_bytes",
        "spec artifact defines the shard ownership schedule, byte convention, crossover formula, and trace evidence rules",
        "multi-MiB sweeps report the rs-ag/xla gap and the trace attributes it; within ~10% is a strong result, not a hidden correctness criterion",
    ],
    "artifacts": [
        "results.jsonl",
        "csvs/results.csv",
        "plots/latency_by_payload.png",
        "plots/bandwidth_by_payload.png",
        "lab_artifacts/*lab11_optimal_all_reduce_spec*",
    ],
    "next_steps": [
        "Use the trace to attribute the residual gap to inter-step dynamic-slice copies and step-boundary stalls",
        "Carry the shard schedule into a fused Pallas pipeline and into topology-aware Lab 9 meshes",
    ],
}


def normalize_direction(direction: str | None) -> str:
    """Validate and default the ring direction."""
    if direction is None:
        return "right"
    direction = str(direction).strip().lower()
    if direction not in SUPPORTED_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {sorted(SUPPORTED_DIRECTIONS)}, got {direction!r}"
        )
    return direction


def normalize_kernel_mode(mode: str | None) -> str:
    """Validate the requested kernel mode and resolve ``auto``."""
    if mode is None:
        return "rs-ag"
    mode = str(mode).strip().lower()
    if mode not in SUPPORTED_KERNEL_MODES:
        raise ValueError(
            f"kernel_mode must be one of {sorted(SUPPORTED_KERNEL_MODES)}, got {mode!r}"
        )
    if mode == "auto":
        return "rs-ag"
    return mode


def normalize_ring_order(policy: str | None) -> str:
    """Validate the ring-order policy."""
    if policy is None:
        return "auto"
    policy = str(policy).strip().lower()
    if policy not in SUPPORTED_RING_ORDERS:
        raise ValueError(
            f"ring_order must be one of {sorted(SUPPORTED_RING_ORDERS)}, got {policy!r}"
        )
    return policy


def ceil_div(numer: int, denom: int) -> int:
    """Integer ceiling division for tile sizing."""
    if denom <= 0:
        raise ValueError(f"denominator must be positive, got {denom}")
    return -(-int(numer) // int(denom))


def direction_sign(direction: str) -> int:
    """Map a ring direction to its rank increment.

    ``right`` means each source ``i`` sends to ``(i + 1) % N``, so the sign is
    ``+1``; ``left`` sends to ``(i - 1) % N``, sign ``-1``. This matches Lab 8's
    convention, where a right ring makes receiver ``r`` see source history
    ``[r, r-1, r-2, ...]``.
    """
    return 1 if normalize_direction(direction) == "right" else -1


def ring_permutation(n_devices: int, direction: str) -> list[tuple[int, int]]:
    """Return the ``lax.ppermute`` (source, destination) pairs for the ring."""
    sgn = direction_sign(direction)
    n = int(n_devices)
    return [(rank, (rank + sgn) % n) for rank in range(n)]


def shard_tile_shape_for_payload(
    *,
    payload_bytes: int,
    itemsize: int,
    tile_rows: int,
    min_cols: int,
    n_shards: int,
    require_even_rows: bool = False,
) -> tuple[int, int, int, int, str]:
    """Choose a per-shard tile shape for the requested full payload.

    ``payload_bytes`` is the requested full per-device input payload. Lab 11
    splits it across ``n_shards == n_devices`` shards (one ring owner each),
    then rounds the per-shard tile up to ``tile_rows * tile_cols`` elements, so
    the actual payload is always an exact multiple of the shard. That keeps the
    optimal byte model exact: ``2 * (N - 1) * shard_bytes`` is precisely
    ``2 * (N - 1) / N`` of the actual payload, with no remainder shard.

    The bidirectional mode splits each shard tile into a top and bottom half
    along the row axis, so it needs an even row count; ``require_even_rows``
    bumps an odd ``tile_rows`` and reports the adjustment.

    Returns:

        ``(rows, cols, shard_payload_bytes, actual_payload_bytes, adjust_note)``
    """
    payload_bytes = max(1, int(payload_bytes))
    itemsize = int(itemsize)
    rows = max(1, int(tile_rows))
    min_cols = max(1, int(min_cols))
    n_shards = max(1, int(n_shards))
    if itemsize <= 0:
        raise ValueError(f"itemsize must be positive, got {itemsize}")

    adjust_note = ""
    if require_even_rows and rows % 2 != 0:
        adjust_note = f"bidir needs even rows; tile_rows {rows}->{rows + 1}"
        rows += 1

    # Round up so the total actual payload is at least the requested payload.
    cols = max(min_cols, ceil_div(payload_bytes, itemsize * rows * n_shards))
    shard_payload_bytes = rows * cols * itemsize
    actual_payload_bytes = n_shards * shard_payload_bytes
    return rows, cols, shard_payload_bytes, actual_payload_bytes, adjust_note


def make_sharded_rank_input(
    jnp: Any,
    *,
    n_devices: int,
    n_shards: int,
    rows: int,
    cols: int,
    dtype: Any,
) -> Any:
    """Create the teaching input tensor for Lab 11.

    Same value pattern as Lab 8, with the chunk axis reinterpreted as the shard
    axis. The scalar at ``[row=0, col=0]`` is the hand-checkable marker:

        x[source, shard, 0, 0] = 10 * source + shard

    so the all-reduced marker is ``sum_s (10*s + c) = 10*N*(N-1)/2 + N*c``; for
    ``N = 4`` that is ``60, 64, 68, 72`` for shards ``0..3``, and those values
    are exactly representable even in bfloat16. The rest of the tile carries
    tiny row/column patterns so a stale-shard or partial-update bug cannot hide
    behind a correct marker at ``[0, 0]``.
    """
    src = jnp.arange(n_devices, dtype=jnp.float32).reshape(n_devices, 1, 1, 1)
    shard = jnp.arange(n_shards, dtype=jnp.float32).reshape(1, n_shards, 1, 1)
    row_pattern = (jnp.arange(rows, dtype=jnp.float32) % 8).reshape(1, 1, rows, 1) * 0.25
    col_pattern = (jnp.arange(cols, dtype=jnp.float32) % 16).reshape(1, 1, 1, cols) * 0.03125
    x_host = src * 10.0 + shard + row_pattern + col_pattern
    return x_host.astype(dtype)


def expected_all_reduce_tiles(
    jnp: Any,
    *,
    n_devices: int,
    n_shards: int,
    rows: int,
    cols: int,
    dtype: Any,
) -> Any:
    """Return the full expected output tile for every receiver and shard.

    The calculation starts from ``make_sharded_rank_input(...).astype(float32)``
    so low-precision input dtypes are modeled the same way as the real kernel:
    the *input quantization* is included, while the sum itself is taken in
    float32. The kernel accumulates partials in the wire dtype (that is the
    whole point of the byte model), so its result can differ from this float32
    reference by accumulated rounding; ``check_result`` uses dtype-aware
    tolerances for exactly this reason.

    All-reduce delivers the same sum everywhere, so the expected tensor is one
    ``[shard, row, col]`` total stacked ``n_devices`` times.
    """
    base = make_sharded_rank_input(
        jnp,
        n_devices=n_devices,
        n_shards=n_shards,
        rows=rows,
        cols=cols,
        dtype=dtype,
    ).astype(jnp.float32)
    total = jnp.sum(base, axis=0)
    return jnp.stack([total] * n_devices, axis=0)


def expected_shard_marker_values(n_devices: int, n_shards: int | None = None) -> list[float]:
    """Expected ``[0, 0]`` marker per shard: ``10*N*(N-1)/2 + N*c``."""
    n = int(n_devices)
    shards = n if n_shards is None else int(n_shards)
    base = 10.0 * n * (n - 1) / 2.0
    return [base + n * c for c in range(shards)]


def reduce_scatter_schedule(
    *,
    n_devices: int,
    direction: str,
    max_devices: int = 16,
) -> dict[str, Any]:
    """Describe the reduce-scatter phase step by step.

    Invariant (provable by induction, and checked by ``simulate_rs_ag``): after
    step ``s`` of a ring with sign ``sgn``, device ``d`` holds

        partial(d, s) = sum over j in [0, s] of C[(d - sgn*j) % N][(d - sgn*s) % N]

    i.e. a partial sum of shard index ``(d - sgn*s) % N`` covering ``s + 1``
    sources. After step ``N - 1`` the partial covers all sources and device
    ``d`` owns the fully reduced shard ``(d + sgn) % N``.
    """
    n = int(n_devices)
    sgn = direction_sign(direction)
    steps: list[dict[str, Any]] = []
    for s in range(1, n):
        entry: dict[str, Any] = {
            "step": s,
            "device_d_sends": f"partial of shard (d - {sgn}*{s - 1}) % {n} to (d + {sgn}) % {n}",
            "device_d_adds_local_shard": f"(d - {sgn}*{s}) % {n}",
            "partial_now_covers_sources": s + 1,
        }
        if n <= max_devices:
            entry["per_device"] = [
                {
                    "device": d,
                    "shard_in_flight": (d - sgn * s) % n,
                    "received_from": (d - sgn) % n,
                    "adds_local_shard": (d - sgn * s) % n,
                }
                for d in range(n)
            ]
        steps.append(entry)
    return {
        "phase": "reduce-scatter",
        "steps_total": max(0, n - 1),
        "bytes_per_step_per_device": "shard_payload_bytes",
        "owner_after_phase": f"device d owns fully reduced shard (d + {sgn}) % {n}",
        "steps": steps,
    }


def all_gather_schedule(
    *,
    n_devices: int,
    direction: str,
    max_devices: int = 16,
) -> dict[str, Any]:
    """Describe the all-gather phase step by step.

    Device ``d`` seeds its output at shard ``(d + sgn) % N`` (the shard it just
    finished reducing), then for ``s in [0, N - 2]`` each ppermute delivers the
    finished shard ``(d - sgn*s) % N``, which is stored verbatim. No arithmetic
    happens in this phase, which is why all device replicas of the final output
    are bitwise identical.
    """
    n = int(n_devices)
    sgn = direction_sign(direction)
    steps: list[dict[str, Any]] = []
    for s in range(n - 1):
        entry: dict[str, Any] = {
            "step": s + 1,
            "device_d_receives_finished_shard": f"(d - {sgn}*{s}) % {n}",
        }
        if n <= max_devices:
            entry["per_device"] = [
                {"device": d, "stores_shard": (d - sgn * s) % n} for d in range(n)
            ]
        steps.append(entry)
    return {
        "phase": "all-gather",
        "steps_total": max(0, n - 1),
        "bytes_per_step_per_device": "shard_payload_bytes",
        "seed": f"device d stores its reduced shard at index (d + {sgn}) % {n} before the first step",
        "steps": steps,
    }


def reduced_shard_owner_map(n_devices: int, direction: str) -> dict[int, int]:
    """Map device -> shard index it owns after reduce-scatter."""
    n = int(n_devices)
    sgn = direction_sign(direction)
    return {d: (d + sgn) % n for d in range(n)}


def optimal_all_reduce_byte_model(
    *,
    n_devices: int,
    full_payload_bytes: int,
    shard_payload_bytes: int,
    kernel_mode: str = "rs-ag",
) -> dict[str, int | float | str]:
    """Return the Lab 11 byte accounting in benchmark-friendly form."""
    n = int(n_devices)
    steps = 2 * max(0, n - 1)
    optimal = steps * int(shard_payload_bytes)
    naive = max(0, n - 1) * int(full_payload_bytes)
    return {
        "kernel_mode": str(kernel_mode),
        "n_devices": n,
        "full_payload_bytes": int(full_payload_bytes),
        "shard_payload_bytes": int(shard_payload_bytes),
        "ring_steps": steps,
        # This is the convention used by pmap_logical_bytes("pmap_psum", ...):
        # the full-duplex bandwidth term. Receive traffic is equal by symmetry
        # but is not added again to the GB/s denominator.
        "send_bytes_per_device": optimal,
        "recv_bytes_per_device": optimal,
        "endpoint_send_plus_recv_bytes_per_device": 2 * optimal,
        "reported_bytes_convention": (
            "benchmark GB/s uses the send-side/full-duplex bandwidth term; "
            "endpoint ingress+egress counters would be twice this number"
        ),
        "naive_whole_token_ring_bytes": naive,
        "naive_over_optimal_ratio": (naive / optimal) if optimal else 1.0,
        "lower_bound_note": (
            "the reported 2*(N-1)/N*B term is the standard ring all-reduce "
            "bandwidth term: (N-1)/N*B to export contributions during "
            "reduce-scatter plus (N-1)/N*B to disseminate finalized shards "
            "during all-gather; receives are equal by symmetry"
        ),
        "xla_psum_assumption": (
            "xla-psum is modeled at the same optimal volume; its actual wire "
            "schedule is the compiler's business, which is the point"
        ),
    }


def alpha_beta_model(n_devices: int) -> dict[str, str]:
    """Latency/bandwidth model strings for the spec artifact.

    With per-message latency ``alpha`` and byte time ``1/beta``:

        T_naive   = (N - 1) * (alpha + B / beta)
        T_optimal = 2 * (N - 1) * (alpha + B / (N * beta))

    Setting them equal gives the crossover payload ``B* = alpha*beta*N/(N-2)``
    (undefined at N = 2, where the two algorithms move identical bytes). Below
    ``B*`` the *naive* ring wins because it pays half the per-step latencies;
    above it the optimal ring's byte savings dominate. Students should fit
    ``alpha`` and ``beta`` from the small-payload sweep and check the measured
    crossover against this formula.
    """
    n = int(n_devices)
    crossover = "undefined (N=2: naive and optimal move identical bytes)"
    if n > 2:
        crossover = f"B* = alpha * beta * {n} / {n - 2}"
    return {
        "t_naive": f"({n} - 1) * (alpha + B / beta)",
        "t_optimal": f"2 * ({n} - 1) * (alpha + B / ({n} * beta))",
        "crossover_payload": crossover,
        "reading": (
            "optimal halves the bytes per step but doubles the step count, so "
            "small payloads are latency-bound and favor the naive ring"
        ),
    }


def simulate_rs_ag(chunks_per_device: list[Any], direction: str) -> list[Any]:
    """Pure-NumPy reference simulation of the shard ring schedule.

    ``chunks_per_device[d]`` is device ``d``'s local ``[n_shards, ...]`` array
    with ``n_shards == len(chunks_per_device)``. The function executes the exact
    reduce-scatter and all-gather index schedule used by the JAX kernel, but
    with explicit message passing between plain arrays, so it validates the
    schedule independently of ``ppermute``. Returns the per-device outputs.

    This doubles as a debugging tool: when a real run fails, reproduce the
    failure here first; if the simulation is correct and the device run is not,
    the bug is in the collective plumbing rather than the index math.
    """
    import numpy as np

    n = len(chunks_per_device)
    if n == 0:
        return []
    sgn = direction_sign(direction)
    chunks = [np.asarray(c) for c in chunks_per_device]
    if any(c.shape[0] != n for c in chunks):
        raise ValueError("each device must hold n_devices shards")

    # Reduce-scatter: v starts as the device's own shard index d.
    v = [chunks[d][d].copy() for d in range(n)]
    for s in range(n - 1):
        # ppermute with perm (i -> i + sgn): device d receives from d - sgn.
        v = [v[(d - sgn) % n] for d in range(n)]
        v = [v[d] + chunks[d][(d - sgn * (s + 1)) % n] for d in range(n)]

    # All-gather: seed at the owned shard, then circulate finished shards.
    out = [np.zeros_like(chunks[d]) for d in range(n)]
    for d in range(n):
        out[d][(d + sgn) % n] = v[d]
    for s in range(n - 1):
        v = [v[(d - sgn) % n] for d in range(n)]
        for d in range(n):
            out[d][(d - sgn * s) % n] = v[d]
    return out


def _device_coords_2d(devices: list[Any]) -> list[tuple[int, int]] | None:
    """Extract dense 2D (x, y) coords from devices, or None if unavailable.

    TPU devices expose ``.coords`` as ``(x, y, z)``; a v5e slice is a 2D mesh
    with ``z == 0`` everywhere. CPU/GPU devices have no coords, and sparse or
    3D layouts are out of scope here, so any irregularity returns None and the
    caller falls back to id order.
    """
    coords: list[tuple[int, int]] = []
    zs: set[int] = set()
    for dev in devices:
        raw = getattr(dev, "coords", None)
        if raw is None or len(raw) < 2:
            return None
        x, y = int(raw[0]), int(raw[1])
        zs.add(int(raw[2]) if len(raw) > 2 else 0)
        coords.append((x, y))
    if len(zs) > 1:
        return None
    xs = {c[0] for c in coords}
    ys = {c[1] for c in coords}
    dim_x = max(xs) - min(xs) + 1
    dim_y = max(ys) - min(ys) + 1
    if dim_x * dim_y != len(devices) or len(set(coords)) != len(devices):
        return None
    # Normalize to a 0-based grid.
    min_x, min_y = min(xs), min(ys)
    return [(x - min_x, y - min_y) for (x, y) in coords]


def _grid_hamiltonian_cycle(dim_x: int, dim_y: int) -> list[tuple[int, int]] | None:
    """Build a unit-step Hamiltonian cycle on a ``dim_x x dim_y`` grid.

    A grid graph has such a cycle iff the vertex count is even (the grid is
    bipartite, so an odd cycle length is impossible) and neither dimension is a
    too-long line. The construction is the standard comb: walk every column
    through rows ``1..dim_y-1`` in a snake, then return home along row ``0``.
    """
    if dim_x * dim_y % 2 != 0 or dim_x * dim_y < 2:
        return None
    if dim_x == 1 or dim_y == 1:
        # A line has a unit-step cycle only when it is a single pair.
        if dim_x * dim_y == 2:
            return [(x, y) for x in range(dim_x) for y in range(dim_y)]
        return None
    if dim_x % 2 != 0:
        transposed = _grid_hamiltonian_cycle(dim_y, dim_x)
        if transposed is None:
            return None
        return [(x, y) for (y, x) in transposed]

    path: list[tuple[int, int]] = [(0, 0)]
    for x in range(dim_x):
        ys = range(1, dim_y) if x % 2 == 0 else range(dim_y - 1, 0, -1)
        path.extend((x, y) for y in ys)
    path.append((dim_x - 1, 0))
    path.extend((x, 0) for x in range(dim_x - 2, 0, -1))
    return path


def _validate_ring(path: list[tuple[int, int]], n: int) -> bool:
    """Check coverage, unit Manhattan steps, and unit closure for a cycle."""
    if len(path) != n or len(set(path)) != n:
        return False
    if n == 1:
        return True
    for a, b in zip(path, path[1:] + path[:1]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    return True


def order_ring_devices(devices: list[Any], policy: str) -> tuple[list[Any], str]:
    """Order devices so consecutive ring neighbors are physical neighbors.

    With the default ``jax.devices()`` id order on a 2x2 v5e slice, the ring
    ``0 -> 1 -> 2 -> 3 -> 0`` includes two 2-hop diagonal-ish edges, so two of
    the four ring links pay double latency and share ICI segments. ``auto``
    instead walks a unit-step Hamiltonian cycle over device ``.coords`` when
    one exists; every ppermute step then crosses exactly one physical link.

    Reordering before building the mesh is safe for an all-reduce: the result
    is the same full sum on every device, and inputs/outputs are addressed by
    logical mesh rank throughout, so only the physical link schedule changes --
    which is precisely the experiment.

    Returns ``(ordered_devices, reason)`` where ``reason`` explains a fallback.
    """
    policy = normalize_ring_order(policy)
    devices = list(devices)
    if policy == "ids" or len(devices) < 3:
        return devices, "" if policy == "ids" else "ring too small to reorder"

    coords = _device_coords_2d(devices)
    if coords is None:
        return devices, "no dense 2D device coords; using id order"

    dim_x = max(c[0] for c in coords) + 1
    dim_y = max(c[1] for c in coords) + 1
    cycle = _grid_hamiltonian_cycle(dim_x, dim_y)
    if cycle is None or not _validate_ring(cycle, len(devices)):
        return devices, (
            f"no unit-step Hamiltonian cycle on {dim_x}x{dim_y}; using id order"
        )

    by_coord = {coord: dev for coord, dev in zip(coords, devices)}
    ordered = [by_coord[coord] for coord in cycle]
    # Rotate so the lowest device id leads; cosmetic, but keeps notes stable.
    ids = [int(getattr(dev, "id", i)) for i, dev in enumerate(ordered)]
    pivot = ids.index(min(ids))
    return ordered[pivot:] + ordered[:pivot], ""


def rs_ag_all_reduce(
    value: Any,
    *,
    mesh: Any,
    axis_name: str,
    direction: str,
    n_devices: int,
    bidirectional: bool = False,
) -> Any:
    """Bandwidth-optimal all-reduce: reduce-scatter + all-gather over shards.

    The input is the Lab 8-style ``[device, shard, row, col]`` tensor with the
    shard axis length pinned to ``N``. Inside ``shard_map`` each device sees
    ``[1, N, row, col]``; the algorithm circulates one ``[row, col]`` shard via
    ``lax.ppermute`` while indexing local shards with the schedule documented
    in ``reduce_scatter_schedule`` / ``all_gather_schedule``.

    The circulating partial stays in the *wire dtype*: that is what makes the
    byte model optimal, and it means low-precision dtypes accumulate rounding
    per step exactly as a wire-dtype ``psum`` would. The assembled result is
    cast to float32 once at the end, matching the course's output convention.

    ``bidirectional`` splits each shard tile into top/bottom row halves and
    runs two independent counter-rotating rings; XLA sees two dataflow-disjoint
    ppermute chains and is free to schedule them onto both ICI directions at
    once. Same total bytes, twice the messages at half the size -- whether the
    overlap is real is a trace question, not a faith question.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    direction = normalize_direction(direction)
    n = int(n_devices)
    partition = jax.sharding.PartitionSpec(axis_name, None, None, None)
    sgn = direction_sign(direction)

    def _rs_ag(chunks: Any, ring_sgn: int) -> Any:
        perm = [(rank, (rank + ring_sgn) % n) for rank in range(n)]
        d = lax.axis_index(axis_name)

        # Reduce-scatter: start from your own shard index d, then for each of
        # the N-1 steps forward your partial and fold in the local shard whose
        # turn it is. Python `%` on a traced int32 lowers to jnp.remainder,
        # which is non-negative for a positive modulus, so the index math is
        # safe for both ring signs.
        v = lax.dynamic_index_in_dim(chunks, d, axis=0, keepdims=False)
        for s in range(n - 1):
            v = lax.ppermute(v, axis_name, perm)
            local_idx = (d - ring_sgn * (s + 1)) % n
            v = v + lax.dynamic_index_in_dim(chunks, local_idx, axis=0, keepdims=False)

        # All-gather: seed the output with the shard this device just finished
        # reducing -- index (d + sgn) % N -- then circulate finished shards.
        out = jnp.zeros_like(chunks)
        out = lax.dynamic_update_index_in_dim(out, v, (d + ring_sgn) % n, axis=0)
        for s in range(n - 1):
            v = lax.ppermute(v, axis_name, perm)
            out = lax.dynamic_update_index_in_dim(out, v, (d - ring_sgn * s) % n, axis=0)
        return out

    def _local(local_value: Any) -> Any:
        chunks = local_value[0]  # [n_shards, rows, cols]
        if bidirectional:
            half = chunks.shape[1] // 2
            top = _rs_ag(chunks[:, :half, :], sgn)
            bottom = _rs_ag(chunks[:, half:, :], -sgn)
            out = jnp.concatenate([top, bottom], axis=1)
        else:
            out = _rs_ag(chunks, sgn)
        return out.astype(jnp.float32)[None]

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_vma=False,
    )(value)


def xla_psum_all_reduce(value: Any, *, mesh: Any, axis_name: str) -> Any:
    """Roofline reference: ``lax.psum`` on the same case, same wire dtype.

    Deliberately unlike Lab 8's XLA path, which cast to float32 *before* the
    psum (inflating wire bytes 2x for bfloat16 inputs), this reference reduces
    in the input dtype and casts after, so the student kernel and the roofline
    move the same bytes per the standard ring model. The remaining latency gap
    is therefore scheduling, not volume.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    partition = jax.sharding.PartitionSpec(axis_name, None, None, None)

    def _local(local_value: Any) -> Any:
        return lax.psum(local_value, axis_name).astype(jnp.float32)

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=partition,
        out_specs=partition,
        check_vma=False,
    )(value)


def build_case(
    *,
    jax: Any,
    jnp: Any,
    devices: list[Any],
    axis_name: str,
    payload_bytes: int,
    dtype: Any,
    direction: str,
    kernel_mode: str | None = None,
    tile_rows: int = 4,
    min_cols: int = 128,
    ring_order: str = "auto",
) -> OptimalAllReduceCase:
    """Build input, expected output, and jitted function for Lab 11."""
    import numpy as np

    n_devices = len(devices)
    if n_devices < 2:
        raise ValueError("Lab 11 requires at least two devices to form a ring")

    direction = normalize_direction(direction)
    selected_kernel_mode = normalize_kernel_mode(kernel_mode)
    ring_order = normalize_ring_order(ring_order)
    n_shards = n_devices

    itemsize = int(jnp.dtype(dtype).itemsize)
    rows, cols, shard_payload_bytes, actual_payload_bytes, adjust_note = (
        shard_tile_shape_for_payload(
            payload_bytes=payload_bytes,
            itemsize=itemsize,
            tile_rows=tile_rows,
            min_cols=min_cols,
            n_shards=n_shards,
            require_even_rows=selected_kernel_mode == "rs-ag-bidir",
        )
    )

    ordered_devices, ring_reason = order_ring_devices(devices, ring_order)
    ring_device_ids = tuple(
        int(getattr(dev, "id", i)) for i, dev in enumerate(ordered_devices)
    )

    mesh = jax.sharding.Mesh(np.array(ordered_devices), (axis_name,))
    sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(axis_name),
    )

    x_host = make_sharded_rank_input(
        jnp,
        n_devices=n_devices,
        n_shards=n_shards,
        rows=rows,
        cols=cols,
        dtype=dtype,
    )
    x = jax.device_put(x_host, sharding)

    def all_reduce_fn(value: Any) -> Any:
        if selected_kernel_mode == "xla-psum":
            return xla_psum_all_reduce(value, mesh=mesh, axis_name=axis_name)
        return rs_ag_all_reduce(
            value,
            mesh=mesh,
            axis_name=axis_name,
            direction=direction,
            n_devices=n_devices,
            bidirectional=selected_kernel_mode == "rs-ag-bidir",
        )

    rtol, atol = default_tolerances(jnp, dtype)
    notes = "; ".join(note for note in (adjust_note, ring_reason) if note)
    if n_devices == 2:
        n2 = "N=2: naive and optimal rings move identical bytes"
        notes = f"{notes}; {n2}" if notes else n2

    return OptimalAllReduceCase(
        x=x,
        fn=jax.jit(all_reduce_fn),
        expected_sums=expected_all_reduce_tiles(
            jnp,
            n_devices=n_devices,
            n_shards=n_shards,
            rows=rows,
            cols=cols,
            dtype=dtype,
        ),
        actual_payload_bytes=actual_payload_bytes,
        shard_payload_bytes=shard_payload_bytes,
        tile_rows=rows,
        tile_cols=cols,
        n_shards=n_shards,
        direction=direction,
        n_devices=n_devices,
        kernel_mode=selected_kernel_mode,
        ring_order=ring_order,
        ring_device_ids=ring_device_ids,
        ring_order_reason=ring_reason,
        check_rtol=rtol,
        check_atol=atol,
        fast_path_reason=notes,
    )


def default_tolerances(jnp: Any, dtype: Any) -> tuple[float, float]:
    """Dtype-aware tolerances for comparing wire-dtype sums to float32 sums.

    The kernel accumulates ``N`` partials in the wire dtype while the expected
    reference sums dtype-quantized inputs in float32, so the permitted error is
    a property of the wire dtype, not of the algorithm. bfloat16 has 8 mantissa
    bits and tile values reach ~75, so a few accumulation steps can wander by
    O(1) absolute; the scalar markers (60, 64, 68, 72 at N=4) remain exact.
    Integer dtypes must match exactly.
    """
    kind_map: dict[str, tuple[float, float]] = {
        "float64": (1e-5, 1e-4),
        "float32": (1e-5, 1e-4),
        "bfloat16": (2e-2, 2.0),
        "float16": (1e-2, 0.5),
    }
    name = str(jnp.dtype(dtype).name)
    if name in kind_map:
        return kind_map[name]
    if jnp.issubdtype(jnp.dtype(dtype), jnp.integer):
        return (0.0, 0.0)
    return (2e-2, 2.0)


def observed_shard_markers(jax: Any, y: Any) -> Any:
    """Return ``y[:, :, 0, 0]`` on the host for quick schedule debugging."""
    return jax.device_get(y)[:, :, 0, 0]


def check_result(
    jax: Any,
    jnp: Any,
    y: Any,
    expected: Any,
    *,
    dtype: Any = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> bool:
    """Validate the full all-reduce output.

    Two checks, in order of diagnostic value:

    1. **Bitwise replica identity.** Every Lab 11 path computes each reduced
       shard exactly once and copies it verbatim during all-gather (and psum is
       a single deterministic collective), so all device replicas must be
       bitwise equal. A replica mismatch means a stale shard or a wrong-owner
       index, never harmless rounding -- it localizes bugs to the schedule.

    2. **Full-tile numeric check** against float32 sums of the dtype-quantized
       input, with dtype-aware tolerances (the kernel accumulates in the wire
       dtype; see ``default_tolerances``). Note rs-ag and xla-psum are *not*
       required to match each other bitwise: their reduction orders differ.

    The bare 3-argument call used by generic harness code falls back to the
    loosest (bfloat16-grade) tolerance; runners should pass ``dtype=``.
    """
    del jnp  # Host-side NumPy makes this a pure correctness check.

    import numpy as np

    y_host = np.asarray(jax.device_get(y))
    expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)

    if y_host.ndim != 4:
        return False

    if not bool(np.all(y_host == y_host[0:1])):
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

    if rtol is None or atol is None:
        if dtype is not None:
            import jax.numpy as _jnp

            d_rtol, d_atol = default_tolerances(_jnp, dtype)
        else:
            d_rtol, d_atol = (2e-2, 2.0)
        rtol = d_rtol if rtol is None else rtol
        atol = d_atol if atol is None else atol

    y_f32 = y_host.astype(np.float32)
    return bool(
        np.allclose(
            y_f32,
            expected_tiles + np.zeros_like(y_f32),
            rtol=rtol,
            atol=atol,
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
            "lab11_ring_order": getattr(args, "lab11_ring_order", None),
            "pallas_tile_rows": getattr(args, "pallas_tile_rows", None),
            "pallas_min_cols": getattr(args, "pallas_min_cols", None),
            "dtype": getattr(args, "dtype", None),
        },
    }


def _fallback_render_markdown(spec: dict[str, Any]) -> str:
    """Small local fallback for ``lab_spec_utils.render_markdown``."""
    lines = [f"# {spec.get('title', 'Lab 11 Spec')}", ""]
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


def _ring_order_preview(jax: Any, policy: str) -> dict[str, Any]:
    """Describe what ``order_ring_devices`` would do for the live devices.

    Guarded: spec generation should never fail because a backend is absent or
    coords are missing; in that case the preview just says so.
    """
    try:
        devices = jax.devices()
        ordered, reason = order_ring_devices(devices, policy)
        return {
            "policy": policy,
            "device_ids_in_ring_order": [
                int(getattr(dev, "id", i)) for i, dev in enumerate(ordered)
            ],
            "coords_in_ring_order": [
                tuple(int(v) for v in getattr(dev, "coords", ())) for dev in ordered
            ],
            "fallback_reason": reason,
        }
    except Exception as exc:  # pragma: no cover - environment-dependent.
        return {"policy": policy, "preview_unavailable": str(exc)}


def build_spec(*, jax: Any, args: Any, payload_bytes: int, n_devices: int) -> dict[str, Any]:
    """Build the Lab 11 course artifact consumed by the benchmark harness."""
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
    ring_order = normalize_ring_order(getattr(args, "lab11_ring_order", None) or "auto")
    tile_rows = int(getattr(args, "pallas_tile_rows", 4) or 4)
    min_cols = int(getattr(args, "pallas_min_cols", 128) or 128)
    dtype_name = str(getattr(args, "dtype", "bfloat16") or "bfloat16")
    itemsize_by_name = {
        "float64": 8, "float32": 4, "bfloat16": 2, "float16": 2,
        "int32": 4, "int16": 2, "int8": 1, "uint8": 1,
    }
    itemsize = itemsize_by_name.get(dtype_name, 2)

    rows, cols, shard_payload_bytes, actual_payload_bytes, adjust_note = (
        shard_tile_shape_for_payload(
            payload_bytes=payload_bytes,
            itemsize=itemsize,
            tile_rows=tile_rows,
            min_cols=min_cols,
            n_shards=n_devices,
        )
    )

    sgn = direction_sign(direction)
    spec["neighbor_direction"] = direction
    spec["direction_sign"] = sgn
    spec["ring_order_policy"] = ring_order
    spec["ring_order_preview"] = _ring_order_preview(jax, ring_order)
    spec["n_shards"] = n_devices
    spec["shard_tile"] = {
        "rows": rows,
        "cols": cols,
        "shard_payload_bytes": shard_payload_bytes,
        "actual_payload_bytes": actual_payload_bytes,
        "rounding_note": adjust_note or "shard rounded up to rows*cols elements; total = N * shard exactly",
    }
    spec["byte_model"] = optimal_all_reduce_byte_model(
        n_devices=n_devices,
        full_payload_bytes=actual_payload_bytes,
        shard_payload_bytes=shard_payload_bytes,
        kernel_mode="rs-ag",
    )
    spec["reported_byte_convention"] = {
        "bench_columns": (
            "logical_bytes/wire_bytes report the send-side/full-duplex "
            "bandwidth term, matching the existing pmap_psum model"
        ),
        "send_bytes_per_device": spec["byte_model"]["send_bytes_per_device"],
        "recv_bytes_per_device": spec["byte_model"]["recv_bytes_per_device"],
        "endpoint_send_plus_recv_bytes_per_device": spec["byte_model"][
            "endpoint_send_plus_recv_bytes_per_device"
        ],
    }
    spec["penalty_ratio_table"] = {
        f"N={n}": f"naive/optimal = {n}/2 = {n / 2:g}"
        for n in (2, 4, 8, 16, 256)
    }
    spec["alpha_beta_model"] = alpha_beta_model(n_devices)
    spec["reduce_scatter_schedule"] = reduce_scatter_schedule(
        n_devices=n_devices, direction=direction
    )
    spec["all_gather_schedule"] = all_gather_schedule(
        n_devices=n_devices, direction=direction
    )
    spec["reduced_shard_owner_after_rs"] = {
        f"device_{d}": owner
        for d, owner in reduced_shard_owner_map(n_devices, direction).items()
    }
    spec["expected_scalar_shard_markers"] = expected_shard_marker_values(n_devices)
    spec["replica_invariant"] = (
        "all device replicas of the output are bitwise identical: each reduced "
        "shard is computed once and copied verbatim during all-gather"
    )
    spec["trace_evidence_rules"] = [
        f"rs-ag shows {2 * (n_devices - 1)} collective-permute start/done pairs, each ~shard_payload_bytes",
        "rs-ag-bidir shows twice as many permutes at half-shard size; total bytes stay optimal",
        "the whole-token ring shows N-1 pairs at ~N times the size; same wall, different bricks",
        "xla_all_reduce shows one fused all-reduce region with no visible hand-written step boundaries",
        "gaps between rs-ag permutes are dynamic-(update-)slice copies and step-boundary stalls: the scheduling tax",
        "rs-ag-bidir should show counter-rotating permute chains overlapping in time; if serialized, the bidir bet failed",
        "per-step achieved GB/s = shard_payload_bytes / step_time; compare against the link roofline",
    ]
    spec["fair_comparison_notes"] = [
        "xla_all_reduce is the apples-to-apples roofline: same case, same shard tile, same wire dtype",
        "pmap_psum is a continuity baseline from earlier labs, not the same shaped case",
        "pmap_token_ring is the N/2 byte-penalty foil; use --lab11-ring-order ids when you want a pure byte-model comparison to that ID-order ring",
        "--lab11-ring-order auto is a topology experiment layered on top of the byte-optimal algorithm",
    ]
    spec["student_checkpoint_questions"] = [
        "Why is 2*(N-1)/N*B a lower bound and not just a clever schedule?",
        "Which shard does device d own after reduce-scatter, and why (d + sgn) % N?",
        "Where does the naive ring beat the optimal one, and what does the crossover say about alpha?",
        "Why must all replicas be bitwise identical for rs-ag but not rs-ag vs xla-psum?",
        "What in the trace accounts for rs-ag landing a few percent behind psum at 4 MiB?",
        "Does ring order (auto vs ids) change bytes, latency, neither, or both on a 2x2 slice?",
        "Which comparison isolates the byte model, and which comparison also includes topology-aware ring ordering?",
    ]
    return spec


def render_markdown(spec: dict[str, Any]) -> str:
    """Render a spec artifact as Markdown."""
    if lab_spec_utils is not None:
        return lab_spec_utils.render_markdown(spec)
    return _fallback_render_markdown(spec)


def _selftest() -> int:  # pragma: no cover - exercised manually / in CI.
    """Standalone correctness sweep on forced host devices.

    Run as ``python labs/lab11_optimal_all_reduce.py``; sets
    ``--xla_force_host_platform_device_count=4`` before importing jax so the
    full 4-device schedule runs on CPU.
    """
    import os

    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = (
            flags + " --xla_force_host_platform_device_count=4"
        ).strip()

    import numpy as np
    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    assert len(devices) >= 4, f"need 4 forced host devices, got {len(devices)}"
    failures = 0

    # Schedule simulator vs direct sum, both directions, including N=3 and N=5
    # (odd device counts never run on this 4-device harness but the index math
    # should not care).
    rng = np.random.default_rng(11)
    for n in (2, 3, 4, 5):
        chunks = [rng.standard_normal((n, 3, 5)).astype(np.float32) for _ in range(n)]
        want = np.sum(np.stack(chunks), axis=0)
        for direction in ("right", "left"):
            got = simulate_rs_ag(chunks, direction)
            for d in range(n):
                if not np.allclose(got[d], want, rtol=1e-5, atol=1e-5):
                    failures += 1
                    print(f"FAIL simulate n={n} direction={direction} device={d}")

    # Device sweep: modes x dtypes x directions x payloads, plus a 2-device case.
    sweep = [
        (devices[:4], mode, dtype_name, direction, payload)
        for mode in ("rs-ag", "rs-ag-bidir", "xla-psum")
        for dtype_name in ("float32", "bfloat16")
        for direction in ("right", "left")
        for payload in (1, 50_000)
    ] + [(devices[:2], "rs-ag", "float32", "right", 4_096)]

    for devs, mode, dtype_name, direction, payload in sweep:
        dtype = jnp.dtype(dtype_name)
        case = build_case(
            jax=jax,
            jnp=jnp,
            devices=list(devs),
            axis_name="x",
            payload_bytes=payload,
            dtype=dtype,
            direction=direction,
            kernel_mode=mode,
            tile_rows=4,
            min_cols=8,
            ring_order="auto",
        )
        n = len(devs)
        assert case.optimal_bytes_per_device == 2 * (n - 1) * case.shard_payload_bytes
        assert case.actual_payload_bytes == n * case.shard_payload_bytes
        assert case.wire_bytes == case.optimal_bytes_per_device
        assert case.byte_model()["endpoint_send_plus_recv_bytes_per_device"] == 2 * case.wire_bytes
        if n == 4:
            assert abs(case.naive_over_optimal_ratio - 2.0) < 1e-9
        y = jax.block_until_ready(case.fn(case.x))
        ok = check_result(jax, jnp, y, case.expected_sums, dtype=dtype)

        # Independent numpy reference: float32 sum of the quantized input.
        x_np = np.asarray(jax.device_get(case.x)).astype(np.float32)
        ref = np.sum(x_np, axis=0)
        y_np = np.asarray(jax.device_get(y)).astype(np.float32)
        rtol, atol = default_tolerances(jnp, dtype)
        ref_ok = all(
            np.allclose(y_np[d], ref, rtol=rtol, atol=atol) for d in range(n)
        )
        markers_ok = np.allclose(
            np.asarray(observed_shard_markers(jax, y), dtype=np.float64)[0],
            np.asarray(expected_shard_marker_values(n), dtype=np.float64),
        )
        status = "ok" if (ok and ref_ok and markers_ok) else "FAIL"
        if status == "FAIL":
            failures += 1
        print(
            f"{status} mode={mode:11s} dtype={dtype_name:8s} dir={direction:5s} "
            f"n={n} payload={payload} tile={case.tile_rows}x{case.tile_cols} "
            f"wire={case.wire_bytes}B"
        )

    # Ring-order construction sanity on synthetic 2D coords.
    class _FakeDev:
        def __init__(self, i: int, coords: tuple[int, int, int]):
            self.id, self.coords = i, coords

    fake = [_FakeDev(i, (i % 2, i // 2, 0)) for i in range(4)]  # 2x2 grid
    ordered, reason = order_ring_devices(fake, "auto")
    path = [(d.coords[0], d.coords[1]) for d in ordered]
    if not _validate_ring(path, 4) or reason:
        failures += 1
        print(f"FAIL ring-order 2x2: path={path} reason={reason!r}")
    else:
        print(f"ok   ring-order 2x2: ids={[d.id for d in ordered]} path={path}")
    fake8 = [_FakeDev(i, (i % 4, i // 4, 0)) for i in range(8)]  # 4x2 grid
    ordered8, reason8 = order_ring_devices(fake8, "auto")
    path8 = [(d.coords[0], d.coords[1]) for d in ordered8]
    if not _validate_ring(path8, 8) or reason8:
        failures += 1
        print(f"FAIL ring-order 4x2: path={path8} reason={reason8!r}")
    else:
        print(f"ok   ring-order 4x2: path={path8}")

    print("selftest:", "PASS" if failures == 0 else f"{failures} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_selftest())
