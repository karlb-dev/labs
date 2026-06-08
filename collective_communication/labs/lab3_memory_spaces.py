"""Lab 3: local Pallas memory spaces before deeper communication.

This file is intentionally a teaching kernel, not a benchmark harness. The
surrounding runner owns command-line parsing, run directories, timing, profiling,
CSV/JSONL output, plots, and diagnostics. This file owns one idea:

    HBM input -> VMEM scratch -> simple arithmetic -> HBM output

Lab 1 and Lab 2 focused on remote movement between devices. Lab 3 removes all
cross-device communication so students can look closely at the local memory path
inside a TPU Pallas kernel.

Every device owns a local tile filled with a deterministic rank-plus-position
pattern. The Pallas kernel copies that tile from HBM into VMEM, computes
``y = x.astype(float32) * scale + bias`` in VMEM, and copies a float32 result
back to HBM.

Why this is Lab 3:
  * It gives students names for the memory hierarchy before collective
    algorithms become more complicated.
  * It shows that DMA semaphores are useful for local async copies, not only for
    remote DMA.
  * It establishes the local staging pattern used later for chunked and
    pipelined collectives.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import Any


# A conservative guardrail for the whole-tile VMEM scratch version of the lab.
# VMEM capacity depends on TPU generation and compiler allocation choices, but a
# small guardrail gives students an actionable message instead of a smoky
# low-level out-of-memory crash. Later labs should handle larger payloads by
# chunking tiles and reusing scratch buffers.
VMEM_SCRATCH_GUARDRAIL_BYTES = 16 * 1024 * 1024

# The kernel intentionally writes float32 output. That keeps the arithmetic
# numerically boring and makes byte accounting visible when the input is bf16.
OUTPUT_DTYPE_NAME = "float32"
OUTPUT_DTYPE_BYTES = 4


@dataclasses.dataclass(frozen=True)
class MemorySpacesCase:
    """A fully specified benchmark case for Lab 3.

    The benchmark harness expects each lab's ``build_case`` function to return a
    compact object containing:

    * ``x``: the sharded input array placed on devices.
    * ``fn``: a jitted function to benchmark.
    * ``expected``: a host-checkable expected result.
    * payload and tile metadata for logging.

    The byte properties below are teaching helpers. They make it easy to talk
    about bytes without asking students to reverse-engineer the tile math from a
    CSV row.
    """

    x: Any
    fn: Callable[[Any], Any]
    expected: Any
    actual_input_bytes: int
    actual_output_bytes: int
    tile_rows: int
    tile_cols: int
    input_dtype_name: str
    output_dtype_name: str
    scale: float
    bias: float

    @property
    def note(self) -> str:
        """Short note suitable for benchmark tables."""
        return (
            "input=HBM scratch=VMEM output=HBM "
            f"tile={self.tile_rows}x{self.tile_cols} "
            f"{self.input_dtype_name}->{self.output_dtype_name} "
            f"local_dma_bytes={self.local_dma_bytes_per_device} "
            f"scratch_bytes={self.scratch_bytes_per_device} "
            f"remote_bytes={self.remote_bytes_per_device} "
            f"y=x*{self.scale:g}+{self.bias:g}"
        )

    @property
    def elements_per_tile(self) -> int:
        """Number of elements in each device-local tile."""
        return self.tile_rows * self.tile_cols

    @property
    def logical_bytes_per_device(self) -> int:
        """First-order local byte model for this lab.

        The kernel reads one input tile from HBM and writes one output tile back
        to HBM. This property intentionally does not pretend to model VREG loads,
        compiler-internal traffic, or cache effects. It is the simple byte model
        students should start with.
        """
        return self.actual_input_bytes + self.actual_output_bytes

    @property
    def local_dma_bytes_per_device(self) -> int:
        """Estimated bytes moved by the two explicit local async copies."""
        return self.actual_input_bytes + self.actual_output_bytes

    @property
    def scratch_bytes_per_device(self) -> int:
        """Approximate VMEM scratch allocation requested by this Pallas call."""
        return self.actual_input_bytes + self.actual_output_bytes

    @property
    def remote_bytes_per_device(self) -> int:
        """Remote communication bytes. Lab 3 has no cross-device traffic."""
        return 0


def _ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division with an explicit name for teaching."""
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    return -(-numerator // denominator)


def _positive_int(value: int, *, name: str) -> int:
    """Normalize a user-provided integer flag and reject non-positive values."""
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return out


def normalize_arith_params(*, scale: float, bias: float) -> tuple[float, float]:
    """Return finite float arithmetic parameters or raise a clear error."""
    scale_f = float(scale)
    bias_f = float(bias)
    if not math.isfinite(scale_f):
        raise ValueError(f"scale must be finite, got {scale!r}")
    if not math.isfinite(bias_f):
        raise ValueError(f"bias must be finite, got {bias!r}")
    return scale_f, bias_f


def tile_shape_for_payload(
    *,
    payload_bytes: int,
    itemsize: int,
    tile_rows: int,
    min_cols: int,
) -> tuple[int, int, int]:
    """Choose a whole-tile shape for a requested per-device payload.

    ``payload_bytes`` is a request from the benchmark CLI. Low-level kernels
    work with concrete shapes, so this function rounds that request into a tile
    with ``rows * cols * itemsize >= payload_bytes``.

    The returned byte count is the actual input bytes per device after rounding.
    The output bytes can differ because this lab always writes float32 output.
    """
    requested = max(1, int(payload_bytes))
    itemsize_i = _positive_int(itemsize, name="itemsize")
    rows = _positive_int(tile_rows, name="tile_rows")
    min_cols_i = _positive_int(min_cols, name="min_cols")

    cols_for_requested_payload = _ceil_div(requested, itemsize_i * rows)
    cols = max(min_cols_i, cols_for_requested_payload)
    actual_payload_bytes = rows * cols * itemsize_i
    return rows, cols, actual_payload_bytes


def output_bytes_for_tile(*, rows: int, cols: int) -> int:
    """Return the float32 output bytes per device for a tile shape."""
    return _positive_int(rows, name="rows") * _positive_int(cols, name="cols") * OUTPUT_DTYPE_BYTES


def validate_devices(devices: list[Any]) -> None:
    """Fail early with a lab-shaped error instead of a cryptic sharding error."""
    if not devices:
        raise ValueError("Lab 3 requires at least one JAX device")


def validate_vmem_scratch_budget(
    *,
    input_bytes: int,
    output_bytes: int,
    guardrail_bytes: int = VMEM_SCRATCH_GUARDRAIL_BYTES,
) -> None:
    """Fail early when the whole-tile VMEM scratch request is too large.

    Lab 3 allocates two whole-tile VMEM buffers: one for the input and one for
    the float32 output. That is perfect for learning the HBM -> VMEM -> HBM path.
    It is not a scalable design for arbitrarily large payloads. Larger tiles
    should be handled with chunking and reused scratch buffers in later labs.
    """
    total = int(input_bytes) + int(output_bytes)
    limit = int(guardrail_bytes)
    if limit <= 0 or total <= limit:
        return

    def mib(num_bytes: int) -> float:
        return num_bytes / 1024 / 1024

    raise ValueError(
        "Lab 3 whole-tile VMEM scratch would exceed the teaching guardrail: "
        f"input={mib(input_bytes):.1f} MiB, "
        f"output={mib(output_bytes):.1f} MiB, "
        f"total={mib(total):.1f} MiB, "
        f"guardrail={mib(limit):.1f} MiB. "
        "Use a smaller --sizes value for Lab 3, or save larger payloads for "
        "the chunked/pipelined labs where scratch buffers are reused."
    )


def make_rank_tile(
    jnp: Any,
    n_devices: int,
    rows: int,
    cols: int,
    dtype: Any,
) -> Any:
    """Create a simple rank-valued tile for compatibility with early tests."""
    if n_devices <= 0:
        raise ValueError(f"n_devices must be positive, got {n_devices}")
    ranks = jnp.arange(n_devices, dtype=jnp.float32).reshape(n_devices, 1, 1)
    x = jnp.broadcast_to(ranks, (n_devices, rows, cols))
    return x.astype(dtype)


def make_patterned_tile(
    jnp: Any,
    n_devices: int,
    rows: int,
    cols: int,
    dtype: Any,
) -> Any:
    """Create a rank-tagged and position-tagged tile for stronger checking.

    A tile filled only with the rank is perfect for ownership tests in Labs 1 and
    2, but it is too weak for Lab 3. A partial local copy can look correct if
    every element contains the same value. This pattern keeps the first element
    equal to the rank while adding binary-friendly row and column offsets.
    """
    if n_devices <= 0:
        raise ValueError(f"n_devices must be positive, got {n_devices}")
    rows_i = _positive_int(rows, name="rows")
    cols_i = _positive_int(cols, name="cols")

    ranks = jnp.arange(n_devices, dtype=jnp.float32).reshape(n_devices, 1, 1)
    row_offsets = jnp.arange(rows_i, dtype=jnp.float32).reshape(1, rows_i, 1) * 0.25
    col_offsets = (
        (jnp.arange(cols_i, dtype=jnp.float32) % 8).reshape(1, 1, cols_i) * 0.03125
    )
    x = ranks + row_offsets + col_offsets
    return x.astype(dtype)


def local_arith_reference(value: Any, *, scale: float, bias: float) -> Any:
    """Reference computation for the local read/compute/write path."""
    import jax.numpy as jnp

    scale_f, bias_f = normalize_arith_params(scale=scale, bias=bias)
    return value.astype(jnp.float32) * scale_f + bias_f


def expected_scalars_by_device(
    jnp: Any,
    *,
    n_devices: int,
    scale: float,
    bias: float,
) -> Any:
    """Return one expected first scalar per device for visual debugging."""
    scale_f, bias_f = normalize_arith_params(scale=scale, bias=bias)
    ranks = jnp.arange(n_devices, dtype=jnp.float32)
    return ranks * scale_f + bias_f


def vmem_arith_kernel(
    x_ref,
    o_ref,
    input_scratch_ref,
    output_scratch_ref,
    copy_in_sem,
    copy_out_sem,
    *,
    scale: float,
    bias: float,
) -> None:
    """Pallas TPU kernel: HBM input -> VMEM scratch -> HBM output.

    Parameters are Pallas ``Ref`` objects rather than ordinary JAX arrays:

    * ``x_ref`` names this device's input tile in HBM.
    * ``o_ref`` names this device's output tile in HBM.
    * ``input_scratch_ref`` names a VMEM scratch tile with the input dtype.
    * ``output_scratch_ref`` names a VMEM scratch tile with float32 dtype.
    * ``copy_in_sem`` tracks the local HBM -> VMEM async copy.
    * ``copy_out_sem`` tracks the local VMEM -> HBM async copy.

    No cross-device communication happens here. Every device runs the same local
    program on its own shard.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    # Stage 1: local HBM -> local VMEM.
    #
    # ``make_async_copy`` creates a descriptor. We start and immediately wait in
    # Lab 3 so the ordering is simple and visible. Later labs can place useful
    # work between start and wait.
    with jax.named_scope("lab3_hbm_to_vmem"):
        copy_in = pltpu.make_async_copy(x_ref, input_scratch_ref, copy_in_sem)
        copy_in.start()
        copy_in.wait()

    # Stage 2: compute from VMEM into another VMEM buffer.
    with jax.named_scope("lab3_vmem_compute"):
        output_scratch_ref[...] = (
            input_scratch_ref[...].astype(jnp.float32) * scale + bias
        )

    # Stage 3: local VMEM -> local HBM.
    with jax.named_scope("lab3_vmem_to_hbm"):
        copy_out = pltpu.make_async_copy(output_scratch_ref, o_ref, copy_out_sem)
        copy_out.start()
        copy_out.wait()


def _get_shard_map(jax_module: Any) -> Callable[..., Any]:
    """Return a public ``jax.shard_map`` when available, with fallbacks."""
    public = getattr(jax_module, "shard_map", None)
    if callable(public):
        return public

    try:  # pragma: no cover, depends on installed JAX version.
        from jax.experimental.shard_map import shard_map as experimental_shard_map

        return experimental_shard_map
    except Exception:  # pragma: no cover
        from jax._src import shard_map as shard_map_module  # type: ignore

        return shard_map_module.shard_map


def _hbm_memory_space(pl: Any, pltpu: Any) -> Any:
    """Return the memory-space object for HBM-like input and output refs.

    The kernel stages these refs through VMEM scratch with its own
    ``make_async_copy`` calls, so the input/output operands must stay
    HBM-resident and *non-windowed*. The right memory space for that is
    ``MemorySpace.ANY`` (a.k.a. ``pl.ANY``): Pallas hands the kernel the HBM ref
    directly instead of trying to copy a window into VMEM. Passing the explicit
    ``MemorySpace.HBM`` makes Mosaic treat the operand as a window and fail with
    "Operand windows can only be requested in VMEM or SMEM".
    """
    return pl.ANY


def vmem_arith(
    x: Any,
    *,
    mesh: Any,
    axis_name: str,
    scale: float,
    bias: float,
) -> Any:
    """Shard-mapped Pallas call for local VMEM scratch arithmetic.

    ``shard_map`` maps the same local Pallas call over each device shard. It is
    not being used as a collective here. There is no remote DMA and no
    cross-device semaphore. The mesh only tells JAX how the input and output are
    partitioned across devices.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    if axis_name not in tuple(mesh.axis_names):
        raise ValueError(f"axis_name {axis_name!r} not found in mesh axes {mesh.axis_names!r}")

    scale_f, bias_f = normalize_arith_params(scale=scale, bias=bias)
    partition = jax.sharding.PartitionSpec
    shard_map_fn = _get_shard_map(jax)
    hbm = _hbm_memory_space(pl, pltpu)

    def local_call(x_shard):
        # ``x_shard.shape`` is the exact local shape seen by one device. Scratch
        # is allocated for that shape so we do not accidentally drop dimensions
        # when JAX changes how it presents mapped shards.
        local_shape = tuple(x_shard.shape)
        out_shape = jax.ShapeDtypeStruct(local_shape, jnp.float32)

        return pl.pallas_call(
            functools.partial(vmem_arith_kernel, scale=scale_f, bias=bias_f),
            out_shape=out_shape,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(),
                has_side_effects=False,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                scratch_shapes=(
                    pltpu.VMEM(local_shape, x_shard.dtype),
                    pltpu.VMEM(local_shape, jnp.float32),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
                in_specs=[pl.BlockSpec(memory_space=hbm)],
                out_specs=pl.BlockSpec(memory_space=hbm),
            ),
        )(x_shard)

    try:
        return shard_map_fn(
            local_call,
            mesh=mesh,
            in_specs=partition(axis_name),
            out_specs=partition(axis_name),
            check_vma=False,
        )(x)
    except TypeError as exc:
        # Very old course images used a different spelling. Do not hide other
        # TypeErrors, because those are usually real Pallas or shape bugs.
        if "check_vma" not in str(exc):
            raise
        return shard_map_fn(
            local_call,
            mesh=mesh,
            in_specs=partition(axis_name),
            out_specs=partition(axis_name),
            check_rep=False,
        )(x)


def build_case(
    *,
    jax: Any,
    jnp: Any,
    devices: list[Any],
    axis_name: str,
    payload_bytes: int,
    dtype: Any,
    tile_rows: int,
    min_cols: int,
    scale: float,
    bias: float,
) -> MemorySpacesCase:
    """Build input, expected output, and jitted function for Lab 3."""
    import numpy as np

    validate_devices(devices)
    scale_f, bias_f = normalize_arith_params(scale=scale, bias=bias)

    mesh = jax.sharding.Mesh(np.array(devices), (axis_name,))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(axis_name)
    )

    input_dtype = jnp.dtype(dtype)
    output_dtype = jnp.dtype(jnp.float32)
    itemsize = int(input_dtype.itemsize)

    rows, cols, actual_input_bytes = tile_shape_for_payload(
        payload_bytes=payload_bytes,
        itemsize=itemsize,
        tile_rows=tile_rows,
        min_cols=min_cols,
    )
    actual_output_bytes = output_bytes_for_tile(rows=rows, cols=cols)
    validate_vmem_scratch_budget(
        input_bytes=actual_input_bytes,
        output_bytes=actual_output_bytes,
    )

    # Host-level shape: [device, row, col]. NamedSharding with PartitionSpec(axis)
    # splits the leading dimension across devices, so each TPU sees one local
    # [row, col] tile in the common one-axis mesh case.
    x_host = make_patterned_tile(jnp, len(devices), rows, cols, dtype)
    x = jax.device_put(x_host, sharding)

    expected = local_arith_reference(x_host, scale=scale_f, bias=bias_f)

    def fn(value):
        return vmem_arith(
            value,
            mesh=mesh,
            axis_name=axis_name,
            scale=scale_f,
            bias=bias_f,
        )

    return MemorySpacesCase(
        x=x,
        fn=jax.jit(fn),
        expected=expected,
        actual_input_bytes=actual_input_bytes,
        actual_output_bytes=actual_output_bytes,
        tile_rows=rows,
        tile_cols=cols,
        input_dtype_name=str(input_dtype),
        output_dtype_name=str(output_dtype),
        scale=scale_f,
        bias=bias_f,
    )


def observed_tile_scalars(jax: Any, y: Any) -> Any:
    """Return ``y[:, 0, 0]`` on the host for quick debugging."""
    return jax.device_get(y)[:, 0, 0]


def observed_tile_summary(jax: Any, y: Any, *, max_devices: int = 4) -> list[dict[str, float]]:
    """Return a compact host-side summary for notebooks or live teaching."""
    y_host = jax.device_get(y)
    rows = []
    for device_idx in range(min(int(max_devices), y_host.shape[0])):
        tile = y_host[device_idx]
        rows.append(
            {
                "device": float(device_idx),
                "first": float(tile[0, 0]),
                "last": float(tile[-1, -1]),
                "mean": float(tile.mean()),
            }
        )
    return rows


def max_abs_error(jax: Any, jnp: Any, y: Any, expected: Any) -> float:
    """Return the maximum absolute full-tile error as a Python float."""
    y_host = jax.device_get(y)
    expected_host = jax.device_get(expected)
    return float(jnp.max(jnp.abs(y_host - expected_host)))


def check_result(jax: Any, jnp: Any, y: Any, expected: Any) -> bool:
    """Validate the full output tile on every device.

    Checking only ``y[:, 0, 0]`` is tempting because rank-coded inputs make that
    scalar easy to read. Lab 3 is a memory movement lab, though, so a partial
    copy or partial write must fail even if the first scalar looks correct.
    """
    y_host = jax.device_get(y)
    expected_host = jax.device_get(expected)
    if getattr(y_host, "shape", None) != getattr(expected_host, "shape", None):
        return False
    return bool(jnp.allclose(y_host, expected_host, rtol=1e-3, atol=1e-3))
