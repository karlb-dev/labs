"""Lab 10: multi-host smoke, process collectives, and hierarchy planning.

This file intentionally contains only the concept code for Lab 10. The benchmark
harness owns run directories, timing rows, CSV/JSONL output, plotting, and
profiler capture.

Lab 10 is different from Labs 1-9:

* It is not a new Pallas remote-DMA kernel.
* It is a launch-control and topology-truth lab.
* It verifies that every Python process can see the same global device world.
* It records the hierarchy plan needed before future multi-host collectives.

The happy path works in both single-process and multi-process mode. On one
process, it is a degenerate smoke test. On a real multi-host launch, it becomes
a gatekeeper: all processes must enter the same sync/all-gather operations and
produce a consistent view of process topology.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from collections import Counter
from typing import Any

try:
    from labs import lab_spec_utils
except Exception:  # pragma: no cover - standalone reading/smoke-test fallback.
    class _FallbackLabSpecUtils:
        """Tiny fallback so this teaching file can be imported by itself.

        The real benchmark repository provides `labs.lab_spec_utils`. This
        fallback is intentionally small and only supports local smoke tests of
        this module outside the repo package layout.
        """

        @staticmethod
        def build_spec(
            base: dict[str, Any],
            *,
            args: Any,
            payload_bytes: int,
            n_devices: int,
        ) -> dict[str, Any]:
            return {
                **base,
                "payload_bytes": int(payload_bytes),
                "n_devices": int(n_devices),
                "args_available": args is not None,
            }

        @staticmethod
        def render_markdown(spec: dict[str, Any]) -> str:
            title = spec.get("title", "Lab Spec")
            goal = spec.get("goal", "")
            op = spec.get("op", "unknown")
            ok = spec.get("ok", "unknown")
            return f"# {title}\n\n{goal}\n\n- op: `{op}`\n- ok: `{ok}`\n"

    lab_spec_utils = _FallbackLabSpecUtils()


# Environment variables that commonly explain distributed launches. Some are
# JAX/PJRT-specific, some are Cloud TPU hints, and some are generic cluster
# launcher variables. We record missing values too: a `None` can be the clue.
PROCESS_ENV_KEYS = (
    "JAX_COORDINATOR_ADDRESS",
    "JAX_COORDINATOR_PORT",
    "JAX_PROCESS_COUNT",
    "JAX_PROCESS_INDEX",
    "JAX_LOCAL_DEVICE_IDS",
    "JAX_PLATFORMS",
    "JAX_PLATFORM_NAME",
    "PJRT_DEVICE",
    "PJRT_LOCAL_PROCESS_RANK",
    "PJRT_LOCAL_PROCESS_COUNT",
    "CLOUD_TPU_TASK_ID",
    "TPU_NAME",
    "TPU_ZONE",
    "TPU_WORKER_ID",
    "TPU_WORKER_HOSTNAMES",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_HOST_BOUNDS",
    "TPU_MESH_CONTROLLER_ADDRESS",
    "TPU_MESH_CONTROLLER_PORT",
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
)

# A large stride makes each process's payload visually distinct in previews.
PROCESS_PAYLOAD_STRIDE = 1_000_003

# Columns gathered by the small process fact all-gather.
PROCESS_FACT_COLUMNS = (
    "process_index",
    "process_count",
    "local_device_count",
    "global_device_count",
)


LAB_SPEC: dict[str, Any] = {
    "lab": "lab10",
    "title": "Lab 10: Multi-Host Smoke And Hierarchy",
    "goal": (
        "Move from one-process/local-device thinking to process topology, "
        "global devices, launch discipline, process collectives, and "
        "hierarchical collective plans."
    ),
    "implemented_ops": [
        "`lab10_topology_smoke`: records local/global/process device facts",
        "`lab10_process_collective_smoke`: syncs processes and all-gathers small payloads",
        "`lab10_multihost_spec`: records launch and hierarchy planning artifacts",
    ],
    "deferred_ops": [
        "Call `jax.distributed.initialize()` in the outer runner before device access when auto-init is absent",
        "Run Pallas remote-DMA collectives over multi-host global meshes",
        "Write a process-0 merged summary from all process-local artifacts",
        "Implement hierarchical reduce-scatter/all-gather over process groups",
    ],
    "byte_model": [
        "process smoke payloads are control-plane validation, not a TPU bandwidth benchmark",
        "hierarchical collectives should report process-local bytes separately from cross-process bytes",
        "future multi-host Pallas work should label every byte by phase and scope",
    ],
    "pass_condition": [
        "all requested expectation checks pass",
        "process index is in range",
        "global-device process indices are dense when metadata is available",
        "process all-gather returns one contribution per process",
        "spec artifact separates process-local and cross-process phases",
    ],
    "artifacts": [
        "run_metadata.json",
        "diagnostics/runtime.json",
        "lab_artifacts/*lab10_topology_smoke*",
        "lab_artifacts/*lab10_process_collective_smoke*",
        "lab_artifacts/*lab10_multihost_spec*",
        "logs/console.log",
    ],
    "next_steps": [
        "Run this smoke under a real multi-host TPU launcher",
        "Use the recorded process groups to design hierarchical reduce-scatter",
        "Extend Lab 9 staged mesh collectives across process boundaries",
    ],
}


# ---------------------------------------------------------------------------
# JSON-safe host/device helpers
# ---------------------------------------------------------------------------


def _safe_int(value: Any) -> int | None:
    """Best-effort conversion for device/runtime attributes."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return repr(value)


def _normalize_coords(coords: Any) -> list[int] | None:
    """Return TPU coordinates as a JSON-safe list, when available."""

    if coords is None:
        return None
    try:
        return [int(x) for x in coords]
    except Exception:
        return None


def _env_report() -> dict[str, str | None]:
    """Capture distributed-launch hints from the process environment."""

    return {key: os.environ.get(key) for key in PROCESS_ENV_KEYS}


def _runtime_report() -> dict[str, Any]:
    """Small host-side report for correlating per-process logs."""

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "python_version": platform.python_version(),
    }


def _device_record(device: Any, *, local: bool | None = None) -> dict[str, Any]:
    """Convert a JAX device object into a stable, JSON-friendly record.

    Important fields for this course:

    * `id`: global-ish device id exposed by JAX.
    * `process_index`: which Python process owns the device.
    * `coords`: TPU topology coordinates, when the backend exposes them.
    * `core_on_chip`: useful on TPU generations that expose chip/core details.
    * `local`: whether the record came from `jax.local_devices()`.
    """

    return {
        "id": _safe_int(getattr(device, "id", None)),
        "process_index": _safe_int(getattr(device, "process_index", None)),
        "platform": _safe_str(getattr(device, "platform", None)),
        "device_kind": _safe_str(getattr(device, "device_kind", None)),
        "coords": _normalize_coords(getattr(device, "coords", None)),
        "core_on_chip": _safe_int(getattr(device, "core_on_chip", None)),
        "local_hardware_id": _safe_int(getattr(device, "local_hardware_id", None)),
        "local": local,
        "repr": repr(device),
    }


def _device_sort_key(device: dict[str, Any]) -> tuple[int, int, tuple[int, ...]]:
    """Sort devices in a readable order for artifacts."""

    process_index = device.get("process_index")
    device_id = device.get("id")
    coords = device.get("coords") or []
    return (
        10**9 if process_index is None else int(process_index),
        10**9 if device_id is None else int(device_id),
        tuple(int(x) for x in coords),
    )


def _devices_by_process(devices: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group global device records by owning process index."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for device in sorted(devices, key=_device_sort_key):
        process_index = device.get("process_index")
        key = "unknown" if process_index is None else str(process_index)
        grouped.setdefault(key, []).append(device)
    return grouped


def _process_group_summaries(
    devices_by_process: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    """Compact process -> device summary for first-pass artifact reading."""

    summaries: dict[str, dict[str, Any]] = {}
    for process, records in sorted(devices_by_process.items()):
        coords = [record.get("coords") for record in records if record.get("coords")]
        summaries[process] = {
            "device_count": len(records),
            "device_ids": [record.get("id") for record in records],
            "device_kinds": sorted(
                {str(record.get("device_kind")) for record in records if record.get("device_kind")}
            ),
            "platforms": sorted(
                {str(record.get("platform")) for record in records if record.get("platform")}
            ),
            "coords_preview": coords[:8],
        }
    return summaries


def _coordinate_bounds(devices: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarize TPU coordinate ranges, if coordinate metadata exists."""

    coords = [device.get("coords") for device in devices if device.get("coords")]
    if not coords:
        return None
    rank = max(len(coord) for coord in coords)
    padded = [coord + [0] * (rank - len(coord)) for coord in coords]
    return {
        "rank": rank,
        "min": [min(coord[axis] for coord in padded) for axis in range(rank)],
        "max": [max(coord[axis] for coord in padded) for axis in range(rank)],
        "unique_count": len({tuple(coord) for coord in padded}),
    }


def _ids(records: list[dict[str, Any]]) -> set[int]:
    return {
        int(record["id"])
        for record in records
        if record.get("id") is not None
    }


def _process_indices_from_devices(devices: list[dict[str, Any]]) -> list[int]:
    indices = {
        int(device["process_index"])
        for device in devices
        if device.get("process_index") is not None
    }
    return sorted(indices)


def _dense_zero_based(indices: list[int]) -> bool | None:
    """Return whether process indices look like 0..N-1.

    `None` means device metadata did not expose process indices.
    """

    if not indices:
        return None
    return indices == list(range(max(indices) + 1))


def _topology_summary(
    *,
    local_devices: list[dict[str, Any]],
    global_devices: list[dict[str, Any]],
    process_count: int,
) -> dict[str, Any]:
    """Compute derived topology facts for students to inspect."""

    process_indices = _process_indices_from_devices(global_devices)
    devices_per_process = Counter(
        str(device.get("process_index"))
        for device in global_devices
        if device.get("process_index") is not None
    )
    local_ids = _ids(local_devices)
    global_ids = _ids(global_devices)
    local_subset = local_ids.issubset(global_ids) if local_ids and global_ids else None

    kinds = sorted(
        {
            str(device.get("device_kind"))
            for device in global_devices
            if device.get("device_kind") is not None
        }
    )
    platforms = sorted(
        {
            str(device.get("platform"))
            for device in global_devices
            if device.get("platform") is not None
        }
    )

    return {
        "process_indices_from_devices": process_indices,
        "process_indices_dense_zero_based": _dense_zero_based(process_indices),
        "process_count_from_api": int(process_count),
        "process_count_from_device_metadata": len(process_indices) or None,
        "devices_per_process": dict(sorted(devices_per_process.items())),
        "local_device_ids": sorted(local_ids),
        "global_device_ids": sorted(global_ids),
        "local_devices_subset_of_global_devices": local_subset,
        "coordinate_bounds": _coordinate_bounds(global_devices),
        "device_kinds": kinds,
        "platforms": platforms,
        "all_global_devices_have_process_index": all(
            device.get("process_index") is not None for device in global_devices
        ),
        "all_global_devices_have_coords": all(
            device.get("coords") is not None for device in global_devices
        ),
    }


# ---------------------------------------------------------------------------
# Checks, plans, and byte worksheets
# ---------------------------------------------------------------------------


def _ok_when_known(value: bool | None) -> bool:
    """Treat unknown metadata checks as non-failing."""

    return True if value is None else bool(value)


def _failed_checks(checks: dict[str, bool | None]) -> list[str]:
    return [name for name, value in checks.items() if value is False]


def _unknown_checks(checks: dict[str, bool | None]) -> list[str]:
    return [name for name, value in checks.items() if value is None]


def _checks_pass(checks: dict[str, bool | None]) -> bool:
    return all(_ok_when_known(value) for value in checks.values())


def _expected_value(args: Any, name: str) -> int | None:
    """Read optional integer expectation flags from the harness args object."""

    value = getattr(args, name, None)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer or None, got {value!r}") from exc


def _expectation_checks(
    *,
    process_index: int,
    process_count: int,
    local_device_count: int,
    global_device_count: int,
    n_devices_from_runner: int,
    topology_summary: dict[str, Any],
    expected_process_count: int | None,
    expected_global_devices: int | None,
) -> dict[str, bool | None]:
    """Build the core Lab 10 topology check table."""

    return {
        "local_device_count_matches_runner": local_device_count == n_devices_from_runner,
        "process_index_in_range": 0 <= process_index < process_count,
        "global_device_count_ge_local_device_count": global_device_count >= local_device_count,
        "local_devices_subset_of_global_devices": topology_summary.get(
            "local_devices_subset_of_global_devices"
        ),
        "process_indices_dense_zero_based": topology_summary.get(
            "process_indices_dense_zero_based"
        ),
        "expected_process_count": (
            expected_process_count is None or process_count == expected_process_count
        ),
        "expected_global_devices": (
            expected_global_devices is None or global_device_count == expected_global_devices
        ),
    }


def _hierarchy_byte_model(
    *,
    payload_bytes: int,
    process_count: int,
    local_device_count: int,
    global_device_count: int,
) -> dict[str, Any]:
    """Return a planning worksheet for labeling hierarchy bytes.

    These numbers are intentionally marked as estimates. They should not be
    reported as measured bandwidth. The point is to split bytes by scope before
    building a cross-host collective.
    """

    payload_bytes = max(0, int(payload_bytes))
    process_count = max(1, int(process_count))
    local_device_count = max(1, int(local_device_count))
    global_device_count = max(1, int(global_device_count))
    host_block_bytes = payload_bytes * local_device_count
    return {
        "kind": "planning_estimate_not_measurement",
        "payload_bytes_per_device": payload_bytes,
        "process_count": process_count,
        "local_device_count_this_process": local_device_count,
        "global_device_count": global_device_count,
        "host_block_bytes_this_process": host_block_bytes,
        "flat_ring_all_gather_send_bytes_per_device": payload_bytes * max(0, global_device_count - 1),
        "local_phase_all_gather_send_bytes_per_device": payload_bytes * max(0, local_device_count - 1),
        "hierarchical_all_gather_cross_process_upper_model": host_block_bytes * max(0, process_count - 1),
        "ring_all_reduce_send_bytes_per_device_model": (
            2 * max(0, global_device_count - 1) * payload_bytes // global_device_count
        ),
        "labels_to_keep_separate": [
            "process_local_bytes",
            "cross_process_bytes",
            "process_local_restore_bytes",
        ],
        "single_process_note": (
            "cross-process estimates degenerate to zero-peer planning"
            if process_count == 1
            else "cross-process phase has real peers"
        ),
        "teaching_note": "label bytes by scope before claiming a hierarchy is faster",
    }


def _hierarchical_collective_examples() -> list[dict[str, Any]]:
    """Examples students can copy into capstone planning docs."""

    return [
        {
            "collective": "all-reduce",
            "phase_1": "local reduce-scatter or local reduce inside each process",
            "phase_2": "cross-process reduction over host blocks or representatives",
            "phase_3": "local all-gather, broadcast, or fanout inside each process",
            "question": "which phase carries the largest payload on this topology?",
        },
        {
            "collective": "all-gather",
            "phase_1": "local gather on each process",
            "phase_2": "cross-process exchange of process-local blocks",
            "phase_3": "local canonical layout restore or fanout",
            "question": "does axis order change the cross-process message size?",
        },
        {
            "collective": "reduce-scatter",
            "phase_1": "local partial reductions into owner chunks",
            "phase_2": "cross-process exchange/reduce for matching owner chunks",
            "phase_3": "optional local scatter to final owner devices",
            "question": "can the output stay sharded to avoid a final all-gather?",
        },
        {
            "collective": "all-to-all or token exchange",
            "phase_1": "bucket or pad local tokens by destination process",
            "phase_2": "cross-process token exchange",
            "phase_3": "local destination-device scatter and combine",
            "question": "how much padding or skew does the cross-process phase carry?",
        },
    ]


def _hierarchy_plan(
    *,
    process_count: int,
    local_device_count: int,
    global_device_count: int,
    payload_bytes: int = 0,
) -> dict[str, Any]:
    """Plan future hierarchical collectives using observed process facts."""

    single_process = process_count == 1
    return {
        "single_process_degenerates_to_local": single_process,
        "process_count": int(process_count),
        "local_device_count_this_process": int(local_device_count),
        "global_device_count": int(global_device_count),
        "phases": [
            {
                "name": "intra_process",
                "scope": "local devices addressable by this Python process",
                "participants_this_process": int(local_device_count),
                "purpose": "do process-local reduction, gather, bucketing, or layout work first",
                "byte_label": "process_local_bytes",
            },
            {
                "name": "inter_process",
                "scope": "one or more representatives or owner shards from each process",
                "process_count": int(process_count),
                "purpose": "move only the data that must cross process boundaries",
                "byte_label": "cross_process_bytes",
            },
            {
                "name": "intra_process_fanout_or_restore",
                "scope": "local devices addressable by this Python process",
                "participants_this_process": int(local_device_count),
                "purpose": "replicate, scatter, or restore final layout for local consumers",
                "byte_label": "process_local_restore_bytes",
            },
        ],
        "byte_model": _hierarchy_byte_model(
            payload_bytes=payload_bytes,
            process_count=process_count,
            local_device_count=local_device_count,
            global_device_count=global_device_count,
        ),
        "examples": _hierarchical_collective_examples(),
        "student_questions": [
            "Which phase owns each output shard?",
            "Which bytes stay inside the process-local group?",
            "Which bytes cross process boundaries?",
            "Can the cross-process phase move reduced shards instead of full tensors?",
            "What artifact would prove that the expected process topology actually launched?",
        ],
    }


def _mesh_recommendations(
    *,
    process_count: int,
    local_device_count: int,
    global_device_count: int,
) -> list[dict[str, Any]]:
    """Suggest logical meshes to try after this smoke test passes."""

    recommendations = [
        {
            "name": f"flat_{max(1, int(global_device_count))}",
            "shape": [max(1, int(global_device_count))],
            "use_when": "baseline flat collective comparison",
        }
    ]
    if process_count > 1 and local_device_count > 0:
        recommendations.append(
            {
                "name": "process_x_local_device",
                "shape": [int(process_count), int(local_device_count)],
                "axes": ["process", "local_device"],
                "use_when": "hierarchical collectives with local and cross-process phases",
            }
        )
    elif local_device_count > 1:
        recommendations.append(
            {
                "name": "single_process_local_devices",
                "shape": [int(local_device_count)],
                "axes": ["local_device"],
                "use_when": "single-host validation before multi-host launch",
            }
        )
    return recommendations


def _launch_plan(process_count: int) -> dict[str, Any]:
    """Record launch rules that are easy to verify from artifacts."""

    return {
        "single_process_command": "python collective_comm_bench/collective_bench.py --lab lab10",
        "multi_process_rule": (
            "run the same command on every process after JAX distributed "
            "initialization is configured by the TPU launcher or by explicit "
            "jax.distributed.initialize(...) arguments"
        ),
        "initialization_rule": (
            "jax.distributed.initialize() must run before jax.devices(), "
            "jax.local_devices(), or device computation in a multi-host job"
        ),
        "all_processes_must_match": [
            "script path and arguments",
            "coordinator address and port",
            "process count",
            "dense process ids from 0 to process_count - 1",
            "global device visibility",
            "collective call order",
        ],
        "process_0_should_write": [
            "merged run summary",
            "global topology summary",
            "class handout artifact links",
        ],
        "every_process_should_write": [
            "local console log",
            "topology smoke artifact",
            "process collective artifact",
        ],
        "current_process_count": int(process_count),
        "needs_multi_process_launch_for_true_multihost": process_count == 1,
    }


def _capstone_report_checklist() -> list[str]:
    return [
        "observed process_count and global_device_count",
        "per-process local_device_count table",
        "device coords and process ownership map when available",
        "phase diagram separating local and cross-process movement",
        "byte model per phase",
        "correctness check that runs on every process",
        "profiler or timing artifact for the slowest phase",
        "one surprising performance result and a topology-based explanation",
    ]


def _operator_checklist() -> list[str]:
    return [
        "distributed JAX initialized before jax.devices() or device computation",
        "same script path and arguments on every process",
        "no process_index-dependent branch skips a collective",
        "per-process logs are saved before process-0 summary generation",
        "expectation flags match the intended slice",
        "single-process smoke is not mistaken for multi-host validation",
    ]


# ---------------------------------------------------------------------------
# Artifact builders imported by the benchmark harness
# ---------------------------------------------------------------------------


def build_topology_smoke(
    *,
    jax: Any,
    args: Any,
    payload_bytes: int,
    n_devices: int,
) -> dict[str, Any]:
    """Build the topology smoke artifact for Lab 10.

    This function intentionally does not call `jax.distributed.initialize()`.
    In a real multi-host runner, initialization must happen before this function
    is reached, because the runner has usually already imported JAX and may have
    enumerated devices.
    """

    local_devices = [_device_record(device, local=True) for device in jax.local_devices()]
    global_devices = [_device_record(device, local=None) for device in jax.devices()]
    devices_by_process = _devices_by_process(global_devices)

    process_index = int(jax.process_index())
    process_count = int(jax.process_count())
    local_device_count = len(local_devices)
    global_device_count = len(global_devices)
    expected_process_count = _expected_value(args, "lab10_expected_process_count")
    expected_global_devices = _expected_value(args, "lab10_expected_global_devices")
    topology_summary = _topology_summary(
        local_devices=local_devices,
        global_devices=global_devices,
        process_count=process_count,
    )
    checks = _expectation_checks(
        process_index=process_index,
        process_count=process_count,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        n_devices_from_runner=int(n_devices),
        topology_summary=topology_summary,
        expected_process_count=expected_process_count,
        expected_global_devices=expected_global_devices,
    )
    hierarchy_byte_model = _hierarchy_byte_model(
        payload_bytes=payload_bytes,
        process_count=process_count,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
    )

    spec = lab_spec_utils.build_spec(
        LAB_SPEC,
        args=args,
        payload_bytes=payload_bytes,
        n_devices=n_devices,
    )
    spec.update(
        {
            "op": "lab10_topology_smoke",
            "ok": _checks_pass(checks),
            "process_index": process_index,
            "process_count": process_count,
            "is_process_0": process_index == 0,
            "local_device_count": local_device_count,
            "global_device_count": global_device_count,
            "local_devices": sorted(local_devices, key=_device_sort_key),
            "global_devices": sorted(global_devices, key=_device_sort_key),
            "devices_by_process": devices_by_process,
            "process_group_summaries": _process_group_summaries(devices_by_process),
            "topology_summary": topology_summary,
            "single_process": process_count == 1,
            "needs_distributed_launch_for_true_multihost": process_count == 1,
            "distributed_env": _env_report(),
            "runtime": _runtime_report(),
            "checks": checks,
            "failed_checks": _failed_checks(checks),
            "unknown_checks": _unknown_checks(checks),
            "expected_process_count": expected_process_count,
            "expected_global_devices": expected_global_devices,
            "hierarchy_plan": _hierarchy_plan(
                process_count=process_count,
                local_device_count=local_device_count,
                global_device_count=global_device_count,
                payload_bytes=payload_bytes,
            ),
            "hierarchy_byte_model": hierarchy_byte_model,
            "mesh_recommendations": _mesh_recommendations(
                process_count=process_count,
                local_device_count=local_device_count,
                global_device_count=global_device_count,
            ),
            "launch_plan": _launch_plan(process_count),
            "operator_checklist": _operator_checklist(),
        }
    )
    return spec


# ---------------------------------------------------------------------------
# Process collective helpers
# ---------------------------------------------------------------------------


def _lab10_sync_name(*, process_count: int, elems: int) -> str:
    return f"collective_comm_bench_lab10_p{process_count}_e{elems}"


def _process_payload(jnp: Any, *, process_index: int, elems: int) -> Any:
    """JAX int32 payload contributed by one process."""

    return (
        jnp.arange(elems, dtype=jnp.int32)
        + jnp.int32(process_index * PROCESS_PAYLOAD_STRIDE)
    )


def _rank_marked_payload(np: Any, *, process_index: int, elems: int) -> Any:
    """NumPy version of one process's expected payload."""

    return (
        np.arange(elems, dtype=np.int32)
        + np.int32(process_index * PROCESS_PAYLOAD_STRIDE)
    )


def _expected_process_allgather(np: Any, *, process_count: int, elems: int) -> Any:
    """Expected flattened payload for `process_allgather(..., tiled=True)`."""

    return np.concatenate(
        [
            _rank_marked_payload(np, process_index=process, elems=elems)
            for process in range(process_count)
        ]
    )


def _chunk_first_values(
    np: Any,
    flat: Any,
    *,
    process_count: int,
    elems: int,
) -> list[int]:
    """Return first value of each process chunk in a tiled all-gather result."""

    arr = np.asarray(flat).reshape(-1)
    if elems <= 0 or arr.size < process_count * elems:
        return []
    return [int(arr[process * elems]) for process in range(process_count)]


def _process_fact_row(
    jnp: Any,
    *,
    process_index: int,
    process_count: int,
    local_device_count: int,
    global_device_count: int,
) -> Any:
    """Small metadata row gathered from every process."""

    return jnp.asarray(
        [process_index, process_count, local_device_count, global_device_count],
        dtype=jnp.int32,
    )


def _process_fact_checks(
    np: Any,
    *,
    fact_rows: Any,
    process_count: int,
    global_device_count: int,
) -> dict[str, bool]:
    """Validate process fact rows gathered with tiled=True."""

    rows = np.asarray(fact_rows)
    if rows.shape != (process_count, len(PROCESS_FACT_COLUMNS)):
        return {
            "process_fact_rows_shape": False,
            "process_indices_cover_range": False,
            "process_counts_agree": False,
            "global_device_counts_agree": False,
            "local_device_counts_are_positive": False,
        }
    return {
        "process_fact_rows_shape": True,
        "process_indices_cover_range": bool(
            np.array_equal(rows[:, 0], np.arange(process_count, dtype=rows.dtype))
        ),
        "process_counts_agree": bool(np.all(rows[:, 1] == process_count)),
        "global_device_counts_agree": bool(np.all(rows[:, 3] == global_device_count)),
        "local_device_counts_are_positive": bool(np.all(rows[:, 2] > 0)),
    }


def build_process_collective_smoke(
    *,
    jax: Any,
    args: Any,
    payload_bytes: int,
    n_devices: int,
) -> dict[str, Any]:
    """Run the tiny process-level sync/all-gather smoke test.

    This is intentionally not a throughput benchmark. It is a launch-validation
    spell: every process must enter the same named barrier and gather an array
    with the same shape. If the job hangs here, a larger collective would have
    hung with a more expensive bonfire.
    """

    import numpy as np
    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    spec = build_topology_smoke(
        jax=jax,
        args=args,
        payload_bytes=payload_bytes,
        n_devices=n_devices,
    )
    spec["op"] = "lab10_process_collective_smoke"

    process_index = int(spec["process_index"])
    process_count = int(spec["process_count"])
    local_device_count = int(spec["local_device_count"])
    global_device_count = int(spec["global_device_count"])
    elems = max(1, -(-int(payload_bytes) // 4))
    sync_name = _lab10_sync_name(process_count=process_count, elems=elems)

    try:
        payload = _process_payload(jnp, process_index=process_index, elems=elems)
        fact_row = _process_fact_row(
            jnp,
            process_index=process_index,
            process_count=process_count,
            local_device_count=local_device_count,
            global_device_count=global_device_count,
        )
        contract = {
            "process_count": process_count,
            "global_device_count": global_device_count,
            "payload_elems": elems,
            "payload_dtype": "int32",
            "sync_name": sync_name,
        }

        start = time.perf_counter()
        contract_assert_equal_reached: bool | None = None
        if hasattr(multihost_utils, "assert_equal"):
            # This preflight catches a mismatched process contract before the
            # gathered payload has to be interpreted. All processes must call it.
            multihost_utils.assert_equal(
                contract,
                fail_message="Lab 10 process collective contract mismatch",
            )
            contract_assert_equal_reached = True

        multihost_utils.sync_global_devices(sync_name + "_start")
        gathered = multihost_utils.process_allgather(payload, tiled=True)
        gathered_facts = multihost_utils.process_allgather(fact_row, tiled=True)
        multihost_utils.sync_global_devices(sync_name + "_done")
        elapsed = time.perf_counter() - start

        gathered_host = np.asarray(gathered).reshape(-1)
        expected = _expected_process_allgather(
            np,
            process_count=process_count,
            elems=elems,
        )
        fact_flat = np.asarray(gathered_facts).reshape(-1)
        if fact_flat.size == process_count * len(PROCESS_FACT_COLUMNS):
            fact_rows = fact_flat.reshape(process_count, len(PROCESS_FACT_COLUMNS))
        else:
            fact_rows = fact_flat.reshape(1, -1)

        checks: dict[str, bool | None] = {
            "contract_assert_equal_reached": contract_assert_equal_reached,
            "gather_shape": gathered_host.size == expected.size,
            "gather_values": bool(np.array_equal(gathered_host, expected)),
            "one_segment_per_process": len(
                _chunk_first_values(np, gathered_host, process_count=process_count, elems=elems)
            ) == process_count,
            "payload_shape_same_on_all_processes_by_construction": True,
            "sync_name_deterministic": bool(sync_name),
        }
        checks.update(
            _process_fact_checks(
                np,
                fact_rows=fact_rows,
                process_count=process_count,
                global_device_count=global_device_count,
            )
        )

        spec.update(
            {
                "ok": bool(spec.get("ok", False)) and _checks_pass(checks),
                "seconds": elapsed,
                "sync_names": [sync_name + "_start", sync_name + "_done"],
                "control_plane_bytes_per_process": elems * 4,
                "payload_elems_per_process": elems,
                "payload_dtype": "int32",
                "contract": contract,
                "process_allgather_tiled": True,
                "process_allgather_shape": list(np.asarray(gathered).shape),
                "process_allgather_flat_size": int(gathered_host.size),
                "expected_flat_size": int(expected.size),
                "process_allgather_preview": gathered_host[: min(16, gathered_host.size)].tolist(),
                "expected_preview": expected[: min(16, expected.size)].tolist(),
                "chunk_first_values": _chunk_first_values(
                    np,
                    gathered_host,
                    process_count=process_count,
                    elems=elems,
                ),
                "expected_chunk_first_values": [
                    int(process * PROCESS_PAYLOAD_STRIDE) for process in range(process_count)
                ],
                "process_fact_columns": list(PROCESS_FACT_COLUMNS),
                "process_fact_rows": fact_rows.astype(int).tolist(),
                "process_fact_rows_shape": list(fact_rows.shape),
                "process_collective_checks": checks,
                "process_collective_failed_checks": _failed_checks(checks),
                "process_collective_unknown_checks": _unknown_checks(checks),
                "failed_checks": _failed_checks(spec.get("checks", {})) + _failed_checks(checks),
                "teaching_note": (
                    "This validates process synchronization and host-visible data "
                    "exchange. It is not a TPU data-plane bandwidth measurement."
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - depends on distributed runtime.
        # Return a rich artifact so quick failures are useful. A true deadlock
        # still requires process-level launcher timeouts outside this function.
        checks = {"process_collective_completed": False}
        spec.update(
            {
                "ok": False,
                "seconds": None,
                "sync_names": [sync_name + "_start", sync_name + "_done"],
                "payload_elems_per_process": elems,
                "payload_dtype": "int32",
                "process_collective_error_type": type(exc).__name__,
                "process_collective_error": repr(exc),
                "process_collective_checks": checks,
                "process_collective_failed_checks": _failed_checks(checks),
                "process_collective_unknown_checks": _unknown_checks(checks),
                "failed_checks": _failed_checks(spec.get("checks", {})) + _failed_checks(checks),
            }
        )
    return spec


def build_spec(*, jax: Any, args: Any, payload_bytes: int, n_devices: int) -> dict[str, Any]:
    """Build the Lab 10 planning artifact without running process collectives."""

    spec = build_topology_smoke(
        jax=jax,
        args=args,
        payload_bytes=payload_bytes,
        n_devices=n_devices,
    )
    spec.update(
        {
            "op": "lab10_multihost_spec",
            "hierarchical_collective_examples": _hierarchical_collective_examples(),
            "launch_invariants": [
                "distributed initialization happens before device access",
                "all processes run the same program and arguments",
                "all processes enter process collectives in the same order",
                "process-local artifacts are written by every process",
                "global summaries are written by process 0",
                "hierarchical byte models separate process-local and cross-process traffic",
            ],
            "future_pallas_plan": [
                "construct a global mesh after Lab 10 topology is reliable",
                "choose process-local and cross-process mesh axes from device records",
                "reuse Lab 6 reduce-scatter and Lab 5 all-gather ownership models",
                "add phase-specific collective IDs and profile scopes",
                "validate first with built-in collectives before custom remote DMA",
            ],
            "capstone_report_checklist": _capstone_report_checklist(),
        }
    )
    return spec


# ---------------------------------------------------------------------------
# Renderers for lab_artifacts/*.json and lab_artifacts/*.md
# ---------------------------------------------------------------------------


def _format_check_value(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "pass" if value else "FAIL"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Render a tiny GitHub-flavored Markdown table."""

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def render_json(spec: dict[str, Any]) -> str:
    """Convenience renderer for standalone use and tests."""

    return json.dumps(spec, indent=2, sort_keys=True, default=str) + "\n"


def render_markdown(spec: dict[str, Any]) -> str:
    """Render a teaching-oriented Markdown artifact for one Lab 10 op."""

    lines = [lab_spec_utils.render_markdown(spec).rstrip(), ""]
    lines.extend(
        [
            "## Topology Smoke",
            "",
            f"- process index: {spec.get('process_index')}",
            f"- process count: {spec.get('process_count')}",
            f"- local devices: {spec.get('local_device_count')}",
            f"- global devices: {spec.get('global_device_count')}",
            f"- single process: {spec.get('single_process')}",
            f"- hostname: {(spec.get('runtime') or {}).get('hostname')}",
            "",
            "### Checks",
            "",
        ]
    )
    checks = spec.get("checks") or {}
    if checks:
        lines.extend(
            _markdown_table(
                ["check", "status"],
                [[name, _format_check_value(value)] for name, value in checks.items()],
            )
        )
    else:
        lines.append("No check table recorded.")

    failed = spec.get("failed_checks") or []
    unknown = spec.get("unknown_checks") or []
    if failed or unknown:
        lines.extend(["", "### Failed / Unknown Checks", ""])
        if failed:
            lines.extend([f"- failed: {name}" for name in failed])
        if unknown:
            lines.extend([f"- unknown: {name}" for name in unknown])

    summary = spec.get("topology_summary") or {}
    if summary:
        lines.extend(
            [
                "",
                "### Topology Summary",
                "",
                f"- process indices from devices: {summary.get('process_indices_from_devices')}",
                f"- devices per process: {summary.get('devices_per_process')}",
                f"- coordinate bounds: {summary.get('coordinate_bounds')}",
                f"- device kinds: {summary.get('device_kinds')}",
                f"- platforms: {summary.get('platforms')}",
            ]
        )

    group_summaries = spec.get("process_group_summaries") or {}
    if group_summaries:
        rows = []
        for process, group in sorted(group_summaries.items()):
            rows.append(
                [
                    process,
                    group.get("device_count"),
                    group.get("device_ids"),
                    group.get("device_kinds"),
                ]
            )
        lines.extend(["", "### Process Group Summaries", ""])
        lines.extend(_markdown_table(["process", "device count", "device ids", "device kinds"], rows))

    if spec.get("op") == "lab10_process_collective_smoke":
        lines.extend(
            [
                "",
                "## Process Collective Smoke",
                "",
                f"- sync names: {spec.get('sync_names')}",
                f"- payload elems/process: {spec.get('payload_elems_per_process')}",
                f"- elapsed seconds: {spec.get('seconds')}",
                f"- payload allgather shape: {spec.get('process_allgather_shape')}",
                f"- gathered preview: {spec.get('process_allgather_preview')}",
                f"- expected preview: {spec.get('expected_preview')}",
                "",
                "### Process Collective Checks",
                "",
            ]
        )
        process_checks = spec.get("process_collective_checks") or {}
        if process_checks:
            lines.extend(
                _markdown_table(
                    ["check", "status"],
                    [[name, _format_check_value(value)] for name, value in process_checks.items()],
                )
            )
        if spec.get("process_collective_error"):
            lines.extend(
                [
                    "",
                    "### Process Collective Error",
                    "",
                    f"- type: {spec.get('process_collective_error_type')}",
                    f"- error: `{spec.get('process_collective_error')}`",
                ]
            )

    hierarchy = spec.get("hierarchy_plan") or {}
    phases = hierarchy.get("phases") or []
    if phases:
        lines.extend(["", "## Hierarchy Plan", ""])
        lines.extend(
            _markdown_table(
                ["phase", "scope", "byte label", "purpose"],
                [
                    [
                        phase.get("name"),
                        phase.get("scope"),
                        phase.get("byte_label"),
                        phase.get("purpose"),
                    ]
                    for phase in phases
                ],
            )
        )

    byte_model = spec.get("hierarchy_byte_model") or hierarchy.get("byte_model") or {}
    if byte_model:
        lines.extend(["", "## Hierarchy Byte Model", ""])
        for key in [
            "kind",
            "payload_bytes_per_device",
            "host_block_bytes_this_process",
            "hierarchical_all_gather_cross_process_upper_model",
            "flat_ring_all_gather_send_bytes_per_device",
            "ring_all_reduce_send_bytes_per_device_model",
            "single_process_note",
            "teaching_note",
        ]:
            if key in byte_model:
                lines.append(f"- {key}: {byte_model.get(key)}")

    launch = spec.get("launch_plan") or {}
    if launch:
        lines.extend(
            [
                "",
                "## Launch Plan",
                "",
                f"- single-process command: `{launch.get('single_process_command')}`",
                f"- multi-process rule: {launch.get('multi_process_rule')}",
                f"- initialization rule: {launch.get('initialization_rule')}",
                f"- needs multi-process launch for true multi-host: {launch.get('needs_multi_process_launch_for_true_multihost')}",
                "",
                "All processes must match:",
                "",
            ]
        )
        lines.extend([f"- {item}" for item in launch.get("all_processes_must_match", [])])

    if spec.get("op") == "lab10_multihost_spec":
        lines.extend(["", "## Future Pallas Plan", ""])
        lines.extend([f"- {item}" for item in spec.get("future_pallas_plan", [])])
        lines.extend(["", "## Capstone Report Checklist", ""])
        lines.extend([f"- {item}" for item in spec.get("capstone_report_checklist", [])])

    return "\n".join(lines).rstrip() + "\n"
