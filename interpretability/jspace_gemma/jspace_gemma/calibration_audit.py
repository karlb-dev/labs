"""Strict audit of immutable, resumable OLMo calibration checkpoints."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .manifests import file_sha256, object_sha256


def _require_inside(path: Path, parent: Path, label: str) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"{label} escapes its checkpoint directory: {path}")
    return resolved


def _require_finite(value: object, path: str) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"non-finite tensor in {path}")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite float in {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _require_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite(child, f"{path}[{index}]")


def audit_completed_checkpoint(
    root: str | Path,
    *,
    expected_evidence_id: str,
    expected_compute_commit: str,
    expected_cells: int,
    expected_rows_per_cell: int,
    expected_parity_rows: int,
    inspect_raw_tensors: bool = True,
) -> dict:
    """Verify state, every durable hash, row metadata, and raw tensor schema."""
    checkpoint_root = Path(root).resolve()
    state_path = checkpoint_root / "state.json"
    input_path = checkpoint_root / "input_manifest.json"
    state = json.loads(state_path.read_text())
    payload = state.get("payload")
    if object_sha256(payload) != state.get("payload_sha256"):
        raise RuntimeError("checkpoint payload hash mismatch")
    header = state.get("header", {})
    if header.get("evidence_id") != expected_evidence_id:
        raise RuntimeError("checkpoint evidence ID mismatch")
    if header.get("code_commit") != expected_compute_commit:
        raise RuntimeError("checkpoint compute commit mismatch")

    input_manifest = json.loads(input_path.read_text())
    for key in ("evidence_id", "code_commit", "model_id", "model_revision"):
        if input_manifest.get(key) != header.get(key):
            raise RuntimeError(f"input/state {key} mismatch")
    if input_manifest.get("environment_sha256") != header.get("environment_sha256"):
        raise RuntimeError("input/state environment mismatch")
    if input_manifest.get("config", {}).get("sha256") != header.get("config_sha256"):
        raise RuntimeError("input/state config mismatch")

    completed = payload.get("completed_cells", {})
    if len(completed) != expected_cells:
        raise RuntimeError(
            f"expected {expected_cells} completed cells, found {len(completed)}"
        )
    parity = payload.get("parity", {})
    if len(parity) != expected_parity_rows or not all(
        row.get("ok") for row in parity.values()
    ):
        raise RuntimeError("clean suffix parity inventory is incomplete or failed")
    if not payload.get("wrong_hook"):
        raise RuntimeError("wrong-hook sentinel is absent")

    inventory = []
    all_rows = []
    runtimes = []
    autodiff_implementation_hashes = set()
    transport_implementation_hashes = set()
    for cell_id in sorted(completed):
        entry = completed[cell_id]
        resolved = {}
        for kind, subdirectory in (("metrics", "cells"), ("raw", "raw")):
            record = entry.get(kind, {})
            path = _require_inside(
                Path(record.get("path", "")), checkpoint_root / subdirectory, kind
            )
            if not path.is_file():
                raise RuntimeError(f"checkpoint file is absent: {cell_id} {kind}")
            digest = file_sha256(path)
            if digest != record.get("sha256"):
                raise RuntimeError(f"checkpoint hash drift: {cell_id} {kind}")
            resolved[kind] = path
            inventory.append(
                {
                    "cell_id": cell_id,
                    "kind": kind,
                    "path": str(path),
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                }
            )

        metrics = json.loads(resolved["metrics"].read_text())
        rows = metrics.get("rows", [])
        if metrics.get("schema_version") != 2 or len(rows) != expected_rows_per_cell:
            raise RuntimeError(f"metrics schema/row-count mismatch: {cell_id}")
        for row in rows:
            expected = {
                "cell_id": cell_id,
                "code_commit": expected_compute_commit,
                "config_sha256": header["config_sha256"],
                "environment_sha256": header["environment_sha256"],
                "model_id": header["model_id"],
                "model_revision": header["model_revision"],
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"row provenance mismatch: {cell_id}")
            autodiff_implementation_hashes.add(row.get("implementation_sha256"))
            transport_implementation_hashes.add(
                row.get("transport_implementation_sha256")
            )
            _require_finite(row, f"metrics.{cell_id}")
        all_rows.extend(rows)

        if inspect_raw_tensors:
            raw = torch.load(resolved["raw"], map_location="cpu", weights_only=False)
            if (
                raw.get("schema_version") != 2
                or raw.get("cell_id") != cell_id
                or len(raw.get("records", [])) != expected_rows_per_cell
            ):
                raise RuntimeError(f"raw schema/row-count mismatch: {cell_id}")
            raw_metadata = raw.get("metadata", {})
            if any(
                raw_metadata.get(key) != rows[0].get(key)
                for key in (
                    "code_commit",
                    "config_sha256",
                    "environment_sha256",
                    "model_id",
                    "model_revision",
                )
            ):
                raise RuntimeError(f"raw/metric provenance mismatch: {cell_id}")
            _require_finite(raw, f"raw.{cell_id}")

        runtime = float(entry.get("runtime_seconds", float("nan")))
        if not math.isfinite(runtime) or runtime <= 0:
            raise RuntimeError(f"invalid cell runtime: {cell_id}")
        runtimes.append(runtime)

    for label, values in (
        ("autodiff", autodiff_implementation_hashes),
        ("transport", transport_implementation_hashes),
    ):
        if len(values) != 1 or not next(iter(values), None):
            raise RuntimeError(f"{label} implementation hash is not uniquely bound")

    return {
        "state_path": str(state_path),
        "state_sha256": file_sha256(state_path),
        "state_payload_sha256": state["payload_sha256"],
        "input_manifest_path": str(input_path),
        "input_manifest_file_sha256": file_sha256(input_path),
        "input_manifest_object_sha256": object_sha256(input_manifest),
        "header": header,
        "completed_cells": len(completed),
        "rows": len(all_rows),
        "parity_rows": len(parity),
        "wrong_hook": payload["wrong_hook"],
        "minimum_runtime_seconds": min(runtimes),
        "maximum_runtime_seconds": max(runtimes),
        "inventory": inventory,
        "inventory_sha256": object_sha256(inventory),
        "autodiff_implementation_sha256": next(
            iter(autodiff_implementation_hashes)
        ),
        "transport_implementation_sha256": next(
            iter(transport_implementation_hashes)
        ),
        "all_rows": all_rows,
    }
