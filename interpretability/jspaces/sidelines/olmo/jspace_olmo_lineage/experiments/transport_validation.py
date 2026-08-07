"""Registered OLMo Study-2 H6 finite-dose transport validation.

The command has explicit scientific boundaries:

* ``freeze`` registers the otherwise-underspecified execution semantics before
  either mandatory H6 checkpoint is loaded;
* ``stage`` downloads an exact Hugging Face revision to local NVMe and hashes
  every file against the immutable Hub tree;
* ``run`` produces atomic, resumable per-cell raw vectors and scalar rows while
  comparing both exact JVP backends on the identical primal/tangent batches;
* ``register`` verifies and registers one mandatory checkpoint result.

The joint relevant-dose router is intentionally separate because it may use
only exact registered intervention-site records.  Per-item aggregates are not
silently promoted to site-level dose coverage.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from jspace_gemma import transport as shared_transport
from jspace_gemma.architecture import audit_loaded_model, manifest_from_config
from jspace_gemma.autodiff import exact_jvp
from jspace_gemma.hooks import ExplicitDecoderSuffix, TargetSpec
from jspace_gemma.staging import stage_snapshot, verify_snapshot

from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths import REPO_ROOT, manifests_dir, metrics_dir, run_root
from ..registry import append_event, create, read_events, resolve
from .checkpoint_inventory import tokenizer_semantics

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BRANCH = "interp_jspace_olmo_lineage_2"
CONFIG = PACKAGE_ROOT / "configs/ol2_transport_validation.yaml"
WEDGE_CONFIG = PACKAGE_ROOT / "configs/ol2_stage_wedge.yaml"
PROMPT_BANK = REPO_ROOT / "interpretability/jspaces/sidelines/gemma/data/g1_prompts_v1.jsonl"
ANCESTRY = "ol2-checkpoint-ancestry-v1"
FOUNDATION = "ol2-foundation-v1"
LICENSE_IMPORT = "ol2-gemma-backend-calibration-import-v1"
EXECUTION_FREEZE = "ol2-transport-execution-freeze-v1"
LOCAL_CACHE = Path("/content/hf_local")
EVALUATION_BATCH_SIZE = 8
EXPECTED_ARCHITECTURE = {
    "model_type": "olmo3",
    "num_hidden_layers": 64,
    "hidden_size": 5120,
}
MODEL_SLUGS = {"base": "base", "olmo31_think": "olmo31-think"}
PRIMARY_BACKEND = "torch.func.jvp"
INDEPENDENT_BACKEND = "torch.autograd.functional.jvp"


class TransportValidationError(RuntimeError):
    pass


def _bos_correction_path() -> Path:
    return manifests_dir() / "ol2_transport_predata_bos_correction.json"


def _shape_correction_path() -> Path:
    return manifests_dir() / (
        "ol2_transport_predata_prompt_tensor_shape_correction.json")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _config() -> dict:
    config = yaml.safe_load(CONFIG.read_text())
    if config.get("status") != "FROZEN_PRE_TRANSPORT_DATA":
        raise TransportValidationError("H6 transport config is not frozen")
    if config["models"]["mandatory_models"] != ["base", "olmo31_think"]:
        raise TransportValidationError("mandatory H6 model order drift")
    if config["layers_zero_indexed"] != [24, 32, 40, 56]:
        raise TransportValidationError("frozen H6 layer set drift")
    if config["assay_band_layers"] != [24, 32, 40]:
        raise TransportValidationError("frozen assay band drift")
    if config["relative_epsilon_ladder"] != [
        0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10
    ]:
        raise TransportValidationError("frozen H6 epsilon ladder drift")
    if config["transport_gate"]["exact_backends"] != [
        PRIMARY_BACKEND, INDEPENDENT_BACKEND
    ]:
        raise TransportValidationError("frozen exact backend pair drift")
    configured = Path(config["run_root"]).resolve()
    if configured != run_root().resolve():
        raise TransportValidationError(
            "JSPACE_OLMO_RUN_ROOT differs from the frozen H6 root")
    return config


def _event_output(event_id: str) -> dict:
    event = resolve(event_id)
    if not event["live"]:
        raise TransportValidationError(f"required event is not live: {event_id}")
    outputs = []
    for row in event.get("outputs", []):
        path = Path(str(row["path"]))
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise TransportValidationError(
                f"required event output hash drift: {event_id}: {path}")
        outputs.append({
            "path": str(path),
            "sha256": row["sha256"],
            "bytes": int(path.stat().st_size),
        })
    if not outputs:
        raise TransportValidationError(f"required event has no outputs: {event_id}")
    return {"event": event, "outputs": outputs}


def load_license() -> dict:
    imported = _event_output(LICENSE_IMPORT)
    event = imported["event"]
    if event.get("source_evidence_id") != (
            "gm2-backend-parity-calibration-v1"):
        raise TransportValidationError("H6 calibration source event drift")
    if event.get("route") != "benign_scheduling_floor":
        raise TransportValidationError("registered G2.1 route does not license H6")
    if event.get("licensed_model_scope_key") != "olmo3_32b_control":
        raise TransportValidationError("import is not OLMo architecture-specific")
    if event.get("pooled_ceiling_imported") is not False:
        raise TransportValidationError("pooled Gemma ceiling reached H6")
    for row in event.get("source_outputs", []):
        path = Path(str(row["path"]))
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise TransportValidationError(
                f"G2.1 imported source output drift: {path}")
    envelope = json.loads(Path(imported["outputs"][0]["path"]).read_text())
    licensed = envelope["licensed_import"]
    ceiling = float(event["licensed_backend_relative_error_ceiling"])
    if (
        float(licensed["backend_relative_error_ceiling"]) != ceiling
        or licensed["model_scope_key"] != "olmo3_32b_control"
        or licensed["pooled_ceiling_imported"] is not False
    ):
        raise TransportValidationError("registered import envelope license drift")
    return {
        "status": "available_and_applicable",
        "event_id": LICENSE_IMPORT,
        "event_output": imported["outputs"][0],
        "source_registry_sha256": event["source_registry_sha256"],
        "source_event_id": event["source_evidence_id"],
        "source_commit": event["source_commit"],
        "route": event["route"],
        "model_scope_key": event["licensed_model_scope_key"],
        "backend_relative_error_ceiling": ceiling,
        "pooled_ceiling_used": False,
    }


def historical_prompt_encoding_hashes() -> dict[str, str]:
    """Exact raw-tokenizer hashes from the imported G2.1 OLMo control rows."""
    event = resolve(LICENSE_IMPORT)
    candidates = [
        Path(str(row["path"])) for row in event.get("source_outputs", [])
        if Path(str(row["path"])).name == "backend_rows.parquet"
    ]
    if len(candidates) != 1:
        raise TransportValidationError(
            "imported G2.1 event lacks one backend row table")
    path = candidates[0]
    source_row = next(
        row for row in event["source_outputs"]
        if Path(str(row["path"])) == path)
    if not path.is_file() or file_sha256(path) != source_row["sha256"]:
        raise TransportValidationError("imported G2.1 backend row hash drift")
    frame = pd.read_parquet(
        path, columns=["model_key", "prompt_id", "token_ids_sha256"])
    frame = frame[frame.model_key == "olmo3_32b_control"]
    result = {}
    for prompt_id, group in frame.groupby("prompt_id", sort=True):
        values = sorted({str(value) for value in group.token_ids_sha256})
        if len(values) != 1:
            raise TransportValidationError(
                f"G2.1 has ambiguous OLMo prompt encoding: {prompt_id}")
        result[str(prompt_id)] = values[0]
    if set(result) != {"gm-p001", "gm-p002", "gm-p003", "gm-p004"}:
        raise TransportValidationError("G2.1 OLMo prompt encoding set drift")
    return result


def register_bos_correction() -> dict:
    """Register the pre-model raw-tokenizer/BOS interface correction."""
    git = require_clean_tree(expected_branch=BRANCH)
    config = _config()
    freeze = _event_output(EXECUTION_FREEZE)
    license_payload = load_license()
    origins = {
        row["evidence_id"] for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
    }
    target_events = {
        config["models"][key]["evidence_id"]
        for key in config["models"]["mandatory_models"]
    }
    if origins & target_events:
        raise TransportValidationError("BOS correction is no longer pre-model")
    for model_key in config["models"]["mandatory_models"]:
        paths = _paths(config, model_key)
        if paths["state"].exists() or any(paths["cells"].iterdir()) or any(
                paths["raw"].iterdir()):
            raise TransportValidationError(
                "H6 model outcome exists; BOS correction is forbidden")
    event = resolve(EXECUTION_FREEZE)
    if any(
        row["event"] == "evidence_corrected"
        and row.get("correction_kind") == "predata_bos_interface"
        for row in event["status_events"]
    ):
        raise TransportValidationError("pre-data BOS correction already registered")
    historical = historical_prompt_encoding_hashes()
    correction = _bos_correction_path()
    payload = {
        "schema_version": 1,
        "evidence_id": EXECUTION_FREEZE,
        "correction_kind": "predata_bos_interface",
        "status": "corrected_before_h6_model_or_transport_outcome",
        "old_execution_code_commit": event["code_commit"],
        "corrected_execution_code_commit": git["code_commit"],
        "reason": (
            "The standalone Base preflight stopped before causal-model load "
            "or any transport outcome "
            "because it incorrectly applied the stage-wedge scoring harness's "
            "manual-BOS expectation to the historical raw-tokenizer transport "
            "interface.  H6 reuses the G1/G2 transport map, whose registered "
            "OLMo-control encodings contain no automatic BOS."
        ),
        "corrected_contract": {
            "tokenization_call": (
                "tokenizer(text, add_special_tokens=True, truncation=False)"),
            "exact_hash_reference": (
                "imported G2.1 backend_rows.parquet OLMo-control token_ids_sha256"),
            "historical_prompt_encoding_hashes": historical,
            "leading_bos_count_must_be_at_most_one": True,
            "manual_bos_insertion": False,
            "chat_template_used": False,
        },
        "execution_freeze_output": freeze["outputs"][0],
        "calibration_import_output": license_payload["event_output"],
        "failed_preflight_log": str(
            run_root() / "logs/base_transport_preflight_producer.log"),
        "model_weights_loaded": False,
        "model_outcome_opened": False,
        "transport_cell_opened": False,
        "thresholds_changed": False,
        "predictions_changed": False,
    }
    payload["payload_sha256"] = object_sha256(payload)
    if correction.exists():
        raise FileExistsError(f"refusing to overwrite correction: {correction}")
    atomic_json(correction, payload)
    append_event({
        "event": "evidence_corrected",
        "evidence_id": EXECUTION_FREEZE,
        "reason": payload["reason"],
        "correction_kind": "predata_bos_interface",
        "corrected_fields": {
            "bos_and_prompt_encoding_gate": payload["corrected_contract"],
            "execution_code_commit": git["code_commit"],
            "model_outcome_opened": False,
        },
        "correction_artifact": {
            "path": str(correction),
            "sha256": file_sha256(correction),
            "bytes": int(correction.stat().st_size),
        },
        "model_weights_loaded": False,
        "transport_cell_opened": False,
        "thresholds_changed": False,
        "predictions_changed": False,
    })
    return {
        "artifact": str(correction),
        "sha256": file_sha256(correction),
        "historical_prompt_encoding_hashes": historical,
    }


def verified_bos_correction() -> dict:
    event = resolve(EXECUTION_FREEZE)
    rows = [
        row for row in event["status_events"]
        if row["event"] == "evidence_corrected"
        and row.get("correction_kind") == "predata_bos_interface"
    ]
    if len(rows) != 1:
        raise TransportValidationError(
            "expected one registered pre-data BOS interface correction")
    row = rows[0]
    artifact = row["correction_artifact"]
    path = Path(artifact["path"])
    if path != _bos_correction_path():
        raise TransportValidationError("BOS correction artifact path drift")
    if not path.is_file() or file_sha256(path) != artifact["sha256"]:
        raise TransportValidationError("BOS correction artifact hash drift")
    return {
        "path": str(path),
        "sha256": artifact["sha256"],
        "bytes": int(path.stat().st_size),
        "correction_event_utc": row["event_utc"],
    }


def register_shape_correction() -> dict:
    """Register a second pre-model correction for `[1,T]` token tensors."""
    git = require_clean_tree(expected_branch=BRANCH)
    config = _config()
    origins = {
        row["evidence_id"] for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
    }
    target_events = {
        config["models"][key]["evidence_id"]
        for key in config["models"]["mandatory_models"]
    }
    if origins & target_events:
        raise TransportValidationError("shape correction is no longer pre-model")
    for model_key in config["models"]["mandatory_models"]:
        paths = _paths(config, model_key)
        if paths["state"].exists() or any(paths["cells"].iterdir()) or any(
                paths["raw"].iterdir()):
            raise TransportValidationError(
                "H6 model outcome exists; shape correction is forbidden")
    event = resolve(EXECUTION_FREEZE)
    if any(
        row["event"] == "evidence_corrected"
        and row.get("correction_kind") == "predata_prompt_tensor_shape"
        for row in event["status_events"]
    ):
        raise TransportValidationError("prompt tensor correction already registered")
    correction = _shape_correction_path()
    payload = {
        "schema_version": 1,
        "evidence_id": EXECUTION_FREEZE,
        "correction_kind": "predata_prompt_tensor_shape",
        "status": "corrected_before_h6_model_or_transport_outcome",
        "old_execution_code_commit": event["effective_metadata"].get(
            "execution_code_commit", event["code_commit"]),
        "corrected_execution_code_commit": git["code_commit"],
        "reason": (
            "The corrected standalone preflight represented token IDs as a "
            "rank-2 [1,T] tensor for the historical numpy-byte hash, but the "
            "human-readable ID extraction iterated its batch row as a scalar. "
            "The preflight stopped with ValueError before causal-model load or "
            "any transport cell; flattening the sole batch row changes no "
            "encoding, threshold, prediction, or outcome."),
        "corrected_contract": {
            "encoded_input_shape": "[1,T]",
            "human_readable_token_ids": "encoded_input_ids[0].tolist()",
            "historical_numpy_byte_hash_input": (
                "unchanged contiguous [1,T] int64 tensor"),
        },
        "failed_preflight_log": str(
            run_root() / "logs/base_transport_preflight_producer.log"),
        "model_weights_loaded": False,
        "model_outcome_opened": False,
        "transport_cell_opened": False,
        "thresholds_changed": False,
        "predictions_changed": False,
    }
    payload["payload_sha256"] = object_sha256(payload)
    if correction.exists():
        raise FileExistsError(f"refusing to overwrite correction: {correction}")
    atomic_json(correction, payload)
    append_event({
        "event": "evidence_corrected",
        "evidence_id": EXECUTION_FREEZE,
        "reason": payload["reason"],
        "correction_kind": "predata_prompt_tensor_shape",
        "corrected_fields": {
            "prompt_tensor_shape_gate": payload["corrected_contract"],
            "execution_code_commit": git["code_commit"],
            "model_outcome_opened": False,
        },
        "correction_artifact": {
            "path": str(correction),
            "sha256": file_sha256(correction),
            "bytes": int(correction.stat().st_size),
        },
        "model_weights_loaded": False,
        "transport_cell_opened": False,
        "thresholds_changed": False,
        "predictions_changed": False,
    })
    return {"artifact": str(correction), "sha256": file_sha256(correction)}


def verified_shape_correction() -> dict:
    event = resolve(EXECUTION_FREEZE)
    rows = [
        row for row in event["status_events"]
        if row["event"] == "evidence_corrected"
        and row.get("correction_kind") == "predata_prompt_tensor_shape"
    ]
    if len(rows) != 1:
        raise TransportValidationError(
            "expected one registered pre-data prompt tensor correction")
    artifact = rows[0]["correction_artifact"]
    path = Path(artifact["path"])
    if path != _shape_correction_path():
        raise TransportValidationError("prompt tensor correction path drift")
    if not path.is_file() or file_sha256(path) != artifact["sha256"]:
        raise TransportValidationError("prompt tensor correction hash drift")
    return {
        "path": str(path),
        "sha256": artifact["sha256"],
        "bytes": int(path.stat().st_size),
        "correction_event_utc": rows[0]["event_utc"],
    }


def _model_spec(config: Mapping, model_key: str) -> dict:
    if model_key not in {"base", "olmo31_think"}:
        raise ValueError(f"unknown mandatory H6 model {model_key!r}")
    return dict(config["models"][model_key])


def _paths(config: Mapping, model_key: str) -> dict[str, Path]:
    spec = _model_spec(config, model_key)
    root = metrics_dir("transport-validation") / MODEL_SLUGS[model_key] / spec[
        "evidence_id"]
    cells = root / "cells"
    raw = root / "raw"
    for path in (root, cells, raw, run_root() / "logs", run_root() / "checkpoints"):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "cells": cells,
        "raw": raw,
        "state": root / "state.json",
        "heartbeat": run_root() / "checkpoints" / f"{spec['evidence_id']}_heartbeat.json",
        "snapshot_manifest": manifests_dir() / f"ol2_transport_snapshot_{model_key}.json",
        "preflight": root / "snapshot_conformance.json",
        "input_manifest": root / "input_manifest.json",
        "rows": root / "transport_rows.parquet",
        "summary": root / "transport_result.json",
        "raw_inventory": root / "raw_inventory.json",
    }


def _write_heartbeat(path: Path, **fields: object) -> None:
    atomic_json(path, {
        "schema_version": 1,
        "updated_utc": _utc(),
        "pid": os.getpid(),
        **fields,
    })


@contextmanager
def _lock(name: str):
    path = Path("/content/olmo_lineage_work/locks") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise TransportValidationError(f"another H6 process owns {path}") from error
    handle.write(str(os.getpid()))
    handle.flush()
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def freeze_execution() -> dict:
    git = require_clean_tree(expected_branch=BRANCH)
    config = _config()
    license_payload = load_license()
    origins = {
        row["evidence_id"] for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
    }
    if EXECUTION_FREEZE in origins:
        raise TransportValidationError("H6 execution freeze is already registered")
    target_events = {
        config["models"][key]["evidence_id"]
        for key in config["models"]["mandatory_models"]
    }
    if origins & target_events:
        raise TransportValidationError("H6 target data predates execution freeze")
    foundation = _event_output(FOUNDATION)
    payload = {
        "schema_version": 1,
        "evidence_id": EXECUTION_FREEZE,
        "tier": "methods",
        "status": "frozen_before_mandatory_h6_model_load",
        "code_commit": git["code_commit"],
        "transport_config": {
            "path": str(CONFIG),
            "sha256": file_sha256(CONFIG),
        },
        "foundation_output": foundation["outputs"][0],
        "calibration_license": license_payload,
        "mandatory_model_order": list(config["models"]["mandatory_models"]),
        "prompt_order": list(config["prompt_ids"]),
        "layer_order": list(config["layers_zero_indexed"]),
        "direction_order": [
            {"family": family, "id": f"{family}-0"}
            for family in config["directions"]["families"]
        ],
        "relative_epsilon_ladder": list(config["relative_epsilon_ladder"]),
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "row_gate": {
            "delivery": (
                "positive, negative, and double delivery all pass frozen "
                "cosine and relative-norm gates"),
            "backend": (
                "identical-batch exact-backend tangent relative error is at "
                "or below the imported OLMo-specific ceiling"),
            "snr": "response_snr at or above the frozen decision floor",
            "forward": "tangent cosine >= floor and relative error <= ceiling",
            "central": (
                "central tangent cosine >= the same frozen cosine floor and "
                "central relative error <= its frozen ceiling"),
            "all_required": True,
        },
        "layer_epsilon_gate": {
            "denominator": "all 4 prompts x all 3 direction families",
            "unmeasurable_rows_count_as_nonpassing": True,
            "passage_floor": config["transport_gate"]["row_passage_floor"],
        },
        "common_regime": {
            "definition": (
                "an epsilon passes in-band only when every assay layer passes "
                "at that same frozen epsilon"),
            "epsilon_0_10_continuity_only": True,
            "small_regime_values": [
                value for value in config["relative_epsilon_ladder"]
                if value < 0.10
            ],
        },
        "dose_mapping": {
            "reuse_registered_position_records_only": True,
            "per_item_mean_or_max_is_not_a_site_distribution": True,
            "missing_exact_site_records": (
                "coverage is missing, not zero; relevant-dose pass and "
                "scale-limited classification remain unresolved"),
            "effective_epsilon": config["dose_matching"][
                "effective_relative_epsilon"],
            "coverage_floor": config["dose_matching"][
                "coverage_floor_for_in_band_pass"],
        },
        "fit": {
            "form": "intercept_plus_slope_times_relative_epsilon",
            "response": "forward_tangent_relative_error",
            "rows": "measurement-SNR rows through epsilon 0.10",
            "minimum_unique_epsilon": 3,
            "estimator": "ordinary_least_squares_descriptive",
        },
        "router_order": list(config["router"]["order"]),
        "no_model_outcome_opened": True,
        "predictions_and_thresholds_changed": False,
    }
    payload["payload_sha256"] = object_sha256(payload)
    output = manifests_dir() / "ol2_transport_execution_freeze_v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite H6 freeze: {output}")
    atomic_json(output, payload)
    create(
        EXECUTION_FREEZE,
        tier="methods",
        what=(
            "Pre-data H6 execution semantics binding identical-batch dual "
            "exact-JVP comparison, row/layer gates, and no aggregate-to-site "
            "dose substitution."
        ),
        command=(
            "python -m jspace_olmo_lineage.experiments.transport_validation "
            "--phase freeze"
        ),
        outputs=[output],
        inputs={
            "transport_config_sha256": file_sha256(CONFIG),
            "foundation_output_sha256": foundation["outputs"][0]["sha256"],
            "calibration_import_output_sha256": license_payload[
                "event_output"]["sha256"],
        },
        model_outcome_opened=False,
        predictions_and_thresholds_changed=False,
    )
    return {"output": str(output), "sha256": file_sha256(output)}


def stage(model_key: str) -> dict:
    clean = require_clean_tree(expected_branch=BRANCH)
    config = _config()
    _event_output(EXECUTION_FREEZE)
    paths = _paths(config, model_key)
    output = paths["snapshot_manifest"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite snapshot manifest: {output}")
    spec = _model_spec(config, model_key)
    result = stage_snapshot(
        repo_id=spec["model_id"],
        revision=spec["revision"],
        cache_root=LOCAL_CACHE,
        seed_model_root=None,
        output_manifest=output,
    )
    result.pop("snapshot_manifest_sha256", None)
    result.update({
        "model_key": model_key,
        "evidence_id": spec["evidence_id"],
        "transport_config_sha256": file_sha256(CONFIG),
        "staging_code_commit": clean["code_commit"],
        "direct_huggingface_download": True,
        "drive_model_copy_used": False,
        "model_loaded": False,
        "model_outcome_opened": False,
    })
    result["snapshot_manifest_sha256"] = object_sha256(result)
    atomic_json(output, result)
    return {
        "model_key": model_key,
        "snapshot": result["snapshot"],
        "manifest": str(output),
        "manifest_sha256": file_sha256(output),
        "remote_inventory_sha256": result["remote_inventory_sha256"],
        "weight_shards": len(result["weight_shards"]),
    }


def _remote_from_staging_manifest(manifest: Mapping) -> dict:
    files = [{
        "path": row["path"],
        "size_bytes": row["expected_size_bytes"],
        "lfs_sha256": row["expected_lfs_sha256"],
        "git_blob_id": row["expected_git_blob_id"],
    } for row in manifest["files"]]
    return {
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "files": files,
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "inventory_sha256": manifest["remote_inventory_sha256"],
    }


def _verified_snapshot(config: Mapping, model_key: str) -> tuple[Path, dict]:
    paths = _paths(config, model_key)
    spec = _model_spec(config, model_key)
    if not paths["snapshot_manifest"].is_file():
        raise TransportValidationError(f"stage exact {model_key} snapshot first")
    manifest = json.loads(paths["snapshot_manifest"].read_text())
    required = {
        "model_key": model_key,
        "repo_id": spec["model_id"],
        "revision": spec["revision"],
        "transport_config_sha256": file_sha256(CONFIG),
        "all_content_hashes_verified": True,
    }
    mismatch = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in required.items() if manifest.get(key) != value
    }
    if mismatch:
        raise TransportValidationError(f"snapshot manifest mismatch: {mismatch}")
    verification = verify_snapshot(
        manifest["snapshot"],
        repo_id=spec["model_id"],
        revision=spec["revision"],
        remote_inventory=_remote_from_staging_manifest(manifest),
    )
    if not verification["all_content_hashes_verified"]:
        raise TransportValidationError("mandatory snapshot rehash failed")
    if verification["remote_inventory_sha256"] != manifest[
            "remote_inventory_sha256"]:
        raise TransportValidationError("snapshot remote inventory drift")
    return Path(manifest["snapshot"]), manifest


def _prompts(config: Mapping) -> list[dict]:
    rows = [
        json.loads(line) for line in PROMPT_BANK.read_text().splitlines()
        if line.strip()
    ]
    by_id = {row["prompt_id"]: row for row in rows}
    if any(prompt_id not in by_id for prompt_id in config["prompt_ids"]):
        raise TransportValidationError("frozen H6 prompt is absent")
    return [by_id[prompt_id] for prompt_id in config["prompt_ids"]]


def _audit_texts() -> tuple[list[str], dict]:
    ancestry = json.loads(
        (run_root(create=False) / "manifests/ol2_checkpoint_ancestry_v1.json")
        .read_text()
    )
    audit = ancestry["tokenizer_semantic_audit"]
    path = Path(audit["path"])
    if file_sha256(path) != audit["file_sha256"]:
        raise TransportValidationError("tokenizer audit corpus hash drift")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    texts = [str(row["text"]) for row in rows] + [
        str(value) for value in audit["edge_cases"]
    ]
    if len(texts) != int(audit["audit_texts"]):
        raise TransportValidationError("tokenizer audit corpus size drift")
    return texts, audit


def preflight(config: Mapping, model_key: str, snapshot: Path,
              staging_manifest: Mapping) -> dict:
    paths = _paths(config, model_key)
    spec = _model_spec(config, model_key)
    raw_config = json.loads((snapshot / "config.json").read_text())
    architecture_checks = {
        key: raw_config.get(key) == expected
        for key, expected in EXPECTED_ARCHITECTURE.items()
    }
    architecture = manifest_from_config(snapshot / "config.json")
    architecture_checks["manifest_family"] = architecture["family"] == "olmo3"
    architecture_checks["requested_layers_exist"] = max(
        config["layers_zero_indexed"]) < architecture["decoder"]["num_layers"]
    if not all(architecture_checks.values()):
        raise TransportValidationError(
            f"H6 architecture hard gate failed: {architecture_checks}")

    audit_texts, audit_source = _audit_texts()
    semantics = tokenizer_semantics(
        snapshot / "tokenizer.json", snapshot / "tokenizer_config.json",
        audit_texts,
    )
    wedge = yaml.safe_load(WEDGE_CONFIG.read_text())
    expected_tokenizer = wedge["tokenizer_contract"]
    semantic_fields = (
        "semantic_fingerprint_sha256", "token_id_map_sha256",
        "normalized_model_sha256", "processing_components_sha256",
    )
    tokenizer_checks = {
        field: semantics[field] == expected_tokenizer[field]
        for field in semantic_fields
    }
    tokenizer_checks["audit_encoding_sha256"] = (
        semantics["audit_encoding_sha256"]
        == expected_tokenizer["frozen_audit_encoding_sha256"]
    )
    tokenizer_checks["audit_text_count"] = semantics["audit_text_count"] == int(
        audit_source["audit_texts"])
    if not all(tokenizer_checks.values()):
        raise TransportValidationError(
            f"H6 semantic tokenizer hard gate failed: {tokenizer_checks}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encodings = []
    bos_checks = {}
    historical_hashes = historical_prompt_encoding_hashes()
    correction = verified_bos_correction()
    shape_correction = verified_shape_correction()
    for prompt in _prompts(config):
        encoded = tokenizer(
            prompt["text"], return_tensors="pt", add_special_tokens=True,
            truncation=False)
        ids = [int(value) for value in encoded["input_ids"][0].tolist()]
        bos = tokenizer.bos_token_id
        leading_bos_count = 0
        if bos is not None:
            for value in ids:
                if value != int(bos):
                    break
                leading_bos_count += 1
        numpy_hash = hashlib.sha256(
            encoded["input_ids"].cpu().numpy().tobytes()).hexdigest()
        encoding_match = numpy_hash == historical_hashes[prompt["prompt_id"]]
        bos_checks[prompt["prompt_id"]] = {
            "bos_token_id": None if bos is None else int(bos),
            "leading_bos_count": leading_bos_count,
            "leading_bos_count_at_most_one": leading_bos_count <= 1,
            "historical_transport_encoding_match": encoding_match,
        }
        encodings.append({
            "prompt_id": prompt["prompt_id"],
            "prompt_sha256": hashlib.sha256(prompt["text"].encode()).hexdigest(),
            "token_ids": ids,
            "token_ids_sha256": object_sha256(ids),
            "token_ids_numpy_sha256": numpy_hash,
            "historical_transport_token_ids_numpy_sha256": historical_hashes[
                prompt["prompt_id"]],
        })
    if not all(
        row["leading_bos_count_at_most_one"]
        and row["historical_transport_encoding_match"]
        for row in bos_checks.values()
    ):
        raise TransportValidationError(f"H6 BOS hard gate failed: {bos_checks}")
    prompt_encoding_sha256 = object_sha256(encodings)
    base_preflight = _paths(config, "base")["preflight"]
    cross_checkpoint = {"required": model_key == "olmo31_think", "passed": True}
    if model_key == "olmo31_think":
        if not base_preflight.is_file():
            raise TransportValidationError(
                "Base prompt-encoding preflight must precede 3.1 Think")
        base = json.loads(base_preflight.read_text())
        cross_checkpoint.update({
            "base_prompt_encoding_sha256": base["prompt_encoding_sha256"],
            "current_prompt_encoding_sha256": prompt_encoding_sha256,
            "passed": base["prompt_encoding_sha256"] == prompt_encoding_sha256,
        })
        if not cross_checkpoint["passed"]:
            raise TransportValidationError(
                "Base and 3.1 Think frozen prompt encodings differ")
    payload = {
        "schema_version": 1,
        "model_key": model_key,
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "snapshot": str(snapshot),
        "snapshot_manifest": {
            "path": str(paths["snapshot_manifest"]),
            "sha256": file_sha256(paths["snapshot_manifest"]),
            "remote_inventory_sha256": staging_manifest[
                "remote_inventory_sha256"],
        },
        "architecture_checks": architecture_checks,
        "architecture_manifest": architecture,
        "tokenizer_semantics": semantics,
        "tokenizer_checks": tokenizer_checks,
        "bos_checks": bos_checks,
        "prompt_encodings": encodings,
        "prompt_encoding_sha256": prompt_encoding_sha256,
        "cross_checkpoint_prompt_encoding": cross_checkpoint,
        "predata_bos_interface_correction": {
            "path": correction["path"],
            "sha256": correction["sha256"],
        },
        "predata_prompt_tensor_shape_correction": {
            "path": shape_correction["path"],
            "sha256": shape_correction["sha256"],
        },
        "all_hard_gates_passed": True,
        "model_outcome_opened": False,
    }
    if paths["preflight"].exists():
        if json.loads(paths["preflight"].read_text()) != payload:
            raise TransportValidationError("H6 preflight changed across replay")
    else:
        atomic_json(paths["preflight"], payload)
    return payload


def _input_manifest(config: Mapping, model_key: str, preflight_payload: Mapping,
                    clean: Mapping) -> dict:
    paths = _paths(config, model_key)
    spec = _model_spec(config, model_key)
    freeze = _event_output(EXECUTION_FREEZE)
    license_payload = load_license()
    ancestry = _event_output(ANCESTRY)
    foundation = _event_output(FOUNDATION)
    correction = verified_bos_correction()
    shape_correction = verified_shape_correction()
    payload = {
        "schema_version": 1,
        "evidence_id": spec["evidence_id"],
        "model_key": model_key,
        "model_id": spec["model_id"],
        "model_revision": spec["revision"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(CONFIG),
        "prompt_bank": {
            "path": str(PROMPT_BANK), "sha256": file_sha256(PROMPT_BANK)},
        "prompt_encoding_sha256": preflight_payload[
            "prompt_encoding_sha256"],
        "snapshot_manifest_sha256": preflight_payload[
            "snapshot_manifest"]["sha256"],
        "tokenizer_semantic_fingerprint_sha256": preflight_payload[
            "tokenizer_semantics"]["semantic_fingerprint_sha256"],
        "execution_freeze_output": freeze["outputs"][0],
        "predata_bos_interface_correction": {
            "path": correction["path"],
            "sha256": correction["sha256"],
        },
        "predata_prompt_tensor_shape_correction": {
            "path": shape_correction["path"],
            "sha256": shape_correction["sha256"],
        },
        "calibration_import_output": license_payload["event_output"],
        "ancestry_output": ancestry["outputs"][0],
        "foundation_output": foundation["outputs"][0],
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "input_manifest_sha256": None,
    }
    payload["input_manifest_sha256"] = object_sha256({
        key: value for key, value in payload.items()
        if key != "input_manifest_sha256"
    })
    if paths["input_manifest"].exists():
        if json.loads(paths["input_manifest"].read_text()) != payload:
            raise TransportValidationError("H6 input manifest changed across replay")
    else:
        atomic_json(paths["input_manifest"], payload)
    return payload


def _as_batch(value: torch.Tensor, count: int) -> torch.Tensor:
    if count == 1 and value.ndim == 1:
        return value.unsqueeze(0)
    if value.ndim < 2 or value.shape[0] != count:
        raise TransportValidationError(
            f"exact backend output lost batch axis: {tuple(value.shape)}")
    return value


def _tensor_comparison(value: torch.Tensor, reference: torch.Tensor) -> dict:
    value = value.detach().double().reshape(-1)
    reference = reference.detach().double().reshape(-1)
    reference_norm = float(reference.norm())
    value_norm = float(value.norm())
    difference = value - reference
    denominator = max(reference_norm, 1e-30)
    cosine = None
    if reference_norm > 0 and value_norm > 0:
        cosine = float(F.cosine_similarity(value, reference, dim=0))
    return {
        "relative_error": float(difference.norm()) / denominator,
        "cosine": cosine,
        "max_absolute_difference": float(difference.abs().max()),
        "reference_norm": reference_norm,
        "value_norm": value_norm,
    }


def evaluate_dual_backend_transport_cell(*args, **kwargs) -> tuple[list[dict], dict]:
    """Run the shared finite-dose evaluator with both exact backends.

    The shared evaluator owns the finite-response, realized-delivery, SNR,
    homogeneity, odd-symmetry, and additivity contracts.  This wrapper replaces
    only its exact-JVP call for the duration of one single-threaded cell, calls
    both frozen backends on the identical batches, and returns the primary JVP
    to the shared scalar estimator.
    """
    original = shared_transport.exact_jvp
    backend_calls: list[dict] = []

    def dual(function, primal, tangent, *, backend="auto"):
        if backend != "auto":
            raise TransportValidationError(
                f"shared evaluator requested unexpected backend {backend!r}")
        primary = exact_jvp(
            function, primal, tangent, backend=PRIMARY_BACKEND)
        independent = exact_jvp(
            function, primal, tangent, backend=INDEPENDENT_BACKEND)
        count = int(primal.shape[0])
        primary_primal = _as_batch(primary.primal, count).detach().float().cpu()
        primary_tangent = _as_batch(primary.tangent, count).detach().float().cpu()
        independent_primal = _as_batch(
            independent.primal, count).detach().float().cpu()
        independent_tangent = _as_batch(
            independent.tangent, count).detach().float().cpu()
        backend_calls.append({
            "primary_primal": primary_primal,
            "primary_tangent": primary_tangent,
            "independent_primal": independent_primal,
            "independent_tangent": independent_tangent,
        })
        return primary

    if original is not exact_jvp:
        raise TransportValidationError(
            "shared transport exact-JVP binding changed before H6")
    shared_transport.exact_jvp = dual
    try:
        rows, raw = shared_transport.evaluate_transport_cell(*args, **kwargs)
    finally:
        shared_transport.exact_jvp = original
    diagnostics = raw["exact_batch_diagnostics"]
    if len(diagnostics) != len(backend_calls):
        raise TransportValidationError("dual-backend call/diagnostic count drift")
    by_key = {}
    raw_backend = []
    for call, diagnostic in zip(backend_calls, diagnostics):
        keys = diagnostic["request_keys"]
        if len(keys) != call["primary_tangent"].shape[0]:
            raise TransportValidationError("dual-backend request slot drift")
        for slot, raw_key in enumerate(keys):
            key = tuple(raw_key)
            tangent = _tensor_comparison(
                call["independent_tangent"][slot],
                call["primary_tangent"][slot],
            )
            primal = _tensor_comparison(
                call["independent_primal"][slot],
                call["primary_primal"][slot],
            )
            record = {
                "request_key": list(key),
                "backend_primary": PRIMARY_BACKEND,
                "backend_independent": INDEPENDENT_BACKEND,
                "tangent": tangent,
                "primal": primal,
                "primary_tangent_sha256": shared_transport.tensor_sha256(
                    call["primary_tangent"][slot]),
                "independent_tangent_sha256": shared_transport.tensor_sha256(
                    call["independent_tangent"][slot]),
                "independent_tangent": call["independent_tangent"][slot],
                "independent_primal": call["independent_primal"][slot],
            }
            by_key[key] = record
            raw_backend.append(record)
    for row in rows:
        key = (
            row["direction_id"], float(row["desired_relative_epsilon"]),
            "positive",
        )
        audit = by_key[key]
        row["exact_jvp_primal_to_finite_baseline_relative_error"] = row.pop(
            "backend_parity_relative_error")
        row.update({
            "backend_primary": PRIMARY_BACKEND,
            "backend_independent": INDEPENDENT_BACKEND,
            "backend_tangent_relative_error": audit["tangent"]["relative_error"],
            "backend_tangent_cosine": audit["tangent"]["cosine"],
            "backend_tangent_max_absolute_difference": audit["tangent"][
                "max_absolute_difference"],
            "backend_primal_relative_error": audit["primal"]["relative_error"],
            "backend_primal_cosine": audit["primal"]["cosine"],
            "backend_primary_tangent_sha256": audit[
                "primary_tangent_sha256"],
            "backend_independent_tangent_sha256": audit[
                "independent_tangent_sha256"],
        })
    raw["dual_backend_records"] = raw_backend
    raw["exact_backends"] = [PRIMARY_BACKEND, INDEPENDENT_BACKEND]
    raw["max_backend_tangent_relative_error"] = max(
        row["tangent"]["relative_error"] for row in raw_backend)
    raw["max_backend_primal_relative_error"] = max(
        row["primal"]["relative_error"] for row in raw_backend)
    raw["shared_field_named_max_backend_parity_relative_error_semantics"] = (
        "primary exact-JVP primal versus identical-batch finite baseline; "
        "retained under this clarified label only")
    return rows, raw


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_model(snapshot: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    ).eval()
    assert_model_on_cuda(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def _state_header(config: Mapping, model_key: str, manifest: Mapping,
                  preflight_payload: Mapping) -> dict:
    spec = _model_spec(config, model_key)
    return {
        "schema_version": 1,
        "evidence_id": spec["evidence_id"],
        "model_key": model_key,
        "model_revision": spec["revision"],
        "config_sha256": file_sha256(CONFIG),
        "input_manifest_sha256": manifest["input_manifest_sha256"],
        "snapshot_manifest_sha256": preflight_payload[
            "snapshot_manifest"]["sha256"],
        "producer_code_commit": manifest["code_commit"],
    }


def _load_state(path: Path, header: Mapping) -> dict:
    if not path.exists():
        return {"completed_cells": {}, "model_audit": None, "gpu": None}
    envelope = json.loads(path.read_text())
    if envelope.get("header") != dict(header):
        raise TransportValidationError("H6 recovery state identity mismatch")
    if object_sha256(envelope["payload"]) != envelope["payload_sha256"]:
        raise TransportValidationError("H6 recovery state hash mismatch")
    return envelope["payload"]


def _write_state(path: Path, header: Mapping, payload: Mapping) -> None:
    atomic_json(path, {
        "schema_version": 1,
        "header": dict(header),
        "payload": dict(payload),
        "payload_sha256": object_sha256(dict(payload)),
    })


def _cell_id(prompt_id: str, layer: int) -> str:
    return f"{prompt_id}-L{int(layer):02d}-single_position"


def _direction_specs(config: Mapping) -> list[dict]:
    return [
        {"type": family, "id": f"{family}-0"}
        for family in config["directions"]["families"]
    ]


def run_model(model_key: str, max_cells: int | None = None) -> dict:
    with _lock("ol2_transport_validation.lock"):
        clean = require_clean_tree(expected_branch=BRANCH)
        gpu = require_cuda_gpu()
        config = _config()
        freeze = _event_output(EXECUTION_FREEZE)
        license_payload = load_license()
        snapshot, staging_manifest = _verified_snapshot(config, model_key)
        paths = _paths(config, model_key)
        preflight_payload = preflight(
            config, model_key, snapshot, staging_manifest)
        manifest = _input_manifest(
            config, model_key, preflight_payload, clean)
        header = _state_header(
            config, model_key, manifest, preflight_payload)
        state = _load_state(paths["state"], header)
        state["gpu"] = gpu
        _write_heartbeat(
            paths["heartbeat"], phase="loading_model", model_key=model_key,
            completed_cells=len(state["completed_cells"]))
        tokenizer, model = _load_model(snapshot)
        audit = audit_loaded_model(model)
        if (
            audit["num_layers"] != EXPECTED_ARCHITECTURE["num_hidden_layers"]
            or audit["residual_width"] != EXPECTED_ARCHITECTURE["hidden_size"]
        ):
            raise TransportValidationError("loaded H6 model architecture drift")
        state["model_audit"] = audit
        _write_state(paths["state"], header, state)

        prompts = _prompts(config)
        spec = _model_spec(config, model_key)
        target = TargetSpec("final_residual", position_indices=(-1,))
        new_cells = 0
        for prompt in prompts:
            encoded = tokenizer(
                prompt["text"], return_tensors="pt", add_special_tokens=True,
                truncation=False)
            input_ids = encoded["input_ids"].to("cuda")
            attention_mask = encoded.get(
                "attention_mask", torch.ones_like(input_ids)).to("cuda")
            token_hash = object_sha256(
                [int(value) for value in input_ids[0].detach().cpu().tolist()])
            expected_encoding = next(
                row for row in preflight_payload["prompt_encodings"]
                if row["prompt_id"] == prompt["prompt_id"])
            if token_hash != expected_encoding["token_ids_sha256"]:
                raise TransportValidationError("post-load prompt encoding drift")
            for layer in config["layers_zero_indexed"]:
                cell_id = _cell_id(prompt["prompt_id"], int(layer))
                if cell_id in state["completed_cells"]:
                    record = state["completed_cells"][cell_id]
                    for name in ("metrics", "raw"):
                        path = Path(record[name]["path"])
                        if not path.is_file() or file_sha256(path) != record[name][
                                "sha256"]:
                            raise TransportValidationError(
                                f"completed H6 cell drift: {cell_id}: {name}")
                    continue
                if max_cells is not None and new_cells >= max_cells:
                    _write_heartbeat(
                        paths["heartbeat"], phase="paused_by_max_cells",
                        model_key=model_key,
                        completed_cells=len(state["completed_cells"]))
                    del model, tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    return {
                        "status": "paused_by_max_cells",
                        "model_key": model_key,
                        "new_cells": new_cells,
                        "completed_cells": len(state["completed_cells"]),
                    }
                _write_heartbeat(
                    paths["heartbeat"], phase="transport_cell",
                    model_key=model_key, cell_id=cell_id,
                    completed_cells=len(state["completed_cells"]))
                suffix = ExplicitDecoderSuffix(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    source_layer=int(layer),
                    target=target,
                )
                parity = suffix.parity(atol=0.0, rtol=0.0)
                if not parity["ok"]:
                    raise TransportValidationError(
                        f"exact clean suffix parity failed: {cell_id}: {parity}")
                started = time.time()
                metadata = {
                    "study_id": "jspace-olmo-lineage-study2",
                    "tier": "methods",
                    "evidence_id": spec["evidence_id"],
                    "model_key": model_key,
                    "model_id": spec["model_id"],
                    "model_revision": spec["revision"],
                    "prompt_id": prompt["prompt_id"],
                    "prompt_sha256": hashlib.sha256(
                        prompt["text"].encode()).hexdigest(),
                    "token_ids_sha256": token_hash,
                    "source_layer": int(layer),
                    "source_position": int(config["source_position"]),
                    "target_stage": "final_decoder_layer",
                    "target_representation": config["target"],
                    "config_sha256": file_sha256(CONFIG),
                    "input_manifest_sha256": manifest[
                        "input_manifest_sha256"],
                    "execution_freeze_sha256": freeze["outputs"][0]["sha256"],
                    "license_import_sha256": license_payload[
                        "event_output"]["sha256"],
                    "code_commit": clean["code_commit"],
                }
                rows, raw = evaluate_dual_backend_transport_cell(
                    suffix,
                    attention_mask=attention_mask,
                    perturbation_mode=config["source_mode"],
                    direction_specs=_direction_specs(config),
                    epsilon_ladder=[
                        float(value) for value in config[
                            "relative_epsilon_ladder"]],
                    seed=int(config["directions"]["base_seed"]),
                    cell_id=cell_id,
                    metadata=metadata,
                    delivery_cosine_floor=float(
                        config["delivery"]["cosine_floor"]),
                    delivery_norm_error_ceiling=float(
                        config["delivery"]["relative_norm_error_ceiling"]),
                    batch_size=EVALUATION_BATCH_SIZE,
                )
                elapsed = time.time() - started
                raw.update({
                    "clean_suffix_parity": parity,
                    "runtime_seconds": elapsed,
                    "license": license_payload,
                })
                raw_path = paths["raw"] / f"{cell_id}.pt"
                metrics_path = paths["cells"] / f"{cell_id}.json"
                if raw_path.exists() or metrics_path.exists():
                    raise FileExistsError(
                        f"uncheckpointed H6 cell output already exists: {cell_id}")
                _atomic_torch_save(raw_path, raw)
                cell_payload = {
                    "schema_version": 1,
                    "cell_id": cell_id,
                    "metadata": metadata,
                    "clean_suffix_parity": parity,
                    "runtime_seconds": elapsed,
                    "rows": rows,
                    "raw": {
                        "path": str(raw_path),
                        "sha256": file_sha256(raw_path),
                        "bytes": int(raw_path.stat().st_size),
                    },
                    "finite_rows": all(
                        all(
                            value is None or not isinstance(value, float)
                            or math.isfinite(value)
                            for value in row.values()
                        ) for row in rows
                    ),
                }
                atomic_json(metrics_path, cell_payload)
                state["completed_cells"][cell_id] = {
                    "metrics": {
                        "path": str(metrics_path),
                        "sha256": file_sha256(metrics_path),
                        "bytes": int(metrics_path.stat().st_size),
                    },
                    "raw": cell_payload["raw"],
                    "runtime_seconds": elapsed,
                }
                _write_state(paths["state"], header, state)
                new_cells += 1
                _write_heartbeat(
                    paths["heartbeat"], phase="cell_banked",
                    model_key=model_key, cell_id=cell_id,
                    completed_cells=len(state["completed_cells"]),
                    runtime_seconds=elapsed)
                print(json.dumps({
                    "model_key": model_key,
                    "cell": cell_id,
                    "completed": len(state["completed_cells"]),
                    "runtime_seconds": elapsed,
                    "raw_sha256": cell_payload["raw"]["sha256"],
                }), flush=True)
                del suffix, raw, rows
                gc.collect()
                torch.cuda.empty_cache()

        expected_cells = len(config["prompt_ids"]) * len(
            config["layers_zero_indexed"])
        if len(state["completed_cells"]) != expected_cells:
            raise TransportValidationError("mandatory H6 model grid is incomplete")
        _write_heartbeat(
            paths["heartbeat"], phase="model_grid_complete",
            model_key=model_key, completed_cells=expected_cells)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return finalize_model(model_key)


def classify_rows(frame: pd.DataFrame, config: Mapping,
                  backend_ceiling: float) -> pd.DataFrame:
    result = frame.copy()
    delivery = result["faithful_delivery"].astype(bool)
    backend = (
        result["backend_tangent_relative_error"].notna()
        & (result["backend_tangent_relative_error"] <= float(backend_ceiling))
    )
    measurement = (
        delivery & backend & result["response_snr"].notna()
        & (result["response_snr"] >= float(
            config["response_snr"]["measurement_floor"]))
    )
    decision = measurement & (
        result["response_snr"] >= float(
            config["response_snr"]["decision_floor"]))
    gate = config["transport_gate"]
    forward = (
        result["tangent_cosine"].notna()
        & (result["tangent_cosine"] >= float(gate["tangent_cosine_floor"]))
        & result["tangent_relative_error"].notna()
        & (result["tangent_relative_error"] <= float(
            gate["forward_relative_error_ceiling"]))
    )
    central = (
        result["central_tangent_cosine"].notna()
        & (result["central_tangent_cosine"] >= float(
            gate["tangent_cosine_floor"]))
        & result["central_tangent_relative_error"].notna()
        & (result["central_tangent_relative_error"] <= float(
            gate["central_relative_error_ceiling"]))
    )
    result["backend_gate_passed"] = backend
    result["measurement_eligible"] = measurement
    result["decision_eligible"] = decision
    result["forward_gate_passed"] = forward
    result["central_gate_passed"] = central
    result["transport_row_passed"] = decision & forward & central
    return result


def _finite_median(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.median()) if len(clean) else None


def _fit_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    keys = ["model_key", "source_layer", "prompt_id", "direction_id"]
    for key, group in frame.groupby(keys, sort=True):
        selected = group[
            group.measurement_eligible
            & (group.desired_relative_epsilon <= 0.10)
        ].sort_values("desired_relative_epsilon")
        x = selected.desired_relative_epsilon.to_numpy(dtype=float)
        y = selected.tangent_relative_error.to_numpy(dtype=float)
        unique = np.unique(x)
        if len(unique) >= 3 and np.isfinite(x).all() and np.isfinite(y).all():
            slope, intercept = np.polyfit(x, y, 1)
            predicted = intercept + slope * x
            residual = float(np.sum((y - predicted) ** 2))
            total = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - residual / total if total > 0 else None
            status = "estimated"
        else:
            slope = intercept = r2 = None
            status = "insufficient_measurement_rows"
        rows.append({
            "model_key": str(key[0]),
            "source_layer": int(key[1]),
            "prompt_id": str(key[2]),
            "direction_id": str(key[3]),
            "n_rows": len(selected),
            "n_unique_epsilon": len(unique),
            "intercept": None if intercept is None else float(intercept),
            "slope": None if slope is None else float(slope),
            "r_squared": None if r2 is None else float(r2),
            "status": status,
        })
    return rows


def summarize_model(frame: pd.DataFrame, config: Mapping,
                    license_payload: Mapping) -> dict:
    passage = []
    floor = float(config["transport_gate"]["row_passage_floor"])
    for (layer, epsilon), group in frame.groupby(
            ["source_layer", "desired_relative_epsilon"], sort=True):
        n = len(group)
        passing = int(group.transport_row_passed.sum())
        row = {
            "source_layer": int(layer),
            "relative_epsilon": float(epsilon),
            "rows": n,
            "measurement_eligible": int(group.measurement_eligible.sum()),
            "decision_eligible": int(group.decision_eligible.sum()),
            "passing_rows": passing,
            "passage_fraction": passing / n,
            "layer_epsilon_passed": passing / n >= floor,
            "median_response_snr": _finite_median(group.response_snr),
            "median_forward_cosine": _finite_median(group.tangent_cosine),
            "median_forward_relative_error": _finite_median(
                group.tangent_relative_error),
            "median_central_cosine": _finite_median(
                group.central_tangent_cosine),
            "median_central_relative_error": _finite_median(
                group.central_tangent_relative_error),
            "median_backend_relative_error": _finite_median(
                group.backend_tangent_relative_error),
            "median_gain": _finite_median(group.gain),
            "median_homogeneity_defect": _finite_median(
                group.homogeneity_defect),
            "median_odd_symmetry_defect": _finite_median(
                group.odd_symmetry_defect),
            "median_additivity_defect": _finite_median(
                group.additivity_defect),
        }
        passage.append(row)
    assay = {int(value) for value in config["assay_band_layers"]}
    late = int(config["late_anchor_layer"])
    epsilon_values = [float(value) for value in config[
        "relative_epsilon_ladder"]]
    common = []
    late_valid = []
    layer_valid = {}
    for layer in [*sorted(assay), late]:
        valid = [
            row["relative_epsilon"] for row in passage
            if row["source_layer"] == layer and row["layer_epsilon_passed"]
        ]
        layer_valid[str(layer)] = valid
    for epsilon in epsilon_values:
        if all(epsilon in layer_valid[str(layer)] for layer in assay):
            common.append(epsilon)
        if epsilon in layer_valid[str(late)]:
            late_valid.append(epsilon)
    small_common = [value for value in common if value < 0.10]
    if small_common:
        intrinsic_route = "common_in_band_regime_measured"
    elif late_valid and any(not layer_valid[str(layer)] for layer in assay):
        intrinsic_route = "late_only_regime_measured"
    else:
        intrinsic_route = "no_common_licensed_regime_measured"
    return {
        "schema_version": 1,
        "rows": len(frame),
        "cells": int(frame.cell_id.nunique()),
        "prompts": int(frame.prompt_id.nunique()),
        "directions": int(frame.direction_id.nunique()),
        "epsilons": epsilon_values,
        "all_rows_finite": bool(
            np.isfinite(frame.select_dtypes(include=[np.number])).all().all()),
        "backend_license": dict(license_payload),
        "passage": passage,
        "valid_epsilons_by_layer": layer_valid,
        "common_assay_valid_epsilons": common,
        "small_common_assay_valid_epsilons": small_common,
        "late_anchor_valid_epsilons": late_valid,
        "intrinsic_transport_route": intrinsic_route,
        "relevant_dose_route": "pending_joint_registered_site_dose_mapping",
        "intercept_slope_fits": _fit_rows(frame),
        "claim_boundary": (
            "Per-checkpoint methods validation of the frozen finite-dose map; "
            "no causal-training attribution and no claim that transport "
            "validates or invalidates paired ablation effects."),
    }


def finalize_model(model_key: str) -> dict:
    config = _config()
    paths = _paths(config, model_key)
    license_payload = load_license()
    if not paths["state"].is_file():
        raise TransportValidationError("H6 model state is absent")
    envelope = json.loads(paths["state"].read_text())
    if object_sha256(envelope["payload"]) != envelope["payload_sha256"]:
        raise TransportValidationError("H6 state hash drift before finalize")
    state = envelope["payload"]
    expected = len(config["prompt_ids"]) * len(config["layers_zero_indexed"])
    if len(state["completed_cells"]) != expected:
        raise TransportValidationError(
            f"H6 model grid incomplete: {len(state['completed_cells'])}/{expected}")
    all_rows = []
    raw_files = []
    metric_files = []
    for cell_id in sorted(state["completed_cells"]):
        record = state["completed_cells"][cell_id]
        for kind in ("metrics", "raw"):
            path = Path(record[kind]["path"])
            if not path.is_file() or file_sha256(path) != record[kind]["sha256"]:
                raise TransportValidationError(
                    f"H6 cell hash drift before finalize: {cell_id}: {kind}")
        metrics = json.loads(Path(record["metrics"]["path"]).read_text())
        if not metrics["finite_rows"]:
            raise TransportValidationError(f"H6 cell has nonfinite row: {cell_id}")
        all_rows.extend(metrics["rows"])
        metric_files.append(record["metrics"])
        raw_files.append(record["raw"])
    frame = pd.DataFrame(all_rows)
    expected_rows = expected * len(config["directions"]["families"]) * len(
        config["relative_epsilon_ladder"])
    if len(frame) != expected_rows:
        raise TransportValidationError(
            f"H6 scalar row count drift: {len(frame)}/{expected_rows}")
    classified = classify_rows(
        frame, config, license_payload["backend_relative_error_ceiling"])
    summary = summarize_model(classified, config, license_payload)
    summary.update({
        "evidence_id": _model_spec(config, model_key)["evidence_id"],
        "model_key": model_key,
        "model_id": _model_spec(config, model_key)["model_id"],
        "model_revision": _model_spec(config, model_key)["revision"],
        "config_sha256": file_sha256(CONFIG),
        "input_manifest_sha256": json.loads(
            paths["input_manifest"].read_text())["input_manifest_sha256"],
        "state_sha256": file_sha256(paths["state"]),
        "snapshot_conformance_sha256": file_sha256(paths["preflight"]),
        "model_audit": state["model_audit"],
        "gpu": state["gpu"],
    })
    summary["payload_sha256"] = object_sha256(summary)
    raw_inventory = {
        "schema_version": 1,
        "evidence_id": summary["evidence_id"],
        "cell_metrics": metric_files,
        "raw_vectors": raw_files,
        "all_files_verified": True,
    }
    raw_inventory["inventory_sha256"] = object_sha256({
        "cell_metrics": metric_files, "raw_vectors": raw_files})
    classified.sort_values([
        "source_layer", "prompt_id", "direction_id",
        "desired_relative_epsilon",
    ]).to_parquet(paths["rows"], index=False)
    atomic_json(paths["summary"], summary)
    atomic_json(paths["raw_inventory"], raw_inventory)
    return {
        "model_key": model_key,
        "evidence_id": summary["evidence_id"],
        "rows": len(classified),
        "cells": expected,
        "intrinsic_route": summary["intrinsic_transport_route"],
        "common_valid_epsilons": summary["common_assay_valid_epsilons"],
        "summary": str(paths["summary"]),
        "summary_sha256": file_sha256(paths["summary"]),
        "rows_sha256": file_sha256(paths["rows"]),
        "raw_inventory_sha256": file_sha256(paths["raw_inventory"]),
    }


def register_model(model_key: str) -> dict:
    clean = require_clean_tree(expected_branch=BRANCH)
    config = _config()
    paths = _paths(config, model_key)
    result = finalize_model(model_key)
    spec = _model_spec(config, model_key)
    origins = {
        row["evidence_id"] for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
    }
    if spec["evidence_id"] in origins:
        raise TransportValidationError(
            f"H6 model event already registered: {spec['evidence_id']}")
    summary = json.loads(paths["summary"].read_text())
    outputs = [
        paths["summary"], paths["rows"], paths["raw_inventory"],
        paths["preflight"], paths["input_manifest"], paths["state"],
    ]
    event = create(
        spec["evidence_id"],
        tier="methods",
        what=(
            f"Frozen H6 finite-dose dual-exact-JVP transport validation for "
            f"{model_key}; relevant-dose routing remains a separate exact "
            "registered-site join."
        ),
        command=(
            "python -m jspace_olmo_lineage.experiments.transport_validation "
            f"--phase register --model {model_key}"
        ),
        outputs=outputs,
        inputs={
            "config_sha256": file_sha256(CONFIG),
            "execution_freeze_output_sha256": _event_output(
                EXECUTION_FREEZE)["outputs"][0]["sha256"],
            "calibration_import_output_sha256": load_license()[
                "event_output"]["sha256"],
            "snapshot_manifest_sha256": file_sha256(
                paths["snapshot_manifest"]),
            "prompt_bank_sha256": file_sha256(PROMPT_BANK),
        },
        model_key=model_key,
        model_id=spec["model_id"],
        model_revision=spec["revision"],
        rows=int(summary["rows"]),
        cells=int(summary["cells"]),
        route=summary["intrinsic_transport_route"],
        common_assay_valid_epsilons=summary[
            "common_assay_valid_epsilons"],
        relevant_dose_route=summary["relevant_dose_route"],
        backend_license_status=summary["backend_license"]["status"],
        pooled_ceiling_used=False,
        code_producer_commit=clean["code_commit"],
    )
    return {
        **result,
        "registry_event_utc": event["event_utc"],
        "registry_code_commit": event["code_commit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True,
        choices=(
            "freeze", "correct-bos", "correct-shape", "stage", "preflight",
            "run", "finalize", "register",
        ))
    parser.add_argument("--model", choices=("base", "olmo31_think"))
    parser.add_argument("--max-cells", type=int)
    arguments = parser.parse_args()
    if arguments.phase == "freeze":
        if arguments.model is not None:
            parser.error("--model is forbidden for freeze")
        result = freeze_execution()
    elif arguments.phase == "correct-bos":
        if arguments.model is not None:
            parser.error("--model is forbidden for correct-bos")
        result = register_bos_correction()
    elif arguments.phase == "correct-shape":
        if arguments.model is not None:
            parser.error("--model is forbidden for correct-shape")
        result = register_shape_correction()
    else:
        if arguments.model is None:
            parser.error("--model is required outside freeze")
        config = _config()
        if arguments.phase == "stage":
            result = stage(arguments.model)
        elif arguments.phase == "preflight":
            clean = require_clean_tree(expected_branch=BRANCH)
            snapshot, manifest = _verified_snapshot(config, arguments.model)
            payload = preflight(config, arguments.model, snapshot, manifest)
            input_payload = _input_manifest(
                config, arguments.model, payload, clean)
            result = {
                "preflight": str(_paths(config, arguments.model)["preflight"]),
                "prompt_encoding_sha256": payload["prompt_encoding_sha256"],
                "input_manifest_sha256": input_payload[
                    "input_manifest_sha256"],
            }
        elif arguments.phase == "run":
            result = run_model(arguments.model, arguments.max_cells)
        elif arguments.phase == "finalize":
            result = finalize_model(arguments.model)
        else:
            result = register_model(arguments.model)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
