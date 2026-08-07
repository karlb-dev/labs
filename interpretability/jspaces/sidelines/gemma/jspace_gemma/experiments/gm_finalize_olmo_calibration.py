"""Pure finalizer for the immutable 56-cell OLMo calibration checkpoint."""
from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from jspace_gemma.calibration_audit import audit_completed_checkpoint
from jspace_gemma.experiments.gm_exact_transport_gate import _aggregate
from jspace_gemma.gpu import require_cuda
from jspace_gemma.manifests import (
    atomic_json,
    environment_payload,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import PACKAGE_ROOT, directory, local_root
from jspace_gemma.registry import create, read_events, resolve

EVIDENCE_ID = "gm-jvp-olmo-calibration-v1"
DIAGNOSTIC_ID = "gm-olmo-calibration-finalize-diagnostic-v1"
COMPUTE_COMMIT = "06b2a3d2fbe42fd5f70abb121573b1e7a62b45ec"
STATE_SHA256 = "f696f28cecc44d3a3d925308dd10226f1f7fa84e09e6e63ff37913ea3960278c"
DIAGNOSTIC_SHA256 = "78d53fca50b2a8ac2e114f71a7900a3581214e5367b0892dadf624ec736e8e25"
DIAGNOSTIC = directory("manifests") / "gm_olmo_calibration_finalize_diagnostic_v1.json"
OUTPUT_ROOT = directory("metrics") / "olmo_control" / EVIDENCE_ID


def _git_source(commit: str, path: str) -> dict:
    spec = f"{commit}:{path}"
    payload = subprocess.check_output(["git", "show", spec])
    return {
        "commit": commit,
        "path": path,
        "git_blob_id": subprocess.check_output(
            ["git", "rev-parse", spec], text=True
        ).strip(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _parquet_frame(rows: list[dict]) -> pd.DataFrame:
    """Make the mixed JSON source-position union explicit for Arrow storage."""
    frame = pd.DataFrame(rows)
    frame["source_position_runtime_type"] = frame["source_position"].map(
        lambda value: type(value).__name__
    )
    frame["source_position"] = frame["source_position"].map(str)
    return frame


def main() -> None:
    lock_path = local_root() / "locks/gm_finalize_olmo_calibration.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    process_lock = lock_path.open("w")
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another OLMo finalizer owns {lock_path}") from exc
    process_lock.write(str(os.getpid()))
    process_lock.flush()

    git = require_clean_tree()
    origins = {
        row["evidence_id"]
        for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
    }
    if EVIDENCE_ID in origins:
        raise RuntimeError("OLMo calibration evidence is already registered")
    diagnostic_event = resolve(DIAGNOSTIC_ID)
    if not diagnostic_event["live"] or diagnostic_event["code_commit"] != (
        "a196c4fdf267944c1b5d9daa467aadcbd65b93ce"
    ):
        raise RuntimeError("the frozen incident diagnostic is not live")
    if file_sha256(DIAGNOSTIC) != DIAGNOSTIC_SHA256:
        raise RuntimeError("incident diagnostic manifest hash drifted")

    outputs = {
        "summary": OUTPUT_ROOT / "olmo_calibration_summary.json",
        "rows": OUTPUT_ROOT / "olmo_calibration_rows.parquet",
        "raw_inventory": OUTPUT_ROOT / "raw_inventory.json",
        "finalization": OUTPUT_ROOT / "olmo_calibration_finalization_v1.json",
    }
    present = [str(path) for path in outputs.values() if path.exists()]
    if present:
        raise RuntimeError(f"refusing to overwrite finalization outputs: {present}")

    audit = audit_completed_checkpoint(
        OUTPUT_ROOT,
        expected_evidence_id=EVIDENCE_ID,
        expected_compute_commit=COMPUTE_COMMIT,
        expected_cells=56,
        expected_rows_per_cell=28,
        expected_parity_rows=28,
        inspect_raw_tensors=True,
    )
    if audit["state_sha256"] != STATE_SHA256:
        raise RuntimeError("complete calibration state hash drifted")

    source_paths = {
        "config": "interpretability/jspaces/sidelines/gemma/configs/gm_g1_design.yaml",
        "prompt_bank": "interpretability/jspaces/sidelines/gemma/data/g1_prompts_v1.jsonl",
        "autodiff": "interpretability/jspaces/sidelines/gemma/jspace_gemma/autodiff.py",
        "transport": "interpretability/jspaces/sidelines/gemma/jspace_gemma/transport.py",
        "compute_producer": (
            "interpretability/jspaces/sidelines/gemma/jspace_gemma/experiments/"
            "gm_exact_transport_gate.py"
        ),
    }
    compute_sources = {
        name: _git_source(COMPUTE_COMMIT, path)
        for name, path in source_paths.items()
    }
    input_manifest = json.loads(Path(audit["input_manifest_path"]).read_text())
    if compute_sources["config"]["sha256"] != input_manifest["config"]["sha256"]:
        raise RuntimeError("compute-commit config does not match the input manifest")
    if compute_sources["prompt_bank"]["sha256"] != input_manifest["prompt_bank"]["sha256"]:
        raise RuntimeError("compute-commit prompt bank does not match the input manifest")
    if compute_sources["autodiff"]["sha256"] != audit["autodiff_implementation_sha256"]:
        raise RuntimeError("compute-commit autodiff source differs from row provenance")
    if compute_sources["transport"]["sha256"] != audit["transport_implementation_sha256"]:
        raise RuntimeError("compute-commit transport source differs from row provenance")
    snapshot_manifest = Path(input_manifest["snapshot_manifest"]["path"])
    if file_sha256(snapshot_manifest) != input_manifest["snapshot_manifest"]["sha256"]:
        raise RuntimeError("OLMo snapshot manifest hash drifted")

    environment = environment_payload(require_gpu=True)
    environment_compat = {
        key: value for key, value in environment.items() if key != "created_utc"
    }
    if object_sha256(environment_compat) != audit["header"]["environment_sha256"]:
        raise RuntimeError("finalization environment differs from compute environment")
    gpu = require_cuda()

    rows = audit["all_rows"]
    summary = _aggregate(
        rows,
        [
            row
            for _, row in sorted(
                json.loads(Path(audit["state_path"]).read_text())["payload"]["parity"].items()
            )
        ],
        audit["wrong_hook"],
    )
    summary.update(
        {
            "schema_version": 2,
            "evidence_id": EVIDENCE_ID,
            "tier": "methods",
            "model_id": audit["header"]["model_id"],
            "model_revision": audit["header"]["model_revision"],
            "input_manifest_sha256": audit["input_manifest_object_sha256"],
            "expected_cells": 56,
            "completed_cells": audit["completed_cells"],
            "cell_compute_code_commit": COMPUTE_COMMIT,
            "finalization_code_commit": git["code_commit"],
            "cell_environment_sha256": audit["header"]["environment_sha256"],
            "finalization_environment": environment,
            "gpu_gate_at_finalization": gpu,
            "state_sha256": audit["state_sha256"],
            "checkpoint_inventory_sha256": audit["inventory_sha256"],
            "incident_diagnostic_sha256": DIAGNOSTIC_SHA256,
            "cells_recomputed_during_finalization": False,
            "model_loaded_during_finalization": False,
            "target_model_opened": False,
            "parquet_normalizations": {
                "source_position": (
                    "canonical string plus source_position_runtime_type; "
                    "per-cell JSON retains the original int/string union"
                )
            },
            "claim_boundary": (
                "OLMo-only threshold calibration; no Gemma target threshold "
                "or target-model result"
            ),
        }
    )
    # Prove the native-scalar repair before creating any final output.
    json.dumps(summary, allow_nan=False, sort_keys=True)

    raw_inventory = {
        "schema_version": 2,
        "compute_code_commit": COMPUTE_COMMIT,
        "state_sha256": audit["state_sha256"],
        "files": audit["inventory"],
        "inventory_sha256": audit["inventory_sha256"],
    }
    atomic_json(outputs["summary"], summary)
    _atomic_parquet(outputs["rows"], _parquet_frame(rows))
    if len(pd.read_parquet(outputs["rows"])) != len(rows):
        raise RuntimeError("finalized Parquet row count differs from checkpoint")
    atomic_json(outputs["raw_inventory"], raw_inventory)

    finalization = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compute_code_commit": COMPUTE_COMMIT,
        "finalization_code_commit": git["code_commit"],
        "compute_sources": compute_sources,
        "finalizer": {
            "path": str(PACKAGE_ROOT / "jspace_gemma/experiments/gm_finalize_olmo_calibration.py"),
            "sha256": file_sha256(
                PACKAGE_ROOT / "jspace_gemma/experiments/gm_finalize_olmo_calibration.py"
            ),
        },
        "aggregate_implementation": {
            "path": str(PACKAGE_ROOT / "jspace_gemma/experiments/gm_exact_transport_gate.py"),
            "sha256": file_sha256(
                PACKAGE_ROOT / "jspace_gemma/experiments/gm_exact_transport_gate.py"
            ),
        },
        "source_checkpoint": {
            "state_sha256": audit["state_sha256"],
            "inventory_sha256": audit["inventory_sha256"],
            "completed_cells": audit["completed_cells"],
            "rows": audit["rows"],
            "parity_rows": audit["parity_rows"],
        },
        "incident_diagnostic": {
            "path": str(DIAGNOSTIC),
            "sha256": DIAGNOSTIC_SHA256,
        },
        "outputs": [
            {"path": str(path), "sha256": file_sha256(path)}
            for name, path in outputs.items()
            if name != "finalization"
        ],
        "cells_recomputed": False,
        "model_loaded_during_finalization": False,
        "target_model_opened": False,
    }
    atomic_json(outputs["finalization"], finalization)
    create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "OLMo-only exact-JVP/secant calibration grid finalized from 56 "
            "immutable hash-verified cells for pre-Gemma threshold freezing"
        ),
        command="python -m jspace_gemma.experiments.gm_finalize_olmo_calibration",
        outputs=[
            Path(audit["input_manifest_path"]),
            outputs["summary"],
            outputs["rows"],
            outputs["raw_inventory"],
            outputs["finalization"],
        ],
        inputs={
            "compute_code_commit": COMPUTE_COMMIT,
            "finalization_code_commit": git["code_commit"],
            "state_sha256": audit["state_sha256"],
            "checkpoint_inventory_sha256": audit["inventory_sha256"],
            "incident_diagnostic_sha256": DIAGNOSTIC_SHA256,
            "snapshot_manifest_sha256": file_sha256(snapshot_manifest),
        },
        cells_recomputed=False,
        control_model_opened_during_compute=True,
        model_loaded_during_finalization=False,
        target_model_opened=False,
    )
    print(
        json.dumps(
            {
                "summary": str(outputs["summary"]),
                "summary_sha256": file_sha256(outputs["summary"]),
                "rows": len(rows),
                "cells": audit["completed_cells"],
                "finalization": str(outputs["finalization"]),
                "finalization_sha256": file_sha256(outputs["finalization"]),
            },
            indent=1,
        )
    )
    gc.collect()


if __name__ == "__main__":
    main()
