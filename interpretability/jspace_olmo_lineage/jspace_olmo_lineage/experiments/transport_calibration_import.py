"""Hash-locked Gemma G2.1 calibration import for OLMo H6.

The source registry is read from an immutable Git blob rather than a mutable
working-tree path.  This command must complete and register its envelope before
any H6 verdict is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ..imports import ImportBoundaryError, resolve_source_event
from ..manifests import atomic_json, file_sha256, object_sha256, require_clean_tree
from ..paths import manifests_dir
from ..registry import append_event, read_events


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BRANCH = "interp_jspace_olmo_lineage_2"
CONFIG = PACKAGE_ROOT / "configs/ol2_transport_validation.yaml"
SOURCE_REPO_DEFAULT = Path("/content/labs_gemma2")
SOURCE_REGISTRY_PATH = Path(
    "interpretability/jspace_gemma/reports/evidence_events.jsonl")
SOURCE_REGISTRY_COMMIT = "428bfe125c204850d9193b1f12b0d790133fa751"
SOURCE_REGISTRY_SHA256 = (
    "c1117711bc26901747a08ce02cf91c80c6d4db05595b10eea07a5fa2f7e09f5d")
SOURCE_EVENT_ID = "gm2-backend-parity-calibration-v1"
IMPORT_EVENT_ID = "ol2-gemma-backend-calibration-import-v1"
MODEL_SCOPE_KEY = "olmo3_32b_control"


def _git_bytes(repo: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *arguments])


def source_registry_events(source_repo: Path) -> tuple[list[dict], dict]:
    """Read and authenticate the source registry at the frozen source commit."""
    try:
        blob = _git_bytes(
            source_repo, "show",
            f"{SOURCE_REGISTRY_COMMIT}:{SOURCE_REGISTRY_PATH.as_posix()}")
    except subprocess.CalledProcessError as error:
        raise ImportBoundaryError(
            "frozen Gemma source registry Git blob is unavailable") from error
    observed = hashlib.sha256(blob).hexdigest()
    if observed != SOURCE_REGISTRY_SHA256:
        raise ImportBoundaryError(
            f"Gemma source registry blob drift: {observed}")
    events = [
        json.loads(line) for line in blob.decode("utf-8").splitlines()
        if line.strip()
    ]
    return events, {
        "repository": str(source_repo.resolve()),
        "relative_path": SOURCE_REGISTRY_PATH.as_posix(),
        "commit": SOURCE_REGISTRY_COMMIT,
        "sha256": observed,
        "bytes": len(blob),
    }


def verified_source_outputs(outputs: Sequence[Mapping]) -> list[dict]:
    verified = []
    for row in outputs:
        path = Path(str(row["path"]))
        if not path.is_file():
            raise ImportBoundaryError(f"Gemma source output is absent: {path}")
        observed = file_sha256(path)
        if observed != row.get("sha256"):
            raise ImportBoundaryError(
                f"Gemma source output hash drift: {path}; {observed}")
        verified.append({
            "path": str(path),
            "sha256": observed,
            "bytes": int(path.stat().st_size),
        })
    if not verified:
        raise ImportBoundaryError("Gemma calibration import has no outputs")
    return verified


def extract_olmo_license(event: Mapping, outputs: Sequence[Mapping]) -> dict:
    """Select only the architecture-specific OLMo ceiling from G2.1."""
    if event.get("route") != "benign_scheduling_floor":
        raise ImportBoundaryError(
            f"G2.1 route does not license H6: {event.get('route')!r}")
    if not event.get("no_target_read_assertion"):
        raise ImportBoundaryError("G2.1 target-isolation assertion is absent")
    by_model = event.get("licensed_ceilings", {}).get("by_model", {})
    if MODEL_SCOPE_KEY not in by_model:
        raise ImportBoundaryError("G2.1 lacks the OLMo-specific ceiling")
    threshold_rows = [
        row for row in outputs
        if Path(str(row["path"])).name == "backend_ceiling_frozen.json"
    ]
    if len(threshold_rows) != 1:
        raise ImportBoundaryError(
            "expected exactly one frozen backend-ceiling artifact")
    threshold_row = dict(threshold_rows[0])
    expected_hash = event.get("inputs", {}).get("threshold_sha256_pre_registry")
    if threshold_row["sha256"] != expected_hash:
        raise ImportBoundaryError("pre-registry threshold hash mismatch")
    threshold = json.loads(Path(threshold_row["path"]).read_text())
    artifact_ceiling = threshold.get("licensed_ceilings", {}).get(
        "by_model", {}).get(MODEL_SCOPE_KEY)
    registry_ceiling = by_model[MODEL_SCOPE_KEY]
    if artifact_ceiling != registry_ceiling:
        raise ImportBoundaryError(
            "OLMo-specific ceiling differs between registry and artifact")
    return {
        "route": event["route"],
        "model_scope_key": MODEL_SCOPE_KEY,
        "backend_relative_error_ceiling": float(registry_ceiling),
        "applicable_scope": event.get("applicable_scope"),
        "threshold_artifact": threshold_row,
        "pooled_ceiling_imported": False,
        "pooled_ceiling_use_forbidden": True,
    }


def _config() -> dict:
    config = yaml.safe_load(CONFIG.read_text())
    if config.get("status") != "FROZEN_PRE_TRANSPORT_DATA":
        raise ImportBoundaryError("H6 contract is not frozen")
    dependency = config["license_dependency"]
    if dependency["evidence_id"] != SOURCE_EVENT_ID:
        raise ImportBoundaryError("frozen H6 license dependency drift")
    if dependency["architecture_dependent_route"] != (
            "use_olmo_specific_ceiling_only"):
        raise ImportBoundaryError("OLMo-specific license rule drift")
    if not dependency["pooled_gemma_ceiling_as_convenience"] == "forbidden":
        raise ImportBoundaryError("pooled-ceiling prohibition drift")
    return config


def run(source_repo: Path = SOURCE_REPO_DEFAULT) -> dict:
    git = require_clean_tree(expected_branch=BRANCH)
    config = _config()
    origins = {
        row["evidence_id"] for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
    }
    if IMPORT_EVENT_ID in origins:
        raise ImportBoundaryError(f"import already registered: {IMPORT_EVENT_ID}")

    events, registry_source = source_registry_events(source_repo)
    source_event = resolve_source_event(events, SOURCE_EVENT_ID)
    if source_event.get("tier") != "methods":
        raise ImportBoundaryError("Gemma G2.1 import must remain methods tier")
    source_outputs = verified_source_outputs(source_event.get("outputs", []))
    license_payload = extract_olmo_license(source_event, source_outputs)

    payload = {
        "schema_version": 1,
        "evidence_id": IMPORT_EVENT_ID,
        "tier": "methods",
        "status": "exact_gemma_g2_1_license_imported_before_h6_verdict",
        "import_code_commit": git["code_commit"],
        "transport_config_sha256": file_sha256(CONFIG),
        "source_study": config["license_dependency"]["source_study"],
        "source_evidence_id": SOURCE_EVENT_ID,
        "source_event_code_commit": source_event["code_commit"],
        "source_event_sha256": object_sha256({
            key: value for key, value in source_event.items()
            if key not in {"live", "status_events"}
        }),
        "source_registry": registry_source,
        "source_outputs": source_outputs,
        "licensed_import": license_payload,
        "import_envelope_registered_before_h6_verdict": True,
        "h6_measurements_opened": False,
        "h6_verdict_opened": False,
    }
    payload["payload_sha256"] = object_sha256(payload)
    output = manifests_dir() / "ol2_gemma_backend_calibration_import_v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite import envelope: {output}")
    atomic_json(output, payload)

    append_event({
        "event": "evidence_imported",
        "evidence_id": IMPORT_EVENT_ID,
        "tier": "methods",
        "what": (
            "Exact registered Gemma G2.1 backend-calibration import exposing "
            "only the OLMo-specific exact-JVP relative-error ceiling for H6."
        ),
        "command": (
            "python -m jspace_olmo_lineage.experiments."
            "transport_calibration_import"
        ),
        "source_study": payload["source_study"],
        "source_evidence_id": SOURCE_EVENT_ID,
        "source_commit": source_event["code_commit"],
        "source_registry_commit": SOURCE_REGISTRY_COMMIT,
        "source_registry_sha256": registry_source["sha256"],
        "source_outputs": source_outputs,
        "import_code_commit": git["code_commit"],
        "outputs": [{
            "path": str(output),
            "sha256": file_sha256(output),
            "bytes": int(output.stat().st_size),
        }],
        "inputs": {
            "transport_config_sha256": file_sha256(CONFIG),
            "source_event_sha256": payload["source_event_sha256"],
        },
        "route": license_payload["route"],
        "licensed_model_scope_key": MODEL_SCOPE_KEY,
        "licensed_backend_relative_error_ceiling": license_payload[
            "backend_relative_error_ceiling"],
        "pooled_ceiling_imported": False,
        "h6_verdict_opened": False,
    })
    return {
        "event": IMPORT_EVENT_ID,
        "output": str(output),
        "sha256": file_sha256(output),
        "licensed_ceiling": license_payload["backend_relative_error_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, default=SOURCE_REPO_DEFAULT)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.source_repo), indent=1))


if __name__ == "__main__":
    main()
