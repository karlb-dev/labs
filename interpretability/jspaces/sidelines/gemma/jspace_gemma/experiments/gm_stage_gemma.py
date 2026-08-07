"""Stage the pinned Gemma target snapshot after the pre-target firewall passes."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from jspace_gemma.gpu import require_cuda
from jspace_gemma.manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import PACKAGE_ROOT, directory
from jspace_gemma.registry import EVENTS, resolve
from jspace_gemma.staging import stage_snapshot

REPO_ID = "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
EVIDENCE_ID = "gm-jvp-olmo-positive-control-v1"
CONFIG = PACKAGE_ROOT / "configs/gm_g1_design.yaml"
THRESHOLDS = PACKAGE_ROOT / "configs/gm_g1_thresholds_frozen.yaml"
SEED = Path("/content/hf_local/models--google--gemma-4-31B-it")
CACHE = Path("/content/hf_gemma_target")


def require_gemma_staging_gate(
    *,
    config_path: Path = CONFIG,
    thresholds_path: Path = THRESHOLDS,
    registry_path: Path = EVENTS,
) -> dict:
    """Bind staging permission to the immutable, passing OLMo control event."""
    config = yaml.safe_load(config_path.read_text())
    role = config["models"]["gemma_target"]
    if role["model_id"] != REPO_ID or role["revision"] != REVISION:
        raise RuntimeError("Gemma staging constants differ from the frozen design")

    calibration = config["threshold_calibration"]
    if calibration["status"] != "FROZEN_PRE_GEMMA_REGISTERED":
        raise RuntimeError("Gemma staging requires the registered pre-target state")
    if calibration["gemma_execution_allowed"] is not True:
        raise RuntimeError("Gemma execution firewall is closed")
    if calibration["threshold_evidence_id"] != EVIDENCE_ID:
        raise RuntimeError("frozen design names a different threshold evidence event")
    expected_threshold_uri = (
        "repo://interpretability/jspaces/sidelines/gemma/configs/"
        "gm_g1_thresholds_frozen.yaml"
    )
    if calibration["frozen_threshold_file"] != expected_threshold_uri:
        raise RuntimeError("frozen design names an unexpected threshold file")

    threshold_sha = file_sha256(thresholds_path)
    event = resolve(EVIDENCE_ID, path=registry_path)
    required_true = ("live", "positive_control_pass", "thresholds_frozen_before_target")
    if any(event.get(field) is not True for field in required_true):
        raise RuntimeError("positive-control event is absent, non-live, or failing")
    if event.get("target_model_opened") is not False:
        raise RuntimeError("positive-control event does not preserve the target firewall")
    if event.get("inputs", {}).get("threshold_config_sha256") != threshold_sha:
        raise RuntimeError("registered threshold hash differs from the tracked file")

    artifact_sha = calibration["positive_control_artifact_sha256"]
    matching_artifacts = [
        row for row in event.get("outputs", []) if row.get("sha256") == artifact_sha
    ]
    if len(matching_artifacts) != 1:
        raise RuntimeError("registered positive-control artifact binding is ambiguous")
    artifact_path = Path(matching_artifacts[0]["path"])
    if not artifact_path.is_file() or file_sha256(artifact_path) != artifact_sha:
        raise RuntimeError("registered positive-control artifact is absent or hash-drifted")

    threshold_outputs = [
        row for row in event.get("outputs", []) if row.get("sha256") == threshold_sha
    ]
    if len(threshold_outputs) != 1:
        raise RuntimeError("positive-control event does not bind the threshold file once")
    return {
        "status": calibration["status"],
        "gemma_execution_allowed": True,
        "threshold_evidence_id": EVIDENCE_ID,
        "threshold_evidence_code_commit": event["code_commit"],
        "threshold_config_path": str(thresholds_path.resolve()),
        "threshold_config_sha256": threshold_sha,
        "positive_control_artifact_path": str(artifact_path.resolve()),
        "positive_control_artifact_sha256": artifact_sha,
        "target_model_opened_by_threshold_event": False,
    }


def main() -> None:
    git = require_clean_tree()
    gpu = require_cuda()
    gate = require_gemma_staging_gate()
    output = directory("manifests") / "gemma_target_local_snapshot_v1.json"
    result = stage_snapshot(
        repo_id=REPO_ID,
        revision=REVISION,
        cache_root=CACHE,
        seed_model_root=SEED,
        output_manifest=output,
    )
    result.pop("snapshot_manifest_sha256", None)
    result["pretarget_gate"] = gate
    result["staging_code_commit"] = git["code_commit"]
    result["target_model_loaded"] = False
    result["target_response_created"] = False
    result["snapshot_manifest_sha256"] = object_sha256(result)
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "snapshot": result["snapshot"],
                "manifest": str(output),
                "manifest_sha256": file_sha256(output),
                "snapshot_manifest_sha256": result["snapshot_manifest_sha256"],
                "weight_shards": len(result["weight_shards"]),
                "all_content_hashes_verified": result["all_content_hashes_verified"],
                "target_model_loaded": False,
                "code_commit": git["code_commit"],
                "gpu": gpu,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
