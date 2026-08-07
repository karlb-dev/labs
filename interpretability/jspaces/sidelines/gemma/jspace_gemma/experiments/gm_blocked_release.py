"""Publish the terminal methods-blocker release without model execution."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import yaml

from jspace_gemma.manifests import (
    atomic_json,
    atomic_text,
    environment_payload,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import PACKAGE_ROOT, directory
from jspace_gemma.registry import EVENTS, create, read_events, resolve, resolve_all

EVIDENCE_ID = "gm-state-of-record-v1"
CONFIG = PACKAGE_ROOT / "configs/gm_blocked_release.yaml"
RELEASE_SOURCE = PACKAGE_ROOT / "release"


def _record(path: str | Path) -> dict:
    source = Path(path)
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "bytes": source.stat().st_size,
    }


def _output_by_hash(event: dict, expected_sha256: str) -> Path:
    rows = [
        Path(row["path"])
        for row in event.get("outputs", [])
        if row["sha256"] == expected_sha256
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one output with hash {expected_sha256}, found {len(rows)}"
        )
    return rows[0]


def _critical_event(spec: dict) -> dict:
    event = resolve(spec["evidence_id"])
    if (
        not event["live"]
        or event["event"] != "evidence_created"
        or event["code_commit"] != spec["code_commit"]
    ):
        raise RuntimeError(f"critical evidence drifted: {spec['evidence_id']}")
    return event


def _registry_inventory(config: dict) -> tuple[dict, bytes]:
    prefix = config["source_registry_prefix"]
    raw = EVENTS.read_bytes()
    if (
        len(raw) != int(prefix["expected_bytes"])
        or file_sha256(EVENTS) != prefix["expected_sha256"]
    ):
        raise RuntimeError("release source-registry prefix drifted")
    rows = read_events()
    if (
        not rows
        or rows[-1]["evidence_id"] != prefix["through_evidence_id"]
        or rows[-1]["event"] != "evidence_created"
    ):
        raise RuntimeError("release registry does not end at the frozen blocker")
    events = []
    checked_outputs = 0
    for event in resolve_all():
        if not event["live"]:
            raise RuntimeError("blocked release refuses non-live source evidence")
        output_field = (
            "source_outputs" if event["event"] == "evidence_imported" else "outputs"
        )
        output_rows = []
        for output in event.get(output_field, []) or []:
            path = Path(output["path"])
            if not path.exists() or file_sha256(path) != output["sha256"]:
                raise RuntimeError(
                    f"release source output hash drift: {event['evidence_id']}: {path}"
                )
            output_rows.append(_record(path))
            checked_outputs += 1
        events.append(
            {
                "evidence_id": event["evidence_id"],
                "event": event["event"],
                "tier": event["tier"],
                "code_commit": event.get("code_commit"),
                "import_code_commit": event.get("import_code_commit"),
                "outputs": output_rows,
                "live": True,
            }
        )
    if (
        len(events) != int(prefix["expected_live_events"])
        or checked_outputs != int(prefix["expected_verified_outputs"])
    ):
        raise RuntimeError("release registry event/output counts drifted")
    return (
        {
            "schema_version": 1,
            "registry": {
                "path": str(EVENTS),
                "prefix_bytes": len(raw),
                "prefix_sha256": file_sha256(EVENTS),
                "through_evidence_id": rows[-1]["evidence_id"],
            },
            "n_live_events": len(events),
            "n_verified_outputs": checked_outputs,
            "failures": [],
            "events": events,
        },
        raw,
    )


def _validate_terminal_sources(config: dict) -> dict:
    critical = config["critical_evidence"]
    goldens = _critical_event(critical["goldens"])
    _output_by_hash(goldens, critical["goldens"]["artifact_sha256"])

    control = _critical_event(critical["positive_control"])
    _output_by_hash(control, critical["positive_control"]["artifact_sha256"])
    if control.get("positive_control_pass") is not True:
        raise RuntimeError("release source positive control is not passing")

    stage1 = _critical_event(critical["gemma_stage1"])
    stage_summary_path = _output_by_hash(
        stage1, critical["gemma_stage1"]["summary_sha256"]
    )
    _output_by_hash(stage1, critical["gemma_stage1"]["rows_sha256"])
    _output_by_hash(stage1, critical["gemma_stage1"]["state_sha256"])
    stage_summary = json.loads(stage_summary_path.read_text())
    decisions = stage_summary["analysis"]["primary_layer_decisions"]
    if (
        int(stage_summary["completed_cells"]) != 40
        or len(decisions) != 5
        or {row["decision"] for row in decisions} != {"local_tangent_mismatch"}
    ):
        raise RuntimeError("release source Stage-1 classification drifted")

    blocker = _critical_event(critical["backend_blocker"])
    blocker_path = _output_by_hash(
        blocker, critical["backend_blocker"]["artifact_sha256"]
    )
    blocker_raw_path = _output_by_hash(
        blocker, critical["backend_blocker"]["raw_sha256"]
    )
    artifact = json.loads(blocker_path.read_text())
    blocker_raw = torch.load(
        blocker_raw_path, map_location="cpu", weights_only=False
    )
    tensor_keys = (
        "finite_response",
        "stored_finite_response",
        "stored_primary_tangent",
        "primary_tangent",
        "fallback_tangent",
    )
    if (
        any(not torch.isfinite(blocker_raw[key]).all() for key in tensor_keys)
        or not torch.equal(
            blocker_raw["finite_response"], blocker_raw["stored_finite_response"]
        )
        or not torch.equal(
            blocker_raw["primary_tangent"], blocker_raw["stored_primary_tangent"]
        )
        or not torch.equal(
            blocker_raw["primary_tangent"], blocker_raw["fallback_tangent"]
        )
        or any(value != 0.0 for value in artifact["stored_metric_absolute_errors"].values())
    ):
        raise RuntimeError("release source backend raw replay drifted")
    failed = {key for key, value in artifact["criteria"].items() if not value}
    comparison = artifact["comparisons"][
        "primary_vs_fallback_tangent_all_slots"
    ]
    spec = critical["backend_blocker"]
    if (
        artifact["backend_parity_pass"] is not spec["expected_backend_parity_pass"]
        or artifact["stage1_mismatch_reproduced"]
        is not spec["expected_stage1_mismatch_reproduced"]
        or failed != {spec["sole_failed_criterion"]}
        or comparison["cosine"] != spec["all_slot_tangent_cosine"]
        or comparison["relative_error"]
        != spec["all_slot_tangent_relative_error"]
        or comparison["relative_error"] <= spec["frozen_relative_error_ceiling"]
        or artifact["backend_errors"]
        or artifact["finite_difference_used_as_exact"] is not False
    ):
        raise RuntimeError("release source backend blocker drifted")
    return {
        "goldens": goldens,
        "positive_control": control,
        "stage1": stage1,
        "stage1_summary": stage_summary,
        "backend_blocker": blocker,
        "backend_artifact": artifact,
    }


def _copy_release_documents(config: dict, destination: Path) -> dict:
    records = {}
    for label, relative in config["release_documents"].items():
        source = PACKAGE_ROOT / relative
        target_name = config["drive_outputs"][label]
        target = destination / target_name
        atomic_text(target, source.read_text())
        records[label] = {
            "source": _record(source),
            "drive_copy": _record(target),
            "byte_identical": source.read_bytes() == target.read_bytes(),
        }
        if not records[label]["byte_identical"]:
            raise RuntimeError(f"release document copy differs: {label}")
    return records


def main() -> None:
    git = require_clean_tree()
    config = yaml.safe_load(CONFIG.read_text())
    if (
        config["status"] != "FROZEN_METHODS_BLOCKER_RELEASE"
        or config["evidence_id"] != EVIDENCE_ID
        or config["terminal_contract"]["status"] != "COMPLETE_METHODS_BLOCKER"
        or config["terminal_contract"]["scientific_expansion_allowed"] is not False
    ):
        raise RuntimeError("blocked-release config is not frozen")
    origins = {
        row["evidence_id"]
        for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
    }
    if EVIDENCE_ID in origins:
        raise RuntimeError("Gemma state-of-record evidence already exists")

    release_root = directory("release")
    outputs = {
        label: release_root / filename
        for label, filename in config["drive_outputs"].items()
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError(
            "refusing to overwrite unregistered release outputs: "
            + json.dumps(existing)
        )

    inventory, registry_raw = _registry_inventory(config)
    sources = _validate_terminal_sources(config)
    documents = _copy_release_documents(config, release_root)
    atomic_json(outputs["inventory"], inventory)
    environment = environment_payload(require_gpu=False)
    atomic_json(outputs["environment_lock"], environment)

    blocker_comparison = sources["backend_artifact"]["comparisons"][
        "primary_vs_fallback_tangent_all_slots"
    ]
    registry_prefix = config["source_registry_prefix"]
    payload = {
        "schema_version": 1,
        "bundle_id": "jspace-gemma-transport-phase4-v1",
        "source_study": config["identity"]["study_id"],
        "source_branch": git["branch"],
        "source_commit": git["code_commit"],
        "scientific_import_boundary": config["identity"]["fork_commit"],
        "status": config["terminal_contract"]["status"],
        "phase4_import_tier": config["identity"]["phase4_import_tier"],
        "source_registry": {
            "path": str(EVENTS),
            "prefix_bytes": len(registry_raw),
            "prefix_sha256": file_sha256(EVENTS),
            "through_evidence_id": registry_prefix["through_evidence_id"],
            "note": (
                "Hash exactly prefix_bytes; the state-of-record registration "
                "event is appended after this bundle is written."
            ),
        },
        "critical_evidence": {
            label: {
                "evidence_id": event["evidence_id"],
                "code_commit": event["code_commit"],
                "outputs": event["outputs"],
            }
            for label, event in (
                ("goldens", sources["goldens"]),
                ("positive_control", sources["positive_control"]),
                ("gemma_stage1", sources["stage1"]),
                ("backend_blocker", sources["backend_blocker"]),
            )
        },
        "methods_result": {
            "backend_parity_pass": False,
            "sole_failed_criterion": config["critical_evidence"][
                "backend_blocker"
            ]["sole_failed_criterion"],
            "selected_slot_backend_tangents_bit_identical": True,
            "stage1_mismatch_reproduced": True,
            "all_slot_tangent_cosine": blocker_comparison["cosine"],
            "all_slot_tangent_relative_error": blocker_comparison[
                "relative_error"
            ],
            "frozen_relative_error_ceiling": config["critical_evidence"][
                "backend_blocker"
            ]["frozen_relative_error_ceiling"],
            "mechanism_interpretation_allowed": False,
        },
        "documents": documents,
        "inventory": _record(outputs["inventory"]),
        "environment_lock": _record(outputs["environment_lock"]),
        "terminal_contract": config["terminal_contract"],
        "claim_boundary": config["claim_boundary"],
        "phase4_model_cell_opened": False,
        "interventions_opened": False,
        "independent_review_or_pi_signoff": False,
        "reproduction": {
            "config": _record(CONFIG),
            "verify_command": "python -m jspace_gemma verify",
            "test_command": "python -m pytest -q interpretability/jspaces/sidelines/gemma/tests",
        },
    }
    envelope = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
        "provenance": {
            "study_id": config["identity"]["study_id"],
            "evidence_id": EVIDENCE_ID,
            "tier": "methods",
            "branch": git["branch"],
            "code_commit": git["code_commit"],
            "dirty_tree": False,
            "command": "python -m jspace_gemma.experiments.gm_blocked_release",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": None,
            "target_model_opened": False,
            "scientific_expansion": False,
        },
    }
    json.dumps(envelope, allow_nan=False, sort_keys=True)
    atomic_json(outputs["import_json"], envelope)

    manifest = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "tier": "methods",
        "status": config["terminal_contract"]["status"],
        "producer_commit": git["code_commit"],
        "config": _record(CONFIG),
        "registry_prefix": inventory["registry"],
        "verified_live_events": inventory["n_live_events"],
        "verified_outputs": inventory["n_verified_outputs"],
        "verification_failures": inventory["failures"],
        "release_files": {
            **{label: value["drive_copy"] for label, value in documents.items()},
            "import_json": _record(outputs["import_json"]),
            "inventory": _record(outputs["inventory"]),
            "environment_lock": _record(outputs["environment_lock"]),
        },
        "source_reports": {
            "development_report": _record(
                PACKAGE_ROOT / "reports/GEMMA_TRANSPORT_DEVELOPMENT_REPORT.md"
            ),
            "development_tex": _record(
                PACKAGE_ROOT / "reports/handout/gemma_transport_development.tex"
            ),
            "restart_bootstrap": _record(
                PACKAGE_ROOT / "reports/RESUME_GEMMA_TRANSPORT.md"
            ),
        },
        "terminal_contract": config["terminal_contract"],
        "claim_boundary": config["claim_boundary"],
        "target_model_opened_during_release": False,
    }
    json.dumps(manifest, allow_nan=False, sort_keys=True)
    atomic_json(outputs["release_manifest"], manifest)

    registered_outputs = [outputs[label] for label in config["drive_outputs"]]
    create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "terminal Gemma transport methods-blocker state of record, "
            "claim ledger, gate protocol, verified inventory, and Phase-4 "
            "methods-only import bundle"
        ),
        command="python -m jspace_gemma.experiments.gm_blocked_release",
        outputs=registered_outputs,
        inputs={
            "config_sha256": file_sha256(CONFIG),
            "registry_prefix_sha256": inventory["registry"]["prefix_sha256"],
            "backend_blocker_artifact_sha256": config["critical_evidence"][
                "backend_blocker"
            ]["artifact_sha256"],
            "stage1_summary_sha256": config["critical_evidence"]["gemma_stage1"][
                "summary_sha256"
            ],
        },
        terminal_status=config["terminal_contract"]["status"],
        methods_blocker=True,
        scientific_expansion=False,
        mechanism_interpretation_allowed=False,
        phase4_import_tier="methods_only",
        target_model_opened=False,
        independent_review_or_pi_signoff=False,
    )
    print(
        json.dumps(
            {
                "status": config["terminal_contract"]["status"],
                "manifest": _record(outputs["release_manifest"]),
                "import_bundle": _record(outputs["import_json"]),
                "payload_sha256": envelope["payload_sha256"],
                "verified_live_events": inventory["n_live_events"],
                "verified_outputs": inventory["n_verified_outputs"],
                "target_model_opened": False,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
