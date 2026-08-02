"""Immutable, hash-verified imports from the frozen campaign namespaces."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from .manifests import commit_reachable, file_sha256, object_sha256
from .paths import resolve_uri


class ImportBoundaryError(RuntimeError):
    pass


def _read_events(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def resolve_source_event(events: Sequence[Mapping], evidence_id: str) -> dict:
    origins = [
        dict(row) for row in events
        if row.get("evidence_id") == evidence_id
        and row.get("event") in {"evidence_created", "evidence_imported"}
    ]
    if len(origins) != 1:
        raise ImportBoundaryError(
            f"expected one source origin for {evidence_id!r}, "
            f"found {len(origins)}")
    statuses = [
        row for row in events
        if row.get("evidence_id") == evidence_id
        and row.get("event") not in {"evidence_created", "evidence_imported"}
    ]
    if any(row.get("event") == "evidence_withdrawn" for row in statuses):
        raise ImportBoundaryError(f"source evidence is withdrawn: {evidence_id}")
    replacement = next((
        row.get("superseded_by") for row in reversed(statuses)
        if row.get("event") == "evidence_superseded"), None)
    if replacement:
        raise ImportBoundaryError(
            f"source evidence {evidence_id} is superseded by {replacement}")
    effective = dict(origins[0])
    for row in statuses:
        if row.get("event") == "evidence_corrected":
            effective.update(row.get("corrected_fields", {}))
    effective["live"] = True
    effective["status_events"] = statuses
    return effective


def _verify_output(output: Mapping) -> dict:
    path = Path(str(output["path"]))
    if not path.is_file():
        raise ImportBoundaryError(f"imported output is absent: {path}")
    actual = file_sha256(path)
    if actual != output.get("sha256"):
        raise ImportBoundaryError(
            f"imported output hash drift: {path}; expected "
            f"{output.get('sha256')}, got {actual}")
    return {
        "path": str(path),
        "sha256": actual,
        "bytes": int(path.stat().st_size),
    }


def verify_source_registry(specification: Mapping) -> dict:
    path = resolve_uri(specification["path"])
    actual_registry_hash = file_sha256(path)
    expected_registry_hash = specification["sha256"]
    if actual_registry_hash != expected_registry_hash:
        raise ImportBoundaryError(
            f"source registry hash drift: {path}; expected "
            f"{expected_registry_hash}, got {actual_registry_hash}")
    events = _read_events(path)
    allowed_tiers = set(specification["allowed_tiers"])
    selected = []
    for evidence_id in specification["event_ids"]:
        event = resolve_source_event(events, evidence_id)
        tier = event.get("tier")
        if tier not in allowed_tiers:
            raise ImportBoundaryError(
                f"forbidden source tier {tier!r} on {evidence_id}")
        commit = event.get("code_commit") or event.get("import_code_commit")
        if not commit or not commit_reachable(commit):
            raise ImportBoundaryError(
                f"source commit is not reachable for {evidence_id}: {commit}")
        output_field = (
            "source_outputs" if event.get("event") == "evidence_imported"
            else "outputs")
        outputs = [_verify_output(row) for row in event.get(output_field, [])]
        selected.append({
            "evidence_id": evidence_id,
            "tier": tier,
            "source_commit": commit,
            "outputs": outputs,
            "inputs": event.get("inputs", {}),
        })
    return {
        "study": specification["study"],
        "path": str(path),
        "sha256": actual_registry_hash,
        "selected_events": selected,
        "selected_event_ids_sha256": object_sha256(
            [row["evidence_id"] for row in selected]),
    }


def verify_direct_artifact(specification: Mapping) -> dict:
    path = resolve_uri(specification["uri"])
    actual = file_sha256(path)
    if actual != specification["sha256"]:
        raise ImportBoundaryError(
            f"direct artifact hash drift for {specification['id']}: "
            f"expected {specification['sha256']}, got {actual}")
    return {
        "id": specification["id"],
        "logical_uri": specification["uri"],
        "path": str(path),
        "sha256": actual,
        "bytes": int(path.stat().st_size),
        "role": specification["role"],
        "read_only": True,
    }


def build_import_manifest(config: Mapping) -> dict:
    boundary = config["scientific_import_boundary"]
    if not commit_reachable(boundary):
        raise ImportBoundaryError(
            f"scientific import boundary is not reachable: {boundary}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", boundary, "HEAD"],
        capture_output=True,
    )
    if ancestor.returncode:
        raise ImportBoundaryError(
            "scientific import boundary is not an ancestor of HEAD")
    registries = [
        verify_source_registry(row) for row in config["source_registries"]
    ]
    artifacts = [
        verify_direct_artifact(row) for row in config["direct_artifacts"]
    ]
    dependencies = [
        verify_direct_artifact(row) for row in config["code_dependencies"]
    ]
    governance = [
        verify_direct_artifact(row) for row in config["governance_documents"]
    ]
    model_roles = [row["role"] for row in config["models"]]
    if model_roles != [
        "pretrained_anchor", "think_endpoint_3_0",
        "think_endpoint_3_1", "sibling_endpoint",
    ]:
        raise ImportBoundaryError("model graph roles or ordering drifted")
    return {
        "schema_version": 1,
        "scientific_import_boundary": boundary,
        "package_parent_commit": config["package_parent_commit"],
        "branch": config["branch"],
        "source_registries": registries,
        "direct_artifacts": artifacts,
        "code_dependencies": dependencies,
        "governance_documents": governance,
        "models": list(config["models"]),
        "prohibitions": list(config["prohibitions"]),
        "no_confirmatory_or_replication_intervention_outcome": True,
        "manifest_content_sha256": object_sha256({
            "registries": registries,
            "artifacts": artifacts,
            "dependencies": dependencies,
            "governance": governance,
            "models": list(config["models"]),
        }),
    }
