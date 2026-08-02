"""Validate isolated development side-track bundles before Phase 4 import."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Callable, Mapping, Sequence

from .manifests import atomic_json, file_sha256, object_sha256
from .registry4 import EVENTS, read_events


REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE_NAMESPACES = {
    "jspace-olmo-lineage": "ol-",
    "jspace-gemma-transport": "gm-",
}
ALLOWED_SOURCE_TIERS = {"development", "methods"}


class ImportBundleError(RuntimeError):
    pass


def _canonical_event_sha256(event: Mapping) -> str:
    return object_sha256(dict(event))


def git_commit_reachable(commit: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
    ).returncode == 0


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        [
            "git", "-C", str(REPOSITORY), "merge-base", "--is-ancestor",
            ancestor, descendant,
        ],
        capture_output=True,
    ).returncode == 0


def _resolve_source_event(events: Sequence[Mapping], evidence_id: str) -> dict:
    origins = [
        dict(row) for row in events
        if row.get("evidence_id") == evidence_id
        and row.get("event") in {"evidence_created", "evidence_imported"}
    ]
    if len(origins) != 1:
        raise ImportBundleError(
            f"expected one source origin for {evidence_id!r}, "
            f"found {len(origins)}")
    statuses = [
        dict(row) for row in events
        if row.get("evidence_id") == evidence_id
        and row.get("event") not in {"evidence_created", "evidence_imported"}
    ]
    if any(row.get("event") == "evidence_withdrawn" for row in statuses):
        raise ImportBundleError(f"source event is withdrawn: {evidence_id}")
    replacement = next((
        row.get("superseded_by") for row in reversed(statuses)
        if row.get("event") == "evidence_superseded"), None)
    if replacement:
        raise ImportBundleError(
            f"source event {evidence_id} is superseded by {replacement}")
    effective = dict(origins[0])
    for row in statuses:
        if row.get("event") == "evidence_corrected":
            effective.update(row.get("corrected_fields", {}))
    effective["status_events"] = statuses
    return effective


def _verify_output(output: Mapping) -> dict:
    path = Path(str(output.get("path", "")))
    expected = str(output.get("sha256", ""))
    if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected):
        raise ImportBundleError(f"source output lacks a full SHA-256: {path}")
    if not path.is_file():
        raise ImportBundleError(f"source output is absent: {path}")
    actual_bytes = int(path.stat().st_size)
    expected_bytes = output.get("bytes")
    if expected_bytes is not None and actual_bytes != int(expected_bytes):
        raise ImportBundleError(
            f"source output byte-count drift: {path}; expected "
            f"{expected_bytes}, got {actual_bytes}")
    actual = file_sha256(path)
    if actual != expected:
        raise ImportBundleError(
            f"source output hash drift: {path}; expected {expected}, "
            f"got {actual}")
    return {
        "path": str(path),
        "sha256": actual,
        "bytes": actual_bytes,
    }


def _validate_governance(value: Mapping) -> None:
    expected = {
        "development_or_methods_only": True,
        "contains_confirmatory_intervention_outcomes": False,
        "contains_replication_intervention_outcomes": False,
        "mainline_registry_was_not_written_by_side_track": True,
    }
    if dict(value) != expected:
        raise ImportBundleError("side-bundle governance contract drift")


def validate_import_bundle(
        bundle_path: str | Path, *, main_events_path: str | Path = EVENTS,
        commit_reachable: Callable[[str], bool] = git_commit_reachable,
        is_ancestor: Callable[[str, str], bool] = git_is_ancestor,
        allow_existing_target: bool = False) -> dict:
    """Validate a bundle and every selected live source output."""
    path = Path(bundle_path)
    envelope = json.loads(path.read_text())
    if envelope.get("schema_version") != 1 or "payload" not in envelope:
        raise ImportBundleError("invalid side-bundle envelope")
    payload = envelope["payload"]
    actual_payload_sha = object_sha256(payload)
    if envelope.get("payload_sha256") != actual_payload_sha:
        raise ImportBundleError("side-bundle payload hash mismatch")
    if payload.get("schema_version") != 1:
        raise ImportBundleError("unsupported side-bundle payload schema")

    source_study = str(payload.get("source_study", ""))
    expected_prefix = SOURCE_NAMESPACES.get(source_study)
    if expected_prefix is None:
        raise ImportBundleError(f"unsupported side study: {source_study!r}")
    if payload.get("evidence_id_prefix") != expected_prefix:
        raise ImportBundleError("side-bundle evidence namespace drift")
    source_commit = str(payload.get("source_commit", ""))
    if len(source_commit) != 40 or not commit_reachable(source_commit):
        raise ImportBundleError(
            f"source bundle commit is not reachable: {source_commit}")
    _validate_governance(payload.get("governance", {}))

    registry_spec = payload.get("source_registry", {})
    registry_path = Path(str(registry_spec.get("path", "")))
    expected_registry_sha = str(registry_spec.get("sha256", ""))
    if not registry_path.is_file():
        raise ImportBundleError(f"source registry is absent: {registry_path}")
    actual_registry_sha = file_sha256(registry_path)
    if actual_registry_sha != expected_registry_sha:
        raise ImportBundleError(
            "source registry hash does not match the bundle")
    source_events = read_events(registry_path)
    if any(row.get("study_id") != source_study for row in source_events):
        raise ImportBundleError("source registry contains a foreign study ID")

    selections = list(payload.get("selected_events", []))
    evidence_ids = [str(row.get("evidence_id", "")) for row in selections]
    if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
        raise ImportBundleError(
            "side bundle requires a nonempty unique selected-event list")
    selected = []
    all_outputs = []
    seen_output_identity = set()
    for specification in selections:
        evidence_id = str(specification.get("evidence_id", ""))
        if not evidence_id.startswith(expected_prefix):
            raise ImportBundleError(
                f"foreign evidence namespace in bundle: {evidence_id}")
        if specification.get("contains_untouched_intervention_outcome") \
                is not False:
            raise ImportBundleError(
                f"selected event lacks a no-untouched-outcome assertion: "
                f"{evidence_id}")
        role = str(specification.get("role", ""))
        if not role:
            raise ImportBundleError(f"selected event lacks a role: {evidence_id}")
        event = _resolve_source_event(source_events, evidence_id)
        tier = event.get("tier")
        if tier not in ALLOWED_SOURCE_TIERS:
            raise ImportBundleError(
                f"forbidden source tier {tier!r} on {evidence_id}")
        event_commit = str(
            event.get("code_commit") or event.get("import_code_commit") or "")
        if (
            len(event_commit) != 40
            or not commit_reachable(event_commit)
            or not is_ancestor(event_commit, source_commit)
        ):
            raise ImportBundleError(
                f"source event commit is not in bundle ancestry: "
                f"{evidence_id} at {event_commit}")
        output_field = (
            "source_outputs"
            if event.get("event") == "evidence_imported" else "outputs")
        outputs = [_verify_output(row) for row in event.get(output_field, [])]
        if not outputs:
            raise ImportBundleError(
                f"selected source event has no outputs: {evidence_id}")
        for output in outputs:
            identity = (output["path"], output["sha256"])
            if identity not in seen_output_identity:
                all_outputs.append(output)
                seen_output_identity.add(identity)
        selected.append({
            "evidence_id": evidence_id,
            "role": role,
            "tier": tier,
            "event_type": event["event"],
            "event_commit": event_commit,
            "effective_event_sha256": _canonical_event_sha256(event),
            "outputs": outputs,
        })

    target = payload.get("target", {})
    if target.get("study_id") != "jspace-phase4":
        raise ImportBundleError("side bundle targets a foreign study")
    target_id = str(target.get("import_evidence_id", ""))
    if not target_id.startswith("p4-import-"):
        raise ImportBundleError("invalid Phase 4 import evidence ID")
    main_origins = {
        row["evidence_id"] for row in read_events(main_events_path)
        if row.get("event") in {"evidence_created", "evidence_imported"}
    }
    if target_id in main_origins and not allow_existing_target:
        raise ImportBundleError(
            f"Phase 4 import evidence ID already exists: {target_id}")

    return {
        "schema_version": 1,
        "ok": True,
        "bundle_path": str(path),
        "bundle_sha256": file_sha256(path),
        "bundle_payload_sha256": actual_payload_sha,
        "bundle_id": payload.get("bundle_id"),
        "source_study": source_study,
        "source_branch": payload.get("source_branch"),
        "source_commit": source_commit,
        "source_registry": {
            "path": str(registry_path),
            "sha256": actual_registry_sha,
        },
        "target_import_evidence_id": target_id,
        "selected_events": selected,
        "selected_event_ids_sha256": object_sha256(evidence_ids),
        "outputs": all_outputs,
        "output_inventory_sha256": object_sha256(all_outputs),
        "governance": dict(payload["governance"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--main-registry", default=str(EVENTS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-existing-target", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = validate_import_bundle(
        arguments.bundle,
        main_events_path=arguments.main_registry,
        allow_existing_target=arguments.allow_existing_target,
    )
    atomic_json(Path(arguments.output), result)
    print(json.dumps({
        "ok": result["ok"],
        "bundle_id": result["bundle_id"],
        "source_study": result["source_study"],
        "source_commit": result["source_commit"],
        "target_import_evidence_id": result["target_import_evidence_id"],
        "n_events": len(result["selected_events"]),
        "n_outputs": len(result["outputs"]),
    }, indent=1))


if __name__ == "__main__":
    main()
