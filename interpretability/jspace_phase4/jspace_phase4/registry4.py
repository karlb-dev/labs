"""Event-sourced Phase 4 registry with immutable Phase 2/3 imports."""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Iterable

from .manifests import file_sha256, git_info

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EVENTS = PACKAGE_ROOT / "reports/evidence_events.jsonl"
STUDY_ID = "jspace-phase4"
SCHEMA_VERSION = 1

IMPORT_TIERS = {
    "phase2-confirmatory-import",
    "phase3-confirmatory-import",
    "phase3-replication-import",
    "side-development-import",
}
NATIVE_TIERS = {
    "phase4-development",
    "phase4-confirmatory",
    "phase4-replication",
    "methods",
}
VALID_TIERS = IMPORT_TIERS | NATIVE_TIERS
VALID_EVENTS = {
    "evidence_created",
    "evidence_imported",
    "evidence_superseded",
    "evidence_withdrawn",
    "evidence_corrected",
}


class RegistryError(RuntimeError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_events(path: str | Path = EVENTS) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        json.loads(line) for line in source.read_text().splitlines()
        if line.strip()
    ]


def _validate(event: dict) -> None:
    event_type = event.get("event")
    evidence_id = event.get("evidence_id")
    if event_type not in VALID_EVENTS:
        raise RegistryError(f"invalid event type {event_type!r}")
    if not evidence_id:
        raise RegistryError("event lacks evidence_id")
    if event_type == "evidence_created":
        tier = event.get("tier")
        if tier not in NATIVE_TIERS:
            raise RegistryError(
                "native creation requires a Phase 4/methods tier; "
                "Phase 2/3 evidence must use an immutable import event")
        for key in ("what", "command", "code_commit"):
            if not event.get(key):
                raise RegistryError(f"creation event lacks {key}")
    elif event_type == "evidence_imported":
        if event.get("tier") not in IMPORT_TIERS:
            raise RegistryError("import event requires an import tier")
        for key in (
                "source_study", "source_evidence_id", "source_commit",
                "source_registry_sha256", "source_outputs"):
            if not event.get(key):
                raise RegistryError(f"import event lacks {key}")
    elif event_type == "evidence_corrected":
        if not event.get("reason") or not event.get("corrected_fields"):
            raise RegistryError(
                "correction requires reason and corrected_fields")
        corrected_tier = event["corrected_fields"].get("tier")
        if corrected_tier is not None \
                and corrected_tier not in VALID_TIERS:
            raise RegistryError("correction has invalid tier")
    elif event_type == "evidence_withdrawn" and not event.get("reason"):
        raise RegistryError("withdrawal requires a reason")


def append_event(event: dict, *, path: str | Path = EVENTS) -> dict:
    destination = Path(path)
    stamped = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "event_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **event,
    }
    _validate(stamped)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            events = read_events(destination)
            known = {
                row["evidence_id"] for row in events
                if row["event"] in {
                    "evidence_created", "evidence_imported"}
            }
            if stamped["event"] in {
                    "evidence_created", "evidence_imported"}:
                if stamped["evidence_id"] in known:
                    raise RegistryError(
                        f"duplicate evidence ID {stamped['evidence_id']!r}")
            elif stamped["evidence_id"] not in known:
                raise RegistryError(
                    f"status event references unknown evidence "
                    f"{stamped['evidence_id']!r}")
            if stamped["event"] == "evidence_superseded":
                replacement = stamped.get("superseded_by")
                if replacement not in known:
                    raise RegistryError(
                        f"unknown replacement evidence {replacement!r}")
            handle.write(canonical_json(stamped) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    return stamped


def create(evidence_id: str, *, tier: str, what: str, command: str,
           outputs: Iterable[str | Path] = (),
           inputs: dict | None = None, **extra) -> dict:
    information = git_info()
    if information["dirty_tree"]:
        raise RegistryError("refusing evidence creation from dirty tree")
    output_rows = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in outputs
    ]
    return append_event({
        "event": "evidence_created",
        "evidence_id": evidence_id,
        "tier": tier,
        "what": what,
        "command": command,
        "code_commit": information["code_commit"],
        "outputs": output_rows,
        "inputs": inputs or {},
        **extra,
    })


def import_evidence(
        evidence_id: str, *, tier: str, what: str,
        source_study: str, source_evidence_id: str,
        source_commit: str, source_registry: str | Path,
        source_outputs: Iterable[str | Path],
        source_tag: str | None = None, **extra) -> dict:
    information = git_info()
    if information["dirty_tree"]:
        raise RegistryError("refusing evidence import from dirty tree")
    outputs = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in source_outputs
    ]
    return append_event({
        "event": "evidence_imported",
        "evidence_id": evidence_id,
        "tier": tier,
        "what": what,
        "source_study": source_study,
        "source_evidence_id": source_evidence_id,
        "source_commit": source_commit,
        "source_tag": source_tag,
        "import_code_commit": information["code_commit"],
        "source_registry_sha256": file_sha256(source_registry),
        "source_outputs": outputs,
        **extra,
    })


def supersede(evidence_id: str, replacement: str, *, reason: str) -> dict:
    return append_event({
        "event": "evidence_superseded",
        "evidence_id": evidence_id,
        "superseded_by": replacement,
        "reason": reason,
    })


def withdraw(evidence_id: str, *, reason: str) -> dict:
    return append_event({
        "event": "evidence_withdrawn",
        "evidence_id": evidence_id,
        "reason": reason,
    })


def resolve(evidence_id: str, *, path: str | Path = EVENTS) -> dict:
    rows = read_events(path)
    creations = [
        row for row in rows
        if row["evidence_id"] == evidence_id
        and row["event"] in {"evidence_created", "evidence_imported"}
    ]
    if len(creations) != 1:
        raise RegistryError(
            f"expected one origin event for {evidence_id!r}, "
            f"found {len(creations)}")
    record = dict(creations[0])
    status = [
        row for row in rows
        if row["evidence_id"] == evidence_id
        and row["event"] not in {"evidence_created", "evidence_imported"}
    ]
    effective = dict(record)
    for row in status:
        if row["event"] == "evidence_corrected":
            effective.update(row["corrected_fields"])
    record["status_events"] = status
    record["effective_metadata"] = effective
    record["effective_tier"] = effective["tier"]
    record["superseded_by"] = next((
        row["superseded_by"] for row in reversed(status)
        if row["event"] == "evidence_superseded"), None)
    record["withdrawn"] = any(
        row["event"] == "evidence_withdrawn" for row in status)
    record["live"] = (
        record["superseded_by"] is None and not record["withdrawn"])
    return record


def resolve_all(*, path: str | Path = EVENTS) -> list[dict]:
    identifiers = [
        row["evidence_id"] for row in read_events(path)
        if row["event"] in {"evidence_created", "evidence_imported"}
    ]
    return [resolve(identifier, path=path) for identifier in identifiers]
