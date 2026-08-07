"""Append-only evidence registry isolated to the Gemma side track."""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Iterable

from .manifests import canonical_json, file_sha256, git_info

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EVENTS = PACKAGE_ROOT / "reports/evidence_events.jsonl"
STUDY_ID = "jspace-gemma-transport"
SCHEMA_VERSION = 1
VALID_EVENTS = {
    "evidence_created",
    "evidence_imported",
    "evidence_superseded",
    "evidence_withdrawn",
    "evidence_corrected",
    "evidence_reproduced",
}
NATIVE_TIERS = {"development", "methods"}
IMPORT_TIERS = {"historical-development-import", "historical-methods-import"}
VALID_TIERS = NATIVE_TIERS | IMPORT_TIERS


class RegistryError(RuntimeError):
    pass


def read_events(path: str | Path = EVENTS) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


def _validate(event: dict) -> None:
    if event.get("event") not in VALID_EVENTS:
        raise RegistryError(f"invalid event type {event.get('event')!r}")
    evidence_id = event.get("evidence_id")
    if not evidence_id or not evidence_id.startswith(("gm-", "gm2-")):
        raise RegistryError(
            "Gemma evidence IDs must use the gm- prefix or gm2- prefix"
        )
    if event["event"] == "evidence_created":
        if event.get("tier") not in NATIVE_TIERS:
            raise RegistryError("native evidence must be development or methods tier")
        for key in ("what", "command", "code_commit"):
            if not event.get(key):
                raise RegistryError(f"creation event lacks {key}")
    elif event["event"] == "evidence_imported":
        if event.get("tier") not in IMPORT_TIERS:
            raise RegistryError("import event requires a historical import tier")
        for key in (
            "what", "source_study", "source_evidence_id", "source_commit",
            "source_registry_sha256", "source_outputs", "import_code_commit",
        ):
            if not event.get(key):
                raise RegistryError(f"import event lacks {key}")
    elif event["event"] == "evidence_withdrawn" and not event.get("reason"):
        raise RegistryError("withdrawal requires a reason")
    elif event["event"] == "evidence_corrected":
        if not event.get("reason") or not event.get("corrected_fields"):
            raise RegistryError("correction requires reason and corrected_fields")


def append_event(event: dict, *, path: str | Path = EVENTS) -> dict:
    destination = Path(path)
    stamped = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "event_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
                if row["event"] in {"evidence_created", "evidence_imported"}
            }
            if stamped["event"] in {"evidence_created", "evidence_imported"}:
                if stamped["evidence_id"] in known:
                    raise RegistryError(f"duplicate evidence ID {stamped['evidence_id']!r}")
            elif stamped["evidence_id"] not in known:
                raise RegistryError("status event references unknown evidence")
            if stamped["event"] == "evidence_superseded":
                if stamped.get("superseded_by") not in known:
                    raise RegistryError("supersession references unknown replacement")
            handle.write(canonical_json(stamped) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    return stamped


def create(
    evidence_id: str,
    *,
    tier: str,
    what: str,
    command: str,
    outputs: Iterable[str | Path] = (),
    inputs: dict | None = None,
    **extra,
) -> dict:
    information = git_info()
    if information["dirty_tree"]:
        raise RegistryError("refusing evidence creation from dirty tree")
    rows = [
        {"path": str(path), "sha256": file_sha256(path)} for path in outputs
    ]
    return append_event(
        {
            "event": "evidence_created",
            "evidence_id": evidence_id,
            "tier": tier,
            "what": what,
            "command": command,
            "code_commit": information["code_commit"],
            "outputs": rows,
            "inputs": inputs or {},
            **extra,
        }
    )


def import_evidence(
    evidence_id: str,
    *,
    tier: str,
    what: str,
    source_study: str,
    source_evidence_id: str,
    source_commit: str,
    source_registry: str | Path,
    source_outputs: Iterable[dict],
    import_code_commit: str,
    source_code_files: Iterable[dict] = (),
    **extra,
) -> dict:
    rows = [dict(row) for row in source_outputs]
    if not rows:
        raise RegistryError("historical import requires at least one source output")
    for row in rows:
        path = Path(row["path"])
        actual = file_sha256(path)
        if actual != row.get("sha256"):
            raise RegistryError(f"source output hash drift: {path}")
    return append_event(
        {
            "event": "evidence_imported",
            "evidence_id": evidence_id,
            "tier": tier,
            "what": what,
            "source_study": source_study,
            "source_evidence_id": source_evidence_id,
            "source_commit": source_commit,
            "source_registry_sha256": file_sha256(source_registry),
            "source_outputs": rows,
            "source_code_files": [dict(row) for row in source_code_files],
            "import_code_commit": import_code_commit,
            **extra,
        }
    )


def resolve(evidence_id: str, *, path: str | Path = EVENTS) -> dict:
    rows = read_events(path)
    origins = [
        row for row in rows
        if row["evidence_id"] == evidence_id
        and row["event"] in {"evidence_created", "evidence_imported"}
    ]
    if len(origins) != 1:
        raise RegistryError(
            f"expected one origin event for {evidence_id!r}, found {len(origins)}"
        )
    record = dict(origins[0])
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
    record["effective_tier"] = effective["tier"]
    record["superseded_by"] = next(
        (
            row["superseded_by"] for row in reversed(status)
            if row["event"] == "evidence_superseded"
        ),
        None,
    )
    record["withdrawn"] = any(row["event"] == "evidence_withdrawn" for row in status)
    record["live"] = record["superseded_by"] is None and not record["withdrawn"]
    return record


def resolve_all(*, path: str | Path = EVENTS) -> list[dict]:
    identifiers = [
        row["evidence_id"] for row in read_events(path)
        if row["event"] in {"evidence_created", "evidence_imported"}
    ]
    return [resolve(identifier, path=path) for identifier in identifiers]
