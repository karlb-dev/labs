"""Append-only event-sourced registry for the official-repro study.

Event vocabulary (plan §4.4): ``evidence_created``, ``evidence_superseded``,
``evidence_reproduced``, ``release_created``. A supersession row never
replaces creation metadata; gated cells are registered as states, never as
zero effects.
"""
from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path
from typing import Iterable

from . import EVIDENCE_PREFIX, STUDY_ID
from .manifests import canonical_json, file_sha256, git_info
from .paths import EVENTS

SCHEMA_VERSION = 1

VALID_EVENTS = {
    "evidence_created",
    "evidence_superseded",
    "evidence_reproduced",
    "release_created",
}

#: Study 1 is development/methods tier throughout (plan §0, §13.5).
VALID_TIERS = {"development", "methods"}


class RegistryError(RuntimeError):
    pass


def read_events(path: str | Path = EVENTS) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        __import__("json").loads(line)
        for line in source.read_text().splitlines()
        if line.strip()
    ]


def _validate(event: dict) -> None:
    kind = event.get("event")
    evidence_id = event.get("evidence_id")
    if kind not in VALID_EVENTS:
        raise RegistryError(f"invalid event type {kind!r}")
    if not evidence_id or not str(evidence_id).startswith(EVIDENCE_PREFIX):
        raise RegistryError(
            f"evidence_id {evidence_id!r} must carry the {EVIDENCE_PREFIX!r} prefix"
        )
    if kind in {"evidence_created", "release_created"}:
        if event.get("tier") not in VALID_TIERS:
            raise RegistryError("creation requires tier development|methods")
        for key in ("what", "command", "code_commit"):
            if not event.get(key):
                raise RegistryError(f"creation event lacks {key}")
    if kind == "evidence_superseded" and not event.get("reason"):
        raise RegistryError("supersession requires a reason")
    if kind == "evidence_reproduced" and not event.get("outputs"):
        raise RegistryError("reproduction event requires rehashed outputs")


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
                row["evidence_id"]
                for row in events
                if row["event"] in {"evidence_created", "release_created"}
            }
            if stamped["event"] in {"evidence_created", "release_created"}:
                if stamped["evidence_id"] in known:
                    raise RegistryError(
                        f"duplicate evidence ID {stamped['evidence_id']!r}"
                    )
            elif stamped["evidence_id"] not in known:
                raise RegistryError(
                    f"status event references unknown evidence "
                    f"{stamped['evidence_id']!r}"
                )
            if stamped["event"] == "evidence_superseded":
                replacement = stamped.get("superseded_by")
                if replacement not in known:
                    raise RegistryError(
                        f"unknown replacement evidence {replacement!r}"
                    )
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
    event_kind: str = "evidence_created",
    **extra,
) -> dict:
    information = git_info()
    if information["dirty_tree"]:
        raise RegistryError("refusing evidence creation from dirty tree")
    output_rows = [
        {"path": str(path), "sha256": file_sha256(path)} for path in outputs
    ]
    return append_event(
        {
            "event": event_kind,
            "evidence_id": evidence_id,
            "tier": tier,
            "what": what,
            "command": command,
            "code_commit": information["code_commit"],
            "branch": information["branch"],
            "outputs": output_rows,
            "inputs": inputs or {},
            **extra,
        }
    )


def supersede(evidence_id: str, replacement: str, *, reason: str) -> dict:
    return append_event(
        {
            "event": "evidence_superseded",
            "evidence_id": evidence_id,
            "superseded_by": replacement,
            "reason": reason,
        }
    )


def resolve(evidence_id: str, *, path: str | Path = EVENTS) -> dict:
    rows = read_events(path)
    creations = [
        row
        for row in rows
        if row["evidence_id"] == evidence_id
        and row["event"] in {"evidence_created", "release_created"}
    ]
    if len(creations) != 1:
        raise RegistryError(
            f"expected one origin event for {evidence_id!r}, found {len(creations)}"
        )
    record = dict(creations[0])
    status = [
        row
        for row in rows
        if row["evidence_id"] == evidence_id and row is not creations[0]
        and row["event"] != "evidence_created"
        and row["event"] != "release_created"
    ]
    record["status_events"] = status
    record["superseded_by"] = next(
        (
            row["superseded_by"]
            for row in reversed(status)
            if row["event"] == "evidence_superseded"
        ),
        None,
    )
    record["live"] = record["superseded_by"] is None
    return record


def live_events(*, path: str | Path = EVENTS) -> list[dict]:
    identifiers = [
        row["evidence_id"]
        for row in read_events(path)
        if row["event"] in {"evidence_created", "release_created"}
    ]
    return [
        record
        for identifier in identifiers
        if (record := resolve(identifier, path=path))["live"]
    ]


def verify_outputs(*, path: str | Path = EVENTS) -> dict:
    """Rehash every live event's outputs; missing or drifted files fail."""
    failures = []
    checked = 0
    for record in live_events(path=path):
        for output in record.get("outputs", []):
            checked += 1
            target = Path(output["path"])
            if not target.is_absolute():
                from .paths import REPO_ROOT

                target = REPO_ROOT / target
            if not target.exists():
                failures.append({"evidence_id": record["evidence_id"],
                                 "path": output["path"], "why": "missing"})
            elif file_sha256(target) != output["sha256"]:
                failures.append({"evidence_id": record["evidence_id"],
                                 "path": output["path"], "why": "hash-drift"})
    return {"ok": not failures, "checked": checked, "failures": failures}
