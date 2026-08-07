"""Read-only, hash-pinned imports from the historical Part-2 registry."""
from __future__ import annotations

import json
from pathlib import Path

from .manifests import file_sha256


class ImportError(RuntimeError):
    pass


def read_registry(path: str | Path) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def resolve_source_event(path: str | Path, evidence_id: str) -> dict:
    rows = read_registry(path)
    origins = [
        row for row in rows
        if row.get("evidence_id") == evidence_id
        and row.get("event") in {"evidence_created", "evidence_imported"}
    ]
    if len(origins) != 1:
        raise ImportError(
            f"expected one source origin for {evidence_id!r}, found {len(origins)}"
        )
    status = [
        row for row in rows
        if row.get("evidence_id") == evidence_id
        and row.get("event") not in {"evidence_created", "evidence_imported"}
    ]
    if any(row.get("event") == "evidence_withdrawn" for row in status):
        raise ImportError(f"source evidence {evidence_id!r} is withdrawn")
    replacement = next(
        (
            row.get("superseded_by") for row in reversed(status)
            if row.get("event") == "evidence_superseded"
        ),
        None,
    )
    if replacement:
        raise ImportError(
            f"source evidence {evidence_id!r} is superseded by {replacement!r}"
        )
    return origins[0]


def verify_source_event(path: str | Path, evidence_id: str) -> dict:
    event = resolve_source_event(path, evidence_id)
    checked = []
    failures = []
    field = "source_outputs" if event["event"] == "evidence_imported" else "outputs"
    for row in event.get(field, []) or []:
        artifact = Path(row["path"])
        actual = file_sha256(artifact) if artifact.exists() else None
        expected = row.get("sha256")
        check = {
            "path": str(artifact),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "ok": actual == expected,
        }
        checked.append(check)
        if not check["ok"]:
            failures.append(check)
    return {
        "source_evidence_id": evidence_id,
        "source_commit": event.get("code_commit") or event.get("source_commit"),
        "source_event": event,
        "source_outputs": checked,
        "ok": not failures,
        "failures": failures,
    }
