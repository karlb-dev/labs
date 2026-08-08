"""Append-only evidence registry (plan §7 schema, jspaces mechanics).

One JSONL file; one canonical-JSON object per line; flock + fsync appends;
supersede/correct/withdraw are new events *about* an event, never edits.
The three/four seed rows written at P1-0 (before this module existed) lack
the ``event`` kind field and are read as origin rows — they are never
rewritten (append-only from birth).
"""

from __future__ import annotations

import fcntl
import os
import pathlib
from typing import Any, Mapping

from . import SCIENTIFIC_TIERS, STUDY_ID
from .canonical import canonical_json
from .provenance import git_info, utc_now
from . import paths


class RegistryError(RuntimeError):
    pass


def _registry_path(path: pathlib.Path | None = None) -> pathlib.Path:
    return pathlib.Path(path) if path else paths.registry_path()


def read_events(path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    p = _registry_path(path)
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return rows
    import json

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _origin_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Origin events by id. Seed rows (no ``event`` key) count as origins."""
    origins: dict[str, dict[str, Any]] = {}
    for row in rows:
        kind = row.get("event")
        if kind in (None, "evidence_created", "evidence_imported"):
            eid = row.get("event_id") or row.get("evidence_id")
            if not eid:
                continue
            if eid in origins:
                raise RegistryError(f"duplicate origin event id: {eid}")
            origins[eid] = row
    return origins


def _append(record: Mapping[str, Any], path: pathlib.Path | None = None) -> None:
    p = _registry_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(dict(record)) + "\n"
    with p.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def register(
    *,
    event_id: str,
    event_type: str,
    scientific_tier: str,
    claim_summary: str,
    status: str = "complete",
    parent_event_ids: list[str] | None = None,
    config_hash: str | None = None,
    source_manifest_hash: str | None = None,
    model_manifest_hash: str | None = None,
    input_artifacts: list[dict[str, Any]] | None = None,
    output_artifacts: list[dict[str, Any]] | None = None,
    row_count: int | None = None,
    limitations: str | None = None,
    supersedes: str | None = None,
    allow_dirty: bool = False,
    registry_file: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Append an origin (evidence-created) event.

    Refuses dirty git trees unless ``allow_dirty`` (development only; the
    event is stamped so a dirty-tree event can never silently become
    frozen-tier evidence).
    """
    if scientific_tier not in SCIENTIFIC_TIERS:
        raise RegistryError(
            f"scientific_tier {scientific_tier!r} not in {sorted(SCIENTIFIC_TIERS)}"
        )
    if not event_id.startswith("pref2-"):
        raise RegistryError(f"event_id must start with 'pref2-': {event_id}")
    rows = read_events(registry_file)
    if event_id in _origin_rows(rows):
        raise RegistryError(f"event id already registered: {event_id}")
    git = git_info()
    # The registry's own file is exempt: appending events IS the registry's
    # operation, and multi-event boundaries would otherwise self-block.
    try:
        registry_rel = str(_registry_path(registry_file).resolve().relative_to(
            paths.repo_root()))
    except ValueError:          # registry outside the repo (tests)
        registry_rel = None
    real_dirty = [p for p in git["dirty_paths"] if p != registry_rel]
    if real_dirty and not allow_dirty:
        raise RegistryError(
            "refusing to register evidence from a dirty git tree "
            f"(dirty: {real_dirty[:5]}...); commit first or pass "
            "allow_dirty=True for development-tier plumbing"
        )
    if real_dirty and scientific_tier in ("frozen_behavioral", "conditional_causal"):
        raise RegistryError("frozen/conditional tiers never register from a dirty tree")
    record = {
        "event": "evidence_created",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "event_id": event_id,
        "event_type": event_type,
        "status": status,
        "created_utc": utc_now(),
        "code_commit": git["commit"],
        "code_branch": git["branch"],
        "dirty_tree": git["dirty_tree"],
        "parent_event_ids": parent_event_ids or [],
        "config_hash": config_hash,
        "source_manifest_hash": source_manifest_hash,
        "model_manifest_hash": model_manifest_hash,
        "input_artifacts": input_artifacts or [],
        "output_artifacts": output_artifacts or [],
        "row_count": row_count,
        "scientific_tier": scientific_tier,
        "claim_summary": claim_summary,
        "limitations": limitations,
        "supersedes": supersedes,
    }
    _append(record, registry_file)
    if supersedes:
        supersede(supersedes, superseded_by=event_id,
                  reason=f"superseded by {event_id}", registry_file=registry_file)
    return record


def supersede(event_id: str, *, superseded_by: str, reason: str,
              registry_file: pathlib.Path | None = None) -> None:
    rows = read_events(registry_file)
    origins = _origin_rows(rows)
    if event_id not in origins:
        raise RegistryError(f"cannot supersede unknown event: {event_id}")
    if superseded_by not in origins:
        raise RegistryError(f"superseding event must exist first: {superseded_by}")
    _append({
        "event": "evidence_superseded",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "event_id": event_id,
        "superseded_by": superseded_by,
        "reason": reason,
        "event_utc": utc_now(),
    }, registry_file)


def correct(event_id: str, *, corrected_fields: Mapping[str, Any], reason: str,
            registry_file: pathlib.Path | None = None) -> None:
    rows = read_events(registry_file)
    if event_id not in _origin_rows(rows):
        raise RegistryError(f"cannot correct unknown event: {event_id}")
    _append({
        "event": "evidence_corrected",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "event_id": event_id,
        "corrected_fields": dict(corrected_fields),
        "reason": reason,
        "event_utc": utc_now(),
    }, registry_file)


def withdraw(event_id: str, *, reason: str,
             registry_file: pathlib.Path | None = None) -> None:
    rows = read_events(registry_file)
    if event_id not in _origin_rows(rows):
        raise RegistryError(f"cannot withdraw unknown event: {event_id}")
    _append({
        "event": "evidence_withdrawn",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "event_id": event_id,
        "reason": reason,
        "event_utc": utc_now(),
    }, registry_file)


def resolve(event_id: str, registry_file: pathlib.Path | None = None) -> dict[str, Any]:
    """Origin row + folded corrections + live status. Never last-row-wins."""
    rows = read_events(registry_file)
    origins = _origin_rows(rows)
    if event_id not in origins:
        raise RegistryError(f"unknown event: {event_id}")
    effective = dict(origins[event_id])
    status_events = []
    superseded_by = None
    withdrawn = False
    for row in rows:
        if row.get("event_id") != event_id:
            continue
        kind = row.get("event")
        if kind == "evidence_corrected":
            effective.update(row.get("corrected_fields", {}))
            status_events.append(row)
        elif kind == "evidence_superseded":
            superseded_by = row.get("superseded_by")
            status_events.append(row)
        elif kind == "evidence_withdrawn":
            withdrawn = True
            status_events.append(row)
    effective["status_events"] = status_events
    effective["superseded_by"] = superseded_by
    effective["withdrawn"] = withdrawn
    effective["live"] = superseded_by is None and not withdrawn
    return effective


def live_events(registry_file: pathlib.Path | None = None) -> list[dict[str, Any]]:
    rows = read_events(registry_file)
    return [resolve(eid, registry_file) for eid in _origin_rows(rows)]
