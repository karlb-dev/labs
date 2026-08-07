"""Verify live side-track evidence and namespace isolation."""
from __future__ import annotations

import json
from pathlib import Path

from .manifests import file_sha256, object_sha256
from .paths import (
    DEFAULT_RUN_ROOT, REPO_ROOT, STUDY2_RUN_ROOT, _rewrite_repo_relative,
)
from .registry import EVENTS, resolve_all


def verify_json_envelope(path: str | Path) -> dict:
    source = Path(path)
    value = json.loads(source.read_text())
    if "payload" not in value:
        return {"format": "plain-json", "ok": None}
    actual = object_sha256(value["payload"])
    return {
        "format": "payload-envelope",
        "ok": actual == value.get("payload_sha256"),
        "actual_payload_sha256": actual,
        "recorded_payload_sha256": value.get("payload_sha256"),
    }


def _repository_materialization(path: Path) -> Path:
    """Map a producer-worktree package output into the merged repository."""
    if path.is_absolute():
        try:
            marker = path.parts.index("interpretability")
        except ValueError:
            return path
        relative = Path(*path.parts[marker:])
    else:
        relative = path
    for spec in dict.fromkeys(
            (_rewrite_repo_relative(relative.as_posix()),
             relative.as_posix())):
        candidate = REPO_ROOT / spec
        if candidate.is_file():
            return candidate
    return path


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def verify_live_evidence() -> dict:
    failures = []
    checked = 0
    for event in resolve_all():
        if not event["live"]:
            continue
        evidence_id = event["evidence_id"]
        if evidence_id.startswith("ol2-"):
            root = STUDY2_RUN_ROOT
        elif evidence_id.startswith("ol-"):
            root = DEFAULT_RUN_ROOT
        else:
            root = None
            failures.append({
                "evidence_id": evidence_id,
                "reason": "foreign evidence prefix",
            })
        for output in event.get("outputs", []) or []:
            path = Path(output["path"])
            materialized = _repository_materialization(path)
            actual = file_sha256(materialized) if materialized.exists() else None
            checked += 1
            if root is not None:
                package_root = REPO_ROOT / "interpretability/jspaces/sidelines/olmo"
                if not (
                    _within(path, root)
                    or _within(materialized, package_root)
                ):
                    failures.append({
                        "evidence_id": event["evidence_id"],
                        "path": str(path),
                        "reason": "native output escapes OLMo isolated roots",
                    })
            if actual != output.get("sha256"):
                failures.append({
                    "evidence_id": event["evidence_id"],
                    "path": str(path),
                    "expected": output.get("sha256"),
                    "actual": actual,
                })
    return {
        "ok": not failures,
        "n_live_events": sum(event["live"] for event in resolve_all()),
        "n_checked_outputs": checked,
        "failures": failures,
    }
