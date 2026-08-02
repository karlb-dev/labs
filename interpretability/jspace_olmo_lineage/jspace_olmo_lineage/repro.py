"""Verify live side-track evidence and namespace isolation."""
from __future__ import annotations

import json
from pathlib import Path

from .manifests import file_sha256, object_sha256
from .paths import run_root
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


def verify_live_evidence() -> dict:
    failures = []
    checked = 0
    root = run_root(create=False) if EVENTS.exists() else None
    for event in resolve_all():
        if not event["live"]:
            continue
        if not event["evidence_id"].startswith("ol-"):
            failures.append({
                "evidence_id": event["evidence_id"],
                "reason": "foreign evidence prefix",
            })
        for output in event.get("outputs", []) or []:
            path = Path(output["path"])
            actual = file_sha256(path) if path.exists() else None
            checked += 1
            if root is not None:
                try:
                    path.resolve().relative_to(root.resolve())
                except ValueError:
                    failures.append({
                        "evidence_id": event["evidence_id"],
                        "path": str(path),
                        "reason": "native output escapes OLMo run root",
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
