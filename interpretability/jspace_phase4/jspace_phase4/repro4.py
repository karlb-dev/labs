"""Phase 4 live-evidence and envelope verifier."""
from __future__ import annotations

import json
from pathlib import Path

from .manifests import file_sha256, object_sha256
from .registry4 import resolve_all


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
    for event in resolve_all():
        if not event["live"]:
            continue
        output_field = (
            "source_outputs"
            if event["event"] == "evidence_imported" else "outputs")
        for output in event.get(output_field, []) or []:
            path = Path(output["path"])
            actual = file_sha256(path) if path.exists() else None
            checked += 1
            if actual != output.get("sha256"):
                failures.append({
                    "evidence_id": event["evidence_id"],
                    "path": str(path),
                    "expected": output.get("sha256"),
                    "actual": actual,
                })
    return {
        "ok": not failures,
        "n_live_events": sum(
            event["live"] for event in resolve_all()),
        "n_checked_outputs": checked,
        "failures": failures,
    }
