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
    from .durability import resolve_output_reference

    failures = []
    checked = 0
    resolutions: dict[str, int] = {}
    for event in resolve_all():
        if not event["live"]:
            continue
        output_field = (
            "source_outputs"
            if event["event"] == "evidence_imported" else "outputs")
        for output in event.get(output_field, []) or []:
            checked += 1
            resolved = resolve_output_reference(
                str(output["path"]), output.get("sha256"), event=event)
            if resolved["status"] != "verified":
                failures.append({
                    "evidence_id": event["evidence_id"],
                    "path": str(output["path"]),
                    "expected": output.get("sha256"),
                    "actual": resolved["actual_sha256"],
                    "status": resolved["status"],
                })
            else:
                mode = str(resolved.get("resolution") or "literal-path")
                if mode.startswith("append-only-registry-prefix"):
                    mode = "append-only-registry-prefix"
                resolutions[mode] = resolutions.get(mode, 0) + 1
    return {
        "ok": not failures,
        "n_live_events": sum(
            event["live"] for event in resolve_all()),
        "n_checked_outputs": checked,
        "n_verified_by_resolution": resolutions,
        "failures": failures,
    }
