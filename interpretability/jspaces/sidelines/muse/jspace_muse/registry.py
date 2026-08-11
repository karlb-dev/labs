"""Append-only evidence registry (development/methods tier)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import DRIVE_ROOT, EVENTS, REPO_ROOT
from .util import append_jsonl, git_head, sha256_file, utc_now


def register(event: dict[str, Any]) -> None:
    """Append one evidence event to repo + Drive mirrors."""
    row = {
        "utc": utc_now(),
        "code_commit": git_head(REPO_ROOT),
        "tier": event.get("tier", "development"),
        **event,
    }
    # Hash outputs if paths provided
    outputs = []
    for item in event.get("outputs", []):
        if isinstance(item, (str, Path)):
            p = Path(item)
            outputs.append({"path": str(p), "sha256": sha256_file(p) if p.exists() else None})
        else:
            outputs.append(item)
    row["outputs"] = outputs
    append_jsonl(EVENTS, row)
    drive_events = DRIVE_ROOT / "manifests" / "evidence_events.jsonl"
    append_jsonl(drive_events, row)
