"""Read-only Phase 3 import boundary."""
from __future__ import annotations

import json
from pathlib import Path

from .manifests import file_sha256
from .paths4 import resolve_uri

PHASE3_COMPLETE_TAG = "jspace-phase3-complete-v1"
PHASE3_COMPLETE_COMMIT = "9e0672b8748b8c53f0bd853dfadda9bc795fd524"
PHASE3_RELEASE_MANIFEST_URI = (
    "artifact://phase3/manifests/phase3_release_manifest.json")


def phase3_release_manifest() -> dict:
    path = resolve_uri(PHASE3_RELEASE_MANIFEST_URI)
    envelope = json.loads(path.read_text())
    return {
        "source_study": "jspace-phase3",
        "source_evidence_id": "p3-release-manifest-v1",
        "source_tag": PHASE3_COMPLETE_TAG,
        "source_commit": PHASE3_COMPLETE_COMMIT,
        "source_path": str(path),
        "source_sha256": file_sha256(path),
        "payload_sha256": envelope["payload_sha256"],
    }


def immutable_phase3_artifact(relative_path: str) -> Path:
    """Resolve a Phase 3 artifact without exposing a writable adapter."""
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise ValueError("relative Phase 3 artifact path required")
    return resolve_uri(f"artifact://phase3/{relative_path}")
