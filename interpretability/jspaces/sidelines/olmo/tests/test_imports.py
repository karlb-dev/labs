import json
from pathlib import Path

import pytest

from jspace_olmo_lineage.imports import (
    ImportBoundaryError,
    resolve_source_event,
    verify_direct_artifact,
)
from jspace_olmo_lineage.manifests import file_sha256


def origin(identifier="p4-example-v1"):
    return {
        "event": "evidence_created",
        "evidence_id": identifier,
        "tier": "phase4-development",
        "code_commit": "b" * 40,
        "outputs": [],
    }


def test_resolve_source_event_rejects_duplicates():
    with pytest.raises(ImportBoundaryError):
        resolve_source_event([origin(), origin()], "p4-example-v1")


def test_resolve_source_event_rejects_withdrawn():
    events = [origin(), {
        "event": "evidence_withdrawn",
        "evidence_id": "p4-example-v1",
        "reason": "test",
    }]
    with pytest.raises(ImportBoundaryError):
        resolve_source_event(events, "p4-example-v1")


def test_direct_artifact_hash_gate(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"ok": True}))
    row = verify_direct_artifact({
        "id": "test",
        "uri": str(path),
        "sha256": file_sha256(path),
        "role": "test",
    })
    assert row["read_only"] is True
    with pytest.raises(ImportBoundaryError):
        verify_direct_artifact({
            "id": "test",
            "uri": str(path),
            "sha256": "0" * 64,
            "role": "test",
        })
