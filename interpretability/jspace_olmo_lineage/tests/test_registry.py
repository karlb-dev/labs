import json

import pytest

from jspace_olmo_lineage.registry import (
    RegistryError,
    append_event,
    read_events,
    resolve,
)


def creation(identifier="ol-test-v1"):
    return {
        "event": "evidence_created",
        "evidence_id": identifier,
        "tier": "methods",
        "what": "synthetic test",
        "command": "pytest",
        "code_commit": "a" * 40,
    }


def test_registry_requires_ol_prefix(tmp_path):
    with pytest.raises(RegistryError):
        append_event(creation("p4-test-v1"), path=tmp_path / "events.jsonl")


def test_registry_rejects_duplicate_origin(tmp_path):
    path = tmp_path / "events.jsonl"
    append_event(creation(), path=path)
    with pytest.raises(RegistryError):
        append_event(creation(), path=path)


def test_registry_is_append_only_and_resolves_correction(tmp_path):
    path = tmp_path / "events.jsonl"
    append_event(creation(), path=path)
    append_event({
        "event": "evidence_corrected",
        "evidence_id": "ol-test-v1",
        "reason": "metadata typo",
        "corrected_fields": {"what": "corrected synthetic test"},
    }, path=path)
    rows = read_events(path)
    assert len(rows) == 2
    assert len(path.read_text().splitlines()) == 2
    assert resolve("ol-test-v1", path=path)["effective_metadata"]["what"] == (
        "corrected synthetic test")
    assert all(json.loads(line) for line in path.read_text().splitlines())
