import json

import pytest

from jspace_gemma.imports import ImportError, verify_source_event
from jspace_gemma.manifests import file_sha256
from jspace_gemma.registry import RegistryError, append_event, resolve
from jspace_gemma.state import StateHeader, StateStore


def test_registry_is_gm_prefixed_and_methods_only(tmp_path):
    events = tmp_path / "events.jsonl"
    with pytest.raises(RegistryError, match="gm- prefix"):
        append_event(
            {
                "event": "evidence_created",
                "evidence_id": "bad-v1",
                "tier": "methods",
                "what": "bad",
                "command": "bad",
                "code_commit": "deadbeef",
            },
            path=events,
        )
    with pytest.raises(RegistryError, match="development or methods"):
        append_event(
            {
                "event": "evidence_created",
                "evidence_id": "gm-bad-v1",
                "tier": "confirmatory",
                "what": "bad",
                "command": "bad",
                "code_commit": "deadbeef",
            },
            path=events,
        )
    append_event(
        {
            "event": "evidence_created",
            "evidence_id": "gm-good-v1",
            "tier": "methods",
            "what": "test",
            "command": "pytest",
            "code_commit": "deadbeef",
        },
        path=events,
    )
    assert resolve("gm-good-v1", path=events)["live"]


def test_state_refuses_manifest_drift_and_tampering(tmp_path):
    header = StateHeader(
        evidence_id="gm-test-v1",
        config_sha256="a" * 64,
        code_commit="b" * 40,
        model_id="org/model",
        model_revision="c" * 40,
        environment_sha256="d" * 64,
    )
    path = tmp_path / "state.json"
    store = StateStore(path, header)
    store.write({"completed_cells": ["one"]})
    assert store.load() == {"completed_cells": ["one"]}
    changed = StateStore(
        path,
        StateHeader(**{**header.__dict__, "model_revision": "e" * 40}),
    )
    with pytest.raises(RuntimeError, match="incompatible checkpoint"):
        changed.load()
    value = json.loads(path.read_text())
    value["payload"]["completed_cells"] = []
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="payload hash"):
        store.load()


def test_hash_pinned_source_import_and_supersession_refusal(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"value":1}\n')
    registry = tmp_path / "source.jsonl"
    origin = {
        "event": "evidence_created",
        "evidence_id": "source-v1",
        "code_commit": "a" * 40,
        "outputs": [{"path": str(artifact), "sha256": file_sha256(artifact)}],
    }
    registry.write_text(json.dumps(origin) + "\n")
    assert verify_source_event(registry, "source-v1")["ok"]
    registry.write_text(
        json.dumps(origin)
        + "\n"
        + json.dumps(
            {
                "event": "evidence_superseded",
                "evidence_id": "source-v1",
                "superseded_by": "source-v2",
            }
        )
        + "\n"
    )
    with pytest.raises(ImportError, match="superseded"):
        verify_source_event(registry, "source-v1")
