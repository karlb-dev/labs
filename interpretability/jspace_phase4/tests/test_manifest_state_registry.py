import json

import pytest

from jspace_phase4.manifests import InputManifest, file_sha256, object_sha256
from jspace_phase4.state import StateHeader, StateStore


def _manifest():
    return InputManifest(
        experiment_id="p4-dev",
        config_sha256="a" * 64,
        model_id="org/model",
        model_revision="b" * 40,
        tokenizer_manifest_sha256="c" * 64,
        lens_sha256="d" * 64,
        bank_sha256="e" * 64,
        partition_sha256="f" * 64,
        scoring_spec_sha256="1" * 64,
        upstream={"p3": "2" * 64},
        code_commit="3" * 40,
    )


def test_input_manifest_is_canonical_and_complete():
    manifest = _manifest()
    assert manifest.sha256() == manifest.envelope()["payload_sha256"]
    assert len(manifest.sha256()) == 64
    assert object_sha256(manifest.payload()) == manifest.sha256()


def test_state_roundtrip_and_manifest_mismatch_refusal(tmp_path):
    manifest = _manifest()
    header = StateHeader(
        evidence_id="p4-dev-v1",
        input_manifest_sha256=manifest.sha256(),
        config_sha256=manifest.config_sha256,
        model_revision=manifest.model_revision,
        bank_sha256=manifest.bank_sha256,
        partition_sha256=manifest.partition_sha256,
    )
    store = StateStore(tmp_path / "state.json", header)
    assert store.load() is None
    store.write({"completed_items": ["a", "b"]})
    assert store.load() == {"completed_items": ["a", "b"]}
    incompatible = StateHeader(
        **{**header.as_dict(), "model_revision": "changed"})
    with pytest.raises(RuntimeError, match="incompatible state"):
        StateStore(tmp_path / "state.json", incompatible).load()


def test_state_payload_tamper_is_rejected(tmp_path):
    manifest = _manifest()
    header = StateHeader(
        evidence_id="p4-dev-v1",
        input_manifest_sha256=manifest.sha256(),
        config_sha256=manifest.config_sha256,
        model_revision=manifest.model_revision,
        bank_sha256=manifest.bank_sha256,
        partition_sha256=manifest.partition_sha256,
    )
    path = tmp_path / "state.json"
    store = StateStore(path, header)
    store.write({"completed": 1})
    value = json.loads(path.read_text())
    value["payload"]["completed"] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="payload hash"):
        store.load()


def test_registry_rejects_native_phase3_and_accepts_immutable_import(
        tmp_path):
    from jspace_phase4 import registry4
    events = tmp_path / "events.jsonl"
    with pytest.raises(Exception, match="immutable import"):
        registry4.append_event({
            "event": "evidence_created",
            "evidence_id": "bad-v1",
            "tier": "phase3-confirmatory",
            "what": "bad",
            "command": "bad",
            "code_commit": "deadbeef",
        }, path=events)
    source = tmp_path / "source.jsonl"
    source.write_text('{"source":"phase3"}\n')
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"value":1}\n')
    registry4.append_event({
        "event": "evidence_imported",
        "evidence_id": "p4-import-p3-v1",
        "tier": "phase3-confirmatory-import",
        "what": "immutable Phase 3 release import",
        "source_study": "jspace-phase3",
        "source_evidence_id": "p3-release-manifest-v1",
        "source_commit": "9e0672b",
        "source_registry_sha256": file_sha256(source),
        "source_outputs": [{
            "path": str(artifact),
            "sha256": file_sha256(artifact),
        }],
    }, path=events)
    record = registry4.resolve("p4-import-p3-v1", path=events)
    assert record["live"]
    assert record["effective_tier"] == "phase3-confirmatory-import"


def test_registry_lifecycle_is_append_only(tmp_path):
    from jspace_phase4 import registry4
    events = tmp_path / "events.jsonl"
    for evidence_id in ("a-v1", "a-v2"):
        registry4.append_event({
            "event": "evidence_created",
            "evidence_id": evidence_id,
            "tier": "phase4-development",
            "what": evidence_id,
            "command": "test",
            "code_commit": "deadbeef",
        }, path=events)
    registry4.append_event({
        "event": "evidence_superseded",
        "evidence_id": "a-v1",
        "superseded_by": "a-v2",
        "reason": "replacement",
    }, path=events)
    assert not registry4.resolve("a-v1", path=events)["live"]
    assert registry4.resolve("a-v2", path=events)["live"]
