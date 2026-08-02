import json
from pathlib import Path

import pytest

from jspace_phase4.import_bundle import (
    ImportBundleError,
    validate_import_bundle,
)
from jspace_phase4.manifests import file_sha256, object_sha256
from jspace_phase4.registry4 import append_event


def _append_side(registry: Path, event: dict):
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a") as handle:
        handle.write(json.dumps({
            "schema_version": 1,
            "event_utc": "2026-08-02T00:00:00Z",
            "study_id": "jspace-olmo-lineage",
            **event,
        }, sort_keys=True) + "\n")


def _source_event(registry: Path, output: Path, *, tier="development"):
    _append_side(registry, {
        "event": "evidence_created",
        "evidence_id": "ol-bank-w-think-v1",
        "tier": tier,
        "what": "baseline capability",
        "command": "test",
        "code_commit": "a" * 40,
        "outputs": [{
            "path": str(output),
            "sha256": file_sha256(output),
            "bytes": output.stat().st_size,
        }],
    })


def _bundle(tmp_path: Path, *, tier="development"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "capability.json"
    output.write_text('{"passes":true}\n')
    registry = tmp_path / "side_events.jsonl"
    _source_event(registry, output, tier=tier)
    payload = {
        "schema_version": 1,
        "bundle_id": "ol-phase4-bank-w-capability-v1",
        "source_study": "jspace-olmo-lineage",
        "source_branch": "interp_jspace_olmo_lineage",
        "source_commit": "b" * 40,
        "evidence_id_prefix": "ol-",
        "source_registry": {
            "path": str(registry), "sha256": file_sha256(registry)},
        "selected_events": [{
            "evidence_id": "ol-bank-w-think-v1",
            "role": "bank-w-baseline-capability",
            "contains_untouched_intervention_outcome": False,
        }],
        "target": {
            "study_id": "jspace-phase4",
            "import_evidence_id": "p4-import-olmo-bank-w-capability-v1",
        },
        "governance": {
            "development_or_methods_only": True,
            "contains_confirmatory_intervention_outcomes": False,
            "contains_replication_intervention_outcomes": False,
            "mainline_registry_was_not_written_by_side_track": True,
        },
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
    }))
    return path, registry, output


def _validate(path: Path, main: Path):
    return validate_import_bundle(
        path, main_events_path=main,
        commit_reachable=lambda commit: commit in {"a" * 40, "b" * 40},
        is_ancestor=lambda ancestor, descendant: (
            ancestor == "a" * 40 and descendant == "b" * 40),
    )


def test_valid_side_bundle_verifies_registry_ancestry_and_outputs(tmp_path):
    bundle, registry, output = _bundle(tmp_path)
    main = tmp_path / "main.jsonl"
    result = _validate(bundle, main)
    assert result["ok"] is True
    assert result["source_registry"]["sha256"] == file_sha256(registry)
    assert result["outputs"] == [{
        "path": str(output),
        "sha256": file_sha256(output),
        "bytes": output.stat().st_size,
    }]
    assert result["target_import_evidence_id"].startswith("p4-import-")


def test_bundle_payload_tamper_is_rejected(tmp_path):
    bundle, _, _ = _bundle(tmp_path)
    value = json.loads(bundle.read_text())
    value["payload"]["source_branch"] = "tampered"
    bundle.write_text(json.dumps(value))
    with pytest.raises(ImportBundleError, match="payload hash"):
        _validate(bundle, tmp_path / "main.jsonl")


def test_bundle_rejects_forbidden_tier_or_untouched_outcome(tmp_path):
    bundle, _, _ = _bundle(tmp_path, tier="confirmatory")
    with pytest.raises(ImportBundleError, match="forbidden source tier"):
        _validate(bundle, tmp_path / "main.jsonl")
    bundle, _, _ = _bundle(tmp_path / "second")
    value = json.loads(bundle.read_text())
    value["payload"]["selected_events"][0][
        "contains_untouched_intervention_outcome"] = True
    value["payload_sha256"] = object_sha256(value["payload"])
    bundle.write_text(json.dumps(value))
    with pytest.raises(ImportBundleError, match="no-untouched-outcome"):
        _validate(bundle, tmp_path / "main.jsonl")


def test_bundle_rejects_superseded_source_event(tmp_path):
    bundle, registry, output = _bundle(tmp_path)
    replacement = tmp_path / "replacement.json"
    replacement.write_text("{}\n")
    _append_side(registry, {
        "event": "evidence_created",
        "evidence_id": "ol-bank-w-think-v2",
        "tier": "development",
        "what": "replacement",
        "command": "test",
        "code_commit": "a" * 40,
        "outputs": [{
            "path": str(replacement),
            "sha256": file_sha256(replacement),
            "bytes": replacement.stat().st_size,
        }],
    })
    _append_side(registry, {
        "event": "evidence_superseded",
        "evidence_id": "ol-bank-w-think-v1",
        "superseded_by": "ol-bank-w-think-v2",
        "reason": "replacement",
    })
    value = json.loads(bundle.read_text())
    value["payload"]["source_registry"]["sha256"] = file_sha256(registry)
    value["payload_sha256"] = object_sha256(value["payload"])
    bundle.write_text(json.dumps(value))
    assert output.exists()
    with pytest.raises(ImportBundleError, match="superseded"):
        _validate(bundle, tmp_path / "main.jsonl")


def test_bundle_rejects_target_collision_and_output_hash_drift(tmp_path):
    bundle, _, output = _bundle(tmp_path)
    main = tmp_path / "main.jsonl"
    append_event({
        "event": "evidence_created",
        "evidence_id": "p4-import-olmo-bank-w-capability-v1",
        "tier": "methods",
        "what": "collision",
        "command": "test",
        "code_commit": "c" * 40,
    }, path=main)
    with pytest.raises(ImportBundleError, match="already exists"):
        _validate(bundle, main)
    output.write_text("tampered\n")
    with pytest.raises(ImportBundleError, match="byte-count drift|hash drift"):
        _validate(bundle, tmp_path / "empty-main.jsonl")


def test_registry_accepts_side_bundle_only_as_an_import_tier(tmp_path):
    events = tmp_path / "events.jsonl"
    source_registry = tmp_path / "source.jsonl"
    source_registry.write_text("{}\n")
    output = tmp_path / "output.json"
    output.write_text("{}\n")
    append_event({
        "event": "evidence_imported",
        "evidence_id": "p4-import-olmo-test-v1",
        "tier": "side-development-import",
        "what": "side development bundle",
        "source_study": "jspace-olmo-lineage",
        "source_evidence_id": "ol-bundle-v1",
        "source_commit": "a" * 40,
        "source_registry_sha256": file_sha256(source_registry),
        "source_outputs": [{
            "path": str(output), "sha256": file_sha256(output)}],
    }, path=events)
    with pytest.raises(Exception, match="native creation"):
        append_event({
            "event": "evidence_created",
            "evidence_id": "bad-side-native-v1",
            "tier": "side-development-import",
            "what": "not an import",
            "command": "test",
            "code_commit": "b" * 40,
        }, path=events)
