import json
from pathlib import Path

from jspace_phase4.durability import (
    compare_durability_passes,
    load_known_deficits,
    verify_registry_durability,
)
from jspace_phase4.manifests import file_sha256
from jspace_phase4.registry4 import append_event


def _create(events: Path, evidence_id: str, output: Path, digest: str):
    append_event({
        "event": "evidence_created",
        "evidence_id": evidence_id,
        "tier": "phase4-development",
        "what": "test output",
        "command": "test",
        "code_commit": "a" * 40,
        "outputs": [{"path": str(output), "sha256": digest}],
    }, path=events)


def test_whole_registry_snapshot_distinguishes_known_missing(tmp_path):
    events = tmp_path / "events.jsonl"
    good = tmp_path / "good.bin"
    good.write_bytes(b"good")
    missing = tmp_path / "missing.bin"
    _create(events, "good-v1", good, file_sha256(good))
    _create(events, "missing-v1", missing, "b" * 64)
    deficits = [{
        "evidence_id": "missing-v1",
        "path_suffix": "missing.bin",
        "expected_sha256": "b" * 64,
    }]
    result = verify_registry_durability(
        events_path=events, known_deficits=deficits, pass_label="first")
    assert result["n_live_events"] == 2
    assert result["n_output_references"] == 2
    assert result["n_verified"] == 1
    assert result["n_known_deficits"] == 1
    assert result["n_unexpected_failures"] == 0
    assert result["only_known_deficits"] is True
    assert result["ok"] is False


def test_snapshot_ignores_nonlive_origin_and_checks_import_outputs(tmp_path):
    events = tmp_path / "events.jsonl"
    old = tmp_path / "old.bin"
    replacement = tmp_path / "replacement.bin"
    imported = tmp_path / "imported.bin"
    replacement.write_bytes(b"replacement")
    imported.write_bytes(b"imported")
    _create(events, "old-v1", old, "c" * 64)
    _create(events, "replacement-v1", replacement, file_sha256(replacement))
    append_event({
        "event": "evidence_superseded",
        "evidence_id": "old-v1",
        "superseded_by": "replacement-v1",
        "reason": "test replacement",
    }, path=events)
    source_registry = tmp_path / "source.jsonl"
    source_registry.write_text("{}\n")
    append_event({
        "event": "evidence_imported",
        "evidence_id": "import-v1",
        "tier": "phase3-confirmatory-import",
        "what": "test import",
        "source_study": "source",
        "source_evidence_id": "source-v1",
        "source_commit": "d" * 40,
        "source_registry_sha256": file_sha256(source_registry),
        "source_outputs": [{
            "path": str(imported), "sha256": file_sha256(imported)}],
    }, path=events)
    result = verify_registry_durability(
        events_path=events, pass_label="import")
    assert result["ok"] is True
    assert result["n_origin_events"] == 3
    assert result["n_live_events"] == 2
    assert {row["evidence_id"] for row in result["references"]} == {
        "replacement-v1", "import-v1"}


def test_snapshot_detects_hash_drift_and_conflicting_live_pins(tmp_path):
    events = tmp_path / "events.jsonl"
    shared = tmp_path / "shared.bin"
    shared.write_bytes(b"one")
    _create(events, "one-v1", shared, file_sha256(shared))
    _create(events, "two-v1", shared, "e" * 64)
    result = verify_registry_durability(
        events_path=events, pass_label="conflict")
    assert result["ok"] is False
    assert result["n_unexpected_failures"] == 1
    assert len(result["path_pin_conflicts"]) == 1


def test_two_pass_comparison_requires_same_registry_and_rows(tmp_path):
    events = tmp_path / "events.jsonl"
    output = tmp_path / "value.bin"
    output.write_bytes(b"stable")
    _create(events, "stable-v1", output, file_sha256(output))
    first = verify_registry_durability(
        events_path=events, pass_label="first")
    second = verify_registry_durability(
        events_path=events, pass_label="second")
    comparison = compare_durability_passes(first, second)
    assert comparison["consistent"] is True
    assert comparison["clean_both"] is True
    second["references"][0]["actual_bytes"] += 1
    comparison = compare_durability_passes(first, second)
    assert comparison["consistent"] is False
    assert comparison["drifts"]


def test_known_deficit_manifest_rejects_duplicates(tmp_path):
    path = tmp_path / "deficits.json"
    row = {
        "evidence_id": "x", "path_suffix": "x.bin",
        "expected_sha256": "f" * 64,
    }
    path.write_text(json.dumps({
        "schema_version": 1, "deficits": [row, row]}))
    try:
        load_known_deficits(path)
    except RuntimeError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate known deficit was accepted")


def test_live_known_deficits_bind_recovery_and_search_records():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json").read_text())
    by_name = {
        Path(row["path_suffix"]).name: row for row in manifest["deficits"]}
    state = by_name["state.json"]
    capacity = by_name["capacity_reconstructions_a120.pt"]
    assert state["status"] == (
        "exact-bytes-not-found-current-vm-external-resolution-required")
    assert capacity["status"] == "exact-bytes-restored-and-verified"
    for key, row in (("search_record", state),
                     ("recovery_config", capacity),
                     ("recovery_record", capacity)):
        uri = row[key]
        assert uri.startswith("repo://interpretability/jspace_phase4/")
        relative = uri.removeprefix(
            "repo://interpretability/jspace_phase4/")
        assert (root / relative).is_file()
