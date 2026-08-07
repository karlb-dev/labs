import json
from pathlib import Path

from jspace_phase4.pre_freeze_inventory import (
    build_inventory,
    load_policy,
    render_markdown,
)
from jspace_phase4.registry4 import resolve_all


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "protocol/PRE_FREEZE_INVENTORY_POLICY_PHASE4.json"


def _write_registry(path: Path, output: Path, *, evidence_id="p4-test-v1"):
    event = {
        "schema_version": 1,
        "study_id": "jspace-phase4",
        "event_utc": "2026-08-02T00:00:00Z",
        "event": "evidence_created",
        "evidence_id": evidence_id,
        "tier": "methods",
        "what": "test",
        "command": "test",
        "code_commit": "a" * 40,
        "outputs": [{"path": str(output), "sha256": "b" * 64}],
        "inputs": {},
    }
    path.write_text(json.dumps(event, sort_keys=True) + "\n")


def _inventory(registry: Path, policy: dict, *, deficits=()):
    return build_inventory(
        events_path=registry,
        policy=policy,
        known_deficits=deficits,
        hash_file=lambda _path: "b" * 64,
        commit_check=lambda commit: commit == "a" * 40,
        repository={
            "code_commit": "c" * 40,
            "branch": "test",
            "dirty_tree": False,
        },
        pass_label="unit-test",
    )


def test_clean_inventory_is_review_ready_and_rendered(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("result")
    registry = tmp_path / "events.jsonl"
    _write_registry(registry, output)
    policy = load_policy(POLICY)
    policy["allowed_registered_recovery_outputs"] = []
    policy["forbidden_path_fragments"] = []

    inventory = _inventory(registry, policy)

    assert inventory["payload"]["review_ready"] is True
    assert inventory["payload"]["durability"]["n_verified"] == 1
    assert inventory["payload_sha256"]
    markdown = render_markdown(inventory)
    assert "**REVIEW_READY — NOT A FREEZE OR APPROVAL RECORD**" in markdown
    assert "`all_live_outputs_verified` | PASS" in markdown


def test_known_missing_output_remains_a_failed_release_gate(tmp_path):
    output = tmp_path / "missing.json"
    registry = tmp_path / "events.jsonl"
    _write_registry(registry, output)
    policy = load_policy(POLICY)
    policy["allowed_registered_recovery_outputs"] = []
    policy["forbidden_path_fragments"] = []
    deficit = {
        "evidence_id": "p4-test-v1",
        "path_suffix": "/missing.json",
        "expected_sha256": "b" * 64,
    }

    inventory = _inventory(registry, policy, deficits=[deficit])

    payload = inventory["payload"]
    assert payload["review_ready"] is False
    assert payload["durability"]["only_known_deficits"] is True
    assert payload["gates"]["all_live_outputs_verified"] is False


def test_native_side_event_and_unreviewed_recovery_path_are_rejected(tmp_path):
    output = tmp_path / "recovery" / "fit.ckpt"
    output.parent.mkdir()
    output.write_text("result")
    registry = tmp_path / "events.jsonl"
    _write_registry(registry, output, evidence_id="ol-native-v1")
    policy = load_policy(POLICY)
    policy["allowed_registered_recovery_outputs"] = []

    inventory = _inventory(registry, policy)

    payload = inventory["payload"]
    assert payload["review_ready"] is False
    assert payload["native_side_namespace_violations"] == ["ol-native-v1"]
    assert payload["temporary_or_recovery_path_violations"]


def test_real_registry_recovery_paths_are_exactly_covered_by_policy():
    policy = load_policy(POLICY)
    events = resolve_all()
    references = []
    for event in events:
        if not event["live"]:
            continue
        field = (
            "source_outputs" if event["event"] == "evidence_imported"
            else "outputs")
        for output in event.get(field, []) or []:
            if "/recovery/" in output["path"]:
                references.append((
                    event["evidence_id"], output["path"], output["sha256"]))

    matched = []
    for row in policy["allowed_registered_recovery_outputs"]:
        hits = [
            reference for reference in references
            if reference[0] == row["evidence_id"]
            and reference[1].endswith(row["path_suffix"])
            and reference[2] == row["expected_sha256"]
        ]
        assert len(hits) == 1
        matched.extend(hits)
    assert sorted(matched) == sorted(references)
