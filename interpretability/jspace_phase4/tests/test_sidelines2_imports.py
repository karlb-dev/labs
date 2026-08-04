import json
from pathlib import Path

import pytest
import yaml

from jspace_phase4.experiments.p4_import_sidelines_study2 import (
    Sidelines2ImportError,
    load_config,
    validate_sidelines2_bundle,
)
from jspace_phase4.manifests import file_sha256, object_sha256
from jspace_phase4.registry4 import append_event

REPO_ROOT = Path(__file__).resolve().parents[3]
GEMMA_CONFIG = (
    REPO_ROOT / "interpretability/jspace_phase4/configs/"
    "p4_import_gemma_transport_study2.yaml")
OLMO_CONFIG = (
    REPO_ROOT / "interpretability/jspace_phase4/configs/"
    "p4_import_olmo_lineage_study2.yaml")
PHASE4_EVENTS = (
    REPO_ROOT / "interpretability/jspace_phase4/reports/"
    "evidence_events.jsonl")
DRIVE_LAB = Path("/content/drive/MyDrive/interpret/special-lab-1")

SOURCE_COMMIT = "b" * 40
MERGED_HEAD = "c" * 40


def _side_row(**event) -> dict:
    return {
        "schema_version": 1,
        "event_utc": "2026-08-03T00:00:00Z",
        "study_id": "jspace-side-study2",
        **event,
    }


def _fixture(tmp_path: Path) -> dict:
    """Build a hermetic repository with a two-event Study-2 side track."""
    repository = tmp_path / "repository"
    external = tmp_path / "external"
    external.mkdir(parents=True)
    output_a = external / "calibration.json"
    output_a.write_text('{"ceiling": 0.5}\n')
    output_b = external / "license.json"
    output_b.write_text('{"licensed": true}\n')

    registry = repository / "interpretability/side/reports/events.jsonl"
    registry.parent.mkdir(parents=True)
    rows = [
        _side_row(
            event="evidence_created",
            evidence_id="gm2-calibration-v1",
            tier="methods",
            what="calibration",
            command="test",
            code_commit="a" * 40,
            outputs=[{
                "path": str(output_a),
                "sha256": file_sha256(output_a),
            }],
        ),
        _side_row(
            event="evidence_created",
            evidence_id="gm2-license-v1",
            tier="development",
            what="license",
            command="test",
            code_commit="a" * 40,
            outputs=[{
                "path": str(output_b),
                "sha256": file_sha256(output_b),
            }],
        ),
    ]
    registry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    prefix = repository / "interpretability/side/release/prefix.jsonl"
    prefix.parent.mkdir(parents=True)
    prefix.write_bytes(registry.read_bytes())
    # The live registry legitimately appends the release event itself.
    with registry.open("a") as handle:
        handle.write(json.dumps(_side_row(
            event="evidence_created",
            evidence_id="gm2-sidelines2-import-bundle-v1",
            tier="methods",
            what="release",
            command="test",
            code_commit=SOURCE_COMMIT,
            outputs=[],
        ), sort_keys=True) + "\n")

    artifact = repository / "interpretability/side/release/STATE_V2.md"
    artifact.write_text("state of record v2\n")

    admitted = []
    for row in rows:
        admitted.append({
            "category": "study2",
            "code_commit": row["code_commit"],
            "evidence_id": row["evidence_id"],
            "origin_event": "evidence_created",
            "outputs": [{
                "path": output["path"],
                "sha256": output["sha256"],
                "bytes": Path(output["path"]).stat().st_size,
            } for output in row["outputs"]],
            "status_events": [],
            "tier": row["tier"],
            "what": row["what"],
        })
    payload = {
        "schema_version": 1,
        "study_id": "jspace-side-study2",
        "bundle_id": "jspace-side-sidelines2-v1",
        "evidence_id": "gm2-sidelines2-import-bundle-v1",
        "source_git": {
            "branch": "side_branch_2",
            "code_commit": SOURCE_COMMIT,
            "dirty_tree": False,
        },
        "shared_parent": "9" * 40,
        "native_tiers": ["development", "methods"],
        "forbidden_uses": ["not confirmatory"],
        "partial_statuses": {"intervention": "not-opened"},
        "claim_boundary": "methods only",
        "result_summary": {"ceiling": 0.5},
        "admitted_evidence": admitted,
        "registry_prefix": {
            "line_count": 2,
            "prefix_bytes": prefix.stat().st_size,
            "prefix_sha256": file_sha256(prefix),
            "path": "interpretability/side/reports/events.jsonl",
            "through_evidence_id": "gm2-license-v1",
        },
        "registry_snapshot": {
            "repo_path": "interpretability/side/release/prefix.jsonl",
            "sha256": file_sha256(prefix),
            "bytes": prefix.stat().st_size,
        },
        "release_artifacts": [{
            "repo_path": "interpretability/side/release/STATE_V2.md",
            "role": "state-of-record-v2",
            "sha256": file_sha256(artifact),
            "bytes": artifact.stat().st_size,
        }],
    }
    bundle = repository / "interpretability/side/release/BUNDLE.json"
    bundle.write_text(json.dumps({
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
    }))
    markdown = repository / "interpretability/side/release/BUNDLE.md"
    markdown.write_text("# side bundle\n")

    main_events = tmp_path / "phase4_events.jsonl"
    source_registry = tmp_path / "study1_source.jsonl"
    source_registry.write_text("{}\n")
    dependency_output = tmp_path / "study1_output.json"
    dependency_output.write_text("{}\n")
    append_event({
        "event": "evidence_imported",
        "evidence_id": "p4-import-side-study1-v1",
        "tier": "side-development-import",
        "what": "study-1 dependency",
        "source_study": "jspace-side",
        "source_evidence_id": "gm-bundle-v1",
        "source_commit": "a" * 40,
        "source_registry_sha256": file_sha256(source_registry),
        "source_outputs": [{
            "path": str(dependency_output),
            "sha256": file_sha256(dependency_output),
        }],
    }, path=main_events)

    config = {
        "schema_version": 1,
        "evidence_id": "p4-import-side-study2-v1",
        "expected_payload_study_id": "jspace-side-study2",
        "expected_bundle_id": "jspace-side-sidelines2-v1",
        "expected_terminal_event": "gm2-sidelines2-import-bundle-v1",
        "expected_source_branch": "side_branch_2",
        "expected_source_commit": SOURCE_COMMIT,
        "allowed_admitted_prefixes": ["gm-", "gm2-"],
        "required_partial_statuses": {"intervention": "not-opened"},
        "bundle": {
            "repo_path": "interpretability/side/release/BUNDLE.json",
            "sha256": file_sha256(bundle),
        },
        "bundle_markdown": {
            "repo_path": "interpretability/side/release/BUNDLE.md",
            "sha256": file_sha256(markdown),
        },
        "frozen_prefix": {
            "repo_path": "interpretability/side/release/prefix.jsonl",
            "sha256": file_sha256(prefix),
        },
        "live_registry_repo_path": "interpretability/side/reports/events.jsonl",
        "required_dependency_imports": ["p4-import-side-study1-v1"],
        "phase4_use": "methods-development-boundary-update",
        "imported_meaning": "methods boundary update only",
        "forbidden_phase4_uses": ["not confirmatory"],
        "validation_output": str(tmp_path / "validation.json"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return {
        "repository": repository,
        "config_path": config_path,
        "config": config,
        "bundle": bundle,
        "registry": registry,
        "prefix": prefix,
        "main_events": main_events,
        "outputs": [output_a, output_b],
    }


def _validate(fixture: dict) -> dict:
    return validate_sidelines2_bundle(
        load_config(fixture["config_path"]),
        repository=fixture["repository"],
        main_events_path=fixture["main_events"],
        commit_reachable=lambda commit: commit in {"a" * 40, SOURCE_COMMIT},
        is_ancestor=lambda ancestor, descendant: (
            ancestor == SOURCE_COMMIT and descendant == MERGED_HEAD),
        head_commit=MERGED_HEAD,
    )


def _rewrite_bundle(fixture: dict, mutate) -> None:
    value = json.loads(fixture["bundle"].read_text())
    mutate(value["payload"])
    value["payload_sha256"] = object_sha256(value["payload"])
    fixture["bundle"].write_text(json.dumps(value))
    config = dict(fixture["config"])
    config["bundle"] = {
        "repo_path": config["bundle"]["repo_path"],
        "sha256": file_sha256(fixture["bundle"]),
    }
    fixture["config"] = config
    fixture["config_path"].write_text(yaml.safe_dump(config))


def test_happy_path_validation_is_deterministic(tmp_path):
    fixture = _fixture(tmp_path)
    first = _validate(fixture)
    second = _validate(fixture)
    assert first["ok"] is True
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True)
    assert first["selected_event_ids"] == [
        "gm2-calibration-v1", "gm2-license-v1"]
    assert first["frozen_prefix"]["sha256"] == file_sha256(fixture["prefix"])
    assert first["live_registry"]["frozen_prefix_intact"] is True
    assert first["n_source_outputs"] == 3
    assert first["native_tier_preserved"] is True


def test_mutated_frozen_prefix_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    raw = bytearray(fixture["registry"].read_bytes())
    raw[0:1] = b"X"
    fixture["registry"].write_bytes(bytes(raw))
    with pytest.raises(Sidelines2ImportError, match="mutated the frozen"):
        _validate(fixture)


def test_later_append_only_rows_are_accepted(tmp_path):
    fixture = _fixture(tmp_path)
    with fixture["registry"].open("a") as handle:
        handle.write(json.dumps(_side_row(
            event="evidence_created",
            evidence_id="gm2-later-v1",
            tier="methods",
            what="later append",
            command="test",
            code_commit="a" * 40,
            outputs=[],
        ), sort_keys=True) + "\n")
    result = _validate(fixture)
    assert result["ok"] is True
    assert result["live_registry"]["n_rows"] == 4


def test_live_registry_may_not_restate_an_admitted_event(tmp_path):
    fixture = _fixture(tmp_path)
    with fixture["registry"].open("a") as handle:
        handle.write(json.dumps(_side_row(
            event="evidence_superseded",
            evidence_id="gm2-license-v1",
            superseded_by="gm2-later-v1",
            reason="post-freeze restatement",
        ), sort_keys=True) + "\n")
    with pytest.raises(Sidelines2ImportError, match="superseded|restates"):
        _validate(fixture)


def test_missing_dependency_import_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["main_events"].write_text("")
    with pytest.raises(
            Sidelines2ImportError, match="dependency import is absent"):
        _validate(fixture)


def test_tier_promotion_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)

    def promote(payload):
        payload["admitted_evidence"][0]["tier"] = "confirmatory"

    _rewrite_bundle(fixture, promote)
    with pytest.raises(Sidelines2ImportError, match="forbidden source tier"):
        _validate(fixture)

    fixture = _fixture(tmp_path / "second")

    def promote_declared(payload):
        payload["native_tiers"] = ["development", "confirmatory"]

    _rewrite_bundle(fixture, promote_declared)
    with pytest.raises(
            Sidelines2ImportError, match="forbidden native tiers"):
        _validate(fixture)


def test_forbidden_phase4_use_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    config = dict(fixture["config"])
    config["phase4_use"] = "confirmatory-import"
    fixture["config_path"].write_text(yaml.safe_dump(config))
    with pytest.raises(Sidelines2ImportError, match="forbidden phase4_use"):
        load_config(fixture["config_path"])


def test_native_side_leakage_in_phase4_registry_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    append_event({
        "event": "evidence_created",
        "evidence_id": "gm2-leaked-native-v1",
        "tier": "methods",
        "what": "leaked native side event",
        "command": "test",
        "code_commit": "a" * 40,
    }, path=fixture["main_events"])
    with pytest.raises(Sidelines2ImportError, match="leaked into Phase 4"):
        _validate(fixture)


def test_duplicate_import_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    source_registry = tmp_path / "duplicate_source.jsonl"
    source_registry.write_text("{}\n")
    append_event({
        "event": "evidence_imported",
        "evidence_id": "p4-import-side-study2-v1",
        "tier": "side-development-import",
        "what": "already registered",
        "source_study": "jspace-side-study2",
        "source_evidence_id": "gm2-sidelines2-import-bundle-v1",
        "source_commit": SOURCE_COMMIT,
        "source_registry_sha256": file_sha256(source_registry),
        "source_outputs": [{
            "path": str(source_registry),
            "sha256": file_sha256(source_registry),
        }],
    }, path=fixture["main_events"])
    with pytest.raises(Sidelines2ImportError, match="already exists"):
        _validate(fixture)


def test_payload_tamper_and_output_drift_are_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    value = json.loads(fixture["bundle"].read_text())
    value["payload"]["claim_boundary"] = "tampered"
    fixture["bundle"].write_text(json.dumps(value))
    config = dict(fixture["config"])
    config["bundle"] = {
        "repo_path": config["bundle"]["repo_path"],
        "sha256": file_sha256(fixture["bundle"]),
    }
    fixture["config_path"].write_text(yaml.safe_dump(config))
    with pytest.raises(Sidelines2ImportError, match="payload hash mismatch"):
        _validate(fixture)

    fixture = _fixture(tmp_path / "drift")
    fixture["outputs"][0].write_text('{"ceiling": 0.9}\n')
    with pytest.raises(Sidelines2ImportError, match="hash drift"):
        _validate(fixture)


def _live_config_check(config_path: Path) -> None:
    config = load_config(config_path)
    events = [
        json.loads(line)
        for line in PHASE4_EVENTS.read_text().splitlines() if line.strip()
    ]
    origins = {
        row["evidence_id"]: row for row in events
        if row.get("event") in {"evidence_created", "evidence_imported"}
    }
    registered = origins.get(config["evidence_id"])
    if registered is None:
        try:
            validation = validate_sidelines2_bundle(config)
        except Sidelines2ImportError as error:
            if ("dependency import is absent: p4-import-" in str(error)
                    and "-study2-" in str(error)):
                # Staged admission order: the sibling Study-2 import is not
                # registered yet. The immutable pins must already hold.
                bundle = REPO_ROOT / config["bundle"]["repo_path"]
                prefix = REPO_ROOT / config["frozen_prefix"]["repo_path"]
                live = REPO_ROOT / config["live_registry_repo_path"]
                assert file_sha256(bundle) == config["bundle"]["sha256"]
                assert file_sha256(prefix) == (
                    config["frozen_prefix"]["sha256"])
                assert live.read_bytes().startswith(prefix.read_bytes())
                return
            raise
        assert validation["ok"] is True
        assert validation["bundle"]["sha256"] == config["bundle"]["sha256"]
        return
    # Already admitted: the registered event must bind the exact pins and
    # the frozen prefix must remain byte-intact in the live side registry.
    assert registered["tier"] == "side-development-import"
    assert registered["bundle_sha256"] == config["bundle"]["sha256"]
    assert registered["frozen_prefix_sha256"] == (
        config["frozen_prefix"]["sha256"])
    assert registered["source_commit"] == config["expected_source_commit"]
    assert registered["source_registry_sha256"] == (
        config["frozen_prefix"]["sha256"])
    prefix = REPO_ROOT / config["frozen_prefix"]["repo_path"]
    live = REPO_ROOT / config["live_registry_repo_path"]
    assert file_sha256(prefix) == config["frozen_prefix"]["sha256"]
    assert live.read_bytes().startswith(prefix.read_bytes())
    for dependency in config["required_dependency_imports"]:
        assert dependency in origins


@pytest.mark.skipif(
    not DRIVE_LAB.exists(), reason="campaign Drive is not mounted")
def test_live_gemma_study2_bundle_materializes_exactly():
    _live_config_check(GEMMA_CONFIG)


@pytest.mark.skipif(
    not DRIVE_LAB.exists(), reason="campaign Drive is not mounted")
def test_live_olmo_study2_bundle_materializes_exactly():
    _live_config_check(OLMO_CONFIG)


def test_phase4_registry_has_no_native_side_origins():
    events = [
        json.loads(line)
        for line in PHASE4_EVENTS.read_text().splitlines() if line.strip()
    ]
    leaked = [
        row["evidence_id"] for row in events
        if row.get("event") in {"evidence_created", "evidence_imported"}
        and row["evidence_id"].startswith(("gm-", "gm2-", "ol-", "ol2-"))
    ]
    assert leaked == []


NARRATIVE_DOCUMENTS = [
    "interpretability/jspace_phase4/paper/PAPER_CONCLUSION_SKELETON.md",
    "interpretability/jspace_phase4/paper/PHASE4_METHODS_DECISION_RECORD.md",
    "interpretability/jspace_phase4/reports/PHASE4_DEVELOPMENT_REPORT.md",
    "interpretability/jspace_phase4/preregistration/"
    "FREEZE_GATE_LEDGER_PHASE4.md",
    "interpretability/jspace_phase4/preregistration/"
    "SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md",
    "interpretability/jspace_phase4/manifests/parallel_import_inventory.md",
    "interpretability/jspace_phase4/reviews/"
    "READY_FOR_PHASE4_FREEZE_REVIEW.md",
]


def test_closeout_narratives_cite_study2_admissions_not_native_ids():
    for relative in NARRATIVE_DOCUMENTS:
        text = (REPO_ROOT / relative).read_text()
        assert "p4-import-gemma-transport-study2-v1" in text, relative
        assert "p4-import-olmo-lineage-study2-v1" in text, relative
        # Native side event IDs may be discussed as source provenance but
        # must never be presented as registered Phase 4 evidence IDs.
        assert "`gm2-sidelines2-import-bundle-v1` is registered" not in text
        assert "`ol2-sidelines2-import-bundle-v1` is registered" not in text


def test_pre_freeze_policy_forbids_all_native_side_prefixes():
    policy = json.loads((
        REPO_ROOT / "interpretability/jspace_phase4/protocol/"
        "PRE_FREEZE_INVENTORY_POLICY_PHASE4.json").read_text())
    assert set(policy["native_side_event_prefixes"]) == {
        "ol-", "ol2-", "gm-", "gm2-"}
