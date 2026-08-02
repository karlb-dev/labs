from pathlib import Path

import pytest
import yaml

from jspace_olmo_lineage.experiments.final_release import (
    _registry_prefix_record,
    _released_state_text,
    _validate_partition,
    _verify_registry_prefix,
)
from jspace_olmo_lineage.manifests import file_sha256
from jspace_olmo_lineage.registry import EVENTS, resolve_all


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/ol_final_release_v1.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_registry_prefix_survives_append_and_rejects_mutation(tmp_path):
    copy = tmp_path / "events.jsonl"
    copy.write_bytes(EVENTS.read_bytes())
    record = _registry_prefix_record(copy)
    with copy.open("ab") as handle:
        handle.write(b'{"future":"event"}\n')
    assert _verify_registry_prefix(record, copy)["ok"] is True

    changed = bytearray(copy.read_bytes())
    changed[0] = ord("[")
    copy.write_bytes(changed)
    with pytest.raises(ValueError, match="prefix hash drift"):
        _verify_registry_prefix(record, copy)


def test_release_categories_exactly_partition_live_pre_release_registry():
    config = _config()
    records = resolve_all()
    live = [
        record["evidence_id"] for record in records
        if record["live"] and record["evidence_id"] != config["evidence_id"]
    ]
    result = _validate_partition(config["evidence_categories"], live)
    assert result["n_evidence_ids"] == 23
    assert "ol-checkpoint-inventory-v1" not in {
        evidence_id
        for identifiers in config["evidence_categories"].values()
        for evidence_id in identifiers
    }
    with pytest.raises(ValueError, match="partition drift"):
        _validate_partition(
            config["evidence_categories"], live + ["ol-unregistered-v1"])


def test_released_state_closes_bundle_and_stops_parallel_track():
    config = _config()
    source = """# Test state

State date: 2026-08-02T05:47:48Z

Status: OLMo parallel-phase scientific execution and the isolated run-specific
paper are complete at the first release boundary. The final import/restart
bundle is being assembled.

## 3. Registry state

At this state date the append-only registry contains 24 origin events, of which
23 are live. `ol-checkpoint-inventory-v1` remains immutable but is explicitly
superseded by version 2. The 23 live events have 88 immutable outputs, all of
which pass byte/hash verification. Fifty-eight package tests and the exact
dependency lock pass.

The latest event is `ol-independent-reconstruction-v1`, created at
2026-08-02T05:26:50Z from clean source commit
`12f21ad5badeac980c11f0817906ad18c6c1d52d`.

## 4. Results

The authoritative mutable ledger for this release is
`reports/OLMO_LINEAGE_CLAIMS_TABLE.md`; the final bundle will contain a
hash-pinned copy.

## 7. Remaining and queued work

Required to finish this release artifact layer:

1. emit and register the self-verifying final OLMo import/restart bundle;
2. stop this side track and hand its queues to Phase 5.

## 10. Release-boundary checklist

- [ ] final import/restart bundle emitted and registered.

When the final two boxes are complete, this workstream stops and joins the
single Phase 5 router only through its hash-pinned handoff.
"""
    released = _released_state_text(
        source, config=config,
        source_commit="a" * 40, release_utc="2026-08-02T06:00:00Z",
        source_sha256="b" * 64)
    assert "bundle is being assembled" not in released
    assert "- [x] final import/restart bundle emitted and registered." in released
    assert "25 origins, 24 live events, and 101 live outputs" in released
    assert "stopped and may join the single Phase 5 router" in released
    assert config["evidence_id"] in released


def test_release_config_pins_paper_and_exact_output_names():
    config = _config()
    source = config["source_artifacts"]
    for key in ("paper_tex", "paper_pdf"):
        path = ROOT / source[key]["path"]
        assert file_sha256(path) == source[key]["sha256"]
    assert len(source["paper_figures"]) == 5
    assert all(
        file_sha256(ROOT / row["path"]) == row["sha256"]
        for row in source["paper_figures"])
    assert config["outputs"]["bundle_json"].endswith(
        "/IMPORT_BUNDLE_PHASE4.json")
    assert config["outputs"]["paper_pdf"].endswith(
        "/OLMO_LINEAGE_PARALLEL_PHASE.pdf")
    assert config["paper_source_date_epoch"] == 1785648410
