import json

import pytest

from jspace_official_repro import layers
from jspace_official_repro import registry as reg


def test_paper_grid_formula_and_bands():
    assert layers.PAPER_GRID == [round(i * 63 / 24) for i in range(25)]
    assert len(layers.PAPER_GRID) == 25
    assert layers.PAPER_GRID[0] == 0 and layers.PAPER_GRID[-1] == 63
    # Banker's rounding sensitivity cells (addendum §2.6):
    assert layers.PAPER_GRID[4] == 10   # 10.5 -> 10, not 11
    assert layers.PAPER_GRID[12] == 32  # 31.5 -> 32
    assert layers.PAPER_GRID[20] == 52  # 52.5 -> 52
    assert layers.PAPER_BAND[0] == 24 and layers.PAPER_BAND[-1] == 58
    assert len(layers.PAPER_BAND) == 14
    assert len(layers.CAMPAIGN_BAND) == 13
    assert len(layers.OLMO_FIT_SOURCE_LAYERS) == 32
    assert len(layers.PAPER_GRID_SOURCES) == 24
    assert layers.PAPER_CAMPAIGN_INTERSECTION == [8, 16, 24, 26, 32, 34, 42, 52, 60]


def test_registry_append_validate_supersede(tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    monkeypatch.setattr(reg, "EVENTS", events)

    with pytest.raises(reg.RegistryError, match="prefix"):
        reg.append_event({"event": "evidence_created", "evidence_id": "bad-1",
                          "tier": "methods", "what": "w", "command": "c",
                          "code_commit": "abc"}, path=events)
    reg.append_event({"event": "evidence_created", "evidence_id": "or1-x-v1",
                      "tier": "methods", "what": "w", "command": "c",
                      "code_commit": "abc"}, path=events)
    with pytest.raises(reg.RegistryError, match="duplicate"):
        reg.append_event({"event": "evidence_created", "evidence_id": "or1-x-v1",
                          "tier": "methods", "what": "w", "command": "c",
                          "code_commit": "abc"}, path=events)
    with pytest.raises(reg.RegistryError, match="unknown replacement"):
        reg.append_event({"event": "evidence_superseded",
                          "evidence_id": "or1-x-v1", "superseded_by": "or1-y-v1",
                          "reason": "r"}, path=events)
    reg.append_event({"event": "evidence_created", "evidence_id": "or1-x-v2",
                      "tier": "development", "what": "w", "command": "c",
                      "code_commit": "abc"}, path=events)
    reg.append_event({"event": "evidence_superseded", "evidence_id": "or1-x-v1",
                      "superseded_by": "or1-x-v2", "reason": "improved"},
                     path=events)
    resolved = reg.resolve("or1-x-v1", path=events)
    assert resolved["superseded_by"] == "or1-x-v2"
    assert not resolved["live"]
    live = reg.live_events(path=events)
    assert [row["evidence_id"] for row in live] == ["or1-x-v2"]
    # Append-only: creation metadata still present in raw rows.
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert rows[0]["evidence_id"] == "or1-x-v1"
    assert rows[0]["event"] == "evidence_created"


def test_raw_record_writer_is_immutable(tmp_path):
    from jspace_official_repro.scoring import RawRecordWriter

    path = tmp_path / "rows.jsonl"
    with RawRecordWriter(path, common={"study_id": "s"}) as writer:
        writer.write({"item_id": "a", "value": 1})
    assert path.exists()
    with pytest.raises(FileExistsError):
        RawRecordWriter(path, common={})
    row = json.loads(path.read_text().splitlines()[0])
    assert row["study_id"] == "s" and row["item_id"] == "a"
