from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_model_graph_roles_and_revisions_are_frozen():
    config = yaml.safe_load((ROOT / "configs/ol_foundation_v1.yaml").read_text())
    assert [row["role"] for row in config["models"]] == [
        "pretrained_anchor",
        "think_endpoint_3_0",
        "think_endpoint_3_1",
        "sibling_endpoint",
    ]
    assert all(len(row["revision"]) == 40 for row in config["models"])
    assert config["scientific_import_boundary"] == (
        "3b041735d8b842de46a9c0a474fccd0c44e0841a")


def test_predictions_and_capacity_margins_are_frozen():
    text = (ROOT / "preregistration/OLMO_LINEAGE_DEVELOPMENT_PREREGISTRATION.md").read_text()
    for required in (
        "stable: absolute centered-excess difference <0.25",
        "small shift: 0.25--1.0",
        "material shift: >1",
        "load-by-externalization mechanism term",
        "full eight-answer",
        "[-0.08, +0.08]",
    ):
        assert required in text


def test_config_contains_no_placeholder():
    text = (ROOT / "configs/ol_foundation_v1.yaml").read_text()
    assert "TO_BE_FILLED" not in text
    assert "TODO" not in text


def test_live_recovery_mirrors_are_not_immutable_evidence_outputs():
    recovery = (ROOT / "jspace_olmo_lineage/recovery.py").read_text()
    foundation = (
        ROOT / "jspace_olmo_lineage/experiments/foundation.py").read_text()
    assert '"mutable_recovery_mirror": True' in recovery
    assert "outputs = list(manifest_paths.values())" in foundation


def test_recovery_includes_claims_and_state_reports():
    recovery = (ROOT / "jspace_olmo_lineage/recovery.py").read_text()
    assert '"reports/OLMO_LINEAGE_CLAIMS_TABLE.md"' in recovery
    assert '"reports/OLMO_LINEAGE_STATE_OF_RECORD.md"' in recovery
