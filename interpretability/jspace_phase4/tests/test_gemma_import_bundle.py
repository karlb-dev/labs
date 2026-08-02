import json
from pathlib import Path

from jspace_phase4.import_bundle import validate_import_bundle


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reports/GEMMA_TRANSPORT_IMPORT_V1.json"
VALIDATION = ROOT / "reports/gemma_transport_import_validation_v1.json"
BUNDLE_RELATIVE = (
    "interpretability/jspace_phase4/reports/GEMMA_TRANSPORT_IMPORT_V1.json")


def test_gemma_terminal_bundle_is_methods_only_and_mechanism_blocked():
    payload = json.loads(BUNDLE.read_text())["payload"]
    result = payload["methods_result"]
    assert payload["source_commit"] == (
        "b0425a441f1b87c33d9bb0b4d08d221942f11923")
    assert payload["target"]["import_evidence_id"] == (
        "p4-import-gemma-transport-v1")
    assert result["terminal_status"] == "COMPLETE_METHODS_BLOCKER"
    assert result["backend_parity_pass"] is False
    assert result["sole_failed_criterion"] == "backend_tangent_all_slots"
    assert result["mechanism_interpretation_allowed"] is False
    assert result["phase4_intervention_authorized"] is False
    assert all(
        row["contains_untouched_intervention_outcome"] is False
        for row in payload["selected_events"])


def test_saved_gemma_validation_matches_fresh_strict_replay():
    recorded = json.loads(VALIDATION.read_text())
    current = validate_import_bundle(
        BUNDLE_RELATIVE, allow_existing_target=True)
    assert current == recorded
    assert len(current["selected_events"]) == 5
    assert len(current["outputs"]) == 21
