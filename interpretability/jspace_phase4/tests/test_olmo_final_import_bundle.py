import json
from pathlib import Path

from jspace_phase4.import_bundle import validate_import_bundle
from jspace_phase4.manifests import file_sha256


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reports/OLMO_LINEAGE_IMPORT_V1.json"
VALIDATION = ROOT / "reports/olmo_lineage_import_validation_v1.json"
BUNDLE_RELATIVE = (
    "interpretability/jspace_phase4/reports/OLMO_LINEAGE_IMPORT_V1.json")
NATIVE_RELEASE = Path(
    "/content/drive/MyDrive/interpret/special-lab-1/"
    "olmo_lineage_20260801/release/IMPORT_BUNDLE_PHASE4.json")


def test_olmo_terminal_bundle_is_methods_only_and_service_blocked():
    payload = json.loads(BUNDLE.read_text())["payload"]
    result = payload["methods_result"]
    native = payload["native_release"]

    assert payload["source_commit"] == (
        "a28cdd54dda335daf55f468e5be8cc65b2fc5253")
    assert payload["target"]["import_evidence_id"] == (
        "p4-import-olmo-lineage-final-v1")
    assert payload["selected_events"] == [{
        "evidence_id": "ol-phase4-final-import-bundle-v1",
        "role": "terminal-self-verifying-methods-release",
        "contains_untouched_intervention_outcome": False,
    }]
    assert native["sha256"] == file_sha256(NATIVE_RELEASE) == (
        "a2486ec5a4759a1f5b21643e7c60766824c48f13ff43240d458ba72147165a2a")
    assert native["source_registry_prefix_sha256"] == (
        "db3fe202026e5cad019ca90a3dceb74efce3b248c02710cfd849dcdbf843e80a")
    assert result["o1_service_status"] == (
        "blocked-at-16-of-20-common-families")
    assert result["o5_status"] == (
        "resolved-no-identifiable-estimand-no-proxy-substitution")
    assert result["independent_reconstruction"] == "pass"
    assert result["phase4_service_ready"] is False
    assert result["intervention_authorized"] is False


def test_saved_olmo_terminal_validation_matches_fresh_strict_replay():
    recorded = json.loads(VALIDATION.read_text())
    current = validate_import_bundle(
        BUNDLE_RELATIVE, allow_existing_target=True)

    assert current == recorded
    assert len(current["selected_events"]) == 1
    assert len(current["outputs"]) == 13
    assert current["selected_events"][0]["tier"] == "methods"
