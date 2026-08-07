import json
from pathlib import Path

from jspace_phase4.import_bundle import validate_import_bundle
from jspace_phase4.manifests import file_sha256


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reports/OLMO_BANK_W_CAPABILITY_IMPORT_V1.json"
VALIDATION = ROOT / "reports/olmo_bank_w_capability_import_validation_v1.json"
REGISTRY_SNAPSHOT = (
    ROOT / "reports/source_registries/olmo_bank_w_early_d76e937.jsonl")
# The frozen bundle records its pre-reorg repository path; the alias table
# maps it onto the current tree.
BUNDLE_RELATIVE = (
    "interpretability/jspace_phase4/reports/"
    "OLMO_BANK_W_CAPABILITY_IMPORT_V1.json")


def test_olmo_early_bundle_uses_immutable_source_registry_snapshot():
    from jspace_phase4.paths4 import REPO_ROOT, _rewrite_repo_relative

    payload = json.loads(BUNDLE.read_text())["payload"]
    registry = payload["source_registry"]
    assert payload["source_commit"] == (
        "d76e937d2e6294b92d3d599581bd0fb029f5735c")
    assert payload["target"]["import_evidence_id"] == (
        "p4-import-olmo-bank-w-capability-v1")
    assert (
        REPO_ROOT / _rewrite_repo_relative(registry["path"])
        == REGISTRY_SNAPSHOT)
    assert registry["sha256"] == file_sha256(REGISTRY_SNAPSHOT) == (
        "1e66b35068dc6489de10cccad206899a726d522872bec3f5fe3586aa0a20cbca")
    assert len(REGISTRY_SNAPSHOT.read_text().splitlines()) == 6
    assert all(
        row["contains_untouched_intervention_outcome"] is False
        for row in payload["selected_events"])


def test_saved_olmo_early_validation_matches_fresh_strict_replay():
    recorded = json.loads(VALIDATION.read_text())
    current = validate_import_bundle(
        BUNDLE_RELATIVE, allow_existing_target=True)
    assert current == recorded
    assert len(current["selected_events"]) == 5
    assert len(current["outputs"]) == 11
