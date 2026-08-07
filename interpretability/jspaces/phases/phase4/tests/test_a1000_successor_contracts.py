import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PLACEHOLDER = "BIND_REGISTERED_A1000_SHA256_AFTER_FIT_COMPLETION"


def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text())


def registered_a1000_sha256() -> str | None:
    events = [
        json.loads(line) for line in (
            ROOT / "reports" / "evidence_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    origins = [
        row for row in events
        if row.get("event") == "evidence_created"
        and row.get("evidence_id") ==
        "p4-qwen-lens-fit-drawA-n1000-dev-v1"
    ]
    if not origins:
        return None
    assert len(origins) == 1
    lenses = [
        row for row in origins[0]["outputs"]
        if Path(row["path"]).name ==
        "qwen36-27b_jlens_drawA_n1000.pt"
    ]
    assert len(lenses) == 1
    return lenses[0]["sha256"]


def test_a1000_successors_are_prospective_or_share_registered_binding():
    convergence = load(
        "p4_qwen_lens_convergence_drawA_n500_n1000_dev.yaml")
    functional = load(
        "p4_qwen_multilens_functional_gate_a500_a1000_dev.yaml")
    influence = load("p4_qwen_lens_influence_prompt323_dev.yaml")

    observed_digests = []
    for config, path in (
        (convergence, ("lenses", "a1000")),
        (functional, ("lenses", "a1000")),
        (influence, ("lenses", "a1000")),
    ):
        lens = config[path[0]][path[1]]
        assert lens["evidence_id"] == "p4-qwen-lens-fit-drawA-n1000-dev-v1"
        assert lens["lens_uri"].endswith(
            "/qwen36-27b_jlens_drawA_n1000.pt")
        observed_digests.append(lens["lens_sha256"])
        assert lens["n_prompts"] == 1000
    registered = registered_a1000_sha256()
    assert set(observed_digests) == {registered or PLACEHOLDER}


def test_a1000_functional_tolerances_are_inherited_unchanged():
    previous = load(
        "p4_qwen_multilens_functional_gate_a250_a500_dev.yaml")
    successor = load(
        "p4_qwen_multilens_functional_gate_a500_a1000_dev.yaml")

    assert successor["analysis"]["thresholds"] == previous["analysis"][
        "thresholds"]
    assert successor["subset"] == previous["subset"]
    assert successor["protocol"] == previous["protocol"]
    assert successor["g4"] == previous["g4"]
    assert successor["bridge_endpoint"] == previous["bridge_endpoint"]
    assert successor["capacity"] == previous["capacity"]
    assert successor["analysis"]["primary_pair"] == ["a500", "a1000"]
    assert successor["analysis"]["pair_order"] == [
        ["a500", "a1000"],
        ["a1000", "published"],
        ["a500", "published"],
    ]


def test_selection_margin_capture_cannot_change_primary_intervention():
    functional = load(
        "p4_qwen_multilens_functional_gate_a500_a1000_dev.yaml")
    standalone = load("p4_qwen_selection_margin_a500_a1000_dev.yaml")
    capture = functional["selection_margin_capture"]
    contract = standalone["contract"]

    assert capture["enabled"] is True
    assert capture["intervention_k"] == functional["protocol"]["k"] == 10
    assert capture["top_n"] == contract["captured_top_n"] == 32
    assert capture["margin_ks"] == contract["margin_ks"]
    assert capture["stable_core_margin_at_k"] == (
        contract["stable_core_margin_at_intervention_k"])
    assert capture["include_all_strata_in_functional_gate"] is True
    assert contract["include_all_strata_in_primary_gate"] is True
    assert contract["no_behavioral_columns_in_stratification"] is True
    assert standalone["analysis"]["no_position_exclusion"] is True


def test_convergence_reuses_frozen_sampling_and_defines_exact_block_mean():
    previous = load(
        "p4_qwen_lens_convergence_drawA_n250_n500_dev.yaml")
    successor = load(
        "p4_qwen_lens_convergence_drawA_n500_n1000_dev.yaml")

    assert successor["fixed_sampling"] == previous["fixed_sampling"]
    assert successor["task_banks"] == previous["task_banks"]
    assert successor["subspace"] == previous["subspace"]
    assert successor["recipe"] == previous["recipe"]
    assert successor["incremental_block"] == {
        "name": "increment501_1000",
        "prefix": "a500",
        "extended": "a1000",
        "prefix_n": 500,
        "extended_n": 1000,
        "formula": "(1000 * J_a1000 - 500 * J_a500) / 500",
    }


def test_prompt323_contract_is_retention_only_and_paired():
    config = load("p4_qwen_lens_influence_prompt323_dev.yaml")
    contract = config["equal_weight_contract"]

    assert config["canonical_lens_unchanged"] is True
    assert config["prompt"]["one_based_index"] == 323
    assert config["prompt"]["zero_based_index"] == 322
    assert config["prompt"]["retained_unconditionally"] is True
    assert contract["tiny_model_direct_refit_required"] is True
    assert contract["adjacent_atomic_checkpoint_required"] is True
    assert contract["earlier_checkpoint"]["n"] == 195
    assert contract["later_checkpoint"]["n"] == 198
    assert config["analysis"]["leave_one_out_formulas"] == {
        "a500": "(500 * J_a500 - J_prompt323) / 499",
        "a1000": "(1000 * J_a1000 - J_prompt323) / 999",
    }


def test_ql2_draft_carries_preoutcome_commit_and_hash():
    amendment = ROOT / "preregistration" / "QL2_ESTIMAND_AMENDMENT_DRAFT.md"
    text = amendment.read_text()
    normalized = " ".join(text.split())
    assert "PROSPECTIVE CONDITIONAL DRAFT" in text
    assert "no draw-A n=1000 lens" in normalized
    assert "Literal vocabulary-row membership" in text
    decision = load("p4_qwen_canonical_lens_decision_a1000_dev.yaml")
    specification = decision["contract"]["ql2_amendment"]
    assert specification["prospective_commit"] == (
        "3a92492f09cf5311949311ca6273acc190ec636d")
    assert hashlib.sha256(amendment.read_bytes()).hexdigest() == (
        specification["sha256"])
