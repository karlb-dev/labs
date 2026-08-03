from pathlib import Path

import yaml

from jspace_gemma.registry import RegistryError, _validate


ROOT = Path(__file__).resolve().parents[1]


def test_study2_registry_prefix_is_isolated():
    _validate({
        "event": "evidence_created",
        "evidence_id": "gm2-foundation-v1",
        "tier": "methods",
        "what": "test",
        "command": "test",
        "code_commit": "a" * 40,
    })
    try:
        _validate({
            "event": "evidence_created",
            "evidence_id": "ol2-foundation-v1",
            "tier": "methods",
            "what": "test",
            "command": "test",
            "code_commit": "a" * 40,
        })
    except RegistryError:
        pass
    else:
        raise AssertionError("Gemma registry accepted an OLMo study-2 prefix")


def test_g21_config_has_216_pairs_and_no_target_literal():
    path = ROOT / "configs/gm2_backend_parity_calibration.yaml"
    raw = path.read_text()
    config = yaml.safe_load(raw)
    count = (
        len(config["models"])
        * 3
        * len(config["prompt_ids"])
        * len(config["batch_sizes"])
        * len(config["directions"]["draws"])
    )
    assert count == config["pair_count_contract"]["expected_backend_pairs"] == 216
    assert "0.002458" not in raw
    assert config["target_firewall"]["ceiling_freeze_precedes_registry_append"] is True


def test_stage1_target_is_confined_to_g22_config():
    raw = (ROOT / "configs/gm2_stage1_relicense.yaml").read_text()
    assert "0.002458" in raw
    design = (ROOT / "preregistration/G2_STUDY2_FROZEN_DESIGN.md").read_text()
    assert "FROZEN_PRE_G2_1" in design
    sentences = (ROOT / "protocol/G2_STAGE1_CANDIDATE_SENTENCES.md").read_text()
    assert "Branch 1" in sentences and "Branch 2" in sentences and "Branch 3" in sentences
