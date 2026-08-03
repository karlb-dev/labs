from pathlib import Path

import yaml

from jspace_olmo_lineage.registry import RegistryError, _validate


ROOT = Path(__file__).resolve().parents[1]


def test_study2_registry_prefix_is_isolated():
    _validate({
        "event": "evidence_created",
        "evidence_id": "ol2-foundation-v1",
        "tier": "methods",
        "what": "test",
        "command": "test",
        "code_commit": "a" * 40,
    })
    try:
        _validate({
            "event": "evidence_created",
            "evidence_id": "gm2-foundation-v1",
            "tier": "methods",
            "what": "test",
            "command": "test",
            "code_commit": "a" * 40,
        })
    except RegistryError:
        pass
    else:
        raise AssertionError("OLMo registry accepted a Gemma study-2 prefix")


def test_wedge_freezes_two_frames_and_three_accounts():
    config = yaml.safe_load((ROOT / "configs/ol2_stage_wedge.yaml").read_text())
    assert config["status"] == "FROZEN_PRE_WEDGE_MODEL_LOAD"
    assert config["tier1"]["frames"] == [
        "base-lens-common", "olmo3-think-endpoint-own"
    ]
    predictions = config["frozen_predictions"]
    assert {"transition_at_sft", "transition_at_dpo", "diffuse"} <= set(predictions)
    assert predictions["diffuse"]["informative_not_failure"] is True
    assert config["exclusions_and_stops"]["no_posthoc_prompt_repair"] is True


def test_transport_thresholds_are_frozen_before_data():
    config = yaml.safe_load((ROOT / "configs/ol2_transport_validation.yaml").read_text())
    assert config["status"] == "FROZEN_PRE_TRANSPORT_DATA"
    assert config["relative_epsilon_ladder"][:3] == [0.001, 0.0025, 0.005]
    assert config["transport_gate"]["tangent_cosine_floor"] == 0.98
    assert config["license_dependency"]["exact_output_hash_required_at_runtime"] is True
