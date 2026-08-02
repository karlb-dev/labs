from jspace_olmo_lineage.experiments.checkpoint_inventory import (
    _apply_group_compatibility,
    model_contract,
    normalize_bpe_model,
    normalize_base_models,
    route_inventory,
)


def test_bpe_merge_serializations_normalize_identically():
    common = {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}}
    pair_form = {**common, "merges": [["a", "b"]]}
    string_form = {**common, "merges": ["a b"]}
    assert normalize_bpe_model(pair_form) == normalize_bpe_model(string_form)


def test_semantic_group_allows_byte_different_equivalent_files():
    rows = [
        {
            "tokenizer_group": "think",
            "tokenizer_json_sha256": "raw-a",
            "tokenizer_semantics": {"semantic_fingerprint_sha256": "same"},
            "public": True,
            "gated": False,
            "declared_ancestry_matches": True,
            "model_contract": {"passes": True},
            "weights": {"weights_available": True},
            "configured_intermediate_candidate": True,
        },
        {
            "tokenizer_group": "think",
            "tokenizer_json_sha256": "raw-b",
            "tokenizer_semantics": {"semantic_fingerprint_sha256": "same"},
            "public": True,
            "gated": False,
            "declared_ancestry_matches": True,
            "model_contract": {"passes": True},
            "weights": {"weights_available": True},
            "configured_intermediate_candidate": True,
        },
    ]
    _apply_group_compatibility(
        rows, method="semantic-content-tokenization-v2")
    assert all(row["tokenizer_group_compatible"] for row in rows)
    assert all(row["intermediate_eligible"] for row in rows)


def test_normalize_base_models_accepts_model_card_shapes():
    assert normalize_base_models(None) == []
    assert normalize_base_models("allenai/base") == ["allenai/base"]
    assert normalize_base_models(["b", "a"]) == ["a", "b"]


def test_model_contract_is_explicit_and_strict():
    expected = {
        "hidden_size": 5120,
        "vocab_size": 100278,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "architecture": "Olmo3ForCausalLM",
    }
    config = {
        **expected,
        "architectures": ["Olmo3ForCausalLM"],
        "model_type": "olmo3",
    }
    config.pop("architecture")
    assert model_contract(config, expected)["passes"]
    config["hidden_size"] = 4096
    result = model_contract(config, expected)
    assert not result["passes"]
    assert not result["checks"]["hidden_size"]


def test_available_sft_dpo_pair_routes_bounded_wedge():
    rows = [
        {"slug": "sft", "stage": "SFT", "intermediate_eligible": True},
        {"slug": "dpo", "stage": "DPO", "intermediate_eligible": True},
    ]
    config = {
        "minimal_wedge_cells": ["sft", "dpo"],
        "required_think_intermediate_stages": ["SFT", "DPO"],
        "existing_anchors": ["base", "final"],
        "queue_position": "after-o4-version1-gate-resolution",
        "scope": "two cells",
    }
    result = route_inventory(rows, config)
    assert result["decision"] == "genuine-32b-intermediates-available"
    assert result["h5_status"] == "testable-with-bounded-stage-wedge"
    assert result["minimal_wedge"]["cells"] == ["sft", "dpo"]
    assert not result["model_outcome_opened"]


def test_missing_stage_is_stated_unresolvable_without_substitution():
    rows = [
        {"slug": "sft", "stage": "SFT", "intermediate_eligible": True},
        {"slug": "dpo", "stage": "DPO", "intermediate_eligible": False},
    ]
    config = {
        "minimal_wedge_cells": ["sft", "dpo"],
        "required_think_intermediate_stages": ["SFT", "DPO"],
        "existing_anchors": ["base", "final"],
        "queue_position": "after-o4-version1-gate-resolution",
        "scope": "two cells",
    }
    result = route_inventory(rows, config)
    assert result["h5_status"] == "stated-unresolvable-at-32b"
    assert result["minimal_wedge"]["cells"] == []
