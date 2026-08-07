import inspect

import pandas as pd
import pytest
import torch

from jspace_phase3.experiments import p3_protected_answer_audit as audit


def test_rank_definition():
    logits = torch.tensor([0.0, 3.0, 1.0, 2.0])
    assert audit.clean_first_token_rank(logits, 1) == 1
    assert audit.clean_first_token_rank(logits, 3) == 2
    assert audit.clean_first_token_rank(logits, 0) == 4


def test_protect_k_must_match_frozen_configs():
    assert audit.validate_protect_k({
        "confirmatory": {"protect_top_k": 10},
        "replication": {"protect_top_k": 10},
    }) == 10
    with pytest.raises(RuntimeError, match="exactly 10"):
        audit.validate_protect_k({
            "confirmatory": {"protect_top_k": 10},
            "replication": {"protect_top_k": 20},
        })


def test_text_hash_mismatch_refused():
    item = {
        "item_id": "f:x#direct", "prompt": "the answer is",
        "accepted_answers": [" x"],
    }
    row = {
        "prompt_text_sha256": audit.text_hash(item["prompt"]),
        "accepted_aliases_text_sha256": audit.canonical_hash(
            item["accepted_answers"]),
    }
    audit.validate_text_hashes(row, item)
    row["prompt_text_sha256"] = "bad"
    with pytest.raises(RuntimeError, match="prompt hash"):
        audit.validate_text_hashes(row, item)


def test_protected_claim_requires_rank_field():
    frame = pd.DataFrame({
        "canonical_family": ["a", "b", "c"],
        "delta_J": [-2.0, -2.0, -2.0],
        "delta_C": [0.0, 0.0, 0.0],
    })
    with pytest.raises(RuntimeError, match="requires rank field"):
        audit.analyze_view(frame, "clean_first_rank", 10)


def test_rank_outcome_item_mismatch_and_partition_overlap_refused():
    ranks = pd.DataFrame({
        "side": ["confirmatory"],
        "item_id": ["a"], "fact_id": ["f"], "variant": ["direct"],
        "bank": ["F"], "canonical_family": ["fam"],
        "relation_group": ["r"], "n_tokens_first_alias": [3],
        "lp_first_alias_remeasured": [-1.0],
    })
    outcomes = pd.DataFrame({
        "item_id": ["b"], "fact_id": ["f"], "variant": ["direct"],
        "bank": ["F"], "canonical_family": ["fam"],
        "relation_group": ["r"], "n_tokens": [3], "lp_baseline": [-1.0],
        "lp_meanJ_span_safe": [-2.0], "lp_ss_matched": [-1.0],
    })
    with pytest.raises(RuntimeError, match="item mismatch"):
        audit.validate_rank_outcome_join(
            ranks, outcomes, side="confirmatory")


def test_baseline_collector_cannot_receive_outcomes():
    params = inspect.signature(audit.measure_item).parameters
    assert "outcomes" not in params
    assert set(params) == {"hf", "session", "item", "side"}

