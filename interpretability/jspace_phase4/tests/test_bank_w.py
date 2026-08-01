import copy
import re


class _Tokens:
    def __init__(self, values):
        self.input_ids = values


class _WordTokenizer:
    """Deterministic stand-in retaining punctuation as separate tokens."""

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        pieces = re.findall(r"[A-Za-z0-9_-]+|[^\w\s]", text)
        return _Tokens([sum(map(ord, piece)) % 997 for piece in pieces])


def _config():
    return {
        "namespace": "bank-w-unit",
        "superfamilies": [
            "key_value", "state_updates", "graph_path", "stack_queue",
            "deferred_recall", "relational_table",
        ],
        "families_per_superfamily": 12,
        "seeds_per_family": 1,
        "loads": {"low": 2, "high": 6},
        "derivations": ["supplied", "derived"],
        "redundancies": ["once", "redundant"],
        "answer_alphabet": [
            "amber", "blue", "coral", "green",
            "ivory", "lilac", "ochre", "silver",
        ],
        "length_matching": {
            "maximum_pair_difference_tokens": 4,
            "filler_sentence": " Neutral marker; ignore it.",
            "maximum_padding_iterations": 256,
        },
        "shortcut_audit": {
            "maximum_excess_over_load_chance": 0.08,
            "require_balanced_target_positions": True,
        },
        "partition": {
            "development_families": 24,
            "confirmatory_families": 24,
            "replication_families": 24,
            "namespace": "bank-w-unit-partition",
        },
        "capability_guard": {"baseline_accuracy_floor": 0.70},
        "primary": {"joint_test": "shared-family-sign-flip-max-t"},
    }


def test_bank_w_renderers_keep_answer_out_of_query_and_in_trace():
    from jspace_phase4.experiments.p4_author_bank_w import render_case

    config = _config()
    for superfamily in config["superfamilies"]:
        for derivation in config["derivations"]:
            case = render_case(
                superfamily, family_index=3, seed_index=2, load_n=6,
                derivation=derivation, redundancy="once",
                alphabet=config["answer_alphabet"],
                namespace="renderer-unit")
            assert case["answer"] in case["body"]
            assert case["answer"] not in case["query"].lower()
            assert case["answer"] in " ".join(case["solution_trace"])
            assert len(case["context_values"]) == 6
            assert len(set(case["context_values"])) == 6


def test_bank_w_candidate_is_fully_crossed_disjoint_and_length_matched():
    from jspace_phase4.experiments.p4_author_bank_w import build_candidate

    rows, partition, audit = build_candidate(
        _config(), tokenizer=_WordTokenizer())
    assert len(rows) == 72 * 1 * 2 * 2 * 2
    assert audit["n_superfamilies"] == 6
    assert audit["n_families"] == 72
    assert audit["all_axes_fully_crossed"]
    assert audit["length_match_pass"]
    assert audit["maximum_within_seed_prompt_token_span"] <= 4
    assert audit["answer_in_query_count"] == 0
    assert audit["shortcut_audit"]["all_pass"]
    assert audit["target_positions_balanced"]
    assert {key: len(value) for key, value in partition.items()} == {
        "development": 24, "confirmatory": 24, "replication": 24}
    assert not (set(partition["development"])
                & set(partition["confirmatory"])
                | set(partition["development"])
                & set(partition["replication"])
                | set(partition["confirmatory"])
                & set(partition["replication"]))


def test_bank_w_shortcut_audit_rejects_first_position_leak():
    from jspace_phase4.experiments.p4_author_bank_w import shortcut_audit

    rows = [{
        "load": "low", "answer": "amber",
        "context_values": ["amber", "blue"],
    } for _ in range(20)]
    audit = shortcut_audit(rows, maximum_excess=0.08)
    assert not audit["all_pass"]
    assert not audit["by_load"]["low"]["first"]["passes"]


def test_bank_w_partition_is_stable_under_irrelevant_config_order():
    from jspace_phase4.experiments.p4_author_bank_w import build_candidate

    first = _config()
    second = copy.deepcopy(first)
    second["superfamilies"] = list(reversed(second["superfamilies"]))
    _, first_partition, _ = build_candidate(first, tokenizer=_WordTokenizer())
    _, second_partition, _ = build_candidate(second, tokenizer=_WordTokenizer())
    assert first_partition == second_partition
