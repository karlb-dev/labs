import json


def rows(indices, texts):
    return [{"idx": index, "text": texts[index]} for index in indices]


def test_hash_order_sample_is_deterministic_and_namespace_separated():
    from jspace_phase4.experiments.p4_qwen_lens_corpora import (
        hash_order_sample,
    )
    population = list(range(100))
    first = hash_order_sample(population, namespace="draw-a", n=20)
    second = hash_order_sample(reversed(population), namespace="draw-a", n=20)
    other = hash_order_sample(population, namespace="draw-b", n=20)
    assert first == second
    assert len(first) == len(set(first)) == 20
    assert first != other


def test_nested_corpora_preserve_prefixes_exclude_spares_and_are_disjoint():
    from jspace_phase4.experiments.p4_qwen_lens_corpora import (
        build_nested_corpora,
    )
    texts = [f"record {index}" for index in range(40)]
    legacy_a = rows([2, 4, 6, 8], texts)
    legacy_b = rows([10, 12], texts)
    kwargs = dict(
        a_prefix_n=2,
        a_total_n=10,
        b_prefix_n=2,
        b_total_n=7,
        a_namespace="a-v1",
        b_namespace="b-v1",
    )
    draw_a, draw_b, audit = build_nested_corpora(
        texts, list(range(40)), legacy_a, legacy_b, **kwargs)
    draw_a_again, draw_b_again, _ = build_nested_corpora(
        texts, list(reversed(range(40))), legacy_a, legacy_b, **kwargs)
    assert draw_a[:2] == legacy_a[:2]
    assert draw_b[:2] == legacy_b
    assert draw_a == draw_a_again
    assert draw_b == draw_b_again
    a_indices = {row["idx"] for row in draw_a}
    b_indices = {row["idx"] for row in draw_b}
    assert a_indices.isdisjoint(b_indices)
    assert {6, 8}.isdisjoint(a_indices | b_indices)
    assert audit == {
        "legacy_a_fit_prefix_n": 2,
        "legacy_a_eval_spares_excluded_n": 2,
        "legacy_b_prefix_n": 2,
        "a_b_overlap_n": 0,
        "a_legacy_eval_spare_overlap_n": 0,
        "b_legacy_eval_spare_overlap_n": 0,
    }


def test_jsonl_encoding_matches_historical_default_json_format():
    from jspace_phase4.experiments.p4_qwen_lens_corpora import jsonl_bytes
    corpus = [{"idx": 3, "text": "caf\u00e9"}]
    expected = (json.dumps(corpus[0]) + "\n").encode("utf-8")
    assert jsonl_bytes(corpus) == expected
