import json

import torch


def test_identical_operators_have_unit_cosine_and_zero_delta():
    from jspace_phase4.experiments.p4_qwen_lens_convergence import (
        operator_pair_metrics,
    )
    generator = torch.Generator().manual_seed(1)
    matrix = torch.randn((12, 12), generator=generator)
    probes = torch.randn((7, 12), generator=generator)
    result = operator_pair_metrics(
        matrix, matrix, probes=probes, quantiles=[0.05, 0.5, 0.95])
    assert abs(result["matrix_cosine"] - 1.0) < 1e-6
    assert result["symmetric_relative_delta"] == 0
    assert abs(result["probe_transport_cosine_q50"] - 1.0) < 1e-6


def test_scalar_identity_change_is_detected_then_removed():
    from jspace_phase4.experiments.p4_qwen_lens_convergence import (
        identity_views,
        operator_pair_metrics,
    )
    identity = torch.eye(10)
    left, left_meta = identity_views(2 * identity)
    right, right_meta = identity_views(3 * identity)
    probes = torch.randn((5, 10), generator=torch.Generator().manual_seed(2))
    raw = operator_pair_metrics(
        left["raw"], right["raw"], probes=probes,
        quantiles=[0.05, 0.5, 0.95])
    residual = operator_pair_metrics(
        left["minus_alpha_identity"], right["minus_alpha_identity"],
        probes=probes, quantiles=[0.05, 0.5, 0.95])
    assert raw["symmetric_relative_delta"] > 0.3
    assert residual["matrix_cosine"] == 1
    assert residual["symmetric_relative_delta"] == 0
    assert left_meta["identity_scale_alpha"] == 2
    assert right_meta["identity_scale_alpha"] == 3


def test_shared_identity_can_hide_different_low_rank_updates():
    from jspace_phase4.experiments.p4_qwen_lens_convergence import (
        identity_views,
        safe_cosine,
    )
    left = 10 * torch.eye(64)
    right = 10 * torch.eye(64)
    left[0, 1] = 3
    right[2, 3] = 3
    left_views, _ = identity_views(left)
    right_views, _ = identity_views(right)
    assert safe_cosine(left_views["raw"], right_views["raw"]) > 0.99
    assert abs(safe_cosine(
        left_views["minus_alpha_identity"],
        right_views["minus_alpha_identity"])) < 1e-7


def test_incremental_block_reconstructs_exact_suffix_mean():
    from jspace_phase4.experiments.p4_qwen_lens_convergence import (
        incremental_block_mean,
    )
    generator = torch.Generator().manual_seed(3)
    contributions = torch.randn((10, 4, 4), generator=generator)
    prefix = contributions[:4].mean(dim=0)
    extended = contributions.mean(dim=0)
    recovered = incremental_block_mean(
        prefix, extended, prefix_n=4, extended_n=10)
    assert torch.allclose(recovered, contributions[4:].mean(dim=0), atol=1e-6)


def test_fixed_sampling_contract_is_not_evidence_id_derived(tmp_path):
    from jspace_phase4.experiments.p4_qwen_lens_convergence import (
        fixed_rademacher_probes,
        load_fixed_sampling_contract,
    )
    from jspace_phase4.manifests import file_sha256, object_sha256
    token_ids = [7, 2, 9, 4]
    probes, probe_hash = fixed_rademacher_probes(seed=17, n=3, d_model=5)
    payload = {
        "token_sample": {
            "ids": token_ids,
            "ids_sha256": object_sha256(token_ids),
            "seed": 123,
            "cka_prefix_n": 2,
        },
        "transport_probes": {
            "seed": 17,
            "shape": [3, 5],
            "packed_uint8_sha256": probe_hash,
        },
    }
    envelope = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(envelope, sort_keys=True))
    kwargs = dict(
        expected_file_sha256=file_sha256(path),
        expected_ids_sha256=object_sha256(token_ids),
        expected_probes_sha256=probe_hash,
        expected_token_n=4,
        expected_probe_shape=[3, 5],
    )
    first_ids, first_probes, _ = load_fixed_sampling_contract(path, **kwargs)
    second_ids, second_probes, _ = load_fixed_sampling_contract(path, **kwargs)
    assert first_ids == second_ids == token_ids
    assert torch.equal(first_probes, second_probes)
    assert torch.equal(first_probes, probes)


class _TinyTokenizer:
    all_special_ids = [0, 1]

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        lowered = text.strip().lower()
        mapping = {
            "alpha": [2, 5, 6],
            "beta": [3, 5, 6],
            "gamma": [4, 5, 6],
            "delta": [7, 5, 6],
        }
        return {"input_ids": mapping.get(lowered, [])}


def test_task_token_strata_are_deterministic_and_disjoint():
    from jspace_phase4.experiments.p4_qwen_lens_convergence import (
        build_task_token_strata,
    )
    rows = [{
        "fact_id": "f1",
        "accepted_answers": [" Alpha"],
        "counterfactual_accepted": [" Beta"],
        "answer": "Alpha",
        "counterfactual_answer": "Beta",
        "bridge": "Gamma",
        "counterfactual_bridge": "Delta",
        "distractor_bridge": None,
    }]
    first, first_contract = build_task_token_strata(_TinyTokenizer(), rows)
    second, second_contract = build_task_token_strata(_TinyTokenizer(), rows)
    assert first == second
    assert first_contract == second_contract
    assert first["task_answer_only"] == [2, 3]
    assert first["task_bridge_only"] == [4, 7]
    assert first["task_answer_bridge_shared"] == [5, 6]
    assert first["special_control"] == [0, 1]
    sets = [set(values) for values in first.values()]
    assert len(set().union(*sets)) == sum(len(values) for values in first.values())


def test_frequency_deciles_are_deterministic_and_disjoint():
    from jspace_phase4.experiments.p4_qwen_lens_convergence import (
        deterministic_frequency_deciles,
    )
    counts = {token_id: token_id + 1 for token_id in range(100)}
    first, first_meta = deterministic_frequency_deciles(
        counts, excluded_ids={2, 9}, max_per_decile=5, namespace="fixed")
    second, second_meta = deterministic_frequency_deciles(
        counts, excluded_ids={2, 9}, max_per_decile=5, namespace="fixed")
    assert first == second
    assert first_meta == second_meta
    sets = [set(values) for values in first.values()]
    assert len(set().union(*sets)) == sum(len(values) for values in first.values())
    assert not set().union(*sets) & {2, 9}


def test_layer_contract_refuses_order_or_set_drift():
    from jspace_phase4.experiments.p4_qwen_lens_convergence import (
        validate_layer_contract,
    )
    valid = {
        "a": {"source_layers": [0, 1], "J": {0: torch.eye(2), 1: torch.eye(2)}},
        "b": {"source_layers": [0, 1], "J": {0: torch.eye(2), 1: torch.eye(2)}},
    }
    assert validate_layer_contract(valid, expected_source_layers=2) == [0, 1]
    valid["b"]["source_layers"] = [1, 0]
    try:
        validate_layer_contract(valid, expected_source_layers=2)
    except RuntimeError as error:
        assert "order/set" in str(error)
    else:
        raise AssertionError("layer-order drift was not refused")


def test_published_reference_requires_exact_partial_recipe_label():
    from jspace_phase4.experiments.p4_qwen_lens_convergence import (
        PUBLISHED_CLASSIFICATION,
        validate_published_provenance,
    )
    config = {
        "lenses": {
            "a": {"kind": "registered"},
            "published": {"kind": "external_published"},
        },
        "published_reference": {
            "classification": PUBLISHED_CLASSIFICATION,
            "unknown_recipe_fields": ["prompt order"],
        },
    }
    comparisons = [{"left": "a", "right": "published"}]
    validate_published_provenance(config, comparisons)
    config["published_reference"]["classification"] = "published n=1000"
    try:
        validate_published_provenance(config, comparisons)
    except RuntimeError as error:
        assert "partial-recipe label" in str(error)
    else:
        raise AssertionError("missing provenance classification was accepted")


def test_successor_prohibits_redundant_merge_shift():
    from jspace_phase4.experiments.p4_qwen_lens_convergence import (
        assert_no_merge_shift,
    )
    assert_no_merge_shift(["matrix_cosine", "incremental_block_delta"])
    try:
        assert_no_merge_shift(["n_weighted_merge_shift_from_reference"])
    except RuntimeError as error:
        assert "prohibited" in str(error)
    else:
        raise AssertionError("merge-shift column was not prohibited")


def test_randomized_subspace_is_deterministic_and_identical():
    from jspace_phase4.experiments.p4_qwen_lens_convergence import (
        principal_subspace_metrics,
        randomized_left_subspace,
    )
    generator = torch.Generator().manual_seed(19)
    matrix = torch.randn((12, 12), generator=generator)
    omega = torch.randn((12, 6), generator=generator)
    first, first_summary = randomized_left_subspace(
        matrix, omega=omega, rank=4, power_iterations=1)
    second, second_summary = randomized_left_subspace(
        matrix, omega=omega, rank=4, power_iterations=1)
    assert torch.allclose(first.abs(), second.abs(), atol=1e-6)
    assert first_summary == second_summary
    metrics = principal_subspace_metrics(first, second)
    assert metrics["principal_subspace_similarity_mean"] > 0.99999
    assert metrics["principal_angle_max_degrees"] < 0.1
