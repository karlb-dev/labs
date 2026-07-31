import hashlib

import torch


def test_stable_sample_ids_is_deterministic_unique_and_ordered():
    from jspace_phase4.experiments.p4_qwen_lens_structural_stability import (
        stable_sample_ids,
    )
    kwargs = dict(
        evidence_id="evidence",
        namespace="sample",
        vocab_size=100,
        n=20,
        base_seed=3,
    )
    first, first_seed = stable_sample_ids(**kwargs)
    second, second_seed = stable_sample_ids(**kwargs)
    assert first == second
    assert first_seed == second_seed
    assert len(first) == len(set(first)) == 20
    assert min(first) >= 0 and max(first) < 100


def test_rademacher_probe_contract_reconstructs():
    from jspace_phase4.experiments.p4_qwen_lens_structural_stability import (
        stable_rademacher_probes,
    )
    probes, contract = stable_rademacher_probes(
        evidence_id="evidence",
        namespace="probes",
        n=4,
        d_model=9,
        base_seed=0,
    )
    assert probes.shape == (4, 9)
    assert torch.allclose(
        torch.unique(probes),
        torch.tensor([-1 / 3, 1 / 3], dtype=torch.float32),
    )
    bits = ((probes * 3 + 1) / 2).to(torch.uint8)
    assert hashlib.sha256(bits.numpy().tobytes()).hexdigest() == (
        contract["packed_uint8_sha256"])


def test_centered_linear_cka_identical_and_scale_invariant():
    from jspace_phase4.experiments.p4_qwen_lens_structural_stability import (
        centered_linear_cka,
    )
    generator = torch.Generator().manual_seed(11)
    matrix = torch.randn((24, 8), generator=generator)
    assert abs(centered_linear_cka(matrix, matrix) - 1.0) < 1e-6
    assert abs(centered_linear_cka(matrix, 3.5 * matrix) - 1.0) < 1e-6


def test_layer_metrics_identical_lenses_have_exact_agreement():
    from jspace_phase4.experiments.p4_qwen_lens_structural_stability import (
        layer_metrics,
    )
    generator = torch.Generator().manual_seed(17)
    lens = torch.randn((12, 12), generator=generator)
    token_rows = torch.randn((20, 12), generator=generator)
    probes = torch.randn((6, 12), generator=generator)
    result = layer_metrics(
        lens,
        lens,
        base_token_rows=token_rows,
        cka_n=10,
        probes=probes,
        quantiles=[0.05, 0.5, 0.95],
        candidate_n=2,
        reference_n=8,
    )
    assert abs(result["matrix_cosine"] - 1.0) < 1e-6
    assert result["relative_frobenius_delta_to_reference"] == 0
    assert result["n_weighted_merge_shift_from_reference"] < 1e-7
    assert abs(result["sampled_token_linear_cka"] - 1.0) < 1e-6
    assert abs(result["sampled_token_direction_cosine_q50"] - 1.0) < 1e-6
    assert abs(result["probe_transport_cosine_q50"] - 1.0) < 1e-6


def test_load_lens_checkpoint_refuses_metadata_drift(tmp_path):
    from jspace_phase4.experiments.p4_qwen_lens_structural_stability import (
        load_lens_checkpoint,
    )
    path = tmp_path / "lens.pt"
    torch.save({
        "J": {0: torch.eye(3, dtype=torch.float16)},
        "n_prompts": 2,
        "source_layers": [0],
        "d_model": 3,
    }, path)
    specification = {"n_prompts": 2}
    recipe = {"expected_source_layers": 1, "expected_d_model": 3}
    loaded = load_lens_checkpoint(path, specification, recipe)
    assert loaded["J"][0].shape == (3, 3)
    specification["n_prompts"] = 3
    try:
        load_lens_checkpoint(path, specification, recipe)
    except RuntimeError as error:
        assert "n_prompts" in str(error)
    else:
        raise AssertionError("metadata drift was not refused")
