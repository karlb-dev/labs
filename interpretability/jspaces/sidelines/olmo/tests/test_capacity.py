import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from jspace_olmo_lineage.capacity import (
    bootstrap_estimates,
    canonical_jsonl,
    classify_shift,
    content_token_manifest,
    curve_summary,
    gradient_pursuit,
    lower_median,
    occupancy_from_errors,
    pursuit_batched,
    select_frozen_corpus,
    stratified_prompt_counts,
)


def _unit(rows: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(rows, dim=1)


def test_positive_support_exhaustion_and_known_support():
    torch.manual_seed(0)
    dictionary = _unit(torch.randn(80, 24))
    zero = gradient_pursuit(
        torch.zeros(1, 24), dictionary, k_max=6)
    assert int(zero.achieved_support[0]) == 0
    assert (zero.selected_indices == -1).all()
    assert torch.equal(zero.errors[0], torch.zeros(7))

    support = torch.tensor([3, 17, 41])
    target = (dictionary[support]
              * torch.tensor([2.5, 1.4, 0.8])[:, None]).sum(0, keepdim=True)
    fitted = gradient_pursuit(target, dictionary, k_max=6)
    selected = set(fitted.selected_indices[0, :3].tolist())
    assert set(support.tolist()) <= selected
    assert fitted.errors[0, 3] < 0.05 * fitted.errors[0, 0]

    # Once no new positive atom exists, later K cannot hide additional refits.
    one_dimensional = torch.tensor([[1.0], [-1.0]])
    exhausted = gradient_pursuit(
        torch.tensor([[1.0]]), one_dimensional, k_max=2,
        refit_iterations=2)
    assert int(exhausted.achieved_support[0]) == 1
    assert exhausted.errors[0, 2] == exhausted.errors[0, 1]


def test_batched_pursuit_is_row_independent():
    torch.manual_seed(4)
    dictionary = _unit(torch.randn(120, 32))
    targets = torch.stack([
        2.0 * dictionary[index] + 0.7 * dictionary[(index + 9) % 120]
        for index in range(19)
    ]).float().cpu()
    direct = gradient_pursuit(targets, dictionary, k_max=7)
    batched = pursuit_batched(
        targets, dictionary, k_max=7, batch_positions=5)
    assert torch.allclose(direct.errors, batched.errors, atol=1e-5)
    assert torch.equal(direct.selected_indices, batched.selected_indices)
    assert torch.equal(direct.achieved_support, batched.achieved_support)


def test_crossing_persistence_and_curve_summary():
    # Four positions; J wins at K=1 then loses for two consecutive steps.
    j_gain = np.asarray([
        [8.0, 0.7, 0.5, 0.2],
        [7.0, 0.8, 0.4, 0.1],
        [9.0, 0.6, 0.5, 0.2],
        [6.0, 0.9, 0.4, 0.1],
    ])
    random_gain = np.asarray([
        [1.0, 1.0, 1.0, 1.0],
        [1.1, 1.1, 1.1, 1.1],
        [0.9, 0.9, 0.9, 0.9],
    ])[:, None, :]
    random_gain = np.repeat(random_gain, 4, axis=1)
    j_errors = np.concatenate([
        np.full((4, 1), 20.0),
        20.0 - np.cumsum(j_gain, axis=1),
    ], axis=1)
    random_errors = np.concatenate([
        np.full((3, 4, 1), 20.0),
        20.0 - np.cumsum(random_gain, axis=2),
    ], axis=2)
    occupancy = occupancy_from_errors(
        j_errors, random_errors, persistence=2)
    assert occupancy.tolist() == [2, 2, 2, 2]
    summary = curve_summary(
        j_errors, random_errors, persistence=2,
        persistence_sensitivity=(1, 2, 3))
    assert summary["occupancy_median"] == 2
    assert summary["occupancy_persistence_sensitivity"] == {
        "1": 2, "2": 2, "3": 2}
    assert np.isfinite(summary["excess_share"])


def test_prompt_bootstrap_preserves_domains_and_is_paired():
    domains = ["a", "a", "b", "b"]
    counts = stratified_prompt_counts(domains, draws=40, seed=19)
    assert counts.shape == (40, 4)
    assert np.all(counts[:, :2].sum(axis=1) == 2)
    assert np.all(counts[:, 2:].sum(axis=1) == 2)

    # One position per prompt, K_max=2. J has a consistent advantage.
    j = np.asarray([
        [10.0, 5.0, 4.0],
        [12.0, 6.0, 5.0],
        [8.0, 4.0, 3.0],
        [14.0, 7.0, 6.0],
    ])
    random = np.stack([j + np.asarray([0.0, 1.0, 1.0]) for _ in range(3)])
    estimates = bootstrap_estimates(
        j, random, owners=np.arange(4), prompt_counts=counts,
        persistence=2)
    assert len(estimates["excess_share"]) == 40
    assert np.all(estimates["excess_share"] > 0)
    again = bootstrap_estimates(
        j, random, owners=np.arange(4), prompt_counts=counts,
        persistence=2)
    assert np.array_equal(
        estimates["excess_share"], again["excess_share"])


def test_shift_router_boundaries():
    common = dict(
        occupancy_interval_low=0.0, occupancy_interval_high=0.0,
        equivalence_margin=0.0025, material_margin=0.01)
    assert classify_shift(
        centered_difference=0.001,
        centered_interval_low=-0.002,
        centered_interval_high=0.002,
        occupancy_difference=0, **common) == "stable"
    assert classify_shift(
        centered_difference=0.004,
        centered_interval_low=0.001,
        centered_interval_high=0.007,
        occupancy_difference=0, **common) == "small_shift"
    assert classify_shift(
        centered_difference=0.014,
        centered_interval_low=0.011,
        centered_interval_high=0.018,
        occupancy_difference=0, **common) == "material_shift"
    assert classify_shift(
        centered_difference=0.008,
        centered_interval_low=-0.004,
        centered_interval_high=0.013,
        occupancy_difference=0, **common) == "unresolved"


def test_corpus_selection_and_hashes_are_order_explicit():
    rows = [
        {"domain": domain, "pid": offset + index, "text": f"{domain}-{index}"}
        for domain, offset in (("a", 0), ("b", 10))
        for index in range(4)
    ]
    selected = select_frozen_corpus(
        rows, domains=["b", "a"], rows_per_domain=2)
    assert [row["pid"] for row in selected] == [10, 11, 0, 1]
    encoded = canonical_jsonl(selected).encode()
    assert hashlib.sha256(encoded).hexdigest() == hashlib.sha256(encoded).hexdigest()
    assert content_token_manifest([[1, 2], [3]]) == content_token_manifest(
        [[1, 2], [3]])


def test_lower_median_is_not_average_for_even_samples():
    assert lower_median([1, 2, 8, 9]) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA executor test")
def test_tiny_layer_checkpoint_reconstructs(tmp_path):
    from jspace_olmo_lineage.experiments.capacity import (
        _process_layer, _verify_checkpoint)

    class IdentityNorm(torch.nn.Module):
        def __init__(self, dimension):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.ones(dimension), requires_grad=False)

        def forward(self, value):
            return value * self.weight

    class TinyHF(torch.nn.Module):
        def __init__(self, vocabulary, dimension):
            super().__init__()
            self.output = torch.nn.Embedding(vocabulary, dimension)

        def get_output_embeddings(self):
            return self.output

    class TinyWrapped:
        def __init__(self, dimension):
            self._final_norm = IdentityNorm(dimension).cuda()

    torch.manual_seed(9)
    dimension, vocabulary, layer = 8, 24, 2
    hf = TinyHF(vocabulary, dimension).cuda().eval()
    wrapped = TinyWrapped(dimension)
    activations = torch.randn(8, dimension)
    positions = pd.DataFrame({
        "owner": np.repeat(np.arange(4), 2),
        "pid": np.repeat(np.arange(4), 2),
        "content_token_position": np.tile(np.arange(2), 4),
    })
    selected = [
        {"domain": domain} for domain in ("a", "a", "b", "b")]
    lens = {"J": {layer: torch.eye(dimension)}}
    config = {
        "estimator": {
            "k_max": 3,
            "refit_iterations": 4,
            "refit_learning_rate_cap": 0.25,
            "pursuit_batch_positions": 4,
            "crossing": {
                "persistence": 2,
                "persistence_sensitivity": [1, 2, 3],
            },
            "dictionary": {"chunk_rows": 8},
            "random_controls": {
                "seeds": [11, 12, 13], "chunk_rows": 8},
        },
        "uncertainty": {
            "draws": 20, "seed": 101, "interval_level": 0.90,
            "centering_during_bootstrap": "frozen-full-mean",
        },
    }
    specification = {
        "evidence_id": "ol-test-capacity", "slug": "tiny"}
    checkpoint = tmp_path / "layer.npz"
    result = _process_layer(
        path=checkpoint, layer=layer, activations=activations,
        positions=positions, selected=selected, hf_model=hf,
        wrapped=wrapped, own_lens=lens, common_lens=lens,
        own_lens_sha256="a" * 64, common_lens_sha256="a" * 64,
        input_manifest_sha256="b" * 64,
        specification=specification, config=config)
    assert checkpoint.is_file()
    assert result["summary"]["common_is_own"] is True
    verified, arrays = _verify_checkpoint(
        checkpoint, input_manifest_sha256="b" * 64,
        estimator=config["estimator"])
    assert verified["summary"]["n_positions"] == 8
    assert arrays["own_centered_errors"].shape == (8, 4)
