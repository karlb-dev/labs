import torch

from jspace_olmo_lineage.experiments.lens_provenance_audit import (
    _sample_indices,
    classify_pair,
    inspect_lens_checkpoint,
    merge_sample_diagnostic,
    validate_fit_metrics,
)


def test_sparse_sample_indices_never_round_past_large_tensor_endpoint():
    size = 5120 * 5120
    indices = _sample_indices(size)
    assert int(indices[0]) == 0
    assert int(indices[-1]) == size - 1


def test_pairwise_recipe_corpus_classification():
    exact = {"recipe_key": "r", "corpus_key": "a"}
    assert classify_pair(exact, dict(exact)) == "EXACT_SAME_RECIPE_CORPUS"
    assert classify_pair(exact, {
        "recipe_key": "r", "corpus_key": "b",
    }) == "SAME_RECIPE_DIFFERENT_CORPUS"
    assert classify_pair(exact, {
        "recipe_key": "other", "corpus_key": "a",
    }) == "DIFFERENT_RECIPE"
    assert classify_pair(exact, {"recipe_key": "r"}) == "UNKNOWN"


def _save_lens(path, tensors, n_prompts):
    torch.save({
        "J": {layer: value.half() for layer, value in tensors.items()},
        "n_prompts": n_prompts,
        "source_layers": sorted(tensors),
        "d_model": next(iter(tensors.values())).shape[0],
    }, path)


def test_lens_container_and_slice_merge_contract(tmp_path):
    layers = [24, 32, 40]
    slices = []
    for index in range(4):
        values = {
            layer: torch.arange(16).reshape(4, 4).float()
            + index / 8 + layer / 100
            for layer in layers
        }
        path = tmp_path / f"slice{index}.pt"
        _save_lens(path, values, 3)
        audit, checkpoint = inspect_lens_checkpoint(
            path, source_layers=layers, d_model=4, n_prompts=3)
        assert audit["sampled_all_finite"]
        slices.append(checkpoint)

    merged_values = {
        layer: torch.stack([
            checkpoint["J"][layer].float() for checkpoint in slices
        ]).mean(0)
        for layer in layers
    }
    merged_path = tmp_path / "merged.pt"
    _save_lens(merged_path, merged_values, 12)
    audit, merged = inspect_lens_checkpoint(
        merged_path, source_layers=layers, d_model=4, n_prompts=12)
    assert audit["n_prompts"] == 12
    diagnostic = merge_sample_diagnostic(merged, slices)
    assert diagnostic["passes"]
    assert diagnostic["max_abs_merged_minus_slice_mean"] <= 0.002


def test_fit_metrics_contract():
    recipe = {
        "source_layers": [24, 32, 40],
        "target_layer": 63,
        "dim_batch": 8,
        "max_sequence_length": 128,
        "skip_first": 16,
        "fit_rows": 120,
        "prompts_per_slice": 30,
    }
    metrics = {
        "source_layers": [24, 32, 40],
        "target_layer": 63,
        "dim_batch": 8,
        "max_seq_len": 128,
        "skip_first": 16,
        "merged": {"n_prompts": 120},
        "slices": {
            str(index): {"n_prompts": 30, "prompts_done": 30}
            for index in range(4)
        },
    }
    result = validate_fit_metrics(metrics, recipe)
    assert result["slice_prompts"] == [30, 30, 30, 30]
