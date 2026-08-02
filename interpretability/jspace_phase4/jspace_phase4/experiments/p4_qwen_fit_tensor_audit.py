"""Independent tensor-level integrity audit for a completed Qwen lens fit.

This engineering audit verifies the final resumable fp32 sum, the registered
fp16 lens, and their exact quantization relationship without loading either
artifact onto the GPU.  It writes no registry event.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch
import yaml

from ..manifests import atomic_json, file_sha256


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_mmap(path: Path) -> Mapping:
    value = torch.load(
        path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected a mapping in {path}")
    return value


def audit_tensors(
    *, config_path: Path, lens_path: Path, checkpoint_path: Path,
    checkpoint_state_path: Path, expected_prompts: int,
    expected_fit_contract_sha256: str | None = None,
) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text())
    recipe = config["recipe"]
    target_layer = int(recipe["target_layer"])
    if recipe["source_layers"] != "all_below_target":
        raise RuntimeError("tensor audit requires all_below_target source layers")
    source_layers = list(range(target_layer))
    d_model = int(recipe["expected_d_model"])
    expected_shape = (d_model, d_model)
    expected_lens_dtype = DTYPES[str(recipe["lens_save_dtype"])]

    header = json.loads(checkpoint_state_path.read_text())
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if int(checkpoint_path.stat().st_size) != int(header["checkpoint_bytes"]):
        raise RuntimeError("checkpoint byte count does not match its header")
    if checkpoint_sha256 != header["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash does not match its header")
    for key in ("n_done", "next_idx"):
        if int(header[key]) != expected_prompts:
            raise RuntimeError(
                f"checkpoint {key}={header[key]}, expected {expected_prompts}")
    if (
        expected_fit_contract_sha256 is not None
        and header["fit_contract_sha256"] != expected_fit_contract_sha256
    ):
        raise RuntimeError("checkpoint fit-contract hash mismatch")

    checkpoint = load_mmap(checkpoint_path)
    lens = load_mmap(lens_path)
    required_checkpoint = {
        "jacobian_sum", "n_done", "next_idx", "source_layers",
        "target_layer", "skip_first",
    }
    required_lens = {"J", "n_prompts", "source_layers", "d_model"}
    if not required_checkpoint.issubset(checkpoint):
        raise RuntimeError("fit checkpoint is missing required fields")
    if not required_lens.issubset(lens):
        raise RuntimeError("lens artifact is missing required fields")

    if int(checkpoint["n_done"]) != expected_prompts:
        raise RuntimeError("checkpoint tensor state has wrong n_done")
    if int(checkpoint["next_idx"]) != expected_prompts:
        raise RuntimeError("checkpoint tensor state has wrong next_idx")
    if list(checkpoint["source_layers"]) != source_layers:
        raise RuntimeError("checkpoint source-layer coverage mismatch")
    if int(checkpoint["target_layer"]) != target_layer:
        raise RuntimeError("checkpoint target-layer mismatch")
    if int(checkpoint["skip_first"]) != int(recipe["skip_first"]):
        raise RuntimeError("checkpoint skip-first mismatch")
    if int(lens["n_prompts"]) != expected_prompts:
        raise RuntimeError("lens prompt count mismatch")
    if list(lens["source_layers"]) != source_layers:
        raise RuntimeError("lens source-layer coverage mismatch")
    if int(lens["d_model"]) != d_model:
        raise RuntimeError("lens d_model mismatch")
    if sorted(checkpoint["jacobian_sum"]) != source_layers:
        raise RuntimeError("checkpoint tensor-layer keys mismatch")
    if sorted(lens["J"]) != source_layers:
        raise RuntimeError("lens tensor-layer keys mismatch")

    layer_results = []
    for layer in source_layers:
        cumulative = checkpoint["jacobian_sum"][layer]
        final = lens["J"][layer]
        if tuple(cumulative.shape) != expected_shape:
            raise RuntimeError(f"checkpoint layer {layer} shape mismatch")
        if tuple(final.shape) != expected_shape:
            raise RuntimeError(f"lens layer {layer} shape mismatch")
        if cumulative.dtype != torch.float32:
            raise RuntimeError(f"checkpoint layer {layer} dtype mismatch")
        if final.dtype != expected_lens_dtype:
            raise RuntimeError(f"lens layer {layer} dtype mismatch")
        cumulative_finite = bool(torch.isfinite(cumulative).all().item())
        final_finite = bool(torch.isfinite(final).all().item())
        if not cumulative_finite or not final_finite:
            raise RuntimeError(f"nonfinite tensor at source layer {layer}")
        expected_final = (cumulative / expected_prompts).to(
            dtype=expected_lens_dtype)
        exact_quantization_match = bool(torch.equal(expected_final, final))
        if not exact_quantization_match:
            raise RuntimeError(
                f"lens layer {layer} is not the exact quantized checkpoint mean")
        layer_results.append({
            "source_layer": layer,
            "shape": list(expected_shape),
            "checkpoint_dtype": str(cumulative.dtype),
            "lens_dtype": str(final.dtype),
            "checkpoint_all_finite": cumulative_finite,
            "lens_all_finite": final_finite,
            "exact_quantized_mean_match": exact_quantization_match,
        })

    return {
        "schema_version": 1,
        "audit_tier": "engineering-integrity-not-registered-evidence",
        "ok": True,
        "expected_prompts": expected_prompts,
        "source_layers": source_layers,
        "target_layer": target_layer,
        "d_model": d_model,
        "config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_path.stat().st_size,
            "state_path": str(checkpoint_state_path),
            "state_sha256": file_sha256(checkpoint_state_path),
            "fit_contract_sha256": header["fit_contract_sha256"],
            "n_done": header["n_done"],
            "next_idx": header["next_idx"],
        },
        "lens": {
            "path": str(lens_path),
            "sha256": file_sha256(lens_path),
            "bytes": lens_path.stat().st_size,
            "n_prompts": int(lens["n_prompts"]),
        },
        "all_layers_finite": True,
        "all_layers_exact_quantized_checkpoint_mean": True,
        "layers": layer_results,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lens", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-state", required=True, type=Path)
    parser.add_argument("--expected-prompts", required=True, type=int)
    parser.add_argument("--fit-contract-sha256")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    result = audit_tensors(
        config_path=arguments.config.resolve(),
        lens_path=arguments.lens.resolve(),
        checkpoint_path=arguments.checkpoint.resolve(),
        checkpoint_state_path=arguments.checkpoint_state.resolve(),
        expected_prompts=arguments.expected_prompts,
        expected_fit_contract_sha256=arguments.fit_contract_sha256,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(arguments.output, result)
    print(json.dumps({
        "ok": result["ok"],
        "prompts": result["expected_prompts"],
        "source_layers": len(result["source_layers"]),
        "checkpoint_sha256": result["checkpoint"]["sha256"],
        "lens_sha256": result["lens"]["sha256"],
        "output": str(arguments.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
