"""GPU-only all-layer structural comparison of two Qwen Jacobian lenses."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from safetensors import safe_open

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..gpu import require_cuda_gpu
from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import (
    figures_dir,
    materialize_local_file,
    metrics_dir,
    resolve_uri,
)
from ..provenance4 import Provenance4, write_result4
from ..registry4 import RegistryError, create, resolve
from ..seeds import SEED_CONTRACT, stable_seed
from .p4_qwen_nested_lens_fit import (
    model_reference,
    verify_package_versions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def quantile_summary(values: torch.Tensor,
                     quantiles: list[float]) -> dict[str, float]:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("quantile input must be a non-empty vector")
    q = torch.tensor(quantiles, device=values.device, dtype=torch.float32)
    result = torch.quantile(values.float(), q)
    return {
        f"q{int(round(level * 100)):02d}": float(value.item())
        for level, value in zip(quantiles, result, strict=True)
    }


def row_cosines(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("row-cosine inputs must be equal-shape matrices")
    left_norm = torch.linalg.vector_norm(left.float(), dim=1)
    right_norm = torch.linalg.vector_norm(right.float(), dim=1)
    denominator = left_norm * right_norm
    if bool((denominator == 0).any().item()):
        raise RuntimeError("zero row encountered in cosine calculation")
    return (left.float() * right.float()).sum(dim=1) / denominator


def centered_linear_cka(left: torch.Tensor,
                        right: torch.Tensor) -> float:
    """Exact centered linear CKA with rows treated as observations."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("CKA inputs must be equal-shape matrices")
    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    cross = left.T @ right
    left_self = left.T @ left
    right_self = right.T @ right
    numerator = cross.square().sum()
    denominator = torch.sqrt(
        left_self.square().sum() * right_self.square().sum())
    if float(denominator.item()) == 0:
        raise RuntimeError("degenerate centered matrix in CKA")
    return float((numerator / denominator).clamp(0, 1).item())


def stable_sample_ids(*, evidence_id: str, namespace: str,
                      vocab_size: int, n: int, base_seed: int) -> tuple[
                          list[int], int]:
    if not 0 < n <= vocab_size:
        raise ValueError("token sample size must be in 1..vocab_size")
    seed = stable_seed(
        experiment_id=evidence_id,
        item_id=namespace,
        condition="uniform-token-id-sample-without-replacement",
        base_seed=base_seed,
    )
    sample = np.random.default_rng(seed).choice(
        vocab_size, size=n, replace=False)
    return [int(value) for value in sample], seed


def stable_rademacher_probes(*, evidence_id: str, namespace: str,
                             n: int, d_model: int,
                             base_seed: int) -> tuple[torch.Tensor, dict]:
    seed = stable_seed(
        experiment_id=evidence_id,
        item_id=namespace,
        condition="rademacher-transport-probes",
        base_seed=base_seed,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bits = torch.randint(
        0, 2, (n, d_model), generator=generator, dtype=torch.uint8)
    contract = {
        "distribution": "Rademacher +/-1/sqrt(d_model)",
        "seed": seed,
        "shape": [n, d_model],
        "packed_uint8_sha256": hashlib.sha256(
            bits.numpy().tobytes()).hexdigest(),
    }
    probes = (
        bits.to(torch.float32).mul_(2).sub_(1).div_(math.sqrt(d_model)))
    return probes, contract


def load_lens_checkpoint(path: Path, specification: Mapping,
                         recipe: Mapping) -> dict:
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True, mmap=True)
    expected_layers = list(range(int(recipe["expected_source_layers"])))
    if sorted(checkpoint) != ["J", "d_model", "n_prompts", "source_layers"]:
        raise RuntimeError(f"unexpected lens keys in {path}")
    if int(checkpoint["n_prompts"]) != int(specification["n_prompts"]):
        raise RuntimeError(
            f"{path} has n_prompts={checkpoint['n_prompts']}, expected "
            f"{specification['n_prompts']}")
    if int(checkpoint["d_model"]) != int(recipe["expected_d_model"]):
        raise RuntimeError(f"unexpected d_model in {path}")
    if list(checkpoint["source_layers"]) != expected_layers:
        raise RuntimeError(f"unexpected source layers in {path}")
    if sorted(checkpoint["J"]) != expected_layers:
        raise RuntimeError(f"unexpected Jacobian keys in {path}")
    for layer in expected_layers:
        tensor = checkpoint["J"][layer]
        if (
            tensor.dtype != torch.float16
            or list(tensor.shape) != [
                int(recipe["expected_d_model"]),
                int(recipe["expected_d_model"]),
            ]
        ):
            raise RuntimeError(
                f"unexpected tensor contract at layer {layer} in {path}")
    return checkpoint


def verified_model_tensor_sample(
        *, model_path: Path, manifest: Mapping, tensor_names: Mapping,
        token_ids: list[int], expected_vocab_size: int,
        expected_d_model: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    requested_tensors = {
        label: tensor_names[label]
        for label in ("unembedding", "final_norm")
    }
    missing = sorted(set(requested_tensors.values()) - set(weight_map))
    if missing:
        raise RuntimeError(f"model tensor names are missing: {missing}")
    entries = {entry["name"]: entry for entry in manifest["files"]}
    tensor_to_shard = {
        label: weight_map[name]
        for label, name in requested_tensors.items()
    }
    verified_shards = {}
    for shard in sorted(set(tensor_to_shard.values())):
        path = model_path / shard
        entry = entries.get(shard)
        if entry is None or int(path.stat().st_size) != int(entry["bytes"]):
            raise RuntimeError(f"model shard manifest mismatch: {shard}")
        actual = file_sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"model shard hash mismatch: {shard}")
        verified_shards[shard] = {
            "bytes": int(path.stat().st_size),
            "sha256": actual,
        }

    unembedding_name = requested_tensors["unembedding"]
    unembedding_shard = model_path / tensor_to_shard["unembedding"]
    with safe_open(
            str(unembedding_shard), framework="pt", device="cpu") as handle:
        unembedding = handle.get_tensor(unembedding_name)
        if list(unembedding.shape) != [
                expected_vocab_size, expected_d_model]:
            raise RuntimeError("unexpected unembedding shape")
        sampled_unembedding = unembedding.index_select(
            0, torch.tensor(token_ids, dtype=torch.int64)).clone()
        del unembedding

    norm_name = requested_tensors["final_norm"]
    norm_shard = model_path / tensor_to_shard["final_norm"]
    with safe_open(
            str(norm_shard), framework="pt", device="cpu") as handle:
        norm_weight = handle.get_tensor(norm_name).clone()
    if list(norm_weight.shape) != [expected_d_model]:
        raise RuntimeError("unexpected final norm weight shape")
    contract = {
        "tensor_to_shard": tensor_to_shard,
        "verified_shards": verified_shards,
        "verified_shards_inventory_sha256":
            object_sha256(verified_shards),
        "sampled_unembedding_dtype": str(sampled_unembedding.dtype),
        "final_norm_dtype": str(norm_weight.dtype),
    }
    return sampled_unembedding, norm_weight, contract


def effective_gain_on_cuda(norm_weight: torch.Tensor, *,
                           d_model: int, eps: float) -> tuple[
                               torch.Tensor, dict]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    module = Qwen3_5RMSNorm(d_model, eps=eps).to("cuda")
    with torch.no_grad():
        module.weight.copy_(norm_weight.to("cuda"))
        gain = module(torch.ones(
            (1, d_model), device="cuda", dtype=torch.float32)).reshape(-1)
        formula = (
            1.0 + norm_weight.to("cuda", dtype=torch.float32)
        ) * torch.rsqrt(torch.tensor(
            1.0 + eps, device="cuda", dtype=torch.float32))
        maximum_difference = float((gain - formula).abs().max().item())
    if maximum_difference > 2e-6:
        raise RuntimeError(
            "measured Qwen effective gain disagrees with the pinned norm "
            f"formula: {maximum_difference}")
    return gain, {
        "module": (
            "transformers.models.qwen3_5.modeling_qwen3_5."
            "Qwen3_5RMSNorm"),
        "probe": "norm(ones)",
        "eps": eps,
        "formula_max_abs_difference": maximum_difference,
    }


def layer_metrics(
        candidate: torch.Tensor, reference: torch.Tensor, *,
        base_token_rows: torch.Tensor, cka_n: int,
        probes: torch.Tensor, quantiles: list[float],
        candidate_n: int, reference_n: int,
) -> dict:
    if candidate.shape != reference.shape:
        raise ValueError("lens matrices disagree in shape")
    difference = candidate - reference
    candidate_norm = torch.linalg.vector_norm(candidate)
    reference_norm = torch.linalg.vector_norm(reference)
    difference_norm = torch.linalg.vector_norm(difference)
    matrix_cosine = (
        (candidate * reference).sum()
        / (candidate_norm * reference_norm)
    )

    candidate_token = torch.nn.functional.normalize(
        base_token_rows @ candidate, dim=1)
    reference_token = torch.nn.functional.normalize(
        base_token_rows @ reference, dim=1)
    token_cosine = row_cosines(candidate_token, reference_token)
    token_cka = centered_linear_cka(
        candidate_token[:cka_n], reference_token[:cka_n])

    candidate_transport = probes @ candidate.T
    reference_transport = probes @ reference.T
    probe_cosine = row_cosines(
        candidate_transport, reference_transport)
    reference_transport_norm = torch.linalg.vector_norm(
        reference_transport, dim=1)
    probe_relative_error = torch.linalg.vector_norm(
        candidate_transport - reference_transport, dim=1
    ) / reference_transport_norm

    jacobian_row_cosine = row_cosines(candidate, reference)
    merged = (
        candidate_n * candidate + reference_n * reference
    ) / (candidate_n + reference_n)
    merge_shift = (
        torch.linalg.vector_norm(merged - reference) / reference_norm)
    result = {
        "candidate_frobenius": float(candidate_norm.item()),
        "reference_frobenius": float(reference_norm.item()),
        "frobenius_ratio_candidate_over_reference":
            float((candidate_norm / reference_norm).item()),
        "matrix_cosine": float(matrix_cosine.item()),
        "relative_frobenius_delta_to_reference":
            float((difference_norm / reference_norm).item()),
        "symmetric_relative_frobenius_delta":
            float((2 * difference_norm
                   / (candidate_norm + reference_norm)).item()),
        "n_weighted_merge_shift_from_reference":
            float(merge_shift.item()),
        "sampled_token_linear_cka": token_cka,
    }
    for prefix, values in (
            ("jacobian_row_cosine", jacobian_row_cosine),
            ("sampled_token_direction_cosine", token_cosine),
            ("probe_transport_cosine", probe_cosine),
            ("probe_transport_relative_error", probe_relative_error)):
        result.update({
            f"{prefix}_{key}": value
            for key, value in quantile_summary(
                values, quantiles).items()
        })
    return result


def aggregate_layers(rows: list[dict], columns: list[str]) -> dict:
    result = {}
    for column in columns:
        values = np.asarray([row[column] for row in rows], dtype=np.float64)
        result[column] = {
            "min": float(values.min()),
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
            "min_layer": int(rows[int(values.argmin())]["layer"]),
            "max_layer": int(rows[int(values.argmax())]["layer"]),
        }
    return result


def plot_layers(rows: list[dict], png_path: Path,
                pdf_path: Path) -> None:
    frame = pd.DataFrame(rows)
    style = {
        "candidate": "#0072B2",
        "reference": "#D55E00",
        "neutral": "#4D4D4D",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    x = frame["layer"]

    axis = axes[0, 0]
    axis.plot(x, frame["matrix_cosine"], color=style["candidate"], lw=1.8)
    axis.set_ylabel("matrix cosine")
    axis.set_title("A · Jacobian-map agreement", loc="left")

    axis = axes[0, 1]
    axis.fill_between(
        x,
        frame["sampled_token_direction_cosine_q05"],
        frame["sampled_token_direction_cosine_q95"],
        color=style["candidate"],
        alpha=0.18,
        linewidth=0,
    )
    axis.plot(
        x,
        frame["sampled_token_direction_cosine_q50"],
        color=style["candidate"],
        lw=1.8,
        label="median (q05–q95 band)",
    )
    axis.set_ylabel("row cosine")
    axis.set_title("B · 4,096 fixed token directions", loc="left")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    axis.plot(
        x,
        frame["sampled_token_linear_cka"],
        color=style["reference"],
        lw=1.8,
    )
    axis.set_ylabel("centered linear CKA")
    axis.set_xlabel("source layer")
    axis.set_title("C · 1,024-token dictionary geometry", loc="left")

    axis = axes[1, 1]
    axis.plot(
        x,
        frame["relative_frobenius_delta_to_reference"],
        color=style["neutral"],
        lw=1.8,
        label="relative Frobenius delta",
    )
    axis.plot(
        x,
        frame["n_weighted_merge_shift_from_reference"],
        color=style["reference"],
        lw=1.5,
        label="n-weighted merge shift",
    )
    axis.set_ylabel("relative magnitude")
    axis.set_xlabel("source layer")
    axis.set_title("D · Difference and merge sensitivity", loc="left")
    axis.legend(frameon=False, fontsize=8)

    for axis in axes.ravel():
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Qwen3.6-27B lens structure: draw-A n=120 vs published n=1000",
        x=0.08,
        ha="left",
        fontsize=13,
    )
    figure.text(
        0.08,
        0.01,
        "Development recipe/corpus-transfer diagnostic; not the pending "
        "same-corpus nested convergence test.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0.05, 0.045, 0.99, 0.94))
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)


def main() -> None:  # noqa: C901
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    try:
        existing = resolve(config["evidence_id"])
    except RegistryError as error:
        if "found 0" not in str(error):
            raise
    else:
        if not existing["live"]:
            raise RuntimeError("existing structural evidence is not live")
        for output in existing["outputs"]:
            if file_sha256(output["path"]) != output["sha256"]:
                raise RuntimeError("existing structural output hash mismatch")
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return

    gpu = require_cuda_gpu()
    packages = verify_package_versions(config["runtime_packages"])
    candidate_event = resolve(config["candidate"]["evidence_id"])
    if not candidate_event["live"]:
        raise RuntimeError("candidate lens evidence is not live")
    candidate_source = resolve_uri(config["candidate"]["lens_uri"])
    candidate_registered_hashes = {
        output["sha256"] for output in candidate_event["outputs"]
        if Path(output["path"]) == candidate_source
    }
    if candidate_registered_hashes != {
            config["candidate"]["lens_sha256"]}:
        raise RuntimeError("candidate lens is not the registered output")
    candidate_path = materialize_local_file(
        config["candidate"]["lens_uri"],
        expected_sha256=config["candidate"]["lens_sha256"],
    )
    reference_path = Path(config["reference"]["lens_path"])
    if file_sha256(reference_path) != config["reference"]["lens_sha256"]:
        raise RuntimeError("published reference lens hash mismatch")

    recipe = config["recipe"]
    candidate = load_lens_checkpoint(
        candidate_path, config["candidate"], recipe)
    reference = load_lens_checkpoint(
        reference_path, config["reference"], recipe)

    token_ids, token_seed = stable_sample_ids(
        evidence_id=config["evidence_id"],
        namespace=config["sampling"]["seed_namespace"],
        vocab_size=int(recipe["expected_vocab_size"]),
        n=int(config["sampling"]["token_ids_n"]),
        base_seed=int(config["sampling"]["base_seed"]),
    )
    probes_cpu, probe_contract = stable_rademacher_probes(
        evidence_id=config["evidence_id"],
        namespace=config["sampling"]["seed_namespace"],
        n=int(config["sampling"]["transport_probes_n"]),
        d_model=int(recipe["expected_d_model"]),
        base_seed=int(config["sampling"]["base_seed"]),
    )

    model_path = resolve_uri(config["model_uri"])
    snapshot_manifest_path = resolve_uri(
        config["model_snapshot_manifest_uri"])
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text())
    sampled_unembedding, norm_weight, tensor_contract = (
        verified_model_tensor_sample(
            model_path=model_path,
            manifest=snapshot_manifest,
            tensor_names=config["model_tensors"],
            token_ids=token_ids,
            expected_vocab_size=int(recipe["expected_vocab_size"]),
            expected_d_model=int(recipe["expected_d_model"]),
        )
    )
    gain, gain_contract = effective_gain_on_cuda(
        norm_weight,
        d_model=int(recipe["expected_d_model"]),
        eps=float(config["model_tensors"]["rms_norm_eps"]),
    )
    base_token_rows = (
        sampled_unembedding.to("cuda", dtype=torch.float32)
        * gain[None, :]
    )
    probes = probes_cpu.to("cuda", dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()

    quantiles = [float(value) for value in recipe["quantiles"]]
    rows = []
    for layer in range(int(recipe["expected_source_layers"])):
        candidate_j = candidate["J"][layer].to(
            "cuda", dtype=torch.float32)
        reference_j = reference["J"][layer].to(
            "cuda", dtype=torch.float32)
        metrics = layer_metrics(
            candidate_j,
            reference_j,
            base_token_rows=base_token_rows,
            cka_n=int(config["sampling"]["cka_token_ids_n"]),
            probes=probes,
            quantiles=quantiles,
            candidate_n=int(config["candidate"]["n_prompts"]),
            reference_n=int(config["reference"]["n_prompts"]),
        )
        row = {"layer": layer, **metrics}
        rows.append(row)
        print(json.dumps({
            "layer": layer,
            "matrix_cosine": round(row["matrix_cosine"], 6),
            "token_cosine_median": round(
                row["sampled_token_direction_cosine_q50"], 6),
            "token_cka": round(row["sampled_token_linear_cka"], 6),
            "relative_frobenius_delta": round(
                row["relative_frobenius_delta_to_reference"], 6),
        }), flush=True)
        del candidate_j, reference_j

    aggregate_columns = [
        "matrix_cosine",
        "frobenius_ratio_candidate_over_reference",
        "relative_frobenius_delta_to_reference",
        "n_weighted_merge_shift_from_reference",
        "jacobian_row_cosine_q50",
        "sampled_token_direction_cosine_q50",
        "sampled_token_direction_cosine_q05",
        "sampled_token_linear_cka",
        "probe_transport_cosine_q50",
        "probe_transport_relative_error_q50",
    ]
    aggregate = aggregate_layers(rows, aggregate_columns)

    output_dir = (
        metrics_dir(config["slug"]) / "lens_structural_stability"
        / config["evidence_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "layer_metrics.csv"
    result_path = output_dir / "structural_stability_result.json"
    manifest_path = output_dir / "input_manifest.json"
    png_path = figures_dir() / "p4f09_qwen_lens_structural_stability.png"
    pdf_path = figures_dir() / "p4f09_qwen_lens_structural_stability.pdf"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    plot_layers(rows, png_path, pdf_path)

    input_payload = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "comparison_scope": config["comparison_scope"],
        "candidate": {
            **dict(config["candidate"]),
            "materialized_path": str(candidate_path),
        },
        "reference": dict(config["reference"]),
        "model": model_reference(config["model_uri"]),
        "model_snapshot_manifest_sha256":
            file_sha256(snapshot_manifest_path),
        "model_tensor_contract": tensor_contract,
        "runtime_packages": packages,
        "gpu": gpu,
        "token_sample": {
            "seed": token_seed,
            "ids": token_ids,
            "ids_sha256": object_sha256(token_ids),
            "n": len(token_ids),
            "cka_prefix_n":
                int(config["sampling"]["cka_token_ids_n"]),
        },
        "transport_probes": probe_contract,
        "effective_gain": gain_contract,
        "recipe": dict(recipe),
        "producer_sha256": file_sha256(Path(__file__)),
    }
    manifest_envelope = {
        "schema_version": 1,
        "payload": input_payload,
        "payload_sha256": object_sha256(input_payload),
    }
    atomic_json(manifest_path, manifest_envelope)
    payload = {
        "schema_version": 1,
        "comparison_scope": config["comparison_scope"],
        "candidate_label": config["candidate"]["label"],
        "reference_label": config["reference"]["label"],
        "n_layers": len(rows),
        "d_model": int(recipe["expected_d_model"]),
        "sampled_token_ids_n": len(token_ids),
        "sampled_token_cka_n":
            int(config["sampling"]["cka_token_ids_n"]),
        "transport_probes_n":
            int(config["sampling"]["transport_probes_n"]),
        "aggregate": aggregate,
        "layer_metrics_sha256": object_sha256(rows),
        "table_sha256": file_sha256(table_path),
        "figure_png_sha256": file_sha256(png_path),
        "figure_pdf_sha256": file_sha256(pdf_path),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": gpu,
        "decision_context": dict(config["decision_context"]),
        "limitations": [
            "The published n=1000 lens is not the pending same-corpus "
            "nested draw-A n=1000 lens, so this is recipe/corpus transfer.",
            "This lens-only analysis does not satisfy the frozen-item "
            "selection, capacity, G4, or causal stability gates.",
            "Sampled token-direction and CKA estimates use the exact fixed "
            "token IDs recorded in the input manifest.",
        ],
    }
    command = (
        "python -m "
        "jspace_phase4.experiments.p4_qwen_lens_structural_stability "
        f"--config {arguments.config}"
    )
    inputs = {
        "candidate_lens": config["candidate"]["lens_sha256"],
        "candidate_evidence": config["candidate"]["evidence_id"],
        "reference_lens": config["reference"]["lens_sha256"],
        "model_snapshot_manifest": file_sha256(snapshot_manifest_path),
        "model_tensor_shards":
            tensor_contract["verified_shards_inventory_sha256"],
        "input_manifest": manifest_envelope["payload_sha256"],
    }
    write_result4(
        payload,
        result_path,
        Provenance4(
            evidence_id=config["evidence_id"],
            tier=config["tier"],
            command=command,
            inputs=inputs,
            input_manifest_sha256=manifest_envelope["payload_sha256"],
            model=model_reference(config["model_uri"]),
            seed_contract=SEED_CONTRACT,
        ),
    )
    create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "All-layer Qwen lens structural recipe/corpus-transfer "
            "diagnostic: draw-A n=120 versus published n=1000; exact "
            "matrix, sampled token-direction, CKA, and transport metrics."),
        command=command,
        outputs=[
            result_path,
            manifest_path,
            table_path,
            png_path,
            pdf_path,
        ],
        inputs=inputs,
    )
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "result": str(result_path),
        "table": str(table_path),
        "figure_png": str(png_path),
        "figure_pdf": str(pdf_path),
        "aggregate": aggregate,
        "gpu": gpu["name"],
    }, indent=1))


if __name__ == "__main__":
    main()
