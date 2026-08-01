"""Exact prompt-112 influence and equal-weight estimator contract checks."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import (
    figures_dir,
    local_work,
    materialize_local_file,
    metrics_dir,
    resolve_uri,
    run_root,
)
from ..provenance4 import Provenance4, write_result4
from ..registry4 import RegistryError, create, resolve
from .p4_qwen_lens_convergence import (
    _load_bank_rows,
    aggregate_numeric,
    build_task_token_strata,
    identity_views,
    load_fixed_sampling_contract,
    operator_pair_metrics,
    token_pair_metrics,
)
from .p4_qwen_lens_structural_stability import (
    effective_gain_on_cuda,
    load_lens_checkpoint,
    verified_model_tensor_sample,
)
from .p4_qwen_nested_lens_fit import (
    copy_atomic_verified,
    ensure_free_space,
    jlens_source_contract,
    load_rows,
    model_reference,
    qwen_fused_kernel_contract,
    verify_model_fused_bindings,
    verify_package_versions,
    verify_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def leave_one_out_mean(full_mean: torch.Tensor,
                       contribution: torch.Tensor, *, n: int) -> torch.Tensor:
    if full_mean.shape != contribution.shape:
        raise ValueError("mean and contribution must have equal shape")
    if n <= 1:
        raise ValueError("leave-one-out mean needs n > 1")
    return (n * full_mean.float() - contribution.float()) / (n - 1)


def adjacent_sum_from_running_sums(earlier_sum: torch.Tensor,
                                   later_sum: torch.Tensor) -> torch.Tensor:
    if earlier_sum.shape != later_sum.shape:
        raise ValueError("running sums must have equal shape")
    return later_sum.float() - earlier_sum.float()


def validate_influence_config(config: Mapping) -> None:
    if config.get("tier") != "phase4-development":
        raise RuntimeError("prompt influence is development sensitivity only")
    if config.get("canonical_lens_unchanged") is not True:
        raise RuntimeError("canonical n=120 lens may never be silently changed")
    prompt = config["prompt"]
    if int(prompt["one_based_index"]) != int(prompt["zero_based_index"]) + 1:
        raise RuntimeError("prompt index conventions disagree")
    adjacent = config["adjacent_checkpoint_contract"]
    earlier = int(adjacent["earlier"]["n"])
    later = int(adjacent["later"]["n"])
    indices = [int(value) for value in adjacent["prompt_indices_one_based"]]
    if later - earlier != len(indices) or indices != list(
            range(earlier + 1, later + 1)):
        raise RuntimeError("adjacent checkpoint prompt indices are not exact")


def checkpoint_contract(path: Path, *, expected_n: int,
                        expected_layers: list[int],
                        expected_sha256: str) -> dict:
    if len(expected_sha256) != 64:
        raise RuntimeError("checkpoint lacks a full SHA-256")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"checkpoint hash mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = {
        "jacobian_sum", "n_done", "next_idx", "source_layers",
        "target_layer", "skip_first",
    }
    if set(state) != required:
        raise RuntimeError(f"unexpected checkpoint keys: {sorted(state)}")
    if int(state["n_done"]) != expected_n or int(state["next_idx"]) != expected_n:
        raise RuntimeError("checkpoint prompt count mismatch")
    if list(state["source_layers"]) != expected_layers \
            or sorted(state["jacobian_sum"]) != expected_layers:
        raise RuntimeError("checkpoint layer order/set mismatch")
    return state


def preferred_or_materialized(specification: Mapping) -> Path:
    preferred = Path(specification["local_preferred_path"])
    expected = specification["sha256"]
    if preferred.exists() and file_sha256(preferred) == expected:
        return preferred
    return materialize_local_file(
        specification["uri"], expected_sha256=expected)


def compare_adjacent_contract(
        earlier: Mapping, later: Mapping,
        contribution_sum: Mapping[int, torch.Tensor], *,
        source_layers: list[int], atol: float, rtol: float,
) -> tuple[list[dict], bool]:
    rows = []
    passed = True
    for layer in source_layers:
        observed = adjacent_sum_from_running_sums(
            earlier["jacobian_sum"][layer], later["jacobian_sum"][layer])
        expected = contribution_sum[layer].float()
        difference = observed - expected
        observed_norm = torch.linalg.vector_norm(observed)
        difference_norm = torch.linalg.vector_norm(difference)
        maximum = float(difference.abs().max().item())
        relative = float((difference_norm / observed_norm).item())
        layer_pass = bool(torch.allclose(
            observed, expected, atol=atol, rtol=rtol))
        passed = passed and layer_pass
        rows.append({
            "layer": layer,
            "allclose_pass": layer_pass,
            "maximum_absolute_error": maximum,
            "relative_frobenius_error": relative,
            "observed_delta_frobenius": float(observed_norm.item()),
        })
    return rows, passed


def save_lens_file(path: Path, jacobians: Mapping[int, torch.Tensor], *,
                   n_prompts: int, d_model: int,
                   source_layers: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save({
        "J": {
            layer: jacobians[layer].to(torch.float16)
            for layer in source_layers
        },
        "n_prompts": int(n_prompts),
        "source_layers": list(source_layers),
        "d_model": int(d_model),
    }, temporary)
    os.replace(temporary, path)


def plot_influence(rows: list[dict], *, assay_band: tuple[int, int],
                   landmark_layers: list[int], png_path: Path,
                   pdf_path: Path) -> None:
    frame = pd.DataFrame(rows).sort_values("layer")
    x = frame["layer"]
    colors = {
        "raw": "#0072B2",
        "residual": "#009E73",
        "answer": "#CC79A7",
        "bridge": "#E69F00",
        "uniform": "#555555",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.7, 7.2), sharex=True)
    axis = axes[0, 0]
    axis.plot(x, frame["raw_matrix_cosine"], color=colors["raw"],
              lw=1.7, label="raw J")
    axis.plot(x, frame["minus_alpha_identity_matrix_cosine"],
              color=colors["residual"], lw=1.7, label="J − alpha I")
    axis.set_ylabel("matrix cosine")
    axis.set_title("A · Full A120 vs leave-prompt-112-out", loc="left")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    for name, label, color, style in (
        ("task_answer_only", "answer-only", colors["answer"], "-"),
        ("task_bridge_only", "bridge-only", colors["bridge"], "-"),
        ("task_answer_bridge_shared", "shared", "#56B4E9", "-"),
        ("uniform4096", "uniform 4,096", colors["uniform"], "--"),
    ):
        axis.plot(
            x, frame[f"token_{name}_direction_cosine_q50"],
            color=color, linestyle=style, lw=1.5, label=label)
    axis.set_ylabel("token-row cosine (median)")
    axis.set_title("B · Task rows lead", loc="left")
    axis.legend(frameon=False, fontsize=7, ncol=2)

    axis = axes[1, 0]
    axis.plot(x, frame["raw_symmetric_relative_delta"],
              color=colors["raw"], lw=1.6, label="raw J")
    axis.plot(x, frame["minus_identity_symmetric_relative_delta"],
              color="#D55E00", lw=1.5, label="J − I")
    axis.plot(x, frame["minus_alpha_identity_symmetric_relative_delta"],
              color=colors["residual"], lw=1.6, label="J − alpha I")
    axis.set_ylabel("symmetric relative delta")
    axis.set_xlabel("source layer")
    axis.set_title("C · Identity-adjusted movement", loc="left")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    axis.plot(
        x, 1 - frame["token_task_answer_only_direction_cosine_q05"],
        color=colors["answer"], lw=1.5, label="answer q05 disagreement")
    axis.plot(
        x, 1 - frame["token_task_bridge_only_direction_cosine_q05"],
        color=colors["bridge"], lw=1.5, label="bridge q05 disagreement")
    axis.plot(
        x, 1 - frame["token_uniform4096_direction_cosine_q05"],
        color=colors["uniform"], linestyle="--", lw=1.4,
        label="uniform q05 disagreement")
    axis.set_ylabel("1 − q05 cosine")
    axis.set_xlabel("source layer")
    axis.set_title("D · Lower-tail token sensitivity", loc="left")
    axis.legend(frameon=False, fontsize=7)

    start, stop = assay_band
    for axis in axes.ravel():
        axis.axvspan(start - 0.5, stop + 0.5, color="#F0E442", alpha=0.10)
        for layer in landmark_layers:
            axis.axvline(layer, color="#222222", alpha=0.22,
                         linestyle=":", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Qwen3.6-27B retained prompt-112 influence sensitivity",
        x=0.075, ha="left", fontsize=13)
    figure.text(
        0.075, 0.012,
        "Development sensitivity only; the canonical A120 lens is unchanged. "
        "Yellow shading marks the frozen L20–L44 assay band.",
        fontsize=8, color="#555555")
    figure.tight_layout(rect=(0.04, 0.05, 0.99, 0.94))
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)


def main() -> None:  # noqa: C901, PLR0915
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    validate_influence_config(config)
    clean = require_clean_tree()
    try:
        existing = resolve(config["evidence_id"])
    except RegistryError as error:
        if "found 0" not in str(error):
            raise
    else:
        if not existing["live"]:
            raise RuntimeError("existing prompt-influence evidence is not live")
        for output in existing["outputs"]:
            if file_sha256(output["path"]) != output["sha256"]:
                raise RuntimeError("existing prompt-influence output mismatch")
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return

    fit_config_path = resolve_uri(config["fit_config_uri"])
    if file_sha256(fit_config_path) != config["fit_config_sha256"]:
        raise RuntimeError("frozen fit-config hash mismatch")
    fit_config = yaml.safe_load(fit_config_path.read_text())
    fit_producer = Path(__file__).with_name("p4_qwen_nested_lens_fit.py")
    if file_sha256(fit_producer) != config["fit_producer_sha256"]:
        raise RuntimeError("frozen fit-producer hash mismatch")
    recipe = fit_config["recipe"]
    source_layers = list(range(int(recipe["target_layer"])))
    analysis = config["analysis"]
    if (
        len(source_layers) != int(analysis["expected_source_layers"])
        or int(recipe["expected_d_model"]) != int(analysis["expected_d_model"])
    ):
        raise RuntimeError("fit and influence shape contracts disagree")

    canonical_spec = config["canonical_lens"]
    canonical_event = resolve(canonical_spec["evidence_id"])
    if not canonical_event["live"]:
        raise RuntimeError("canonical A120 lens is not live")
    canonical_source = resolve_uri(canonical_spec["lens_uri"])
    registered_hashes = {
        output["sha256"] for output in canonical_event["outputs"]
        if Path(output["path"]) == canonical_source
    }
    if registered_hashes != {canonical_spec["lens_sha256"]}:
        raise RuntimeError("canonical lens path/hash is not registered")
    canonical_path = materialize_local_file(
        canonical_spec["lens_uri"],
        expected_sha256=canonical_spec["lens_sha256"])
    lens_recipe = {
        "expected_source_layers": len(source_layers),
        "expected_d_model": int(recipe["expected_d_model"]),
    }
    canonical = load_lens_checkpoint(
        canonical_path, canonical_spec, lens_recipe)

    adjacent = config["adjacent_checkpoint_contract"]
    earlier_path = preferred_or_materialized(adjacent["earlier"])
    later_path = preferred_or_materialized(adjacent["later"])
    earlier = checkpoint_contract(
        earlier_path,
        expected_n=int(adjacent["earlier"]["n"]),
        expected_layers=source_layers,
        expected_sha256=adjacent["earlier"]["sha256"],
    )
    later = checkpoint_contract(
        later_path,
        expected_n=int(adjacent["later"]["n"]),
        expected_layers=source_layers,
        expected_sha256=adjacent["later"]["sha256"],
    )
    if (
        int(earlier["target_layer"]) != int(recipe["target_layer"])
        or int(later["target_layer"]) != int(recipe["target_layer"])
        or int(earlier["skip_first"]) != int(recipe["skip_first"])
        or int(later["skip_first"]) != int(recipe["skip_first"])
    ):
        raise RuntimeError("adjacent checkpoints disagree with fit estimator")

    gpu = require_cuda_gpu()
    runtime = {
        "packages": verify_package_versions(fit_config["runtime"]["packages"]),
        "qwen_kernels": qwen_fused_kernel_contract(fit_config["runtime"]),
    }
    model_path = resolve_uri(config["model_uri"])
    snapshot_manifest_path = resolve_uri(config["model_snapshot_manifest_uri"])
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text())
    model_snapshot = verify_snapshot(model_path, snapshot_manifest)
    jlens_contract, _ = jlens_source_contract(fit_config["jlens"])

    draw_specification = fit_config["draws"]["draw_a"]
    corpus_event = resolve(fit_config["corpus_evidence_id"])
    corpus_path = resolve_uri(draw_specification["corpus_uri"])
    corpus_sha = file_sha256(corpus_path)
    if not corpus_event["live"] or corpus_sha != draw_specification["corpus_sha256"]:
        raise RuntimeError("draw-A corpus evidence/hash mismatch")
    rows = load_rows(corpus_path)
    prompt_index = int(config["prompt"]["zero_based_index"])
    adjacent_indices = [
        int(value) - 1 for value in adjacent["prompt_indices_one_based"]]
    if max([prompt_index, *adjacent_indices]) >= len(rows):
        raise RuntimeError("requested prompt index exceeds frozen corpus")

    import jlens
    import transformers
    from jlens.fitting import jacobian_for_prompt

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to("cuda").eval()
    assert_model_on_cuda(hf_model)
    binding_contract = verify_model_fused_bindings(
        hf_model, fit_config["runtime"])
    lens_model = jlens.from_hf(hf_model, tokenizer)
    if (
        lens_model.n_layers != int(recipe["expected_n_layers"])
        or lens_model.d_model != int(recipe["expected_d_model"])
    ):
        raise RuntimeError("loaded model disagrees with frozen fit shape")

    invocation_started = time.time()
    prompt_text = rows[prompt_index]["text"]
    prompt_contribution, prompt_seq_len, prompt_n_valid = jacobian_for_prompt(
        lens_model,
        prompt_text,
        source_layers,
        target_layer=int(recipe["target_layer"]),
        dim_batch=int(recipe["dim_batch"]),
        max_seq_len=int(recipe["max_seq_len"]),
        skip_first=int(recipe["skip_first"]),
    )
    prompt_norms = {
        layer: float(torch.linalg.vector_norm(prompt_contribution[layer]).item())
        for layer in source_layers
    }
    observed_max_norm = (
        max(prompt_norms.values()) / math.sqrt(int(recipe["expected_d_model"])))
    logged_max_norm = float(
        config["prompt"]["logged_max_jacobian_norm_over_sqrt_d"])
    if abs(observed_max_norm - logged_max_norm) > 0.01:
        raise RuntimeError(
            "recomputed prompt-112 norm disagrees with the frozen fit log: "
            f"{observed_max_norm} versus {logged_max_norm}")

    local_output = local_work() / "qwen_lens_influence" / config["evidence_id"]
    ensure_free_space(local_output, needed_bytes=8_000_000_000, label="local")
    local_contribution = local_output / "qwen36-27b_prompt112_contribution.pt"
    local_minus = local_output / "qwen36-27b_jlens_drawA_n0120_minus_prompt112.pt"
    save_lens_file(
        local_contribution, prompt_contribution,
        n_prompts=1, d_model=int(recipe["expected_d_model"]),
        source_layers=source_layers,
    )
    minus_jacobians = {
        layer: leave_one_out_mean(
            canonical["J"][layer], prompt_contribution[layer],
            n=int(canonical_spec["n_prompts"]),
        )
        for layer in source_layers
    }
    save_lens_file(
        local_minus, minus_jacobians,
        n_prompts=int(canonical_spec["n_prompts"]) - 1,
        d_model=int(recipe["expected_d_model"]),
        source_layers=source_layers,
    )
    del prompt_contribution, minus_jacobians
    gc.collect()

    contribution_sum = None
    adjacent_prompt_metadata = []
    for zero_based in adjacent_indices:
        per_prompt, seq_len, n_valid = jacobian_for_prompt(
            lens_model,
            rows[zero_based]["text"],
            source_layers,
            target_layer=int(recipe["target_layer"]),
            dim_batch=int(recipe["dim_batch"]),
            max_seq_len=int(recipe["max_seq_len"]),
            skip_first=int(recipe["skip_first"]),
        )
        if contribution_sum is None:
            contribution_sum = per_prompt
        else:
            for layer in source_layers:
                contribution_sum[layer].add_(per_prompt[layer])
            del per_prompt
        adjacent_prompt_metadata.append({
            "one_based_index": zero_based + 1,
            "text_sha256": hashlib_sha256_text(rows[zero_based]["text"]),
            "seq_len": seq_len,
            "n_valid": n_valid,
        })
        gc.collect()
    assert contribution_sum is not None
    adjacent_rows, adjacent_pass = compare_adjacent_contract(
        earlier, later, contribution_sum,
        source_layers=source_layers,
        atol=float(adjacent["allclose_atol"]),
        rtol=float(adjacent["allclose_rtol"]),
    )
    del contribution_sum, earlier, later, lens_model, hf_model
    gc.collect()
    torch.cuda.empty_cache()
    if not adjacent_pass:
        raise RuntimeError(
            "adjacent-checkpoint equal-weight estimator contract failed")

    drive_lens_dir = (
        run_root() / "lens" / "qwen36-27b" / "influence" / "prompt112")
    drive_contribution = drive_lens_dir / local_contribution.name
    drive_minus = drive_lens_dir / local_minus.name
    contribution_sha = file_sha256(local_contribution)
    minus_sha = file_sha256(local_minus)
    copy_atomic_verified(
        local_contribution, drive_contribution,
        expected_sha256=contribution_sha)
    copy_atomic_verified(local_minus, drive_minus, expected_sha256=minus_sha)

    minus_spec = {"n_prompts": int(canonical_spec["n_prompts"]) - 1}
    minus_checkpoint = load_lens_checkpoint(
        local_minus, minus_spec, lens_recipe)
    sampling_spec = config["fixed_sampling"]
    uniform_ids, probes_cpu, sampling_contract = load_fixed_sampling_contract(
        resolve_uri(sampling_spec["source_manifest_uri"]),
        expected_file_sha256=sampling_spec["source_manifest_sha256"],
        expected_ids_sha256=sampling_spec["token_ids_sha256"],
        expected_probes_sha256=sampling_spec["packed_probes_sha256"],
        expected_token_n=int(sampling_spec["token_ids_n"]),
        expected_probe_shape=[int(value) for value in sampling_spec["probe_shape"]],
    )
    task_rows, bank_contract = _load_bank_rows(config["task_banks"])
    task_strata, task_contract = build_task_token_strata(tokenizer, task_rows)
    strata = {**task_strata, "uniform4096": uniform_ids}
    all_token_ids = sorted(set().union(*(set(values) for values in strata.values())))
    token_index = {token_id: index for index, token_id in enumerate(all_token_ids)}
    indices = {
        name: torch.tensor(
            [token_index[value] for value in values],
            device="cuda", dtype=torch.int64)
        for name, values in strata.items()
    }
    sampled_unembedding, norm_weight, tensor_contract = (
        verified_model_tensor_sample(
            model_path=model_path,
            manifest=snapshot_manifest,
            tensor_names=config["model_tensors"],
            token_ids=all_token_ids,
            expected_vocab_size=int(analysis["expected_vocab_size"]),
            expected_d_model=int(analysis["expected_d_model"]),
        )
    )
    gain, gain_contract = effective_gain_on_cuda(
        norm_weight,
        d_model=int(analysis["expected_d_model"]),
        eps=float(config["model_tensors"]["rms_norm_eps"]),
    )
    base_rows = sampled_unembedding.to("cuda", dtype=torch.float32) * gain[None, :]
    probes = probes_cpu.to("cuda", dtype=torch.float32)
    quantiles = [float(value) for value in analysis["quantiles"]]
    structural_rows = []
    for layer in source_layers:
        full = canonical["J"][layer].to("cuda", dtype=torch.float32)
        minus = minus_checkpoint["J"][layer].to("cuda", dtype=torch.float32)
        full_views, full_identity = identity_views(full)
        minus_views, minus_identity = identity_views(minus)
        row = {
            "layer": layer,
            "full_identity_scale_alpha":
                full_identity["identity_scale_alpha"],
            "minus112_identity_scale_alpha":
                minus_identity["identity_scale_alpha"],
        }
        for view in ("raw", "minus_identity", "minus_alpha_identity"):
            metrics = operator_pair_metrics(
                full_views[view], minus_views[view],
                probes=probes, quantiles=quantiles)
            row.update({f"{view}_{key}": value
                        for key, value in metrics.items()})
        full_token = base_rows @ full
        minus_token = base_rows @ minus
        for name, stratum_indices in indices.items():
            metrics = token_pair_metrics(
                full_token.index_select(0, stratum_indices),
                minus_token.index_select(0, stratum_indices),
                quantiles=quantiles,
                cka_n=min(int(analysis["cka_max_n"]), len(strata[name])),
            )
            row.update({f"token_{name}_{key}": value
                        for key, value in metrics.items()})
        structural_rows.append(row)
        print(json.dumps({
            "layer": layer,
            "raw_cosine": round(row["raw_matrix_cosine"], 7),
            "residual_cosine": round(
                row["minus_alpha_identity_matrix_cosine"], 7),
            "task_answer_cosine": round(
                row["token_task_answer_only_direction_cosine_q50"], 7),
        }), flush=True)
        del full, minus, full_views, minus_views, full_token, minus_token

    numeric_columns = [column for column in structural_rows[0] if column != "layer"]
    assay_start, assay_stop = [int(value) for value in analysis["assay_band"]]
    assay_rows = [row for row in structural_rows
                  if assay_start <= row["layer"] <= assay_stop]
    aggregate = {
        "all_layers": aggregate_numeric(structural_rows, numeric_columns),
        f"assay_L{assay_start}_L{assay_stop}":
            aggregate_numeric(assay_rows, numeric_columns),
    }
    task_names = [
        "task_answer_only", "task_bridge_only",
        "task_answer_bridge_shared",
    ]
    median_disagreement = max(
        1 - float(np.median([
            row[f"token_{name}_direction_cosine_q50"] for row in assay_rows]))
        for name in task_names
    )
    q05_disagreement = max(
        1 - float(np.median([
            row[f"token_{name}_direction_cosine_q05"] for row in assay_rows]))
        for name in task_names
    )
    residual_disagreement = 1 - float(np.median([
        row["minus_alpha_identity_matrix_cosine"] for row in assay_rows]))
    thresholds = analysis["material_thresholds"]
    structural_material = (
        median_disagreement
        > float(thresholds["assay_task_token_median_disagreement"])
        or q05_disagreement
        > float(thresholds["assay_task_token_q05_disagreement"])
        or residual_disagreement
        > float(thresholds[
            "assay_identity_adjusted_matrix_disagreement"])
    )

    output_dir = (
        metrics_dir(config["slug"]) / "prompt_influence"
        / config["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    structural_path = output_dir / "layer_influence_metrics.csv"
    adjacent_path = output_dir / "adjacent_checkpoint_contract.csv"
    result_path = output_dir / "influence_result.json"
    manifest_path = output_dir / "input_manifest.json"
    png_path = figures_dir() / "p4f12_qwen_prompt112_influence.png"
    pdf_path = figures_dir() / "p4f12_qwen_prompt112_influence.pdf"
    pd.DataFrame(structural_rows).to_csv(structural_path, index=False)
    pd.DataFrame(adjacent_rows).to_csv(adjacent_path, index=False)
    plot_influence(
        structural_rows,
        assay_band=(assay_start, assay_stop),
        landmark_layers=[int(value) for value in analysis["landmark_layers"]],
        png_path=png_path,
        pdf_path=pdf_path,
    )

    input_payload = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "canonical_lens": {
            **dict(canonical_spec),
            "materialized_path": str(canonical_path),
        },
        "canonical_lens_unchanged": True,
        "prompt": {
            **dict(config["prompt"]),
            "text_sha256": hashlib_sha256_text(prompt_text),
            "seq_len": prompt_seq_len,
            "n_valid": prompt_n_valid,
        },
        "adjacent_checkpoint_contract": {
            **dict(adjacent),
            "earlier_materialized_path": str(earlier_path),
            "later_materialized_path": str(later_path),
            "prompt_metadata": adjacent_prompt_metadata,
        },
        "fit_config_sha256": config["fit_config_sha256"],
        "fit_producer_sha256": config["fit_producer_sha256"],
        "model": model_reference(config["model_uri"]),
        "model_snapshot": model_snapshot,
        "jlens": jlens_contract,
        "runtime": runtime,
        "model_fused_kernel_bindings": binding_contract,
        "gpu": gpu,
        "fixed_sampling": sampling_contract,
        "task_bank_inputs": bank_contract,
        "task_token_contract": task_contract,
        "model_tensor_contract": tensor_contract,
        "effective_gain": gain_contract,
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
        "canonical_lens_unchanged": True,
        "sensitivity": "full A120 versus algebraic leave-prompt-112-out",
        "registered_lens_quantization_note": (
            "The canonical registered A120 matrix is fp16; the exact fp32 "
            "per-prompt contribution and leave-one-out arithmetic are applied "
            "to that immutable stored representation, then the sensitivity "
            "lens is saved fp16."),
        "prompt_contribution": {
            "one_based_index": int(config["prompt"]["one_based_index"]),
            "seq_len": prompt_seq_len,
            "n_valid": prompt_n_valid,
            "max_jacobian_norm_over_sqrt_d": observed_max_norm,
            "layer_norms_sha256": object_sha256(prompt_norms),
            "stored_fp16_sha256": contribution_sha,
        },
        "minus112_lens": {
            "n_prompts": int(canonical_spec["n_prompts"]) - 1,
            "sha256": minus_sha,
            "canonical_replaced": False,
        },
        "adjacent_checkpoint_equal_weight_contract": {
            "earlier_n": int(adjacent["earlier"]["n"]),
            "later_n": int(adjacent["later"]["n"]),
            "prompt_indices_one_based": adjacent[
                "prompt_indices_one_based"],
            "all_layers_pass": adjacent_pass,
            "max_absolute_error": max(
                row["maximum_absolute_error"] for row in adjacent_rows),
            "max_relative_frobenius_error": max(
                row["relative_frobenius_error"] for row in adjacent_rows),
            "table_sha256": file_sha256(adjacent_path),
        },
        "aggregate": aggregate,
        "materiality": {
            "assay_task_token_median_disagreement": median_disagreement,
            "assay_task_token_q05_disagreement": q05_disagreement,
            "assay_identity_adjusted_matrix_disagreement":
                residual_disagreement,
            "structurally_material": structural_material,
            "thresholds": dict(thresholds),
            "selected_id_and_capacity_checks":
                "pending fixed multi-lens functional gate",
        },
        "elapsed_s": round(time.time() - invocation_started, 1),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": gpu,
        "limitations": [
            "This is a development sensitivity, never a trimming rule; prompt "
            "112 remains in the canonical A120 lens.",
            "Selected-ID, selected-span, capacity, and causal consequences are "
            "not licensed by this structural producer.",
            "The immutable registered A120 lens is stored fp16; no unavailable "
            "historical fp32 n=120 checkpoint is reconstructed or guessed.",
        ],
    }
    command = (
        "python -m jspace_phase4.experiments.p4_qwen_lens_influence "
        f"--config {arguments.config}")
    inputs = {
        "canonical_lens": canonical_spec["lens_sha256"],
        "canonical_lens_evidence": canonical_spec["evidence_id"],
        "draw_a_corpus": corpus_sha,
        "model_snapshot_inventory": model_snapshot["inventory_sha256"],
        "fit_config": config["fit_config_sha256"],
        "fit_producer": config["fit_producer_sha256"],
        "adjacent_checkpoint_earlier": adjacent["earlier"]["sha256"],
        "adjacent_checkpoint_later": adjacent["later"]["sha256"],
        "fixed_token_ids": sampling_spec["token_ids_sha256"],
        "fixed_rademacher_probes": sampling_spec["packed_probes_sha256"],
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
            seed_contract=(
                "exact frozen prompt indices; equal-prompt running mean; "
                "fixed p4f09 token/probe sample"),
        ),
    )
    create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "Exact retained prompt-112 leave-one-out sensitivity with tiny-test "
            "and adjacent n=195/n=198 checkpoint verification of the "
            "equal-prompt estimator contract."),
        command=command,
        outputs=[
            drive_contribution, drive_minus,
            resolve_uri(adjacent["earlier"]["uri"]),
            resolve_uri(adjacent["later"]["uri"]),
            result_path, manifest_path, structural_path, adjacent_path,
            png_path, pdf_path,
        ],
        inputs=inputs,
    )
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "prompt_contribution": str(drive_contribution),
        "minus112_lens": str(drive_minus),
        "adjacent_contract_pass": adjacent_pass,
        "structurally_material": structural_material,
        "result": str(result_path),
        "figure_png": str(png_path),
        "gpu": gpu["name"],
    }, indent=1))


def hashlib_sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
