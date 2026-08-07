"""Paired exact retained-prompt influence for Qwen draw-A A500/A1000."""
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
    verify_distribution_content_inventories,
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
from .p4_qwen_lens_influence import (
    hashlib_sha256_text,
    leave_one_out_mean,
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


PAIR = ("a500", "a1000")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def validate_paired_config(config: Mapping) -> None:
    if config.get("tier") != "phase4-development":
        raise RuntimeError("paired prompt influence is development only")
    if config.get("canonical_lens_unchanged") is not True:
        raise RuntimeError("paired influence may never replace a lens")
    prompt = config["prompt"]
    if int(prompt["one_based_index"]) != int(prompt["zero_based_index"]) + 1:
        raise RuntimeError("prompt index conventions disagree")
    if prompt.get("retained_unconditionally") is not True:
        raise RuntimeError("prompt 323 must be retained unconditionally")
    historical_tolerance = float(
        prompt["historical_reference_absolute_tolerance"])
    repeat_tolerance = float(
        prompt["current_runtime_repeat_absolute_tolerance"])
    if historical_tolerance != 0.5 or repeat_tolerance != 0.5:
        raise RuntimeError("prompt-323 runtime-control tolerance drift")
    amendment = config["runtime_amendment"]
    if amendment.get("contract_version") != "current-runtime-shape-v1":
        raise RuntimeError("prompt-323 runtime amendment version drift")
    if amendment.get("phase_branch_decision_critical") is not False \
            or amendment.get("historical_runtime_reproduction_claimed") \
            is not False:
        raise RuntimeError("prompt-323 runtime claim exceeds its scope")
    if int(amendment.get("primary_computation_ordinal", -1)) != 1 \
            or int(amendment.get("repeat_computation_ordinal", -1)) != 2 \
            or amendment.get("repeat_computation_role") \
            != "diagnostic-only-discarded":
        raise RuntimeError("prompt-323 computation selection drift")
    distributions = amendment.get("exact_distribution_content_inventories")
    if not isinstance(distributions, list) or {
            row.get("distribution") for row in distributions} != {
                "fla-core", "flash-linear-attention", "transformers",
                "triton", "torch"}:
        raise RuntimeError("prompt-323 distribution-content lock drift")
    if set(config["lenses"]) != set(PAIR):
        raise RuntimeError("paired influence requires exactly A500/A1000")
    for lens in PAIR:
        specification = config["lenses"][lens]
        expected_n = int(lens[1:])
        if int(specification["n_prompts"]) != expected_n:
            raise RuntimeError(f"{lens} prompt count drift")
        digest = str(specification["lens_sha256"])
        if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"{lens} lacks a registered full SHA-256")
    formulas = config["analysis"]["leave_one_out_formulas"]
    if formulas != {
            "a500": "(500 * J_a500 - J_prompt323) / 499",
            "a1000": "(1000 * J_a1000 - J_prompt323) / 999"}:
        raise RuntimeError("paired leave-one-out formula drift")
    contract = config["equal_weight_contract"]
    if contract.get("tiny_model_direct_refit_required") is not True \
            or contract.get("adjacent_atomic_checkpoint_required") is not True:
        raise RuntimeError("equal-weight assertion was weakened")
    earlier = int(contract["earlier_checkpoint"]["n"])
    later = int(contract["later_checkpoint"]["n"])
    indices = [int(value) for value in contract[
        "prompt_indices_one_based"]]
    if later - earlier != len(indices) or indices != list(
            range(earlier + 1, later + 1)):
        raise RuntimeError("adjacent checkpoint block drift")
    load_bearing = set(config["analysis"]["decision_load_bearing_metrics"])
    if load_bearing != {
            "assay_task_token_median_disagreement",
            "assay_task_token_q05_disagreement",
            "assay_identity_adjusted_matrix_disagreement"}:
        raise RuntimeError("paired influence decision metric drift")


def tiny_direct_refit_contract() -> dict:
    generator = torch.Generator().manual_seed(323)
    contributions = torch.randn((7, 5, 5), generator=generator)
    full = contributions.mean(dim=0)
    recovered = leave_one_out_mean(full, contributions[3], n=7)
    direct = torch.cat((contributions[:3], contributions[4:])).mean(dim=0)
    maximum = float((recovered - direct).abs().max().item())
    return {
        "seed": 323,
        "n_prompts": 7,
        "removed_zero_based_index": 3,
        "maximum_absolute_error": maximum,
        "allclose_atol": 1e-6,
        "pass": bool(torch.allclose(recovered, direct, atol=1e-6, rtol=0)),
    }


def classify_materiality(by_lens: Mapping[str, Mapping],
                         thresholds: Mapping) -> str:
    def material(lens: str) -> bool:
        metrics = by_lens[lens]
        return any(
            float(metrics[name]) > float(thresholds[name])
            for name in (
                "assay_task_token_median_disagreement",
                "assay_task_token_q05_disagreement",
                "assay_identity_adjusted_matrix_disagreement"))
    if material("a1000"):
        return "material_at_a1000"
    if material("a500"):
        return "material_small_fit_only"
    return "negligible"


def _event_output(event: Mapping, *, name: str | None = None,
                  sha256: str | None = None) -> Path:
    matches = [row for row in event["outputs"]
               if (name is None or Path(row["path"]).name == name)
               and (sha256 is None or row["sha256"] == sha256)]
    if len(matches) != 1:
        raise RuntimeError(
            f"event {event['evidence_id']} lacks one requested output")
    row = matches[0]
    path = Path(row["path"])
    if file_sha256(path) != row["sha256"]:
        raise RuntimeError(f"registered source output hash mismatch: {path}")
    return path


def registered_equal_weight_assertion(config: Mapping) -> dict:
    contract = config["equal_weight_contract"]
    source = contract["registered_assertion_source"]
    event = resolve(source["evidence_id"])
    if not event["live"]:
        raise RuntimeError("registered equal-weight assertion is not live")
    result_path = _event_output(
        event, name="influence_result.json",
        sha256=source["result_sha256"])
    table_path = _event_output(
        event, name="adjacent_checkpoint_contract.csv",
        sha256=source["table_sha256"])
    result = json.loads(result_path.read_text())["payload"]
    assertion = result["adjacent_checkpoint_equal_weight_contract"]
    if assertion.get("all_layers_pass") is not True:
        raise RuntimeError("registered adjacent equal-weight assertion failed")
    for key in ("earlier_checkpoint", "later_checkpoint"):
        specification = contract[key]
        _event_output(event, sha256=specification["sha256"])
    return {
        "source_evidence_id": source["evidence_id"],
        "source_result_sha256": file_sha256(result_path),
        "source_table_sha256": file_sha256(table_path),
        "earlier_n": assertion["earlier_n"],
        "later_n": assertion["later_n"],
        "prompt_indices_one_based": assertion["prompt_indices_one_based"],
        "all_layers_pass": True,
        "max_running_mean_relative_frobenius_error": assertion[
            "max_running_mean_relative_frobenius_error"],
        "role": (
            "registered global equal-prompt estimator assertion; the same "
            "immutable fitter/checkpoint contract applies to prompt 323"),
    }


def _resolve_lenses(config: Mapping, recipe: Mapping) -> tuple[dict, dict]:
    paths, checkpoints = {}, {}
    for lens in PAIR:
        specification = config["lenses"][lens]
        event = resolve(specification["evidence_id"])
        if not event["live"]:
            raise RuntimeError(f"registered lens is not live: {lens}")
        logical = resolve_uri(specification["lens_uri"], must_exist=False)
        registered = {
            row["sha256"] for row in event["outputs"]
            if Path(row["path"]) == logical}
        if registered != {specification["lens_sha256"]}:
            raise RuntimeError(f"registered lens binding mismatch: {lens}")
        path = materialize_local_file(
            specification["lens_uri"],
            expected_sha256=specification["lens_sha256"])
        paths[lens] = path
        checkpoints[lens] = load_lens_checkpoint(
            path, specification, recipe)
    return paths, checkpoints


def _atomic_contribution(path: Path, jacobians: Mapping[int, torch.Tensor],
                         *, d_model: int, source_layers: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save({
        "schema_version": 1,
        "J": {layer: jacobians[layer].float().cpu()
              for layer in source_layers},
        "d_model": int(d_model),
        "source_layers": list(source_layers),
        "n_prompts": 1,
        "dtype": "float32",
    }, temporary)
    os.replace(temporary, path)


def _load_contribution(path: Path, *, d_model: int,
                       source_layers: list[int]) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if sorted(value) != [
            "J", "d_model", "dtype", "n_prompts", "schema_version",
            "source_layers"]:
        raise RuntimeError("prompt-323 contribution schema drift")
    if int(value["n_prompts"]) != 1 or int(value["d_model"]) != d_model \
            or value["dtype"] != "float32" \
            or list(value["source_layers"]) != source_layers:
        raise RuntimeError("prompt-323 contribution metadata drift")
    for layer in source_layers:
        tensor = value["J"][layer]
        if tensor.dtype != torch.float32 or tensor.shape != (d_model, d_model):
            raise RuntimeError(f"prompt-323 contribution tensor drift L{layer}")
    return value


def contribution_repeatability(
        primary: Mapping[int, torch.Tensor],
        repeat: Mapping[int, torch.Tensor], *,
        primary_seq_len: int, repeat_seq_len: int,
        primary_n_valid: int, repeat_n_valid: int,
        d_model: int, source_layers: list[int],
        absolute_tolerance: float) -> dict:
    """Check a discarded repeat without selecting or averaging estimates."""
    primary_norms = {}
    repeat_norms = {}
    layer_differences = {}
    all_finite = True
    for layer in source_layers:
        first = primary[layer]
        second = repeat[layer]
        if first.shape != (d_model, d_model) \
                or second.shape != (d_model, d_model):
            raise RuntimeError(f"prompt-323 repeat shape drift L{layer}")
        first_finite = bool(torch.isfinite(first).all().item())
        second_finite = bool(torch.isfinite(second).all().item())
        all_finite = all_finite and first_finite and second_finite
        first_norm = float(torch.linalg.vector_norm(first).item()) \
            / math.sqrt(d_model)
        second_norm = float(torch.linalg.vector_norm(second).item()) \
            / math.sqrt(d_model)
        primary_norms[layer] = first_norm
        repeat_norms[layer] = second_norm
        layer_differences[layer] = abs(first_norm - second_norm)
    maximum_difference = max(layer_differences.values())
    metadata_match = (
        int(primary_seq_len) == int(repeat_seq_len)
        and int(primary_n_valid) == int(repeat_n_valid))
    passed = bool(
        all_finite and metadata_match
        and maximum_difference <= float(absolute_tolerance))
    return {
        "pass": passed,
        "primary_computation_ordinal": 1,
        "repeat_computation_ordinal": 2,
        "repeat_computation_role": "diagnostic-only-discarded",
        "same_seq_len_and_n_valid": metadata_match,
        "all_tensors_finite": all_finite,
        "absolute_tolerance": float(absolute_tolerance),
        "maximum_layer_norm_over_sqrt_d_absolute_difference":
            maximum_difference,
        "primary_max_jacobian_norm_over_sqrt_d": max(
            primary_norms.values()),
        "repeat_max_jacobian_norm_over_sqrt_d": max(
            repeat_norms.values()),
        "primary_layer_norms_sha256": object_sha256(primary_norms),
        "repeat_layer_norms_sha256": object_sha256(repeat_norms),
        "layer_norm_absolute_differences_sha256": object_sha256(
            layer_differences),
    }


def _plot(rows: pd.DataFrame, *, config: Mapping,
          png_path: Path, pdf_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True)
    colors = {"a500": "#b6782c", "a1000": "#315a9a"}
    labels = {"a500": "A500", "a1000": "A1000"}
    for lens in PAIR:
        frame = rows[rows["lens"] == lens].sort_values("layer")
        x = frame["layer"]
        axes[0, 0].plot(
            x, 1 - frame["minus_alpha_identity_matrix_cosine"],
            color=colors[lens], label=labels[lens])
        axes[0, 1].plot(
            x, 1 - frame["token_task_answer_only_direction_cosine_q50"],
            color=colors[lens], label=labels[lens])
        axes[1, 0].plot(
            x, 1 - frame["token_task_bridge_only_direction_cosine_q05"],
            color=colors[lens], label=labels[lens])
        axes[1, 1].plot(
            x, frame["minus_alpha_identity_symmetric_relative_delta"],
            color=colors[lens], label=labels[lens])
    titles = [
        "A · Identity-adjusted matrix disagreement",
        "B · Answer-row median disagreement",
        "C · Bridge-row q05 disagreement",
        "D · Identity-adjusted relative delta",
    ]
    ylabels = ["1 − cosine", "1 − cosine", "1 − q05 cosine", "relative delta"]
    start, stop = [int(value) for value in config["analysis"]["assay_band"]]
    for axis, title, ylabel in zip(axes.ravel(), titles, ylabels):
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.axvspan(start - .5, stop + .5, color="#F0E442", alpha=.10)
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("source layer")
    axes[1, 1].set_xlabel("source layer")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(config["figure"]["title"], fontsize=13)
    figure.text(.5, .012, config["figure"]["footer"], ha="center",
                fontsize=8, color="#555555")
    figure.tight_layout(rect=(.02, .04, .99, .95))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=190, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def _materiality(rows: pd.DataFrame, config: Mapping) -> dict:
    analysis = config["analysis"]
    start, stop = [int(value) for value in analysis["assay_band"]]
    output = {}
    task_names = [
        "task_answer_only", "task_bridge_only",
        "task_answer_bridge_shared"]
    for lens in PAIR:
        assay = rows[
            (rows["lens"] == lens) & rows["layer"].between(start, stop)]
        output[lens] = {
            "assay_task_token_median_disagreement": max(
                1 - float(assay[
                    f"token_{name}_direction_cosine_q50"].median())
                for name in task_names),
            "assay_task_token_q05_disagreement": max(
                1 - float(assay[
                    f"token_{name}_direction_cosine_q05"].median())
                for name in task_names),
            "assay_identity_adjusted_matrix_disagreement": (
                1 - float(assay[
                    "minus_alpha_identity_matrix_cosine"].median())),
        }
    decision = classify_materiality(
        output, analysis["material_thresholds"])
    for lens in PAIR:
        output[lens]["material"] = bool(any(
            output[lens][name] > float(analysis["material_thresholds"][name])
            for name in analysis["decision_load_bearing_metrics"]))
    return {"by_lens": output, "decision": decision}


@torch.no_grad()
def main() -> None:  # noqa: C901, PLR0915
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    validate_paired_config(config)
    clean = require_clean_tree()
    try:
        existing = resolve(config["evidence_id"])
    except RegistryError as error:
        if "found 0" not in str(error):
            raise
    else:
        if not existing["live"]:
            raise RuntimeError("existing prompt-323 evidence is not live")
        for output in existing["outputs"]:
            if file_sha256(output["path"]) != output["sha256"]:
                raise RuntimeError("registered prompt-323 output drift")
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return

    fit_config_path = resolve_uri(config["fit_config_uri"])
    if file_sha256(fit_config_path) != config["fit_config_sha256"]:
        raise RuntimeError("paired influence fit-config hash mismatch")
    fit_config = yaml.safe_load(fit_config_path.read_text())
    fit_producer = Path(__file__).with_name("p4_qwen_nested_lens_fit.py")
    if file_sha256(fit_producer) != config["fit_producer_sha256"]:
        raise RuntimeError("paired influence fit-producer hash mismatch")
    recipe = fit_config["recipe"]
    source_layers = list(range(int(recipe["target_layer"])))
    d_model = int(recipe["expected_d_model"])
    lens_recipe = {
        "expected_source_layers": len(source_layers),
        "expected_d_model": d_model,
    }
    lens_paths, lenses = _resolve_lenses(config, lens_recipe)
    tiny = tiny_direct_refit_contract()
    if not tiny["pass"]:
        raise RuntimeError("tiny direct-refit equal-weight contract failed")
    adjacent = registered_equal_weight_assertion(config)

    gpu = require_cuda_gpu()
    runtime = {
        "packages": verify_package_versions(fit_config["runtime"]["packages"]),
        "qwen_kernels": qwen_fused_kernel_contract(fit_config["runtime"]),
    }
    amendment = config["runtime_amendment"]
    amendment_sources = {}
    for name in ("precommit", "incident_report", "incident_identity"):
        path = resolve_uri(amendment[f"{name}_uri"])
        expected_sha256 = amendment[f"{name}_sha256"]
        if file_sha256(path) != expected_sha256:
            raise RuntimeError(
                f"prompt-323 runtime-amendment source drift: {name}")
        amendment_sources[name] = {
            "path": str(path), "sha256": expected_sha256}
    distribution_lock = verify_distribution_content_inventories(
        amendment["exact_distribution_content_inventories"])
    runtime["distribution_content_lock"] = distribution_lock
    runtime["prompt323_amendment"] = {
        "contract_version": amendment["contract_version"],
        "historical_runtime_reproduction_claimed": False,
        "phase_branch_decision_critical": False,
        "sources": amendment_sources,
    }
    model_path = resolve_uri(config["model_uri"])
    snapshot_manifest_path = resolve_uri(config["model_snapshot_manifest_uri"])
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text())
    model_snapshot = verify_snapshot(model_path, snapshot_manifest)
    jlens_contract, _ = jlens_source_contract(fit_config["jlens"])
    draw = fit_config["draws"]["draw_a"]
    corpus_event = resolve(fit_config["corpus_evidence_id"])
    corpus_path = resolve_uri(draw["corpus_uri"])
    corpus_sha = file_sha256(corpus_path)
    if not corpus_event["live"] or corpus_sha != draw["corpus_sha256"]:
        raise RuntimeError("paired influence corpus binding mismatch")
    rows = load_rows(corpus_path)
    prompt_index = int(config["prompt"]["zero_based_index"])
    if prompt_index >= len(rows):
        raise RuntimeError("prompt 323 exceeds the frozen corpus")

    output_dir = (
        metrics_dir(config["slug"]) / "prompt_influence"
        / config["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "analysis_state.json"
    local_dir = local_work() / "qwen_lens_influence" / config["evidence_id"]
    ensure_free_space(local_dir, needed_bytes=8_000_000_000, label="local")
    local_contribution = local_dir / "qwen36-27b_prompt323_contribution_fp32.pt"
    drive_contribution = (
        run_root() / "lens" / "qwen36-27b" / "influence" / "prompt323"
        / local_contribution.name)
    header = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "lens_hashes": {lens: config["lenses"][lens]["lens_sha256"]
                        for lens in PAIR},
        "corpus_sha256": corpus_sha,
        "prompt_zero_based_index": prompt_index,
        "source_layers": source_layers,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("header") != header:
            raise RuntimeError("prompt-323 resume header mismatch")
    else:
        state = {"header": header, "contribution": None,
                 "completed_layers": []}
        atomic_json(state_path, state)

    prompt_text = rows[prompt_index]["text"]
    binding_contract = None
    prompt_metadata = None
    if state["contribution"] is None:
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
        log("computing preselected prompt-323 primary contribution")
        contribution, seq_len, n_valid = jacobian_for_prompt(
            lens_model, prompt_text, source_layers,
            target_layer=int(recipe["target_layer"]),
            dim_batch=int(recipe["dim_batch"]),
            max_seq_len=int(recipe["max_seq_len"]),
            skip_first=int(recipe["skip_first"]),
        )
        torch.manual_seed(0)
        log("computing diagnostic-only discarded prompt-323 repeat")
        repeat_contribution, repeat_seq_len, repeat_n_valid = (
            jacobian_for_prompt(
                lens_model, prompt_text, source_layers,
                target_layer=int(recipe["target_layer"]),
                dim_batch=int(recipe["dim_batch"]),
                max_seq_len=int(recipe["max_seq_len"]),
                skip_first=int(recipe["skip_first"]),
            ))
        repeatability = contribution_repeatability(
            contribution, repeat_contribution,
            primary_seq_len=seq_len, repeat_seq_len=repeat_seq_len,
            primary_n_valid=n_valid, repeat_n_valid=repeat_n_valid,
            d_model=d_model, source_layers=source_layers,
            absolute_tolerance=float(config["prompt"][
                "current_runtime_repeat_absolute_tolerance"]),
        )
        del repeat_contribution
        gc.collect()
        if not repeatability["pass"]:
            raise RuntimeError(
                "current-runtime prompt-323 repeatability gate failed: "
                + json.dumps(repeatability, sort_keys=True))
        maximum_repeat_difference = repeatability[
            "maximum_layer_norm_over_sqrt_d_absolute_difference"]
        log(
            "prompt-323 current-runtime repeatability passed; maximum "
            f"layer-norm difference {maximum_repeat_difference:.6f}")
        observed = repeatability[
            "primary_max_jacobian_norm_over_sqrt_d"]
        logged = float(config["prompt"][
            "logged_max_jacobian_norm_over_sqrt_d"])
        difference = abs(observed - logged)
        historical_tolerance = float(config["prompt"][
            "historical_reference_absolute_tolerance"])
        _atomic_contribution(
            local_contribution, contribution,
            d_model=d_model, source_layers=source_layers)
        contribution_sha = file_sha256(local_contribution)
        copy_atomic_verified(
            local_contribution, drive_contribution,
            expected_sha256=contribution_sha)
        prompt_metadata = {
            "text_sha256": hashlib_sha256_text(prompt_text),
            "seq_len": int(seq_len), "n_valid": int(n_valid),
            "max_jacobian_norm_over_sqrt_d": observed,
            "historical_logged_max_jacobian_norm_over_sqrt_d": logged,
            "historical_norm_absolute_difference": difference,
            "historical_reference_absolute_tolerance":
                historical_tolerance,
            "historical_reference_pass": bool(
                difference <= historical_tolerance),
            "historical_reference_role": "reported-non-gating",
            "runtime_claim": "current-runtime-sensitivity-shape-only",
            "repeatability": repeatability,
        }
        state["contribution"] = {
            "local_path": str(local_contribution),
            "drive_path": str(drive_contribution),
            "sha256": contribution_sha,
            "prompt_metadata": prompt_metadata,
            "model_fused_kernel_bindings": binding_contract,
        }
        atomic_json(state_path, state)
        del contribution, lens_model, hf_model
        gc.collect()
        torch.cuda.empty_cache()
        log("prompt-323 contribution durably banked")
    else:
        contribution_sha = state["contribution"]["sha256"]
        prompt_metadata = state["contribution"]["prompt_metadata"]
        binding_contract = state["contribution"][
            "model_fused_kernel_bindings"]
        if (not local_contribution.exists()
                or file_sha256(local_contribution) != contribution_sha):
            if file_sha256(drive_contribution) != contribution_sha:
                raise RuntimeError("durable prompt-323 contribution drift")
            copy_atomic_verified(
                drive_contribution, local_contribution,
                expected_sha256=contribution_sha)
    contribution = _load_contribution(
        local_contribution, d_model=d_model, source_layers=source_layers)

    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    sampling = config["fixed_sampling"]
    uniform_ids, probes_cpu, sampling_contract = load_fixed_sampling_contract(
        resolve_uri(sampling["source_manifest_uri"]),
        expected_file_sha256=sampling["source_manifest_sha256"],
        expected_ids_sha256=sampling["token_ids_sha256"],
        expected_probes_sha256=sampling["packed_probes_sha256"],
        expected_token_n=int(sampling["token_ids_n"]),
        expected_probe_shape=[int(value) for value in sampling["probe_shape"]],
    )
    task_rows, bank_contract = _load_bank_rows(config["task_banks"])
    task_strata, task_contract = build_task_token_strata(tokenizer, task_rows)
    strata = {**task_strata, "uniform4096": uniform_ids}
    all_token_ids = sorted(set().union(
        *(set(values) for values in strata.values())))
    token_index = {token_id: index for index, token_id in enumerate(all_token_ids)}
    indices = {
        name: torch.tensor(
            [token_index[value] for value in values],
            device="cuda", dtype=torch.long)
        for name, values in strata.items()}
    analysis = config["analysis"]
    sampled_unembedding, norm_weight, tensor_contract = (
        verified_model_tensor_sample(
            model_path=model_path, manifest=snapshot_manifest,
            tensor_names=config["model_tensors"], token_ids=all_token_ids,
            expected_vocab_size=int(analysis["expected_vocab_size"]),
            expected_d_model=d_model,
        ))
    gain, gain_contract = effective_gain_on_cuda(
        norm_weight, d_model=d_model,
        eps=float(config["model_tensors"]["rms_norm_eps"]))
    base_rows = sampled_unembedding.to("cuda", torch.float32) * gain[None, :]
    probes = probes_cpu.to("cuda", torch.float32)
    quantiles = [float(value) for value in analysis["quantiles"]]

    for layer in source_layers:
        part_path = parts_dir / f"influence_L{layer:02d}.parquet"
        if layer in state["completed_layers"]:
            if not part_path.exists():
                raise RuntimeError("completed influence layer lacks part")
            continue
        layer_rows = []
        contribution_layer = contribution["J"][layer].to(
            "cuda", torch.float32)
        for lens in PAIR:
            full = lenses[lens]["J"][layer].to("cuda", torch.float32)
            minus = leave_one_out_mean(
                full, contribution_layer,
                n=int(config["lenses"][lens]["n_prompts"]))
            full_views, full_identity = identity_views(full)
            minus_views, minus_identity = identity_views(minus)
            row = {
                "lens": lens, "layer": int(layer),
                "full_identity_scale_alpha": full_identity[
                    "identity_scale_alpha"],
                "minus_prompt323_identity_scale_alpha": minus_identity[
                    "identity_scale_alpha"],
            }
            for view in ("raw", "minus_identity", "minus_alpha_identity"):
                metrics = operator_pair_metrics(
                    full_views[view], minus_views[view], probes=probes,
                    quantiles=quantiles)
                row.update({f"{view}_{key}": value
                            for key, value in metrics.items()})
            full_token = base_rows @ full
            minus_token = base_rows @ minus
            for name, index in indices.items():
                metrics = token_pair_metrics(
                    full_token.index_select(0, index),
                    minus_token.index_select(0, index),
                    quantiles=quantiles,
                    cka_n=min(int(analysis["cka_max_n"]), len(strata[name])))
                row.update({f"token_{name}_{key}": value
                            for key, value in metrics.items()})
            layer_rows.append(row)
            del (full, minus, full_views, minus_views, full_token,
                 minus_token)
        atomic_parquet = pd.DataFrame(layer_rows)
        temporary = part_path.with_suffix(part_path.suffix + f".tmp{os.getpid()}")
        atomic_parquet.to_parquet(temporary, index=False)
        os.replace(temporary, part_path)
        state["completed_layers"].append(layer)
        state["completed_layers"].sort()
        atomic_json(state_path, state)
        del contribution_layer
        torch.cuda.empty_cache()
        log(f"prompt-323 paired influence L{layer} complete")

    part_paths = [parts_dir / f"influence_L{layer:02d}.parquet"
                  for layer in source_layers]
    structural = pd.concat(
        [pd.read_parquet(path) for path in part_paths], ignore_index=True)
    materiality = _materiality(structural, config)
    decision = materiality["decision"]
    structural_path = output_dir / "layer_influence_metrics.csv"
    assertion_path = output_dir / "equal_weight_contract.json"
    result_path = output_dir / "influence_result.json"
    manifest_path = output_dir / "input_manifest.json"
    png_path = figures_dir() / f"{config['figure']['stem']}.png"
    pdf_path = figures_dir() / f"{config['figure']['stem']}.pdf"
    structural.to_csv(structural_path, index=False)
    atomic_json(assertion_path, {
        "tiny_direct_refit": tiny,
        "registered_adjacent_checkpoint_assertion": adjacent,
        "all_contracts_pass": bool(tiny["pass"] and adjacent["all_layers_pass"]),
    })
    _plot(structural, config=config, png_path=png_path, pdf_path=pdf_path)

    aggregate = {}
    numeric = [column for column in structural.columns
               if column not in {"lens", "layer"}]
    start, stop = [int(value) for value in analysis["assay_band"]]
    for lens in PAIR:
        frame = structural[structural["lens"] == lens]
        assay = frame[frame["layer"].between(start, stop)]
        aggregate[lens] = {
            "all_layers": aggregate_numeric(
                frame.to_dict("records"), numeric),
            f"assay_L{start}_L{stop}": aggregate_numeric(
                assay.to_dict("records"), numeric),
        }
    payload = {
        "schema_version": 1,
        "canonical_lens_unchanged": True,
        "prompt_retained_unconditionally": True,
        "lens_hashes": {lens: config["lenses"][lens]["lens_sha256"]
                        for lens in PAIR},
        "sensitivity": (
            "paired registered A500/A1000 versus exact algebraic "
            "leave-prompt-323-out means"),
        "runtime_scope": "current-runtime-sensitivity-shape-only",
        "historical_runtime_reproduction_claimed": False,
        "runtime_amendment_precommit_sha256": amendment[
            "precommit_sha256"],
        "prompt_contribution": {
            **prompt_metadata,
            "stored_float32_sha256": contribution_sha,
        },
        "equal_weight_contract": {
            "tiny_direct_refit": tiny,
            "registered_adjacent_checkpoint_assertion": adjacent,
            "all_contracts_pass": True,
        },
        "aggregate": aggregate,
        "materiality": {
            **materiality,
            "thresholds": dict(analysis["material_thresholds"]),
            "decision_load_bearing_metrics": list(
                analysis["decision_load_bearing_metrics"]),
            "downstream_nondecision_diagnostics": list(
                analysis["downstream_nondecision_diagnostics"]),
            "downstream_note": analysis["downstream_note"],
        },
        "decision": decision,
        "decision_wording": analysis["decision_wording"][decision],
        "limitations": [
            "Prompt 323 remains in both registered fits under every verdict.",
            "The audit cannot authorize trimming, refitting, or a new fit size.",
            "Selected-row, selected-span, capacity, and causal stability are "
            "decided by the separately registered all-position functional "
            "and margin gates.",
            "The historical prompt-323 Jacobian norm did not reproduce. This "
            "event is explicitly limited to the exact-pinned current-runtime "
            "sensitivity shape and does not establish historical-runtime "
            "reproducibility.",
        ],
        "gpu": gpu,
    }
    command = (
        "python -m "
        "jspace_phase4.experiments.p4_qwen_lens_influence_paired "
        f"--config {arguments.config}")
    manifest_payload = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "fit_config_sha256": config["fit_config_sha256"],
        "fit_producer_sha256": config["fit_producer_sha256"],
        "model": model_reference(config["model_uri"]),
        "model_snapshot": model_snapshot,
        "jlens": jlens_contract,
        "runtime": runtime,
        "runtime_amendment_contract": {
            "contract_version": amendment["contract_version"],
            "prior_config_sha256": amendment["prior_config_sha256"],
            "primary_computation_ordinal": amendment[
                "primary_computation_ordinal"],
            "repeat_computation_ordinal": amendment[
                "repeat_computation_ordinal"],
            "repeat_computation_role": amendment[
                "repeat_computation_role"],
            "sources": amendment_sources,
        },
        "model_fused_kernel_bindings": binding_contract,
        "gpu": gpu,
        "fixed_sampling": sampling_contract,
        "task_bank_inputs": bank_contract,
        "task_token_contract": task_contract,
        "model_tensor_contract": tensor_contract,
        "effective_gain": gain_contract,
        "lens_paths": {lens: str(lens_paths[lens]) for lens in PAIR},
        "equal_weight_contract": payload["equal_weight_contract"],
    }
    manifest = {
        "schema_version": 1, "payload": manifest_payload,
        "payload_sha256": object_sha256(manifest_payload),
    }
    atomic_json(manifest_path, manifest)
    inputs = {
        **{f"lens_{lens}": config["lenses"][lens]["lens_sha256"]
           for lens in PAIR},
        "draw_a_corpus": corpus_sha,
        "fit_config": config["fit_config_sha256"],
        "fit_producer": config["fit_producer_sha256"],
        "equal_weight_source_result": adjacent["source_result_sha256"],
        "equal_weight_source_table": adjacent["source_table_sha256"],
        "fixed_token_ids": sampling["token_ids_sha256"],
        "fixed_rademacher_probes": sampling["packed_probes_sha256"],
        "runtime_amendment_precommit": amendment["precommit_sha256"],
        "runtime_incident_report": amendment["incident_report_sha256"],
        "runtime_incident_identity": amendment["incident_identity_sha256"],
        "input_manifest": manifest["payload_sha256"],
    }
    write_result4(
        payload, result_path,
        Provenance4(
            evidence_id=config["evidence_id"], tier=config["tier"],
            command=command, inputs=inputs,
            input_manifest_sha256=manifest["payload_sha256"],
            model=model_reference(config["model_uri"]),
            seed_contract=(
                "exact frozen prompt 323; equal-prompt algebra; registered "
                "tiny/direct and adjacent-checkpoint assertions; fixed token "
                "and probe samples; preselected current-runtime computation 1 "
                "with computation 2 discarded after repeatability QA"),
        ))
    outputs = [
        drive_contribution, result_path, manifest_path, structural_path,
        assertion_path, state_path, *part_paths, png_path, pdf_path,
    ]
    create(
        config["evidence_id"], tier=config["tier"],
        what=config["registry_what"], command=command,
        outputs=outputs, inputs=inputs)
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "decision": decision,
        "prompt_retained_unconditionally": True,
        "contribution": str(drive_contribution),
        "result": str(result_path), "figure": str(png_path),
    }, indent=1))


if __name__ == "__main__":
    main()
