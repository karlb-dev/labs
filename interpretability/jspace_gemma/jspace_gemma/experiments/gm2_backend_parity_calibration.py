"""Resumable G2.1 exact-backend calibration, freeze, and registration.

The process boundary is part of the scientific firewall:

* ``stage`` and ``run`` never import the evidence registry;
* ``freeze`` reads only the frozen G2.1 config and its raw rows;
* ``register`` verifies that the immutable ceiling already exists before it
  imports or reads the registry.

No finite difference is used as an exact derivative anywhere in this module.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import subprocess
import time
import warnings
from collections import UserDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from jspace_gemma.architecture import audit_loaded_model, decoder_components
from jspace_gemma.autodiff import exact_jvp
from jspace_gemma.backend_calibration import (
    FINITE,
    FULL_SUFFIX,
    bfloat16_quantum_metrics,
    compare_tensors,
    derive_calibration,
    direction_tensor,
    stable_seed,
    tensor_sha256,
)
from jspace_gemma.gpu import require_cuda
from jspace_gemma.hooks import ExplicitDecoderSuffix, TargetSpec
from jspace_gemma.manifests import (
    atomic_json,
    environment_payload,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import PACKAGE_ROOT, local_root, resolve_uri, run_root
from jspace_gemma.staging import stage_snapshot, verify_snapshot


EVIDENCE_ID = "gm2-backend-parity-calibration-v1"
FOUNDATION_ID = "gm2-foundation-v1"
BRANCH = "interp_jspace_gemma_transport_2"
CONFIG = PACKAGE_ROOT / "configs/gm2_backend_parity_calibration.yaml"
AUDIT_CORRECTION = (
    PACKAGE_ROOT / "protocol/G2_POSTDATA_RECONSTRUCTION_AUDIT_CORRECTION.md"
)
LOCAL_CACHE = Path("/content/jspace_g2_models")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _config() -> dict:
    payload = yaml.safe_load(CONFIG.read_text())
    if payload.get("status") != "FROZEN_PRE_G2_1" or payload.get("evidence_id") != EVIDENCE_ID:
        raise RuntimeError("G2.1 calibration config is not frozen")
    if payload["exact_backends"] != {
        "primary": "torch.func.jvp",
        "independent": "torch.autograd.functional.jvp",
        "finite_difference_as_exact": "forbidden",
    }:
        raise RuntimeError("G2.1 exact-backend contract drifted")
    configured_root = Path(payload["run_root"]).resolve()
    if run_root().resolve() != configured_root:
        raise RuntimeError("JSPACE_GEMMA_RUN_ROOT differs from the frozen G2.1 root")
    return payload


def _paths(config: dict) -> dict[str, Path]:
    root = Path(config["run_root"])
    raw = root / "raw" / EVIDENCE_ID
    derived = root / "derived" / EVIDENCE_ID
    figures = root / "figures" / EVIDENCE_ID
    manifests = root / "manifests"
    for path in (raw, derived, figures, manifests, root / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "raw_root": raw,
        "derived_root": derived,
        "figure_root": figures,
        "manifest_root": manifests,
        "state": raw / "raw_rows_state.json",
        "heartbeat": root / "checkpoints" / "gm2_backend_calibration_heartbeat.json",
        "rows_parquet": raw / "backend_rows.parquet",
        "pairs": raw / "pair_summaries.json",
        "summary": derived / "calibration_summary.json",
        "threshold": derived / "backend_ceiling_frozen.json",
        "figure": figures / "backend_disagreement_by_model_batch.png",
        "registration_receipt": manifests / "gm2_backend_calibration_registration.json",
    }


def _snapshot_manifest_path(paths: dict[str, Path], model_key: str) -> Path:
    return paths["manifest_root"] / f"gm2_snapshot_{model_key}.json"


def _lock(name: str):
    path = local_root() / "locks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another G2.1 process owns {path}") from exc
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _write_heartbeat(paths: dict[str, Path], **fields) -> None:
    atomic_json(
        paths["heartbeat"],
        {
            "schema_version": 1,
            "evidence_id": EVIDENCE_ID,
            "updated_utc": _utc(),
            "pid": os.getpid(),
            **fields,
        },
    )


def _load_state(
    path: Path, *, config_sha: str, code_commit: str | None
) -> dict:
    if not path.exists():
        return {
            "schema_version": 1,
            "evidence_id": EVIDENCE_ID,
            "config_sha256": config_sha,
            "code_commit": code_commit,
            "created_utc": _utc(),
            "updated_utc": _utc(),
            "producer_pids": [],
            "rows": [],
            "pair_summaries": [],
            "model_audits": {},
            "snapshot_manifests": {},
            "fresh_process_replays": [],
        }
    state = json.loads(path.read_text())
    required = {
        "evidence_id": EVIDENCE_ID,
        "config_sha256": config_sha,
    }
    if code_commit is not None:
        required["code_commit"] = code_commit
    mismatch = {
        key: {"expected": value, "actual": state.get(key)}
        for key, value in required.items()
        if state.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"G2.1 recovery-state identity mismatch: {mismatch}")
    return state


def _checkpoint(path: Path, state: dict) -> None:
    state["updated_utc"] = _utc()
    atomic_json(path, state)


def _remote_from_staging_manifest(manifest: dict) -> dict:
    files = [
        {
            "path": row["path"],
            "size_bytes": row["expected_size_bytes"],
            "lfs_sha256": row["expected_lfs_sha256"],
            "git_blob_id": row["expected_git_blob_id"],
        }
        for row in manifest["files"]
    ]
    return {
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "files": files,
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "inventory_sha256": manifest["remote_inventory_sha256"],
    }


def _verified_snapshot(
    config: dict,
    paths: dict[str, Path],
    model_key: str,
) -> tuple[Path, dict]:
    spec = config["models"][model_key]
    manifest_path = _snapshot_manifest_path(paths, model_key)
    if not manifest_path.is_file():
        raise RuntimeError(f"stage the exact {model_key} snapshot first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    required = {
        "repo_id": spec["model_id"],
        "revision": spec["revision"],
        "config_sha256": file_sha256(CONFIG),
        "all_content_hashes_verified": True,
    }
    mismatch = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in required.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"snapshot staging manifest mismatch: {mismatch}")
    verification = verify_snapshot(
        manifest["snapshot"],
        repo_id=spec["model_id"],
        revision=spec["revision"],
        remote_inventory=_remote_from_staging_manifest(manifest),
    )
    if (
        not verification["all_content_hashes_verified"]
        or verification["remote_inventory_sha256"]
        != manifest["remote_inventory_sha256"]
    ):
        raise RuntimeError("exact local snapshot failed mandatory pre-load rehash")
    return Path(manifest["snapshot"]), manifest


def stage(model_key: str) -> None:
    git = require_clean_tree(branch=BRANCH)
    config = _config()
    paths = _paths(config)
    if paths["threshold"].exists():
        raise RuntimeError("G2.1 is already frozen; staging is closed")
    if model_key not in config["models"]:
        raise ValueError(f"unknown frozen model key {model_key!r}")
    output = _snapshot_manifest_path(paths, model_key)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite staged manifest {output}")
    spec = config["models"][model_key]
    result = stage_snapshot(
        repo_id=spec["model_id"],
        revision=spec["revision"],
        cache_root=LOCAL_CACHE,
        seed_model_root=None,
        output_manifest=output,
    )
    result.pop("snapshot_manifest_sha256", None)
    result.update(
        {
            "model_key": model_key,
            "config_sha256": file_sha256(CONFIG),
            "staging_code_commit": git["code_commit"],
            "direct_huggingface_download": True,
            "drive_model_copy_used": False,
            "target_model_loaded": False,
            "model_outcome_created": False,
        }
    )
    result["snapshot_manifest_sha256"] = object_sha256(result)
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "model_key": model_key,
                "snapshot": result["snapshot"],
                "manifest": str(output),
                "manifest_file_sha256": file_sha256(output),
                "remote_inventory_sha256": result["remote_inventory_sha256"],
                "weight_shards": len(result["weight_shards"]),
            },
            indent=1,
        )
    )


def _load_prompts(config: dict) -> list[dict]:
    source = resolve_uri(config["prompt_bank"])
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    by_id = {row["prompt_id"]: row for row in rows}
    missing = set(config["prompt_ids"]) - set(by_id)
    if missing:
        raise RuntimeError(f"frozen G2.1 prompts are absent: {sorted(missing)}")
    return [by_id[prompt_id] for prompt_id in config["prompt_ids"]]


def _load_model(snapshot: Path, spec: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    ).eval()
    base, _, text_config, _ = decoder_components(model)
    failures = []
    if len(base.layers) != int(spec["expected_decoder_layers"]):
        failures.append("decoder layer count")
    if getattr(text_config, "_attn_implementation", None) != "eager":
        failures.append("attention implementation")
    if "expected_outer_model_type" in spec and model.config.model_type != spec["expected_outer_model_type"]:
        failures.append("outer model type")
    if "expected_text_model_type" in spec and text_config.model_type != spec["expected_text_model_type"]:
        failures.append("text model type")
    if "expected_model_type" in spec and model.config.model_type != spec["expected_model_type"]:
        failures.append("model type")
    if failures:
        raise RuntimeError(f"loaded model violates frozen architecture lock: {failures}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model, base


class _LocalComponentSuffix:
    """Frozen diagnostic next-block attention or MLP branch map."""

    def __init__(self, suffix: ExplicitDecoderSuffix, variant: str):
        if variant not in {"attention_only", "mlp_only"}:
            raise ValueError(variant)
        self.suffix = suffix
        self.variant = variant
        self.clean_source = suffix.clean_source
        self.index = suffix.source_layer + 1

    def _kwargs(self) -> dict:
        layer_type = self.suffix.config.layer_types[self.index]
        result = {
            "attention_mask": self.suffix.mask_mapping[layer_type],
            "position_ids": self.suffix.position_ids,
            "position_embeddings": self.suffix.position_embeddings[layer_type],
            "past_key_values": None,
            "use_cache": False,
        }
        return result

    def __call__(self, explicit_source_fp32: torch.Tensor) -> torch.Tensor:
        hidden = explicit_source_fp32.to(self.clean_source.dtype)
        layer = self.suffix.base.layers[self.index]
        kwargs = self._kwargs()
        outer_type = getattr(self.suffix.causal_lm.config, "model_type", None)
        residual = hidden
        if outer_type in {"gemma4", "gemma4_text"}:
            attention_input = layer.input_layernorm(hidden)
            attention, _ = layer.self_attn(
                hidden_states=attention_input,
                position_embeddings=kwargs["position_embeddings"],
                attention_mask=kwargs["attention_mask"],
                shared_kv_states=UserDict(),
                position_ids=kwargs["position_ids"],
                past_key_values=None,
                use_cache=False,
            )
            attention = layer.post_attention_layernorm(attention)
            post_attention = residual + attention
            if self.variant == "attention_only":
                result = attention
            else:
                mlp_input = layer.pre_feedforward_layernorm(post_attention)
                branch = layer.mlp(mlp_input)
                if layer.enable_moe_block:
                    branch_one = layer.post_feedforward_layernorm_1(branch)
                    flattened = post_attention.reshape(-1, post_attention.shape[-1])
                    _, top_weights, top_index = layer.router(flattened)
                    branch_two = layer.pre_feedforward_layernorm_2(flattened)
                    branch_two = layer.experts(branch_two, top_index, top_weights)
                    branch_two = branch_two.reshape(post_attention.shape)
                    branch_two = layer.post_feedforward_layernorm_2(branch_two)
                    branch = branch_one + branch_two
                result = layer.post_feedforward_layernorm(branch)
        elif outer_type == "olmo3":
            attention, _ = layer.self_attn(
                hidden_states=hidden,
                attention_mask=kwargs["attention_mask"],
                position_ids=kwargs["position_ids"],
                past_key_values=None,
                use_cache=False,
                position_embeddings=kwargs["position_embeddings"],
            )
            attention = layer.post_attention_layernorm(attention)
            post_attention = residual + attention
            result = (
                attention
                if self.variant == "attention_only"
                else layer.post_feedforward_layernorm(layer.mlp(post_attention))
            )
        else:
            raise RuntimeError(f"unsupported local component architecture {outer_type!r}")
        return self.suffix._select_positions(result).float()


def _batched_source_and_tangent(
    suffix,
    *,
    model_key: str,
    layer: int,
    prompt_id: str,
    batch_size: int,
    draw: dict,
    base_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    clean = suffix.clean_source.float()
    clean_batch = clean.expand(batch_size, *clean.shape[1:]).clone()
    tangent = torch.zeros_like(clean_batch, dtype=torch.float32)
    direction_rows = []
    selected = clean[0, -1]
    for slot in range(batch_size):
        seed = stable_seed(
            int(base_seed) + int(draw["seed_offset"]),
            model_key,
            int(layer),
            prompt_id,
            draw["draw_id"],
            int(slot),
        )
        direction = direction_tensor(selected, family=draw["family"], seed=seed)
        tangent[slot, -1] = direction.to(clean.device)
        direction_rows.append(
            {
                "slot": slot,
                "seed": seed,
                "sha256": tensor_sha256(direction),
            }
        )
    return clean_batch, tangent, direction_rows


def _backend_call(function, primal, tangent, backend: str) -> tuple[dict | None, dict | None, list[str]]:
    captured = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = exact_jvp(function, primal, tangent, backend=backend)
        captured = [f"{type(row.message).__name__}: {row.message}" for row in caught]
        output = result.primal.detach().float().cpu()
        derivative = result.tangent.detach().float().cpu()
        if int(primal.shape[0]) == 1 and output.ndim == 1:
            output = output.unsqueeze(0)
            derivative = derivative.unsqueeze(0)
        if output.shape[0] != int(primal.shape[0]):
            raise RuntimeError(
                "exact-JVP output does not preserve the requested batch axis"
            )
        payload = {
            "primal": output,
            "tangent": derivative,
            "backend": result.backend,
        }
        del result
        return payload, None, captured
    except Exception as exc:
        return None, {
            "backend": backend,
            "type": type(exc).__name__,
            "message": str(exc),
        }, captured


def _evaluate_pair(
    function,
    *,
    config: dict,
    model_key: str,
    model_spec: dict,
    layer: int,
    prompt: dict,
    prompt_sha: str,
    token_sha: str,
    batch_size: int,
    draw: dict,
    suffix_variant: str,
) -> tuple[list[dict], dict]:
    clean, tangent, directions = _batched_source_and_tangent(
        function,
        model_key=model_key,
        layer=layer,
        prompt_id=prompt["prompt_id"],
        batch_size=batch_size,
        draw=draw,
        base_seed=int(config["directions"]["base_seed"]),
    )
    pair_id = "|".join(
        [
            model_key,
            str(layer),
            prompt["prompt_id"],
            str(batch_size),
            draw["draw_id"],
            suffix_variant,
        ]
    )
    primary_name = config["exact_backends"]["primary"]
    independent_name = config["exact_backends"]["independent"]
    primary, primary_error, warnings_primary = _backend_call(
        function, clean, tangent, primary_name
    )
    independent, independent_error, warnings_independent = _backend_call(
        function, clean, tangent, independent_name
    )
    replay, replay_error, warnings_replay = _backend_call(
        function, clean, tangent, primary_name
    )
    kernel_warnings = sorted(
        set(warnings_primary + warnings_independent + warnings_replay)
    )
    errors = [
        error
        for error in (primary_error, independent_error, replay_error)
        if error is not None
    ]
    rows = []
    if primary is None or independent is None or replay is None:
        state = "exception:" + ",".join(error["backend"] for error in errors)
        for slot, direction in enumerate(directions):
            rows.append(
                {
                    "row_id": f"{pair_id}|slot={slot}",
                    "pair_id": pair_id,
                    "model_key": model_key,
                    "model_id": model_spec["model_id"],
                    "model_revision": model_spec["revision"],
                    "layer": int(layer),
                    "prompt_id": prompt["prompt_id"],
                    "prompt_sha256": prompt_sha,
                    "token_ids_sha256": token_sha,
                    "batch_size": int(batch_size),
                    "slot": int(slot),
                    "direction_family": draw["family"],
                    "direction_id": draw["draw_id"],
                    "direction_seed": int(direction["seed"]),
                    "direction_sha256": direction["sha256"],
                    "dtype": "torch.bfloat16",
                    "backend_primary": primary_name,
                    "backend_independent": independent_name,
                    "primal_relative_error": None,
                    "tangent_cosine": None,
                    "tangent_relative_error": None,
                    "max_absolute_difference": None,
                    "max_difference_dtype_quanta": None,
                    "one_dtype_quantum_relative_equivalent": None,
                    "ten_dtype_quanta_relative_equivalent": None,
                    "deterministic_replay": False,
                    "suffix_variant": suffix_variant,
                    "finite_or_exception_state": state,
                    "backend_errors": errors,
                    "kernel_backend_warnings": kernel_warnings,
                    "primary_tangent_sha256": None,
                    "primary_primal_sha256": None,
                }
            )
        summary = {
            "pair_id": pair_id,
            "model_key": model_key,
            "layer": int(layer),
            "prompt_id": prompt["prompt_id"],
            "batch_size": int(batch_size),
            "direction_id": draw["draw_id"],
            "suffix_variant": suffix_variant,
            "finite_or_exception_state": state,
            "backend_errors": errors,
            "kernel_backend_warnings": kernel_warnings,
            "producer_pid": os.getpid(),
        }
        del clean, tangent
        torch.cuda.empty_cache()
        gc.collect()
        return rows, summary

    deterministic_all = (
        tensor_sha256(primary["tangent"]) == tensor_sha256(replay["tangent"])
        and tensor_sha256(primary["primal"]) == tensor_sha256(replay["primal"])
    )
    for slot, direction in enumerate(directions):
        tangent_comparison = compare_tensors(
            independent["tangent"][slot], primary["tangent"][slot]
        )
        primal_comparison = compare_tensors(
            independent["primal"][slot], primary["primal"][slot]
        )
        difference = independent["tangent"][slot] - primary["tangent"][slot]
        quantum = bfloat16_quantum_metrics(
            difference, primary["tangent"][slot]
        )
        finite = all(
            np.isfinite(value)
            for value in (
                tangent_comparison["relative_error"],
                tangent_comparison["cosine"],
                primal_comparison["relative_error"],
                quantum["max_difference_dtype_quanta"],
                quantum["ten_dtype_quanta_relative_equivalent"],
            )
        )
        state = FINITE if finite else "nonfinite"
        replay_slot = (
            tensor_sha256(primary["tangent"][slot])
            == tensor_sha256(replay["tangent"][slot])
            and tensor_sha256(primary["primal"][slot])
            == tensor_sha256(replay["primal"][slot])
        )
        rows.append(
            {
                "row_id": f"{pair_id}|slot={slot}",
                "pair_id": pair_id,
                "model_key": model_key,
                "model_id": model_spec["model_id"],
                "model_revision": model_spec["revision"],
                "layer": int(layer),
                "prompt_id": prompt["prompt_id"],
                "prompt_sha256": prompt_sha,
                "token_ids_sha256": token_sha,
                "batch_size": int(batch_size),
                "slot": int(slot),
                "direction_family": draw["family"],
                "direction_id": draw["draw_id"],
                "direction_seed": int(direction["seed"]),
                "direction_sha256": direction["sha256"],
                "dtype": "torch.bfloat16",
                "backend_primary": primary_name,
                "backend_independent": independent_name,
                "primal_relative_error": primal_comparison["relative_error"],
                "tangent_cosine": tangent_comparison["cosine"],
                "tangent_relative_error": tangent_comparison["relative_error"],
                "max_absolute_difference": tangent_comparison[
                    "max_absolute_difference"
                ],
                "max_difference_dtype_quanta": quantum[
                    "max_difference_dtype_quanta"
                ],
                "one_dtype_quantum_relative_equivalent": quantum[
                    "one_dtype_quantum_relative_equivalent"
                ],
                "ten_dtype_quanta_relative_equivalent": quantum[
                    "ten_dtype_quanta_relative_equivalent"
                ],
                "deterministic_replay": bool(replay_slot),
                "suffix_variant": suffix_variant,
                "finite_or_exception_state": state,
                "backend_errors": [],
                "kernel_backend_warnings": kernel_warnings,
                "primary_tangent_sha256": tensor_sha256(primary["tangent"][slot]),
                "primary_primal_sha256": tensor_sha256(primary["primal"][slot]),
                "primary_tangent_norm": tangent_comparison["reference_norm"],
                "independent_tangent_norm": tangent_comparison["value_norm"],
                "tangent_difference_norm": float(difference.double().norm()),
                "tangent_dot_product": float(
                    torch.dot(
                        independent["tangent"][slot].reshape(-1).double(),
                        primary["tangent"][slot].reshape(-1).double(),
                    )
                ),
                "primary_primal_norm": primal_comparison["reference_norm"],
                "independent_primal_norm": primal_comparison["value_norm"],
                "primal_difference_norm": float(
                    (independent["primal"][slot] - primary["primal"][slot])
                    .double()
                    .norm()
                ),
            }
        )
    selected = compare_tensors(independent["tangent"][0], primary["tangent"][0])
    all_slots = compare_tensors(independent["tangent"], primary["tangent"])
    summary = {
        "pair_id": pair_id,
        "model_key": model_key,
        "layer": int(layer),
        "prompt_id": prompt["prompt_id"],
        "batch_size": int(batch_size),
        "direction_id": draw["draw_id"],
        "direction_family": draw["family"],
        "suffix_variant": suffix_variant,
        "finite_or_exception_state": FINITE,
        "selected_slot": selected,
        "all_slots": all_slots,
        "deterministic_replay_all_slots": bool(deterministic_all),
        "backend_errors": [],
        "kernel_backend_warnings": kernel_warnings,
        "producer_pid": os.getpid(),
    }
    del clean, tangent, primary, independent, replay
    torch.cuda.empty_cache()
    gc.collect()
    return rows, summary


def _pending_for_suffix(
    completed: set[str],
    *,
    model_key: str,
    layer: int,
    prompt_id: str,
    config: dict,
) -> bool:
    variants = [FULL_SUFFIX]
    op = config["op_screen"]
    if (
        layer == config["models"][model_key]["layers_zero_indexed"][1]
        and prompt_id in op["frozen_subset"]["prompts"]
    ):
        variants.extend(["attention_only", "mlp_only"])
    for variant in variants:
        batches = (
            config["batch_sizes"]
            if variant == FULL_SUFFIX
            else op["frozen_subset"]["batch_sizes"]
        )
        draws = (
            config["directions"]["draws"]
            if variant == FULL_SUFFIX
            else [
                row
                for row in config["directions"]["draws"]
                if row["draw_id"] in op["frozen_subset"]["directions"]
            ]
        )
        for batch_size in batches:
            for draw in draws:
                pair_id = "|".join(
                    [
                        model_key,
                        str(layer),
                        prompt_id,
                        str(batch_size),
                        draw["draw_id"],
                        variant,
                    ]
                )
                if pair_id not in completed:
                    return True
    return False


def run(model_key: str, *, max_pairs: int | None) -> None:
    lock = _lock("gm2_backend_parity_calibration.lock")
    try:
        git = require_clean_tree(branch=BRANCH)
        require_cuda()
        config = _config()
        paths = _paths(config)
        if paths["threshold"].exists():
            raise RuntimeError("G2.1 ceiling is frozen; raw-row mutation is closed")
        if model_key not in config["models"]:
            raise ValueError(f"unknown frozen model key {model_key!r}")
        snapshot, staging_manifest = _verified_snapshot(config, paths, model_key)
        state = _load_state(
            paths["state"],
            config_sha=file_sha256(CONFIG),
            code_commit=git["code_commit"],
        )
        if os.getpid() not in state["producer_pids"]:
            state["producer_pids"].append(os.getpid())
        state["snapshot_manifests"][model_key] = {
            "path": str(_snapshot_manifest_path(paths, model_key)),
            "sha256": file_sha256(_snapshot_manifest_path(paths, model_key)),
            "remote_inventory_sha256": staging_manifest["remote_inventory_sha256"],
        }
        completed = {row["pair_id"] for row in state["pair_summaries"]}
        prompts = _load_prompts(config)
        spec = config["models"][model_key]
        environment = environment_payload(require_gpu=True)
        _write_heartbeat(
            paths,
            phase="loading_model",
            model_key=model_key,
            completed_pairs=len(completed),
        )
        tokenizer, model, _ = _load_model(snapshot, spec)
        model_audit = audit_loaded_model(model)
        model_audit.update(
            {
                "model_key": model_key,
                "model_id": spec["model_id"],
                "revision": spec["revision"],
                "environment": environment,
                "snapshot_manifest_sha256": file_sha256(
                    _snapshot_manifest_path(paths, model_key)
                ),
            }
        )
        state["model_audits"][model_key] = model_audit
        checkpoint_every = int(config["acceptance"]["checkpoint_every_pairs"])
        new_since_checkpoint = 0
        new_total = 0
        stop = False
        target = TargetSpec("final_residual", position_indices=(-1,))
        for prompt in prompts:
            if stop:
                break
            encoded = tokenizer(
                prompt["text"],
                return_tensors="pt",
                add_special_tokens=True,
                truncation=False,
            )
            input_ids = encoded["input_ids"].to("cuda")
            attention_mask = encoded.get(
                "attention_mask", torch.ones_like(input_ids)
            ).to("cuda")
            prompt_sha = hashlib.sha256(prompt["text"].encode()).hexdigest()
            token_sha = hashlib.sha256(
                input_ids.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            for layer in spec["layers_zero_indexed"]:
                if not _pending_for_suffix(
                    completed,
                    model_key=model_key,
                    layer=int(layer),
                    prompt_id=prompt["prompt_id"],
                    config=config,
                ):
                    continue
                suffix = ExplicitDecoderSuffix(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    source_layer=int(layer),
                    target=target,
                )
                variants = [(FULL_SUFFIX, suffix)]
                op = config["op_screen"]
                if (
                    int(layer) == int(spec["layers_zero_indexed"][1])
                    and prompt["prompt_id"] in op["frozen_subset"]["prompts"]
                ):
                    variants.extend(
                        (variant, _LocalComponentSuffix(suffix, variant))
                        for variant in ("attention_only", "mlp_only")
                    )
                for variant, function in variants:
                    batches = (
                        config["batch_sizes"]
                        if variant == FULL_SUFFIX
                        else op["frozen_subset"]["batch_sizes"]
                    )
                    draws = (
                        config["directions"]["draws"]
                        if variant == FULL_SUFFIX
                        else [
                            row
                            for row in config["directions"]["draws"]
                            if row["draw_id"]
                            in op["frozen_subset"]["directions"]
                        ]
                    )
                    for batch_size in batches:
                        for draw in draws:
                            pair_id = "|".join(
                                [
                                    model_key,
                                    str(layer),
                                    prompt["prompt_id"],
                                    str(batch_size),
                                    draw["draw_id"],
                                    variant,
                                ]
                            )
                            if pair_id in completed:
                                continue
                            _write_heartbeat(
                                paths,
                                phase="backend_pair",
                                model_key=model_key,
                                pair_id=pair_id,
                                completed_pairs=len(completed),
                            )
                            rows, summary = _evaluate_pair(
                                function,
                                config=config,
                                model_key=model_key,
                                model_spec=spec,
                                layer=int(layer),
                                prompt=prompt,
                                prompt_sha=prompt_sha,
                                token_sha=token_sha,
                                batch_size=int(batch_size),
                                draw=draw,
                                suffix_variant=variant,
                            )
                            state["rows"].extend(rows)
                            state["pair_summaries"].append(summary)
                            completed.add(pair_id)
                            new_since_checkpoint += 1
                            new_total += 1
                            if new_since_checkpoint >= checkpoint_every:
                                _checkpoint(paths["state"], state)
                                new_since_checkpoint = 0
                            if max_pairs is not None and new_total >= max_pairs:
                                stop = True
                                break
                        if stop:
                            break
                    if stop:
                        break
                del suffix
                torch.cuda.empty_cache()
                gc.collect()
                if stop:
                    break
        _checkpoint(paths["state"], state)
        model_pairs = [
            row for row in state["pair_summaries"] if row["model_key"] == model_key
        ]
        _write_heartbeat(
            paths,
            phase="model_run_complete" if max_pairs is None else "model_run_checkpoint",
            model_key=model_key,
            model_pairs=len(model_pairs),
            total_pairs=len(state["pair_summaries"]),
            total_rows=len(state["rows"]),
        )
        print(
            json.dumps(
                {
                    "model_key": model_key,
                    "new_pairs": new_total,
                    "model_pairs_in_state": len(model_pairs),
                    "total_pairs_in_state": len(state["pair_summaries"]),
                    "total_rows_in_state": len(state["rows"]),
                    "state": str(paths["state"]),
                    "state_sha256": file_sha256(paths["state"]),
                },
                indent=1,
            )
        )
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    finally:
        lock.close()


def fresh_replay(model_key: str) -> None:
    lock = _lock("gm2_backend_parity_calibration.lock")
    try:
        git = require_clean_tree(branch=BRANCH)
        require_cuda()
        config = _config()
        paths = _paths(config)
        if paths["threshold"].exists():
            raise RuntimeError("G2.1 ceiling is frozen; replay mutation is closed")
        snapshot, _ = _verified_snapshot(config, paths, model_key)
        state = _load_state(
            paths["state"],
            config_sha=file_sha256(CONFIG),
            code_commit=git["code_commit"],
        )
        candidates = sorted(
            (
                row
                for row in state["pair_summaries"]
                if row["model_key"] == model_key
                and row["suffix_variant"] == FULL_SUFFIX
                and row["finite_or_exception_state"] == FINITE
                and int(row["producer_pid"]) != os.getpid()
            ),
            key=lambda row: row["pair_id"],
        )
        if not candidates:
            raise RuntimeError("no completed finite pair is available for fresh replay")
        selected = candidates[0]
        if any(
            row["pair_id"] == selected["pair_id"]
            for row in state["fresh_process_replays"]
        ):
            raise RuntimeError("the frozen fresh-process sentinel is already replayed")
        spec = config["models"][model_key]
        prompts = {row["prompt_id"]: row for row in _load_prompts(config)}
        prompt = prompts[selected["prompt_id"]]
        tokenizer, model, _ = _load_model(snapshot, spec)
        encoded = tokenizer(
            prompt["text"], return_tensors="pt", add_special_tokens=True, truncation=False
        )
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded.get(
            "attention_mask", torch.ones_like(input_ids)
        ).to("cuda")
        suffix = ExplicitDecoderSuffix(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            source_layer=int(selected["layer"]),
            target=TargetSpec("final_residual", position_indices=(-1,)),
        )
        draw = next(
            row
            for row in config["directions"]["draws"]
            if row["draw_id"] == selected["direction_id"]
        )
        clean, tangent, _ = _batched_source_and_tangent(
            suffix,
            model_key=model_key,
            layer=int(selected["layer"]),
            prompt_id=selected["prompt_id"],
            batch_size=int(selected["batch_size"]),
            draw=draw,
            base_seed=int(config["directions"]["base_seed"]),
        )
        result = exact_jvp(
            suffix, clean, tangent, backend=config["exact_backends"]["primary"]
        )
        expected = sorted(
            (row for row in state["rows"] if row["pair_id"] == selected["pair_id"]),
            key=lambda row: row["slot"],
        )
        replay_tangent = result.tangent.detach().float().cpu()
        replay_primal = result.primal.detach().float().cpu()
        if len(expected) == 1 and replay_tangent.ndim == 1:
            replay_tangent = replay_tangent.unsqueeze(0)
            replay_primal = replay_primal.unsqueeze(0)
        observed_tangent = [tensor_sha256(replay_tangent[slot]) for slot in range(len(expected))]
        observed_primal = [tensor_sha256(replay_primal[slot]) for slot in range(len(expected))]
        expected_tangent = [row["primary_tangent_sha256"] for row in expected]
        expected_primal = [row["primary_primal_sha256"] for row in expected]
        passed = observed_tangent == expected_tangent and observed_primal == expected_primal
        record = {
            "pair_id": selected["pair_id"],
            "model_key": model_key,
            "original_pid": int(selected["producer_pid"]),
            "replay_pid": os.getpid(),
            "fresh_process": int(selected["producer_pid"]) != os.getpid(),
            "expected_tangent_sha256": expected_tangent,
            "observed_tangent_sha256": observed_tangent,
            "expected_primal_sha256": expected_primal,
            "observed_primal_sha256": observed_primal,
            "passed": bool(passed),
            "created_utc": _utc(),
        }
        state["fresh_process_replays"].append(record)
        _checkpoint(paths["state"], state)
        _write_heartbeat(
            paths,
            phase="fresh_process_replay_complete",
            model_key=model_key,
            pair_id=selected["pair_id"],
            passed=passed,
        )
        print(json.dumps(record, indent=1))
        del result, clean, tangent, suffix, model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        if not passed:
            raise RuntimeError("fresh-process exact-JVP replay drifted")
    finally:
        lock.close()


def _validate_pair_summaries(state: dict) -> dict:
    rows_by_pair: dict[str, list[dict]] = {}
    for row in state["rows"]:
        rows_by_pair.setdefault(row["pair_id"], []).append(row)
    checked = 0
    failures = []
    cosine_residuals = []
    for pair in state["pair_summaries"]:
        if pair["finite_or_exception_state"] != FINITE:
            continue
        rows = sorted(rows_by_pair[pair["pair_id"]], key=lambda row: row["slot"])
        difference_norm = float(
            np.sqrt(sum(float(row["tangent_difference_norm"]) ** 2 for row in rows))
        )
        primary_norm = float(
            np.sqrt(sum(float(row["primary_tangent_norm"]) ** 2 for row in rows))
        )
        independent_norm = float(
            np.sqrt(sum(float(row["independent_tangent_norm"]) ** 2 for row in rows))
        )
        dot = sum(float(row["tangent_dot_product"]) for row in rows)
        reconstructed = {
            "relative_error": difference_norm / max(primary_norm, 1e-300),
            "cosine": dot / max(primary_norm * independent_norm, 1e-300),
            "max_absolute_difference": max(
                float(row["max_absolute_difference"]) for row in rows
            ),
        }
        selected = rows[0]
        comparisons = {
            "all_relative": (
                reconstructed["relative_error"], pair["all_slots"]["relative_error"]
            ),
            "all_cosine": (reconstructed["cosine"], pair["all_slots"]["cosine"]),
            "all_max": (
                reconstructed["max_absolute_difference"],
                pair["all_slots"]["max_absolute_difference"],
            ),
            "selected_relative": (
                float(selected["tangent_relative_error"]),
                pair["selected_slot"]["relative_error"],
            ),
            "selected_cosine": (
                float(selected["tangent_cosine"]), pair["selected_slot"]["cosine"]
            ),
        }
        bad = {}
        for key, values in comparisons.items():
            if key == "all_cosine":
                # The producer's all-slot summary used PyTorch's float32
                # cosine reduction.  Reconstruction uses saved float64 dot
                # products and norms.  Bound their reduction-order difference
                # prospectively by eight fp32 epsilons per binary reduction
                # level; this remains orders below the frozen 0.995 router.
                elements = max(int(pair["all_slots"]["elements"]), 2)
                tolerance = (
                    8.0
                    * float(np.finfo(np.float32).eps)
                    * float(np.ceil(np.log2(elements)))
                )
                residual = abs(values[0] - values[1])
                cosine_residuals.append(
                    {
                        "pair_id": pair["pair_id"],
                        "elements": elements,
                        "absolute_residual": residual,
                        "float32_reduction_bound": tolerance,
                    }
                )
                close = residual <= tolerance
            else:
                close = bool(
                    np.isclose(values[0], values[1], rtol=1e-7, atol=1e-9)
                )
            if not close:
                bad[key] = values
        if bad:
            failures.append({"pair_id": pair["pair_id"], "mismatches": bad})
        checked += 1
    worst = max(
        cosine_residuals,
        key=lambda row: row["absolute_residual"]
        / max(row["float32_reduction_bound"], np.finfo(np.float64).tiny),
    )
    return {
        "checked_pairs": checked,
        "failures": failures,
        "passed": not failures,
        "all_slot_cosine_reconstruction": {
            "stored_accumulation": "float32 torch cosine reduction",
            "reconstruction_accumulation": "float64 saved dot products and norms",
            "bound_formula": "8 * float32_epsilon * ceil(log2(elements))",
            "max_absolute_residual": max(
                row["absolute_residual"] for row in cosine_residuals
            ),
            "worst_bound_fraction": worst["absolute_residual"]
            / worst["float32_reduction_bound"],
            "worst_pair": worst,
        },
    }


def _validate_quantum_rows(rows: list[dict]) -> dict:
    failures = []
    checked = 0
    for row in rows:
        if row["finite_or_exception_state"] != FINITE:
            continue
        one = float(row["one_dtype_quantum_relative_equivalent"])
        ten = float(row["ten_dtype_quanta_relative_equivalent"])
        quanta = float(row["max_difference_dtype_quanta"])
        if not np.isclose(ten, 10.0 * one, rtol=1e-12, atol=0.0) or quanta < 0:
            failures.append(row["row_id"])
        checked += 1
    return {"checked_rows": checked, "failures": failures[:20], "passed": not failures}


def _write_figure(rows: list[dict], destination: Path) -> dict:
    import matplotlib.pyplot as plt

    finite = [
        row
        for row in rows
        if row["suffix_variant"] == FULL_SUFFIX
        and row["finite_or_exception_state"] == FINITE
    ]
    positive = [float(row["tangent_relative_error"]) for row in finite if float(row["tangent_relative_error"]) > 0]
    display_floor = min(positive) / 2.0 if positive else np.finfo(np.float64).tiny
    clipped_zeros = sum(float(row["tangent_relative_error"]) == 0.0 for row in finite)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, model_key in zip(axes, sorted({row["model_key"] for row in finite})):
        for batch_size in (1, 4, 8):
            values = sorted(
                max(float(row["tangent_relative_error"]), display_floor)
                for row in finite
                if row["model_key"] == model_key
                and int(row["batch_size"]) == batch_size
            )
            y = np.arange(1, len(values) + 1) / max(len(values), 1)
            axis.step(values, y, where="post", label=f"batch {batch_size}")
        axis.set_xscale("log")
        axis.set_title(model_key)
        axis.set_xlabel("backend tangent relative disagreement")
        axis.set_ylabel("empirical CDF")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    fig.suptitle(
        f"G2.1 exact-backend calibration; zero display floor={display_floor:.3e} "
        f"({clipped_zeros} zeros clipped only for display)"
    )
    temporary = destination.with_suffix(destination.suffix + f".tmp{os.getpid()}")
    fig.savefig(temporary, dpi=180, format="png")
    plt.close(fig)
    os.replace(temporary, destination)
    return {
        "display_floor": float(display_floor),
        "zero_values_clipped_for_display_only": int(clipped_zeros),
        "log_scale_used": True,
        "scientific_values_modified": False,
    }


def freeze() -> None:
    lock = _lock("gm2_backend_parity_calibration.lock")
    try:
        git = require_clean_tree(branch=BRANCH)
        config = _config()
        paths = _paths(config)
        if paths["threshold"].exists():
            raise FileExistsError(f"refusing to overwrite frozen ceiling {paths['threshold']}")
        state = _load_state(
            paths["state"],
            config_sha=file_sha256(CONFIG),
            code_commit=None,
        )
        rows = state["rows"]
        result = derive_calibration(rows, config)
        reverse = derive_calibration(list(reversed(rows)), config)
        if object_sha256(result) != object_sha256(reverse):
            raise RuntimeError("G2.1 aggregate changes under row-order permutation")
        expected_op_rows = (
            len(config["models"])
            * len(config["op_screen"]["frozen_subset"]["prompts"])
            * sum(config["op_screen"]["frozen_subset"]["batch_sizes"])
            * len(config["op_screen"]["frozen_subset"]["directions"])
            * (len(config["op_screen"]["suffixes"]) - 1)
        )
        if result["row_counts"]["nested_op_rows"] != expected_op_rows:
            raise RuntimeError(
                f"expected {expected_op_rows} nested op rows, found "
                f"{result['row_counts']['nested_op_rows']}"
            )
        replay_required = int(config["acceptance"]["fresh_process_replays"])
        passing_fresh = [row for row in state["fresh_process_replays"] if row["passed"]]
        if len(passing_fresh) < replay_required:
            raise RuntimeError(
                f"G2.1 requires {replay_required} passing fresh-process replay(s)"
            )
        pair_audit = _validate_pair_summaries(state)
        quantum_audit = _validate_quantum_rows(rows)
        if not pair_audit["passed"] or not quantum_audit["passed"]:
            raise RuntimeError("G2.1 reconstruction audit failed")
        if not result["audits"]["all_rows_deterministically_replayed"]:
            raise RuntimeError("one or more G2.1 rows failed deterministic replay")

        frame = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
        for column in ("backend_errors", "kernel_backend_warnings"):
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
            )
        temporary = paths["rows_parquet"].with_suffix(
            paths["rows_parquet"].suffix + f".tmp{os.getpid()}"
        )
        frame.to_parquet(temporary, index=False)
        round_trip = pd.read_parquet(temporary)
        if len(round_trip) != len(frame):
            temporary.unlink(missing_ok=True)
            raise RuntimeError("G2.1 Parquet row-count round trip failed")
        os.replace(temporary, paths["rows_parquet"])
        atomic_json(
            paths["pairs"],
            {
                "schema_version": 1,
                "evidence_id": EVIDENCE_ID,
                "pair_summaries": state["pair_summaries"],
                "fresh_process_replays": state["fresh_process_replays"],
            },
        )
        figure_audit = _write_figure(rows, paths["figure"])
        summary = {
            **result,
            "tier": "methods",
            "created_utc": _utc(),
            "code_commit": git["code_commit"],
            "raw_producer_code_commit": state["code_commit"],
            "config": {"path": str(CONFIG), "sha256": file_sha256(CONFIG)},
            "raw_state": {
                "path": str(paths["state"]),
                "sha256": file_sha256(paths["state"]),
            },
            "row_table": {
                "path": str(paths["rows_parquet"]),
                "sha256": file_sha256(paths["rows_parquet"]),
                "rows": len(frame),
            },
            "pair_reconstruction": pair_audit,
            "dtype_quantum_reconstruction": quantum_audit,
            "row_order_permutation": {
                "permutations": int(config["acceptance"]["row_order_permutations"]),
                "passed": True,
                "canonical_result_sha256": object_sha256(result),
            },
            "figure_audit": figure_audit,
            "target_firewall": {
                "derivation_inputs": [str(CONFIG), str(paths["state"])],
                "config_sha256": file_sha256(CONFIG),
                "raw_state_sha256": file_sha256(paths["state"]),
                "stage1_target_read": False,
                "stage1_outcome_field_joined": False,
                "no_target_read_assertion": True,
            },
            "finite_difference_used_as_exact": False,
            "claim_tier": "methods/development",
        }
        atomic_json(paths["summary"], summary)
        threshold = {
            "schema_version": 1,
            "status": "FROZEN_PRE_G2_2",
            "source_event_id": EVIDENCE_ID,
            "created_utc": _utc(),
            "code_commit": git["code_commit"],
            "raw_producer_code_commit": state["code_commit"],
            "config_path": str(CONFIG),
            "config_sha256": file_sha256(CONFIG),
            "raw_row_table_path": str(paths["rows_parquet"]),
            "raw_row_table_sha256": file_sha256(paths["rows_parquet"]),
            "raw_state_path": str(paths["state"]),
            "raw_state_sha256": file_sha256(paths["state"]),
            "summary_path": str(paths["summary"]),
            "summary_sha256": file_sha256(paths["summary"]),
            "formula": config["ceiling_rule"]["formula"],
            "distribution_summary": result["pooled_all_batches_measurement"],
            "bootstrap_90pct": result["prompt_bootstrap_90pct"],
            "batch_and_slot_sensitivity": result["router"][
                "batch_composition_nuisance"
            ],
            "per_model_route": result["router"]["architecture_dependent_floor"],
            "route": result["router"]["route"],
            "applicable_scope": result["applicable_scope"],
            "licensed_ceilings": result["licensed_ceilings"],
            "no_target_read_assertion": True,
            "stage1_target_read": False,
            "ceiling_frozen_before_registry_read": True,
            "finite_difference_used_as_exact": False,
        }
        threshold["payload_sha256"] = object_sha256(threshold)
        # This atomic rename is the unblinding boundary.  No registry module has
        # been imported by this process.
        atomic_json(paths["threshold"], threshold)
        _write_heartbeat(
            paths,
            phase="ceiling_frozen_pre_registry",
            threshold=str(paths["threshold"]),
            threshold_sha256=file_sha256(paths["threshold"]),
            route=threshold["route"],
        )
        print(
            json.dumps(
                {
                    "threshold": str(paths["threshold"]),
                    "threshold_sha256": file_sha256(paths["threshold"]),
                    "route": threshold["route"],
                    "licensed_ceilings": threshold["licensed_ceilings"],
                    "row_table_sha256": threshold["raw_row_table_sha256"],
                    "stage1_target_read": False,
                    "registry_read": False,
                },
                indent=1,
            )
        )
    finally:
        lock.close()


def register() -> None:
    git = require_clean_tree(branch=BRANCH)
    config = _config()
    paths = _paths(config)
    # The ceiling must be fully read and reconstructed before the registry is
    # imported.  This enforces the precommitted process ordering mechanically.
    if not paths["threshold"].is_file():
        raise RuntimeError("freeze the G2.1 ceiling before registry access")
    threshold_bytes = paths["threshold"].read_bytes()
    threshold_sha = hashlib.sha256(threshold_bytes).hexdigest()
    threshold = json.loads(threshold_bytes)
    if (
        threshold.get("status") != "FROZEN_PRE_G2_2"
        or threshold.get("ceiling_frozen_before_registry_read") is not True
        or threshold.get("no_target_read_assertion") is not True
        or threshold.get("code_commit") != git["code_commit"]
    ):
        raise RuntimeError("frozen G2.1 threshold identity is invalid")
    state = _load_state(
        paths["state"],
        config_sha=file_sha256(CONFIG),
        code_commit=None,
    )
    reconstructed = derive_calibration(state["rows"], config)
    summary = json.loads(paths["summary"].read_text())
    if object_sha256(reconstructed) != summary["row_order_permutation"][
        "canonical_result_sha256"
    ]:
        raise RuntimeError("frozen G2.1 aggregate no longer reconstructs")
    if file_sha256(paths["rows_parquet"]) != threshold["raw_row_table_sha256"]:
        raise RuntimeError("frozen G2.1 row-table hash drifted")
    frozen_output_hashes = {
        path: file_sha256(path)
        for path in (
            paths["state"],
            paths["rows_parquet"],
            paths["pairs"],
            paths["summary"],
            paths["threshold"],
            paths["figure"],
        )
    }

    from jspace_gemma.registry import create, read_events, resolve

    origins = {
        row["evidence_id"]
        for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
    }
    if EVIDENCE_ID in origins:
        raise RuntimeError("G2.1 calibration evidence is already registered")
    foundation = resolve(FOUNDATION_ID)
    if not foundation["live"] or foundation.get("model_outcome_opened") is not False:
        raise RuntimeError("frozen study-2 foundation is absent or incompatible")
    architecture_corrections = [
        row
        for row in foundation["status_events"]
        if row["event"] == "evidence_corrected"
        and row.get("correction_kind") == "predata_architecture_label"
    ]
    if (
        len(architecture_corrections) != 1
        or architecture_corrections[0].get("corrected_config_sha256")
        != file_sha256(CONFIG)
    ):
        raise RuntimeError("registered pre-data architecture correction is absent")
    architecture_correction = architecture_corrections[0]
    correction_artifact = architecture_correction["correction_artifact"]
    if file_sha256(correction_artifact["path"]) != correction_artifact["sha256"]:
        raise RuntimeError("pre-data architecture correction artifact drifted")
    event = create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "Target-isolated Gemma/OLMo exact-JVP backend calibration with a "
            "precommitted bfloat16-quantum ceiling and G2.1 route."
        ),
        command=(
            "python -m jspace_gemma.experiments.gm2_backend_parity_calibration "
            "register"
        ),
        outputs=list(frozen_output_hashes),
        inputs={
            "foundation_evidence_id": FOUNDATION_ID,
            "foundation_code_commit": foundation["code_commit"],
            "config_sha256": file_sha256(CONFIG),
            "row_table_sha256": threshold["raw_row_table_sha256"],
            "threshold_sha256_pre_registry": threshold_sha,
            "raw_producer_code_commit": state["code_commit"],
            "analysis_correction_protocol_sha256": file_sha256(AUDIT_CORRECTION),
            "predata_architecture_correction_artifact_sha256": correction_artifact[
                "sha256"
            ],
            "snapshot_manifests": state["snapshot_manifests"],
        },
        route=threshold["route"],
        licensed_ceilings=threshold["licensed_ceilings"],
        applicable_scope=threshold["applicable_scope"],
        backend_pairs=reconstructed["row_counts"]["full_backend_pairs"],
        raw_rows=reconstructed["row_counts"]["all"],
        fresh_process_replays=len(state["fresh_process_replays"]),
        target_model_opened=True,
        stage1_target_opened=False,
        no_target_read_assertion=True,
        ceiling_frozen_before_registry_read=True,
        finite_difference_used_as_exact=False,
    )
    receipt = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "registered_utc": _utc(),
        "threshold_sha256_pre_registry": threshold_sha,
        "event": event,
    }
    atomic_json(paths["registration_receipt"], receipt)
    print(
        json.dumps(
            {
                "evidence_id": EVIDENCE_ID,
                "route": threshold["route"],
                "threshold_sha256_pre_registry": threshold_sha,
                "registry_event_code_commit": event["code_commit"],
                "registration_receipt": str(paths["registration_receipt"]),
            },
            indent=1,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("stage", "run", "fresh-replay"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--model-key", required=True)
        if name == "run":
            subparser.add_argument("--max-pairs", type=int, default=None)
    subparsers.add_parser("freeze")
    subparsers.add_parser("register")
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.model_key)
    elif args.command == "run":
        if args.max_pairs is not None and args.max_pairs < 1:
            raise ValueError("--max-pairs must be positive")
        run(args.model_key, max_pairs=args.max_pairs)
    elif args.command == "fresh-replay":
        fresh_replay(args.model_key)
    elif args.command == "freeze":
        freeze()
    elif args.command == "register":
        register()


if __name__ == "__main__":
    main()
