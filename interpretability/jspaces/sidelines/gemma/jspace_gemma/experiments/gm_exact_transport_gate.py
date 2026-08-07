"""Resumable OLMo calibration and, after the firewall, Gemma G1 producer."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from jspace_gemma.gpu import require_cuda
from jspace_gemma.hooks import ExplicitDecoderSuffix, TargetSpec
from jspace_gemma.manifests import (
    atomic_json,
    environment_payload,
    file_sha256,
    git_info,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import PACKAGE_ROOT, directory, local_root
from jspace_gemma.registry import create
from jspace_gemma.state import StateHeader, StateStore
from jspace_gemma.stats import robust_floor_curvature_fit
from jspace_gemma.transport import evaluate_transport_cell

EVIDENCE_ID = "gm-jvp-olmo-calibration-v1"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/gm_g1_design.yaml"
DEFAULT_SNAPSHOT_MANIFEST = Path(
    "/content/drive/MyDrive/interpret/special-lab-1/gemma_transport_20260802/"
    "manifests/olmo_control_local_snapshot_v1.json"
)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_prompts(path: Path, ids: list[str]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {row["prompt_id"]: row for row in rows}
    if set(ids) - set(by_id):
        raise RuntimeError("frozen prompt ID is absent from prompt bank")
    return [by_id[value] for value in ids]


def _resolve_repo_uri(text: str) -> Path:
    prefix = "repo://"
    if not text.startswith(prefix):
        raise ValueError("this producer requires a repo:// prompt bank")
    return PACKAGE_ROOT.parents[1] / text[len(prefix) :]


def _aggregate(rows: list[dict], parity_rows: list[dict], wrong_hook: dict) -> dict:
    frame = pd.DataFrame(rows)
    faithful = frame[frame["faithful_delivery"]]
    groups = []
    for (layer, mode), group in faithful.groupby(["source_layer", "perturbation_mode"]):
        groups.append(
            {
                "source_layer": int(layer),
                "perturbation_mode": mode,
                "n_rows": len(group),
                "median_tangent_cosine": float(group["tangent_cosine"].median()),
                "median_tangent_relative_error": float(group["tangent_relative_error"].median()),
                "median_central_tangent_relative_error": float(
                    group["central_tangent_relative_error"].median()
                ),
                "median_homogeneity_defect": float(group["homogeneity_defect"].median()),
                "median_homogeneity_nonlinear_remainder_defect": float(
                    group["homogeneity_nonlinear_remainder_defect"].median()
                ),
                "median_odd_symmetry_defect": float(group["odd_symmetry_defect"].median()),
                "median_odd_nonlinear_remainder_defect": float(
                    group["odd_nonlinear_remainder_defect"].median()
                ),
                "median_response_snr": float(group["response_snr"].median()),
                "delivery_failure_rate": 1 - len(group) / len(
                    frame[(frame["source_layer"] == layer) & (frame["perturbation_mode"] == mode)]
                ),
            }
        )
    fits = []
    fit_keys = ["prompt_id", "source_layer", "perturbation_mode", "direction_id"]
    for key, group in faithful.groupby(fit_keys):
        if len(group) < 3:
            continue
        fit = robust_floor_curvature_fit(
            group["desired_relative_epsilon"].tolist(),
            group["tangent_relative_error"].tolist(),
        )
        identity = dict(zip(fit_keys, key, strict=True))
        identity["source_layer"] = int(identity["source_layer"])
        fits.append({**identity, **fit})
    return {
        "calibration_only_no_target_thresholds_applied": True,
        "n_rows": len(frame),
        "n_faithful_rows": len(faithful),
        "group_aggregates": groups,
        "floor_curvature_fits_unfiltered_pre_snr_threshold": fits,
        "clean_suffix_parity": parity_rows,
        "max_clean_relative_l2_error": max(row["relative_l2_error"] for row in parity_rows),
        "max_backend_parity_relative_error": float(
            frame["backend_parity_relative_error"].max()
        ),
        "wrong_hook_sentinel": wrong_hook,
        "response_snr_definition": (
            "min(secant_norm, exact_jvp_norm) / "
            "max(in_batch_clean_repeat_norm, target_dtype_half_step_norm)"
        ),
        "batch_alignment": (
            "finite baseline, finite perturbation, and exact JVP use identical "
            "batch shape and request slot"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot-manifest", type=Path, default=DEFAULT_SNAPSHOT_MANIFEST)
    parser.add_argument("--max-cells", type=int, default=None)
    args = parser.parse_args()

    lock_path = local_root() / "locks/gm_exact_transport_gate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    process_lock = lock_path.open("w")
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another exact transport producer owns {lock_path}") from exc
    process_lock.write(str(os.getpid()))
    process_lock.flush()

    git = require_clean_tree()
    gpu = require_cuda()
    config = yaml.safe_load(args.config.read_text())
    role = config["models"]["olmo_positive_control"]
    snapshot_manifest = json.loads(args.snapshot_manifest.read_text())
    if not snapshot_manifest.get("all_content_hashes_verified"):
        raise RuntimeError("OLMo snapshot manifest is not fully hash-verified")
    if (
        snapshot_manifest["repo_id"] != role["model_id"]
        or snapshot_manifest["revision"] != role["revision"]
    ):
        raise RuntimeError("snapshot manifest model/revision mismatch")
    snapshot = Path(snapshot_manifest["snapshot"])
    prompt_bank = _resolve_repo_uri(config["prompt_bank"])
    prompts = _load_prompts(prompt_bank, config["stage1_prompt_ids"])
    environment = environment_payload(require_gpu=True)
    environment_compat = {key: value for key, value in environment.items() if key != "created_utc"}
    environment_sha = object_sha256(environment_compat)
    input_manifest = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "tier": "methods",
        "code_commit": git["code_commit"],
        "config": {"path": str(args.config), "sha256": file_sha256(args.config)},
        "prompt_bank": {"path": str(prompt_bank), "sha256": file_sha256(prompt_bank)},
        "snapshot_manifest": {
            "path": str(args.snapshot_manifest),
            "sha256": file_sha256(args.snapshot_manifest),
            "payload_sha256": snapshot_manifest["snapshot_manifest_sha256"],
        },
        "model_id": role["model_id"],
        "model_revision": role["revision"],
        "environment_sha256": environment_sha,
        "seed": config["directions"]["seed"],
        "target_thresholds_applied": False,
        "purpose": "OLMo-only calibration for thresholds frozen before Gemma",
    }
    input_manifest_sha = object_sha256(input_manifest)
    output_root = directory("metrics") / "olmo_control" / EVIDENCE_ID
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = output_root / "input_manifest.json"
    if input_path.exists():
        if json.loads(input_path.read_text()) != input_manifest:
            raise RuntimeError("existing calibration input manifest is incompatible")
    else:
        atomic_json(input_path, input_manifest)
    state = StateStore(
        output_root / "state.json",
        StateHeader(
            evidence_id=EVIDENCE_ID,
            config_sha256=input_manifest["config"]["sha256"],
            code_commit=git["code_commit"],
            model_id=role["model_id"],
            model_revision=role["revision"],
            environment_sha256=environment_sha,
        ),
    )
    progress = state.load() or {"completed_cells": {}, "parity": {}}

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
    if model.config.model_type != "olmo3" or len(model.model.layers) != 64:
        raise RuntimeError("loaded model is not the exact 64-layer OLMo control")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if "wrong_hook" not in progress:
        sentinel_prompt = prompts[0]
        sentinel_tokens = tokenizer(
            sentinel_prompt["text"], return_tensors="pt", truncation=True,
            max_length=96, add_special_tokens=True,
        )
        sentinel_ids = sentinel_tokens["input_ids"].to("cuda")
        sentinel_mask = sentinel_tokens.get(
            "attention_mask", torch.ones_like(sentinel_ids)
        ).to("cuda")
        shallow = int(role["shallow_negative_control_layer"])
        correct_source = ExplicitDecoderSuffix(
            model,
            input_ids=sentinel_ids,
            attention_mask=sentinel_mask,
            source_layer=shallow,
            target=TargetSpec("final_residual", position_indices=(-1,)),
        )
        wrong_receiver = ExplicitDecoderSuffix(
            model,
            input_ids=sentinel_ids,
            attention_mask=sentinel_mask,
            source_layer=shallow + 1,
            target=TargetSpec("final_residual", position_indices=(-1,)),
        )
        with torch.no_grad():
            wrong_target = wrong_receiver(correct_source.clean_source.float()).float()
            expected_target = wrong_receiver(wrong_receiver.clean_source.float()).float()
        relative_error = float(
            (wrong_target - expected_target).norm()
            / expected_target.norm().clamp_min(1e-30)
        )
        if relative_error == 0:
            raise RuntimeError("wrong-hook sentinel unexpectedly matches the clean path")
        progress["wrong_hook"] = {
            "prompt_id": sentinel_prompt["prompt_id"],
            "source_activation_layer": shallow,
            "suffix_expected_source_layer": shallow + 1,
            "relative_l2_error": relative_error,
            "purpose": "calibration baseline only; threshold frozen before Gemma",
        }
        state.write(progress)
        del correct_source, wrong_receiver
        torch.cuda.empty_cache()

    completed_this_invocation = 0
    target = TargetSpec("final_residual", position_indices=(-1,))
    for prompt in prompts:
        encoded = tokenizer(
            prompt["text"],
            return_tensors="pt",
            truncation=True,
            max_length=96,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to("cuda")
        prompt_sha = hashlib.sha256(prompt["text"].encode()).hexdigest()
        token_sha = hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest()
        for layer in role["layers_zero_indexed"]:
            suffix = ExplicitDecoderSuffix(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                source_layer=int(layer),
                target=target,
            )
            parity_key = f"{prompt['prompt_id']}:L{layer}"
            if parity_key not in progress["parity"]:
                parity = suffix.parity(atol=0.0, rtol=0.0)
                if not parity["ok"]:
                    raise RuntimeError(f"clean suffix parity failed: {parity_key}: {parity}")
                progress["parity"][parity_key] = parity
                state.write(progress)
            for source_mode in config["source_modes"]:
                mode = source_mode["mode"]
                cell_id = f"{prompt['prompt_id']}-L{int(layer):02d}-{mode}"
                if cell_id in progress["completed_cells"]:
                    row = progress["completed_cells"][cell_id]
                    for field in ("metrics", "raw"):
                        path = Path(row[field]["path"])
                        if not path.exists() or file_sha256(path) != row[field]["sha256"]:
                            raise RuntimeError(f"completed cell hash drift: {cell_id} {field}")
                    continue
                if args.max_cells is not None and completed_this_invocation >= args.max_cells:
                    print(json.dumps({"status": "paused_by_max_cells", "completed": completed_this_invocation}, indent=1))
                    return
                started = time.time()
                metadata = {
                    "prompt_id": prompt["prompt_id"],
                    "prompt_sha256": prompt_sha,
                    "prompt_stratum": prompt["stratum"],
                    "prompt_family": prompt["family"],
                    "token_ids_sha256": token_sha,
                    "sequence_length": int(input_ids.shape[1]),
                    "model_id": role["model_id"],
                    "model_revision": role["revision"],
                    "source_layer": int(layer),
                    "source_position": -1 if mode == "single_position" else "all_valid",
                    "target_stage": "final_decoder_layer_output",
                    "target_representation": "final_residual",
                    "block_type": model.config.layer_types[int(layer)],
                    "variant": "R0-full-live",
                    "implementation_sha256": file_sha256(
                        PACKAGE_ROOT / "jspace_gemma/autodiff.py"
                    ),
                    "transport_implementation_sha256": file_sha256(
                        PACKAGE_ROOT / "jspace_gemma/transport.py"
                    ),
                    "code_commit": git["code_commit"],
                    "config_sha256": input_manifest["config"]["sha256"],
                    "environment_sha256": environment_sha,
                }
                rows, raw = evaluate_transport_cell(
                    suffix,
                    attention_mask=attention_mask,
                    perturbation_mode=mode,
                    direction_specs=config["directions"]["stage1"],
                    epsilon_ladder=config["relative_epsilon_ladder"],
                    seed=config["directions"]["seed"],
                    cell_id=cell_id,
                    metadata=metadata,
                    delivery_cosine_floor=config["delivery_gate"]["cosine_floor"],
                    delivery_norm_error_ceiling=config["delivery_gate"]["relative_norm_error_ceiling"],
                    batch_size=config["finite_response_batch_size"],
                )
                elapsed = time.time() - started
                for row in rows:
                    row["cell_runtime_seconds"] = elapsed
                raw["cell_runtime_seconds"] = elapsed
                raw_path = output_root / "raw" / f"{cell_id}.pt"
                metrics_path = output_root / "cells" / f"{cell_id}.json"
                _atomic_torch_save(raw_path, raw)
                atomic_json(metrics_path, {"schema_version": 2, "rows": rows})
                progress["completed_cells"][cell_id] = {
                    "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
                    "raw": {"path": str(raw_path), "sha256": file_sha256(raw_path)},
                    "runtime_seconds": elapsed,
                }
                state.write(progress)
                completed_this_invocation += 1
                print(
                    f"completed {cell_id} in {elapsed:.1f}s; "
                    f"total={len(progress['completed_cells'])}",
                    flush=True,
                )
                if elapsed > config["cell_runtime_target_minutes"] * 60:
                    raise RuntimeError(
                        f"cell exceeded checkpoint runtime target: {cell_id} {elapsed:.1f}s"
                    )
            del suffix
            torch.cuda.empty_cache()

    all_rows = []
    for cell_id in sorted(progress["completed_cells"]):
        path = Path(progress["completed_cells"][cell_id]["metrics"]["path"])
        all_rows.extend(json.loads(path.read_text())["rows"])
    expected_cells = len(prompts) * len(role["layers_zero_indexed"]) * len(config["source_modes"])
    if len(progress["completed_cells"]) != expected_cells:
        raise RuntimeError("calibration grid is incomplete")
    summary_payload = _aggregate(
        all_rows, list(progress["parity"].values()), progress["wrong_hook"]
    )
    summary_payload.update(
        {
            "schema_version": 1,
            "evidence_id": EVIDENCE_ID,
            "tier": "methods",
            "model_id": role["model_id"],
            "model_revision": role["revision"],
            "input_manifest_sha256": input_manifest_sha,
            "expected_cells": expected_cells,
            "completed_cells": len(progress["completed_cells"]),
            "environment": environment,
            "gpu_gate": gpu,
            "claim_boundary": "OLMo-only threshold calibration; no Gemma target threshold or result",
        }
    )
    summary_path = output_root / "olmo_calibration_summary.json"
    rows_path = output_root / "olmo_calibration_rows.parquet"
    inventory_path = output_root / "raw_inventory.json"
    atomic_json(summary_path, summary_payload)
    pd.DataFrame(all_rows).to_parquet(rows_path, index=False)
    raw_inventory = {
        "schema_version": 1,
        "files": [
            {"path": value[field]["path"], "sha256": value[field]["sha256"]}
            for value in progress["completed_cells"].values()
            for field in ("metrics", "raw")
        ],
    }
    raw_inventory["inventory_sha256"] = object_sha256(raw_inventory["files"])
    atomic_json(inventory_path, raw_inventory)
    create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "OLMo-only exact-JVP/secant calibration grid used to freeze G1 "
            "thresholds before any Gemma target result"
        ),
        command="python -m jspace_gemma.experiments.gm_exact_transport_gate",
        outputs=[input_path, summary_path, rows_path, inventory_path],
        inputs={
            "input_manifest_sha256": input_manifest_sha,
            "snapshot_manifest_sha256": file_sha256(args.snapshot_manifest),
        },
        target_model_opened=False,
    )
    print(json.dumps({
        "summary": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "rows": len(all_rows),
        "cells": expected_cells,
    }, indent=1))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
