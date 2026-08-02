"""Resumable Gemma G1 Stage-1 exact-JVP/secant producer."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from jspace_gemma.architecture import audit_loaded_model, decoder_components
from jspace_gemma.calibration_audit import audit_completed_checkpoint
from jspace_gemma.experiments.gm_stage_gemma import require_gemma_staging_gate
from jspace_gemma.gpu import require_cuda
from jspace_gemma.hooks import ExplicitDecoderSuffix, TargetSpec
from jspace_gemma.manifests import (
    atomic_json,
    environment_payload,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import PACKAGE_ROOT, directory, local_root, resolve_uri
from jspace_gemma.registry import create, read_events
from jspace_gemma.stage1_analysis import analyze_stage1
from jspace_gemma.staging import verify_snapshot
from jspace_gemma.state import StateHeader, StateStore
from jspace_gemma.transport import evaluate_transport_cell

EVIDENCE_ID = "gm-jvp-gemma-stage1-v1"
DEFAULT_EXECUTION = PACKAGE_ROOT / "configs/gm_g1_stage1_execution.yaml"


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    if len(pd.read_parquet(temporary)) != len(frame):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Parquet round-trip row-count mismatch: {path}")
    os.replace(temporary, path)


def _parquet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "source_position" in result:
        result["source_position_runtime_type"] = result["source_position"].map(
            lambda value: type(value).__name__
        )
        result["source_position"] = result["source_position"].map(str)
    return result


def _json_native(value):
    if isinstance(value, dict):
        return {str(key): _json_native(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_prompts(path: Path, identifiers: list[str]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {row["prompt_id"]: row for row in rows}
    missing = set(identifiers) - set(by_id)
    if missing:
        raise RuntimeError(f"frozen Stage-1 prompts are absent: {sorted(missing)}")
    return [by_id[value] for value in identifiers]


def _snapshot_payload(path: Path, execution: dict) -> dict:
    expected = execution["snapshot_manifest"]
    if file_sha256(path) != expected["file_sha256"]:
        raise RuntimeError("Gemma snapshot manifest file hash drifted")
    payload = json.loads(path.read_text())
    recorded_payload_sha = payload.pop("snapshot_manifest_sha256", None)
    if (
        recorded_payload_sha != expected["payload_sha256"]
        or object_sha256(payload) != recorded_payload_sha
    ):
        raise RuntimeError("Gemma snapshot manifest payload hash drifted")
    payload["snapshot_manifest_sha256"] = recorded_payload_sha
    required = {
        "repo_id": execution["model"]["model_id"],
        "revision": execution["model"]["revision"],
        "snapshot": execution["model"]["local_snapshot"],
        "remote_inventory_sha256": expected["remote_inventory_sha256"],
        "staging_code_commit": expected["staging_code_commit"],
        "all_content_hashes_verified": True,
        "target_model_loaded": False,
        "target_response_created": False,
    }
    mismatches = {
        key: {"expected": wanted, "actual": payload.get(key)}
        for key, wanted in required.items()
        if payload.get(key) != wanted
    }
    if mismatches:
        raise RuntimeError(
            "Gemma snapshot manifest contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return payload


def _remote_inventory_from_manifest(payload: dict) -> dict:
    files = [
        {
            "path": row["path"],
            "size_bytes": row["expected_size_bytes"],
            "lfs_sha256": row["expected_lfs_sha256"],
            "git_blob_id": row["expected_git_blob_id"],
        }
        for row in payload["files"]
    ]
    return {
        "repo_id": payload["repo_id"],
        "revision": payload["revision"],
        "files": files,
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "inventory_sha256": payload["remote_inventory_sha256"],
    }


def _validate_execution(execution_path: Path) -> tuple[dict, dict, dict, Path, Path, Path]:
    execution = yaml.safe_load(execution_path.read_text())
    if execution["status"] != "READY_PRE_TARGET" or execution["evidence_id"] != EVIDENCE_ID:
        raise RuntimeError("Stage-1 execution manifest is not at READY_PRE_TARGET")
    design_path = resolve_uri(execution["frozen_inputs"]["design_path"])
    thresholds_path = resolve_uri(execution["frozen_inputs"]["threshold_path"])
    prompt_path = resolve_uri(execution["frozen_inputs"]["prompt_bank_path"])
    for label, path in (
        ("design", design_path),
        ("threshold", thresholds_path),
        ("prompt_bank", prompt_path),
    ):
        if file_sha256(path) != execution["frozen_inputs"][f"{label}_sha256"]:
            raise RuntimeError(f"frozen Stage-1 {label} hash drifted")
    design = yaml.safe_load(design_path.read_text())
    thresholds = yaml.safe_load(thresholds_path.read_text())
    gate = require_gemma_staging_gate(
        config_path=design_path,
        thresholds_path=thresholds_path,
    )
    if (
        gate["positive_control_artifact_sha256"]
        != execution["frozen_inputs"]["positive_control_artifact_sha256"]
        or gate["threshold_config_sha256"]
        != execution["frozen_inputs"]["threshold_sha256"]
    ):
        raise RuntimeError("live pre-target gate differs from the execution manifest")
    grid = execution["grid"]
    role = design["models"]["gemma_target"]
    expected_modes = [row["mode"] for row in design["source_modes"]]
    expected_directions = [row["id"] for row in design["directions"]["stage1"]]
    frozen_grid = {
        "prompt_ids": design["stage1_prompt_ids"],
        "layers_zero_indexed": role["layers_zero_indexed"],
        "modes": expected_modes,
        "direction_ids": expected_directions,
        "epsilon_ladder": design["relative_epsilon_ladder"],
    }
    for key, expected_value in frozen_grid.items():
        if grid[key] != expected_value:
            raise RuntimeError(f"execution grid differs from frozen design: {key}")
    expected_cells = len(grid["prompt_ids"]) * len(grid["layers_zero_indexed"]) * len(grid["modes"])
    expected_rows = expected_cells * len(grid["direction_ids"]) * len(grid["epsilon_ladder"])
    if (
        grid["expected_cells"] != expected_cells
        or grid["expected_rows_per_cell"]
        != len(grid["direction_ids"]) * len(grid["epsilon_ladder"])
        or grid["expected_rows"] != expected_rows
        or grid["expected_clean_parity_rows"]
        != len(grid["prompt_ids"]) * len(grid["layers_zero_indexed"])
    ):
        raise RuntimeError("Stage-1 expected grid counts are inconsistent")
    snapshot_path = Path(execution["snapshot_manifest"]["path"])
    snapshot = _snapshot_payload(snapshot_path, execution)
    return execution, design, thresholds, design_path, thresholds_path, prompt_path


def _verify_staged_bytes(execution: dict) -> tuple[dict, dict]:
    manifest_path = Path(execution["snapshot_manifest"]["path"])
    snapshot = _snapshot_payload(manifest_path, execution)
    verification = verify_snapshot(
        snapshot["snapshot"],
        repo_id=snapshot["repo_id"],
        revision=snapshot["revision"],
        remote_inventory=_remote_inventory_from_manifest(snapshot),
    )
    if (
        not verification["all_content_hashes_verified"]
        or verification["remote_inventory_sha256"]
        != execution["snapshot_manifest"]["remote_inventory_sha256"]
    ):
        raise RuntimeError("Gemma local snapshot failed the mandatory pre-load rehash")
    return snapshot, verification


def _model_audit_payload(model, *, git: dict, environment_sha: str, snapshot_sha: str) -> dict:
    payload = audit_loaded_model(model)
    payload.update(
        {
            "schema_version": 1,
            "model_id": "google/gemma-4-31B-it",
            "model_revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "code_commit": git["code_commit"],
            "environment_sha256": environment_sha,
            "snapshot_manifest_sha256": snapshot_sha,
        }
    )
    return payload


def _record_or_validate(path: Path, payload: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"existing immutable output differs: {path}")
    else:
        atomic_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None)
    args = parser.parse_args()
    if args.smoke and args.max_cells is not None:
        raise RuntimeError("--smoke and --max-cells are mutually exclusive")
    if args.max_cells is not None and args.max_cells < 1:
        raise RuntimeError("--max-cells must be positive")

    lock_path = local_root() / "locks/gm_gemma_stage1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    process_lock = lock_path.open("w")
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another Gemma Stage-1 producer owns {lock_path}") from exc
    process_lock.write(str(os.getpid()))
    process_lock.flush()

    git = require_clean_tree()
    gpu = require_cuda()
    execution, design, thresholds, design_path, thresholds_path, prompt_path = (
        _validate_execution(args.execution)
    )
    origins = {
        row["evidence_id"]
        for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
    }
    if EVIDENCE_ID in origins:
        raise RuntimeError("Gemma Stage-1 evidence is already registered")

    output_root = directory("metrics") / "gemma_target" / EVIDENCE_ID
    final_outputs = {
        "summary": output_root / "gemma_stage1_summary.json",
        "rows": output_root / "gemma_stage1_rows.parquet",
        "selected": output_root / "gemma_stage1_smallest_evaluable.parquet",
        "fits": output_root / "gemma_stage1_curvature_fits.parquet",
        "raw_inventory": output_root / "raw_inventory.json",
    }
    existing_final = [str(path) for path in final_outputs.values() if path.exists()]
    if existing_final:
        raise RuntimeError(
            "refusing to overwrite unregistered Stage-1 final outputs: "
            + json.dumps(existing_final)
        )

    snapshot, snapshot_verification = _verify_staged_bytes(execution)
    prompts = _load_prompts(prompt_path, execution["grid"]["prompt_ids"])
    environment = environment_payload(require_gpu=True)
    environment_compat = {
        key: value for key, value in environment.items() if key != "created_utc"
    }
    environment_sha = object_sha256(environment_compat)
    execution_sha = file_sha256(args.execution)
    combined_config_sha = object_sha256(
        {
            "execution": execution_sha,
            "design": file_sha256(design_path),
            "thresholds": file_sha256(thresholds_path),
        }
    )
    input_manifest = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "tier": "methods",
        "code_commit": git["code_commit"],
        "config": {"path": str(args.execution), "sha256": combined_config_sha},
        "execution": {"path": str(args.execution), "sha256": execution_sha},
        "design": {"path": str(design_path), "sha256": file_sha256(design_path)},
        "thresholds": {
            "path": str(thresholds_path),
            "sha256": file_sha256(thresholds_path),
            "frozen_before_first_gemma_result": True,
        },
        "prompt_bank": {"path": str(prompt_path), "sha256": file_sha256(prompt_path)},
        "snapshot_manifest": {
            "path": execution["snapshot_manifest"]["path"],
            "sha256": file_sha256(execution["snapshot_manifest"]["path"]),
            "payload_sha256": snapshot["snapshot_manifest_sha256"],
            "remote_inventory_sha256": snapshot_verification[
                "remote_inventory_sha256"
            ],
        },
        "positive_control": require_gemma_staging_gate(
            config_path=design_path, thresholds_path=thresholds_path
        ),
        "model_id": execution["model"]["model_id"],
        "model_revision": execution["model"]["revision"],
        "environment_sha256": environment_sha,
        "combined_config_sha256": combined_config_sha,
        "seed": design["directions"]["seed"],
        "target_thresholds_applied": True,
        "expected_cells": execution["grid"]["expected_cells"],
        "expected_rows": execution["grid"]["expected_rows"],
        "purpose": "Gemma Stage-1 exact-JVP transport validity gate",
    }
    input_manifest_sha = object_sha256(input_manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = output_root / "input_manifest.json"
    _record_or_validate(input_path, input_manifest)
    state = StateStore(
        output_root / "state.json",
        StateHeader(
            evidence_id=EVIDENCE_ID,
            config_sha256=combined_config_sha,
            code_commit=git["code_commit"],
            model_id=execution["model"]["model_id"],
            model_revision=execution["model"]["revision"],
            environment_sha256=environment_sha,
        ),
    )
    progress = state.load() or {"completed_cells": {}, "parity": {}}

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(snapshot["snapshot"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot["snapshot"],
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    ).eval()
    base, _, text_config, _ = decoder_components(model)
    if (
        model.config.model_type != execution["model"]["expected_outer_model_type"]
        or text_config.model_type != execution["model"]["expected_text_model_type"]
        or len(base.layers) != execution["model"]["expected_decoder_layers"]
        or getattr(text_config, "_attn_implementation", None) != "eager"
    ):
        raise RuntimeError("loaded Gemma model fails the frozen architecture contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    audit_path = output_root / "loaded_model_audit.json"
    _record_or_validate(
        audit_path,
        _model_audit_payload(
            model,
            git=git,
            environment_sha=environment_sha,
            snapshot_sha=file_sha256(execution["snapshot_manifest"]["path"]),
        ),
    )

    target = TargetSpec("final_residual", position_indices=(-1,))
    if "wrong_hook" not in progress:
        prompt = prompts[0]
        encoded = tokenizer(
            prompt["text"],
            return_tensors="pt",
            truncation=True,
            max_length=96,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
            "cuda"
        )
        source_layer = int(execution["grid"]["layers_zero_indexed"][0])
        correct_source = ExplicitDecoderSuffix(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            source_layer=source_layer,
            target=target,
        )
        wrong_receiver = ExplicitDecoderSuffix(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            source_layer=source_layer + 1,
            target=target,
        )
        with torch.no_grad():
            wrong_target = wrong_receiver(correct_source.clean_source.float()).float()
            expected_target = wrong_receiver(wrong_receiver.clean_source.float()).float()
        relative_error = float(
            (wrong_target - expected_target).norm()
            / expected_target.norm().clamp_min(1e-30)
        )
        progress["wrong_hook"] = {
            "prompt_id": prompt["prompt_id"],
            "source_activation_layer": source_layer,
            "suffix_expected_source_layer": source_layer + 1,
            "relative_l2_error": relative_error,
            "frozen_floor": thresholds["wrong_hook_relative_error_floor"],
            "pass": relative_error >= thresholds["wrong_hook_relative_error_floor"],
        }
        state.write(progress)
        del correct_source, wrong_receiver
        torch.cuda.empty_cache()
        if not progress["wrong_hook"]["pass"]:
            raise RuntimeError("Gemma wrong-hook sentinel failed the frozen mismatch floor")
    elif not progress["wrong_hook"].get("pass"):
        raise RuntimeError("checkpoint contains a failed wrong-hook sentinel")

    cells = []
    modes = {row["mode"]: row for row in design["source_modes"]}
    for prompt in prompts:
        for layer in execution["grid"]["layers_zero_indexed"]:
            for mode in execution["grid"]["modes"]:
                cell_id = f"{prompt['prompt_id']}-L{int(layer):02d}-{mode}"
                cells.append((cell_id, prompt, int(layer), modes[mode]))
    if args.smoke:
        smoke_id = execution["smoke"]["cell_id"]
        cells = [row for row in cells if row[0] == smoke_id]
        if len(cells) != 1:
            raise RuntimeError("predeclared smoke cell is absent from the frozen grid")

    completed_this_invocation = 0
    for cell_id, prompt, layer, source_mode in cells:
        if cell_id in progress["completed_cells"]:
            entry = progress["completed_cells"][cell_id]
            for field in ("metrics", "raw"):
                path = Path(entry[field]["path"])
                if not path.exists() or file_sha256(path) != entry[field]["sha256"]:
                    raise RuntimeError(f"completed Stage-1 cell hash drift: {cell_id} {field}")
            continue
        if args.max_cells is not None and completed_this_invocation >= args.max_cells:
            print(
                json.dumps(
                    {
                        "status": "paused_by_max_cells",
                        "completed_this_invocation": completed_this_invocation,
                        "completed_total": len(progress["completed_cells"]),
                    },
                    indent=1,
                )
            )
            return
        encoded = tokenizer(
            prompt["text"],
            return_tensors="pt",
            truncation=True,
            max_length=96,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(
            "cuda"
        )
        prompt_sha = hashlib.sha256(prompt["text"].encode()).hexdigest()
        token_sha = hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest()
        suffix = ExplicitDecoderSuffix(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            source_layer=layer,
            target=target,
        )
        parity_key = f"{prompt['prompt_id']}:L{layer}"
        if parity_key not in progress["parity"]:
            parity = suffix.parity(
                atol=execution["runtime"]["clean_suffix_parity_atol"],
                rtol=execution["runtime"]["clean_suffix_parity_rtol"],
            )
            if not parity["ok"]:
                raise RuntimeError(f"Gemma clean suffix parity failed: {parity_key}: {parity}")
            progress["parity"][parity_key] = parity
            state.write(progress)
        started = time.time()
        mode = source_mode["mode"]
        metadata = {
            "prompt_id": prompt["prompt_id"],
            "prompt_sha256": prompt_sha,
            "prompt_stratum": prompt["stratum"],
            "prompt_family": prompt["family"],
            "token_ids_sha256": token_sha,
            "sequence_length": int(input_ids.shape[1]),
            "model_id": execution["model"]["model_id"],
            "model_revision": execution["model"]["revision"],
            "source_layer": int(layer),
            "source_position": -1 if mode == "single_position" else "all_valid",
            "target_stage": "final_decoder_layer_output",
            "target_representation": "final_residual",
            "block_type": text_config.layer_types[layer],
            "variant": "G1-stage1-R0-full-live",
            "implementation_sha256": file_sha256(
                PACKAGE_ROOT / "jspace_gemma/autodiff.py"
            ),
            "transport_implementation_sha256": file_sha256(
                PACKAGE_ROOT / "jspace_gemma/transport.py"
            ),
            "analysis_implementation_sha256": file_sha256(
                PACKAGE_ROOT / "jspace_gemma/stage1_analysis.py"
            ),
            "code_commit": git["code_commit"],
            "config_sha256": combined_config_sha,
            "threshold_config_sha256": file_sha256(thresholds_path),
            "positive_control_artifact_sha256": input_manifest["positive_control"][
                "positive_control_artifact_sha256"
            ],
            "environment_sha256": environment_sha,
        }
        rows, raw = evaluate_transport_cell(
            suffix,
            attention_mask=attention_mask,
            perturbation_mode=mode,
            direction_specs=design["directions"]["stage1"],
            epsilon_ladder=design["relative_epsilon_ladder"],
            seed=design["directions"]["seed"],
            cell_id=cell_id,
            metadata=metadata,
            delivery_cosine_floor=thresholds["delivery"]["cosine_floor"],
            delivery_norm_error_ceiling=thresholds["delivery"][
                "relative_norm_error_ceiling"
            ],
            batch_size=execution["runtime"]["finite_response_batch_size"],
        )
        elapsed = time.time() - started
        for row in rows:
            row["cell_runtime_seconds"] = elapsed
        raw["cell_runtime_seconds"] = elapsed
        json.dumps({"schema_version": 2, "rows": rows}, allow_nan=False)
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
        if elapsed > execution["runtime"]["cell_runtime_target_minutes"] * 60:
            raise RuntimeError(f"Stage-1 cell exceeded runtime target: {cell_id}")
        del suffix
        torch.cuda.empty_cache()

    if args.smoke:
        smoke_audit = audit_completed_checkpoint(
            output_root,
            expected_evidence_id=EVIDENCE_ID,
            expected_compute_commit=git["code_commit"],
            expected_cells=len(progress["completed_cells"]),
            expected_rows_per_cell=execution["grid"]["expected_rows_per_cell"],
            expected_parity_rows=len(progress["parity"]),
            inspect_raw_tensors=True,
        )
        print(
            json.dumps(
                {
                    "status": "smoke_complete_unregistered",
                    "cell_id": execution["smoke"]["cell_id"],
                    "state_sha256": smoke_audit["state_sha256"],
                    "inventory_sha256": smoke_audit["inventory_sha256"],
                    "rows": smoke_audit["rows"],
                    "target_model_opened": True,
                },
                indent=1,
            )
        )
        return

    if len(progress["completed_cells"]) != execution["grid"]["expected_cells"]:
        raise RuntimeError("Gemma Stage-1 grid is incomplete")
    if len(progress["parity"]) != execution["grid"]["expected_clean_parity_rows"]:
        raise RuntimeError("Gemma clean-parity grid is incomplete")
    checkpoint = audit_completed_checkpoint(
        output_root,
        expected_evidence_id=EVIDENCE_ID,
        expected_compute_commit=git["code_commit"],
        expected_cells=execution["grid"]["expected_cells"],
        expected_rows_per_cell=execution["grid"]["expected_rows_per_cell"],
        expected_parity_rows=execution["grid"]["expected_clean_parity_rows"],
        inspect_raw_tensors=True,
    )
    analysis, selected, fits = analyze_stage1(
        checkpoint["all_rows"],
        thresholds=thresholds,
        prompt_ids=execution["grid"]["prompt_ids"],
        layers=execution["grid"]["layers_zero_indexed"],
        direction_ids=execution["grid"]["direction_ids"],
    )
    state_payload = json.loads(Path(checkpoint["state_path"]).read_text())["payload"]
    parity = [row for _, row in sorted(state_payload["parity"].items())]
    summary = _json_native(
        {
            "schema_version": 1,
            "evidence_id": EVIDENCE_ID,
            "tier": "methods",
            "model_id": execution["model"]["model_id"],
            "model_revision": execution["model"]["revision"],
            "cell_compute_code_commit": git["code_commit"],
            "input_manifest_sha256": input_manifest_sha,
            "execution_config_sha256": execution_sha,
            "threshold_config_sha256": file_sha256(thresholds_path),
            "positive_control_evidence_id": input_manifest["positive_control"][
                "threshold_evidence_id"
            ],
            "positive_control_artifact_sha256": input_manifest["positive_control"][
                "positive_control_artifact_sha256"
            ],
            "snapshot_manifest_sha256": file_sha256(
                execution["snapshot_manifest"]["path"]
            ),
            "loaded_model_audit_sha256": file_sha256(audit_path),
            "environment": environment,
            "gpu_gate": gpu,
            "expected_cells": execution["grid"]["expected_cells"],
            "completed_cells": checkpoint["completed_cells"],
            "expected_rows": execution["grid"]["expected_rows"],
            "state_sha256": checkpoint["state_sha256"],
            "checkpoint_inventory_sha256": checkpoint["inventory_sha256"],
            "clean_suffix_parity": parity,
            "max_clean_relative_l2_error": max(
                row["relative_l2_error"] for row in parity
            ),
            "max_backend_parity_relative_error": max(
                row["backend_parity_relative_error"]
                for row in checkpoint["all_rows"]
            ),
            "wrong_hook_sentinel": checkpoint["wrong_hook"],
            "analysis": analysis,
            "thresholds": thresholds,
            "model_loaded_during_compute": True,
            "target_model_opened": True,
            "model_response_data_created": True,
            "thresholds_frozen_before_target": True,
            "j_selected_direction_state": execution["grid"]["j_selected_direction"],
            "parquet_normalizations": {
                "source_position": (
                    "canonical string plus source_position_runtime_type; "
                    "per-cell JSON retains the original int/string union"
                )
            },
            "claim_boundary": execution["claim_boundary"],
        }
    )
    json.dumps(summary, allow_nan=False, sort_keys=True)
    raw_inventory = {
        "schema_version": 1,
        "compute_code_commit": git["code_commit"],
        "state_sha256": checkpoint["state_sha256"],
        "files": checkpoint["inventory"],
        "inventory_sha256": checkpoint["inventory_sha256"],
    }
    _atomic_parquet(
        final_outputs["rows"], _parquet_frame(pd.DataFrame(checkpoint["all_rows"]))
    )
    _atomic_parquet(final_outputs["selected"], _parquet_frame(selected))
    _atomic_parquet(final_outputs["fits"], fits)
    atomic_json(final_outputs["raw_inventory"], raw_inventory)
    atomic_json(final_outputs["summary"], summary)
    create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "Gemma 4 31B Stage-1 prompt-specific exact-JVP/secant transport "
            "gate under frozen pre-target OLMo-calibrated thresholds"
        ),
        command="python -m jspace_gemma.experiments.gm_run_gemma_stage1",
        outputs=[
            input_path,
            audit_path,
            Path(checkpoint["state_path"]),
            final_outputs["summary"],
            final_outputs["rows"],
            final_outputs["selected"],
            final_outputs["fits"],
            final_outputs["raw_inventory"],
        ],
        inputs={
            "execution_config_sha256": execution_sha,
            "threshold_config_sha256": file_sha256(thresholds_path),
            "positive_control_artifact_sha256": input_manifest["positive_control"][
                "positive_control_artifact_sha256"
            ],
            "snapshot_manifest_sha256": file_sha256(
                execution["snapshot_manifest"]["path"]
            ),
        },
        target_model_opened=True,
        model_response_data_created=True,
        thresholds_frozen_before_target=True,
        positive_control_evidence_id=input_manifest["positive_control"][
            "threshold_evidence_id"
        ],
        j_selected_direction_state=execution["grid"]["j_selected_direction"],
    )
    print(
        json.dumps(
            {
                "summary": str(final_outputs["summary"]),
                "summary_sha256": file_sha256(final_outputs["summary"]),
                "rows": checkpoint["rows"],
                "cells": checkpoint["completed_cells"],
                "state_sha256": checkpoint["state_sha256"],
                "layer_decisions": analysis["primary_layer_decisions"],
            },
            indent=1,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
