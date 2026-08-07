"""Actual-Gemma replay comparing two exact autodiff JVP implementations."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from jspace_gemma.architecture import decoder_components
from jspace_gemma.autodiff import exact_jvp
from jspace_gemma.experiments.gm_run_gemma_stage1 import (
    DEFAULT_EXECUTION,
    _load_prompts,
    _validate_execution,
    _verify_staged_bytes,
)
from jspace_gemma.gpu import require_cuda
from jspace_gemma.hooks import (
    ExplicitDecoderSuffix,
    TargetSpec,
    delivery_audit,
    patterned_direction,
    source_mask,
)
from jspace_gemma.manifests import (
    atomic_json,
    environment_payload,
    file_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import PACKAGE_ROOT, directory, local_root
from jspace_gemma.registry import create, read_events, resolve
from jspace_gemma.transport import make_directions, tensor_sha256
from jspace_gemma.transport_metrics import tangent_metrics

EVIDENCE_ID = "gm-jvp-gemma-backend-parity-v1"
CONFIG = PACKAGE_ROOT / "configs/gm_g1_backend_parity.yaml"
STAGE1_ROOT = (
    directory("metrics") / "gemma_target" / "gm-jvp-gemma-stage1-v1"
)


def _relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (value.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-30)
    )


def _comparison(value: torch.Tensor, reference: torch.Tensor) -> dict:
    left = value.detach().float().cpu().reshape(-1)
    right = reference.detach().float().cpu().reshape(-1)
    return {
        "relative_error": _relative_error(left, right),
        "cosine": float(F.cosine_similarity(left, right, dim=0)),
        "max_absolute_error": float((left - right).abs().max()),
        "value_norm": float(left.norm()),
        "reference_norm": float(right.norm()),
        "elements": int(left.numel()),
    }


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _source_cell(config: dict) -> tuple[dict, dict, dict, Path, Path]:
    stage1 = resolve(config["source_stage1"]["evidence_id"])
    if (
        not stage1["live"]
        or stage1["code_commit"] != config["source_stage1"]["compute_commit"]
        or stage1.get("target_model_opened") is not True
    ):
        raise RuntimeError("registered Stage-1 source event is absent or incompatible")
    registered = {Path(row["path"]): row["sha256"] for row in stage1["outputs"]}
    summary_path = STAGE1_ROOT / "gemma_stage1_summary.json"
    state_path = STAGE1_ROOT / "state.json"
    for path, expected in (
        (summary_path, config["source_stage1"]["summary_sha256"]),
        (state_path, config["source_stage1"]["state_sha256"]),
    ):
        if file_sha256(path) != expected or registered.get(path) != expected:
            raise RuntimeError(f"registered Stage-1 source hash drifted: {path}")
    state = json.loads(state_path.read_text())
    cell_id = config["source_stage1"]["cell_id"]
    entry = state["payload"]["completed_cells"].get(cell_id)
    if entry is None:
        raise RuntimeError("diagnostic source cell is absent from Stage-1 state")
    metrics_path = Path(entry["metrics"]["path"])
    raw_path = Path(entry["raw"]["path"])
    if (
        file_sha256(metrics_path) != entry["metrics"]["sha256"]
        or file_sha256(raw_path) != entry["raw"]["sha256"]
    ):
        raise RuntimeError("diagnostic source cell hash drifted")
    metrics = json.loads(metrics_path.read_text())
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    return stage1, state, metrics, metrics_path, raw_path


def main() -> None:
    lock_path = local_root() / "locks/gm_gemma_backend_parity.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    process_lock = lock_path.open("w")
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            f"another Gemma backend-parity producer owns {lock_path}"
        ) from exc
    process_lock.write(str(os.getpid()))
    process_lock.flush()

    git = require_clean_tree()
    gpu = require_cuda()
    config = yaml.safe_load(CONFIG.read_text())
    if (
        config["status"] != "FROZEN_PRE_DIAGNOSTIC"
        or config["evidence_id"] != EVIDENCE_ID
    ):
        raise RuntimeError("backend-parity config is not frozen pre-diagnostic")
    if config["exact_backends"] != {
        "primary": "torch.func.jvp",
        "independent_fallback": "torch.autograd.functional.jvp",
        "finite_difference_as_exact": "forbidden",
    }:
        raise RuntimeError("backend-parity config does not bind both exact backends")
    origins = {
        row["evidence_id"]
        for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
    }
    if EVIDENCE_ID in origins:
        raise RuntimeError("Gemma backend-parity evidence already exists")
    output_root = directory("metrics") / "gemma_target"
    artifact_path = output_root / f"{EVIDENCE_ID}.json"
    raw_output = output_root / f"{EVIDENCE_ID}.pt"
    existing = [str(path) for path in (artifact_path, raw_output) if path.exists()]
    if existing:
        raise RuntimeError(
            "refusing to overwrite unregistered backend-parity outputs: "
            + json.dumps(existing)
        )

    stage1, stage1_state, metrics, metrics_path, raw_path = _source_cell(config)
    execution, design, thresholds, _, _, prompt_path = _validate_execution(
        DEFAULT_EXECUTION
    )
    snapshot, verification = _verify_staged_bytes(execution)
    prompt_id = config["source_stage1"]["prompt_id"]
    prompt = _load_prompts(prompt_path, [prompt_id])[0]
    target_rows = [
        row
        for row in metrics["rows"]
        if row["direction_id"] == config["source_stage1"]["direction_id"]
        and row["desired_relative_epsilon"]
        == config["source_stage1"]["relative_epsilon"]
    ]
    if len(target_rows) != 1:
        raise RuntimeError("expected one frozen diagnostic metric row")
    stored_row = target_rows[0]
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    stored_records = [
        row
        for row in raw["records"]
        if row["direction_id"] == config["source_stage1"]["direction_id"]
        and row["epsilon"] == config["source_stage1"]["relative_epsilon"]
    ]
    if len(stored_records) != 1:
        raise RuntimeError("expected one frozen diagnostic raw record")
    stored_raw = stored_records[0]
    source = config["source_stage1"]
    source_contract = {
        "prompt_id": source["prompt_id"],
        "source_layer": source["source_layer"],
        "perturbation_mode": source["perturbation_mode"],
        "exact_jvp_backend": source["stored_exact_backend"],
    }
    mismatches = {
        key: {"expected": expected, "actual": stored_row.get(key)}
        for key, expected in source_contract.items()
        if stored_row.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "frozen diagnostic row contract drifted: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if (
        raw.get("cell_id") != source["cell_id"]
        or raw.get("perturbation_mode") != source["perturbation_mode"]
        or raw.get("metadata", {}).get("token_ids_sha256")
        != stored_row["token_ids_sha256"]
    ):
        raise RuntimeError("frozen diagnostic raw cell metadata drifted")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(snapshot["snapshot"], local_files_only=True)
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
    token_sha = hashlib.sha256(input_ids.cpu().numpy().tobytes()).hexdigest()
    if token_sha != stored_row["token_ids_sha256"]:
        raise RuntimeError("diagnostic tokenization differs from Stage 1")

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
    ):
        raise RuntimeError("loaded Gemma fails backend-diagnostic architecture lock")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source_layer = int(config["source_stage1"]["source_layer"])
    suffix = ExplicitDecoderSuffix(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        source_layer=source_layer,
        target=TargetSpec("final_residual", position_indices=(-1,)),
    )
    acceptance = config["acceptance"]
    parity = suffix.parity(
        atol=acceptance["clean_suffix_parity_atol"],
        rtol=acceptance["clean_suffix_parity_rtol"],
    )
    clean = suffix.clean_source.float()
    mask = source_mask(attention_mask, mode="single_position", position=-1)
    directions = make_directions(
        clean,
        mask,
        design["directions"]["stage1"],
        base_seed=design["directions"]["seed"],
        cell_id=config["source_stage1"]["cell_id"],
    )
    selected_direction = [
        row for row in directions if row["id"] == config["source_stage1"]["direction_id"]
    ]
    if len(selected_direction) != 1:
        raise RuntimeError("frozen diagnostic direction is absent")
    direction = selected_direction[0]
    stored_directions = [
        row for row in raw["directions"] if row["id"] == direction["id"]
    ]
    if (
        len(stored_directions) != 1
        or stored_directions[0]["sha256"] != direction["sha256"]
        or tensor_sha256(stored_directions[0]["tensor"]) != direction["sha256"]
    ):
        raise RuntimeError("frozen diagnostic direction payload drifted")

    requests = []
    for epsilon in design["relative_epsilon_ladder"]:
        desired = patterned_direction(clean, direction["tensor"], mask, float(epsilon))
        for label, multiplier in (
            ("positive", 1.0),
            ("negative", -1.0),
            ("double", 2.0),
        ):
            perturbation = desired * multiplier
            realized, audit = delivery_audit(
                clean,
                perturbation,
                model_dtype=suffix.clean_source.dtype,
                selected_mask=mask,
                cosine_floor=thresholds["delivery"]["cosine_floor"],
                relative_norm_error_ceiling=thresholds["delivery"][
                    "relative_norm_error_ceiling"
                ],
            )
            requests.append(
                {
                    "key": [direction["id"], float(epsilon), label],
                    "source": clean + perturbation,
                    "tangent": realized,
                    "audit": audit,
                }
            )
    batch = config["batch_replay"]
    start = int(batch["chunk_start_index"])
    chunk = requests[start : start + int(batch["batch_size"])]
    offset = int(batch["selected_offset"])
    if chunk[offset]["key"] != batch["expected_request_key"]:
        raise RuntimeError("frozen request slot does not resolve to the expected key")
    original_batch_index = start // int(batch["batch_size"])
    original_diagnostic = raw["exact_batch_diagnostics"][original_batch_index]
    if (
        int(execution["runtime"]["finite_response_batch_size"])
        != int(batch["batch_size"])
        or original_diagnostic["request_keys"]
        != [row["key"] for row in chunk]
        or original_diagnostic["exact_jvp_backend"]
        != config["source_stage1"]["stored_exact_backend"]
    ):
        raise RuntimeError("frozen original batch/slot contract drifted")
    clean_batch = clean.expand(len(chunk), *clean.shape[1:]).clone()
    source_batch = torch.cat([row["source"] for row in chunk], dim=0)
    tangent_batch = torch.cat([row["tangent"] for row in chunk], dim=0)
    with torch.no_grad():
        baseline = suffix(clean_batch).detach().float().cpu()
        finite = suffix(source_batch).detach().float().cpu()
    response = finite[offset] - baseline[offset]

    backend_errors = []
    backend_results = {}
    for name in (
        config["exact_backends"]["primary"],
        config["exact_backends"]["independent_fallback"],
    ):
        try:
            result = exact_jvp(suffix, clean_batch, tangent_batch, backend=name)
            backend_results[name] = {
                "primal": result.primal.detach().float().cpu(),
                "tangent": result.tangent.detach().float().cpu(),
            }
            del result
            torch.cuda.empty_cache()
        except Exception as exc:
            backend_errors.append(
                {"backend": name, "type": type(exc).__name__, "message": str(exc)}
            )

    primary_name = config["exact_backends"]["primary"]
    fallback_name = config["exact_backends"]["independent_fallback"]
    primary = backend_results.get(primary_name)
    fallback = backend_results.get(fallback_name)
    comparisons = {}
    if primary is not None:
        comparisons["primary_primal_vs_identical_batch_clean"] = _comparison(
            primary["primal"], baseline
        )
        comparisons["primary_tangent_vs_stored"] = _comparison(
            primary["tangent"][offset], stored_raw["tangent_positive"]
        )
    if fallback is not None:
        comparisons["fallback_primal_vs_identical_batch_clean"] = _comparison(
            fallback["primal"], baseline
        )
    if primary is not None and fallback is not None:
        comparisons["primary_vs_fallback_tangent_all_slots"] = _comparison(
            primary["tangent"], fallback["tangent"]
        )
        comparisons["primary_vs_fallback_tangent_selected_slot"] = _comparison(
            primary["tangent"][offset], fallback["tangent"][offset]
        )
    comparisons["finite_response_vs_stored"] = _comparison(
        response, stored_raw["positive"]
    )
    comparisons["source_activation_vs_stored"] = _comparison(
        clean[mask], raw["clean_source_selected"]
    )
    comparisons["clean_target_vs_stored"] = _comparison(
        baseline[offset], stored_raw["finite_clean_target"]
    )
    recomputed_metrics = (
        tangent_metrics(response, primary["tangent"][offset])
        if primary is not None
        else None
    )
    metric_fields = (
        "response_norm",
        "tangent_prediction_norm",
        "tangent_cosine",
        "gain",
        "tangent_relative_error",
    )
    metric_absolute_errors = {
        field: (
            abs(float(recomputed_metrics[field]) - float(stored_row[field]))
            if recomputed_metrics is not None
            else None
        )
        for field in metric_fields
    }
    realized_sha = tensor_sha256(chunk[offset]["tangent"])
    criteria = {
        "clean_suffix_parity": parity["ok"],
        "direction_sha256_exact": direction["sha256"] == stored_row["direction_sha256"],
        "realized_perturbation_sha256_exact": realized_sha
        == stored_raw["realized_positive_sha256"],
        "primary_backend_succeeded": primary is not None,
        "fallback_backend_succeeded": fallback is not None,
        "primary_primal_parity": primary is not None
        and comparisons["primary_primal_vs_identical_batch_clean"]["relative_error"]
        <= acceptance["exact_primal_relative_error_ceiling"],
        "fallback_primal_parity": fallback is not None
        and comparisons["fallback_primal_vs_identical_batch_clean"]["relative_error"]
        <= acceptance["exact_primal_relative_error_ceiling"],
        "backend_tangent_cosine": primary is not None
        and fallback is not None
        and comparisons["primary_vs_fallback_tangent_selected_slot"]["cosine"]
        >= acceptance["backend_tangent_cosine_floor"],
        "backend_tangent_relative_error": primary is not None
        and fallback is not None
        and comparisons["primary_vs_fallback_tangent_selected_slot"]["relative_error"]
        <= acceptance["backend_tangent_relative_error_ceiling"],
        "backend_tangent_all_slots": primary is not None
        and fallback is not None
        and comparisons["primary_vs_fallback_tangent_all_slots"]["cosine"]
        >= acceptance["backend_tangent_cosine_floor"]
        and comparisons["primary_vs_fallback_tangent_all_slots"]["relative_error"]
        <= acceptance["backend_tangent_relative_error_ceiling"],
        "stored_source_activation_replay": comparisons[
            "source_activation_vs_stored"
        ]["relative_error"]
        <= acceptance["stored_source_activation_relative_error_ceiling"],
        "stored_clean_target_replay": comparisons["clean_target_vs_stored"][
            "relative_error"
        ]
        <= acceptance["stored_clean_target_relative_error_ceiling"],
        "stored_forward_tangent_replay": primary is not None
        and comparisons["primary_tangent_vs_stored"]["relative_error"]
        <= acceptance["stored_forward_tangent_relative_error_ceiling"],
        "stored_finite_response_replay": comparisons["finite_response_vs_stored"][
            "relative_error"
        ]
        <= acceptance["stored_finite_response_relative_error_ceiling"],
        "stored_metric_replay": recomputed_metrics is not None
        and max(metric_absolute_errors.values())
        <= acceptance["stored_metric_absolute_tolerance"],
    }
    passed = all(criteria.values())
    raw_payload = {
        "schema_version": 1,
        "request_key": chunk[offset]["key"],
        "direction": direction["tensor"].detach().float().cpu(),
        "realized_tangent": chunk[offset]["tangent"].detach().float().cpu(),
        "finite_response": response,
        "stored_finite_response": stored_raw["positive"],
        "stored_primary_tangent": stored_raw["tangent_positive"],
        "primary_tangent": None if primary is None else primary["tangent"][offset],
        "fallback_tangent": None if fallback is None else fallback["tangent"][offset],
        "clean_target": baseline[offset],
    }
    _atomic_torch_save(raw_output, raw_payload)
    artifact = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "tier": "methods",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": git["code_commit"],
        "config": {"path": str(CONFIG), "sha256": file_sha256(CONFIG)},
        "source_stage1": {
            "evidence_id": stage1["evidence_id"],
            "state_sha256": file_sha256(STAGE1_ROOT / "state.json"),
            "cell_metrics": {
                "path": str(metrics_path),
                "sha256": file_sha256(metrics_path),
            },
            "cell_raw": {"path": str(raw_path), "sha256": file_sha256(raw_path)},
            "stored_row": stored_row,
        },
        "model": {
            "model_id": execution["model"]["model_id"],
            "revision": execution["model"]["revision"],
            "snapshot_manifest_sha256": file_sha256(
                execution["snapshot_manifest"]["path"]
            ),
            "remote_inventory_sha256": verification["remote_inventory_sha256"],
        },
        "request": {
            "cell_id": config["source_stage1"]["cell_id"],
            "batch_size": len(chunk),
            "selected_offset": offset,
            "key": chunk[offset]["key"],
            "direction_sha256": direction["sha256"],
            "realized_perturbation_sha256": realized_sha,
            "delivery_audit": {
                "desired_norm": chunk[offset]["audit"].desired_norm,
                "realized_norm": chunk[offset]["audit"].realized_norm,
                "cosine": chunk[offset]["audit"].cosine,
                "relative_norm_error": chunk[offset]["audit"].relative_norm_error,
                "faithful": chunk[offset]["audit"].faithful,
            },
        },
        "clean_suffix_parity": parity,
        "backend_errors": backend_errors,
        "comparisons": comparisons,
        "recomputed_metrics": recomputed_metrics,
        "stored_metric_absolute_errors": metric_absolute_errors,
        "criteria": criteria,
        "backend_parity_pass": passed,
        "stage1_mismatch_reproduced": bool(
            recomputed_metrics is not None
            and (
                recomputed_metrics["tangent_cosine"]
                < thresholds["smallest_faithful_secant"]["tangent_cosine_floor"]
                or recomputed_metrics["tangent_relative_error"]
                > thresholds["smallest_faithful_secant"][
                    "tangent_relative_error_ceiling"
                ]
            )
        ),
        "environment": environment_payload(require_gpu=True),
        "gpu_gate": gpu,
        "target_model_opened": True,
        "model_response_data_replayed": True,
        "finite_difference_used_as_exact": False,
        "claim_boundary": (
            "actual-Gemma exact-backend/path diagnostic only; a pass validates "
            "the Stage-1 derivative implementation but does not identify mechanism"
        ),
    }
    json.dumps(artifact, allow_nan=False, sort_keys=True)
    atomic_json(artifact_path, artifact)
    create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "actual-Gemma same-batch replay comparing forward and fallback "
            "exact-JVP backends on a frozen Stage-1 mismatch cell"
        ),
        command="python -m jspace_gemma.experiments.gm_gemma_backend_parity",
        outputs=[artifact_path, raw_output],
        inputs={
            "config_sha256": file_sha256(CONFIG),
            "stage1_state_sha256": file_sha256(STAGE1_ROOT / "state.json"),
            "stage1_cell_metrics_sha256": file_sha256(metrics_path),
            "stage1_cell_raw_sha256": file_sha256(raw_path),
            "snapshot_manifest_sha256": file_sha256(
                execution["snapshot_manifest"]["path"]
            ),
        },
        backend_parity_pass=passed,
        stage1_mismatch_reproduced=artifact["stage1_mismatch_reproduced"],
        target_model_opened=True,
        model_response_data_replayed=True,
        finite_difference_used_as_exact=False,
    )
    print(
        json.dumps(
            {
                "output": str(artifact_path),
                "sha256": file_sha256(artifact_path),
                "raw": str(raw_output),
                "raw_sha256": file_sha256(raw_output),
                "backend_parity_pass": passed,
                "stage1_mismatch_reproduced": artifact[
                    "stage1_mismatch_reproduced"
                ],
                "criteria": criteria,
            },
            indent=1,
        )
    )
    del model
    torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError("actual-Gemma exact-JVP backend parity diagnostic failed")


if __name__ == "__main__":
    main()
