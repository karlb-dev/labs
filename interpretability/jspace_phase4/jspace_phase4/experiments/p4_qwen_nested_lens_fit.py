"""Resumable, GPU-only all-layer Qwen Jacobian-lens fitting."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Mapping

import torch
import yaml

from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    atomic_json,
    file_sha256,
    git_info,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import local_work, metrics_dir, resolve_uri, run_root
from ..provenance4 import Provenance4, write_result4
from ..registry4 import RegistryError, create, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--draw", required=True, choices=("draw_a", "draw_b"))
    parser.add_argument("--stop-at", required=True, type=int)
    return parser.parse_args()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def model_reference(uri: str) -> dict:
    if not uri.startswith("model://") or "@" not in uri:
        raise ValueError("model URI must pin an exact revision")
    model_id, revision = uri[len("model://"):].rsplit("@", 1)
    return {"model_id": model_id, "revision": revision}


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def registered_output_check(evidence_id: str) -> dict | None:
    try:
        event = resolve(evidence_id)
    except RegistryError as error:
        if "found 0" in str(error):
            return None
        raise
    if not event["live"]:
        raise RuntimeError(
            f"milestone evidence {evidence_id} exists but is not live")
    failures = []
    for output in event.get("outputs", []):
        path = Path(output["path"])
        actual = file_sha256(path) if path.exists() else None
        if actual != output["sha256"]:
            failures.append({
                "path": str(path),
                "expected": output["sha256"],
                "actual": actual,
            })
    if failures:
        raise RuntimeError(
            "registered milestone output verification failed: "
            + json.dumps(failures, sort_keys=True))
    return event


def verify_snapshot(model_path: Path, manifest: Mapping) -> dict:
    verified = []
    weight_bytes = 0
    for entry in manifest["files"]:
        path = model_path / entry["name"]
        if not path.is_file():
            raise RuntimeError(f"model snapshot file is missing: {path}")
        size = int(path.stat().st_size)
        if size != int(entry["bytes"]):
            raise RuntimeError(
                f"model snapshot size mismatch for {entry['name']}: {size}")
        print(
            f"verifying model file {entry['name']} "
            f"({size / 1e9:.3f} GB)",
            flush=True,
        )
        actual = file_sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"model snapshot hash mismatch for {entry['name']}: "
                f"{actual}")
        if entry["name"].endswith(".safetensors"):
            weight_bytes += size
        verified.append({
            "name": entry["name"],
            "bytes": size,
            "sha256": actual,
        })
    if weight_bytes != int(manifest["weight_bytes"]):
        raise RuntimeError(
            f"weight byte total is {weight_bytes}, expected "
            f"{manifest['weight_bytes']}")
    return {
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "architecture": manifest["architecture"],
        "n_layers": int(manifest["n_layers"]),
        "d_model": int(manifest["d_model"]),
        "weight_bytes": weight_bytes,
        "files": verified,
        "inventory_sha256": object_sha256(verified),
    }


def jlens_source_contract(specification: Mapping) -> tuple[dict, Path]:
    import jlens

    repository = Path(jlens.__file__).resolve().parent.parent

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments],
            text=True,
        ).strip()

    revision = git("rev-parse", "HEAD")
    if revision != specification["revision"]:
        raise RuntimeError(
            f"jlens is at {revision}, expected {specification['revision']}")
    dirty = git("status", "--porcelain")
    if dirty:
        raise RuntimeError(
            "jlens checkout is dirty; refuse scientific fitting")
    sources = {}
    for relative in specification["source_files"]:
        path = repository / relative
        if not path.is_file():
            raise RuntimeError(f"jlens source file is missing: {relative}")
        sources[relative] = file_sha256(path)
    payload = {
        "repository": specification["repository"],
        "revision": revision,
        "source_files": sources,
    }
    payload["source_inventory_sha256"] = object_sha256(payload)
    return payload, repository


def copy_atomic_verified(
        source: Path, destination: Path, *,
        expected_sha256: str | None = None,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(
        destination.suffix + f".tmp{os.getpid()}")
    shutil.copyfile(source, temporary)
    actual = file_sha256(temporary)
    if expected_sha256 is not None and actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"copied file hash mismatch for {destination}: {actual}")
    os.replace(temporary, destination)
    return actual


def choose_recovery_candidate(
        local_header: Mapping | None, drive_header: Mapping | None, *,
        fit_contract_sha256: str,
) -> str | None:
    candidates = {}
    for label, header in (
            ("local", local_header), ("drive", drive_header)):
        if header is None:
            continue
        if header.get("fit_contract_sha256") != fit_contract_sha256:
            raise RuntimeError(
                f"{label} recovery state belongs to an incompatible fit")
        candidates[label] = int(header["next_idx"])
    if not candidates:
        return None
    highest = max(candidates.values())
    return "local" if candidates.get("local") == highest else "drive"


def read_recovery_header(
        checkpoint: Path, header_path: Path,
) -> dict | None:
    checkpoint_exists = checkpoint.exists()
    header_exists = header_path.exists()
    if checkpoint_exists != header_exists:
        raise RuntimeError(
            f"incomplete recovery pair: {checkpoint}, {header_path}")
    return json.loads(header_path.read_text()) if header_exists else None


def verify_recovery_file(checkpoint: Path, header: Mapping) -> None:
    size = int(checkpoint.stat().st_size)
    if size != int(header["checkpoint_bytes"]):
        raise RuntimeError(
            f"recovery checkpoint size mismatch for {checkpoint}")
    actual = file_sha256(checkpoint)
    if actual != header["checkpoint_sha256"]:
        raise RuntimeError(
            f"recovery checkpoint hash mismatch for {checkpoint}: {actual}")


def prepare_recovery(
        local_checkpoint: Path, local_header_path: Path,
        drive_checkpoint: Path, drive_header_path: Path, *,
        fit_contract_sha256: str,
) -> tuple[int, dict | None]:
    local_header = read_recovery_header(
        local_checkpoint, local_header_path)
    drive_header = read_recovery_header(
        drive_checkpoint, drive_header_path)
    selected = choose_recovery_candidate(
        local_header,
        drive_header,
        fit_contract_sha256=fit_contract_sha256,
    )
    if selected is None:
        return 0, None
    header = local_header if selected == "local" else drive_header
    checkpoint = (
        local_checkpoint if selected == "local" else drive_checkpoint)
    assert header is not None
    verify_recovery_file(checkpoint, header)
    if selected == "drive":
        copy_atomic_verified(
            drive_checkpoint,
            local_checkpoint,
            expected_sha256=header["checkpoint_sha256"],
        )
        atomic_json(local_header_path, header)
    return int(header["next_idx"]), dict(header)


def sync_recovery(
        local_checkpoint: Path, local_header_path: Path,
        drive_checkpoint: Path, drive_header_path: Path, *,
        fit_contract_sha256: str, next_idx: int, n_done: int,
        gpu: Mapping,
) -> dict:
    checkpoint_sha = file_sha256(local_checkpoint)
    header = {
        "schema_version": 1,
        "fit_contract_sha256": fit_contract_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": int(local_checkpoint.stat().st_size),
        "next_idx": int(next_idx),
        "n_done": int(n_done),
        "synced_utc": utc_now(),
        "gpu": dict(gpu),
    }
    atomic_json(local_header_path, header)
    copy_atomic_verified(
        local_checkpoint,
        drive_checkpoint,
        expected_sha256=checkpoint_sha,
    )
    atomic_json(drive_header_path, header)
    return header


def ensure_free_space(path: Path, *, needed_bytes: int, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < needed_bytes:
        raise RuntimeError(
            f"{label} has {free / 1e9:.1f} GB free; need at least "
            f"{needed_bytes / 1e9:.1f} GB")


def fit_contract_payload(
        *, config_path: Path, config: Mapping, draw: str,
        corpus_path: Path, corpus_sha256: str,
        model_snapshot_manifest_path: Path,
        model_snapshot: Mapping, jlens_contract: Mapping,
        fitter_source_sha256: str,
) -> dict:
    return {
        "schema_version": 1,
        "config_sha256": file_sha256(config_path),
        "draw": draw,
        "corpus_uri": config["draws"][draw]["corpus_uri"],
        "corpus_sha256": corpus_sha256,
        "corpus_bytes": int(corpus_path.stat().st_size),
        "model_uri": config["model_uri"],
        "model_snapshot_manifest_sha256":
            file_sha256(model_snapshot_manifest_path),
        "model_snapshot_inventory_sha256":
            model_snapshot["inventory_sha256"],
        "recipe": dict(config["recipe"]),
        "jlens": dict(jlens_contract),
        "fitter_source_sha256": fitter_source_sha256,
    }


def main() -> None:  # noqa: C901
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    draw_specification = config["draws"][arguments.draw]
    milestones = {
        int(n): evidence_id
        for n, evidence_id in draw_specification["milestones"].items()
    }
    if arguments.stop_at not in milestones:
        raise RuntimeError(
            f"--stop-at must be one of {sorted(milestones)} for "
            f"{arguments.draw}")
    evidence_id = milestones[arguments.stop_at]
    existing = registered_output_check(evidence_id)
    if existing is not None:
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": evidence_id,
            "outputs": existing["outputs"],
        }, indent=1))
        return
    for n, prior_evidence_id in sorted(milestones.items()):
        if n >= arguments.stop_at:
            break
        if registered_output_check(prior_evidence_id) is None:
            raise RuntimeError(
                f"prior nested milestone n={n} is not registered; "
                "run milestones in order")

    clean = require_clean_tree()
    gpu = require_cuda_gpu()
    print(json.dumps({"cuda_hard_gate": gpu}, indent=1), flush=True)
    model = model_reference(config["model_uri"])
    model_path = resolve_uri(config["model_uri"])
    snapshot_manifest_path = resolve_uri(
        config["model_snapshot_manifest_uri"])
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text())
    if (
        snapshot_manifest["model_id"] != model["model_id"]
        or snapshot_manifest["revision"] != model["revision"]
    ):
        raise RuntimeError("model URI and snapshot manifest disagree")
    model_snapshot = verify_snapshot(model_path, snapshot_manifest)
    jlens_contract, _ = jlens_source_contract(config["jlens"])

    corpus_event = resolve(config["corpus_evidence_id"])
    if not corpus_event["live"]:
        raise RuntimeError("registered nested corpus evidence is not live")
    corpus_path = resolve_uri(draw_specification["corpus_uri"])
    corpus_sha = file_sha256(corpus_path)
    if corpus_sha != draw_specification["corpus_sha256"]:
        raise RuntimeError(f"corpus hash mismatch: {corpus_sha}")
    registered_corpus = {
        output["sha256"] for output in corpus_event["outputs"]
        if Path(output["path"]) == corpus_path
    }
    if registered_corpus != {corpus_sha}:
        raise RuntimeError(
            "corpus path/hash is not the registered corpus output")
    rows = load_rows(corpus_path)
    if len(rows) != int(draw_specification["total_n"]):
        raise RuntimeError(
            f"corpus has {len(rows)} rows, expected "
            f"{draw_specification['total_n']}")

    fitter_source = Path(__file__)
    contract_payload = fit_contract_payload(
        config_path=config_path,
        config=config,
        draw=arguments.draw,
        corpus_path=corpus_path,
        corpus_sha256=corpus_sha,
        model_snapshot_manifest_path=snapshot_manifest_path,
        model_snapshot=model_snapshot,
        jlens_contract=jlens_contract,
        fitter_source_sha256=file_sha256(fitter_source),
    )
    fit_contract_sha = object_sha256(contract_payload)
    recipe = config["recipe"]
    target_layer = int(recipe["target_layer"])
    source_layers = list(range(target_layer))
    sync_every = int(recipe["checkpoint_sync_every"])

    drive_fit_root = (
        run_root() / "lens" / config["slug"] / "nested_fit"
        / arguments.draw
    )
    drive_recovery = drive_fit_root / "recovery"
    local_fit_root = (
        local_work() / "qwen_nested_lens_fit" / config["slug"]
        / arguments.draw / fit_contract_sha
    )
    local_checkpoint = local_fit_root / "fit.ckpt"
    local_header_path = local_fit_root / "checkpoint_state.json"
    drive_checkpoint = drive_recovery / "fit.ckpt"
    drive_header_path = drive_recovery / "checkpoint_state.json"
    progress_path = (
        drive_recovery / f"progress_{fit_contract_sha}.json")
    expected_checkpoint = int(
        recipe["expected_checkpoint_tensor_bytes"])
    expected_lens = int(recipe["expected_lens_tensor_bytes"])
    ensure_free_space(
        local_fit_root,
        needed_bytes=expected_checkpoint + expected_lens + 3_000_000_000,
        label="local NVMe",
    )
    ensure_free_space(
        drive_fit_root,
        needed_bytes=expected_checkpoint + expected_lens + 2_000_000_000,
        label="Drive",
    )
    next_idx, recovered_header = prepare_recovery(
        local_checkpoint,
        local_header_path,
        drive_checkpoint,
        drive_header_path,
        fit_contract_sha256=fit_contract_sha,
    )
    if next_idx > arguments.stop_at:
        raise RuntimeError(
            f"recovery state is already at n={next_idx}, beyond requested "
            f"unregistered n={arguments.stop_at}")
    print(json.dumps({
        "fit_contract_sha256": fit_contract_sha,
        "recovered_next_idx": next_idx,
        "recovered": recovered_header is not None,
        "target_milestone": arguments.stop_at,
        "checkpoint_sync_every": sync_every,
        "drive_recovery": str(drive_recovery),
    }, indent=1), flush=True)

    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists() else {
            "schema_version": 1,
            "fit_contract_sha256": fit_contract_sha,
            "draw": arguments.draw,
            "checkpoints": [],
            "started_utc": utc_now(),
        }
    )
    if progress["fit_contract_sha256"] != fit_contract_sha:
        raise RuntimeError("progress file belongs to an incompatible fit")
    progress["active_target"] = arguments.stop_at
    progress["last_invocation_code_commit"] = clean["code_commit"]
    progress["gpu"] = gpu
    atomic_json(progress_path, progress)

    import jlens
    import transformers

    jlens.configure_logging()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    print("loading exact Qwen snapshot to CUDA; CPU fallback is forbidden",
          flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    assert_model_on_cuda(hf_model)
    lens_model = jlens.from_hf(hf_model, tokenizer)
    if (
        lens_model.n_layers != int(recipe["expected_n_layers"])
        or lens_model.d_model != int(recipe["expected_d_model"])
    ):
        raise RuntimeError(
            f"expected {recipe['expected_n_layers']} x "
            f"{recipe['expected_d_model']}, got "
            f"{lens_model.n_layers} x {lens_model.d_model}")
    prompts = [row["text"] for row in rows]

    invocation_started = time.time()
    invocation_start_idx = next_idx
    lens = None
    while next_idx < arguments.stop_at:
        end = min(
            next_idx + sync_every,
            arguments.stop_at,
        )
        print(
            f"GPU fit chunk {next_idx}:{end} / {arguments.stop_at}; "
            "checkpoint will be atomically mirrored to Drive",
            flush=True,
        )
        lens = jlens.fit(
            lens_model,
            prompts[:end],
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=int(recipe["dim_batch"]),
            max_seq_len=int(recipe["max_seq_len"]),
            skip_first=int(recipe["skip_first"]),
            checkpoint_path=str(local_checkpoint),
            checkpoint_every=None,
            resume=True,
        )
        if lens.n_prompts != end:
            raise RuntimeError(
                f"fit reached {lens.n_prompts} valid prompts, expected {end}; "
                "do not silently change the nested sample")
        header = sync_recovery(
            local_checkpoint,
            local_header_path,
            drive_checkpoint,
            drive_header_path,
            fit_contract_sha256=fit_contract_sha,
            next_idx=end,
            n_done=lens.n_prompts,
            gpu=gpu,
        )
        next_idx = end
        elapsed = time.time() - invocation_started
        newly_done = next_idx - invocation_start_idx
        checkpoint_record = {
            **header,
            "elapsed_this_invocation_s": round(elapsed, 1),
            "seconds_per_new_prompt": (
                round(elapsed / newly_done, 2) if newly_done else None),
            "peak_vram_bytes":
                int(torch.cuda.max_memory_allocated()),
        }
        progress["checkpoints"].append(checkpoint_record)
        progress["last_checkpoint"] = checkpoint_record
        atomic_json(progress_path, progress)
        print(json.dumps({
            "checkpoint_synced": next_idx,
            "checkpoint_sha256": header["checkpoint_sha256"],
            "checkpoint_bytes": header["checkpoint_bytes"],
            "seconds_per_new_prompt":
                checkpoint_record["seconds_per_new_prompt"],
            "peak_vram_gb":
                checkpoint_record["peak_vram_bytes"] / 1e9,
        }, indent=1), flush=True)
        if next_idx < arguments.stop_at:
            del lens
            lens = None
            gc.collect()

    if lens is None:
        lens = jlens.fit(
            lens_model,
            prompts[:arguments.stop_at],
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=int(recipe["dim_batch"]),
            max_seq_len=int(recipe["max_seq_len"]),
            skip_first=int(recipe["skip_first"]),
            checkpoint_path=str(local_checkpoint),
            checkpoint_every=None,
            resume=True,
        )
    if lens.n_prompts != arguments.stop_at:
        raise RuntimeError("milestone lens prompt count mismatch")

    label = draw_specification["label"]
    lens_name = (
        f"{config['slug']}_jlens_{label}_n{arguments.stop_at:04d}.pt")
    local_lens = local_fit_root / lens_name
    local_lens_temporary = local_lens.with_suffix(
        local_lens.suffix + f".tmp{os.getpid()}")
    lens.save(str(local_lens_temporary), dtype=torch.float16)
    os.replace(local_lens_temporary, local_lens)
    lens_sha = file_sha256(local_lens)
    drive_lens = drive_fit_root / lens_name
    copy_atomic_verified(
        local_lens, drive_lens, expected_sha256=lens_sha)

    output_dir = (
        metrics_dir(config["slug"]) / "lens_fit" / evidence_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "input_manifest.json"
    result_path = output_dir / "lens_fit_result.json"
    final_header = json.loads(drive_header_path.read_text())
    input_payload = {
        "schema_version": 1,
        "evidence_id": evidence_id,
        "code_commit": clean["code_commit"],
        "fit_contract": contract_payload,
        "fit_contract_sha256": fit_contract_sha,
        "corpus_evidence_id": config["corpus_evidence_id"],
        "checkpoint_at_milestone": final_header,
        "model_snapshot": model_snapshot,
        "gpu": gpu,
    }
    manifest_envelope = {
        "schema_version": 1,
        "payload": input_payload,
        "payload_sha256": object_sha256(input_payload),
    }
    atomic_json(manifest_path, manifest_envelope)
    elapsed = time.time() - invocation_started
    payload = {
        "schema_version": 1,
        "draw": arguments.draw,
        "n_prompts": lens.n_prompts,
        "source_layers": source_layers,
        "target_layer": target_layer,
        "d_model": lens.d_model,
        "corpus_sha256": corpus_sha,
        "fit_contract_sha256": fit_contract_sha,
        "lens_sha256": lens_sha,
        "lens_bytes": int(drive_lens.stat().st_size),
        "checkpoint_sha256": final_header["checkpoint_sha256"],
        "checkpoint_bytes": final_header["checkpoint_bytes"],
        "new_prompts_this_invocation":
            arguments.stop_at - invocation_start_idx,
        "elapsed_this_invocation_s": round(elapsed, 1),
        "seconds_per_new_prompt": (
            round(
                elapsed / (arguments.stop_at - invocation_start_idx), 2)
            if arguments.stop_at > invocation_start_idx else None),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": gpu,
        "jlens": jlens_contract,
    }
    command = (
        "python -m "
        "jspace_phase4.experiments.p4_qwen_nested_lens_fit "
        f"--config {arguments.config} --draw {arguments.draw} "
        f"--stop-at {arguments.stop_at}"
    )
    inputs = {
        "corpus": corpus_sha,
        "corpus_evidence": config["corpus_evidence_id"],
        "model_snapshot_manifest":
            file_sha256(snapshot_manifest_path),
        "model_snapshot_inventory":
            model_snapshot["inventory_sha256"],
        "fit_contract": fit_contract_sha,
        "milestone_checkpoint": final_header["checkpoint_sha256"],
    }
    write_result4(
        payload,
        result_path,
        Provenance4(
            evidence_id=evidence_id,
            tier=config["tier"],
            command=command,
            inputs=inputs,
            input_manifest_sha256=
                manifest_envelope["payload_sha256"],
            model=model,
            seed_contract=(
                "nested-prefix-corpus; deterministic mean-J estimator"),
        ),
    )
    create(
        evidence_id,
        tier=config["tier"],
        what=(
            f"Qwen3.6-27B all-layer Jacobian lens on leakage-safe "
            f"{label} nested prefix n={arguments.stop_at}; exact pinned "
            "model/corpus/jlens sources and GPU-only fit."),
        command=command,
        outputs=[drive_lens, result_path, manifest_path],
        inputs=inputs,
    )
    progress["last_registered_milestone"] = {
        "evidence_id": evidence_id,
        "n": arguments.stop_at,
        "lens": str(drive_lens),
        "lens_sha256": lens_sha,
        "registered_utc": utc_now(),
    }
    atomic_json(progress_path, progress)
    local_lens.unlink(missing_ok=True)
    print(json.dumps({
        "evidence_id": evidence_id,
        "lens": str(drive_lens),
        "lens_sha256": lens_sha,
        "result": str(result_path),
        "input_manifest": str(manifest_path),
        "progress": str(progress_path),
        "gpu": gpu["name"],
    }, indent=1))


if __name__ == "__main__":
    main()
