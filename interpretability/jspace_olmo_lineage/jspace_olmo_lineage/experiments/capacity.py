"""Freeze, run, and aggregate symmetric OLMo four-checkpoint capacity."""
from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from ..capacity import (
    bootstrap_estimates,
    canonical_jsonl,
    classify_shift,
    content_token_manifest,
    curve_summary,
    frame_summary,
    lower_median,
    occupancy_from_errors,
    percentile_interval,
    pursuit_batched,
    select_frozen_corpus,
    stratified_prompt_counts,
)
from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    atomic_json,
    atomic_text,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths import metrics_dir, resolve_uri, run_root
from ..provenance import Provenance, write_result
from ..registry import RegistryError, create, read_events, resolve

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]


def _load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("capacity config must be a mapping")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def _model_reference(uri: str) -> dict:
    if not uri.startswith("model://") or "@" not in uri:
        raise ValueError("model URI must pin an exact revision")
    model_id, revision = uri[len("model://"):].rsplit("@", 1)
    if len(revision) != 40:
        raise ValueError("model revision must be a full commit")
    return {"model_id": model_id, "revision": revision}


def _registered_output_check(evidence_id: str) -> dict | None:
    origins = [
        row for row in read_events()
        if row.get("evidence_id") == evidence_id
        and row.get("event") in {"evidence_created", "evidence_imported"}
    ]
    if not origins:
        return None
    event = resolve(evidence_id)
    if not event["live"]:
        raise RegistryError(f"existing evidence is not live: {evidence_id}")
    for output in event.get("outputs", []):
        path = Path(output["path"])
        if not path.is_file() or file_sha256(path) != output["sha256"]:
            raise RegistryError(f"registered output failed verification: {path}")
    return event


def _verify_method_sources(config: Mapping) -> dict:
    verified = {}
    for name, specification in config["method_sources"].items():
        path = resolve_uri(specification["uri"])
        digest = file_sha256(path)
        if digest != specification["sha256"]:
            raise RuntimeError(f"capacity method source drift: {name}")
        verified[name] = {
            "uri": specification["uri"], "path": str(path),
            "sha256": digest, "bytes": int(path.stat().st_size),
        }
    return verified


def _require_provenance_audit(config: Mapping) -> dict:
    specification = config["provenance_audit"]
    event = resolve(specification["evidence_id"])
    if not event["live"]:
        raise RuntimeError("O3 provenance audit is not live")
    path = resolve_uri(specification["result_uri"])
    digest = file_sha256(path)
    if digest != specification["result_sha256"]:
        raise RuntimeError("O3 provenance audit result hash drift")
    if not any(row["sha256"] == digest for row in event["outputs"]):
        raise RuntimeError("O3 audit registry event does not pin result")
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    if envelope["payload_sha256"] != object_sha256(payload):
        raise RuntimeError("O3 provenance audit envelope hash drift")
    if payload["summary"]["refit_decision"] != specification[
            "required_refit_decision"]:
        raise RuntimeError("O3 audit does not authorize existing-lens use")
    if payload["summary"]["all_pairs_exact_same_recipe_corpus"] is not True:
        raise RuntimeError("O3 audit did not establish lens comparability")
    return {
        "evidence_id": specification["evidence_id"],
        "result_sha256": digest,
        "refit_decision": payload["summary"]["refit_decision"],
        "all_pairs_exact_same_recipe_corpus": True,
    }


def _selected_corpus(config: Mapping) -> tuple[list[dict], Path]:
    specification = config["source_corpus"]
    source = resolve_uri(specification["uri"])
    if file_sha256(source) != specification["sha256"]:
        raise RuntimeError("capacity source corpus hash drift")
    rows = _read_jsonl(source)
    if len(rows) != int(specification["rows"]):
        raise RuntimeError("capacity source corpus row-count drift")
    domains = list(specification["source_domains"])
    counts = {domain: sum(row.get("domain") == domain for row in rows)
              for domain in domains}
    if any(value != int(specification["source_rows_per_domain"])
           for value in counts.values()):
        raise RuntimeError(f"capacity source domain-count drift: {counts}")
    selected = select_frozen_corpus(
        rows, domains=domains,
        rows_per_domain=int(specification["selected_rows_per_domain"]))
    if len(selected) != int(specification["selected_rows"]):
        raise RuntimeError("capacity selected-corpus row-count drift")
    canonical = canonical_jsonl(selected)
    encoded = canonical.encode("utf-8")
    import hashlib
    if hashlib.sha256(encoded).hexdigest() != specification[
            "selected_canonical_jsonl_sha256"]:
        raise RuntimeError("selected capacity corpus hash drift")
    if len(encoded) != int(specification["selected_canonical_jsonl_bytes"]):
        raise RuntimeError("selected capacity corpus byte-count drift")
    if hashlib.sha256("".join(
            row["text"] for row in selected).encode("utf-8")).hexdigest() != (
            specification["selected_concatenated_text_sha256"]):
        raise RuntimeError("selected capacity text hash drift")
    expected_pids = []
    for domain in domains:
        lo, hi = specification["selected_pid_ranges"][domain]
        expected_pids.extend(range(int(lo), int(hi) + 1))
    if [int(row["pid"]) for row in selected] != expected_pids:
        raise RuntimeError("selected capacity PID order drift")
    return selected, source


def _tokenizer_protocol_audit(
    config: Mapping, selected: Sequence[Mapping],
) -> dict:
    import transformers

    maximum = int(config["activation_population"][
        "maximum_content_tokens"])
    by_model = {}
    sequences_by_model = {}
    for model in config["models"]:
        snapshot = Path(model["tokenizer_snapshot"])
        tokenizer_json = snapshot / "tokenizer.json"
        if file_sha256(tokenizer_json) != model["tokenizer_json_sha256"]:
            raise RuntimeError(f"tokenizer.json hash drift: {model['slug']}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True)
        observed_bos = tokenizer.bos_token_id
        if observed_bos != model["expected_bos_token_id"]:
            raise RuntimeError(f"BOS token drift: {model['slug']}")
        sequences = [
            list(map(int, tokenizer(
                row["text"], add_special_tokens=False, truncation=True,
                max_length=maximum)["input_ids"]))
            for row in selected
        ]
        manifest = content_token_manifest(sequences)
        sequences_by_model[model["slug"]] = sequences
        by_model[model["slug"]] = {
            "tokenizer_snapshot": str(snapshot),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_json_sha256": model["tokenizer_json_sha256"],
            "bos_token_id": observed_bos,
            "content_sequence_manifest_sha256": manifest,
            "minimum_content_tokens": min(map(len, sequences)),
            "maximum_content_tokens": max(map(len, sequences)),
        }
    expected = config["activation_population"][
        "expected_content_sequence_manifest_sha256"]
    if any(row["content_sequence_manifest_sha256"] != expected
           for row in by_model.values()):
        raise RuntimeError("capacity tokenizer sequence manifest drift")
    first_slug = config["models"][0]["slug"]
    if any(sequences != sequences_by_model[first_slug]
           for sequences in sequences_by_model.values()):
        raise RuntimeError("capacity content token sequences differ by model")
    sequence = sequences_by_model[first_slug]
    position_count = sum(
        (len(ids) - 1) - min(
            int(config["activation_population"][
                "skip_first_content_positions"]),
            max(len(ids) - int(config["activation_population"][
                "short_prompt_tail_positions"]), 1))
        for ids in sequence
    )
    if position_count != int(config["activation_population"][
            "expected_positions_under_shared_tokenization"]):
        raise RuntimeError("capacity retained-position count drift")
    return {
        "models": by_model,
        "all_content_sequences_identical": True,
        "content_sequence_manifest_sha256": expected,
        "retained_positions_per_model": position_count,
    }


def freeze_protocol(config_path: Path, config: Mapping) -> dict:
    existing = _registered_output_check(config["protocol_evidence_id"])
    if existing is not None:
        return {"status": "already_registered", "event": existing}
    clean = require_clean_tree(expected_branch=config["branch"])
    selected, source = _selected_corpus(config)
    methods = _verify_method_sources(config)
    audit = _require_provenance_audit(config)
    tokenization = _tokenizer_protocol_audit(config, selected)
    corpus_path = resolve_uri(config["outputs"]["corpus"], must_exist=False)
    protocol_path = resolve_uri(config["outputs"]["protocol"], must_exist=False)
    collisions = [str(path) for path in (corpus_path, protocol_path)
                  if path.exists()]
    if collisions:
        raise FileExistsError(
            "unregistered capacity protocol outputs exist: "
            + ", ".join(collisions))
    atomic_text(corpus_path, canonical_jsonl(selected))
    if file_sha256(corpus_path) != config["source_corpus"][
            "selected_canonical_jsonl_sha256"]:
        raise RuntimeError("written capacity corpus failed its frozen hash")
    payload = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": config["protocol_evidence_id"],
        "status": "frozen-before-any-four-model-capacity-outcome",
        "git": clean,
        "source_corpus": {
            "uri": config["source_corpus"]["uri"],
            "path": str(source),
            "sha256": config["source_corpus"]["sha256"],
            "selected_uri": config["outputs"]["corpus"],
            "selected_path": str(corpus_path),
            "selected_sha256": file_sha256(corpus_path),
            "selected_rows": len(selected),
            "selected_pids": [int(row["pid"]) for row in selected],
            "domain_order": config["source_corpus"]["source_domains"],
            "rows_per_domain": config["source_corpus"][
                "selected_rows_per_domain"],
            "phase4_untouched_family_content_present": False,
        },
        "tokenization": tokenization,
        "activation_population": config["activation_population"],
        "estimator": config["estimator"],
        "uncertainty": config["uncertainty"],
        "decision_rules": config["decision_rules"],
        "frames": config["frames"],
        "models": [{
            key: row[key] for key in (
                "slug", "role", "evidence_id", "model_uri",
                "own_lens_uri", "own_lens_sha256")
        } for row in config["models"]],
        "common_lens": config["common_lens"],
        "provenance_audit": audit,
        "method_sources": methods,
        "claim_boundary": config["claim_boundary"],
        "model_capacity_outcomes_opened": False,
        "intervention_outcomes_opened": False,
        "phase4_outputs_opened": False,
    }
    payload["all_protocol_gates_pass"] = True
    atomic_json(protocol_path, payload)
    command = (
        "python -m jspace_olmo_lineage.experiments.capacity "
        f"--config {config_path} --freeze-protocol")
    event = create(
        config["protocol_evidence_id"], tier="methods",
        what=(
            "Prospective symmetric OLMo four-checkpoint capacity protocol; "
            "shared 120-prompt corpus, centered-target pursuit, raw "
            "sensitivity, paired bootstrap, and decision margins frozen "
            "before any O2 model outcome."),
        command=command, outputs=[corpus_path, protocol_path],
        inputs={
            "config": file_sha256(config_path),
            "source_corpus": config["source_corpus"]["sha256"],
            "provenance_audit": config["provenance_audit"][
                "result_sha256"],
            **{f"method:{key}": value["sha256"]
               for key, value in config["method_sources"].items()},
        },
        capacity_outcomes_opened=False,
        interventions_opened=False,
        phase4_outputs_opened=False,
    )
    return {"protocol": payload, "event": event}


def _require_protocol(config: Mapping) -> tuple[dict, str, Path]:
    event = resolve(config["protocol_evidence_id"])
    if not event["live"]:
        raise RuntimeError("capacity protocol is not live")
    path = resolve_uri(config["outputs"]["protocol"])
    digest = file_sha256(path)
    if not any(row["sha256"] == digest for row in event["outputs"]):
        raise RuntimeError("capacity protocol registry hash drift")
    payload = json.loads(path.read_text())
    if payload.get("all_protocol_gates_pass") is not True:
        raise RuntimeError("capacity protocol gates did not pass")
    corpus = resolve_uri(config["outputs"]["corpus"])
    if file_sha256(corpus) != config["source_corpus"][
            "selected_canonical_jsonl_sha256"]:
        raise RuntimeError("frozen capacity corpus hash drift")
    _require_provenance_audit(config)
    _verify_method_sources(config)
    return payload, digest, corpus


def _verify_runtime(config: Mapping) -> dict:
    observed = {}
    for package, expected in config["runtime"]["packages"].items():
        actual = importlib.metadata.version(package)
        observed[package] = actual
        if actual != str(expected):
            raise RuntimeError(
                f"runtime package drift: {package} {actual} != {expected}")
    checkout = Path(config["runtime"]["jlens_checkout"])
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain"],
        text=True).strip())
    if revision != config["runtime"]["jlens_revision"] or dirty:
        raise RuntimeError("jlens checkout revision/cleanliness drift")
    return {
        "packages": observed,
        "jlens_checkout": str(checkout),
        "jlens_revision": revision,
        "jlens_dirty": dirty,
    }


def _model_specification(config: Mapping, slug: str) -> dict:
    matches = [dict(row) for row in config["models"] if row["slug"] == slug]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate capacity model: {slug}")
    return matches[0]


def _tokenizer_source_hash(snapshot: Path) -> str:
    names = (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "chat_template.jinja",
    )
    files = {name: file_sha256(snapshot / name)
             for name in names if (snapshot / name).is_file()}
    if not files:
        raise RuntimeError("model snapshot has no tokenizer sources")
    return object_sha256(files)


def _snapshot_manifest(model_path: Path, reference: Mapping) -> dict:
    if model_path.name != reference["revision"]:
        raise RuntimeError("staged model snapshot revision path drift")
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise RuntimeError("staged model lacks config or safetensors index")
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    shard_rows = []
    for name in shards:
        path = model_path / name
        if not path.is_file():
            raise RuntimeError(f"staged model shard is absent: {name}")
        shard_rows.append({
            "name": name, "bytes": int(path.stat().st_size),
            "blob_target": path.resolve().name,
        })
    payload = {
        "model_id": reference["model_id"],
        "revision": reference["revision"],
        "snapshot_path": str(model_path),
        "config_sha256": file_sha256(config_path),
        "index_sha256": file_sha256(index_path),
        "index_metadata": index.get("metadata", {}),
        "n_weight_map_entries": len(index["weight_map"]),
        "shards": shard_rows,
        "tokenizer_source_sha256": _tokenizer_source_hash(model_path),
    }
    payload["semantic_manifest_sha256"] = object_sha256(payload)
    return payload


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_once(path: Path, payload: Mapping) -> None:
    if path.exists():
        if json.loads(path.read_text()) != dict(payload):
            raise RuntimeError(f"existing manifest differs: {path}")
        return
    atomic_json(path, payload)


def _capture_activations(
    hf_model,
    tokenizer,
    *,
    selected: Sequence[Mapping],
    layers: Sequence[int],
    population: Mapping,
) -> tuple[dict[int, torch.Tensor], pd.DataFrame, dict, object]:
    """Capture the same content-relative activation population for one model."""
    import jlens
    from jlens.hooks import ActivationRecorder

    wrapped = jlens.from_hf(hf_model, tokenizer, force_bos=False)
    maximum = int(population["maximum_content_tokens"])
    skip_first = int(population["skip_first_content_positions"])
    tail = int(population["short_prompt_tail_positions"])
    bos_token_id = tokenizer.bos_token_id
    bos_count = int(bos_token_id is not None)
    activations = {int(layer): [] for layer in layers}
    position_rows = []
    content_sequences = []
    prompt_input_hashes = []
    started = time.time()
    for owner, row in enumerate(selected):
        content_ids = list(map(int, tokenizer(
            row["text"], add_special_tokens=False, truncation=True,
            max_length=maximum)["input_ids"]))
        if len(content_ids) < 2:
            raise RuntimeError(f"capacity prompt is too short: PID {row['pid']}")
        content_sequences.append(content_ids)
        input_ids_list = (
            [int(bos_token_id)] + content_ids
            if bos_token_id is not None else content_ids)
        input_ids = torch.tensor(
            [input_ids_list], dtype=torch.long, device=wrapped.input_device)
        content_length = len(content_ids)
        lo = min(skip_first, max(content_length - tail, 1))
        hi = content_length - 1
        if hi <= lo:
            raise RuntimeError(f"capacity prompt has no retained positions: {row}")
        with torch.no_grad(), ActivationRecorder(
                wrapped.layers, at=layers) as recorder:
            wrapped.forward(input_ids)
        model_lo = bos_count + lo
        model_hi = bos_count + hi
        n_positions = hi - lo
        for layer in layers:
            captured = recorder.activations[int(layer)][
                0, model_lo:model_hi].detach().float().cpu()
            if captured.shape != (n_positions, wrapped.d_model):
                raise RuntimeError("captured activation shape drift")
            activations[int(layer)].append(captured)
        input_hash = object_sha256(input_ids_list)
        prompt_input_hashes.append(input_hash)
        import hashlib
        text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        for content_position in range(lo, hi):
            position_rows.append({
                "row_index": len(position_rows),
                "owner": int(owner),
                "pid": int(row["pid"]),
                "domain": row["domain"],
                "text_sha256": text_hash,
                "prompt_input_ids_sha256": input_hash,
                "content_token_count": content_length,
                "bos_token_id": bos_token_id,
                "model_token_position": bos_count + content_position,
                "content_token_position": content_position,
                "token_id": int(content_ids[content_position]),
                "next_token_id": int(content_ids[content_position + 1]),
            })
        if (owner + 1) % 20 == 0:
            print(
                f"capture {owner + 1}/{len(selected)} prompts; "
                f"positions={len(position_rows)}; "
                f"elapsed={time.time() - started:.0f}s", flush=True)
    combined = {layer: torch.cat(chunks, dim=0)
                for layer, chunks in activations.items()}
    frame = pd.DataFrame(position_rows)
    expected = int(population["expected_positions_under_shared_tokenization"])
    if len(frame) != expected:
        raise RuntimeError(
            f"retained activation positions {len(frame)} != {expected}")
    if any(len(value) != len(frame) for value in combined.values()):
        raise RuntimeError("activation layers and position manifest misalign")
    manifest = content_token_manifest(content_sequences)
    if manifest != population[
            "expected_content_sequence_manifest_sha256"]:
        raise RuntimeError("runtime content-token manifest drift")
    tokenization = {
        "content_sequence_manifest_sha256": manifest,
        "prompt_input_ids_manifest_sha256": object_sha256(
            prompt_input_hashes),
        "bos_token_id": bos_token_id,
        "bos_explicitly_prepended": bool(bos_count),
        "n_prompts": len(selected),
        "n_positions": len(frame),
        "minimum_content_tokens": int(frame.groupby("owner")[
            "content_token_count"].first().min()),
        "maximum_content_tokens": int(frame.groupby("owner")[
            "content_token_count"].first().max()),
    }
    return combined, frame, tokenization, wrapped


@torch.no_grad()
def _effective_gain(wrapped) -> torch.Tensor:
    weight = wrapped._final_norm.weight.detach()  # reference adapter member
    ones = torch.ones(
        1, weight.shape[0], device=weight.device, dtype=torch.float32)
    return wrapped._final_norm(ones).detach().float().reshape(-1)


@torch.no_grad()
def _build_j_dictionary(
    hf_model,
    wrapped,
    jacobian: torch.Tensor,
    *,
    chunk_rows: int,
) -> tuple[torch.Tensor, dict]:
    """Build full-vocabulary ``normalize((W*g)@J)`` one layer at a time."""
    weight = hf_model.get_output_embeddings().weight.detach()
    device = weight.device
    vocabulary, dimension = weight.shape
    if tuple(jacobian.shape) != (dimension, dimension):
        raise RuntimeError("lens Jacobian dimension differs from model")
    gain = _effective_gain(wrapped).to(device)
    transported = jacobian.to(device=device, dtype=torch.float32)
    dictionary = torch.empty(
        vocabulary, dimension, device=device, dtype=torch.bfloat16)
    zero_rows = 0
    for start in range(0, vocabulary, int(chunk_rows)):
        stop = min(start + int(chunk_rows), vocabulary)
        block = (weight[start:stop].float() * gain[None, :]) @ transported
        norms = block.norm(dim=1, keepdim=True)
        zero_rows += int((norms.squeeze(1) == 0).sum().item())
        block = block / norms.clamp_min(1e-12)
        dictionary[start:stop] = block.to(torch.bfloat16)
        del block, norms
    sample_indices = torch.linspace(
        0, vocabulary - 1, steps=min(257, vocabulary),
        device=device, dtype=torch.float64).round().long()
    sample = dictionary[sample_indices].float()
    diagnostics = {
        "rows": int(vocabulary),
        "dimensions": int(dimension),
        "stored_dtype": str(dictionary.dtype).replace("torch.", ""),
        "zero_rows_before_normalization": int(zero_rows),
        "sampled_rows": int(len(sample_indices)),
        "sampled_all_finite": bool(torch.isfinite(sample).all()),
        "sampled_norm_min": float(sample.norm(dim=1).min()),
        "sampled_norm_max": float(sample.norm(dim=1).max()),
    }
    del transported, gain, sample
    if zero_rows or not diagnostics["sampled_all_finite"]:
        raise RuntimeError(f"invalid J dictionary: {diagnostics}")
    return dictionary, diagnostics


@torch.no_grad()
def _build_random_dictionary(
    rows: int,
    dimensions: int,
    *,
    seed: int,
    chunk_rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    dictionary = torch.empty(
        int(rows), int(dimensions), device=device, dtype=torch.bfloat16)
    for start in range(0, int(rows), int(chunk_rows)):
        stop = min(start + int(chunk_rows), int(rows))
        block = torch.randn(
            stop - start, int(dimensions), generator=generator,
            device=device, dtype=torch.float32)
        block = torch.nn.functional.normalize(block, dim=1)
        dictionary[start:stop] = block.to(torch.bfloat16)
        del block
    sample = dictionary[::max(1, int(rows) // 257)].float()
    diagnostics = {
        "seed": int(seed), "rows": int(rows),
        "dimensions": int(dimensions),
        "stored_dtype": "bfloat16",
        "sampled_rows": int(len(sample)),
        "sampled_all_finite": bool(torch.isfinite(sample).all()),
        "sampled_norm_min": float(sample.norm(dim=1).min()),
        "sampled_norm_max": float(sample.norm(dim=1).max()),
    }
    if not diagnostics["sampled_all_finite"]:
        raise RuntimeError("non-finite random dictionary")
    return dictionary, diagnostics


def _run_targets(
    centered: torch.Tensor,
    raw: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    estimator: Mapping,
    label: str,
) -> tuple[object, object]:
    arguments = {
        "k_max": int(estimator["k_max"]),
        "batch_positions": int(estimator["pursuit_batch_positions"]),
        "refit_iterations": int(estimator["refit_iterations"]),
        "learning_rate_cap": float(estimator["refit_learning_rate_cap"]),
    }
    started = time.time()
    print(f"{label}: centered pursuit start", flush=True)
    centered_result = pursuit_batched(centered, dictionary, **arguments)
    print(
        f"{label}: centered pursuit done in {time.time() - started:.0f}s; "
        "raw sensitivity start", flush=True)
    raw_started = time.time()
    raw_result = pursuit_batched(raw, dictionary, **arguments)
    print(
        f"{label}: raw sensitivity done in {time.time() - raw_started:.0f}s",
        flush=True)
    return centered_result, raw_result


def _summary_with_bootstrap(
    centered_j: np.ndarray,
    centered_random: np.ndarray,
    raw_j: np.ndarray,
    raw_random: np.ndarray,
    *,
    owners: np.ndarray,
    prompt_domains: Sequence[str],
    estimator: Mapping,
    uncertainty: Mapping,
    prompt_counts: np.ndarray,
) -> tuple[dict, dict]:
    persistence = int(estimator["crossing"]["persistence"])
    sensitivity = list(estimator["crossing"][
        "persistence_sensitivity"])
    summary = frame_summary(
        centered_j, centered_random, raw_j, raw_random,
        owners=owners, prompt_domains=prompt_domains,
        persistence=persistence, persistence_sensitivity=sensitivity)
    centered_bootstrap = bootstrap_estimates(
        centered_j, centered_random, owners=owners,
        prompt_counts=prompt_counts, persistence=persistence)
    raw_bootstrap = bootstrap_estimates(
        raw_j, raw_random, owners=owners,
        prompt_counts=prompt_counts, persistence=persistence)
    level = float(uncertainty["interval_level"])
    summary["primary_centered"]["prompt_bootstrap"] = {
        "excess_share": percentile_interval(
            centered_bootstrap["excess_share"], level),
        "occupancy_median": percentile_interval(
            centered_bootstrap["occupancy_median"], level),
        "draws": int(uncertainty["draws"]),
        "centering": uncertainty["centering_during_bootstrap"],
    }
    summary["raw_sensitivity"]["prompt_bootstrap"] = {
        "excess_share": percentile_interval(
            raw_bootstrap["excess_share"], level),
        "occupancy_median": percentile_interval(
            raw_bootstrap["occupancy_median"], level),
        "draws": int(uncertainty["draws"]),
    }
    return summary, {
        "centered_excess": centered_bootstrap["excess_share"],
        "centered_occupancy": centered_bootstrap["occupancy_median"],
        "raw_excess": raw_bootstrap["excess_share"],
        "raw_occupancy": raw_bootstrap["occupancy_median"],
    }


def _lens_container(path: Path, expected_sha256: str,
                    layers: Sequence[int]) -> dict:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"capacity lens hash drift: {path}")
    value = torch.load(
        path, map_location="cpu", mmap=True, weights_only=True)
    if set(value) != {"J", "n_prompts", "source_layers", "d_model"}:
        raise RuntimeError("capacity lens container keys drift")
    if any(int(layer) not in value["J"] for layer in layers):
        raise RuntimeError("capacity lens lacks a requested layer")
    return value


def _checkpoint_payload(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as value:
        metadata = json.loads(str(value["metadata_json"].item()))
        summary = json.loads(str(value["summary_json"].item()))
        arrays = {name: value[name].copy() for name in value.files
                  if name not in {"metadata_json", "summary_json"}}
    return {"metadata": metadata, "summary": summary}, arrays


def _add_solver_support(
    summary: dict,
    *,
    centered_achieved: np.ndarray,
    raw_achieved: np.ndarray,
    k_max: int,
) -> None:
    summary["primary_centered"]["solver"].update({
        "achieved_support_mean": float(np.mean(centered_achieved)),
        "achieved_support_median": float(np.median(centered_achieved)),
        "rows_exhausted_before_kmax": int(np.sum(
            np.asarray(centered_achieved) < int(k_max))),
    })
    summary["raw_sensitivity"]["solver"].update({
        "achieved_support_mean": float(np.mean(raw_achieved)),
        "achieved_support_median": float(np.median(raw_achieved)),
        "rows_exhausted_before_kmax": int(np.sum(
            np.asarray(raw_achieved) < int(k_max))),
    })


def _process_layer(
    *,
    path: Path,
    layer: int,
    activations: torch.Tensor,
    positions: pd.DataFrame,
    selected: Sequence[Mapping],
    hf_model,
    wrapped,
    own_lens: Mapping,
    common_lens: Mapping,
    own_lens_sha256: str,
    common_lens_sha256: str,
    input_manifest_sha256: str,
    specification: Mapping,
    config: Mapping,
) -> dict:
    estimator = config["estimator"]
    uncertainty = config["uncertainty"]
    k_max = int(estimator["k_max"])
    owners = positions["owner"].to_numpy(dtype=np.int64)
    prompt_domains = [str(row["domain"]) for row in selected]
    prompt_counts = stratified_prompt_counts(
        prompt_domains, draws=int(uncertainty["draws"]),
        seed=int(uncertainty["seed"]) + int(layer))
    raw = activations.contiguous().float().cpu()
    global_mean = raw.mean(dim=0)
    centered = (raw - global_mean[None, :]).contiguous()
    if not torch.isfinite(raw).all() or not torch.isfinite(centered).all():
        raise RuntimeError("non-finite captured activation population")
    dictionary_config = estimator["dictionary"]
    random_config = estimator["random_controls"]

    own_dictionary, own_dictionary_diagnostics = _build_j_dictionary(
        hf_model, wrapped, own_lens["J"][int(layer)],
        chunk_rows=int(dictionary_config["chunk_rows"]))
    own_centered, own_raw = _run_targets(
        centered, raw, own_dictionary, estimator=estimator,
        label=f"{specification['slug']} L{layer} own")
    del own_dictionary
    torch.cuda.empty_cache()

    common_is_own = own_lens_sha256 == common_lens_sha256
    if common_is_own:
        common_centered, common_raw = own_centered, own_raw
        common_dictionary_diagnostics = {
            **own_dictionary_diagnostics,
            "reused_exact_own_dictionary": True,
        }
    else:
        common_dictionary, common_dictionary_diagnostics = _build_j_dictionary(
            hf_model, wrapped, common_lens["J"][int(layer)],
            chunk_rows=int(dictionary_config["chunk_rows"]))
        common_centered, common_raw = _run_targets(
            centered, raw, common_dictionary, estimator=estimator,
            label=f"{specification['slug']} L{layer} base-common")
        del common_dictionary
        torch.cuda.empty_cache()

    random_centered_results = []
    random_raw_results = []
    random_diagnostics = []
    vocabulary, dimension = hf_model.get_output_embeddings().weight.shape
    device = hf_model.get_output_embeddings().weight.device
    for seed in random_config["seeds"]:
        random_dictionary, diagnostics = _build_random_dictionary(
            int(vocabulary), int(dimension), seed=int(seed),
            chunk_rows=int(random_config["chunk_rows"]), device=device)
        centered_result, raw_result = _run_targets(
            centered, raw, random_dictionary, estimator=estimator,
            label=f"{specification['slug']} L{layer} random-seed-{seed}")
        random_centered_results.append(centered_result)
        random_raw_results.append(raw_result)
        random_diagnostics.append(diagnostics)
        del random_dictionary
        torch.cuda.empty_cache()

    own_centered_errors = own_centered.errors.numpy().astype(np.float32)
    own_raw_errors = own_raw.errors.numpy().astype(np.float32)
    common_centered_errors = common_centered.errors.numpy().astype(np.float32)
    common_raw_errors = common_raw.errors.numpy().astype(np.float32)
    random_centered_errors = np.stack([
        result.errors.numpy().astype(np.float32)
        for result in random_centered_results])
    random_raw_errors = np.stack([
        result.errors.numpy().astype(np.float32)
        for result in random_raw_results])
    own_summary, own_bootstrap = _summary_with_bootstrap(
        own_centered_errors, random_centered_errors,
        own_raw_errors, random_raw_errors,
        owners=owners, prompt_domains=prompt_domains,
        estimator=estimator, uncertainty=uncertainty,
        prompt_counts=prompt_counts)
    common_summary, common_bootstrap = _summary_with_bootstrap(
        common_centered_errors, random_centered_errors,
        common_raw_errors, random_raw_errors,
        owners=owners, prompt_domains=prompt_domains,
        estimator=estimator, uncertainty=uncertainty,
        prompt_counts=prompt_counts)
    _add_solver_support(
        own_summary,
        centered_achieved=own_centered.achieved_support.numpy(),
        raw_achieved=own_raw.achieved_support.numpy(), k_max=k_max)
    _add_solver_support(
        common_summary,
        centered_achieved=common_centered.achieved_support.numpy(),
        raw_achieved=common_raw.achieved_support.numpy(), k_max=k_max)
    own_summary["dictionary"] = own_dictionary_diagnostics
    common_summary["dictionary"] = common_dictionary_diagnostics
    summary = {
        "layer": int(layer),
        "own": own_summary,
        "base_common": common_summary,
        "common_is_own": bool(common_is_own),
        "random_dictionaries": random_diagnostics,
        "global_mean_l2": float(global_mean.norm()),
        "n_positions": int(len(raw)),
    }
    point_hashes = {
        "own_centered": object_sha256(curve_summary(
            own_centered_errors, random_centered_errors,
            persistence=int(estimator["crossing"]["persistence"]),
            persistence_sensitivity=estimator["crossing"][
                "persistence_sensitivity"])),
        "own_raw": object_sha256(curve_summary(
            own_raw_errors, random_raw_errors,
            persistence=int(estimator["crossing"]["persistence"]),
            persistence_sensitivity=estimator["crossing"][
                "persistence_sensitivity"])),
        "common_centered": object_sha256(curve_summary(
            common_centered_errors, random_centered_errors,
            persistence=int(estimator["crossing"]["persistence"]),
            persistence_sensitivity=estimator["crossing"][
                "persistence_sensitivity"])),
        "common_raw": object_sha256(curve_summary(
            common_raw_errors, random_raw_errors,
            persistence=int(estimator["crossing"]["persistence"]),
            persistence_sensitivity=estimator["crossing"][
                "persistence_sensitivity"])),
    }
    metadata = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": specification["evidence_id"],
        "model_slug": specification["slug"],
        "layer": int(layer),
        "input_manifest_sha256": input_manifest_sha256,
        "own_lens_sha256": own_lens_sha256,
        "common_lens_sha256": common_lens_sha256,
        "point_summary_sha256": point_hashes,
        "summary_sha256": object_sha256(summary),
        "bootstrap_seed": int(uncertainty["seed"]) + int(layer),
    }
    _atomic_npz(
        path,
        metadata_json=np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":"))),
        summary_json=np.asarray(json.dumps(
            summary, sort_keys=True, separators=(",", ":"))),
        owners=owners.astype(np.int16),
        global_mean=global_mean.numpy().astype(np.float32),
        own_centered_errors=own_centered_errors,
        own_raw_errors=own_raw_errors,
        common_centered_errors=common_centered_errors,
        common_raw_errors=common_raw_errors,
        random_centered_errors=random_centered_errors,
        random_raw_errors=random_raw_errors,
        own_centered_selected=own_centered.selected_indices.numpy().astype(
            np.int32),
        common_centered_selected=common_centered.selected_indices.numpy().astype(
            np.int32),
        own_centered_achieved=own_centered.achieved_support.numpy().astype(
            np.int16),
        own_raw_achieved=own_raw.achieved_support.numpy().astype(np.int16),
        common_centered_achieved=common_centered.achieved_support.numpy().astype(
            np.int16),
        common_raw_achieved=common_raw.achieved_support.numpy().astype(np.int16),
        random_centered_achieved=np.stack([
            result.achieved_support.numpy().astype(np.int16)
            for result in random_centered_results]),
        random_raw_achieved=np.stack([
            result.achieved_support.numpy().astype(np.int16)
            for result in random_raw_results]),
        own_bootstrap_centered_excess=own_bootstrap["centered_excess"],
        own_bootstrap_centered_occupancy=own_bootstrap["centered_occupancy"],
        own_bootstrap_raw_excess=own_bootstrap["raw_excess"],
        own_bootstrap_raw_occupancy=own_bootstrap["raw_occupancy"],
        common_bootstrap_centered_excess=common_bootstrap["centered_excess"],
        common_bootstrap_centered_occupancy=common_bootstrap[
            "centered_occupancy"],
        common_bootstrap_raw_excess=common_bootstrap["raw_excess"],
        common_bootstrap_raw_occupancy=common_bootstrap["raw_occupancy"],
    )
    return {"metadata": metadata, "summary": summary}


def _verify_checkpoint(
    path: Path, *, input_manifest_sha256: str, estimator: Mapping,
) -> tuple[dict, dict]:
    record, arrays = _checkpoint_payload(path)
    metadata = record["metadata"]
    if metadata["input_manifest_sha256"] != input_manifest_sha256:
        raise RuntimeError(f"capacity layer checkpoint input drift: {path}")
    if metadata["summary_sha256"] != object_sha256(record["summary"]):
        raise RuntimeError(f"capacity layer summary hash drift: {path}")
    persistence = int(estimator["crossing"]["persistence"])
    sensitivity = estimator["crossing"]["persistence_sensitivity"]
    recomputed = {
        "own_centered": curve_summary(
            arrays["own_centered_errors"], arrays["random_centered_errors"],
            persistence=persistence, persistence_sensitivity=sensitivity),
        "own_raw": curve_summary(
            arrays["own_raw_errors"], arrays["random_raw_errors"],
            persistence=persistence, persistence_sensitivity=sensitivity),
        "common_centered": curve_summary(
            arrays["common_centered_errors"],
            arrays["random_centered_errors"], persistence=persistence,
            persistence_sensitivity=sensitivity),
        "common_raw": curve_summary(
            arrays["common_raw_errors"], arrays["random_raw_errors"],
            persistence=persistence, persistence_sensitivity=sensitivity),
    }
    hashes = {key: object_sha256(value) for key, value in recomputed.items()}
    if hashes != metadata["point_summary_sha256"]:
        raise RuntimeError(f"capacity checkpoint reconstruction drift: {path}")
    return record, arrays


@torch.inference_mode()
def run_model(config_path: Path, config: Mapping, slug: str) -> dict:
    specification = _model_specification(config, slug)
    existing = _registered_output_check(specification["evidence_id"])
    if existing is not None:
        return {"status": "already_registered", "event": existing}
    clean = require_clean_tree(expected_branch=config["branch"])
    _, protocol_sha256, corpus_path = _require_protocol(config)
    selected = _read_jsonl(corpus_path)
    runtime = _verify_runtime(config)
    gpu = require_cuda_gpu()
    reference = _model_reference(specification["model_uri"])
    model_path = resolve_uri(specification["model_uri"])
    snapshot = _snapshot_manifest(model_path, reference)
    if file_sha256(model_path / "tokenizer.json") != specification[
            "tokenizer_json_sha256"]:
        raise RuntimeError("runtime tokenizer.json differs from protocol")
    layers = [int(value) for value in config["activation_population"][
        "layers"]]
    own_lens_path = resolve_uri(specification["own_lens_uri"])
    common_lens_path = resolve_uri(config["common_lens"]["uri"])
    own_lens_sha256 = file_sha256(own_lens_path)
    common_lens_sha256 = file_sha256(common_lens_path)
    if own_lens_sha256 != specification["own_lens_sha256"]:
        raise RuntimeError("own capacity lens hash drift")
    if common_lens_sha256 != config["common_lens"]["sha256"]:
        raise RuntimeError("common capacity lens hash drift")

    output_dir = (metrics_dir(slug) / "capacity"
                  / specification["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    position_path = output_dir / "position_manifest.parquet"
    input_manifest_path = output_dir / "input_manifest.json"
    checkpoint_paths = {
        layer: output_dir / f"capacity_layer_{layer}.npz" for layer in layers}
    result_path = output_dir / "capacity_result.json"
    if result_path.exists():
        raise FileExistsError(
            "unregistered capacity result exists; audit before rerunning")

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True)
    if tokenizer.bos_token_id != specification["expected_bos_token_id"]:
        raise RuntimeError("runtime BOS token differs from protocol")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True).to("cuda").eval()
    assert_model_on_cuda(hf_model)
    print(
        f"{slug}: model loaded on {gpu['name']}; capturing {len(selected)} "
        f"prompts at layers {layers}", flush=True)
    activations, positions, tokenization, wrapped = _capture_activations(
        hf_model, tokenizer, selected=selected, layers=layers,
        population=config["activation_population"])
    position_payload_sha256 = object_sha256(
        positions.to_dict(orient="records"))
    if position_path.exists():
        existing_positions = pd.read_parquet(position_path)
        if object_sha256(existing_positions.to_dict(
                orient="records")) != position_payload_sha256:
            raise RuntimeError("existing capacity position manifest drift")
    else:
        _atomic_parquet(position_path, positions)
    input_payload = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "experiment_id": specification["evidence_id"],
        "git": clean,
        "config_sha256": file_sha256(config_path),
        "protocol_evidence_id": config["protocol_evidence_id"],
        "protocol_sha256": protocol_sha256,
        "corpus_sha256": file_sha256(corpus_path),
        "model": reference,
        "model_snapshot": snapshot,
        "tokenization": tokenization,
        "position_manifest_payload_sha256": position_payload_sha256,
        "position_manifest_file_sha256": file_sha256(position_path),
        "own_lens": {
            "uri": specification["own_lens_uri"],
            "sha256": own_lens_sha256,
        },
        "common_lens": {
            "uri": config["common_lens"]["uri"],
            "sha256": common_lens_sha256,
        },
        "runtime": runtime,
        "gpu": gpu,
        "activation_population_sha256": object_sha256(
            config["activation_population"]),
        "estimator_sha256": object_sha256(config["estimator"]),
        "uncertainty_sha256": object_sha256(config["uncertainty"]),
        "decision_rules_sha256": object_sha256(config["decision_rules"]),
    }
    input_envelope = {
        "schema_version": 1, "payload": input_payload,
        "payload_sha256": object_sha256(input_payload),
    }
    _write_json_once(input_manifest_path, input_envelope)
    input_manifest_sha256 = input_envelope["payload_sha256"]
    own_lens = _lens_container(
        own_lens_path, own_lens_sha256, layers)
    common_lens = (
        own_lens if own_lens_sha256 == common_lens_sha256 else
        _lens_container(common_lens_path, common_lens_sha256, layers))
    for layer in layers:
        checkpoint = checkpoint_paths[layer]
        if checkpoint.exists():
            _verify_checkpoint(
                checkpoint, input_manifest_sha256=input_manifest_sha256,
                estimator=config["estimator"])
            print(f"{slug}: verified existing L{layer} checkpoint", flush=True)
            continue
        print(f"{slug}: processing capacity layer {layer}", flush=True)
        _process_layer(
            path=checkpoint, layer=layer, activations=activations[layer],
            positions=positions, selected=selected, hf_model=hf_model,
            wrapped=wrapped, own_lens=own_lens, common_lens=common_lens,
            own_lens_sha256=own_lens_sha256,
            common_lens_sha256=common_lens_sha256,
            input_manifest_sha256=input_manifest_sha256,
            specification=specification, config=config)
        _verify_checkpoint(
            checkpoint, input_manifest_sha256=input_manifest_sha256,
            estimator=config["estimator"])
        print(
            f"{slug}: durable L{layer} checkpoint "
            f"{file_sha256(checkpoint)[:12]}...", flush=True)

    per_layer = {}
    for layer, checkpoint in checkpoint_paths.items():
        record, _ = _verify_checkpoint(
            checkpoint, input_manifest_sha256=input_manifest_sha256,
            estimator=config["estimator"])
        per_layer[str(layer)] = record["summary"]
    payload = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": specification["evidence_id"],
        "status": "complete",
        "model_slug": slug,
        "model_role": specification["role"],
        "model": reference,
        "model_snapshot_semantic_manifest_sha256": snapshot[
            "semantic_manifest_sha256"],
        "protocol_evidence_id": config["protocol_evidence_id"],
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "tokenization": tokenization,
        "position_manifest_payload_sha256": position_payload_sha256,
        "frames": config["frames"],
        "per_layer": per_layer,
        "definitions": {
            "primary_centered": (
                "Sparse nonnegative pursuit of h minus the full frozen "
                "activation-population mean; J centered R2 minus the mean "
                "of three matched random centered R2 values at the lower-"
                "median crossing occupancy."),
            "raw_sensitivity": (
                "Separate pursuit of uncentered h with its own crossing; "
                "raw energy R2 excess, never labeled centered variance."),
            "own": "checkpoint activation and checkpoint-own lens",
            "base_common": (
                "checkpoint activation transported with frozen Base lens"),
        },
        "claim_boundary": config["claim_boundary"],
        "interventions_opened": False,
        "phase4_outputs_opened": False,
    }
    command = (
        "python -m jspace_olmo_lineage.experiments.capacity "
        f"--config {config_path} --model-slug {slug}")
    write_result(
        payload, result_path,
        Provenance(
            evidence_id=specification["evidence_id"], tier="development",
            command=command,
            inputs={
                "config": file_sha256(config_path),
                "protocol": protocol_sha256,
                "corpus": file_sha256(corpus_path),
                "own_lens": own_lens_sha256,
                "common_lens": common_lens_sha256,
            },
            input_manifest_sha256=input_manifest_sha256,
            model=reference,
            seed_contract=(
                "random dictionaries 11/12/13; stratified prompt bootstrap "
                f"{config['uncertainty']['seed']}+layer"),
        ))
    outputs = [input_manifest_path, position_path,
               *[checkpoint_paths[layer] for layer in layers], result_path]
    event = create(
        specification["evidence_id"], tier="development",
        what=(
            f"Symmetric centered-target sparse-capacity measurement for "
            f"{slug} at layers 24/32/40, own and Base-common lens frames, "
            "with three random dictionaries, raw sensitivity, strata, and "
            "prompt-bootstrap intervals; no intervention outcome."),
        command=command, outputs=outputs,
        inputs={
            "config": file_sha256(config_path),
            "protocol": protocol_sha256,
            "corpus": file_sha256(corpus_path),
            "own_lens": own_lens_sha256,
            "common_lens": common_lens_sha256,
            "input_manifest": input_manifest_sha256,
        },
        model_slug=slug,
        capacity_complete=True,
        interventions_opened=False,
        phase4_outputs_opened=False,
    )
    del activations, own_lens, common_lens, wrapped, hf_model
    torch.cuda.empty_cache()
    return {"payload": payload, "event": event}


def _registered_model_data(config: Mapping, specification: Mapping) -> dict:
    event = _registered_output_check(specification["evidence_id"])
    if event is None:
        raise RuntimeError(
            f"capacity model evidence is absent: {specification['evidence_id']}")
    output_dir = (metrics_dir(specification["slug"]) / "capacity"
                  / specification["evidence_id"])
    result_path = output_dir / "capacity_result.json"
    position_path = output_dir / "position_manifest.parquet"
    layers = [int(value) for value in config["activation_population"][
        "layers"]]
    checkpoint_paths = {
        layer: output_dir / f"capacity_layer_{layer}.npz" for layer in layers}
    output_records = {Path(row["path"]).resolve(): row
                      for row in event["outputs"]}
    for path in [result_path, position_path, *checkpoint_paths.values()]:
        record = output_records.get(path.resolve())
        if record is None or file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"registered capacity model output drift: {path}")
    result_envelope = json.loads(result_path.read_text())
    if result_envelope["payload_sha256"] != object_sha256(
            result_envelope["payload"]):
        raise RuntimeError("capacity model result envelope drift")
    checkpoints = {}
    for layer, path in checkpoint_paths.items():
        record, arrays = _checkpoint_payload(path)
        if record["metadata"]["summary_sha256"] != object_sha256(
                record["summary"]):
            raise RuntimeError("capacity checkpoint summary drift")
        checkpoints[layer] = {"record": record, "arrays": arrays,
                              "path": path}
    return {
        "event": event,
        "result": result_envelope["payload"],
        "result_sha256": file_sha256(result_path),
        "positions": pd.read_parquet(position_path),
        "position_sha256": file_sha256(position_path),
        "checkpoints": checkpoints,
    }


def _frame_arrays(checkpoint: Mapping, frame: str) -> tuple[np.ndarray, ...]:
    arrays = checkpoint["arrays"]
    prefix = "own" if frame == "own" else "common"
    return (
        arrays[f"{prefix}_centered_errors"],
        arrays["random_centered_errors"],
        arrays[f"{prefix}_centered_selected"],
        arrays[f"{prefix}_centered_achieved"],
        arrays["owners"].astype(np.int64),
    )


def _support_overlap(
    left_checkpoint: Mapping,
    right_checkpoint: Mapping,
    left_positions: pd.DataFrame,
    right_positions: pd.DataFrame,
    *,
    frame: str,
    persistence: int,
) -> dict:
    left_keys = list(zip(
        left_positions["pid"].astype(int),
        left_positions["content_token_position"].astype(int)))
    right_keys = list(zip(
        right_positions["pid"].astype(int),
        right_positions["content_token_position"].astype(int)))
    right_lookup = {key: index for index, key in enumerate(right_keys)}
    left_errors, left_random, left_selected, left_achieved, _ = _frame_arrays(
        left_checkpoint, frame)
    right_errors, right_random, right_selected, right_achieved, _ = _frame_arrays(
        right_checkpoint, frame)
    left_occupancy = occupancy_from_errors(
        left_errors, left_random, persistence=persistence)
    right_occupancy = occupancy_from_errors(
        right_errors, right_random, persistence=persistence)
    jaccards = []
    exact = []
    left_sizes = []
    right_sizes = []
    for left_index, key in enumerate(left_keys):
        right_index = right_lookup.get(key)
        if right_index is None:
            continue
        left_k = min(
            int(left_occupancy[left_index]), int(left_achieved[left_index]))
        right_k = min(
            int(right_occupancy[right_index]), int(right_achieved[right_index]))
        left_set = set(map(int, left_selected[left_index, :left_k]))
        right_set = set(map(int, right_selected[right_index, :right_k]))
        union = left_set | right_set
        jaccards.append(
            1.0 if not union else len(left_set & right_set) / len(union))
        exact.append(left_set == right_set)
        left_sizes.append(len(left_set))
        right_sizes.append(len(right_set))
    if not jaccards:
        raise RuntimeError("no content positions align for support overlap")
    return {
        "n_aligned_positions": len(jaccards),
        "n_left_positions": len(left_keys),
        "n_right_positions": len(right_keys),
        "mean_jaccard": float(np.mean(jaccards)),
        "median_jaccard": float(np.median(jaccards)),
        "exact_support_fraction": float(np.mean(exact)),
        "mean_left_support": float(np.mean(left_sizes)),
        "mean_right_support": float(np.mean(right_sizes)),
    }


def _lower_median_axis(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("lower-median axis input must be [layers,draws]")
    index = (array.shape[0] - 1) // 2
    return np.partition(array, index, axis=0)[index]


def aggregate_joint(config_path: Path, config: Mapping) -> dict:
    existing = _registered_output_check(config["joint_evidence_id"])
    if existing is not None:
        return {"status": "already_registered", "event": existing}
    clean = require_clean_tree(expected_branch=config["branch"])
    _, protocol_sha256, corpus_path = _require_protocol(config)
    model_data = {
        row["slug"]: _registered_model_data(config, row)
        for row in config["models"]
    }
    models = [row["slug"] for row in config["models"]]
    layers = [int(value) for value in config["activation_population"][
        "layers"]]
    prompt_domains = [row["domain"] for row in _read_jsonl(corpus_path)]
    estimator = config["estimator"]
    uncertainty = config["uncertainty"]
    decision = config["decision_rules"]
    persistence = int(estimator["crossing"]["persistence"])
    level = float(uncertainty["interval_level"])
    frames = ("own", "base_common")
    frame_array_name = {"own": "own", "base_common": "common"}
    table_rows = []
    bootstrap_arrays = {}
    model_bootstrap = {}
    model_points = {}
    pair_layer_rows = {}

    for layer in layers:
        counts = stratified_prompt_counts(
            prompt_domains, draws=int(uncertainty["draws"]),
            seed=int(uncertainty["seed"]) + layer)
        for frame in frames:
            stored_frame = frame_array_name[frame]
            for slug in models:
                checkpoint = model_data[slug]["checkpoints"][layer]
                j_errors, random_errors, _, _, owners = _frame_arrays(
                    checkpoint, stored_frame)
                point = curve_summary(
                    j_errors, random_errors, persistence=persistence,
                    persistence_sensitivity=estimator["crossing"][
                        "persistence_sensitivity"])
                boot = bootstrap_estimates(
                    j_errors, random_errors, owners=owners,
                    prompt_counts=counts, persistence=persistence)
                model_points[(slug, frame, layer)] = point
                model_bootstrap[(slug, frame, layer)] = boot
                prefix = f"model__{slug}__{frame}__L{layer}"
                bootstrap_arrays[f"{prefix}__centered_excess"] = boot[
                    "excess_share"]
                bootstrap_arrays[f"{prefix}__occupancy"] = boot[
                    "occupancy_median"]
                excess_interval = percentile_interval(
                    boot["excess_share"], level)
                occupancy_interval = percentile_interval(
                    boot["occupancy_median"], level)
                table_rows.append({
                    "row_type": "model_estimate",
                    "left": slug,
                    "right": None,
                    "frame": frame,
                    "layer": str(layer),
                    "centered_excess": point["excess_share"],
                    "centered_difference": None,
                    "centered_ci_low": excess_interval["low"],
                    "centered_ci_high": excess_interval["high"],
                    "occupancy_median": point["occupancy_median"],
                    "occupancy_difference": None,
                    "occupancy_ci_low": occupancy_interval["low"],
                    "occupancy_ci_high": occupancy_interval["high"],
                    "classification": None,
                })
            for left, right in itertools.combinations(models, 2):
                left_point = model_points[(left, frame, layer)]
                right_point = model_points[(right, frame, layer)]
                left_boot = model_bootstrap[(left, frame, layer)]
                right_boot = model_bootstrap[(right, frame, layer)]
                excess_difference = (
                    right_boot["excess_share"] - left_boot["excess_share"])
                occupancy_difference = (
                    right_boot["occupancy_median"]
                    - left_boot["occupancy_median"])
                excess_interval = percentile_interval(excess_difference, level)
                occupancy_interval = percentile_interval(
                    occupancy_difference, level)
                point_excess_difference = float(
                    right_point["excess_share"] - left_point["excess_share"])
                point_occupancy_difference = int(
                    right_point["occupancy_median"]
                    - left_point["occupancy_median"])
                classification = classify_shift(
                    centered_difference=point_excess_difference,
                    centered_interval_low=excess_interval["low"],
                    centered_interval_high=excess_interval["high"],
                    occupancy_difference=point_occupancy_difference,
                    occupancy_interval_low=occupancy_interval["low"],
                    occupancy_interval_high=occupancy_interval["high"],
                    equivalence_margin=decision[
                        "centered_excess_equivalence_margin"],
                    material_margin=decision[
                        "centered_excess_material_margin"])
                support = _support_overlap(
                    model_data[left]["checkpoints"][layer],
                    model_data[right]["checkpoints"][layer],
                    model_data[left]["positions"],
                    model_data[right]["positions"],
                    frame=stored_frame, persistence=persistence)
                row = {
                    "row_type": "pair_contrast",
                    "left": left,
                    "right": right,
                    "frame": frame,
                    "layer": str(layer),
                    "centered_excess": None,
                    "centered_difference": point_excess_difference,
                    "centered_ci_low": excess_interval["low"],
                    "centered_ci_high": excess_interval["high"],
                    "occupancy_median": None,
                    "occupancy_difference": point_occupancy_difference,
                    "occupancy_ci_low": occupancy_interval["low"],
                    "occupancy_ci_high": occupancy_interval["high"],
                    "classification": classification,
                    "support_overlap": support,
                }
                table_rows.append(row)
                pair_layer_rows[(left, right, frame, layer)] = row
                prefix = f"pair__{left}__{right}__{frame}__L{layer}"
                bootstrap_arrays[f"{prefix}__centered_difference"] = (
                    excess_difference)
                bootstrap_arrays[f"{prefix}__occupancy_difference"] = (
                    occupancy_difference)

    aggregate_rows = {}
    for frame in frames:
        for left, right in itertools.combinations(models, 2):
            left_points = [model_points[(left, frame, layer)] for layer in layers]
            right_points = [model_points[(right, frame, layer)] for layer in layers]
            point_excess = float(np.mean([
                right_value["excess_share"] - left_value["excess_share"]
                for left_value, right_value in zip(
                    left_points, right_points, strict=True)]))
            left_occ = lower_median([
                value["occupancy_median"] for value in left_points])
            right_occ = lower_median([
                value["occupancy_median"] for value in right_points])
            point_occ = int(right_occ - left_occ)
            layer_excess_boot = np.stack([
                model_bootstrap[(right, frame, layer)]["excess_share"]
                - model_bootstrap[(left, frame, layer)]["excess_share"]
                for layer in layers])
            excess_boot = layer_excess_boot.mean(axis=0)
            left_occ_boot = _lower_median_axis(np.stack([
                model_bootstrap[(left, frame, layer)]["occupancy_median"]
                for layer in layers]))
            right_occ_boot = _lower_median_axis(np.stack([
                model_bootstrap[(right, frame, layer)]["occupancy_median"]
                for layer in layers]))
            occ_boot = right_occ_boot - left_occ_boot
            excess_interval = percentile_interval(excess_boot, level)
            occ_interval = percentile_interval(occ_boot, level)
            classification = classify_shift(
                centered_difference=point_excess,
                centered_interval_low=excess_interval["low"],
                centered_interval_high=excess_interval["high"],
                occupancy_difference=point_occ,
                occupancy_interval_low=occ_interval["low"],
                occupancy_interval_high=occ_interval["high"],
                equivalence_margin=decision[
                    "centered_excess_equivalence_margin"],
                material_margin=decision["centered_excess_material_margin"])
            row = {
                "row_type": "pair_contrast_equal_layer",
                "left": left, "right": right, "frame": frame,
                "layer": "equal_layer_mean",
                "centered_excess": None,
                "centered_difference": point_excess,
                "centered_ci_low": excess_interval["low"],
                "centered_ci_high": excess_interval["high"],
                "occupancy_median": None,
                "occupancy_difference": point_occ,
                "occupancy_ci_low": occ_interval["low"],
                "occupancy_ci_high": occ_interval["high"],
                "classification": classification,
            }
            table_rows.append(row)
            aggregate_rows[(left, right, frame)] = row
            prefix = f"pair__{left}__{right}__{frame}__equal_layer"
            bootstrap_arrays[f"{prefix}__centered_difference"] = excess_boot
            bootstrap_arrays[f"{prefix}__occupancy_difference"] = occ_boot

    primary = aggregate_rows[("olmo3-base", "olmo3-think", "own")]
    positive_material_layers = [
        pair_layer_rows[("olmo3-base", "olmo3-think", "own", layer)]
        for layer in layers
        if pair_layer_rows[("olmo3-base", "olmo3-think", "own", layer)][
            "classification"] == "material_shift"
        and pair_layer_rows[("olmo3-base", "olmo3-think", "own", layer)][
            "centered_difference"] > 0
    ]
    if (primary["classification"] == "material_shift"
            and primary["centered_difference"] > 0):
        lineage_verdict = "partial_channel_formation_or_growth_supported"
    elif (primary["classification"] in {"stable", "small_shift"}
          and not positive_material_layers):
        lineage_verdict = (
            "broadly_conserved_capacity_recruitment_consistent")
    else:
        lineage_verdict = "mixed_or_unresolved"

    result_path = resolve_uri(config["outputs"]["joint_result"], must_exist=False)
    table_path = resolve_uri(config["outputs"]["joint_table"], must_exist=False)
    bootstrap_path = resolve_uri(
        config["outputs"]["joint_bootstrap"], must_exist=False)
    collisions = [str(path) for path in (result_path, table_path, bootstrap_path)
                  if path.exists()]
    if collisions:
        raise FileExistsError(
            "unregistered joint capacity outputs exist: "
            + ", ".join(collisions))
    serial_table = []
    for row in table_rows:
        serial = dict(row)
        if "support_overlap" in serial:
            serial["support_overlap_json"] = json.dumps(
                serial.pop("support_overlap"), sort_keys=True)
        serial_table.append(serial)
    _atomic_parquet(table_path, pd.DataFrame(serial_table))
    _atomic_npz(bootstrap_path, **bootstrap_arrays)
    payload = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": config["joint_evidence_id"],
        "status": "complete",
        "protocol_evidence_id": config["protocol_evidence_id"],
        "protocol_sha256": protocol_sha256,
        "model_order": models,
        "model_roles": {row["slug"]: row["role"]
                        for row in config["models"]},
        "model_result_sha256": {
            slug: data["result_sha256"] for slug, data in model_data.items()},
        "layers": layers,
        "frames": config["frames"],
        "table": table_rows,
        "headline_base_to_3_0_think": primary,
        "lineage_verdict": lineage_verdict,
        "positive_material_individual_layers": [
            row["layer"] for row in positive_material_layers],
        "decision_rules": decision,
        "bootstrap": {
            "draws": int(uncertainty["draws"]),
            "interval_level": level,
            "unit": uncertainty["unit"],
            "stratified": True,
            "paired_across_models": True,
            "distribution_sha256": file_sha256(bootstrap_path),
        },
        "interpretation": {
            "instruct_role": "sibling endpoint, not fourth trajectory point",
            "natural_experiment_limit": (
                "Matched released checkpoints do not randomize training "
                "objective or isolate its causal contribution."),
        },
        "claim_boundary": config["claim_boundary"],
        "interventions_opened": False,
        "phase4_outputs_opened": False,
    }
    command = (
        "python -m jspace_olmo_lineage.experiments.capacity "
        f"--config {config_path} --aggregate-joint")
    write_result(
        payload, result_path,
        Provenance(
            evidence_id=config["joint_evidence_id"], tier="development",
            command=command,
            inputs={
                "config": file_sha256(config_path),
                "protocol": protocol_sha256,
                "corpus": file_sha256(corpus_path),
                **{f"model:{slug}": data["result_sha256"]
                   for slug, data in model_data.items()},
            },
            input_manifest_sha256=object_sha256({
                "protocol": protocol_sha256,
                "models": {slug: data["result_sha256"]
                           for slug, data in model_data.items()},
                "uncertainty": object_sha256(uncertainty),
                "decision": object_sha256(decision),
            }),
            seed_contract=(
                "shared stratified prompt draws, seed "
                f"{uncertainty['seed']}+layer"),
        ))
    event = create(
        config["joint_evidence_id"], tier="development",
        what=(
            "Paired four-checkpoint OLMo capacity table and H1-versus-"
            "formation router across layers 24/32/40, own and Base-common "
            "frames; Instruct retained as sibling endpoint."),
        command=command, outputs=[result_path, table_path, bootstrap_path],
        inputs={
            "config": file_sha256(config_path),
            "protocol": protocol_sha256,
            **{f"model:{slug}": data["result_sha256"]
               for slug, data in model_data.items()},
        },
        lineage_verdict=lineage_verdict,
        interventions_opened=False,
        phase4_outputs_opened=False,
    )
    return {"payload": payload, "event": event}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-protocol", action="store_true")
    group.add_argument("--model-slug")
    group.add_argument("--aggregate-joint", action="store_true")
    arguments = parser.parse_args()
    config_path = Path(arguments.config)
    config = _load_config(config_path)
    if arguments.freeze_protocol:
        result = freeze_protocol(config_path, config)
    elif arguments.model_slug:
        result = run_model(config_path, config, arguments.model_slug)
    else:
        result = aggregate_joint(config_path, config)
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
