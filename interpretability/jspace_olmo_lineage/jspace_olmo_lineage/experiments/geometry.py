"""O3 same-corpus geometry trajectory and staged readout extraction.

The producer has four explicit evidence boundaries:

``freeze-protocol``
    Resolve the complete stable/token/selection row population before any new
    operator comparison is computed.
``extract-readout``
    Extract only the frozen unembedding rows and final norm from one exact Hub
    revision, allowing the large model shards to be rotated safely.
``aggregate``
    Compare the four registered lenses/readouts and the already-registered O2
    support arrays without loading a language model.
``figures``
    Render only from registered tables.

No command reads an untouched Phase 4 intervention outcome or writes outside
the isolated OLMo run root (apart from caller-owned local staging).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from tokenizers import Tokenizer

from ..capacity import occupancy_from_errors
from ..geometry import (
    aggregate_projector_metrics,
    centered_linear_cka_gram,
    id_selection_metrics,
    marginal_crossing_margins,
    neighbor_overlap,
    operator_pair_metrics,
    persistent_direction_summary,
    quantile_summary,
    random_transport_metrics,
    randomized_spectrum,
    row_cosines,
    selection_prefixes,
    set_jaccard,
)
from ..manifests import (
    atomic_json,
    file_sha256,
    git_info,
    object_sha256,
    require_clean_tree,
)
from ..paths import figures_dir, local_work, manifests_dir, metrics_dir, resolve_uri
from ..registry import RegistryError, create, resolve

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/ol_geometry_v1.yaml"


def load_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("geometry config must be a mapping")
    return value


def _model(config: Mapping, slug: str) -> dict:
    rows = [dict(row) for row in config["models"] if row["slug"] == slug]
    if len(rows) != 1:
        raise RuntimeError(f"unknown or duplicate model slug: {slug}")
    return rows[0]


def _verify_event(evidence_id: str) -> dict:
    event = resolve(evidence_id)
    if not event["live"]:
        raise RuntimeError(f"required evidence is not live: {evidence_id}")
    for output in event.get("outputs", []):
        path = Path(output["path"])
        if not path.is_file():
            raise RuntimeError(f"registered output is absent: {path}")
        observed = file_sha256(path)
        if observed != output["sha256"]:
            raise RuntimeError(
                f"registered output hash drift: {path}: {observed}")
    return event


def _already_registered(evidence_id: str) -> dict | None:
    try:
        event = _verify_event(evidence_id)
    except RegistryError as error:
        if "found 0" in str(error):
            return None
        raise
    return event


def _registered_output(event: Mapping, name: str) -> dict:
    matches = [row for row in event.get("outputs", [])
               if Path(row["path"]).name == name]
    if len(matches) != 1:
        raise RuntimeError(
            f"{event['evidence_id']} expected one registered {name}, "
            f"found {len(matches)}")
    return dict(matches[0])


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _flatten_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_strings(item))
        return result
    return []


def _tokenizer_inventory(config: Mapping) -> tuple[list[Tokenizer], dict]:
    tokenizers = []
    rows = []
    vocabulary = int(config["geometry_series"]["vocabulary_size"])
    for model in config["models"]:
        snapshot = Path(model["tokenizer_snapshot"])
        source = snapshot / "tokenizer.json"
        observed = file_sha256(source)
        if observed != model["tokenizer_json_sha256"]:
            raise RuntimeError(
                f"tokenizer hash drift for {model['slug']}: {observed}")
        tokenizer = Tokenizer.from_file(str(source))
        if tokenizer.get_vocab_size(with_added_tokens=True) != vocabulary:
            raise RuntimeError(f"vocabulary size drift for {model['slug']}")
        tokenizers.append(tokenizer)
        rows.append({
            "slug": model["slug"],
            "snapshot": str(snapshot),
            "tokenizer_json_sha256": observed,
            "vocabulary_size": tokenizer.get_vocab_size(
                with_added_tokens=True),
        })
    return tokenizers, {"models": rows}


def _is_special_token(value: str | None) -> bool:
    if value is None:
        return True
    return (
        value.startswith("<|") or value.startswith("<extra_id_")
        or (value.startswith("<") and value.endswith(">"))
    )


def _stable_vocabulary(config: Mapping, tokenizers: Sequence[Tokenizer]
                       ) -> tuple[list[int], dict[int, str], dict]:
    vocabulary = int(config["geometry_series"]["vocabulary_size"])
    common = []
    labels = {}
    excluded_special = 0
    excluded_string_mismatch = 0
    for token_id in range(vocabulary):
        values = [tokenizer.id_to_token(token_id) for tokenizer in tokenizers]
        if len(set(values)) != 1:
            excluded_string_mismatch += 1
            continue
        if _is_special_token(values[0]):
            excluded_special += 1
            continue
        common.append(token_id)
        labels[token_id] = str(values[0])
    sample_size = int(config["sampling"]["stable_vocabulary_rows"])
    if len(common) < sample_size:
        raise RuntimeError("stable common vocabulary is smaller than sample")
    priority = sorted(common, key=lambda token_id: hashlib.sha256(
        f"ol-geometry-stable-vocab-v1:{token_id}:{labels[token_id]}".encode(
            "utf-8")).digest())
    sampled = sorted(priority[:sample_size])
    return sampled, labels, {
        "common_nonspecial_token_ids": len(common),
        "excluded_special_token_ids": excluded_special,
        "excluded_token_string_mismatch_ids": excluded_string_mismatch,
        "stable_sample_token_ids": sampled,
        "stable_sample_token_ids_sha256": object_sha256(sampled),
        "selection_rule": config["sampling"][
            "stable_vocabulary_selection"],
    }


def _task_token_inventory(config: Mapping, tokenizers: Sequence[Tokenizer],
                          stable_labels: Mapping[int, str]) -> tuple[dict, dict]:
    strata: dict[str, set[int]] = defaultdict(set)
    sources = []
    mismatched_texts = []
    bank_all: dict[str, set[int]] = defaultdict(set)
    for source in config["task_token_sources"]:
        path = resolve_uri(source["uri"])
        observed = file_sha256(path)
        if observed != source["sha256"]:
            raise RuntimeError(f"task-token source hash drift: {source['id']}")
        rows = _jsonl(path)
        field_counts = defaultdict(int)
        for row in rows:
            for field in source["fields"]:
                for text in _flatten_strings(row.get(field)):
                    sequences = [tokenizer.encode(
                        text, add_special_tokens=False).ids
                        for tokenizer in tokenizers]
                    if any(sequence != sequences[0]
                           for sequence in sequences[1:]):
                        mismatched_texts.append({
                            "source": source["id"], "field": field,
                            "text_sha256": hashlib.sha256(
                                text.encode("utf-8")).hexdigest(),
                        })
                        continue
                    stable_ids = [int(value) for value in sequences[0]
                                  if int(value) in stable_labels]
                    key = f"{source['id']}_{field}"
                    strata[key].update(stable_ids)
                    bank_all[source["id"]].update(stable_ids)
                    field_counts[field] += 1
        sources.append({
            "id": source["id"], "uri": source["uri"],
            "path": str(path), "sha256": observed, "rows": len(rows),
            "field_text_counts": dict(field_counts),
            "intervention_outcomes_present": bool(
                source.get("intervention_outcomes_present", False)),
        })
    if set(bank_all) >= {"bank_f", "bank_s", "bank_w"}:
        shared = set.intersection(
            bank_all["bank_f"], bank_all["bank_s"], bank_all["bank_w"])
    else:
        shared = set()
    strata["cross_bank_shared"] = shared
    frozen = {key: sorted(value) for key, value in sorted(strata.items())}
    return frozen, {
        "sources": sources,
        "strata_token_ids": frozen,
        "strata_token_ids_sha256": object_sha256(frozen),
        "tokenization_mismatch_texts": mismatched_texts,
        "tokenization_mismatch_count": len(mismatched_texts),
    }


def _capacity_layer_inputs(config: Mapping) -> tuple[dict, set[int]]:
    layers = [int(value) for value in config["geometry_series"][
        "assay_layers"]]
    rows = {}
    union: set[int] = set()
    expected_positions = None
    for model in config["models"]:
        event = _verify_event(model["capacity_evidence_id"])
        model_rows = {}
        for layer in layers:
            output = _registered_output(event, f"capacity_layer_{layer}.npz")
            path = Path(output["path"])
            with np.load(path, allow_pickle=False) as data:
                occupancy = occupancy_from_errors(
                    data["own_centered_errors"],
                    data["random_centered_errors"], persistence=2)
                prefixes = selection_prefixes(
                    data["own_centered_selected"], occupancy,
                    data["own_centered_achieved"])
                selected_ids = {value for prefix in prefixes for value in prefix}
                union.update(selected_ids)
                positions = len(prefixes)
                if expected_positions is None:
                    expected_positions = positions
                if positions != expected_positions:
                    raise RuntimeError("capacity position population drift")
                model_rows[str(layer)] = {
                    "path": str(path), "sha256": output["sha256"],
                    "bytes": output["bytes"], "n_positions": positions,
                    "selected_prefix_token_ids": len(selected_ids),
                    "selected_prefix_cells": int(sum(map(len, prefixes))),
                    "maximum_crossing_occupancy": int(occupancy.max()),
                    "crossing_occupancy_sha256": object_sha256(
                        occupancy.astype(int).tolist()),
                }
        rows[model["slug"]] = {
            "capacity_evidence_id": model["capacity_evidence_id"],
            "layers": model_rows,
        }
    return {
        "models": rows,
        "n_positions": int(expected_positions or 0),
        "selection_union_token_ids": sorted(union),
        "selection_union_token_ids_sha256": object_sha256(sorted(union)),
    }, union


def freeze_protocol(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    existing = _already_registered(config["protocol_evidence_id"])
    if existing is not None:
        return {"status": "already-registered-and-verified",
                "evidence_id": existing["evidence_id"]}
    clean = require_clean_tree(expected_branch=config["branch"])
    provenance = _verify_event(
        config["required_upstream"]["provenance_audit"]["evidence_id"])
    _verify_event(config["required_upstream"]["capacity_protocol"][
        "evidence_id"])
    _verify_event(config["required_upstream"]["capacity_joint"][
        "evidence_id"])
    audit_output = _registered_output(
        provenance, "ol_lens_provenance_audit_v1.json")
    if audit_output["sha256"] != config["required_upstream"][
            "provenance_audit"]["result_sha256"]:
        raise RuntimeError("provenance-audit result hash drift")

    tokenizers, tokenizer_inventory = _tokenizer_inventory(config)
    stable_sample, stable_labels, stable_inventory = _stable_vocabulary(
        config, tokenizers)
    task_strata, task_inventory = _task_token_inventory(
        config, tokenizers, stable_labels)
    capacity_inputs, selection_union = _capacity_layer_inputs(config)
    causal_source = Path(config["prior_causal_utilization"]["table_path"])
    causal_source_sha256 = file_sha256(causal_source)
    if causal_source_sha256 != config["prior_causal_utilization"][
            "table_sha256"]:
        raise RuntimeError("prior causal-utilization table hash drift")
    stable_selection_union = sorted(
        value for value in selection_union if value in stable_labels)
    excluded_selection = sorted(selection_union - set(stable_selection_union))
    task_union = sorted({value for row in task_strata.values() for value in row})
    all_rows = sorted(set(stable_sample) | set(task_union)
                      | set(selection_union))
    token_strings = {
        str(value): str(tokenizers[0].id_to_token(value))
        for value in all_rows
    }
    row_manifest = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": config["protocol_evidence_id"],
        "code_commit": clean["code_commit"],
        "stable_sample_token_ids": stable_sample,
        "cka_token_ids": stable_sample[:int(config["sampling"]["cka_rows"])],
        "neighbor_token_ids": stable_sample[:int(
            config["sampling"]["neighbor_rows"])],
        "task_strata_token_ids": task_strata,
        "task_union_token_ids": task_union,
        "selection_union_token_ids": sorted(selection_union),
        "selection_common_semantic_token_ids": stable_selection_union,
        "selection_token_ids_excluded_as_noncommon_or_special": (
            excluded_selection),
        "all_extracted_token_ids": all_rows,
        "token_strings": token_strings,
        "hashes": {
            "stable_sample_token_ids": object_sha256(stable_sample),
            "cka_token_ids": object_sha256(
                stable_sample[:int(config["sampling"]["cka_rows"])]),
            "neighbor_token_ids": object_sha256(
                stable_sample[:int(config["sampling"]["neighbor_rows"])]),
            "task_strata_token_ids": object_sha256(task_strata),
            "selection_union_token_ids": object_sha256(
                sorted(selection_union)),
            "selection_common_semantic_token_ids": object_sha256(
                stable_selection_union),
            "all_extracted_token_ids": object_sha256(all_rows),
            "token_strings": object_sha256(token_strings),
        },
        "counts": {
            "stable_sample": len(stable_sample),
            "task_union": len(task_union),
            "selection_union_before_common_filter": len(selection_union),
            "selection_union": len(selection_union),
            "selection_common_semantic": len(stable_selection_union),
            "selection_excluded": len(excluded_selection),
            "all_extracted": len(all_rows),
        },
    }
    row_manifest["semantic_sha256"] = object_sha256(row_manifest)
    row_path = resolve_uri(config["outputs"]["row_manifest"], must_exist=False)
    protocol_path = resolve_uri(config["outputs"]["protocol"], must_exist=False)
    atomic_json(row_path, row_manifest)

    lens_inputs = []
    for model in config["models"]:
        lens_path = resolve_uri(model["lens_uri"])
        observed = file_sha256(lens_path)
        if observed != model["lens_sha256"]:
            raise RuntimeError(f"lens hash drift for {model['slug']}")
        lens_inputs.append({
            "slug": model["slug"], "uri": model["lens_uri"],
            "path": str(lens_path), "sha256": observed,
            "bytes": int(lens_path.stat().st_size),
        })
    protocol = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": config["protocol_evidence_id"],
        "status": "frozen-before-new-o3-geometry-outcomes",
        "code_commit": clean["code_commit"],
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "required_upstream": config["required_upstream"],
        "provenance_audit_output": audit_output,
        "tokenizer_inventory": tokenizer_inventory,
        "stable_vocabulary_inventory": stable_inventory,
        "task_token_inventory": task_inventory,
        "capacity_selection_inputs": capacity_inputs,
        "prior_causal_utilization": {
            **config["prior_causal_utilization"],
            "observed_sha256": causal_source_sha256,
            "bytes": int(causal_source.stat().st_size),
        },
        "lens_inputs": lens_inputs,
        "geometry_series": config["geometry_series"],
        "sampling": config["sampling"],
        "operator_metrics": config["operator_metrics"],
        "token_metrics": config["token_metrics"],
        "selection_metrics": config["selection_metrics"],
        "readout_metrics": config["readout_metrics"],
        "selection_margin_boundary": config["selection_margin_boundary"],
        "decision_router": config["decision_router"],
        "model_tensor_contract": config["model_tensor_contract"],
        "models": config["models"],
        "row_manifest": {
            "path": str(row_path), "sha256": file_sha256(row_path),
            "semantic_sha256": row_manifest["semantic_sha256"],
            "counts": row_manifest["counts"],
        },
        "claim_boundary": config["claim_boundary"],
    }
    protocol["semantic_sha256"] = object_sha256(protocol)
    atomic_json(protocol_path, protocol)
    event = create(
        config["protocol_evidence_id"], tier="methods",
        what=("Outcome-blind O3 geometry protocol and exhaustive frozen row "
              "manifest; records unrecoverable margin/causal fields as null."),
        command=("python -m jspace_olmo_lineage.experiments.geometry "
                 f"freeze-protocol --config {config_path}"),
        outputs=[protocol_path, row_path],
        inputs={
            "config_sha256": file_sha256(config_path),
            "provenance_audit": audit_output["sha256"],
            "capacity_selection_union_sha256": capacity_inputs[
                "selection_union_token_ids_sha256"],
        },
        claim_boundary=config["claim_boundary"],
    )
    return {"status": "registered", "event": event,
            "protocol": protocol, "row_manifest": row_manifest}


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    save_safetensors(tensors, str(temporary))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _snapshot_tensor_contract(snapshot: Path, model: Mapping,
                              contract: Mapping) -> tuple[dict, dict]:
    config_path = snapshot / contract["config_name"]
    index_path = snapshot / contract["index_name"]
    if not config_path.is_file() or not index_path.is_file():
        raise RuntimeError("snapshot lacks config/index metadata")
    model_config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    checks = {
        "hidden_size": int(model_config.get("hidden_size", -1))
        == int(contract["expected_hidden_size"]),
        "vocab_size": int(model_config.get("vocab_size", -1))
        == int(contract["expected_vocabulary_size"]),
        "rms_norm_eps": float(model_config.get("rms_norm_eps", -1.0))
        == float(contract["expected_rms_norm_eps"]),
        "tie_word_embeddings": bool(model_config.get(
            "tie_word_embeddings"))
        == bool(contract["expected_tie_word_embeddings"]),
        "unembedding_shard": index["weight_map"].get(
            contract["unembedding_tensor"]) == contract["unembedding_shard"],
        "final_norm_shard": index["weight_map"].get(
            contract["final_norm_tensor"]) == contract["final_norm_shard"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"model tensor contract failed: {checks}")
    files = {}
    for name in (
            contract["config_name"], contract["index_name"],
            contract["unembedding_shard"], contract["final_norm_shard"]):
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError(f"snapshot file absent: {name}")
        files[name] = {
            "path": str(path), "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    return checks, {
        "repository": model["repository"], "revision": model["revision"],
        "snapshot": str(snapshot), "files": files,
        "weight_map_entries": len(index["weight_map"]),
        "index_metadata": index.get("metadata", {}),
        "model_config": {
            "hidden_size": model_config["hidden_size"],
            "vocab_size": model_config["vocab_size"],
            "rms_norm_eps": model_config["rms_norm_eps"],
            "tie_word_embeddings": model_config["tie_word_embeddings"],
            "architectures": model_config.get("architectures"),
        },
    }


def extract_readout(config_path: str | Path, *, slug: str,
                    snapshot: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    model = _model(config, slug)
    existing = _already_registered(model["readout_evidence_id"])
    if existing is not None:
        return {"status": "already-registered-and-verified",
                "evidence_id": existing["evidence_id"]}
    clean = require_clean_tree(expected_branch=config["branch"])
    protocol_event = _verify_event(config["protocol_evidence_id"])
    protocol_output = _registered_output(
        protocol_event, "ol_geometry_protocol_v1.json")
    row_output = _registered_output(
        protocol_event, "ol_geometry_row_manifest_v1.json")
    row_manifest = json.loads(Path(row_output["path"]).read_text())
    token_ids = torch.tensor(
        row_manifest["all_extracted_token_ids"], dtype=torch.int64)
    if object_sha256(token_ids.tolist()) != row_manifest["hashes"][
            "all_extracted_token_ids"]:
        raise RuntimeError("frozen token row manifest drift")

    snapshot = Path(snapshot)
    contract = config["model_tensor_contract"]
    checks, source_manifest = _snapshot_tensor_contract(
        snapshot, model, contract)
    unembedding_path = snapshot / contract["unembedding_shard"]
    norm_path = snapshot / contract["final_norm_shard"]
    with safe_open(str(unembedding_path), framework="pt", device="cpu") as file:
        if contract["unembedding_tensor"] not in file.keys():
            raise RuntimeError("unembedding tensor missing from frozen shard")
        unembedding = file.get_tensor(contract["unembedding_tensor"])
    unembedding_source_dtype = str(unembedding.dtype).replace("torch.", "")
    expected_shape = (
        int(contract["expected_vocabulary_size"]),
        int(contract["expected_hidden_size"]),
    )
    if tuple(unembedding.shape) != expected_shape:
        raise RuntimeError(f"unembedding shape drift: {tuple(unembedding.shape)}")
    extracted = unembedding.index_select(0, token_ids).to(torch.float16)
    del unembedding
    with safe_open(str(norm_path), framework="pt", device="cpu") as file:
        if contract["final_norm_tensor"] not in file.keys():
            raise RuntimeError("final norm tensor missing from frozen shard")
        norm_weight = file.get_tensor(contract["final_norm_tensor"]).float()
    if tuple(norm_weight.shape) != (int(contract["expected_hidden_size"]),):
        raise RuntimeError("final norm shape drift")
    if not bool(torch.isfinite(extracted.float()).all()
                and torch.isfinite(norm_weight).all()):
        raise RuntimeError("non-finite extracted readout tensor")

    output_dir = (metrics_dir(slug) / "geometry_readout"
                  / model["readout_evidence_id"])
    tensor_path = output_dir / "readout_rows.safetensors"
    manifest_path = output_dir / "input_manifest.json"
    _atomic_safetensors(tensor_path, {
        "token_ids": token_ids,
        "lm_head_rows": extracted.contiguous(),
        "final_norm_weight": norm_weight.contiguous(),
    })
    manifest = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": model["readout_evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "protocol": protocol_output,
        "row_manifest": row_output,
        "model": {
            "slug": slug, "role": model["role"],
            "repository": model["repository"],
            "revision": model["revision"],
            "model_uri": model["model_uri"],
        },
        "tensor_contract_checks": checks,
        "source_snapshot": source_manifest,
        "extraction": {
            "token_ids": int(len(token_ids)),
            "token_ids_sha256": object_sha256(token_ids.tolist()),
            "lm_head_rows_shape": list(extracted.shape),
            "lm_head_source_dtype": unembedding_source_dtype,
            "lm_head_storage_dtype": "float16",
            "final_norm_shape": list(norm_weight.shape),
            "final_norm_storage_dtype": "float32",
            "all_finite": True,
        },
        "output": {
            "path": str(tensor_path), "sha256": file_sha256(tensor_path),
            "bytes": int(tensor_path.stat().st_size),
        },
        "claim_boundary": (
            "Hash-pinned methods-only extraction of frozen model tensor rows; "
            "contains no activation, task, intervention, or geometry outcome."),
    }
    manifest["semantic_sha256"] = object_sha256(manifest)
    atomic_json(manifest_path, manifest)
    event = create(
        model["readout_evidence_id"], tier="methods",
        what=(f"Exact-revision frozen unembedding rows and final norm for "
              f"{slug}; methods-only geometry input."),
        command=("python -m jspace_olmo_lineage.experiments.geometry "
                 f"extract-readout --config {config_path} --slug {slug} "
                 f"--snapshot {snapshot}"),
        outputs=[manifest_path, tensor_path],
        inputs={
            "protocol_sha256": protocol_output["sha256"],
            "row_manifest_sha256": row_output["sha256"],
            "model_revision": model["revision"],
            "source_config_sha256": source_manifest["files"][
                contract["config_name"]]["sha256"],
            "source_index_sha256": source_manifest["files"][
                contract["index_name"]]["sha256"],
            "source_unembedding_shard_sha256": source_manifest["files"][
                contract["unembedding_shard"]]["sha256"],
            "source_final_norm_shard_sha256": source_manifest["files"][
                contract["final_norm_shard"]]["sha256"],
        },
        claim_boundary=manifest["claim_boundary"],
    )
    return {"status": "registered", "event": event, "manifest": manifest}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _materialize_lens(model: Mapping) -> Path:
    """Stream one imported lens to local NVMe while verifying its hash."""
    source = resolve_uri(model["lens_uri"])
    destination_dir = local_work() / "geometry_lenses"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (
        f"{model['slug']}-{model['lens_sha256'][:16]}.pt")
    if destination.is_file():
        if file_sha256(destination) == model["lens_sha256"]:
            return destination
        raise RuntimeError(f"local staged lens hash drift: {destination}")
    temporary = destination.with_suffix(destination.suffix + f".tmp{os.getpid()}")
    digest = hashlib.sha256()
    with source.open("rb") as reader, temporary.open("wb") as writer:
        while True:
            block = reader.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    if digest.hexdigest() != model["lens_sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"imported lens hash drift for {model['slug']}")
    os.replace(temporary, destination)
    return destination


def _load_lenses(config: Mapping) -> tuple[dict, list[dict]]:
    expected_layers = [int(value) for value in config["geometry_series"][
        "source_layers"]]
    dimension = int(config["geometry_series"]["d_model"])
    lenses = {}
    manifests = []
    for model in config["models"]:
        path = _materialize_lens(model)
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True)
        if [int(value) for value in checkpoint["source_layers"]] != expected_layers:
            raise RuntimeError(f"lens layer drift for {model['slug']}")
        if int(checkpoint["d_model"]) != dimension:
            raise RuntimeError(f"lens dimension drift for {model['slug']}")
        if set(map(int, checkpoint["J"])) != set(expected_layers):
            raise RuntimeError(f"lens J keys drift for {model['slug']}")
        lenses[model["slug"]] = checkpoint
        manifests.append({
            "slug": model["slug"], "source_uri": model["lens_uri"],
            "local_path": str(path), "sha256": model["lens_sha256"],
            "bytes": int(path.stat().st_size),
        })
    return lenses, manifests


def _load_readouts(config: Mapping, row_manifest: Mapping
                   ) -> tuple[dict, list[dict]]:
    expected_ids = torch.tensor(
        row_manifest["all_extracted_token_ids"], dtype=torch.int64)
    readouts = {}
    inputs = []
    for model in config["models"]:
        event = _verify_event(model["readout_evidence_id"])
        tensor_output = _registered_output(event, "readout_rows.safetensors")
        manifest_output = _registered_output(event, "input_manifest.json")
        tensors = load_safetensors(tensor_output["path"], device="cpu")
        if not torch.equal(tensors["token_ids"], expected_ids):
            raise RuntimeError(f"readout row drift for {model['slug']}")
        if tuple(tensors["lm_head_rows"].shape) != (
                len(expected_ids), int(config["geometry_series"]["d_model"])):
            raise RuntimeError(f"readout tensor shape drift for {model['slug']}")
        manifest = json.loads(Path(manifest_output["path"]).read_text())
        readouts[model["slug"]] = {
            "token_ids": tensors["token_ids"],
            "raw_rows": tensors["lm_head_rows"],
            "norm_weight": tensors["final_norm_weight"],
            "manifest": manifest,
        }
        inputs.append({
            "slug": model["slug"],
            "evidence_id": model["readout_evidence_id"],
            "tensor": tensor_output, "manifest": manifest_output,
        })
    return readouts, inputs


def _indices(row_manifest: Mapping, token_ids: Sequence[int]) -> torch.Tensor:
    lookup = {int(value): index for index, value in enumerate(
        row_manifest["all_extracted_token_ids"])}
    missing = [int(value) for value in token_ids if int(value) not in lookup]
    if missing:
        raise RuntimeError(f"frozen readout rows omit token IDs: {missing[:8]}")
    return torch.tensor([lookup[int(value)] for value in token_ids],
                        dtype=torch.long)


def _effective_rows(readout: Mapping, *, eps: float) -> torch.Tensor:
    gain = readout["norm_weight"].float() / float(np.sqrt(1.0 + eps))
    return readout["raw_rows"].float() * gain[None, :]


def _pair_id(left: str, right: str) -> str:
    return f"{left}__{right}"


def _edge_type(config: Mapping, left: str, right: str) -> str:
    pair = [left, right]
    if pair == list(config["geometry_series"]["primary_edge"]):
        return "primary"
    if pair in [list(value) for value in config["geometry_series"][
            "trajectory_edges"]]:
        return "trajectory"
    if pair == list(config["geometry_series"]["sibling_edge"]):
        return "sibling"
    return "secondary-pairwise"


def _capacity_state(config: Mapping, model: Mapping, layer: int) -> dict:
    event = _verify_event(model["capacity_evidence_id"])
    output = _registered_output(event, f"capacity_layer_{layer}.npz")
    with np.load(output["path"], allow_pickle=False) as data:
        occupancy = occupancy_from_errors(
            data["own_centered_errors"], data["random_centered_errors"],
            persistence=2)
        prefixes = selection_prefixes(
            data["own_centered_selected"], occupancy,
            data["own_centered_achieved"])
        margin = marginal_crossing_margins(
            data["own_centered_errors"], data["random_centered_errors"],
            occupancy)
        owners = data["owners"].copy()
    return {
        "prefixes": prefixes, "occupancy": occupancy,
        "crossing_margin": margin, "owners": owners,
        "input": output,
    }


def _readout_pair_rows(config: Mapping, row_manifest: Mapping,
                       readouts: Mapping) -> list[dict]:
    stable_ids = row_manifest["stable_sample_token_ids"]
    stable_indices = _indices(row_manifest, stable_ids)
    eps = float(config["model_tensor_contract"]["expected_rms_norm_eps"])
    rows = []
    for left_model, right_model in itertools.combinations(config["models"], 2):
        left, right = left_model["slug"], right_model["slug"]
        left_raw = readouts[left]["raw_rows"][stable_indices].float()
        right_raw = readouts[right]["raw_rows"][stable_indices].float()
        left_effective = _effective_rows(readouts[left], eps=eps)[
            stable_indices]
        right_effective = _effective_rows(readouts[right], eps=eps)[
            stable_indices]
        left_plain = torch.nn.functional.normalize(left_effective, dim=1)
        right_plain = torch.nn.functional.normalize(right_effective, dim=1)
        raw_cosine = quantile_summary(
            row_cosines(left_raw, right_raw),
            config["sampling"]["token_quantiles"])
        effective_cosine = quantile_summary(
            row_cosines(left_effective, right_effective),
            config["sampling"]["token_quantiles"])
        plain_neighbors = neighbor_overlap(
            left_plain, right_plain,
            k=int(config["sampling"]["neighbor_k"]))
        left_gain = (readouts[left]["norm_weight"].float()
                     / float(np.sqrt(1.0 + eps)))
        right_gain = (readouts[right]["norm_weight"].float()
                      / float(np.sqrt(1.0 + eps)))
        gain_ratio = right_gain.norm() / left_gain.norm().clamp_min(1e-12)
        rows.append({
            "pair_id": _pair_id(left, right), "left": left, "right": right,
            "edge_type": _edge_type(config, left, right),
            "raw_unembedding_row_cosine_q05": raw_cosine["q05"],
            "raw_unembedding_row_cosine_q50": raw_cosine["q50"],
            "raw_unembedding_row_cosine_q95": raw_cosine["q95"],
            "effective_unembedding_row_cosine_q05": effective_cosine["q05"],
            "effective_unembedding_row_cosine_q50": effective_cosine["q50"],
            "effective_unembedding_row_cosine_q95": effective_cosine["q95"],
            "final_norm_gain_cosine": float(torch.nn.functional.cosine_similarity(
                left_gain, right_gain, dim=0).item()),
            "final_norm_gain_l2_ratio_right_over_left": float(gain_ratio.item()),
            "left_tied_embeddings": bool(left_model.get(
                "tie_word_embeddings", False)),
            "right_tied_embeddings": bool(right_model.get(
                "tie_word_embeddings", False)),
            "plain_neighbor_overlap_fraction": plain_neighbors[
                "overlap_fraction_mean"],
            "plain_neighbor_jaccard": plain_neighbors["jaccard_mean"],
            "n_stable_rows": len(stable_ids),
            "raw_cosine_summary_json": json.dumps(
                raw_cosine, sort_keys=True),
            "effective_cosine_summary_json": json.dumps(
                effective_cosine, sort_keys=True),
            "plain_neighbor_summary_json": json.dumps(
                plain_neighbors, sort_keys=True),
        })
    return rows


def _operator_and_token_rows(config: Mapping, row_manifest: Mapping,
                             readouts: Mapping, lenses: Mapping,
                             readout_rows: Sequence[Mapping]
                             ) -> tuple[list[dict], dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("O3 operator geometry requires CUDA")
    device = torch.device("cuda")
    model_order = [row["slug"] for row in config["models"]]
    model_specs = {row["slug"]: row for row in config["models"]}
    pairs = list(itertools.combinations(model_order, 2))
    readout_by_pair = {row["pair_id"]: row for row in readout_rows}
    analysis_ids = sorted(set(row_manifest["stable_sample_token_ids"])
                          | set(row_manifest["task_union_token_ids"]))
    analysis_indices = _indices(row_manifest, analysis_ids)
    analysis_lookup = {token_id: index for index, token_id in enumerate(
        analysis_ids)}
    stable_positions = torch.tensor([
        analysis_lookup[int(value)]
        for value in row_manifest["stable_sample_token_ids"]],
        dtype=torch.long, device=device)
    cka_positions = torch.tensor([
        analysis_lookup[int(value)] for value in row_manifest["cka_token_ids"]],
        dtype=torch.long, device=device)
    neighbor_positions = torch.tensor([
        analysis_lookup[int(value)]
        for value in row_manifest["neighbor_token_ids"]],
        dtype=torch.long, device=device)
    task_positions = {
        key: torch.tensor([analysis_lookup[int(value)] for value in values],
                          dtype=torch.long, device=device)
        for key, values in row_manifest["task_strata_token_ids"].items()
        if values
    }
    eps = float(config["model_tensor_contract"]["expected_rms_norm_eps"])
    effective_analysis = {
        slug: _effective_rows(readouts[slug], eps=eps)[analysis_indices]
        for slug in model_order
    }
    rows = []
    spectra = {slug: {} for slug in model_order}
    dimension = int(config["geometry_series"]["d_model"])
    base_seed = int(config["sampling"]["base_seed"])
    quantiles = config["sampling"]["token_quantiles"]
    for layer in config["geometry_series"]["source_layers"]:
        layer = int(layer)
        generator = torch.Generator(device="cpu").manual_seed(
            base_seed + 1009 * layer)
        probes = (torch.randint(
            0, 2, (int(config["sampling"]["transport_probes"]), dimension),
            generator=generator, dtype=torch.int8).float().mul_(2).sub_(1)
            / float(np.sqrt(dimension))).to(device)
        omega = torch.randn(
            dimension,
            int(config["sampling"]["spectral_rank"])
            + int(config["sampling"]["spectral_oversample"]),
            generator=generator, dtype=torch.float32).to(device)
        operators = {
            slug: lenses[slug]["J"][layer].to(
                device=device, dtype=torch.float32)
            for slug in model_order
        }
        mapped = {}
        mapped_norms = {}
        for slug in model_order:
            unnormalized = effective_analysis[slug].to(device) @ operators[slug]
            mapped_norms[slug] = unnormalized.norm(dim=1)
            mapped[slug] = torch.nn.functional.normalize(unnormalized, dim=1)
            spectra[slug][str(layer)] = randomized_spectrum(
                operators[slug], omega=omega,
                rank=int(config["sampling"]["spectral_rank"]),
                power_iterations=int(config["sampling"][
                    "spectral_power_iterations"]))
        for left, right in pairs:
            pair_id = _pair_id(left, right)
            operator = operator_pair_metrics(operators[left], operators[right])
            transport = random_transport_metrics(
                operators[left], operators[right], probes, quantiles)
            token_cosine = quantile_summary(
                row_cosines(mapped[left][stable_positions],
                            mapped[right][stable_positions]), quantiles)
            token_cka = centered_linear_cka_gram(
                mapped[left][cka_positions], mapped[right][cka_positions])
            mapped_neighbors = neighbor_overlap(
                mapped[left][neighbor_positions],
                mapped[right][neighbor_positions],
                k=int(config["sampling"]["neighbor_k"]))
            task = {}
            for key, positions in task_positions.items():
                task[key] = quantile_summary(
                    row_cosines(mapped[left][positions],
                                mapped[right][positions]), quantiles)
            row = {
                "pair_id": pair_id, "left": left, "right": right,
                "edge_type": _edge_type(config, left, right), "layer": layer,
                **operator,
                "probe_transport_cosine_q05": transport[
                    "probe_cosine"]["q05"],
                "probe_transport_cosine_q50": transport[
                    "probe_cosine"]["q50"],
                "probe_transport_relative_error_q50": transport[
                    "probe_symmetric_relative_error"]["q50"],
                "mapped_token_cosine_q05": token_cosine["q05"],
                "mapped_token_cosine_q50": token_cosine["q50"],
                "mapped_token_cosine_q95": token_cosine["q95"],
                "mapped_token_centered_linear_cka": token_cka,
                "mapped_neighbor_overlap_fraction": mapped_neighbors[
                    "overlap_fraction_mean"],
                "mapped_neighbor_jaccard": mapped_neighbors["jaccard_mean"],
                "j_minus_plain_neighbor_overlap": (
                    mapped_neighbors["overlap_fraction_mean"]
                    - float(readout_by_pair[pair_id][
                        "plain_neighbor_overlap_fraction"])),
                "left_mapped_row_norm_q50": float(
                    mapped_norms[left].median().item()),
                "right_mapped_row_norm_q50": float(
                    mapped_norms[right].median().item()),
                "task_strata_json": json.dumps(task, sort_keys=True),
                "token_cosine_summary_json": json.dumps(
                    token_cosine, sort_keys=True),
                "transport_summary_json": json.dumps(
                    transport, sort_keys=True),
                "mapped_neighbor_summary_json": json.dumps(
                    mapped_neighbors, sort_keys=True),
                "left_spectrum_json": json.dumps(
                    spectra[left][str(layer)], sort_keys=True),
                "right_spectrum_json": json.dumps(
                    spectra[right][str(layer)], sort_keys=True),
                "left_estimated_stable_rank": spectra[left][str(layer)][
                    "estimated_stable_rank"],
                "right_estimated_stable_rank": spectra[right][str(layer)][
                    "estimated_stable_rank"],
                "n_stable_token_rows": len(
                    row_manifest["stable_sample_token_ids"]),
                "n_cka_rows": len(row_manifest["cka_token_ids"]),
            }
            rows.append(row)
        print(json.dumps({
            "stage": "operator-token", "layer": layer,
            "primary_matrix_cosine": next(
                value["raw_matrix_cosine"] for value in rows
                if value["layer"] == layer
                and value["edge_type"] == "primary"),
        }), flush=True)
        del operators, mapped, mapped_norms, probes, omega
        torch.cuda.empty_cache()
    return rows, spectra


def _selection_rows(config: Mapping, row_manifest: Mapping,
                    readouts: Mapping, lenses: Mapping) -> tuple[list[dict], dict]:
    device = torch.device("cuda")
    model_order = [row["slug"] for row in config["models"]]
    model_specs = {row["slug"]: row for row in config["models"]}
    pairs = list(itertools.combinations(model_order, 2))
    selection_ids = [int(value) for value in row_manifest[
        "selection_union_token_ids"]]
    selection_indices = _indices(row_manifest, selection_ids)
    selection_lookup = {token_id: index for index, token_id in enumerate(
        selection_ids)}
    token_labels = {int(key): value for key, value in row_manifest[
        "token_strings"].items()}
    eps = float(config["model_tensor_contract"]["expected_rms_norm_eps"])
    effective_selection = {
        slug: _effective_rows(readouts[slug], eps=eps)[selection_indices]
        for slug in model_order
    }
    rows = []
    input_rows = {}
    for layer in config["geometry_series"]["assay_layers"]:
        layer = int(layer)
        states = {
            slug: _capacity_state(config, model_specs[slug], layer)
            for slug in model_order
        }
        reference_owners = states[model_order[0]]["owners"]
        if any(not np.array_equal(reference_owners, states[slug]["owners"])
               for slug in model_order[1:]):
            raise RuntimeError("capacity position ownership drift")
        input_rows[str(layer)] = {
            slug: states[slug]["input"] for slug in model_order}
        dictionaries = {}
        for slug in model_order:
            operator = lenses[slug]["J"][layer].to(
                device=device, dtype=torch.float32)
            values = effective_selection[slug].to(device) @ operator
            dictionaries[slug] = torch.nn.functional.normalize(values, dim=1)
            del operator, values
        persistent = {
            slug: persistent_direction_summary(
                states[slug]["prefixes"],
                minimum_fraction=float(config["sampling"][
                    "persistent_direction_minimum_position_fraction"]))
            for slug in model_order
        }
        for left, right in pairs:
            id_metrics = id_selection_metrics(
                states[left]["prefixes"], states[right]["prefixes"],
                rbo_p=float(config["sampling"]["rbo_p"]),
                token_labels=token_labels)
            projector = aggregate_projector_metrics(
                dictionaries[left], dictionaries[right],
                states[left]["prefixes"], states[right]["prefixes"],
                row_id_to_index=selection_lookup,
                relative_tolerance=float(config["sampling"][
                    "projector_rank_relative_tolerance"]),
                batch_positions=int(config["sampling"][
                    "projector_batch_positions"]))
            left_margin = states[left]["crossing_margin"]
            right_margin = states[right]["crossing_margin"]
            left_margin_summary = quantile_summary(
                left_margin, (0.05, 0.25, 0.5, 0.75, 0.95))
            right_margin_summary = quantile_summary(
                right_margin, (0.05, 0.25, 0.5, 0.75, 0.95))
            margin_difference = quantile_summary(
                right_margin - left_margin,
                (0.05, 0.25, 0.5, 0.75, 0.95))
            persistent_jaccard = set_jaccard(
                persistent[left]["persistent_token_ids"],
                persistent[right]["persistent_token_ids"])
            rows.append({
                "pair_id": _pair_id(left, right), "left": left,
                "right": right, "edge_type": _edge_type(
                    config, left, right), "layer": layer,
                "selected_id_jaccard_q05": id_metrics[
                    "selected_id_jaccard"]["q05"],
                "selected_id_jaccard_q50": id_metrics[
                    "selected_id_jaccard"]["q50"],
                "rank_biased_overlap_q50": id_metrics[
                    "rank_biased_overlap"]["q50"],
                "exact_aligned_slot_fraction": id_metrics[
                    "exact_aligned_slot_fraction"],
                "normalized_alias_equivalent_swap_fraction": id_metrics[
                    "normalized_alias_equivalent_swap_fraction"],
                "projector_overlap_q05": projector[
                    "normalized_projector_overlap"]["q05"],
                "projector_overlap_q50": projector[
                    "normalized_projector_overlap"]["q50"],
                "principal_angle_median_degrees_q50": projector[
                    "principal_angle_median_degrees"]["q50"],
                "principal_angle_max_degrees_q95": projector[
                    "principal_angle_max_degrees"]["q95"],
                "left_numerical_rank_q50": projector[
                    "left_numerical_rank"]["q50"],
                "right_numerical_rank_q50": projector[
                    "right_numerical_rank"]["q50"],
                "persistent_direction_jaccard": persistent_jaccard,
                "left_persistent_direction_count": persistent[left][
                    "n_persistent"],
                "right_persistent_direction_count": persistent[right][
                    "n_persistent"],
                "left_crossing_margin_q50": left_margin_summary["q50"],
                "right_crossing_margin_q50": right_margin_summary["q50"],
                "crossing_margin_difference_q50": margin_difference["q50"],
                "exact_kth_kplus1_score_gap": None,
                "protected_span_overlap": None,
                "causal_core_fringe_dose": None,
                "causal_core_fringe_status": "blocked-by-service-gate",
                "id_metrics_json": json.dumps(id_metrics, sort_keys=True),
                "projector_metrics_json": json.dumps(
                    projector, sort_keys=True),
                "left_crossing_margin_json": json.dumps(
                    left_margin_summary, sort_keys=True),
                "right_crossing_margin_json": json.dumps(
                    right_margin_summary, sort_keys=True),
                "crossing_margin_difference_json": json.dumps(
                    margin_difference, sort_keys=True),
                "left_persistent_json": json.dumps(
                    persistent[left], sort_keys=True),
                "right_persistent_json": json.dumps(
                    persistent[right], sort_keys=True),
                "n_positions": len(states[left]["prefixes"]),
            })
        print(json.dumps({
            "stage": "selection", "layer": layer,
            "primary_jaccard_q50": next(
                value["selected_id_jaccard_q50"] for value in rows
                if value["layer"] == layer
                and value["edge_type"] == "primary"),
        }), flush=True)
        del dictionaries, states
        torch.cuda.empty_cache()
    return rows, input_rows


def _aggregate_pair_table(rows: Sequence[Mapping], columns: Sequence[str]
                          ) -> dict:
    result = {}
    pair_ids = sorted({row["pair_id"] for row in rows})
    for pair_id in pair_ids:
        selected = [row for row in rows if row["pair_id"] == pair_id]
        result[pair_id] = {
            column: quantile_summary(
                np.asarray([float(row[column]) for row in selected]),
                (0.05, 0.5, 0.95))
            for column in columns
        }
    return result


def _geometry_router(config: Mapping, layer_rows: Sequence[Mapping],
                     selection_rows: Sequence[Mapping]) -> dict:
    primary = [row for row in layer_rows if row["edge_type"] == "primary"]
    assay = set(map(int, config["geometry_series"]["assay_layers"]))
    broad_operator = (
        float(np.median([row["raw_matrix_cosine"] for row in primary])) >= 0.90
        and min(row["raw_matrix_cosine"] for row in primary
                if row["layer"] in assay) >= 0.85)
    broad_token = (
        float(np.median([row["mapped_token_cosine_q50"]
                         for row in primary])) >= 0.85
        and float(np.median([row["mapped_token_cosine_q05"]
                             for row in primary])) >= 0.70)
    trajectory_30_31 = [
        row for row in layer_rows
        if row["left"] == "olmo3-think" and row["right"] == "olmo31-think"]
    base_movement = 1.0 - float(np.median([
        row["mapped_token_cosine_q50"] for row in primary]))
    later_movement = 1.0 - float(np.median([
        row["mapped_token_cosine_q50"] for row in trajectory_30_31]))
    formation = (
        base_movement >= 1.5 * max(later_movement, 1e-12)
        and base_movement - later_movement >= 0.03)
    primary_selection = [
        row for row in selection_rows if row["edge_type"] == "primary"]
    selection_divergence = (
        float(np.median([row["selected_id_jaccard_q50"]
                         for row in primary_selection])) < 0.75
        or float(np.median([row["projector_overlap_q50"]
                            for row in primary_selection])) < 0.85)
    sibling = [row for row in layer_rows if row["edge_type"] == "sibling"]
    early = set(map(int, config["geometry_series"]["early_layers"]))
    late = set(map(int, config["geometry_series"]["late_layers"]))
    early_movement = float(np.median([
        1.0 - row["mapped_token_cosine_q50"] for row in sibling
        if row["layer"] in early]))
    late_movement = float(np.median([
        1.0 - row["mapped_token_cosine_q50"] for row in sibling
        if row["layer"] in late]))
    instruct_late = late_movement - early_movement >= 0.03
    if broad_operator and broad_token and selection_divergence:
        verdict = "broad-continuity-with-selection-change"
    elif formation:
        verdict = "dictionary-formation-pattern"
    elif not broad_operator and broad_token:
        verdict = "coordinate-drift-with-common-coarse-channel"
    else:
        verdict = "mixed-or-unresolved"
    return {
        "verdict": verdict,
        "broad_operator_continuity": broad_operator,
        "broad_token_continuity": broad_token,
        "selection_divergence_flag": selection_divergence,
        "dictionary_formation_pattern": formation,
        "base_to_3_0_mapped_movement": base_movement,
        "3_0_to_3_1_mapped_movement": later_movement,
        "instruct_late_shift": instruct_late,
        "sibling_early_mapped_movement": early_movement,
        "sibling_late_mapped_movement": late_movement,
        "rules": config["decision_router"],
    }


def aggregate(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    existing = _already_registered(config["joint_evidence_id"])
    if existing is not None:
        return {"status": "already-registered-and-verified",
                "evidence_id": existing["evidence_id"]}
    clean = require_clean_tree(expected_branch=config["branch"])
    protocol_event = _verify_event(config["protocol_evidence_id"])
    protocol_output = _registered_output(
        protocol_event, "ol_geometry_protocol_v1.json")
    row_output = _registered_output(
        protocol_event, "ol_geometry_row_manifest_v1.json")
    row_manifest = json.loads(Path(row_output["path"]).read_text())
    readouts, readout_inputs = _load_readouts(config, row_manifest)
    lenses, lens_inputs = _load_lenses(config)
    readout_rows = _readout_pair_rows(config, row_manifest, readouts)
    layer_rows, spectra = _operator_and_token_rows(
        config, row_manifest, readouts, lenses, readout_rows)
    selection_rows, selection_inputs = _selection_rows(
        config, row_manifest, readouts, lenses)
    router = _geometry_router(config, layer_rows, selection_rows)
    layer_aggregate = _aggregate_pair_table(layer_rows, (
        "raw_matrix_cosine", "symmetric_relative_frobenius_delta",
        "j_minus_identity_cosine", "j_minus_alpha_identity_cosine",
        "probe_transport_cosine_q50", "mapped_token_cosine_q05",
        "mapped_token_cosine_q50", "mapped_token_centered_linear_cka",
        "mapped_neighbor_overlap_fraction"))
    selection_aggregate = _aggregate_pair_table(selection_rows, (
        "selected_id_jaccard_q50", "rank_biased_overlap_q50",
        "projector_overlap_q50", "principal_angle_median_degrees_q50",
        "persistent_direction_jaccard"))
    capacity_joint = _verify_event(config["required_upstream"][
        "capacity_joint"]["evidence_id"])
    capacity_result = _registered_output(
        capacity_joint, "ol-capacity-joint-dev-v1.json")
    result_path = resolve_uri(config["outputs"]["joint_result"],
                              must_exist=False)
    layer_path = resolve_uri(config["outputs"]["layer_table"],
                             must_exist=False)
    selection_path = resolve_uri(config["outputs"]["selection_table"],
                                 must_exist=False)
    readout_path = resolve_uri(config["outputs"]["readout_table"],
                               must_exist=False)
    _atomic_parquet(layer_path, pd.DataFrame(layer_rows))
    _atomic_parquet(selection_path, pd.DataFrame(selection_rows))
    _atomic_parquet(readout_path, pd.DataFrame(readout_rows))
    payload = {
        "study_id": "jspace-olmo-lineage",
        "evidence_id": config["joint_evidence_id"],
        "tier": config["tier"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "protocol": protocol_output,
        "row_manifest": row_output,
        "readout_inputs": readout_inputs,
        "lens_inputs": lens_inputs,
        "capacity_selection_inputs": selection_inputs,
        "capacity_joint_input": capacity_result,
        "prior_causal_utilization": config["prior_causal_utilization"],
        "router": router,
        "pairwise_layer_aggregate": layer_aggregate,
        "pairwise_selection_aggregate": selection_aggregate,
        "readout_pairs": readout_rows,
        "model_spectra": spectra,
        "selection_margin_audit": {
            "exact_kth_kplus1_score_gap": None,
            "exact_gap_status": (
                "not-estimable-from-registered-o2-sufficient-statistics"),
            "available_threshold_margin": (
                "J marginal gain minus random-median gain at registered "
                "per-position crossing K"),
            "protected_span_overlap": None,
            "causal_core_fringe_dose": None,
            "causal_status": "blocked-by-olmo-phase4-service-gate",
            "boundary": config["selection_margin_boundary"],
        },
        "tables": {
            "layers": {"path": str(layer_path),
                       "sha256": file_sha256(layer_path),
                       "rows": len(layer_rows)},
            "selection": {"path": str(selection_path),
                           "sha256": file_sha256(selection_path),
                           "rows": len(selection_rows)},
            "readout": {"path": str(readout_path),
                         "sha256": file_sha256(readout_path),
                         "rows": len(readout_rows)},
        },
        "hypothesis_interpretation": {
            "h1": (
                "Router describes operator/token continuity separately from "
                "registered capacity and selected-span change."),
            "formation": (
                "Formation label requires prospectively larger Base-to-3.0 "
                "mapped-row movement than 3.0-to-3.1 movement."),
            "instruct": (
                "Instruct remains a sibling comparison; late concentration "
                "is not placed on the Think trajectory."),
            "lens_artifact": (
                "All four lenses were exact-same-recipe/corpus and no refit "
                "was justified; this does not prove absence of fit noise."),
        },
        "claim_boundary": config["claim_boundary"],
    }
    payload["payload_sha256"] = object_sha256(payload)
    atomic_json(result_path, payload)
    event = create(
        config["joint_evidence_id"], tier=config["tier"],
        what=("Four-checkpoint same-corpus OLMo operator, token, readout, and "
              "registered sparse-selection geometry trajectory."),
        command=("python -m jspace_olmo_lineage.experiments.geometry "
                 f"aggregate --config {config_path}"),
        outputs=[result_path, layer_path, selection_path, readout_path],
        inputs={
            "protocol_sha256": protocol_output["sha256"],
            "row_manifest_sha256": row_output["sha256"],
            "capacity_joint_sha256": capacity_result["sha256"],
            "readout_evidence_ids": [
                row["evidence_id"] for row in readout_inputs],
            "lens_sha256": {
                row["slug"]: row["sha256"] for row in lens_inputs},
        },
        verdict=router["verdict"],
        claim_boundary=config["claim_boundary"],
    )
    return {"status": "registered", "event": event, "result": payload}


def _save_figure(figure, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def figures(
    config_path: str | Path,
    *,
    reconstruction_output_dir: str | Path | None = None,
) -> dict:
    """Render the five O3 deliverables strictly from registered tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config_path = Path(config_path)
    config = load_config(config_path)
    if reconstruction_output_dir is None:
        existing = _already_registered(config["figure_evidence_id"])
        if existing is not None:
            return {"status": "already-registered-and-verified",
                    "evidence_id": existing["evidence_id"]}
    clean = require_clean_tree(expected_branch=config["branch"])
    geometry_event = _verify_event(config["joint_evidence_id"])
    layer_output = _registered_output(
        geometry_event, "ol-geometry-joint-dev-v1_layers.parquet")
    selection_output = _registered_output(
        geometry_event, "ol-geometry-joint-dev-v1_selection.parquet")
    readout_output = _registered_output(
        geometry_event, "ol-geometry-joint-dev-v1_readout.parquet")
    result_output = _registered_output(
        geometry_event, "ol-geometry-joint-dev-v1.json")
    capacity_event = _verify_event(config["required_upstream"][
        "capacity_joint"]["evidence_id"])
    capacity_output = _registered_output(
        capacity_event, "ol-capacity-joint-dev-v1.parquet")
    causal_path = Path(config["prior_causal_utilization"]["table_path"])
    if file_sha256(causal_path) != config["prior_causal_utilization"][
            "table_sha256"]:
        raise RuntimeError("prior causal-utilization table hash drift")

    layer = pd.read_parquet(layer_output["path"])
    selection = pd.read_parquet(selection_output["path"])
    readout = pd.read_parquet(readout_output["path"])
    capacity = pd.read_parquet(capacity_output["path"])
    causal = pd.read_csv(causal_path)
    output_dir = (
        figures_dir() if reconstruction_output_dir is None
        else Path(reconstruction_output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    labels = {
        "olmo3-base": "Base", "olmo3-think": "3.0 Think",
        "olmo31-think": "3.1 Think",
        "olmo31-instruct": "3.1 Instruct",
    }
    edge_labels = {
        "olmo3-base__olmo3-think": "Base → 3.0 Think",
        "olmo3-think__olmo31-think": "3.0 → 3.1 Think",
        "olmo31-think__olmo31-instruct": "3.1 Think ↔ Instruct",
    }
    key_pairs = list(edge_labels)

    # 1. All-layer operator similarity heatmap.
    pivot = (layer.pivot(index="pair_id", columns="layer",
                         values="raw_matrix_cosine").loc[
                             sorted(layer["pair_id"].unique())])
    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis",
                      vmin=float(pivot.min().min()), vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    ax.set_yticks(range(len(pivot.index)), [
        value.replace("__", " ↔ ") for value in pivot.index])
    ax.set_xlabel("Source layer")
    ax.set_title("OLMo same-corpus J-operator cosine")
    fig.colorbar(image, ax=ax, label="raw matrix cosine")
    outputs.extend(_save_figure(
        fig, output_dir / "olf01_operator_similarity_heatmap"))
    plt.close(fig)

    # 2. Assay/trajectory token-row similarity.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for pair_id in key_pairs:
        values = layer[layer["pair_id"] == pair_id].sort_values("layer")
        ax.plot(values["layer"], values["mapped_token_cosine_q50"],
                marker="o", ms=3, label=edge_labels[pair_id])
        ax.fill_between(values["layer"], values["mapped_token_cosine_q05"],
                        values["mapped_token_cosine_q95"], alpha=0.12)
    for assay_layer in config["geometry_series"]["assay_layers"]:
        ax.axvline(assay_layer, color="0.8", lw=0.7, zorder=0)
    ax.set_xlabel("Source layer")
    ax.set_ylabel("mapped token-row cosine")
    ax.set_title("Stable sampled-token geometry across lineage edges")
    ax.legend(frameon=False)
    ax.set_ylim(-0.05, 1.02)
    outputs.extend(_save_figure(
        fig, output_dir / "olf02_token_row_similarity"))
    plt.close(fig)

    # 3. Selected-ID and span overlap trajectory at capacity layers.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for pair_id in key_pairs:
        values = selection[selection["pair_id"] == pair_id].sort_values(
            "layer")
        axes[0].plot(values["layer"], values["selected_id_jaccard_q50"],
                     marker="o", label=edge_labels[pair_id])
        axes[1].plot(values["layer"], values["projector_overlap_q50"],
                     marker="o", label=edge_labels[pair_id])
    axes[0].axhline(0.75, color="0.45", ls="--", lw=1,
                    label="descriptive ID flag")
    axes[1].axhline(0.85, color="0.45", ls="--", lw=1,
                    label="descriptive span flag")
    axes[0].set_ylabel("median selected-ID Jaccard")
    axes[1].set_ylabel("median projector overlap")
    for ax in axes:
        ax.set_xlabel("Source layer")
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sparse selection and selected-span continuity")
    outputs.extend(_save_figure(
        fig, output_dir / "olf03_selected_span_trajectory"))
    plt.close(fig)

    # 4. Capacity versus already-imported Bank-S causal utilization.  This is
    # deliberately not Bank W: the Bank-W causal axis remains gate-blocked.
    own_capacity = capacity[
        (capacity["row_type"] == "model_estimate")
        & (capacity["frame"] == "own")].copy()
    own_capacity["layer"] = own_capacity["layer"].astype(int)
    capacity_mean = own_capacity.groupby("left", as_index=False).agg(
        centered_excess=("centered_excess", "mean"))
    causal_filter = config["prior_causal_utilization"]["estimand_filter"]
    causal_selected = causal[
        (causal["frame"] == causal_filter["frame"])
        & (causal["metric_key"] == causal_filter["metric_key"])]
    state = capacity_mean.merge(
        causal_selected[["checkpoint_key", "estimate", "ci95_low",
                         "ci95_high", "lineage_role"]],
        left_on="left", right_on="checkpoint_key", validate="one_to_one")
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    trajectory = state[state["left"].isin([
        "olmo3-base", "olmo3-think", "olmo31-think"])].set_index("left").loc[
            ["olmo3-base", "olmo3-think", "olmo31-think"]]
    ax.plot(trajectory["centered_excess"], trajectory["estimate"],
            color="0.45", lw=1.2, zorder=1)
    for _, value in state.iterrows():
        sibling = value["left"] == "olmo31-instruct"
        ax.errorbar(value["centered_excess"], value["estimate"],
                    yerr=[[value["estimate"] - value["ci95_low"]],
                          [value["ci95_high"] - value["estimate"]]],
                    fmt="D" if sibling else "o", capsize=3,
                    label=labels[value["left"]], zorder=2)
        ax.annotate(labels[value["left"]],
                    (value["centered_excess"], value["estimate"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="0.75", lw=0.8)
    ax.axvline(0, color="0.75", lw=0.8)
    ax.set_xlabel("mean own-frame centered excess capacity (L24/32/40)")
    ax.set_ylabel("Bank-S composition-specific causal effect (nats)")
    ax.set_title("Capacity versus known causal utilization\n"
                 "Bank W remains blocked (common support 16 < 20)")
    outputs.extend(_save_figure(
        fig, output_dir / "olf04_capacity_causal_state_space"))
    plt.close(fig)

    # 5. Readout versus transport movement decomposition.
    readout_lookup = readout.set_index("pair_id")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, pair_id in zip(axes, key_pairs):
        values = layer[layer["pair_id"] == pair_id].sort_values("layer")
        plain = 1.0 - float(readout_lookup.loc[
            pair_id, "effective_unembedding_row_cosine_q50"])
        ax.plot(values["layer"], 1.0 - values["raw_matrix_cosine"],
                label="operator", marker="o", ms=3)
        ax.plot(values["layer"], 1.0 - values["mapped_token_cosine_q50"],
                label="J-mapped rows", marker="s", ms=3)
        ax.axhline(plain, label="unembedding + norm", color="tab:green",
                   ls="--")
        ax.set_title(edge_labels[pair_id])
        ax.set_xlabel("Source layer")
    axes[0].set_ylabel("movement (1 - cosine)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Transport, readout, and mapped-row movement decomposition")
    outputs.extend(_save_figure(
        fig, output_dir / "olf05_readout_transport_decomposition"))
    plt.close(fig)

    manifest_path = (
        manifests_dir() / "ol_geometry_figures_v1.json"
        if reconstruction_output_dir is None
        else output_dir / "ol_geometry_figures_reconstruction.json")
    manifest = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": config["figure_evidence_id"],
        "code_commit": clean["code_commit"],
        "source_geometry_evidence_id": config["joint_evidence_id"],
        "source_outputs": {
            "result": result_output, "layers": layer_output,
            "selection": selection_output, "readout": readout_output,
            "capacity": capacity_output,
            "prior_causal_utilization": {
                "path": str(causal_path),
                "sha256": config["prior_causal_utilization"]["table_sha256"],
                "source_evidence_id": config["prior_causal_utilization"][
                    "source_evidence_id"],
            },
        },
        "figures": [{
            "path": str(path), "sha256": file_sha256(path),
            "bytes": int(path.stat().st_size),
        } for path in outputs],
        "important_boundary": (
            "Figure 4 uses the foundation-imported development Bank-S causal "
            "trajectory. It does not depict a Bank-W or O4 result; O4 remains "
            "blocked. Exact kth/k+1 and core/fringe causal dose remain null."),
        "claim_boundary": config["claim_boundary"],
    }
    manifest["semantic_sha256"] = object_sha256(manifest)
    atomic_json(manifest_path, manifest)
    if reconstruction_output_dir is not None:
        return {
            "status": "reconstructed-without-registry-mutation",
            "manifest_path": str(manifest_path),
            "manifest": manifest,
            "outputs": [str(path) for path in outputs],
        }
    event = create(
        config["figure_evidence_id"], tier=config["tier"],
        what=("Five O3 geometry-trajectory figures rendered exclusively from "
              "registered OLMo tables and foundation-imported Bank-S data."),
        command=("python -m jspace_olmo_lineage.experiments.geometry "
                 f"figures --config {config_path}"),
        outputs=[manifest_path, *outputs],
        inputs={
            "geometry_result_sha256": result_output["sha256"],
            "geometry_layer_table_sha256": layer_output["sha256"],
            "geometry_selection_table_sha256": selection_output["sha256"],
            "geometry_readout_table_sha256": readout_output["sha256"],
            "capacity_table_sha256": capacity_output["sha256"],
            "prior_causal_table_sha256": config[
                "prior_causal_utilization"]["table_sha256"],
        },
        claim_boundary=config["claim_boundary"],
    )
    return {"status": "registered", "event": event,
            "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "freeze-protocol", "extract-readout", "aggregate", "figures"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--slug")
    parser.add_argument("--snapshot")
    parser.add_argument("--reconstruction-output-dir")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.command == "freeze-protocol":
        result = freeze_protocol(arguments.config)
    elif arguments.command == "extract-readout":
        if not arguments.slug or not arguments.snapshot:
            raise SystemExit("extract-readout requires --slug and --snapshot")
        result = extract_readout(
            arguments.config, slug=arguments.slug,
            snapshot=arguments.snapshot)
    elif arguments.command == "aggregate":
        result = aggregate(arguments.config)
    else:
        result = figures(
            arguments.config,
            reconstruction_output_dir=arguments.reconstruction_output_dir)
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
