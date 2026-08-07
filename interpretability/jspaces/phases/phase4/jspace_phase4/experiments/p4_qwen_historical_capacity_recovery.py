"""Hash-gated exact-byte recovery for a missing historical capacity artifact.

This is deliberately narrower than the functional producer.  It consumes the
registered fixed activation cache, registered A120 lens, and the exact
capacity algorithm used at the source commit.  It never runs a language-model
forward pass.  The candidate replaces the absent registered file only when
its complete SHA-256 equals the immutable registry pin.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path

def _find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError("cannot locate git repository root")

from types import SimpleNamespace
from typing import Mapping

import torch
import yaml
from safetensors import safe_open

from jspace_part2.dictionaries import build_j_dictionaries

from ..gpu import require_cuda_gpu
from ..manifests import file_sha256, require_clean_tree
from ..paths4 import local_work, materialize_local_file, resolve_uri
from ..registry4 import resolve
from .p4_qwen_lens_structural_stability import load_lens_checkpoint
from .p4_qwen_multilens_functional_gate import _capacity_layer


REPO_ROOT = _find_repo_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--recover", action="store_true")
    return parser.parse_args()


def function_ast_sha256(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one function {name!r} in {path}")
    payload = ast.dump(matches[0], include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _event_output(event: Mapping, path: Path) -> dict:
    matches = [
        row for row in event["outputs"]
        if Path(row["path"]) == path
    ]
    if len(matches) != 1:
        raise RuntimeError(f"registered event does not pin exactly one {path}")
    return matches[0]


def _verify_source_contract(config: Mapping) -> dict:
    source = config["source_event"]
    event = resolve(str(source["evidence_id"]))
    if not event["live"]:
        raise RuntimeError("historical functional event is not live")
    if event["code_commit"] != source["code_commit"]:
        raise RuntimeError("historical functional code commit drift")
    source_config = REPO_ROOT / source["config"]
    if file_sha256(source_config) != source["config_sha256"]:
        raise RuntimeError("historical functional config no longer matches")
    manifest_path = Path(source["input_manifest"])
    if file_sha256(manifest_path) != source["input_manifest_sha256"]:
        raise RuntimeError("historical functional input manifest hash drift")
    manifest_pin = _event_output(event, manifest_path)
    if manifest_pin["sha256"] != source["input_manifest_sha256"]:
        raise RuntimeError("historical event input-manifest pin drift")
    cache = config["fixed_capacity_input"]
    cache_path = Path(cache["path"])
    if file_sha256(cache_path) != cache["sha256"]:
        raise RuntimeError("fixed capacity activation hash drift")
    if _event_output(event, cache_path)["sha256"] != cache["sha256"]:
        raise RuntimeError("fixed capacity input is absent from source event")
    target = config["target"]
    target_path = Path(target["path"])
    if _event_output(event, target_path)["sha256"] \
            != target["expected_sha256"]:
        raise RuntimeError("historical target registry pin drift")
    lens = config["lens"]
    lens_event = resolve(str(lens["evidence_id"]))
    if not lens_event["live"]:
        raise RuntimeError("A120 lens event is not live")
    lens_path = resolve_uri(lens["uri"])
    if file_sha256(lens_path) != lens["sha256"]:
        raise RuntimeError("A120 lens hash drift")
    registered_lens_hashes = {
        row["sha256"] for row in lens_event["outputs"]
        if Path(row["path"]) == lens_path
    }
    if registered_lens_hashes != {lens["sha256"]}:
        raise RuntimeError("A120 lens is absent from its registered event")
    manifest = json.loads(manifest_path.read_text())
    payload = manifest.get("payload", {})
    if payload.get("code_commit") != source["code_commit"]:
        raise RuntimeError("input manifest source commit drift")
    if payload.get("capacity") != yaml.safe_load(
            source_config.read_text())["capacity"]:
        raise RuntimeError("input manifest capacity contract drift")
    return {
        "source_event": event,
        "source_config_path": source_config,
        "source_config": yaml.safe_load(source_config.read_text()),
        "input_manifest": manifest,
        "lens_source_path": lens_path,
    }


def _verify_algorithm_contract(config: Mapping) -> dict:
    contract = config["algorithm_contract"]
    files = {}
    for relative, expected in contract["exact_files"].items():
        path = REPO_ROOT / relative
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"capacity dependency drift: {relative}")
        files[relative] = actual
    module = REPO_ROOT / contract["functional_module"]
    functions = {}
    for name, expected in contract["function_ast_sha256"].items():
        actual = function_ast_sha256(module, name)
        if actual != expected:
            raise RuntimeError(f"capacity function AST drift: {name}")
        functions[name] = actual
    return {"exact_files": files, "function_ast_sha256": functions}


def _verify_runtime(config: Mapping, *, require_free_gpu: bool) -> dict:
    import transformers

    runtime = config["runtime"]
    if torch.__version__ != runtime["torch"]:
        raise RuntimeError("historical recovery torch version drift")
    if transformers.__version__ != runtime["transformers"]:
        raise RuntimeError("historical recovery transformers version drift")
    gpu = require_cuda_gpu()
    if gpu["name"] != runtime["expected_gpu_name"]:
        raise RuntimeError("historical recovery GPU model drift")
    if gpu["capability"] != runtime["expected_cuda_capability"]:
        raise RuntimeError("historical recovery GPU capability drift")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    ready = int(free_bytes) >= int(runtime["minimum_free_gpu_bytes"])
    if require_free_gpu and not ready:
        raise RuntimeError(
            "historical recovery requires an idle GPU after A1000; "
            f"free={free_bytes}, required={runtime['minimum_free_gpu_bytes']}")
    return {
        "gpu": gpu,
        "free_gpu_bytes": int(free_bytes),
        "total_gpu_bytes": int(total_bytes),
        "free_gpu_gate_pass": ready,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _verify_model_files(config: Mapping) -> tuple[Path, dict]:
    model = config["model"]
    snapshot = resolve_uri(model["uri"])
    manifest_path = REPO_ROOT / model["snapshot_manifest"]
    if file_sha256(manifest_path) != model["snapshot_manifest_sha256"]:
        raise RuntimeError("Qwen snapshot manifest hash drift")
    checks = {
        "model.safetensors.index.json": model["index_sha256"],
        model["lm_head_shard"]: model["lm_head_shard_sha256"],
        model["final_norm_shard"]: model["final_norm_shard_sha256"],
    }
    verified = {}
    for relative, expected in checks.items():
        path = snapshot / relative
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"Qwen recovery model file drift: {relative}")
        verified[relative] = actual
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    for tensor_key, shard_key in (
            ("lm_head_tensor", "lm_head_shard"),
            ("final_norm_tensor", "final_norm_shard")):
        if index["weight_map"].get(model[tensor_key]) != model[shard_key]:
            raise RuntimeError(f"Qwen tensor-to-shard mapping drift: {tensor_key}")
    return snapshot, verified


def preflight(config: Mapping, *, require_free_gpu: bool = False) -> dict:
    source = _verify_source_contract(config)
    algorithms = _verify_algorithm_contract(config)
    runtime = _verify_runtime(config, require_free_gpu=require_free_gpu)
    _snapshot, model_files = _verify_model_files(config)
    target = Path(config["target"]["path"])
    target_status = "missing"
    if target.exists():
        actual = file_sha256(target)
        if actual != config["target"]["expected_sha256"]:
            raise RuntimeError("historical recovery target exists with wrong hash")
        target_status = "already-restored-and-verified"
    return {
        "recovery_id": config["recovery_id"],
        "source_evidence_id": config["source_event"]["evidence_id"],
        "source_event_live": source["source_event"]["live"],
        "source_code_commit": source["source_event"]["code_commit"],
        "algorithm_contract": algorithms,
        "model_files": model_files,
        "runtime": runtime,
        "target": str(target),
        "target_status": target_status,
        "recovery_ready": bool(
            target_status == "missing" and runtime["free_gpu_gate_pass"]),
        "claim_boundary": config["claim_boundary"],
    }


class _MinimalQwenDictionaryModel:
    def __init__(self, lm_head: torch.Tensor, norm: torch.nn.Module):
        self._output = SimpleNamespace(weight=lm_head)
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(norm=norm))

    def get_output_embeddings(self):
        return self._output


def _load_dictionary_model(snapshot: Path, config: Mapping):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    model = config["model"]
    source_config = yaml.safe_load(
        (REPO_ROOT / config["source_event"]["config"]).read_text())
    d_model = int(source_config["runtime"]["expected_d_model"])
    hf_config = json.loads((snapshot / "config.json").read_text())
    text_config = hf_config["text_config"]
    if int(text_config["hidden_size"]) != d_model:
        raise RuntimeError("Qwen hidden-size drift during recovery")
    epsilon = float(text_config["rms_norm_eps"])
    with safe_open(
            snapshot / model["lm_head_shard"], framework="pt",
            device="cpu") as handle:
        lm_head_cpu = handle.get_tensor(model["lm_head_tensor"])
    with safe_open(
            snapshot / model["final_norm_shard"], framework="pt",
            device="cpu") as handle:
        norm_weight = handle.get_tensor(model["final_norm_tensor"])
    if list(lm_head_cpu.shape) != [
            int(source_config["runtime"]["expected_vocab_size"]), d_model]:
        raise RuntimeError("Qwen lm-head tensor shape drift")
    if lm_head_cpu.dtype != torch.bfloat16:
        raise RuntimeError("Qwen lm-head dtype drift")
    norm = Qwen3_5RMSNorm(d_model, eps=epsilon)
    with torch.no_grad():
        norm.weight.copy_(norm_weight.float())
    norm = norm.to("cuda").eval()
    lm_head = lm_head_cpu.to("cuda")
    del lm_head_cpu, norm_weight
    return _MinimalQwenDictionaryModel(lm_head, norm)


@torch.no_grad()
def recover(config: Mapping) -> dict:
    if config["target"]["replace_only_on_exact_hash_match"] is not True \
            or config["target"]["never_register_a_partial_or_near_match"] \
            is not True:
        raise RuntimeError("historical recovery target safety gates drift")
    if config["runtime"]["require_clean_tree"] is True:
        require_clean_tree()
    before = preflight(config, require_free_gpu=True)
    if before["target_status"] == "already-restored-and-verified":
        return {**before, "status": "already-restored-and-verified"}
    source = _verify_source_contract(config)
    source_config = source["source_config"]
    snapshot, _model_files = _verify_model_files(config)
    lens_specification = {
        "n_prompts": int(config["lens"]["n_prompts"]),
    }
    lens_path = materialize_local_file(
        config["lens"]["uri"], expected_sha256=config["lens"]["sha256"])
    checkpoint = load_lens_checkpoint(
        lens_path, lens_specification, source_config["runtime"])
    minimal_model = _load_dictionary_model(snapshot, config)
    layers = [int(value) for value in
              config["algorithm_contract"]["layer_order"]]
    capacity = source_config["capacity"]
    if layers != [int(value) for value in capacity["layers"]]:
        raise RuntimeError("historical recovery capacity layer order drift")
    lens = SimpleNamespace(jacobians=checkpoint["J"])
    dictionaries = build_j_dictionaries(
        minimal_model, lens, layers, dtype=torch.float16)
    cache = torch.load(
        config["fixed_capacity_input"]["path"], map_location="cpu",
        weights_only=True, mmap=True)
    owners = list(cache["owner"])
    if not owners:
        raise RuntimeError("historical fixed capacity cache is empty")
    first = dictionaries[layers[0]]
    vocab_size, d_model = first.shape
    random_dictionaries = []
    for seed in capacity["random_seeds"]:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        cpu = torch.randn(
            (vocab_size, d_model), generator=generator,
            dtype=torch.float32)
        cpu = torch.nn.functional.normalize(cpu, dim=1)
        random_dictionaries.append(cpu.to("cuda", torch.float16))
        del cpu
    payload = {"schema_version": 1, "lens": config["lens"]["name"],
               "layers": {}}
    metric_hashes = {}
    for layer in layers:
        h = torch.stack(cache["H"][str(layer)])
        metrics, reconstructions = _capacity_layer(
            h, dictionaries[layer], random_dictionaries,
            owners=owners, k_max=int(capacity["k_max"]),
            persistence=int(capacity["persistence"]),
            persistence_sensitivity=[int(value) for value in
                                     capacity["persistence_sensitivity"]],
            bootstrap_draws=int(capacity["bootstrap_draws"]),
            bootstrap_seed=int(capacity["bootstrap_seed"]))
        payload["layers"][str(layer)] = reconstructions
        metric_hashes[str(layer)] = hashlib.sha256(json.dumps(
            metrics, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    target = Path(config["target"]["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = target.with_suffix(target.suffix + f".tmp{os.getpid()}")
    if candidate.exists():
        raise RuntimeError(f"historical recovery candidate already exists: {candidate}")
    torch.save(payload, candidate)
    actual = file_sha256(candidate)
    expected = config["target"]["expected_sha256"]
    if actual != expected:
        quarantine = local_work() / "historical_recovery_quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        destination = quarantine / (
            f"{config['recovery_id']}-{actual}.pt")
        os.replace(candidate, destination)
        raise RuntimeError(
            "historical recovery candidate hash mismatch; candidate was "
            f"quarantined at {destination}, actual={actual}, expected={expected}")
    os.replace(candidate, target)
    if file_sha256(target) != expected:
        raise RuntimeError("historical recovery destination failed post-move hash")
    del payload, random_dictionaries, dictionaries, checkpoint, lens
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "status": "exact-registered-bytes-restored",
        "recovery_id": config["recovery_id"],
        "source_evidence_id": config["source_event"]["evidence_id"],
        "target": str(target),
        "sha256": expected,
        "metric_hashes_not_registered": metric_hashes,
        "scientific_result_created": False,
        "registry_changed": False,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    arguments = parse_args()
    config = yaml.safe_load(Path(arguments.config).read_text())
    result = preflight(config) if arguments.preflight else recover(config)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
