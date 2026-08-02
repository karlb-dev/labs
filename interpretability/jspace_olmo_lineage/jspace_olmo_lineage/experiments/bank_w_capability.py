"""OLMo-only Bank-W baseline capability gates and Phase 4 import bundle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch
import yaml
from jspace_phase4.state import StateHeader, StateStore

from ..compat import (
    DEFAULT_SPEC,
    ScoringSession,
    aggregate_model_payloads,
    analyze_model_rows,
    candidate_scores,
    select_development_rows,
    verify_sources,
)
from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..imports import resolve_source_event
from ..manifests import (
    InputManifest,
    atomic_json,
    atomic_text,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths import metrics_dir, resolve_uri
from ..provenance import Provenance, write_result
from ..registry import EVENTS, RegistryError, create, read_events, resolve


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("Bank-W capability config must be a mapping")
    return value


def _model_reference(uri: str) -> dict:
    if not uri.startswith("model://") or "@" not in uri:
        raise ValueError("model URI must pin an exact revision")
    model_id, revision = uri[len("model://"):].rsplit("@", 1)
    if len(revision) != 40:
        raise ValueError("model revision must be a full 40-character commit")
    return {"model_id": model_id, "revision": revision}


def _tokenizer_source_hash(model_path: Path) -> str:
    names = (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "chat_template.jinja",
    )
    files = {
        name: file_sha256(model_path / name)
        for name in names if (model_path / name).is_file()
    }
    if not files:
        raise RuntimeError("pinned model snapshot has no tokenizer files")
    return object_sha256(files)


def _verify_packages(expected: Mapping[str, str]) -> dict:
    observed = {}
    for package, version in expected.items():
        actual = importlib.metadata.version(package)
        observed[package] = actual
        if actual != str(version):
            raise RuntimeError(
                f"runtime package drift: {package} expected {version}, "
                f"got {actual}")
    return observed


def _source_registry(config: Mapping) -> tuple[list[dict], dict]:
    specification = config["source_phase4"]
    path = resolve_uri(specification["registry_uri"])
    raw = path.read_bytes()
    expected = specification["registry_sha256"]
    actual = hashlib.sha256(raw).hexdigest()
    prefix_bytes = len(raw)
    if actual != expected:
        digest = hashlib.sha256()
        prefix_bytes = 0
        for line in raw.splitlines(keepends=True):
            digest.update(line)
            prefix_bytes += len(line)
            if digest.hexdigest() == expected:
                break
        else:
            raise RuntimeError("frozen Phase 4 source registry hash drift")
    prefix = raw[:prefix_bytes]
    events = [
        json.loads(line) for line in prefix.decode("utf-8").splitlines()
        if line.strip()
    ]
    appended = raw[prefix_bytes:]
    if appended:
        try:
            appended_events = [
                json.loads(line)
                for line in appended.decode("utf-8").splitlines()
                if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Phase 4 registry extension is not valid append-only JSONL"
            ) from error
    else:
        appended_events = []
    return events, {
        "path": str(path),
        "sha256": expected,
        "bytes": prefix_bytes,
        "current_sha256": actual,
        "current_bytes": len(raw),
        "append_only_extension": bool(appended),
        "appended_events": len(appended_events),
    }


def _source_event(config: Mapping, evidence_id: str) -> dict:
    events, registry = _source_registry(config)
    event = resolve_source_event(events, evidence_id)
    event["verified_registry"] = registry
    return event


def _verify_bank_inputs(config: Mapping) -> tuple[Path, dict]:
    paths = {
        "bank": resolve_uri(config["bank_uri"]),
        "audit": resolve_uri(config["audit_uri"]),
        "partition": resolve_uri(config["partition_uri"]),
    }
    expected = {
        "bank": config["bank_file_sha256"],
        "audit": config["audit_file_sha256"],
        "partition": config["partition_file_sha256"],
    }
    hashes = {}
    for name, path in paths.items():
        actual = file_sha256(path)
        if actual != expected[name]:
            raise RuntimeError(f"Bank-W {name} hash drift")
        hashes[name] = actual
    audit = json.loads(paths["audit"].read_text())
    partition = json.loads(paths["partition"].read_text())
    checks = {
        "audit_evidence": audit.get("evidence_id") == config[
            "bank_evidence_id"],
        "bank_rows": audit.get("bank_rows_sha256") == config[
            "bank_rows_sha256"],
        "audit_partition": audit.get("partition_sha256") == config[
            "partition_payload_sha256"],
        "partition_payload": partition.get("payload_sha256") == config[
            "partition_payload_sha256"],
        "all_axes_crossed": bool(audit.get("all_axes_fully_crossed")),
        "shortcut_audit": bool(audit.get(
            "shortcut_audit", {}).get("all_pass")),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Bank-W structural gate failed: {checks}")
    return paths["bank"], {
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": hashes,
        "checks": checks,
    }


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
            raise RegistryError(
                f"registered output failed verification: {path}")
    return event


def _write_manifest_once(path: Path, envelope: Mapping) -> None:
    if path.exists():
        observed = json.loads(path.read_text())
        if observed != envelope:
            raise RuntimeError(
                "existing input manifest differs; refusing mixed-state resume")
        return
    atomic_json(path, envelope)


def _critical_source_conformance(config: Mapping) -> dict:
    source_spec = config["source_phase4"]
    source_config_path = resolve_uri(source_spec["config_uri"])
    if file_sha256(source_config_path) != source_spec["config_sha256"]:
        raise RuntimeError("Phase 4 Bank-W source config hash drift")
    source_config = yaml.safe_load(source_config_path.read_text())
    source_protocol_path = resolve_uri(source_spec["protocol_uri"])
    if file_sha256(source_protocol_path) != source_spec["protocol_sha256"]:
        raise RuntimeError("Phase 4 Bank-W source protocol hash drift")
    source_protocol = json.loads(source_protocol_path.read_text())
    protocol_event = _source_event(
        config, source_spec["protocol_evidence_id"])
    protocol_outputs = protocol_event.get("outputs", [])
    if not any(
            row.get("sha256") == source_spec["protocol_sha256"]
            for row in protocol_outputs):
        raise RuntimeError("Phase 4 protocol event does not pin the file")

    comparisons = {}
    for key in ("selection", "answer_contract", "capability_guard"):
        comparisons[key] = {
            "source_sha256": object_sha256(source_config[key]),
            "side_sha256": object_sha256(config[key]),
            "exact": source_config[key] == config[key],
        }
    source_models = {row["slug"]: row for row in source_config["models"]}
    model_checks = {}
    for row in [*config["models"], *config["reference_models"]]:
        source = source_models[row["slug"]]
        fields = (
            "model_uri", "expected_tokenizer_class",
            "expected_answer_token_ids",
        )
        model_checks[row["slug"]] = all(
            row[field] == source[field] for field in fields)
    checks = {
        "source_protocol_all_gates_pass": bool(
            source_protocol.get("all_protocol_gates_pass")),
        "source_protocol_event_live": bool(protocol_event["live"]),
        "critical_config_exact": all(
            row["exact"] for row in comparisons.values()),
        "model_contracts_exact": all(model_checks.values()),
        "model_order_exact": config["joint_model_order"] == [
            row["slug"] for row in source_config["models"]],
    }
    return {
        "checks": checks,
        "comparisons": comparisons,
        "model_checks": model_checks,
        "source_protocol": {
            "evidence_id": source_spec["protocol_evidence_id"],
            "sha256": source_spec["protocol_sha256"],
            "source_commit": protocol_event.get("code_commit"),
        },
        "source_config_sha256": source_spec["config_sha256"],
        "all_pass": all(checks.values()),
    }


def freeze_protocol(config_path: Path, config: Mapping) -> dict:
    existing = _registered_output_check(config["protocol_evidence_id"])
    if existing is not None:
        return {"status": "already_registered", "event": existing}
    clean = require_clean_tree(expected_branch=config["branch"])
    compatibility = verify_sources(config["compatibility_sources"])
    bank_path, bank_audit = _verify_bank_inputs(config)
    selected = select_development_rows(
        _read_jsonl(bank_path), config["selection"])
    conformance = _critical_source_conformance(config)
    if not conformance["all_pass"]:
        raise RuntimeError(
            "side Bank-W protocol differs from frozen Phase 4 protocol: "
            + json.dumps(conformance["checks"], sort_keys=True))
    forbidden = {
        "correct", "candidate_scores", "baseline_answer_margin",
        "intervention", "condition", "generated", "generation",
    }
    if any(forbidden & set(row) for row in selected):
        raise RuntimeError("selected Bank-W rows contain outcome columns")
    aliases = list(config["answer_contract"]["aliases"])
    labels = list(config["answer_contract"]["labels"])
    payload = {
        "schema_version": 1,
        "evidence_id": config["protocol_evidence_id"],
        "status": "frozen-before-olmo-baseline-outcomes",
        "code_commit": clean["code_commit"],
        "source_phase4_conformance": conformance,
        "compatibility": compatibility,
        "bank_audit": bank_audit,
        "selection": config["selection"],
        "selected_item_ids_sha256": object_sha256([
            row["item_id"] for row in selected]),
        "selected_family_ids_sha256": object_sha256(sorted({
            row["canonical_family"] for row in selected})),
        "answer_contract": config["answer_contract"],
        "capability_guard": config["capability_guard"],
        "model_evidence_ids": {
            row["slug"]: row["evidence_id"] for row in config["models"]},
        "reference_evidence_id": config["source_phase4"][
            "qwen_evidence_id"],
        "joint_evidence_id": config["joint_evidence_id"],
        "early_bundle_evidence_id": config["early_bundle_evidence_id"],
        "answer_labels_align": [alias.strip() for alias in aliases] == labels,
        "rows_expected_per_olmo_model": len(selected),
        "intervention_columns_present": False,
        "model_outcomes_opened_by_freeze": False,
        "claim_boundary": config["claim_boundary"],
    }
    payload["all_protocol_gates_pass"] = bool(
        conformance["all_pass"]
        and payload["answer_labels_align"]
        and len(selected) == int(config["selection"][
            "expected_rows_per_model"]))
    output = resolve_uri(config["outputs"]["protocol"], must_exist=False)
    if output.exists():
        raise FileExistsError(
            "unregistered side protocol output already exists; audit first")
    atomic_json(output, payload)
    command = (
        "python -m jspace_olmo_lineage.experiments.bank_w_capability "
        f"--config {config_path} --freeze-protocol")
    event = create(
        config["protocol_evidence_id"], tier="methods",
        what=(
            "Outcome-blind OLMo side compatibility freeze for the exact "
            "Phase 4 Bank-W baseline capability protocol."),
        command=command, outputs=[output],
        inputs={
            "config": file_sha256(config_path),
            "source_phase4_config": config["source_phase4"][
                "config_sha256"],
            "source_phase4_protocol": config["source_phase4"][
                "protocol_sha256"],
            "bank_file": config["bank_file_sha256"],
            "bank_rows": config["bank_rows_sha256"],
            "partition_payload": config["partition_payload_sha256"],
        },
        model_outcomes_opened=False,
        interventions_opened=False,
    )
    return {"protocol": payload, "event": event}


def _require_protocol(config: Mapping) -> tuple[dict, str]:
    path = resolve_uri(config["outputs"]["protocol"])
    event = resolve(config["protocol_evidence_id"])
    if not event["live"]:
        raise RuntimeError("side Bank-W protocol is not live")
    digest = file_sha256(path)
    if not any(row["sha256"] == digest for row in event["outputs"]):
        raise RuntimeError("side Bank-W protocol registry hash drift")
    payload = json.loads(path.read_text())
    if payload.get("all_protocol_gates_pass") is not True:
        raise RuntimeError("side Bank-W protocol did not pass")
    _critical_source_conformance(config)
    verify_sources(config["compatibility_sources"])
    return payload, digest


def _model_specification(config: Mapping, slug: str) -> dict:
    matches = [dict(row) for row in config["models"] if row["slug"] == slug]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate OLMo model slug: {slug}")
    return matches[0]


@torch.inference_mode()
def run_model(config_path: Path, config: Mapping, slug: str) -> dict:
    specification = _model_specification(config, slug)
    existing = _registered_output_check(specification["evidence_id"])
    if existing is not None:
        return {"status": "already_registered", "event": existing}
    clean = require_clean_tree(expected_branch=config["branch"])
    protocol, protocol_sha = _require_protocol(config)
    bank_path, bank_audit = _verify_bank_inputs(config)
    selected = select_development_rows(
        _read_jsonl(bank_path), config["selection"])
    model_reference = _model_reference(specification["model_uri"])
    model_path = resolve_uri(specification["model_uri"])
    tokenizer_hash = _tokenizer_source_hash(model_path)
    gpu = require_cuda_gpu()
    packages = _verify_packages(config["runtime"]["packages"])
    manifest = InputManifest(
        experiment_id=specification["evidence_id"],
        config_sha256=file_sha256(config_path),
        model_id=model_reference["model_id"],
        model_revision=model_reference["revision"],
        tokenizer_manifest_sha256=tokenizer_hash,
        lens_sha256="not-applicable-baseline-capability",
        bank_sha256=config["bank_rows_sha256"],
        partition_sha256=config["partition_payload_sha256"],
        scoring_spec_sha256=object_sha256(DEFAULT_SPEC.as_dict()),
        upstream={
            config["protocol_evidence_id"]: protocol_sha,
            config["source_phase4"]["protocol_evidence_id"]: config[
                "source_phase4"]["protocol_sha256"],
            config["bank_evidence_id"]: config["audit_file_sha256"],
        },
        code_commit=clean["code_commit"],
    )
    output_dir = (
        metrics_dir(slug) / "bank_w_capability"
        / specification["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "input_manifest.json"
    _write_manifest_once(manifest_path, manifest.envelope())
    state_store = StateStore(
        output_dir / "state.json",
        StateHeader(
            evidence_id=specification["evidence_id"],
            input_manifest_sha256=manifest.sha256(),
            config_sha256=manifest.config_sha256,
            model_revision=manifest.model_revision,
            bank_sha256=manifest.bank_sha256,
            partition_sha256=manifest.partition_sha256,
        ),
    )
    state = state_store.load() or {
        "done": {}, "rows": [], "gpu": gpu,
        "started_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if len(state["done"]) != len(state["rows"]):
        raise RuntimeError("Bank-W state done/row count mismatch")

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True)
    if type(tokenizer).__name__ != specification[
            "expected_tokenizer_class"]:
        raise RuntimeError(
            "Bank-W capability tokenizer class drift: "
            f"{type(tokenizer).__name__}")
    session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
    aliases = list(config["answer_contract"]["aliases"])
    labels = list(config["answer_contract"]["labels"])
    expected_tokens = {
        alias: [int(value) for value in ids]
        for alias, ids in specification[
            "expected_answer_token_ids"].items()
    }
    observed_tokens = {
        alias: [int(value) for value in session.answer_ids(alias)[0].tolist()]
        for alias in aliases
    }
    if observed_tokens != expected_tokens:
        raise RuntimeError("Bank-W capability answer token IDs drift")
    token_manifest_hash = object_sha256(observed_tokens)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True).to("cuda").eval()
    assert_model_on_cuda(hf_model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("OLMo tokenizer lacks pad and EOS tokens")
    alias_by_label = dict(zip(labels, aliases))
    checkpoint_every = int(config["checkpoint_every_rows"])
    newly_done = 0
    started = time.time()
    print(
        f"{slug}: {len(state['done'])}/{len(selected)} rows already banked; "
        f"GPU={gpu['name']}; BOS={session.bos_prefixed}", flush=True)
    for item in selected:
        item_id = item["item_id"]
        if item_id in state["done"]:
            continue
        scores, prompt_tokens, token_manifest = candidate_scores(
            hf_model, session, item["prompt"], aliases,
            batch_size=int(config["answer_contract"][
                "runtime_candidate_batch_size"]),
            pad_token_id=int(pad_token_id),
        )
        predicted = max(aliases, key=lambda alias: scores[alias])
        true_alias = alias_by_label[item["answer"]]
        best_wrong = max(
            scores[alias] for alias in aliases if alias != true_alias)
        row = {
            "study_id": "jspace-olmo-lineage",
            "phase": "olmo-lineage-o1",
            "tier": "development",
            "evidence_id": specification["evidence_id"],
            "model_slug": slug,
            "model_id": model_reference["model_id"],
            "model_revision": model_reference["revision"],
            "item_id": item_id,
            "canonical_family": item["canonical_family"],
            "item_seed": int(item["item_seed"]),
            "load": item["load"],
            "derivation": item["derivation"],
            "redundancy": item["redundancy"],
            "answer": item["answer"],
            "true_alias": true_alias,
            "predicted_alias": predicted,
            "correct": bool(predicted == true_alias),
            "baseline_answer_margin": float(
                scores[true_alias] - best_wrong),
            "true_answer_sequence_lp": float(scores[true_alias]),
            "candidate_scores_json": json.dumps(
                scores, sort_keys=True, ensure_ascii=False),
            "prompt_token_count": int(prompt_tokens),
            "answer_token_count": len(token_manifest[true_alias]),
            "answer_token_manifest_sha256": object_sha256(token_manifest),
        }
        state["rows"].append(row)
        state["done"][item_id] = True
        newly_done += 1
        if newly_done % checkpoint_every == 0:
            state_store.write(state)
            elapsed = max(time.time() - started, 1e-9)
            rate = newly_done / elapsed
            remaining = len(selected) - len(state["done"])
            print(
                f"{slug}: banked {len(state['done'])}/{len(selected)}; "
                f"{rate:.3f} rows/s; ETA={remaining / rate / 60:.1f}m",
                flush=True)
    state["completed_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state_store.write(state)
    rows = sorted(state["rows"], key=lambda row: row["item_id"])
    analysis = analyze_model_rows(
        rows, selection=config["selection"],
        guard=config["capability_guard"], aliases=aliases)
    payload = {
        "schema_version": 1,
        "model_slug": slug,
        "model_role": specification["role"],
        "model": model_reference,
        "gpu": gpu,
        "runtime_packages": packages,
        "protocol_evidence_id": config["protocol_evidence_id"],
        "protocol_sha256": protocol_sha,
        "source_phase4_protocol_evidence_id": config[
            "source_phase4"]["protocol_evidence_id"],
        "source_phase4_protocol_sha256": config[
            "source_phase4"]["protocol_sha256"],
        "tokenizer_manifest_sha256": tokenizer_hash,
        "answer_token_ids": observed_tokens,
        "answer_token_manifest_sha256": token_manifest_hash,
        "bank_audit": bank_audit,
        "analysis": analysis,
        "claim_boundary": config["claim_boundary"],
        "interventions_opened": False,
        "freeze_ready": False,
    }
    rows_path = output_dir / "bank_w_capability_rows.parquet"
    result_path = output_dir / "bank_w_capability_result.json"
    _atomic_parquet(rows_path, pd.DataFrame(rows))
    command = (
        "python -m jspace_olmo_lineage.experiments.bank_w_capability "
        f"--config {config_path} --model-slug {slug}")
    write_result(
        payload, result_path,
        Provenance(
            evidence_id=specification["evidence_id"],
            tier="development", command=command,
            inputs={
                "config": file_sha256(config_path),
                "bank_file": config["bank_file_sha256"],
                "bank_rows": config["bank_rows_sha256"],
                "partition_payload": config["partition_payload_sha256"],
                "side_protocol": protocol_sha,
                "source_phase4_protocol": config[
                    "source_phase4"]["protocol_sha256"],
            },
            input_manifest_sha256=manifest.sha256(),
            model=model_reference,
        ),
    )
    event = create(
        specification["evidence_id"], tier="development",
        what=(
            f"Bank-W baseline capability development gate for {slug}; "
            "full eight-answer sequence LP; no intervention outcome."),
        command=command,
        outputs=[manifest_path, rows_path, result_path],
        inputs={
            "config": file_sha256(config_path),
            "bank_file": config["bank_file_sha256"],
            "bank_rows": config["bank_rows_sha256"],
            "partition_payload": config["partition_payload_sha256"],
            "side_protocol": protocol_sha,
            "source_phase4_protocol": config[
                "source_phase4"]["protocol_sha256"],
        },
        baseline_capability_eligible=bool(
            analysis["independently_capability_eligible"]),
        interventions_opened=False,
    )
    return {"payload": payload, "event": event}


def _registered_envelope_payload(
        evidence_id: str, *, expected_path: Path) -> tuple[dict, str, dict]:
    event = resolve(evidence_id)
    if not event["live"]:
        raise RuntimeError(f"evidence is not live: {evidence_id}")
    rows = [
        row for row in event["outputs"]
        if Path(row["path"]).resolve() == expected_path.resolve()
    ]
    if len(rows) != 1:
        raise RuntimeError(f"result path is ambiguous for {evidence_id}")
    record = rows[0]
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"result hash drift for {evidence_id}")
    envelope = json.loads(path.read_text())
    if envelope["payload_sha256"] != object_sha256(envelope["payload"]):
        raise RuntimeError(f"result payload hash drift for {evidence_id}")
    return envelope["payload"], record["sha256"], event


def _registered_result_payload(
        config: Mapping, specification: Mapping) -> tuple[dict, str, dict]:
    path = (
        metrics_dir(specification["slug"]) / "bank_w_capability"
        / specification["evidence_id"] / "bank_w_capability_result.json")
    return _registered_envelope_payload(
        specification["evidence_id"], expected_path=path)


def _qwen_payload(config: Mapping) -> tuple[dict, str, dict]:
    source = config["source_phase4"]
    event = _source_event(config, source["qwen_evidence_id"])
    path = resolve_uri(source["qwen_result_uri"])
    actual = file_sha256(path)
    if actual != source["qwen_result_sha256"]:
        raise RuntimeError("imported Qwen Bank-W result hash drift")
    if not any(row.get("sha256") == actual for row in event["outputs"]):
        raise RuntimeError("Qwen source event does not pin result hash")
    envelope = json.loads(path.read_text())
    if envelope["payload_sha256"] != object_sha256(envelope["payload"]):
        raise RuntimeError("Qwen Bank-W result payload hash drift")
    return envelope["payload"], actual, event


def aggregate_joint(config_path: Path, config: Mapping) -> dict:
    existing = _registered_output_check(config["joint_evidence_id"])
    if existing is not None:
        return {"status": "already_registered", "event": existing}
    require_clean_tree(expected_branch=config["branch"])
    _, protocol_sha = _require_protocol(config)
    payloads = {}
    hashes = {}
    events = {}
    for specification in config["models"]:
        payload, digest, event = _registered_result_payload(
            config, specification)
        payloads[specification["slug"]] = payload
        hashes[specification["evidence_id"]] = digest
        events[specification["slug"]] = event
    qwen, qwen_hash, qwen_event = _qwen_payload(config)
    qwen_slug = config["reference_models"][0]["slug"]
    payloads[qwen_slug] = qwen
    hashes[config["source_phase4"]["qwen_evidence_id"]] = qwen_hash
    source_config = {
        "models": [{"slug": slug} for slug in config["joint_model_order"]],
        "capability_guard": config["capability_guard"],
        "claim_boundary": config["claim_boundary"],
    }
    source_rule = aggregate_model_payloads(payloads, config=source_config)
    required = list(config["joint_model_order"])
    strict_common = set(payloads[required[0]]["analysis"][
        "capable_family_ids"])
    for slug in required[1:]:
        strict_common &= set(payloads[slug]["analysis"][
            "capable_family_ids"])
    common = sorted(strict_common)
    all_eligible = all(payloads[slug]["analysis"][
        "independently_capability_eligible"] for slug in required)
    minimum = int(config["capability_guard"][
        "minimum_joint_common_families"])
    service_ready = bool(all_eligible and len(common) >= minimum)
    joint = {
        "schema_version": 1,
        "evidence_id": config["joint_evidence_id"],
        "model_order": required,
        "all_required_models_independently_eligible": all_eligible,
        "required_model_gate_status": {
            slug: bool(payloads[slug]["analysis"][
                "independently_capability_eligible"])
            for slug in required
        },
        "joint_common_capable_family_ids": common,
        "joint_common_capable_family_ids_sha256": object_sha256(common),
        "n_joint_common_capable_families": len(common),
        "minimum_joint_common_families": minimum,
        "olmo_phase4_service_ready": service_ready,
        "decision": (
            "PASS: OLMo Think/Instruct capability and three-model common "
            "support satisfy the prospective service gate"
            if service_ready else
            "BLOCKED: report all baselines and revise prospectively; do not "
            "open a Bank-W intervention outcome on the failed service set"),
        "model_analyses": {
            slug: payloads[slug]["analysis"] for slug in required},
        "phase4_source_rule_analysis": source_rule,
        "input_result_hashes": hashes,
        "source_events": {
            **{
                slug: {
                    "evidence_id": event["evidence_id"],
                    "code_commit": event.get("code_commit"),
                    "study_id": event.get("study_id"),
                }
                for slug, event in events.items()
            },
            qwen_slug: {
                "evidence_id": qwen_event["evidence_id"],
                "code_commit": qwen_event.get("code_commit"),
                "study_id": qwen_event.get("study_id"),
                "source_registry_sha256": config["source_phase4"][
                    "registry_sha256"],
            },
        },
        "claim_boundary": config["claim_boundary"],
        "interventions_opened": False,
        "freeze_ready": False,
    }
    table_rows = []
    for slug in required:
        analysis = payloads[slug]["analysis"]
        table_rows.append({
            "model_slug": slug,
            "accuracy_low": analysis["load_summaries"]["low"]["accuracy"],
            "accuracy_high": analysis["load_summaries"]["high"]["accuracy"],
            "high_minus_low": analysis[
                "paired_high_minus_low_accuracy"]["mean"],
            "ci90_low": analysis[
                "paired_high_minus_low_accuracy"][
                    "family_bootstrap_ci90"][0],
            "ci90_high": analysis[
                "paired_high_minus_low_accuracy"][
                    "family_bootstrap_ci90"][1],
            "n_capable_families": analysis["n_capable_families"],
            "independently_eligible": analysis[
                "independently_capability_eligible"],
        })
    result_path = resolve_uri(
        config["outputs"]["joint_result"], must_exist=False)
    table_path = resolve_uri(
        config["outputs"]["joint_table"], must_exist=False)
    if result_path.exists() or table_path.exists():
        raise FileExistsError("unregistered joint output exists; audit first")
    command = (
        "python -m jspace_olmo_lineage.experiments.bank_w_capability "
        f"--config {config_path} --aggregate-joint")
    input_hash = object_sha256(hashes)
    write_result(
        joint, result_path,
        Provenance(
            evidence_id=config["joint_evidence_id"], tier="development",
            command=command, inputs={
                "config": file_sha256(config_path),
                "side_protocol": protocol_sha,
                **hashes,
            },
            input_manifest_sha256=input_hash,
        ),
    )
    _atomic_parquet(table_path, pd.DataFrame(table_rows))
    event = create(
        config["joint_evidence_id"], tier="development",
        what=(
            "Joint OLMo Think/Instruct plus imported Qwen Bank-W baseline "
            "capability support gate; no intervention outcome."),
        command=command, outputs=[result_path, table_path],
        inputs={
            "config": file_sha256(config_path),
            "side_protocol": protocol_sha,
            **hashes,
        },
        olmo_phase4_service_ready=service_ready,
        interventions_opened=False,
    )
    return {"joint": joint, "event": event}


def emit_early_bundle(config_path: Path, config: Mapping) -> dict:
    existing = _registered_output_check(config["early_bundle_evidence_id"])
    if existing is not None:
        return {"status": "already_registered", "event": existing}
    clean = require_clean_tree(expected_branch=config["branch"])
    _, protocol_sha = _require_protocol(config)
    model_records = {}
    for specification in config["models"]:
        payload, digest, event = _registered_result_payload(
            config, specification)
        model_records[specification["slug"]] = {
            "evidence_id": specification["evidence_id"],
            "result_sha256": digest,
            "code_commit": event.get("code_commit"),
            "outputs": event["outputs"],
            "analysis": payload["analysis"],
        }
    joint_payload, joint_hash, joint_event = _registered_envelope_payload(
        config["joint_evidence_id"],
        expected_path=resolve_uri(config["outputs"]["joint_result"]))
    qwen_payload, qwen_hash, qwen_event = _qwen_payload(config)
    registry_sha = file_sha256(EVENTS)
    registry_bytes = int(EVENTS.stat().st_size)
    bundle = {
        "schema_version": 1,
        "bundle_id": "jspace-olmo-lineage-phase4-early-v1",
        "evidence_id": config["early_bundle_evidence_id"],
        "source_study": "jspace-olmo-lineage",
        "source_branch": clean["branch"],
        "source_commit": clean["code_commit"],
        "scientific_import_boundary": (
            "3b041735d8b842de46a9c0a474fccd0c44e0841a"),
        "source_registry": {
            "path": str(EVENTS),
            "prefix_bytes": registry_bytes,
            "prefix_sha256": registry_sha,
            "through_evidence_id": config["joint_evidence_id"],
            "note": (
                "Hash exactly the first prefix_bytes of the later registry; "
                "the bundle registration event is appended afterward."),
        },
        "protocol": {
            "evidence_id": config["protocol_evidence_id"],
            "sha256": protocol_sha,
            "source_phase4_evidence_id": config["source_phase4"][
                "protocol_evidence_id"],
            "source_phase4_sha256": config["source_phase4"][
                "protocol_sha256"],
        },
        "olmo_model_records": model_records,
        "joint_record": {
            "evidence_id": config["joint_evidence_id"],
            "result_sha256": joint_hash,
            "code_commit": joint_event.get("code_commit"),
            "outputs": joint_event["outputs"],
            "service_summary": {
                key: joint_payload[key] for key in (
                    "all_required_models_independently_eligible",
                    "n_joint_common_capable_families",
                    "minimum_joint_common_families",
                    "olmo_phase4_service_ready",
                    "decision",
                )
            },
        },
        "imported_qwen_reference": {
            "source_study": qwen_event["study_id"],
            "evidence_id": qwen_event["evidence_id"],
            "source_commit": qwen_event.get("code_commit"),
            "source_registry_sha256": config["source_phase4"][
                "registry_sha256"],
            "result_sha256": qwen_hash,
            "analysis": qwen_payload["analysis"],
        },
        "reproduction": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "verify_command": "python -m jspace_olmo_lineage verify",
            "no_intervention_columns": True,
        },
        "claim_boundary": config["claim_boundary"],
        "interventions_opened": False,
        "independent_review_or_pi_signoff": False,
    }
    json_path = resolve_uri(
        config["outputs"]["early_bundle_json"], must_exist=False)
    markdown_path = resolve_uri(
        config["outputs"]["early_bundle_markdown"], must_exist=False)
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("unregistered early bundle exists; audit first")
    command = (
        "python -m jspace_olmo_lineage.experiments.bank_w_capability "
        f"--config {config_path} --emit-early-bundle")
    write_result(
        bundle, json_path,
        Provenance(
            evidence_id=config["early_bundle_evidence_id"], tier="methods",
            command=command,
            inputs={
                "config": file_sha256(config_path),
                "registry_prefix": registry_sha,
                "joint_result": joint_hash,
                "qwen_result": qwen_hash,
                **{
                    row["evidence_id"]: row["result_sha256"]
                    for row in model_records.values()
                },
            },
            input_manifest_sha256=object_sha256({
                "registry_prefix": registry_sha,
                "joint_result": joint_hash,
                "qwen_result": qwen_hash,
            }),
        ),
    )
    summary = joint_payload
    lines = [
        "# OLMo Bank-W early Phase 4 import bundle",
        "",
        f"Source branch: `{clean['branch']}`  ",
        f"Source commit: `{clean['code_commit']}`  ",
        f"Registry prefix SHA-256: `{registry_sha}`  ",
        "",
        "This bundle contains baseline capability only. It opens no Bank-W "
        "intervention, confirmatory, or replication outcome.",
        "",
        "| model | low acc. | high acc. | 90% CI high-low | capable families | eligible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for slug in config["joint_model_order"]:
        analysis = summary["model_analyses"][slug]
        low = analysis["load_summaries"]["low"]["accuracy"]
        high = analysis["load_summaries"]["high"]["accuracy"]
        interval = analysis["paired_high_minus_low_accuracy"][
            "family_bootstrap_ci90"]
        lines.append(
            f"| {slug} | {low:.4f} | {high:.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
            f"{analysis['n_capable_families']} | "
            f"{analysis['independently_capability_eligible']} |")
    lines.extend([
        "",
        f"Joint common capable families: "
        f"{summary['n_joint_common_capable_families']} "
        f"(required {summary['minimum_joint_common_families']}).",
        "",
        f"Service gate: **{'PASS' if summary['olmo_phase4_service_ready'] else 'BLOCKED'}**.",
        "",
        "Importers must verify the JSON envelope, registry prefix, source "
        "events, and every output hash before use.",
        "",
    ])
    atomic_text(markdown_path, "\n".join(lines))
    event = create(
        config["early_bundle_evidence_id"], tier="methods",
        what=(
            "Hash-pinned early Phase 4 import bundle for OLMo Bank-W "
            "baseline capability and joint support."),
        command=command, outputs=[json_path, markdown_path],
        inputs={
            "config": file_sha256(config_path),
            "registry_prefix": registry_sha,
            "joint_result": joint_hash,
            "qwen_result": qwen_hash,
            **{
                row["evidence_id"]: row["result_sha256"]
                for row in model_records.values()
            },
        },
        olmo_phase4_service_ready=bool(
            joint_payload["olmo_phase4_service_ready"]),
        interventions_opened=False,
    )
    return {"bundle": bundle, "event": event}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-protocol", action="store_true")
    group.add_argument("--model-slug")
    group.add_argument("--aggregate-joint", action="store_true")
    group.add_argument("--emit-early-bundle", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = _load_config(config_path)
    if arguments.freeze_protocol:
        result = freeze_protocol(config_path, config)
    elif arguments.model_slug:
        result = run_model(config_path, config, arguments.model_slug)
    elif arguments.aggregate_joint:
        result = aggregate_joint(config_path, config)
    elif arguments.emit_early_bundle:
        result = emit_early_bundle(config_path, config)
    else:  # pragma: no cover
        raise RuntimeError("Bank-W capability action is missing")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
