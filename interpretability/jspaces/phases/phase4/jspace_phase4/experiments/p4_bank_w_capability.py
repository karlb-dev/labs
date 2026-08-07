"""Prospective Bank W baseline-capability protocol and model gates.

The methods action validates only the already-authored bank and the frozen
candidate-score contract. Model actions open development baseline outcomes
only; they never run a J or matched-control intervention. The joint action
selects the primary model set from those baseline outcomes using a rule fixed
before any Bank W intervention outcome exists.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    InputManifest,
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import figures_dir, metrics_dir, resolve_uri
from ..provenance4 import Provenance4, write_result4
from ..registry4 import RegistryError, create, resolve
from ..scoring4 import DEFAULT_SPEC, ScoringSession
from ..state import StateHeader, StateStore
from .p4_g5_bank_scoring import tokenizer_source_hash
from .p4_qwen_nested_lens_fit import (
    registered_output_check,
    verify_model_fused_bindings,
    verify_package_versions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--author-protocol", action="store_true")
    group.add_argument("--register-protocol", action="store_true")
    group.add_argument("--model-slug")
    group.add_argument("--aggregate-joint", action="store_true")
    group.add_argument("--register-joint", action="store_true")
    return parser.parse_args()


def model_reference(uri: str) -> dict:
    if not uri.startswith("model://") or "@" not in uri:
        raise ValueError("model URI must pin an exact revision")
    model_id, revision = uri[len("model://"):].rsplit("@", 1)
    return {"model_id": model_id, "revision": revision}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _verify_inputs(config: Mapping) -> tuple[Path, Path, Path, dict, dict]:
    bank = resolve_uri(config["bank_uri"])
    audit = resolve_uri(config["audit_uri"])
    partition = resolve_uri(config["partition_uri"])
    expected = {
        bank: config["bank_file_sha256"],
        audit: config["audit_file_sha256"],
        partition: config["partition_file_sha256"],
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"Bank W capability input hash drift: {path}")
    audit_payload = json.loads(audit.read_text())
    partition_envelope = json.loads(partition.read_text())
    if audit_payload.get("evidence_id") != config["bank_evidence_id"]:
        raise RuntimeError("Bank W audit evidence ID drift")
    if audit_payload.get("bank_rows_sha256") != config[
            "bank_rows_sha256"]:
        raise RuntimeError("Bank W audit row payload hash drift")
    if audit_payload.get("partition_sha256") != config[
            "partition_payload_sha256"]:
        raise RuntimeError("Bank W audit partition hash drift")
    if partition_envelope.get("payload_sha256") != config[
            "partition_payload_sha256"]:
        raise RuntimeError("Bank W partition payload hash drift")
    return bank, audit, partition, audit_payload, partition_envelope


def select_development_rows(rows: Sequence[Mapping],
                            selection: Mapping) -> list[dict]:
    selected = [
        dict(row) for row in rows
        if row.get("partition") == selection["partition"]
        and row.get("derivation") == selection["derivation"]
        and row.get("redundancy") == selection["redundancy"]
        and row.get("load") in set(selection["loads"])
    ]
    selected.sort(key=lambda row: row["item_id"])
    if len(selected) != int(selection["expected_rows_per_model"]):
        raise RuntimeError("unexpected Bank W capability row count")
    families = sorted({row["canonical_family"] for row in selected})
    if len(families) != int(selection["expected_families"]):
        raise RuntimeError("unexpected Bank W capability family count")
    keys = [(row["canonical_family"], int(row["item_seed"]), row["load"])
            for row in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate Bank W capability family/seed/load row")
    expected_seeds = set(range(int(selection["expected_seeds_per_family"])))
    for family in families:
        for load in selection["loads"]:
            observed = {int(row["item_seed"]) for row in selected
                        if row["canonical_family"] == family
                        and row["load"] == load}
            if observed != expected_seeds:
                raise RuntimeError(
                    f"incomplete Bank W capability grid: {family}/{load}")
    forbidden = {
        "correct", "candidate_scores", "baseline_answer_margin",
        "intervention", "condition", "generated", "generation",
    }
    if any(forbidden & set(row) for row in selected):
        raise RuntimeError("Bank W bank contains forbidden outcome columns")
    return selected


def author_protocol(config: Mapping) -> dict:
    bank, audit, partition, audit_payload, partition_envelope = _verify_inputs(
        config)
    rows = select_development_rows(_read_jsonl(bank), config["selection"])
    contract = config["answer_contract"]
    aliases = list(contract["aliases"])
    labels = list(contract["labels"])
    if [alias.strip() for alias in aliases] != labels:
        raise RuntimeError("Bank W answer label/alias alignment drift")
    if sorted({row["answer"] for row in rows}) != sorted(labels):
        raise RuntimeError("Bank W selected answers do not match contract")
    token_audits = {}
    for specification in config["models"]:
        token_ids = {
            alias: [int(value) for value in
                    specification["expected_answer_token_ids"][alias]]
            for alias in aliases
        }
        sequences = list(token_ids.values())
        distinct = len({tuple(value) for value in sequences}) == len(sequences)
        maximum = max(map(len, sequences))
        token_audits[specification["slug"]] = {
            "tokenizer_class": specification["expected_tokenizer_class"],
            "answer_token_ids": token_ids,
            "answer_token_manifest_sha256": object_sha256(token_ids),
            "distinct_sequences": distinct,
            "maximum_answer_tokens": maximum,
            "passes": bool(
                distinct
                and maximum <= int(contract["maximum_answer_tokens"])),
        }
    checks = {
        "bank_candidate_v2_matches": (
            audit_payload["evidence_id"] == config["bank_evidence_id"]),
        "bank_axes_and_shortcuts_pass": bool(
            audit_payload["all_axes_fully_crossed"]
            and audit_payload["shortcut_audit"]["all_pass"]),
        "partition_payload_matches": (
            partition_envelope["payload_sha256"]
            == config["partition_payload_sha256"]),
        "development_primary_grid_complete": (
            len(rows) == int(config["selection"][
                "expected_rows_per_model"])),
        "answer_token_contract_passes": all(
            value["passes"] for value in token_audits.values()),
        "model_order_matches_bank_primary": (
            [row["slug"] for row in config["models"]]
            == audit_payload["primary"]["model_slugs"]),
        "guard_matches_registered_candidate": (
            float(config["capability_guard"]["baseline_accuracy_floor"])
            == float(audit_payload["capability_guard"][
                "baseline_accuracy_floor"])
            and float(config["capability_guard"][
                "low_high_accuracy_difference_sesoi"])
            == float(audit_payload["capability_guard"][
                "low_high_accuracy_difference_sesoi"])
            and int(config["capability_guard"][
                "minimum_joint_common_families"])
            == int(audit_payload["capability_guard"][
                "minimum_common_families_per_model"])),
    }
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "bank_evidence_id": config["bank_evidence_id"],
        "power_evidence_id": config["power_evidence_id"],
        "input_hashes": {
            "bank": file_sha256(bank), "audit": file_sha256(audit),
            "partition": file_sha256(partition),
        },
        "selection": dict(config["selection"]),
        "selected_item_ids_sha256": object_sha256(
            [row["item_id"] for row in rows]),
        "selected_family_ids_sha256": object_sha256(sorted(
            {row["canonical_family"] for row in rows})),
        "answer_contract": dict(contract),
        "tokenizer_answer_audits": token_audits,
        "capability_guard": dict(config["capability_guard"]),
        "model_evidence_ids": {
            row["slug"]: row["evidence_id"] for row in config["models"]},
        "joint_evidence_id": config["joint_evidence_id"],
        "gate_checks": checks,
        "all_protocol_gates_pass": bool(all(checks.values())),
        "outcome_blinding": (
            "Bank structure, tokenizer answer IDs, and analysis rules only; "
            "no baseline or intervention model outcome loaded."),
        "claim_boundary": config["claim_boundary"],
        "freeze_ready": False,
        "remaining_freeze_blockers": [
            "three model-specific Bank W baseline capability outcomes",
            "joint common-family capability intersection",
            "independent protocol review and PI sign-off",
        ],
    }


def analyze_model_rows(rows: Sequence[Mapping], *, selection: Mapping,
                       guard: Mapping) -> dict:
    expected_loads = list(selection["loads"])
    keys = [(str(row["canonical_family"]), int(row["item_seed"]),
             str(row["load"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate scored Bank W capability row")
    families = sorted({key[0] for key in keys})
    expected_families = int(selection["expected_families"])
    expected_seeds = int(selection["expected_seeds_per_family"])
    if len(rows) != int(selection["expected_rows_per_model"]) \
            or len(families) != expected_families:
        raise RuntimeError("incomplete scored Bank W capability grid")
    by_key = {key: row for key, row in zip(keys, rows)}
    if not all((family, seed, load) in by_key
               for family in families
               for seed in range(expected_seeds)
               for load in expected_loads):
        raise RuntimeError("missing scored Bank W capability cell")

    load_summaries = {}
    for load in expected_loads:
        subset = [row for row in rows if row["load"] == load]
        load_summaries[load] = {
            "n_rows": len(subset),
            "accuracy": float(np.mean([bool(row["correct"])
                                       for row in subset])),
            "mean_answer_margin": float(np.mean([
                float(row["baseline_answer_margin"]) for row in subset])),
            "mean_prompt_tokens": float(np.mean([
                int(row["prompt_token_count"]) for row in subset])),
            "mean_answer_tokens": float(np.mean([
                int(row["answer_token_count"]) for row in subset])),
        }

    family_accuracy = {}
    for family in families:
        family_accuracy[family] = {}
        for load in expected_loads:
            family_accuracy[family][load] = float(np.mean([
                bool(by_key[(family, seed, load)]["correct"])
                for seed in range(expected_seeds)]))
    differences = np.asarray([
        family_accuracy[family]["high"]
        - family_accuracy[family]["low"]
        for family in families], dtype=np.float64)
    generator = np.random.default_rng(int(guard["family_bootstrap_seed"]))
    draws = int(guard["family_bootstrap_draws"])
    indices = generator.integers(
        0, len(families), size=(draws, len(families)))
    bootstraps = differences[indices].mean(axis=1)
    alpha = 1.0 - float(guard["equivalence_interval_level"])
    interval = [float(value) for value in np.quantile(
        bootstraps, [alpha / 2, 1 - alpha / 2])]
    sesoi = float(guard["low_high_accuracy_difference_sesoi"])
    floor = float(guard["baseline_accuracy_floor"])
    family_floor = float(guard[
        "family_capability_accuracy_floor_by_load"])
    capable = [
        family for family in families
        if all(family_accuracy[family][load] >= family_floor
               for load in expected_loads)
    ]
    checks = {
        "complete_development_grid": True,
        "accuracy_floor_both_loads": all(
            load_summaries[load]["accuracy"] >= floor
            for load in expected_loads),
        "load_difference_equivalent": (
            interval[0] >= -sesoi and interval[1] <= sesoi),
    }

    def correlation(field: str, outcome: str) -> float | None:
        left = np.asarray([float(row[field]) for row in rows])
        right = np.asarray([float(row[outcome]) for row in rows])
        if np.std(left) == 0 or np.std(right) == 0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    return {
        "schema_version": 1,
        "n_rows": len(rows),
        "n_families": len(families),
        "load_summaries": load_summaries,
        "paired_high_minus_low_accuracy": {
            "mean": float(differences.mean()),
            "family_bootstrap_ci90": interval,
            "family_bootstrap_draws": draws,
            "family_bootstrap_seed": int(guard[
                "family_bootstrap_seed"]),
            "equivalence_sesoi": sesoi,
        },
        "family_accuracy": family_accuracy,
        "capable_family_ids": capable,
        "capable_family_ids_sha256": object_sha256(capable),
        "n_capable_families": len(capable),
        "locked_covariate_sensitivities": {
            "correct_vs_prompt_token_count_r": correlation(
                "prompt_token_count", "correct"),
            "correct_vs_answer_token_count_r": correlation(
                "answer_token_count", "correct"),
            "margin_vs_prompt_token_count_r": correlation(
                "prompt_token_count", "baseline_answer_margin"),
            "margin_vs_answer_token_count_r": correlation(
                "answer_token_count", "baseline_answer_margin"),
            "status": "secondary-only; cannot rescue a failed primary gate",
        },
        "independent_gate_checks": checks,
        "independently_capability_eligible": bool(all(checks.values())),
        "claim_boundary": (
            "Bank W baseline development capability only; no intervention, "
            "confirmatory, or replication outcome."),
    }


def aggregate_model_payloads(payloads: Mapping[str, Mapping], *,
                             config: Mapping) -> dict:
    order = [row["slug"] for row in config["models"]]
    if set(payloads) != set(order):
        raise RuntimeError("joint capability aggregation lacks a frozen model")
    independently_eligible = [
        slug for slug in order
        if payloads[slug]["analysis"]["independently_capability_eligible"]
    ]
    if independently_eligible:
        common = set(payloads[independently_eligible[0]]["analysis"][
            "capable_family_ids"])
        for slug in independently_eligible[1:]:
            common &= set(payloads[slug]["analysis"][
                "capable_family_ids"])
    else:
        common = set()
    common_ids = sorted(common)
    minimum = int(config["capability_guard"][
        "minimum_joint_common_families"])
    joint_pass = bool(independently_eligible and len(common_ids) >= minimum)
    return {
        "schema_version": 1,
        "model_order": order,
        "independently_eligible_models": independently_eligible,
        "independently_failed_models": [
            slug for slug in order if slug not in independently_eligible],
        "would_be_primary_model_set": independently_eligible,
        "primary_model_set": independently_eligible if joint_pass else [],
        "joint_common_capable_family_ids": common_ids,
        "joint_common_capable_family_ids_sha256": object_sha256(common_ids),
        "n_joint_common_capable_families": len(common_ids),
        "minimum_joint_common_families": minimum,
        "joint_common_support_pass": joint_pass,
        "p4p3_baseline_capability_ready": joint_pass,
        "decision": (
            "PASS: freeze the independently eligible model set before any "
            "Bank W intervention outcome"
            if joint_pass else
            "BLOCKED: do not drop an otherwise eligible model to manufacture "
            "common support; revise capability/bank design prospectively"),
        "model_analyses": {
            slug: payloads[slug]["analysis"] for slug in order},
        "claim_boundary": config["claim_boundary"],
        "freeze_ready": False,
        "remaining_freeze_blockers": [
            "independent protocol review and PI sign-off",
            "Bank W intervention producer and exact model/lens manifests",
        ] if joint_pass else [
            "prospective Bank W capability/bank revision",
            "independent protocol review and PI sign-off",
        ],
    }


def _protocol_path(config: Mapping) -> Path:
    return resolve_uri(config["outputs"]["protocol"], must_exist=False)


def _require_protocol(config: Mapping) -> tuple[dict, str]:
    path = _protocol_path(config)
    if not path.exists():
        raise RuntimeError("Bank W capability protocol output is absent")
    payload = json.loads(path.read_text())
    if payload.get("all_protocol_gates_pass") is not True:
        raise RuntimeError("Bank W capability protocol did not pass")
    event = resolve(config["evidence_id"])
    if not event["live"]:
        raise RuntimeError("Bank W capability protocol evidence is not live")
    digest = file_sha256(path)
    registered = [
        row for row in event["outputs"]
        if row["sha256"] == digest
        and row["path"].endswith(config["outputs"]["protocol"])
    ]
    if len(registered) != 1:
        raise RuntimeError("registered Bank W capability protocol hash drift")
    return payload, digest


def _model_specification(config: Mapping, slug: str) -> dict:
    matches = [dict(row) for row in config["models"]
               if row["slug"] == slug]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate Bank W model slug {slug!r}")
    return matches[0]


def _candidate_scores(hf_model, session: ScoringSession, prompt: str,
                      aliases: Sequence[str], *, batch_size: int,
                      pad_token_id: int) -> tuple[dict[str, float], int, dict]:
    prompt_ids = session.prompt_ids(prompt)
    prompt_length = int(prompt_ids.shape[1])
    answer_ids = {alias: session.answer_ids(alias) for alias in aliases}
    scores = {}
    for start in range(0, len(aliases), batch_size):
        batch_aliases = list(aliases[start:start + batch_size])
        full = [torch.cat([prompt_ids, answer_ids[alias]], dim=1)[0]
                for alias in batch_aliases]
        lengths = [int(value.shape[0]) for value in full]
        width = max(lengths)
        input_ids = torch.full(
            (len(full), width), int(pad_token_id), dtype=torch.long,
            device=prompt_ids.device)
        attention = torch.zeros_like(input_ids)
        for index, values in enumerate(full):
            input_ids[index, :values.shape[0]] = values
            attention[index, :values.shape[0]] = 1
        logits = hf_model(
            input_ids=input_ids, attention_mask=attention,
            use_cache=False).logits
        for index, alias in enumerate(batch_aliases):
            answer = answer_ids[alias][0]
            answer_logits = logits[
                index, prompt_length - 1:prompt_length + answer.shape[0] - 1,
            ].float()
            token_lps = torch.log_softmax(answer_logits, dim=-1).gather(
                1, answer.to(answer_logits.device).unsqueeze(1)).squeeze(1)
            scores[alias] = float(token_lps.sum().item())
        del logits, input_ids, attention
    token_manifest = {
        alias: [int(value) for value in answer_ids[alias][0].tolist()]
        for alias in aliases}
    return scores, prompt_length, token_manifest


@torch.no_grad()
def run_model(config_path: Path, config: Mapping, slug: str) -> None:
    specification = _model_specification(config, slug)
    existing = registered_output_check(specification["evidence_id"])
    if existing is not None:
        print(
            f"{specification['evidence_id']} is already registered and "
            "all outputs verify; nothing to do",
            flush=True,
        )
        return
    clean = require_clean_tree()
    protocol, protocol_sha = _require_protocol(config)
    bank, _, _, _, _ = _verify_inputs(config)
    selected = select_development_rows(
        _read_jsonl(bank), config["selection"])
    model = model_reference(specification["model_uri"])
    model_path = resolve_uri(specification["model_uri"])
    tokenizer_hash = tokenizer_source_hash(model_path)
    gpu = require_cuda_gpu()
    manifest = InputManifest(
        experiment_id=specification["evidence_id"],
        config_sha256=file_sha256(config_path),
        model_id=model["model_id"], model_revision=model["revision"],
        tokenizer_manifest_sha256=tokenizer_hash,
        lens_sha256="not-applicable-baseline-capability",
        bank_sha256=config["bank_rows_sha256"],
        partition_sha256=config["partition_payload_sha256"],
        scoring_spec_sha256=object_sha256(DEFAULT_SPEC.as_dict()),
        upstream={
            config["evidence_id"]: protocol_sha,
            config["bank_evidence_id"]: config["audit_file_sha256"],
            config["power_evidence_id"]: file_sha256(resolve_uri(
                "repo://interpretability/jspaces/phases/phase4/reports/"
                "bank_w_power_dev_v1.json")),
        },
        code_commit=clean["code_commit"],
    )
    output_dir = (metrics_dir("bank-w-capability") / slug
                  / specification["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "input_manifest.json"
    atomic_json(manifest_path, manifest.envelope())
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
    state = state_store.load() or {"done": {}, "rows": [], "gpu": gpu}

    import transformers
    verify_package_versions(config["runtime"]["packages"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    if type(tokenizer).__name__ != specification[
            "expected_tokenizer_class"]:
        raise RuntimeError("Bank W capability tokenizer class drift")
    session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
    expected_tokens = {
        alias: [int(value) for value in ids]
        for alias, ids in specification[
            "expected_answer_token_ids"].items()}
    observed_tokens = {
        alias: [int(value) for value in session.answer_ids(alias)[0].tolist()]
        for alias in config["answer_contract"]["aliases"]}
    if observed_tokens != expected_tokens:
        raise RuntimeError("Bank W capability answer token IDs drift")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    assert_model_on_cuda(hf_model)
    fused = None
    if slug == "qwen36-27b":
        fused = verify_model_fused_bindings(hf_model, {
            "expected_linear_attention_modules": int(config["runtime"][
                "expected_qwen_linear_attention_modules"]),
            "qwen_kernel_modules": config["runtime"][
                "qwen_kernel_modules"],
        })
    aliases = list(config["answer_contract"]["aliases"])
    labels = list(config["answer_contract"]["labels"])
    alias_by_label = dict(zip(labels, aliases))
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Bank W capability tokenizer lacks pad/EOS token")
    checkpoint_every = int(config["checkpoint_every_rows"])
    newly_done = 0
    for index, item in enumerate(selected, start=1):
        item_id = item["item_id"]
        if item_id in state["done"]:
            continue
        scores, prompt_tokens, token_manifest = _candidate_scores(
            hf_model, session, item["prompt"], aliases,
            batch_size=int(config["answer_contract"][
                "runtime_candidate_batch_size"]),
            pad_token_id=int(pad_token_id),
        )
        predicted = max(aliases, key=lambda alias: scores[alias])
        true_alias = alias_by_label[item["answer"]]
        best_wrong = max(scores[alias] for alias in aliases
                         if alias != true_alias)
        row = {
            "study_id": "jspace-phase4", "phase": "development",
            "tier": "phase4-development",
            "evidence_id": specification["evidence_id"],
            "model_id": model["model_id"],
            "model_revision": model["revision"],
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
            "prompt_token_count": prompt_tokens,
            "answer_token_count": len(token_manifest[true_alias]),
            "answer_token_manifest_sha256": object_sha256(token_manifest),
        }
        state["rows"].append(row)
        state["done"][item_id] = True
        newly_done += 1
        if newly_done % checkpoint_every == 0:
            state_store.write(state)
            print(f"{slug}: banked {len(state['done'])}/{len(selected)} rows",
                  flush=True)
    state_store.write(state)
    rows = sorted(state["rows"], key=lambda row: row["item_id"])
    analysis = analyze_model_rows(
        rows, selection=config["selection"],
        guard=config["capability_guard"])
    payload = {
        "schema_version": 1,
        "model_slug": slug,
        "model": model,
        "gpu": gpu,
        "qwen_fused_binding_audit": fused,
        "protocol_evidence_id": config["evidence_id"],
        "protocol_sha256": protocol_sha,
        "tokenizer_manifest_sha256": tokenizer_hash,
        "answer_token_ids": observed_tokens,
        "analysis": analysis,
        "claim_boundary": config["claim_boundary"],
        "freeze_ready": False,
    }
    rows_path = output_dir / "bank_w_capability_rows.parquet"
    result_path = output_dir / "bank_w_capability_result.json"
    _atomic_parquet(rows_path, pd.DataFrame(rows))
    command = (
        "python -m jspace_phase4.experiments.p4_bank_w_capability "
        f"--config {config_path} --model-slug {slug}")
    write_result4(
        payload, result_path,
        Provenance4(
            evidence_id=specification["evidence_id"],
            tier="phase4-development", command=command,
            inputs={
                "config": file_sha256(config_path),
                "bank_file": config["bank_file_sha256"],
                "bank_rows": config["bank_rows_sha256"],
                "partition_payload": config["partition_payload_sha256"],
                "protocol": protocol_sha,
            },
            input_manifest_sha256=manifest.sha256(), model=model,
        ),
    )
    create(
        specification["evidence_id"], tier="phase4-development",
        what=(f"Bank W baseline capability development gate for {slug}; "
              "no intervention outcome."),
        command=command,
        outputs=[manifest_path, rows_path, result_path],
        inputs={
            "config": file_sha256(config_path),
            "bank_file": config["bank_file_sha256"],
            "bank_rows": config["bank_rows_sha256"],
            "partition_payload": config["partition_payload_sha256"],
            "protocol": protocol_sha,
        },
    )


def _registered_model_payload(config: Mapping,
                              specification: Mapping) -> tuple[dict, str]:
    event = resolve(specification["evidence_id"])
    if not event["live"]:
        raise RuntimeError("Bank W model capability evidence is not live")
    result_rows = [row for row in event["outputs"]
                   if row["path"].endswith("bank_w_capability_result.json")]
    if len(result_rows) != 1:
        raise RuntimeError("Bank W model capability result path is ambiguous")
    row = result_rows[0]
    path = Path(row["path"])
    if file_sha256(path) != row["sha256"]:
        raise RuntimeError("Bank W model capability result hash drift")
    envelope = json.loads(path.read_text())
    if envelope["payload_sha256"] != object_sha256(envelope["payload"]):
        raise RuntimeError("Bank W model capability payload hash drift")
    return envelope["payload"], row["sha256"]


def _plot_joint(payload: Mapping, *, png: Path, pdf: Path) -> None:
    order = payload["model_order"]
    analyses = payload["model_analyses"]
    labels = [slug.replace("olmo31-", "OLMo ").replace(
        "qwen36-27b", "Qwen") for slug in order]
    x = np.arange(len(order))
    width = 0.34
    low = [analyses[slug]["load_summaries"]["low"]["accuracy"]
           for slug in order]
    high = [analyses[slug]["load_summaries"]["high"]["accuracy"]
            for slug in order]
    capable = [analyses[slug]["n_capable_families"] for slug in order]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    axes[0].bar(x - width / 2, low, width=width, label="low load",
                color="#56B4E9")
    axes[0].bar(x + width / 2, high, width=width, label="high load",
                color="#009E73")
    axes[0].axhline(0.70, color="#555555", linestyle="--", linewidth=1,
                    label="accuracy floor")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("candidate-set accuracy")
    axes[0].set_title("A · Frozen baseline endpoints", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    colors = ["#009E73" if slug in payload[
        "independently_eligible_models"] else "#999999" for slug in order]
    axes[1].bar(x, capable, color=colors)
    axes[1].axhline(payload["minimum_joint_common_families"],
                    color="#555555", linestyle="--", linewidth=1,
                    label="joint family floor")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, max(24, max(capable)) + 1)
    axes[1].set_ylabel("families capable at both loads")
    axes[1].set_title("B · Model and joint support", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    status = "PASS" if payload[
        "p4p3_baseline_capability_ready"] else "BLOCKED"
    figure.suptitle(
        "Bank W baseline capability gate · Phase 4 development\n"
        f"{status}; joint intersection n="
        f"{payload['n_joint_common_capable_families']}", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def aggregate_joint(config_path: Path, config: Mapping) -> None:
    require_clean_tree()
    _require_protocol(config)
    payloads = {}
    input_hashes = {}
    for specification in config["models"]:
        payload, digest = _registered_model_payload(config, specification)
        payloads[specification["slug"]] = payload
        input_hashes[specification["evidence_id"]] = digest
    joint = aggregate_model_payloads(payloads, config=config)
    joint["input_result_hashes"] = input_hashes
    report = resolve_uri(config["outputs"]["joint_report"], must_exist=False)
    stem = config["outputs"]["joint_figure_stem"]
    png = figures_dir() / f"{stem}.png"
    pdf = figures_dir() / f"{stem}.pdf"
    atomic_json(report, joint)
    _plot_joint(joint, png=png, pdf=pdf)


def register_protocol(config_path: Path, config: Mapping) -> None:
    require_clean_tree()
    path = _protocol_path(config)
    payload = json.loads(path.read_text())
    if payload.get("all_protocol_gates_pass") is not True:
        raise RuntimeError("cannot register a failed Bank W protocol")
    create(
        config["evidence_id"], tier="methods",
        what=("Outcome-blind Bank W baseline capability scoring, model "
              "selection, equivalence, and joint-support protocol."),
        command=(
            "python -m jspace_phase4.experiments.p4_bank_w_capability "
            f"--config {config_path} --register-protocol"),
        outputs=[config["outputs"]["protocol"]],
        inputs={
            "config": file_sha256(config_path),
            "bank_file": config["bank_file_sha256"],
            "bank_rows": config["bank_rows_sha256"],
            "audit_file": config["audit_file_sha256"],
            "partition_file": config["partition_file_sha256"],
            "partition_payload": config["partition_payload_sha256"],
        },
    )


def register_joint(config_path: Path, config: Mapping) -> None:
    require_clean_tree()
    report = resolve_uri(config["outputs"]["joint_report"])
    payload = json.loads(report.read_text())
    stem = config["outputs"]["joint_figure_stem"]
    png = figures_dir() / f"{stem}.png"
    pdf = figures_dir() / f"{stem}.pdf"
    inputs = {
        row["evidence_id"]: _registered_model_payload(config, row)[1]
        for row in config["models"]}
    create(
        config["joint_evidence_id"], tier="phase4-development",
        what=("Joint Bank W baseline capability model-set and common-family "
              "gate; no intervention outcome."),
        command=(
            "python -m jspace_phase4.experiments.p4_bank_w_capability "
            f"--config {config_path} --register-joint"),
        outputs=[report, png, pdf], inputs={
            "config": file_sha256(config_path), **inputs},
        baseline_capability_ready=bool(
            payload["p4p3_baseline_capability_ready"]),
        freeze_ready=False,
    )


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    if arguments.author_protocol:
        require_clean_tree()
        atomic_json(_protocol_path(config), author_protocol(config))
    elif arguments.register_protocol:
        register_protocol(config_path, config)
    elif arguments.model_slug:
        run_model(config_path, config, arguments.model_slug)
    elif arguments.aggregate_joint:
        aggregate_joint(config_path, config)
    elif arguments.register_joint:
        register_joint(config_path, config)
    else:  # pragma: no cover - argparse enforces one action
        raise RegistryError("Bank W capability action missing")


if __name__ == "__main__":
    main()
