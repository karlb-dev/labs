"""GPU producer for the consumed-development P4-P2 variance pilot.

The producer is deliberately unable to open untouched families.  Full
execution additionally requires a live canonical A1000 decision, a passing
one-family CUDA smoke, and an independently registered code-review artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jspace_part2.dictionaries import build_j_dictionaries
from jspace_phase3.bank import FactBundle

from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..mode_intervention import (
    ExactProfileModeAblator,
    accepted_alias_token_ids,
    answer_prediction_mask,
    combined_protection_sets,
    prediction_phase,
)
from ..paths4 import (
    figures_dir,
    materialize_local_file,
    metrics_dir,
    resolve_uri,
)
from ..provenance4 import Provenance4, write_result4
from ..registry4 import RegistryError, create, resolve
from .p4_qwen_lens_structural_stability import load_lens_checkpoint
from .p4_qwen_mode_model_gate import (
    _delimiter_spec,
    _load_bundles,
    _load_methods_gate,
    _load_selection,
    _render_prompt,
    generated_phase_ids,
    normalized_exact_alias,
)
from .p4_qwen_mode_variance_pilot import (
    _parse_cell_name,
    analyze_pilot_rows,
)
from .p4_qwen_nested_lens_fit import (
    model_reference,
    qwen_fused_kernel_contract,
    registered_output_check,
    verify_model_fused_bindings,
    verify_package_versions,
    verify_snapshot,
)
from ..phase_hooks import DelimiterSpec, Phase, classify_token_phases


class PilotExecutionBlocked(RuntimeError):
    """A prospective execution precondition is absent."""


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _registered_path(
        evidence_id: str, path: str | Path, expected: str) -> tuple[dict, Path]:
    event = resolve(evidence_id)
    if not event["live"]:
        raise RuntimeError(f"required evidence is not live: {evidence_id}")
    resolved = Path(path)
    if file_sha256(resolved) != expected:
        raise RuntimeError(f"registered artifact hash drift: {resolved}")
    rows = [
        row for row in event["outputs"]
        if (Path(row["path"]).resolve() == resolved.resolve()
            and row["sha256"] == expected)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"{evidence_id} does not pin {resolved} at {expected}")
    return event, resolved


def _registered_basename(evidence_id: str, basename: str,
                         expected: str | None = None) -> tuple[dict, Path, str]:
    event = resolve(evidence_id)
    if not event["live"]:
        raise RuntimeError(f"required evidence is not live: {evidence_id}")
    rows = [
        row for row in event["outputs"]
        if Path(row["path"]).name == basename
        and (expected is None or row["sha256"] == expected)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"{evidence_id} lacks one registered {basename}")
    path = Path(rows[0]["path"])
    digest = file_sha256(path)
    if digest != rows[0]["sha256"]:
        raise RuntimeError(f"registered output hash drift: {path}")
    return event, path, digest


def _load_envelope(path: Path) -> tuple[dict, dict]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"result envelope lacks payload: {path}")
    if envelope.get("payload_sha256") != object_sha256(payload):
        raise RuntimeError(f"result payload hash drift: {path}")
    return envelope, payload


def _protocol_and_baseline(config: Mapping) -> tuple[dict, dict, dict]:
    protocol_spec = config["registered_protocol"]
    protocol_config_path = resolve_uri(protocol_spec["config_uri"])
    protocol_result_path = resolve_uri(protocol_spec["result_uri"])
    if file_sha256(protocol_config_path) != protocol_spec["config_sha256"]:
        raise RuntimeError("registered mode-pilot protocol config drift")
    _registered_path(
        protocol_spec["evidence_id"], protocol_result_path,
        protocol_spec["result_sha256"])
    protocol = json.loads(protocol_result_path.read_text())
    if protocol.get("all_protocol_gates_pass") is not True:
        raise RuntimeError("registered mode-pilot protocol did not pass")

    baseline_spec = config["passing_mode_baseline"]
    baseline_config_path = resolve_uri(baseline_spec["config_uri"])
    if file_sha256(baseline_config_path) != baseline_spec["config_sha256"]:
        raise RuntimeError("passing mode-baseline config drift")
    _registered_path(
        baseline_spec["evidence_id"], baseline_spec["result_path"],
        baseline_spec["result_sha256"])
    _, baseline = _load_envelope(Path(baseline_spec["result_path"]))
    if baseline.get("all_model_backed_development_gates_pass") is not True:
        raise PilotExecutionBlocked("official mode-v2 baseline is not passing")
    baseline_config = yaml.safe_load(baseline_config_path.read_text())

    for field in (
            "model_uri", "model_snapshot_manifest_uri",
            "model_snapshot_manifest_sha256"):
        if config[field] != baseline_config[field]:
            raise RuntimeError(f"GPU producer {field} differs from baseline")
    baseline_runtime = baseline_config["runtime"]
    for field in (
            "packages", "qwen_kernel_modules",
            "expected_linear_attention_modules", "causal_conv1d_required"):
        if config["runtime"][field] != baseline_runtime[field]:
            raise RuntimeError(
                f"GPU producer runtime.{field} differs from baseline")
    baseline_selection = baseline_config["selection"]
    selection_pairs = {
        "evidence_id": "source_evidence_id",
        "manifest_uri": "source_uri",
        "manifest_sha256": "source_sha256",
        "payload_sha256": "source_payload_sha256",
        "subset_key": "subset_key",
        "expected_families": "expected_families",
        "expected_facts": "expected_facts",
    }
    for field, baseline_field in selection_pairs.items():
        if config["selection"][field] != baseline_selection[baseline_field]:
            raise RuntimeError(
                f"GPU producer selection.{field} differs from baseline")
    if list(config["task_banks"]) != list(baseline_config["task_banks"]):
        raise RuntimeError("GPU producer task-bank contract differs from baseline")

    generation = config["generation"]
    baseline_generation = baseline_config["protocol"]
    for field in (
            "mode_order", "phase_parser_version", "prompt_instruction",
            "requested_reasoning_token_cap", "max_new_tokens", "do_sample",
            "answer_rule"):
        if generation[field] != baseline_generation[field]:
            raise RuntimeError(
                f"GPU producer generation.{field} differs from baseline")
    if generation["prompt_variant"] != baseline_selection["prompt_variant"]:
        raise RuntimeError("GPU producer prompt variant differs from baseline")
    if generation["parse_failure_counts_as_incorrect"] is not True:
        raise RuntimeError("GPU producer must count parse failure as incorrect")
    if generation["do_sample"] is not False:
        raise RuntimeError("GPU producer decoding must be deterministic")

    if list(protocol["pilot"]["cell_order"]) != list(
            generation["cell_order"]):
        raise RuntimeError("GPU producer cell order differs from protocol")
    if list(protocol["pilot"]["interaction_coefficients"]) != list(
            generation["interaction_coefficients"]):
        raise RuntimeError("GPU producer interaction coefficients drift")
    if list(protocol["pilot"]["modes"]) != list(generation["mode_order"]):
        raise RuntimeError("GPU producer mode order differs from protocol")
    if list(protocol["pilot"]["arms"]) != list(generation["arm_order"]):
        raise RuntimeError("GPU producer arm order differs from protocol")
    if list(protocol["pilot"]["primary_phases"]) != [
            "prefill", "final_answer"]:
        raise RuntimeError("registered pilot common-phase contract drift")
    if protocol["selection"]["payload_sha256"] != config[
            "selection"]["payload_sha256"]:
        raise RuntimeError("GPU producer selection payload drift")
    for field in (
            "maximum_wrong_phase_hook_fires",
            "minimum_expected_phase_hook_fires_per_row",
            "require_zero_selected_protected_overlap",
            "require_exact_rank_match", "maximum_energy_relative_error"):
        if config["intervention"][field] != protocol[
                "mechanical_gates"][field]:
            raise RuntimeError(
                f"GPU producer mechanical gate {field} differs from protocol")
    for field in (
            "sample_sd_ddof", "bootstrap_draws", "bootstrap_seed",
            "bootstrap_upper_quantile", "planning_sd_rule"):
        if config["variance_summary"][field] != protocol[
                "variance_summary"][field]:
            raise RuntimeError(
                f"GPU producer variance rule {field} differs from protocol")
    sesoi_path = resolve_uri(config["variance_summary"]["sesoi_source_uri"])
    if file_sha256(sesoi_path) != config[
            "variance_summary"]["sesoi_source_sha256"]:
        raise RuntimeError("prospective P4-P2 SESOI memo hash drift")
    return protocol, baseline, baseline_config


def canonical_lens_binding(config: Mapping, *,
                           require_bound: bool) -> dict:
    specification = config["canonical_lens"]
    decision_hash = str(specification["decision_result_sha256"])
    lens_hash = str(specification["lens_sha256"])
    placeholders = [
        value for value in (decision_hash, lens_hash)
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value)
    ]
    if placeholders:
        if require_bound:
            raise PilotExecutionBlocked(
                "canonical A1000 decision/lens hashes are not bound")
        return {"bound": False, "placeholders": placeholders}
    decision_event, decision_path, observed_decision_hash = \
        _registered_basename(
            specification["decision_evidence_id"],
            "canonical_lens_decision.json", decision_hash)
    _, decision = _load_envelope(decision_path)
    if decision.get("branch") not in specification["permitted_branches"]:
        raise PilotExecutionBlocked(
            f"canonical branch {decision.get('branch')} blocks P4-P2")
    if decision.get("canonical_lens") != specification[
            "required_canonical_lens"]:
        raise PilotExecutionBlocked("canonical decision did not nominate A1000")
    if not str(decision.get("p4_p2_status", "")).startswith(
            "canonical-lens-precondition-passes"):
        raise PilotExecutionBlocked("canonical decision blocks P4-P2")
    lens_event = resolve(specification["lens_evidence_id"])
    if not lens_event["live"]:
        raise PilotExecutionBlocked("registered A1000 lens event is not live")
    lens_source = resolve_uri(specification["lens_uri"], must_exist=False)
    matching = [
        row for row in lens_event["outputs"]
        if Path(row["path"]).resolve() == lens_source.resolve()
        and row["sha256"] == lens_hash
    ]
    if len(matching) != 1:
        raise RuntimeError("A1000 path/hash is absent from its fit event")
    lens_path = materialize_local_file(
        specification["lens_uri"], expected_sha256=lens_hash)
    if decision.get("source_hashes", {}).get("a1000_lens") != lens_hash:
        raise RuntimeError("canonical decision/lens hash binding drift")
    return {
        "bound": True,
        "decision_evidence_id": decision_event["evidence_id"],
        "decision_result_path": str(decision_path),
        "decision_result_sha256": observed_decision_hash,
        "branch": decision["branch"],
        "p4_p2_status": decision["p4_p2_status"],
        "lens_evidence_id": lens_event["evidence_id"],
        "lens_path": str(lens_path),
        "lens_sha256": lens_hash,
    }


def independent_review_binding(
        config: Mapping, *, config_path: Path, producer_path: Path,
        require_review: bool) -> dict:
    specification = config["review_contract"]
    evidence_id = specification["independent_review_evidence_id"]
    try:
        event, path, digest = _registered_basename(
            evidence_id, specification["independent_review_output_basename"])
    except (RegistryError, RuntimeError):
        if require_review:
            raise PilotExecutionBlocked(
                "independent P4-P2 GPU producer review is absent") from None
        return {"complete": False, "evidence_id": evidence_id}
    envelope, review = _load_envelope(path)
    expected = {
        "reviewed_config_sha256": file_sha256(config_path),
        "reviewed_producer_sha256": file_sha256(producer_path),
        "reviewed_mode_intervention_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "mode_intervention.py"),
        "reviewed_gpu_test_sha256": file_sha256(
            Path(__file__).resolve().parents[2]
            / "tests/test_qwen_mode_variance_gpu.py"),
        "reviewed_mode_test_sha256": file_sha256(
            Path(__file__).resolve().parents[2]
            / "tests/test_mode_intervention.py"),
        "verdict": specification["required_verdict"],
        "reviewer_independent": True,
        "intervention_outcome_opened": False,
        "untouched_family_opened": False,
        "confirmatory_or_replication_outcome_opened": False,
    }
    expected.update({
        str(field): True
        for field in specification["required_boolean_findings"]
    })
    if any(review.get(key) != value for key, value in expected.items()):
        raise PilotExecutionBlocked("independent GPU review contract drift")
    if not str(review.get("reviewer_identity", "")).strip():
        raise PilotExecutionBlocked("independent GPU review lacks identity")
    if not str(review.get("review_completed_utc", "")).endswith("Z"):
        raise PilotExecutionBlocked("independent GPU review lacks UTC stamp")
    if envelope.get("provenance", {}).get("dirty_tree") is not False:
        raise PilotExecutionBlocked("independent review came from a dirty tree")
    return {
        "complete": True,
        "evidence_id": event["evidence_id"],
        "path": str(path),
        "sha256": digest,
        "review_payload_sha256": envelope["payload_sha256"],
    }


def preflight(config_path: Path, config: Mapping) -> dict:
    protocol, baseline, baseline_config = _protocol_and_baseline(config)
    lens = canonical_lens_binding(config, require_bound=False)
    review = independent_review_binding(
        config, config_path=config_path, producer_path=Path(__file__),
        require_review=False)
    facts, selection = _load_selection(baseline_config)
    bundles, banks = _load_bundles(baseline_config, facts)
    if len(facts) != 20 or len(bundles) != 20:
        raise RuntimeError("P4-P2 consumed-family selection drift")
    smoke_path = _smoke_path(config)
    return {
        "schema_version": 1,
        "protocol_pass": protocol["all_protocol_gates_pass"],
        "mode_baseline_pass": baseline[
            "all_model_backed_development_gates_pass"],
        "canonical_lens": lens,
        "independent_review": review,
        "selection": {
            **selection,
            "n_facts": len(facts),
            "n_families": len({row.canonical_family for row in bundles}),
        },
        "task_banks_sha256": object_sha256(banks),
        "smoke_present": smoke_path.is_file(),
        "full_execution_ready": bool(
            lens.get("bound") and review.get("complete")
            and smoke_path.is_file()),
        "untouched_families_opened": False,
    }


def _cache_signature(cache) -> dict:
    if cache is None:
        return {"type": None}
    result = {"type": type(cache).__name__}
    get_length = getattr(cache, "get_seq_length", None)
    if callable(get_length):
        result["sequence_length"] = int(get_length())
    layers = getattr(cache, "layers", None)
    if layers is not None:
        layer_rows = []
        for layer in layers:
            tensors = {}
            for name, value in vars(layer).items():
                if torch.is_tensor(value):
                    tensors[name] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
            layer_rows.append({
                "type": type(layer).__name__, "tensors": tensors})
        result["layers_sha256"] = object_sha256(layer_rows)
        result["n_layers"] = len(layer_rows)
    elif isinstance(cache, (tuple, list)):
        result["n_layers"] = len(cache)
        result["tensor_shapes"] = [
            [list(value.shape) for value in row if torch.is_tensor(value)]
            for row in cache
        ]
    return result


def _clean_sentinel(logits: torch.Tensor, prompt_ids: Sequence[int],
                    cache) -> dict:
    values, indices = logits[-1].float().topk(32)
    payload = {
        "prompt_token_ids_sha256": object_sha256(
            [int(value) for value in prompt_ids]),
        "top32_ids": [int(value) for value in indices.cpu().tolist()],
        "top32_logits_rounded_1e4": [
            round(float(value), 4) for value in values.cpu().tolist()],
        "cache_signature": _cache_signature(cache),
    }
    return {**payload, "sentinel_sha256": object_sha256(payload)}


def _forward(
        hf, ablator: ExactProfileModeAblator, *, input_ids: torch.Tensor,
        past_key_values, protection: torch.Tensor,
        active_mask: torch.Tensor, phase: str,
        position_phases: Sequence[str], forward_index: int,
        arm: str, dictionaries: Mapping[int, torch.Tensor], config: Mapping,
        item_id: str, condition: str):
    intervention = config["intervention"]
    ablator.configure(
        arm=arm, dictionaries=dictionaries,
        protection_sets=protection,
        active_position_mask=active_mask,
        target_phase=phase, position_phases=position_phases,
        forward_index=forward_index,
        k=int(intervention["k"]), evidence_id=config["evidence_id"],
        item_id=item_id, condition=condition,
        base_seed=int(intervention["matched_seed"]),
        energy_relative_floor=float(
            intervention["energy_relative_floor"]),
    )
    try:
        return hf(
            input_ids=input_ids, past_key_values=past_key_values,
            use_cache=True)
    finally:
        ablator.mode = None


@torch.no_grad()
def generate_intervened_tokens(
        hf, ablator: ExactProfileModeAblator, *,
        prompt_ids: Sequence[int], alias_ids: Sequence[int],
        delimiters: DelimiterSpec, mode: str, phase: str, arm: str,
        dictionaries: Mapping[int, torch.Tensor], config: Mapping,
        item_id: str) -> dict:
    """Generate one deterministic arm while keeping clean caches isolated."""
    device = next(hf.parameters()).device
    prompt = torch.tensor([list(prompt_ids)], device=device, dtype=torch.long)
    clean = hf(input_ids=prompt, use_cache=True)
    sentinel = _clean_sentinel(clean.logits[0], prompt_ids,
                               clean.past_key_values)
    top_k = int(config["intervention"]["clean_protect_top_k"])
    condition = f"{mode}|{phase}|{arm}"
    initial_phase = prediction_phase(
        prompt_ids, prompt_length=len(prompt_ids), delimiters=delimiters)
    if phase == Phase.PREFILL.value:
        protection = combined_protection_sets(
            clean.logits[0], alias_token_ids=alias_ids, top_k=top_k)
        actual = _forward(
            hf, ablator, input_ids=prompt, past_key_values=None,
            protection=protection,
            active_mask=torch.ones(len(prompt_ids), dtype=torch.bool,
                                   device=device),
            phase=phase,
            position_phases=[Phase.PREFILL.value] * len(prompt_ids),
            forward_index=0, arm=arm,
            dictionaries=dictionaries, config=config,
            item_id=item_id, condition=condition)
    elif initial_phase == Phase.FINAL_ANSWER.value:
        if len(prompt_ids) < 2:
            raise RuntimeError("final-answer prompt is too short to split")
        prefix = hf(input_ids=prompt[:, :-1], use_cache=True)
        protection = combined_protection_sets(
            clean.logits[0, -1:].contiguous(),
            alias_token_ids=alias_ids, top_k=top_k)
        actual = _forward(
            hf, ablator, input_ids=prompt[:, -1:],
            past_key_values=prefix.past_key_values,
            protection=protection,
            active_mask=torch.ones(1, dtype=torch.bool, device=device),
            phase=phase, position_phases=[initial_phase],
            forward_index=0, arm=arm,
            dictionaries=dictionaries, config=config,
            item_id=item_id, condition=condition)
    else:
        # A second clean forward owns a distinct, unmodified cache stream.
        actual = hf(input_ids=prompt, use_cache=True)
        difference = float((
            clean.logits[0, -1].float()
            - actual.logits[0, -1].float()).abs().max())
        if difference != 0.0:
            raise RuntimeError(
                f"deterministic prompt replay drifted by {difference}")

    clean_past = clean.past_key_values
    actual_past = actual.past_key_values
    actual_logits = actual.logits[0, -1]
    generated: list[int] = []
    eos_ids = set(int(value) for value in delimiters.eos_token_ids)
    max_new = int(config["generation"]["max_new_tokens"])
    for step in range(max_new):
        token = int(actual_logits.argmax())
        generated.append(token)
        if token in eos_ids:
            break
        token_tensor = torch.tensor([[token]], device=device, dtype=torch.long)
        clean_step = hf(
            input_ids=token_tensor, past_key_values=clean_past,
            use_cache=True)
        clean_past = clean_step.past_key_values
        current_phase = prediction_phase(
            [*prompt_ids, *generated], prompt_length=len(prompt_ids),
            delimiters=delimiters)
        if current_phase == phase:
            protection = combined_protection_sets(
                clean_step.logits[0], alias_token_ids=alias_ids,
                top_k=top_k)
            actual_step = _forward(
                hf, ablator, input_ids=token_tensor,
                past_key_values=actual_past,
                protection=protection,
                active_mask=torch.ones(1, dtype=torch.bool, device=device),
                phase=phase, position_phases=[current_phase],
                forward_index=step + 1, arm=arm,
                dictionaries=dictionaries, config=config,
                item_id=item_id, condition=condition)
        else:
            actual_step = hf(
                input_ids=token_tensor, past_key_values=actual_past,
                use_cache=True)
        actual_past = actual_step.past_key_values
        actual_logits = actual_step.logits[0, -1]
    return {
        "generated_token_ids": generated,
        "clean_sentinel": sentinel,
        "cache_identity_sha256": object_sha256(
            _cache_signature(clean.past_key_values)),
        "intervention_log": ablator.log,
    }


def _generated_context(
        full_ids: Sequence[int], *, prompt_length: int,
        parsed, delimiters: DelimiterSpec) -> list[int] | None:
    initial = prediction_phase(
        full_ids[:prompt_length], prompt_length=prompt_length,
        delimiters=delimiters)
    if initial == Phase.FINAL_ANSWER.value:
        return [int(value) for value in full_ids[:prompt_length]]
    if parsed.end_index is None:
        return None
    end = int(parsed.end_index) + len(delimiters.reasoning_end_ids)
    return [int(value) for value in full_ids[:end]]


@torch.no_grad()
def secondary_answer_lp(
        hf, ablator: ExactProfileModeAblator, *,
        context_ids: Sequence[int], answer_ids: Sequence[int],
        prompt_length: int, alias_ids: Sequence[int], phase: str, arm: str,
        dictionaries: Mapping[int, torch.Tensor], config: Mapping,
        item_id: str, condition: str) -> tuple[float, list[dict]]:
    device = next(hf.parameters()).device
    if not answer_ids:
        raise RuntimeError("secondary accepted answer tokenizes empty")
    sequence = [*context_ids, *answer_ids]
    ids = torch.tensor([sequence], device=device, dtype=torch.long)
    clean = hf(input_ids=ids, use_cache=False)
    protection = combined_protection_sets(
        clean.logits[0], alias_token_ids=alias_ids,
        top_k=int(config["intervention"]["clean_protect_top_k"]))
    if phase == Phase.PREFILL.value:
        active = torch.arange(len(sequence), device=device) < prompt_length
        position_phases = [
            Phase.PREFILL.value if position < prompt_length
            else Phase.REASONING.value
            for position in range(len(sequence))]
    else:
        active = answer_prediction_mask(
            sequence_length=len(sequence), context_length=len(context_ids),
            device=device)
        position_phases = [
            Phase.FINAL_ANSWER.value if bool(active[position])
            else Phase.REASONING.value
            for position in range(len(sequence))]
    output = _forward(
        hf, ablator, input_ids=ids, past_key_values=None,
        protection=protection, active_mask=active,
        phase=phase, position_phases=position_phases,
        forward_index=0, arm=arm,
        dictionaries=dictionaries, config=config,
        item_id=item_id, condition=f"{condition}|secondary-lp")
    logits = output.logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    start = len(context_ids) - 1
    values = [
        log_probs[start + offset, int(token)]
        for offset, token in enumerate(answer_ids)
    ]
    return float(torch.stack(values).sum()), ablator.log.rows()


def _enforce_intervention_summary(
        summary: Mapping, config: Mapping, *, endpoint: str) -> None:
    contract = config["intervention"]
    wrong = int(summary.get("wrong_phase_hook_fires", 0))
    fires = int(summary.get("hook_fires", {}).get(
        str(summary.get("target_phase", "")), 0))
    # Logs do not need to duplicate their target; take the only primary phase
    # with a nonzero count when the caller did not add it to the summary.
    if fires == 0:
        fires = max(
            [int(value) for value in summary.get("hook_fires", {}).values()]
            or [0])
    overlap = int(summary.get("maximum_selected_protected_overlap", 0))
    rank_exact = bool(summary.get("rank_match_exact", False))
    energy = float(summary.get("maximum_energy_relative_error", math.inf))
    protected_cosine = float(summary.get("maximum_protected_cosine", math.inf))
    failures = []
    if wrong > int(contract["maximum_wrong_phase_hook_fires"]):
        failures.append(f"wrong_phase_hook_fires={wrong}")
    if fires < int(contract["minimum_expected_phase_hook_fires_per_row"]):
        failures.append(f"expected_phase_hook_fires={fires}")
    if contract["require_zero_selected_protected_overlap"] and overlap:
        failures.append(f"selected_protected_overlap={overlap}")
    if contract["require_exact_rank_match"] and not rank_exact:
        failures.append("rank_match_exact=false")
    if energy > float(contract["maximum_energy_relative_error"]):
        failures.append(f"energy_relative_error={energy}")
    if protected_cosine > float(contract["maximum_protected_cosine"]):
        failures.append(f"maximum_protected_cosine={protected_cosine}")
    if failures:
        raise RuntimeError(
            f"{endpoint} intervention mechanics failed: " + ", ".join(failures))


def _validate_profile_rows(rows: Sequence[Mapping]) -> None:
    seen = set()
    for row in rows:
        key = (
            str(row["endpoint"]), int(row["layer"]),
            int(row["forward_index"]), int(row["position"]),
        )
        if key in seen:
            raise RuntimeError(f"duplicate intervention profile site: {key}")
        seen.add(key)
        if int(row["requested_rank"]) != int(row["span_safe_effective_rank"]):
            raise RuntimeError("profile requested/span-safe rank drift")
        if int(row["lost_rank"]) != max(
                int(row["selected_effective_rank"])
                - int(row["span_safe_effective_rank"]), 0):
            raise RuntimeError("profile lost-rank arithmetic drift")


@torch.no_grad()
def run_pilot_cell(
        hf, tokenizer, ablator: ExactProfileModeAblator, *,
        bundle: FactBundle, mode: str, phase: str, arm: str,
        methods: Mapping, delimiters: DelimiterSpec,
        dictionaries: Mapping[int, torch.Tensor], config: Mapping,
        baseline_sentinel: Mapping | None) -> tuple[dict, list[dict], dict]:
    generation = config["generation"]
    content = (
        bundle.prompts[generation["prompt_variant"]].rstrip() + "\n\n"
        + generation["prompt_instruction"])
    rendered = _render_prompt(
        tokenizer, content, enable_thinking=mode == "thinking_on")
    prompt_ids = rendered["token_ids"]
    prompt_parse = classify_token_phases(
        prompt_ids, prompt_length=len(prompt_ids), delimiters=delimiters)
    if bool(prompt_parse.reasoning_open_at_generation) != (
            mode == "thinking_on"):
        raise RuntimeError(f"official prompt phase drift for {mode}")
    if phase not in {Phase.PREFILL.value, Phase.FINAL_ANSWER.value}:
        raise RuntimeError(f"unsupported P4-P2 phase: {phase}")
    if arm not in ExactProfileModeAblator.ALLOWED_ARMS:
        raise RuntimeError(f"unsupported P4-P2 arm: {arm}")
    aliases = list(bundle.accepted_answers)
    alias_ids = accepted_alias_token_ids(tokenizer, aliases)
    ablator.reset()
    started = time.time()
    generated = generate_intervened_tokens(
        hf, ablator, prompt_ids=prompt_ids, alias_ids=alias_ids,
        delimiters=delimiters, mode=mode, phase=phase, arm=arm,
        dictionaries=dictionaries, config=config, item_id=bundle.fact_id)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    sentinel = generated["clean_sentinel"]
    if baseline_sentinel is not None and sentinel != dict(baseline_sentinel):
        raise RuntimeError("clean deterministic replay sentinel drift")
    generation_summary = generated["intervention_log"].summary()
    _enforce_intervention_summary(
        generation_summary, config, endpoint="generation")
    generation_profile = [
        {"endpoint": "generation", **row}
        for row in generated["intervention_log"].rows()]
    generated_ids = generated["generated_token_ids"]
    full_ids = [*prompt_ids, *generated_ids]
    parsed = classify_token_phases(
        full_ids, prompt_length=len(prompt_ids), delimiters=delimiters)
    phase_ids = generated_phase_ids(
        full_ids, prompt_length=len(prompt_ids), parsed=parsed,
        delimiters=delimiters)
    final_text = tokenizer.decode(
        phase_ids[Phase.FINAL_ANSWER.value],
        skip_special_tokens=True).strip()
    matched_alias = normalized_exact_alias(final_text, aliases)
    eos = bool(generated_ids and generated_ids[-1] in set(
        delimiters.eos_token_ids))
    truncated = bool(not eos and len(generated_ids) >= int(
        generation["max_new_tokens"]))
    stop_reason = "eos" if eos else "length" if truncated else "error"
    correct = bool(
        parsed.valid and eos and not truncated and matched_alias is not None)
    context = _generated_context(
        full_ids, prompt_length=len(prompt_ids), parsed=parsed,
        delimiters=delimiters)
    secondary_lp = None
    lp_profile: list[dict] = []
    lp_summary = None
    if context is not None and config["secondary_answer_lp"]["enabled"]:
        answer_ids = tokenizer(
            aliases[0], add_special_tokens=False).input_ids
        ablator.reset()
        secondary_lp, lp_profile = secondary_answer_lp(
            hf, ablator, context_ids=context, answer_ids=answer_ids,
            prompt_length=len(prompt_ids), alias_ids=alias_ids,
            phase=phase, arm=arm, dictionaries=dictionaries,
            config=config, item_id=bundle.fact_id,
            condition=f"{mode}|{phase}|{arm}")
        lp_summary = ablator.log.summary()
        _enforce_intervention_summary(
            lp_summary, config, endpoint="teacher-forced answer LP")
        lp_profile = [
            {"endpoint": "teacher_forced_answer_lp", **row}
            for row in lp_profile]
    profile = [*generation_profile, *lp_profile]
    _validate_profile_rows(profile)
    requested = int(generation_summary.get("requested_rank_total", 0))
    delivered = int(generation_summary.get("delivered_rank_total", 0))
    row = {
        "fact_id": bundle.fact_id,
        "canonical_family": bundle.canonical_family,
        "bank": bundle.bank,
        "mode": mode,
        "enable_thinking": mode == "thinking_on",
        "phase": phase,
        "arm": arm,
        "prompt_variant": generation["prompt_variant"],
        "rendered_prompt_text": rendered["text"],
        "prompt_text_sha256": rendered["text_sha256"],
        "prompt_token_ids_json": json.dumps(prompt_ids),
        "prompt_token_ids_sha256": rendered["token_ids_sha256"],
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated_ids),
        "reasoning_content_tokens": len(
            phase_ids[Phase.REASONING.value]),
        "final_answer_tokens": len(
            phase_ids[Phase.FINAL_ANSWER.value]),
        "parse_valid": bool(parsed.valid),
        "parse_errors_json": json.dumps(list(parsed.errors)),
        "generated_token_phases_json": json.dumps(
            list(parsed.phases[len(prompt_ids):])),
        "reasoning_open_at_generation": bool(
            parsed.reasoning_open_at_generation),
        "stop_reason": stop_reason,
        "truncated": truncated,
        "matched_alias": matched_alias,
        "correct": correct,
        "final_answer_text": final_text,
        "generated_token_ids_json": json.dumps(generated_ids),
        "raw_completion_sha256": object_sha256(generated_ids),
        "accepted_answers_json": json.dumps(aliases, ensure_ascii=False),
        "accepted_alias_token_ids_sha256": object_sha256(alias_ids),
        "original_answer_sequence_lp": secondary_lp,
        "expected_phase_hook_fires": int(
            generation_summary.get("hook_fires", {}).get(phase, 0)),
        "wrong_phase_hook_fires": int(
            generation_summary.get("wrong_phase_hook_fires", 0)),
        "selected_protected_overlap": int(generation_summary.get(
            "maximum_selected_protected_overlap", 0)),
        "requested_rank": requested,
        "delivered_rank": delivered,
        "selected_effective_rank": int(generation_summary.get(
            "selected_effective_rank_total", 0)),
        "span_safe_effective_rank": int(generation_summary.get(
            "span_safe_effective_rank_total", 0)),
        "lost_rank": int(generation_summary.get("lost_rank_total", 0)),
        "rank_match_exact": bool(generation_summary.get(
            "rank_match_exact", False)),
        "energy_relative_error": float(generation_summary.get(
            "maximum_energy_relative_error", 0.0)),
        "maximum_protected_cosine": float(generation_summary.get(
            "maximum_protected_cosine", 0.0)),
        "control_clamped_positions": int(generation_summary.get(
            "control_clamped_positions", 0)),
        "generation_profile_positions": int(generation_summary.get(
            "n_positions", 0)),
        "teacher_forced_profile_positions": len(lp_profile),
        "teacher_forced_mechanics_json": json.dumps(
            lp_summary, sort_keys=True) if lp_summary is not None else None,
        "clean_replay_sentinel_sha256": sentinel["sentinel_sha256"],
        "baseline_cache_identity_sha256": generated[
            "cache_identity_sha256"],
        "elapsed_seconds": round(elapsed, 3),
        "parser_version": methods["parser"]["version"],
    }
    return row, profile, sentinel


def _safe_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_"
                   for character in value)


def _expected_cells(config: Mapping, bundles: Sequence[FactBundle]):
    pilot = {
        "modes": config["generation"]["mode_order"],
        "primary_phases": ["prefill", "final_answer"],
        "arms": config["generation"]["arm_order"],
    }
    rows = []
    for bundle in bundles:
        for cell in config["generation"]["cell_order"]:
            mode, phase, arm = _parse_cell_name(
                cell, pilot["modes"], pilot["primary_phases"],
                pilot["arms"])
            key = f"{bundle.fact_id}|{mode}|{phase}|{arm}"
            rows.append((key, bundle, mode, phase, arm))
    keys = [row[0] for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate P4-P2 expected row key")
    return rows


def _validate_clean_sentinel(value: Mapping) -> None:
    if set(value) != {
            "prompt_token_ids_sha256", "top32_ids",
            "top32_logits_rounded_1e4", "cache_signature",
            "sentinel_sha256"}:
        raise RuntimeError("clean replay sentinel schema drift")
    payload = {
        key: value[key] for key in value if key != "sentinel_sha256"}
    if value["sentinel_sha256"] != object_sha256(payload):
        raise RuntimeError("clean replay sentinel hash drift")


def _state(path: Path, header: Mapping) -> dict:
    if not path.exists():
        result = {
            "schema_version": 1,
            "header": dict(header),
            "rows": {},
            "baseline_sentinels": {},
            "runtime": None,
        }
        atomic_json(path, result)
        return result
    result = json.loads(path.read_text())
    if result.get("header") != dict(header):
        raise RuntimeError("refusing incompatible P4-P2 pilot resume")
    sentinels = result.get("baseline_sentinels", {})
    for value in sentinels.values():
        _validate_clean_sentinel(value)
    expected_sentinel_keys = set()
    for key, record in result.get("rows", {}).items():
        if record.get("row_sha256") != object_sha256(record.get("row")):
            raise RuntimeError(f"pilot state row hash drift: {key}")
        row = record["row"]
        sentinel_key = f"{row['fact_id']}|{row['mode']}"
        expected_sentinel_keys.add(sentinel_key)
        if sentinel_key not in sentinels:
            raise RuntimeError(f"pilot state lacks replay sentinel: {key}")
        if row.get("clean_replay_sentinel_sha256") != sentinels[
                sentinel_key]["sentinel_sha256"]:
            raise RuntimeError(f"pilot row/sentinel hash drift: {key}")
        profile = Path(record["profile_part_path"])
        if file_sha256(profile) != record["profile_part_sha256"]:
            raise RuntimeError(f"pilot profile-part hash drift: {key}")
    if set(sentinels) != expected_sentinel_keys:
        raise RuntimeError("pilot state has orphan clean replay sentinel")
    return result


def _runtime(
        config: Mapping, lens: Mapping, methods: Mapping):
    gpu = require_cuda_gpu()
    packages = verify_package_versions(config["runtime"]["packages"])
    fused_runtime = qwen_fused_kernel_contract(config["runtime"])
    model_path = resolve_uri(config["model_uri"])
    snapshot_path = resolve_uri(config["model_snapshot_manifest_uri"])
    if file_sha256(snapshot_path) != config[
            "model_snapshot_manifest_sha256"]:
        raise RuntimeError("Qwen model snapshot manifest hash drift")
    snapshot_manifest = json.loads(snapshot_path.read_text())
    snapshot = verify_snapshot(model_path, snapshot_manifest)
    import transformers
    import jlens
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
    if type(tokenizer).__name__ != methods[
            "official_template"]["tokenizer_class"]:
        raise RuntimeError("official Qwen tokenizer class drift")
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    if template_hash != methods[
            "official_template"]["chat_template_sha256"]:
        raise RuntimeError("official Qwen chat template hash drift")
    delimiters = _delimiter_spec(methods["parser"])
    if delimiters.version != config["generation"]["phase_parser_version"]:
        raise RuntimeError("mode phase-parser version drift")
    torch.cuda.reset_peak_memory_stats()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.bfloat16).to("cuda").eval()
    assert_model_on_cuda(hf)
    embeddings = hf.get_output_embeddings().weight
    if list(embeddings.shape) != [
            int(config["runtime"]["expected_vocab_size"]),
            int(config["runtime"]["expected_d_model"])]:
        raise RuntimeError("Qwen output-embedding shape drift")
    fused_model = verify_model_fused_bindings(hf, config["runtime"])
    wrapped = jlens.from_hf(hf, tokenizer)
    checkpoint = load_lens_checkpoint(
        Path(lens["lens_path"]), config["canonical_lens"],
        config["runtime"])
    lens_object = SimpleNamespace(jacobians=checkpoint["J"])
    band = [int(value) for value in config["intervention"]["band"]]
    if band != sorted(set(band)) or not band \
            or band[-1] >= int(config["runtime"]["expected_source_layers"]):
        raise RuntimeError("P4-P2 intervention band is invalid")
    if config["runtime"]["dictionary_dtype"] != "float16":
        raise RuntimeError("P4-P2 dictionary dtype must remain float16")
    dictionaries = build_j_dictionaries(
        hf, lens_object, band, dtype=torch.float16)
    del checkpoint, lens_object
    gc.collect()
    torch.cuda.empty_cache()
    contract = {
        "gpu": gpu,
        "packages": packages,
        "fused_runtime": fused_runtime,
        "fused_model": fused_model,
        "snapshot_inventory_sha256": snapshot["inventory_sha256"],
        "snapshot_manifest_sha256": file_sha256(snapshot_path),
        "chat_template_sha256": template_hash,
    }
    return hf, wrapped, tokenizer, delimiters, dictionaries, contract


def _output_dir(config: Mapping) -> Path:
    return (metrics_dir(config["slug"]) / "mode_variance_pilot"
            / config["evidence_id"])


def _smoke_path(config: Mapping) -> Path:
    return _output_dir(config) / "smoke" / "one_family_smoke.json"


def _smoke_header(config_path: Path, config: Mapping, lens: Mapping) -> dict:
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "mode_intervention_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "mode_intervention.py"),
        "lens_sha256": lens["lens_sha256"],
        "selection_payload_sha256": config["selection"]["payload_sha256"],
        "cell_order_sha256": object_sha256(
            config["generation"]["cell_order"]),
    }


def verify_smoke(config_path: Path, config: Mapping,
                 lens: Mapping) -> dict:
    path = _smoke_path(config)
    if not path.is_file():
        raise PilotExecutionBlocked("one-family CUDA smoke is absent")
    envelope, payload = _load_envelope(path)
    if payload.get("header") != _smoke_header(config_path, config, lens):
        raise PilotExecutionBlocked("one-family CUDA smoke header drift")
    if payload.get("passed") is not True or payload.get("n_rows") != 8:
        raise PilotExecutionBlocked("one-family CUDA smoke did not pass")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "payload_sha256": envelope["payload_sha256"],
    }


@torch.no_grad()
def run_smoke(config_path: Path, config: Mapping) -> None:
    require_clean_tree()
    protocol, _baseline, baseline_config = _protocol_and_baseline(config)
    lens = canonical_lens_binding(config, require_bound=True)
    methods, _ = _load_methods_gate(baseline_config)
    facts, _ = _load_selection(baseline_config)
    bundles, _ = _load_bundles(baseline_config, facts)
    bundle = bundles[0]
    hf, wrapped, tokenizer, delimiters, dictionaries, runtime = _runtime(
        config, lens, methods)
    ablator = ExactProfileModeAblator(
        wrapped.layers, config["intervention"]["band"])
    rows, profiles = [], []
    sentinels = {}
    with ablator:
        for key, _bundle, mode, phase, arm in _expected_cells(
                config, [bundle]):
            sentinel_key = f"{bundle.fact_id}|{mode}"
            row, profile, sentinel = run_pilot_cell(
                hf, tokenizer, ablator, bundle=bundle, mode=mode,
                phase=phase, arm=arm, methods=methods,
                delimiters=delimiters, dictionaries=dictionaries,
                config=config, baseline_sentinel=sentinels.get(sentinel_key))
            sentinels.setdefault(sentinel_key, sentinel)
            rows.append(row)
            profiles.extend({"row_key": key, **value} for value in profile)
    smoke_protocol = json.loads(json.dumps(protocol))
    smoke_protocol["selection"]["n_families"] = 1
    smoke_protocol["variance_summary"]["sample_sd_ddof"] = 0
    analysis = analyze_pilot_rows(rows, smoke_protocol)
    passed = bool(
        analysis["pilot_analysis_valid"]
        and len(rows) == 8
        and all(row["stop_reason"] in {"eos", "length"} for row in rows))
    payload = {
        "schema_version": 1,
        "header": _smoke_header(config_path, config, lens),
        "fact_id": bundle.fact_id,
        "canonical_family": bundle.canonical_family,
        "n_rows": len(rows),
        "n_profile_positions": len(profiles),
        "row_hashes": [object_sha256(row) for row in rows],
        "mechanical_gate_checks": analysis["mechanical_gate_checks"],
        "passed": passed,
        "runtime": runtime,
        "untouched_families_opened": False,
    }
    path = _smoke_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
    }
    atomic_json(path, envelope)
    del dictionaries, hf
    torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError("one-family CUDA smoke failed")
    print(json.dumps({
        "status": "PASS",
        "path": str(path),
        "fact_id": bundle.fact_id,
        "n_rows": len(rows),
    }, indent=1))


def _analysis_extensions(rows: Sequence[Mapping], protocol: Mapping) -> dict:
    families = sorted({str(row["canonical_family"]) for row in rows})
    cells = list(protocol["pilot"]["cell_order"])
    modes = protocol["pilot"]["modes"]
    phases = protocol["pilot"]["primary_phases"]
    arms = protocol["pilot"]["arms"]
    by_key = {
        (str(row["canonical_family"]), str(row["mode"]),
         str(row["phase"]), str(row["arm"])): row
        for row in rows
    }
    accuracy_matrix, lp_matrix = [], []
    lp_complete_families = []
    for family in families:
        accuracy_row, lp_row = [], []
        lp_complete = True
        for cell in cells:
            mode, phase, arm = _parse_cell_name(
                cell, modes, phases, arms)
            row = by_key[(family, mode, phase, arm)]
            accuracy_row.append(float(bool(row["correct"])))
            value = row.get("original_answer_sequence_lp")
            if value is None or not math.isfinite(float(value)):
                lp_complete = False
            lp_row.append(float(value) if value is not None else np.nan)
        accuracy_matrix.append(accuracy_row)
        if lp_complete:
            lp_matrix.append(lp_row)
            lp_complete_families.append(family)
    coefficients = np.asarray(
        protocol["pilot"]["interaction_coefficients"], dtype=np.float64)
    accuracy = np.asarray(accuracy_matrix, dtype=np.float64)
    interactions = accuracy @ coefficients
    lp_interactions = (
        np.asarray(lp_matrix, dtype=np.float64) @ coefficients
        if lp_matrix else np.asarray([], dtype=np.float64))

    def correlation(matrix: np.ndarray) -> list[list[float | None]]:
        output = []
        for left in range(matrix.shape[1]):
            row = []
            for right in range(matrix.shape[1]):
                if np.std(matrix[:, left]) == 0 \
                        or np.std(matrix[:, right]) == 0:
                    row.append(None)
                else:
                    row.append(float(np.corrcoef(
                        matrix[:, left], matrix[:, right])[0, 1]))
            output.append(row)
        return output

    return {
        "accuracy_interaction_atom_masses": {
            str(value): int(count)
            for value, count in sorted(Counter(
                float(value) for value in interactions).items())
        },
        "eight_cell_accuracy_correlation": correlation(accuracy),
        "cell_order": cells,
        "continuous_answer_lp_interaction": {
            "n_complete_families": len(lp_complete_families),
            "complete_family_ids_sha256": object_sha256(
                lp_complete_families),
            "mean": (float(lp_interactions.mean())
                     if len(lp_interactions) else None),
            "sample_sd": (float(np.std(lp_interactions, ddof=1))
                          if len(lp_interactions) > 1 else None),
            "family_values": [float(value) for value in lp_interactions],
            "status": "named design sensitivity; cannot replace accuracy post hoc",
        },
    }


def _plot(analysis: Mapping, *, png: Path, pdf: Path) -> None:
    interactions = np.asarray(
        analysis["family_interactions"], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    values, counts = np.unique(interactions, return_counts=True)
    axes[0].bar(values, counts, width=0.18, color="#0072B2")
    axes[0].set_xlabel("family accuracy interaction")
    axes[0].set_ylabel("families")
    axes[0].set_title("A · Discrete pilot support", loc="left")
    axes[1].bar(
        ["sample SD", "bootstrap\n90% upper", "planning SD"],
        [analysis["family_interaction_sample_sd"],
         analysis["family_interaction_bootstrap_sd_upper"],
         analysis["planning_family_sd"]],
        color=["#56B4E9", "#E69F00", "#D55E00"])
    axes[1].set_ylabel("accuracy points")
    axes[1].set_title("B · Variance-only planning ruler", loc="left")
    figure.suptitle(
        "P4-P2 consumed-development variance pilot\n"
        "Pilot mean cannot select the pre-stated SESOI", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def run_full(config_path: Path, config: Mapping) -> None:  # noqa: C901
    existing = registered_output_check(config["evidence_id"])
    if existing is not None:
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return
    clean = require_clean_tree()
    protocol, _baseline, baseline_config = _protocol_and_baseline(config)
    lens = canonical_lens_binding(config, require_bound=True)
    review = independent_review_binding(
        config, config_path=config_path, producer_path=Path(__file__),
        require_review=True)
    smoke = verify_smoke(config_path, config, lens)
    methods, methods_contract = _load_methods_gate(baseline_config)
    facts, selection = _load_selection(baseline_config)
    bundles, banks = _load_bundles(baseline_config, facts)
    expected = _expected_cells(config, bundles)
    if len(expected) != 160:
        raise RuntimeError("P4-P2 pilot must contain exactly 160 rows")
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    parts_dir = output_dir / "profile_parts"
    header = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "mode_intervention_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "mode_intervention.py"),
        "protocol_result_sha256": config[
            "registered_protocol"]["result_sha256"],
        "mode_baseline_result_sha256": config[
            "passing_mode_baseline"]["result_sha256"],
        "selection_payload_sha256": selection["payload_sha256"],
        "selected_fact_ids_sha256": object_sha256(facts),
        "task_banks_sha256": object_sha256(banks),
        "canonical_lens": lens,
        "independent_review": review,
        "smoke": smoke,
        "generation_sha256": object_sha256(config["generation"]),
        "intervention_sha256": object_sha256(config["intervention"]),
        "variance_summary_sha256": object_sha256(
            config["variance_summary"]),
    }
    state = _state(state_path, header)
    expected_keys = [row[0] for row in expected]
    unexpected = set(state["rows"]) - set(expected_keys)
    if unexpected:
        raise RuntimeError(f"pilot state has unexpected rows: {unexpected}")
    hf = wrapped = tokenizer = delimiters = dictionaries = runtime = None
    incomplete = [row for row in expected if row[0] not in state["rows"]]
    if incomplete:
        hf, wrapped, tokenizer, delimiters, dictionaries, runtime = _runtime(
            config, lens, methods)
        state["runtime"] = runtime
        atomic_json(state_path, state)
        ablator = ExactProfileModeAblator(
            wrapped.layers, config["intervention"]["band"])
        with ablator:
            for ordinal, (key, bundle, mode, phase, arm) in enumerate(
                    expected, start=1):
                if key in state["rows"]:
                    continue
                sentinel_key = f"{bundle.fact_id}|{mode}"
                row, profile, sentinel = run_pilot_cell(
                    hf, tokenizer, ablator, bundle=bundle, mode=mode,
                    phase=phase, arm=arm, methods=methods,
                    delimiters=delimiters, dictionaries=dictionaries,
                    config=config,
                    baseline_sentinel=state["baseline_sentinels"].get(
                        sentinel_key))
                state["baseline_sentinels"].setdefault(
                    sentinel_key, sentinel)
                profile_rows = [
                    {
                        "row_key": key,
                        "fact_id": bundle.fact_id,
                        "canonical_family": bundle.canonical_family,
                        "mode": mode, "phase": phase, "arm": arm,
                        **record,
                    }
                    for record in profile
                ]
                part = parts_dir / f"{ordinal:03d}_{_safe_key(key)}.parquet"
                _atomic_parquet(part, pd.DataFrame(profile_rows))
                state["rows"][key] = {
                    "row": row,
                    "row_sha256": object_sha256(row),
                    "profile_part_path": str(part),
                    "profile_part_sha256": file_sha256(part),
                    "n_profile_rows": len(profile_rows),
                }
                atomic_json(state_path, state)
                print(
                    f"{len(state['rows'])}/160 {key} "
                    f"parse={row['parse_valid']} correct={row['correct']} "
                    f"hooks={row['expected_phase_hook_fires']} "
                    f"tokens={row['generated_tokens']} "
                    f"{row['elapsed_seconds']:.1f}s", flush=True)
        state["runtime"]["peak_vram_bytes"] = int(
            torch.cuda.max_memory_allocated())
        atomic_json(state_path, state)
        del dictionaries, hf
        torch.cuda.empty_cache()
    elif state.get("runtime") is None:
        raise RuntimeError("complete P4-P2 state lacks runtime provenance")
    if set(state["rows"]) != set(expected_keys):
        raise RuntimeError("P4-P2 pilot ended with incomplete row grid")

    rows = [state["rows"][key]["row"] for key in expected_keys]
    analysis = analyze_pilot_rows(rows, protocol)
    analysis.update(_analysis_extensions(rows, protocol))
    analysis.update({
        "evidence_id": config["evidence_id"],
        "substantive_sesoi_accuracy_points": float(
            config["variance_summary"][
                "substantive_sesoi_accuracy_points"]),
        "sesoi_fixed_before_pilot": True,
        "pilot_mean_used_for_sesoi_selection": False,
        "canonical_lens": lens,
        "independent_review": review,
        "claim_boundary": config["claim_boundary"],
        "freeze_ready": False,
    })
    output_rows = output_dir / "mode_variance_pilot_rows.parquet"
    output_profiles = output_dir / "intervention_profile_rows.parquet"
    profile_manifest_path = output_dir / "profile_part_manifest.json"
    manifest_path = output_dir / "input_manifest.json"
    result_path = output_dir / "mode_variance_pilot_result.json"
    stop_path = output_dir / "stop_record.json"
    profile_frames = [
        pd.read_parquet(state["rows"][key]["profile_part_path"])
        for key in expected_keys
    ]
    _atomic_parquet(output_rows, pd.DataFrame(rows))
    _atomic_parquet(output_profiles, pd.concat(profile_frames,
                                               ignore_index=True))
    profile_manifest = {
        "schema_version": 1,
        "parts": [
            {
                "row_key": key,
                "path": state["rows"][key]["profile_part_path"],
                "sha256": state["rows"][key]["profile_part_sha256"],
                "n_rows": state["rows"][key]["n_profile_rows"],
            }
            for key in expected_keys
        ],
    }
    profile_manifest["parts_sha256"] = object_sha256(
        profile_manifest["parts"])
    atomic_json(profile_manifest_path, profile_manifest)
    manifest_payload = {
        "schema_version": 1,
        "header": header,
        "runtime": state["runtime"],
        "methods_gate": methods_contract,
        "selection": selection,
        "task_banks": banks,
        "protocol": protocol,
        "generation": config["generation"],
        "intervention": config["intervention"],
        "secondary_answer_lp": config["secondary_answer_lp"],
        "variance_summary": config["variance_summary"],
        "row_table_sha256": file_sha256(output_rows),
        "profile_table_sha256": file_sha256(output_profiles),
        "profile_manifest_sha256": file_sha256(profile_manifest_path),
    }
    manifest = {
        "schema_version": 1,
        "payload": manifest_payload,
        "payload_sha256": object_sha256(manifest_payload),
    }
    atomic_json(manifest_path, manifest)
    command = (
        "python -m jspace_phase4.experiments.p4_qwen_mode_variance_gpu "
        f"--config {config_path} --run")
    inputs = {
        "input_manifest": manifest["payload_sha256"],
        "canonical_decision": lens["decision_result_sha256"],
        "canonical_lens": lens["lens_sha256"],
        "mode_baseline": config[
            "passing_mode_baseline"]["result_sha256"],
        "pilot_protocol": config[
            "registered_protocol"]["result_sha256"],
        "selection": selection["payload_sha256"],
        "independent_review": review["sha256"],
        "cuda_smoke": smoke["sha256"],
    }
    write_result4(
        analysis, result_path,
        Provenance4(
            evidence_id=config["evidence_id"], tier=config["tier"],
            command=command, inputs=inputs,
            input_manifest_sha256=manifest["payload_sha256"],
            model=model_reference(config["model_uri"]),
            seed_contract=(
                "fixed consumed 20-family order; exact eight-cell order; "
                "stable SHA-256 exact-profile matched-control seeds"),
        ))
    stop = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "n_rows": len(rows),
        "n_eos": sum(row["stop_reason"] == "eos" for row in rows),
        "n_length": sum(row["stop_reason"] == "length" for row in rows),
        "n_error": sum(row["stop_reason"] == "error" for row in rows),
        "n_parse_failures": sum(not row["parse_valid"] for row in rows),
        "complete_grid": len(rows) == 160,
        "pilot_analysis_valid": analysis["pilot_analysis_valid"],
        "untouched_families_opened": False,
        "confirmatory_or_replication_outcomes_opened": False,
    }
    atomic_json(stop_path, stop)
    stem = config["outputs"]["figure_stem"]
    png = figures_dir() / f"{stem}.png"
    pdf = figures_dir() / f"{stem}.pdf"
    _plot(analysis, png=png, pdf=pdf)
    outputs = [
        result_path, manifest_path, state_path, output_rows,
        output_profiles, profile_manifest_path, stop_path,
        Path(smoke["path"]), png, pdf,
    ]
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            "Consumed-development P4-P2 official-mode by phase variance "
            "pilot over the exact 160-row grid; variance calibration only, "
            "no untouched intervention outcome."),
        command=command, outputs=outputs, inputs=inputs,
        interventions_opened=True,
        intervention_tier="consumed-development-only",
        untouched_families_opened=False,
        confirmatory_or_replication_outcomes_opened=False,
        freeze_ready=False,
    )
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "n_rows": len(rows),
        "pilot_analysis_valid": analysis["pilot_analysis_valid"],
        "planning_family_sd": analysis["planning_family_sd"],
        "result": str(result_path),
        "figure": str(png),
    }, indent=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--smoke-one-family", action="store_true")
    action.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("tier") != "phase4-development":
        raise RuntimeError("P4-P2 variance pilot must be development tier")
    if arguments.preflight:
        print(json.dumps(preflight(config_path, config), indent=1))
    elif arguments.smoke_one_family:
        run_smoke(config_path, config)
    else:
        run_full(config_path, config)


if __name__ == "__main__":
    main()
