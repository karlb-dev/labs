"""Model-backed development gate for the official Qwen thinking toggle.

This producer consumes only the already-released Phase 3 development banks
and an outcome-blind, registered 20-family subset.  It validates the official
template/parser contract against real deterministic completions before any
Phase 4 confirmatory or replication family can be opened.  It does not run an
intervention or estimate the P4-P2 primary effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jspace_phase3.bank import FactBundle, load_bank

from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import figures_dir, metrics_dir, resolve_uri
from ..phase_hooks import DelimiterSpec, Phase, classify_token_phases
from ..provenance4 import Provenance4, write_result4
from ..registry4 import create, resolve
from ..scoring4 import DEFAULT_SPEC
from .p4_qwen_nested_lens_fit import (
    model_reference,
    qwen_fused_kernel_contract,
    registered_output_check,
    verify_model_fused_bindings,
    verify_package_versions,
    verify_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def normalized_exact_alias(
        generated: str, aliases: Sequence[str]) -> str | None:
    """Return the longest accepted alias with exactly equal normalization."""
    normalized = DEFAULT_SPEC.normalize_generation(generated)
    matches = [
        alias for alias in aliases
        if DEFAULT_SPEC.normalize_generation(alias) == normalized
    ]
    if not matches:
        return None
    return max(
        matches,
        key=lambda value: len(DEFAULT_SPEC.normalize_generation(value)),
    )


def generated_phase_ids(
        token_ids: Sequence[int], *, prompt_length: int, parsed,
        delimiters: DelimiterSpec) -> dict[str, list[int]]:
    """Extract generated phase content, excluding structural delimiters."""
    excluded = {
        *delimiters.reasoning_start_ids,
        *delimiters.reasoning_end_ids,
        *delimiters.eos_token_ids,
    }
    result = {
        Phase.REASONING.value: [],
        Phase.FINAL_ANSWER.value: [],
    }
    for index in range(prompt_length, len(token_ids)):
        phase = parsed.phases[index]
        token = int(token_ids[index])
        if phase in result and token not in excluded:
            result[phase].append(token)
    return result


def analyze_mode_rows(rows: Sequence[Mapping], gates: Mapping) -> dict:
    """Aggregate the complete paired family grid and apply frozen dev gates."""
    mode_order = ["thinking_on", "thinking_off"]
    by_key = {}
    for row in rows:
        key = (str(row["canonical_family"]), str(row["mode"]))
        if key in by_key:
            raise RuntimeError(f"duplicate mode/family row: {key}")
        by_key[key] = row
    families = sorted({key[0] for key in by_key})
    complete = bool(families) and all(
        (family, mode) in by_key
        for family in families for mode in mode_order)
    if not complete:
        raise RuntimeError("mode model gate lacks a complete family grid")

    summaries = {}
    for mode in mode_order:
        selected = [by_key[(family, mode)] for family in families]
        n = len(selected)
        summaries[mode] = {
            "n_families": n,
            "accuracy": float(np.mean([
                bool(row["correct"]) for row in selected])),
            "parse_failure_rate": float(np.mean([
                not bool(row["parse_valid"]) for row in selected])),
            "truncation_rate": float(np.mean([
                bool(row["truncated"]) for row in selected])),
            "final_answer_nonempty_rate": float(np.mean([
                int(row["final_answer_tokens"]) > 0 for row in selected])),
            "reasoning_content_rate": float(np.mean([
                int(row["reasoning_content_tokens"]) > 0
                for row in selected])),
            "generated_tokens_median": float(np.median([
                int(row["generated_tokens"]) for row in selected])),
            "reasoning_content_tokens_median": float(np.median([
                int(row["reasoning_content_tokens"])
                for row in selected])),
            "final_answer_tokens_median": float(np.median([
                int(row["final_answer_tokens"]) for row in selected])),
        }

    paired_differences = np.asarray([
        float(bool(by_key[(family, "thinking_on")]["correct"]))
        - float(bool(by_key[(family, "thinking_off")]["correct"]))
        for family in families
    ], dtype=np.float64)
    draws = int(gates["family_bootstrap_draws"])
    generator = np.random.default_rng(
        int(gates["family_bootstrap_seed"]))
    indices = generator.integers(
        0, len(families), size=(draws, len(families)))
    bootstrap = paired_differences[indices].mean(axis=1)
    common_parse_valid = sum(
        bool(by_key[(family, "thinking_on")]["parse_valid"])
        and bool(by_key[(family, "thinking_off")]["parse_valid"])
        for family in families)
    common_correct = sum(
        bool(by_key[(family, "thinking_on")]["correct"])
        and bool(by_key[(family, "thinking_off")]["correct"])
        for family in families)

    checks = {
        "complete_paired_family_grid": complete,
        "parse_failure_within_tolerance": all(
            summaries[mode]["parse_failure_rate"] <= float(
                gates["maximum_parse_failure_rate_by_mode"])
            for mode in mode_order),
        "truncation_within_tolerance": all(
            summaries[mode]["truncation_rate"] <= float(
                gates["maximum_truncation_rate_by_mode"])
            for mode in mode_order),
        "accuracy_measurable_in_both_modes": all(
            summaries[mode]["accuracy"] >= float(
                gates["minimum_accuracy_by_mode"])
            for mode in mode_order),
        "common_correct_family_floor": common_correct >= int(
            gates["minimum_common_correct_families"]),
        "thinking_on_has_reasoning_content": summaries[
            "thinking_on"]["reasoning_content_rate"] >= float(
                gates["minimum_thinking_on_reasoning_content_rate"]),
        "thinking_off_has_no_reasoning_content": (
            not bool(gates[
                "require_zero_thinking_off_reasoning_content"])
            or summaries["thinking_off"]["reasoning_content_rate"] == 0.0),
        "final_answer_nonempty_in_both_modes": all(
            summaries[mode]["final_answer_nonempty_rate"] >= float(
                gates["minimum_final_answer_nonempty_rate_by_mode"])
            for mode in mode_order),
    }
    return {
        "schema_version": 1,
        "mode_summaries": summaries,
        "paired_accuracy": {
            "thinking_on_minus_thinking_off": float(
                paired_differences.mean()),
            "family_bootstrap_ci90": [
                float(value) for value in np.quantile(
                    bootstrap, [0.05, 0.95])],
            "bootstrap_draws": draws,
            "bootstrap_seed": int(gates["family_bootstrap_seed"]),
        },
        "common_support": {
            "n_families": len(families),
            "n_parse_valid_both_modes": int(common_parse_valid),
            "n_correct_both_modes": int(common_correct),
            "family_ids_sha256": object_sha256(families),
        },
        "development_gate_thresholds": dict(gates),
        "development_gate_checks": checks,
        "all_model_backed_development_gates_pass": bool(
            all(checks.values())),
        "claim_boundary": (
            "Baseline model/parser/correctness development gate only; no "
            "P4-P2 intervention, confirmatory, or replication outcome."),
        "freeze_ready": False,
        "remaining_freeze_blockers": [
            "P4-P2 prospective canonical-family split",
            "P4-P2 intervention SESOI and power ruler",
            "independent protocol review and PI sign-off",
        ],
    }


def _plot_analysis(analysis: Mapping, *, png_path: Path,
                   pdf_path: Path) -> None:
    modes = ["thinking_on", "thinking_off"]
    labels = ["thinking on", "thinking off"]
    colors = ["#0072B2", "#E69F00"]
    summary = analysis["mode_summaries"]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))

    axis = axes[0, 0]
    accuracy = [summary[mode]["accuracy"] for mode in modes]
    axis.bar(labels, accuracy, color=colors)
    axis.axhline(
        analysis["development_gate_thresholds"][
            "minimum_accuracy_by_mode"],
        color="#555555", linestyle="--", linewidth=1, label="dev floor")
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("normalized exact accuracy")
    axis.set_title("A · Baseline answer quality", loc="left")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    x = np.arange(2)
    parse = [summary[mode]["parse_failure_rate"] for mode in modes]
    truncation = [summary[mode]["truncation_rate"] for mode in modes]
    axis.bar(x - 0.18, parse, width=0.36, color="#D55E00",
             label="parse failure")
    axis.bar(x + 0.18, truncation, width=0.36, color="#CC79A7",
             label="truncation")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, max(0.05, max(parse + truncation) * 1.2))
    axis.set_ylabel("rate")
    axis.set_title("B · Parser and stop outcomes", loc="left")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    reasoning = [
        summary[mode]["reasoning_content_tokens_median"] for mode in modes]
    final = [summary[mode]["final_answer_tokens_median"] for mode in modes]
    axis.bar(x - 0.18, reasoning, width=0.36, color="#56B4E9",
             label="reasoning content")
    axis.bar(x + 0.18, final, width=0.36, color="#009E73",
             label="final answer")
    axis.set_xticks(x, labels)
    axis.set_ylabel("median generated tokens")
    axis.set_title("C · Structural phase occupancy", loc="left")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    support = analysis["common_support"]
    values = [support["n_parse_valid_both_modes"],
              support["n_correct_both_modes"]]
    axis.bar(["parse-valid\nboth", "correct\nboth"], values,
             color=["#999999", "#009E73"])
    axis.axhline(
        analysis["development_gate_thresholds"][
            "minimum_common_correct_families"],
        color="#555555", linestyle="--", linewidth=1,
        label="common-correct floor")
    axis.set_ylim(0, support["n_families"] + 1)
    axis.set_ylabel("canonical families")
    axis.set_title("D · Paired common support", loc="left")
    axis.legend(frameon=False, fontsize=8)

    status = "PASS" if analysis[
        "all_model_backed_development_gates_pass"] else "BLOCKED"
    figure.suptitle(
        "Official Qwen mode model gate · Phase 4 development\n"
        f"{status}; no intervention outcome",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _load_methods_gate(config: Mapping) -> tuple[dict, dict]:
    specification = config["methods_gate"]
    event = resolve(specification["evidence_id"])
    if not event["live"]:
        raise RuntimeError("mode parser methods gate is not live")
    config_path = resolve_uri(specification["config_uri"])
    result_path = resolve_uri(specification["result_uri"])
    if file_sha256(config_path) != specification["config_sha256"]:
        raise RuntimeError("mode parser config hash drift")
    if file_sha256(result_path) != specification["result_sha256"]:
        raise RuntimeError("mode parser result hash drift")
    result = json.loads(result_path.read_text())
    if result.get("all_protocol_gates_pass") is not True:
        raise RuntimeError("mode parser methods gate did not pass")
    return yaml.safe_load(config_path.read_text()), {
        "event_code_commit": event["code_commit"],
        "config_sha256": specification["config_sha256"],
        "result_sha256": specification["result_sha256"],
    }


def _load_selection(config: Mapping) -> tuple[list[str], dict]:
    specification = config["selection"]
    path = resolve_uri(specification["source_uri"])
    if file_sha256(path) != specification["source_sha256"]:
        raise RuntimeError("mode selection manifest hash drift")
    envelope = json.loads(path.read_text())
    if envelope.get("payload_sha256") != specification[
            "source_payload_sha256"]:
        raise RuntimeError("mode selection payload hash drift")
    payload = envelope["payload"]
    if payload.get("evidence_id") != specification[
            "source_evidence_id"]:
        raise RuntimeError("mode selection evidence ID drift")
    if payload.get("selection_is_outcome_blind") is not True:
        raise RuntimeError("mode selection is not outcome-blind")
    if payload.get("selection_uses_only_consumed_phase3_families") \
            is not True:
        raise RuntimeError("mode selection is not confined to Phase 3")
    subset = payload["subset"][specification["subset_key"]]
    fact_ids = [str(value) for value in subset["fact_ids"]]
    if len(fact_ids) != int(specification["expected_facts"]) \
            or len(set(fact_ids)) != len(fact_ids):
        raise RuntimeError("mode selection fact count/uniqueness drift")
    return fact_ids, {
        "path": str(path),
        "file_sha256": specification["source_sha256"],
        "payload_sha256": specification["source_payload_sha256"],
        "fact_ids_sha256": object_sha256(fact_ids),
        "selection_contract": payload["selection_contract"],
    }


def _load_bundles(config: Mapping, fact_ids: Sequence[str]) -> tuple[
        list[FactBundle], dict]:
    requested = set(fact_ids)
    resolved = {}
    contract = {}
    for specification in config["task_banks"]:
        path = resolve_uri(specification["uri"])
        actual = file_sha256(path)
        if actual != specification["sha256"]:
            raise RuntimeError(f"mode task-bank hash drift: {path}")
        bundles = load_bank(path)
        contract[specification["uri"]] = {
            "sha256": actual, "n_facts": len(bundles)}
        for bundle in bundles:
            if bundle.fact_id in requested:
                if bundle.fact_id in resolved:
                    raise RuntimeError(
                        f"duplicate selected fact {bundle.fact_id}")
                resolved[bundle.fact_id] = bundle
    if set(resolved) != requested:
        missing = sorted(requested - set(resolved))
        raise RuntimeError(f"mode selected facts do not resolve: {missing}")
    ordered = [resolved[fact_id] for fact_id in fact_ids]
    families = [bundle.canonical_family for bundle in ordered]
    expected_families = int(config["selection"]["expected_families"])
    if len(set(families)) != expected_families:
        raise RuntimeError("mode selected canonical-family count drift")
    return ordered, contract


def _delimiter_spec(parser: Mapping) -> DelimiterSpec:
    return DelimiterSpec(
        reasoning_start_ids=tuple(int(value) for value in
                                  parser["reasoning_start_ids"]),
        reasoning_end_ids=tuple(int(value) for value in
                                parser["reasoning_end_ids"]),
        eos_token_ids=tuple(int(value) for value in parser["eos_token_ids"]),
        version=str(parser["version"]),
        require_closed_reasoning=bool(parser["require_closed_reasoning"]),
    )


def _token_ids(value) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise RuntimeError("expected one rendered prompt")
        value = value[0]
    return [int(token) for token in value]


def _render_prompt(tokenizer, content: str, *, enable_thinking: bool) -> dict:
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking)
    ids = _token_ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=enable_thinking))
    if tokenizer.decode(ids, skip_special_tokens=False) != text:
        raise RuntimeError("official mode prompt text/token round trip drift")
    return {
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_ids": ids,
        "token_ids_sha256": object_sha256(ids),
    }


@torch.no_grad()
def _run_completion(
        hf_model, tokenizer, *, bundle: FactBundle, mode: str,
        methods: Mapping, protocol: Mapping,
        delimiters: DelimiterSpec) -> dict:
    prompt_variant = str(protocol["prompt_variant"])
    user_content = (
        bundle.prompts[prompt_variant].rstrip() + "\n\n"
        + str(protocol["prompt_instruction"]))
    enable_thinking = mode == "thinking_on"
    rendered = _render_prompt(
        tokenizer, user_content, enable_thinking=enable_thinking)
    prompt_ids = rendered["token_ids"]
    prompt_parse = classify_token_phases(
        prompt_ids, prompt_length=len(prompt_ids), delimiters=delimiters)
    if bool(prompt_parse.reasoning_open_at_generation) != enable_thinking:
        raise RuntimeError(f"official prompt phase drift for {mode}")
    input_ids = torch.tensor(
        [prompt_ids], device="cuda", dtype=torch.long)
    max_new_tokens = int(protocol["max_new_tokens"])
    started = time.time()
    output = hf_model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=bool(protocol["do_sample"]),
        eos_token_id=delimiters.eos_token_ids[0],
        pad_token_id=delimiters.eos_token_ids[0],
    )
    torch.cuda.synchronize()
    elapsed = time.time() - started
    full_ids = [int(value) for value in output[0].tolist()]
    generated = full_ids[len(prompt_ids):]
    parsed = classify_token_phases(
        full_ids, prompt_length=len(prompt_ids), delimiters=delimiters)
    phases = generated_phase_ids(
        full_ids, prompt_length=len(prompt_ids), parsed=parsed,
        delimiters=delimiters)
    final_text = tokenizer.decode(
        phases[Phase.FINAL_ANSWER.value], skip_special_tokens=True).strip()
    matched_alias = normalized_exact_alias(
        final_text, bundle.accepted_answers)
    eos = bool(generated and generated[-1] in delimiters.eos_token_ids)
    truncated = bool(not eos and len(generated) >= max_new_tokens)
    stop_reason = "eos" if eos else "length" if truncated else "error"
    correct = bool(
        parsed.valid and not truncated and eos and matched_alias is not None)
    return {
        "fact_id": bundle.fact_id,
        "canonical_family": bundle.canonical_family,
        "bank": bundle.bank,
        "prompt_variant": prompt_variant,
        "mode": mode,
        "enable_thinking": enable_thinking,
        "prompt_text_sha256": rendered["text_sha256"],
        "prompt_token_ids_sha256": rendered["token_ids_sha256"],
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "reasoning_content_tokens": len(
            phases[Phase.REASONING.value]),
        "final_answer_tokens": len(
            phases[Phase.FINAL_ANSWER.value]),
        "parse_valid": bool(parsed.valid),
        "parse_errors_json": json.dumps(list(parsed.errors)),
        "reasoning_open_at_generation": bool(
            parsed.reasoning_open_at_generation),
        "stop_reason": stop_reason,
        "truncated": truncated,
        "accepted_answers_json": json.dumps(
            list(bundle.accepted_answers), ensure_ascii=False),
        "matched_alias": matched_alias,
        "correct": correct,
        "final_answer_text": final_text,
        "generated_text": tokenizer.decode(
            generated, skip_special_tokens=False),
        "generated_token_ids_json": json.dumps(generated),
        "elapsed_seconds": round(elapsed, 3),
        "parser_version": methods["parser"]["version"],
    }


def _load_or_create_state(path: Path, header: Mapping) -> dict:
    if not path.exists():
        state = {
            "schema_version": 1,
            "header": dict(header),
            "rows": {},
            "runtime": None,
        }
        atomic_json(path, state)
        return state
    state = json.loads(path.read_text())
    if state.get("header") != dict(header):
        raise RuntimeError("refusing incompatible mode-gate resume")
    return state


@torch.no_grad()
def main() -> None:  # noqa: C901
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("tier") != "phase4-development":
        raise RuntimeError("mode model gate is development only")
    existing = registered_output_check(config["evidence_id"])
    if existing is not None:
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return
    clean = require_clean_tree()
    methods, methods_contract = _load_methods_gate(config)
    protocol = {
        **dict(config["protocol"]),
        "prompt_variant": config["selection"]["prompt_variant"],
    }
    if protocol["mode_order"] != ["thinking_on", "thinking_off"]:
        raise RuntimeError("mode order drift")
    if protocol["phase_parser_version"] != methods["parser"]["version"]:
        raise RuntimeError("mode parser version drift")
    if int(protocol["max_new_tokens"]) != int(
            methods["generation"]["max_new_tokens"]) \
            or bool(protocol["do_sample"]) != bool(
                methods["generation"]["do_sample"]):
        raise RuntimeError("mode generation contract drift")
    expected_model = model_reference(config["model_uri"])
    if methods["model_id"] != expected_model["model_id"] \
            or methods["model_revision"] != expected_model["revision"]:
        raise RuntimeError("mode methods/model identity drift")
    if protocol["primary_phases"] != ["prefill", "final_answer"]:
        raise RuntimeError("mode common-support primary drift")
    if protocol["structurally_absent_cell"] != \
            "thinking_off_x_reasoning":
        raise RuntimeError("mode structural-cell declaration drift")
    if protocol["answer_rule"] != \
            "normalized-exact-accepted-alias":
        raise RuntimeError("mode answer rule drift")

    fact_ids, selection_contract = _load_selection(config)
    bundles, bank_contract = _load_bundles(config, fact_ids)
    expected_keys = [
        f"{bundle.fact_id}|{mode}"
        for bundle in bundles for mode in protocol["mode_order"]
    ]
    header = {
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "methods_gate": methods_contract,
        "selection_file_sha256": selection_contract["file_sha256"],
        "selection_payload_sha256": selection_contract["payload_sha256"],
        "selected_fact_ids_sha256": object_sha256(fact_ids),
        "task_banks_sha256": object_sha256(bank_contract),
        "protocol_sha256": object_sha256(protocol),
        "development_gates_sha256": object_sha256(
            config["development_gates"]),
        "model": model_reference(config["model_uri"]),
    }
    output_dir = (
        metrics_dir(config["slug"]) / "mode_gate"
        / config["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    state = _load_or_create_state(state_path, header)
    unexpected = set(state["rows"]) - set(expected_keys)
    if unexpected:
        raise RuntimeError(f"mode state has unexpected rows: {unexpected}")

    incomplete = [key for key in expected_keys if key not in state["rows"]]
    if incomplete:
        gpu = require_cuda_gpu()
        package_versions = verify_package_versions(
            config["runtime"]["packages"])
        fused_runtime = qwen_fused_kernel_contract(config["runtime"])
        model_path = resolve_uri(config["model_uri"])
        snapshot_manifest_path = resolve_uri(
            config["model_snapshot_manifest_uri"])
        if file_sha256(snapshot_manifest_path) != config[
                "model_snapshot_manifest_sha256"]:
            raise RuntimeError("model snapshot manifest hash drift")
        snapshot_manifest = json.loads(snapshot_manifest_path.read_text())
        model = model_reference(config["model_uri"])
        if snapshot_manifest["model_id"] != model["model_id"] \
                or snapshot_manifest["revision"] != model["revision"]:
            raise RuntimeError("model snapshot identity drift")
        snapshot = verify_snapshot(model_path, snapshot_manifest)

        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_path))
        if type(tokenizer).__name__ != methods[
                "official_template"]["tokenizer_class"]:
            raise RuntimeError("official tokenizer class drift")
        expected_template_hash = methods[
            "official_template"]["chat_template_sha256"]
        # The methods gate uses raw-byte SHA-256, while object_sha256 adds
        # canonical JSON quoting. Recompute the raw hash for the hard gate.
        template_hash = hashlib.sha256(
            tokenizer.chat_template.encode()).hexdigest()
        if template_hash != expected_template_hash:
            raise RuntimeError("official chat template hash drift")
        delimiters = _delimiter_spec(methods["parser"])
        for text, expected_ids in (
                (methods["parser"]["reasoning_start_text"],
                 delimiters.reasoning_start_ids),
                (methods["parser"]["reasoning_end_text"],
                 delimiters.reasoning_end_ids),
                (methods["parser"]["eos_text"],
                 delimiters.eos_token_ids)):
            actual_ids = _token_ids(tokenizer(
                text, add_special_tokens=False))
            if actual_ids != list(expected_ids):
                raise RuntimeError(
                    f"official delimiter token IDs drifted for {text!r}")
        torch.cuda.reset_peak_memory_stats()
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=torch.bfloat16).to("cuda").eval()
        assert_model_on_cuda(hf_model)
        fused_model = verify_model_fused_bindings(
            hf_model, config["runtime"])
        state["runtime"] = {
            "gpu": gpu,
            "package_versions": package_versions,
            "fused_runtime": fused_runtime,
            "fused_model": fused_model,
            "model_snapshot_inventory_sha256": snapshot[
                "inventory_sha256"],
            "model_snapshot_manifest_sha256": config[
                "model_snapshot_manifest_sha256"],
            "chat_template_sha256": template_hash,
        }
        atomic_json(state_path, state)

        for bundle in bundles:
            for mode in protocol["mode_order"]:
                key = f"{bundle.fact_id}|{mode}"
                if key in state["rows"]:
                    continue
                row = _run_completion(
                    hf_model, tokenizer, bundle=bundle, mode=mode,
                    methods=methods, protocol=protocol,
                    delimiters=delimiters)
                state["rows"][key] = row
                atomic_json(state_path, state)
                print(
                    f"{len(state['rows'])}/{len(expected_keys)} "
                    f"{bundle.fact_id} {mode} "
                    f"parse={row['parse_valid']} correct={row['correct']} "
                    f"tokens={row['generated_tokens']} "
                    f"{row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
        state["runtime"]["peak_vram_bytes"] = int(
            torch.cuda.max_memory_allocated())
        atomic_json(state_path, state)
        del hf_model
        torch.cuda.empty_cache()
    elif state.get("runtime") is None:
        raise RuntimeError("complete mode state lacks runtime provenance")

    if set(state["rows"]) != set(expected_keys):
        raise RuntimeError("mode model gate ended with incomplete rows")
    rows = [state["rows"][key] for key in expected_keys]
    analysis = analyze_mode_rows(rows, config["development_gates"])
    analysis.update({
        "evidence_id": config["evidence_id"],
        "model": model_reference(config["model_uri"]),
        "methods_evidence_id": config["methods_gate"]["evidence_id"],
        "selection_evidence_id": config["selection"][
            "source_evidence_id"],
    })

    rows_path = output_dir / "mode_completion_rows.parquet"
    manifest_path = output_dir / "input_manifest.json"
    result_path = output_dir / "mode_gate_result.json"
    _atomic_parquet(rows_path, pd.DataFrame(rows))
    input_payload = {
        "schema_version": 1,
        "header": header,
        "runtime": state["runtime"],
        "methods_gate": methods_contract,
        "selection": selection_contract,
        "task_banks": bank_contract,
        "protocol": protocol,
        "development_gates": dict(config["development_gates"]),
    }
    manifest = {
        "schema_version": 1,
        "payload": input_payload,
        "payload_sha256": object_sha256(input_payload),
    }
    atomic_json(manifest_path, manifest)
    png_path = figures_dir() / f"{config['figure']['stem']}.png"
    pdf_path = figures_dir() / f"{config['figure']['stem']}.pdf"
    _plot_analysis(
        analysis, png_path=png_path, pdf_path=pdf_path)
    command = (
        "python -m "
        "jspace_phase4.experiments.p4_qwen_mode_model_gate "
        f"--config {arguments.config}")
    inputs = {
        "methods_gate": methods_contract["result_sha256"],
        "selection": selection_contract["payload_sha256"],
        "task_banks": object_sha256(bank_contract),
        "model_snapshot": state["runtime"][
            "model_snapshot_inventory_sha256"],
        "input_manifest": manifest["payload_sha256"],
    }
    write_result4(
        analysis, result_path,
        Provenance4(
            evidence_id=config["evidence_id"], tier=config["tier"],
            command=command, inputs=inputs,
            input_manifest_sha256=manifest["payload_sha256"],
            model=model_reference(config["model_uri"]),
            seed_contract=(
                "registered outcome-blind 20-family subset; one composed "
                "fact per family; thinking-on then thinking-off within "
                "family; deterministic greedy generation; frozen family "
                "bootstrap seed"),
        ),
    )
    outputs = [
        result_path, manifest_path, state_path, rows_path,
        png_path, pdf_path,
    ]
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            "Model-backed official-Qwen thinking on/off development gate "
            "over 20 paired consumed Phase 3 families: real completion "
            "parsing, truncation, normalized exact answer quality, and "
            "common-mode support; no intervention outcome."),
        command=command, outputs=outputs, inputs=inputs,
    )
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "all_model_backed_development_gates_pass": analysis[
            "all_model_backed_development_gates_pass"],
        "mode_summaries": analysis["mode_summaries"],
        "common_support": analysis["common_support"],
        "result": str(result_path),
        "figure": str(png_path),
    }, indent=1))


if __name__ == "__main__":
    main()
