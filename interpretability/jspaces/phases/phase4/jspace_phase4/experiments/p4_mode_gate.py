"""Author the official Qwen mode/template/parser protocol gate.

This CPU methods gate does not load model weights and does not inspect any
Phase 4 intervention outcome. It pins the official template behavior,
delimiter-aware phase parser, structural phase support, rationale-control
token matching, and the exact future primary interaction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ..manifests import atomic_json, file_sha256, object_sha256, require_clean_tree
from ..paths4 import resolve_uri
from ..phase_hooks import DelimiterSpec, Phase, classify_token_phases
from ..registry4 import create


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--author-protocol", action="store_true")
    group.add_argument("--register-existing", action="store_true")
    return parser.parse_args()


def _ids(encoded) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected one rendered chat prompt")
        encoded = encoded[0]
    return [int(value) for value in encoded]


def render_official_prompt(
        tokenizer, messages: list[dict], *, enable_thinking: bool,
        add_generation_prompt: bool = True) -> dict:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking)
    token_ids = _ids(tokenizer.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking))
    if tokenizer.decode(token_ids, skip_special_tokens=False) != text:
        raise RuntimeError("official template text/token round trip drifted")
    return {
        "enable_thinking": bool(enable_thinking),
        "rendered_text": text,
        "rendered_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_ids": token_ids,
        "token_ids_sha256": object_sha256(token_ids),
        "n_tokens": len(token_ids),
    }


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def token_match_controls(
        tokenizer, controls: Mapping[str, str], *,
        neutral_fragments: Sequence[str], maximum_span: int) -> dict:
    target = max(_token_count(tokenizer, text) for text in controls.values())
    matched = {}
    for name, initial in controls.items():
        text = initial
        iterations = 0
        while _token_count(tokenizer, text) < target:
            current = _token_count(tokenizer, text)
            candidates = []
            for fragment in neutral_fragments:
                candidate = text + fragment
                count = _token_count(tokenizer, candidate)
                if current < count <= target:
                    candidates.append((count, len(fragment), candidate))
            if not candidates:
                raise RuntimeError(
                    f"cannot exactly token-match rationale control {name}")
            _, _, text = max(candidates, key=lambda row: (row[0], -row[1]))
            iterations += 1
            if iterations > 128:
                raise RuntimeError("rationale control padding did not converge")
        token_ids = [int(value) for value in tokenizer(
            text, add_special_tokens=False).input_ids]
        matched[name] = {
            "text": text, "token_ids": token_ids,
            "token_ids_sha256": object_sha256(token_ids),
            "n_tokens": len(token_ids),
            "neutral_padding_fragments": iterations,
        }
    counts = [row["n_tokens"] for row in matched.values()]
    span = max(counts) - min(counts)
    return {
        "controls": matched,
        "target_tokens": target,
        "token_span": span,
        "maximum_token_span": int(maximum_span),
        "passes": bool(span <= maximum_span),
    }


def _subsequence_starts(values: Sequence[int], target: Sequence[int]) -> list[int]:
    width = len(target)
    return [index for index in range(len(values) - width + 1)
            if list(values[index:index + width]) == list(target)]


def answer_boundary_status(
        token_ids: Sequence[int], *, prompt_length: int, parsed,
        accepted_token_sequences: Sequence[Sequence[int]]) -> dict:
    final_matches = []
    reasoning_matches = []
    for accepted in accepted_token_sequences:
        for start in _subsequence_starts(token_ids, accepted):
            if start < prompt_length:
                continue
            phases = set(parsed.phases[start:start + len(accepted)])
            if phases == {Phase.FINAL_ANSWER.value}:
                final_matches.append(start)
            elif Phase.REASONING.value in phases:
                reasoning_matches.append(start)
    if final_matches:
        status = "accepted_answer_in_final"
    elif reasoning_matches:
        status = "answer_before_reasoning_close"
    else:
        status = "answer_omitted"
    return {
        "status": status,
        "final_match_indices": final_matches,
        "reasoning_match_indices": reasoning_matches,
        "parse_failure": not parsed.valid,
    }


def completion_outcome(parsed, *, stop_reason: str,
                       answer_status: Mapping) -> dict:
    if stop_reason not in {"eos", "length", "error"}:
        raise ValueError(stop_reason)
    return {
        "parse_valid": bool(parsed.valid),
        "parse_errors": list(parsed.errors),
        "parse_failure": not parsed.valid,
        "truncated": stop_reason == "length",
        "stop_reason": stop_reason,
        "answer_boundary_status": answer_status["status"],
        "answer_omission": answer_status["status"] == "answer_omitted",
        # Correctness is deliberately absent: parse/truncation are separate
        # outcomes and cannot be silently coerced to an incorrect answer.
        "eligible_for_final_answer_grading": bool(
            parsed.valid
            and answer_status["status"] == "accepted_answer_in_final"),
    }


def _goldens(tokenizer, prompts: Mapping[str, Mapping],
             delimiters: DelimiterSpec, generation: Mapping) -> dict:
    reasoning_ids = [int(value) for value in tokenizer(
        generation["golden_reasoning_text"],
        add_special_tokens=False).input_ids]
    separator_ids = [int(value) for value in tokenizer(
        "\n\n", add_special_tokens=False).input_ids]
    answer_ids = [int(value) for value in tokenizer(
        generation["golden_final_answer"],
        add_special_tokens=False).input_ids]
    eos_id = delimiters.eos_token_ids[0]
    on_prompt = list(prompts["thinking_on"]["token_ids"])
    on_tokens = (on_prompt + reasoning_ids
                 + list(delimiters.reasoning_end_ids)
                 + separator_ids + answer_ids + [eos_id])
    on_parse = classify_token_phases(
        on_tokens, prompt_length=len(on_prompt), delimiters=delimiters)
    on_answer = answer_boundary_status(
        on_tokens, prompt_length=len(on_prompt), parsed=on_parse,
        accepted_token_sequences=[answer_ids])

    off_prompt = list(prompts["thinking_off"]["token_ids"])
    off_tokens = off_prompt + answer_ids + [eos_id]
    off_parse = classify_token_phases(
        off_tokens, prompt_length=len(off_prompt), delimiters=delimiters)
    off_answer = answer_boundary_status(
        off_tokens, prompt_length=len(off_prompt), parsed=off_parse,
        accepted_token_sequences=[answer_ids])

    truncated_tokens = on_prompt + reasoning_ids
    truncated_parse = classify_token_phases(
        truncated_tokens, prompt_length=len(on_prompt),
        delimiters=delimiters)
    truncated_answer = answer_boundary_status(
        truncated_tokens, prompt_length=len(on_prompt), parsed=truncated_parse,
        accepted_token_sequences=[answer_ids])
    eos_inside_tokens = on_prompt + reasoning_ids + [eos_id]
    eos_inside_parse = classify_token_phases(
        eos_inside_tokens, prompt_length=len(on_prompt),
        delimiters=delimiters)
    return {
        "thinking_on_complete": {
            "parse": {
                "valid": on_parse.valid, "errors": list(on_parse.errors),
                "reasoning_open_at_generation":
                    on_parse.reasoning_open_at_generation,
                "phase_counts": {
                    phase.value: on_parse.phases.count(phase.value)
                    for phase in Phase},
            },
            "answer": on_answer,
            "outcome": completion_outcome(
                on_parse, stop_reason="eos", answer_status=on_answer),
            "full_token_ids_sha256": object_sha256(on_tokens),
        },
        "thinking_off_complete": {
            "parse": {
                "valid": off_parse.valid, "errors": list(off_parse.errors),
                "reasoning_open_at_generation":
                    off_parse.reasoning_open_at_generation,
                "phase_counts": {
                    phase.value: off_parse.phases.count(phase.value)
                    for phase in Phase},
            },
            "answer": off_answer,
            "outcome": completion_outcome(
                off_parse, stop_reason="eos", answer_status=off_answer),
            "full_token_ids_sha256": object_sha256(off_tokens),
        },
        "thinking_on_length_truncation": {
            "parse_errors": list(truncated_parse.errors),
            "outcome": completion_outcome(
                truncated_parse, stop_reason="length",
                answer_status=truncated_answer),
        },
        "thinking_on_eos_inside_reasoning": {
            "parse_valid": eos_inside_parse.valid,
            "parse_errors": list(eos_inside_parse.errors),
        },
    }


def author_protocol(config: Mapping, tokenizer) -> dict:
    template = config["official_template"]
    if type(tokenizer).__name__ != template["tokenizer_class"]:
        raise RuntimeError("unexpected tokenizer class")
    chat_template_sha = hashlib.sha256(
        tokenizer.chat_template.encode()).hexdigest()
    if chat_template_sha != template["chat_template_sha256"]:
        raise RuntimeError("official chat template hash drifted")
    prompts = {}
    for name, mode in template["modes"].items():
        rendered = render_official_prompt(
            tokenizer, list(template["messages"]),
            enable_thinking=bool(mode["enable_thinking"]),
            add_generation_prompt=bool(template["add_generation_prompt"]))
        if rendered["rendered_prompt_sha256"] != \
                mode["rendered_prompt_sha256"]:
            raise RuntimeError(f"official prompt rendering drifted for {name}")
        prompts[name] = rendered
    parser = config["parser"]
    delimiters = DelimiterSpec(
        reasoning_start_ids=tuple(parser["reasoning_start_ids"]),
        reasoning_end_ids=tuple(parser["reasoning_end_ids"]),
        eos_token_ids=tuple(parser["eos_token_ids"]),
        version=parser["version"],
        require_closed_reasoning=bool(parser["require_closed_reasoning"]))
    if _ids(tokenizer(
            parser["reasoning_start_text"],
            add_special_tokens=False)) != list(delimiters.reasoning_start_ids):
        raise RuntimeError("reasoning-start token IDs drifted")
    if _ids(tokenizer(
            parser["reasoning_end_text"],
            add_special_tokens=False)) != list(delimiters.reasoning_end_ids):
        raise RuntimeError("reasoning-end token IDs drifted")
    if _ids(tokenizer(
            parser["eos_text"],
            add_special_tokens=False)) != list(delimiters.eos_token_ids):
        raise RuntimeError("EOS token IDs drifted")
    controls_config = config["rationale_controls"]
    controls = token_match_controls(
        tokenizer,
        {name: controls_config[name] for name in
         ("correct", "wrong", "shuffled", "filler")},
        neutral_fragments=controls_config["neutral_fragments"],
        maximum_span=int(controls_config["maximum_token_span"]))
    goldens = _goldens(
        tokenizer, prompts, delimiters, config["generation"])
    structural = config["structural_common_support"]
    hook_support = {
        "thinking_on": {
            "prefill": True, "reasoning": True,
            "final_answer": True, "all": True},
        "thinking_off": {
            "prefill": True, "reasoning": False,
            "final_answer": True, "all": True},
    }
    gate_checks = {
        "template_hash_matches": True,
        "official_mode_prompts_distinct": (
            prompts["thinking_on"]["rendered_prompt_sha256"]
            != prompts["thinking_off"]["rendered_prompt_sha256"]),
        "thinking_on_opens_reasoning_at_generation": goldens[
            "thinking_on_complete"]["parse"][
                "reasoning_open_at_generation"],
        "thinking_off_starts_final_answer_at_generation": not goldens[
            "thinking_off_complete"]["parse"][
                "reasoning_open_at_generation"],
        "complete_goldens_parse": (
            goldens["thinking_on_complete"]["parse"]["valid"]
            and goldens["thinking_off_complete"]["parse"]["valid"]),
        "answers_only_graded_in_final_phase": (
            goldens["thinking_on_complete"]["answer"]["status"]
            == "accepted_answer_in_final"
            and goldens["thinking_off_complete"]["answer"]["status"]
            == "accepted_answer_in_final"),
        "truncation_stays_separate": goldens[
            "thinking_on_length_truncation"]["outcome"]["truncated"],
        "eos_inside_reasoning_fails_parse": not goldens[
            "thinking_on_eos_inside_reasoning"]["parse_valid"],
        "rationale_controls_token_matched": controls["passes"],
        "primary_uses_only_common_mode_phases": (
            structural["primary_phases"] == ["prefill", "final_answer"]),
        "thinking_off_reasoning_cell_marked_structurally_absent": not (
            hook_support["thinking_off"]["reasoning"]),
    }
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": chat_template_sha,
        "parser": dict(parser),
        "official_prompts": prompts,
        "generation": dict(config["generation"]),
        "rationale_control_audit": controls,
        "goldens": goldens,
        "hook_phase_support": hook_support,
        "structural_common_support": dict(structural),
        "primary": dict(config["primary"]),
        "gate_checks": gate_checks,
        "all_protocol_gates_pass": all(gate_checks.values()),
        "outcome_blinding": (
            "Tokenizer/template methods only; no model weights or Phase 4 "
            "intervention outcomes loaded."),
        "freeze_ready": False,
        "remaining_freeze_blockers": [
            "model-backed development parser/final-answer gate",
            "P4-P2 family split, power, and SESOI",
            "independent protocol review and PI sign-off",
        ],
    }


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    output = Path(config["outputs"]["protocol"])
    if arguments.author_protocol:
        import transformers
        model_path = resolve_uri(config["model_uri"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
        result = author_protocol(config, tokenizer)
        if not result["all_protocol_gates_pass"]:
            raise RuntimeError("Qwen mode parser protocol gate failed")
        result.update({
            "generation_code_commit": clean["code_commit"],
            "config_sha256": file_sha256(config_path),
        })
        atomic_json(output, result)
        print(json.dumps({
            "status": "authored-unregistered",
            "gate_checks": result["gate_checks"],
            "structural_common_support":
                result["structural_common_support"],
        }, indent=1))
        return
    if not output.exists():
        raise RuntimeError("Qwen mode parser protocol output is missing")
    result = json.loads(output.read_text())
    if not result["all_protocol_gates_pass"]:
        raise RuntimeError("refusing to register a failed mode protocol")
    command = (
        "python -m jspace_phase4.experiments.p4_mode_gate "
        f"--config {arguments.config} --register-existing")
    create(
        config["evidence_id"], tier=config["tier"], command=command,
        what=(
            "Official Qwen thinking on/off template and phase-parser methods "
            "gate: v2 recognizes the thinking-on delimiter opened in prefill; "
            "thinking-off × reasoning is structurally absent, so the primary "
            "mode-by-phase interaction uses prefill and final-answer phases."),
        outputs=[output],
        inputs={
            "config": file_sha256(config_path),
            "chat_template": result["chat_template_sha256"],
            "thinking_on_prompt": result["official_prompts"][
                "thinking_on"]["rendered_prompt_sha256"],
            "thinking_off_prompt": result["official_prompts"][
                "thinking_off"]["rendered_prompt_sha256"],
        })
    print(json.dumps({
        "status": "registered", "evidence_id": config["evidence_id"]},
        indent=1))


if __name__ == "__main__":
    main()
