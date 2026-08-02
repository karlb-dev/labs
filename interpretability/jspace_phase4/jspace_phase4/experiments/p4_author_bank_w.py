"""Programmatically author the fully crossed Phase 4 Bank W candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import yaml

from ..manifests import atomic_json, file_sha256, object_sha256, require_clean_tree
from ..paths4 import resolve_uri
from ..registry4 import create, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true")
    group.add_argument("--register-existing", action="store_true")
    return parser.parse_args()


def stable_seed(namespace: str, *parts: object) -> int:
    text = ":".join([namespace, *(str(value) for value in parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def stable_family_order(values: list[str], namespace: str) -> list[str]:
    return sorted(values, key=lambda value: (
        hashlib.sha256(f"{namespace}:{value}".encode()).digest(), value))


def _labels(count: int, *, prefix: str) -> list[str]:
    return [f"{prefix}{index + 1}" for index in range(count)]


def _render_key_value(labels: list[str], values: list[str], target: int,
                      derivation: str) -> tuple[str, str, list[str]]:
    if derivation == "supplied":
        facts = [f"Key {label} stores {value}."
                 for label, value in zip(labels, values)]
        trace = [f"lookup({labels[target]})={values[target]}"]
    else:
        slots = _labels(len(labels), prefix="S")
        facts = [f"Key {label} points to slot {slot}. Slot {slot} stores {value}."
                 for label, slot, value in zip(labels, slots, values)]
        trace = [f"{labels[target]}->{slots[target]}->{values[target]}"]
    query = f"Question: What value is stored for key {labels[target]}? Answer:"
    return " ".join(facts), query, trace


def _render_state_updates(labels: list[str], values: list[str], target: int,
                          derivation: str) -> tuple[str, str, list[str]]:
    if derivation == "supplied":
        facts = [f"Register {label} now equals {value}."
                 for label, value in zip(labels, values)]
        trace = [f"final({labels[target]})={values[target]}"]
    else:
        starts = list(reversed(values))
        facts = [f"Register {label} starts as {start}; set it to {value}."
                 for label, start, value in zip(labels, starts, values)]
        trace = [f"{labels[target]}:{starts[target]}->{values[target]}"]
    query = f"Question: What is the final value of register {labels[target]}? Answer:"
    return " ".join(facts), query, trace


def _render_graph_path(labels: list[str], values: list[str], target: int,
                       derivation: str) -> tuple[str, str, list[str]]:
    if derivation == "supplied":
        facts = [f"Node {label} reaches terminal {value}."
                 for label, value in zip(labels, values)]
        trace = [f"{labels[target]}->{values[target]}"]
    else:
        middle = _labels(len(labels), prefix="M")
        facts = [f"Node {label} links to {mid}. {mid} reaches terminal {value}."
                 for label, mid, value in zip(labels, middle, values)]
        trace = [f"{labels[target]}->{middle[target]}->{values[target]}"]
    query = f"Question: Which terminal is reached from node {labels[target]}? Answer:"
    return " ".join(facts), query, trace


def _render_stack_queue(labels: list[str], values: list[str], target: int,
                        derivation: str) -> tuple[str, str, list[str]]:
    container = f"C{target + 1}"
    if derivation == "supplied":
        facts = [f"Container C{index + 1} has top item {value}."
                 for index, value in enumerate(values)]
        trace = [f"top({container})={values[target]}"]
    else:
        facts = [
            f"On container C{index + 1}, push spare-{label}, remove "
            f"spare-{label}, then push {value}."
            for index, (label, value) in enumerate(zip(labels, values))]
        trace = [f"execute(C{target + 1})->top={values[target]}"]
    query = f"Question: What is the top item of container {container}? Answer:"
    return " ".join(facts), query, trace


def _render_deferred_recall(labels: list[str], values: list[str], target: int,
                            derivation: str) -> tuple[str, str, list[str]]:
    if derivation == "supplied":
        facts = [f"At marker {label}, remember {value}."
                 for label, value in zip(labels, values)]
        trace = [f"recall({labels[target]})={values[target]}"]
    else:
        destinations = labels[1:] + labels[:1]
        facts = [
            f"Marker {label} points ahead one place to marker {destination}. "
            f"Marker {destination} holds {value}."
            for label, destination, value in zip(labels, destinations, values)]
        trace = [f"{labels[target]}->{destinations[target]}->{values[target]}"]
    query = f"Question: What value should be recalled from marker {labels[target]}? Answer:"
    return " ".join(facts), query, trace


def _render_relational_table(labels: list[str], values: list[str], target: int,
                             derivation: str) -> tuple[str, str, list[str]]:
    if derivation == "supplied":
        facts = [f"Row {label} has result {value}."
                 for label, value in zip(labels, values)]
        trace = [f"row({labels[target]}).result={values[target]}"]
    else:
        foreign = _labels(len(labels), prefix="F")
        facts = [f"Row {label} joins foreign key {key}. Foreign key {key} has result {value}."
                 for label, key, value in zip(labels, foreign, values)]
        trace = [f"row({labels[target]})->{foreign[target]}->{values[target]}"]
    query = f"Question: What result joins to row {labels[target]}? Answer:"
    return " ".join(facts), query, trace


RENDERERS = {
    "key_value": _render_key_value,
    "state_updates": _render_state_updates,
    "graph_path": _render_graph_path,
    "stack_queue": _render_stack_queue,
    "deferred_recall": _render_deferred_recall,
    "relational_table": _render_relational_table,
}


def render_case(superfamily: str, *, family_index: int, seed_index: int,
                load_n: int, derivation: str, redundancy: str,
                alphabet: list[str], namespace: str) -> dict:
    generator = np.random.default_rng(stable_seed(
        namespace, superfamily, family_index, seed_index))
    labels = _labels(load_n, prefix=chr(65 + family_index % 12))
    answer = alphabet[(family_index + seed_index) % len(alphabet)]
    other = [value for value in alphabet if value != answer]
    generator.shuffle(other)
    target = seed_index % load_n
    values = other[:load_n]
    values[target] = answer
    # Balance answer locations exactly across template × seed combinations,
    # while independently shuffling all non-target entries.
    desired_position = (family_index + seed_index) % load_n
    order = [index for index in range(load_n) if index != target]
    generator.shuffle(order)
    order.insert(desired_position, target)
    labels = [labels[index] for index in order]
    values = [values[index] for index in order]
    target = desired_position
    body, query, trace = RENDERERS[superfamily](
        labels, values, target, derivation)
    if redundancy == "redundant":
        # Repeat the entire state, not only the target, so frequency cannot
        # identify the requested answer while external redundancy is real.
        body = body + " Summary repetition: " + body
    return {
        "body": body,
        "query": query,
        "answer": answer,
        "context_values": values,
        "target_position": int(target),
        "solution_trace": trace,
    }


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def pad_case(case: dict, *, tokenizer, target_tokens: int,
             filler_sentence: str, maximum_iterations: int) -> dict:
    body = case["body"]
    prompt = f"{body} {case['query']}"
    iterations = 0
    # Tokenization at a text boundary is not generally additive. Prefer the
    # configured filler, then successively finer neutral fragments so a
    # sentence-sized tokenization step cannot violate the matching tolerance.
    fragments = tuple(dict.fromkeys(
        (filler_sentence, " Neutral.", " N.", " .")))
    while _token_count(tokenizer, prompt) < target_tokens:
        current = _token_count(tokenizer, prompt)
        candidates = []
        for fragment in fragments:
            candidate_body = body + fragment
            candidate_prompt = f"{candidate_body} {case['query']}"
            count = _token_count(tokenizer, candidate_prompt)
            if count > current:
                candidates.append((count, len(fragment), candidate_body,
                                   candidate_prompt))
        if not candidates:
            raise RuntimeError("Bank W length padding made no token progress")
        non_overshooting = [row for row in candidates
                            if row[0] <= target_tokens]
        if non_overshooting:
            _, _, body, prompt = max(non_overshooting,
                                     key=lambda row: (row[0], -row[1]))
        else:
            _, _, body, prompt = min(candidates,
                                     key=lambda row: (row[0], row[1]))
        iterations += 1
        if iterations >= maximum_iterations:
            raise RuntimeError("Bank W length padding did not converge")
    return {**case, "prompt": prompt, "padding_sentences": iterations,
            "prompt_tokens": _token_count(tokenizer, prompt)}


def _query_contains_answer(row: Mapping) -> bool:
    query = re.sub(r"[^a-z0-9]+", " ", row["query"].lower()).strip()
    answer = re.sub(r"[^a-z0-9]+", " ", row["answer"].lower()).strip()
    return bool(answer and re.search(rf"\b{re.escape(answer)}\b", query))


def shortcut_audit(rows: list[dict], *, maximum_excess: float) -> dict:
    frame = []
    for row in rows:
        values = row["context_values"]
        counts = {value: values.count(value) for value in set(values)}
        maximum_count = max(counts.values())
        most_frequent = [value for value, count in counts.items()
                         if count == maximum_count]
        frame.append({
            "load": row["load"],
            "n": len(values),
            "first": values[0] == row["answer"],
            "last": values[-1] == row["answer"],
            # A frequency-only classifier must randomize across tied modes.
            "lexical_frequency": (
                1 / len(most_frequent)
                if row["answer"] in most_frequent else 0.0),
        })
    metrics = {}
    for load in sorted({row["load"] for row in frame}):
        subset = [row for row in frame if row["load"] == load]
        chance = float(np.mean([1 / row["n"] for row in subset]))
        metrics[load] = {"chance": chance}
        for heuristic in ("first", "last", "lexical_frequency"):
            accuracy = float(np.mean([row[heuristic] for row in subset]))
            metrics[load][heuristic] = {
                "accuracy": accuracy,
                "excess_over_load_chance": accuracy - chance,
                "passes": bool(accuracy - chance <= maximum_excess),
            }
    return {
        "by_load": metrics,
        "all_pass": all(
            value[heuristic]["passes"]
            for value in metrics.values()
            for heuristic in ("first", "last", "lexical_frequency")),
    }


def validate_power_binding(config: Mapping) -> dict | None:
    """Verify an optional registered power result before repartitioning."""
    binding = config.get("power_result")
    if binding is None:
        return None
    path = Path(binding["path"])
    actual_sha = file_sha256(path)
    if actual_sha != binding["sha256"]:
        raise RuntimeError("Bank W power-result hash mismatch")
    event = resolve(binding["evidence_id"])
    registered = {row["sha256"] for row in event["outputs"]}
    if actual_sha not in registered:
        raise RuntimeError("Bank W power result is not registered")
    result = json.loads(path.read_text())
    if result.get("evidence_id") != binding["evidence_id"]:
        raise RuntimeError("Bank W power result evidence ID mismatch")
    decision = result["decision"]
    licensed = int(binding["licensed_minimum_common_families"])
    if decision["minimum_common_families_for_power_target"] != licensed:
        raise RuntimeError("Bank W licensed family count does not match power result")
    requested = int(config["partition"]["confirmatory_families"])
    if requested != licensed:
        raise RuntimeError("Bank W confirmatory count is not power-licensed")
    power = decision["minimum_power_at_sesoi_by_common_family_count"].get(
        str(requested))
    target = float(result["simulation"]["power_target"])
    if power is None or float(power) < target:
        raise RuntimeError("Bank W requested confirmatory count is underpowered")
    return {
        "evidence_id": binding["evidence_id"],
        "path": str(path),
        "sha256": actual_sha,
        "licensed_minimum_common_families": licensed,
        "minimum_power_at_licensed_count": float(power),
        "power_target": target,
        "conservative_alpha": float(result["primary"]["alpha"]),
    }


def validate_development_reference(config: Mapping, rows: list[dict]) -> dict | None:
    """Prove a repartition leaves the already-consumed development rows fixed."""
    reference = config.get("development_rows_reference")
    if reference is None:
        return None
    path = Path(reference["path"])
    actual_file_sha = file_sha256(path)
    if actual_file_sha != reference["file_sha256"]:
        raise RuntimeError("Bank W development-reference file hash mismatch")
    previous = [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]
    previous_development = [
        row for row in previous if row["partition"] == "development"]
    current_development = [
        row for row in rows if row["partition"] == "development"]
    previous_sha = object_sha256(previous_development)
    current_sha = object_sha256(current_development)
    if previous_sha != reference["development_rows_sha256"]:
        raise RuntimeError("Bank W development-reference payload hash mismatch")
    if current_sha != previous_sha:
        raise RuntimeError("Bank W repartition changed consumed development rows")
    return {
        "path": str(path),
        "file_sha256": actual_file_sha,
        "n_development_rows": len(current_development),
        "development_rows_sha256": current_sha,
        "byte_identical_payload": True,
    }


def build_candidate(config: Mapping, *, tokenizer) -> tuple[list[dict], dict, dict]:
    family_ids = [
        f"{superfamily}:template-{index:02d}"
        for superfamily in config["superfamilies"]
        for index in range(int(config["families_per_superfamily"]))
    ]
    ordered = stable_family_order(
        family_ids, config["partition"]["namespace"])
    requested_partition = {
        "development": int(config["partition"]["development_families"]),
        "confirmatory": int(config["partition"]["confirmatory_families"]),
        "replication": int(config["partition"]["replication_families"]),
    }
    if any(value <= 0 for value in requested_partition.values()):
        raise ValueError("Bank W partition counts must all be positive")
    if sum(requested_partition.values()) != len(ordered):
        raise ValueError("Bank W partition counts must exhaust the bank exactly")
    power_binding = validate_power_binding(config)
    first = requested_partition["development"]
    second = first + requested_partition["confirmatory"]
    partition = {
        "development": ordered[:first],
        "confirmatory": ordered[first:second],
        "replication": ordered[second:],
    }
    side_by_family = {
        family: side for side, families in partition.items()
        for family in families}
    rows = []
    for family_id in ordered:
        superfamily, template = family_id.split(":")
        family_index = int(template.split("-")[-1])
        for seed_index in range(int(config["seeds_per_family"])):
            cases = []
            for load, load_n in config["loads"].items():
                for derivation in config["derivations"]:
                    for redundancy in config["redundancies"]:
                        case = render_case(
                            superfamily, family_index=family_index,
                            seed_index=seed_index, load_n=int(load_n),
                            derivation=derivation, redundancy=redundancy,
                            alphabet=list(config["answer_alphabet"]),
                            namespace=config["namespace"])
                        cases.append({
                            **case, "load": load,
                            "load_n": int(load_n),
                            "derivation": derivation,
                            "redundancy": redundancy,
                        })
            target_tokens = max(_token_count(
                tokenizer, f"{case['body']} {case['query']}")
                for case in cases)
            for case in cases:
                padded = pad_case(
                    case, tokenizer=tokenizer, target_tokens=target_tokens,
                    filler_sentence=config["length_matching"][
                        "filler_sentence"],
                    maximum_iterations=int(config["length_matching"][
                        "maximum_padding_iterations"]))
                item_id = (
                    f"bank-w:{family_id}:seed-{seed_index:02d}:"
                    f"{padded['load']}:{padded['derivation']}:"
                    f"{padded['redundancy']}")
                rows.append({
                    "schema_version": 1,
                    "item_id": item_id,
                    "canonical_family": family_id,
                    "superfamily": superfamily,
                    "template_index": family_index,
                    "item_seed": seed_index,
                    "partition": side_by_family[family_id],
                    "load": padded["load"],
                    "load_n": padded["load_n"],
                    "derivation": padded["derivation"],
                    "redundancy": padded["redundancy"],
                    "prompt": padded["prompt"],
                    "query": padded["query"],
                    "answer": padded["answer"],
                    "accepted_answers": [f" {padded['answer']}"],
                    "prompt_token_ids": [int(value) for value in tokenizer(
                        padded["prompt"], add_special_tokens=False).input_ids],
                    "answer_token_ids": [int(value) for value in tokenizer(
                        f" {padded['answer']}",
                        add_special_tokens=False).input_ids],
                    "prompt_token_count": padded["prompt_tokens"],
                    "padding_sentences": padded["padding_sentences"],
                    "context_values": padded["context_values"],
                    "target_position": padded["target_position"],
                    "solution_trace": padded["solution_trace"],
                    "answer_in_query": _query_contains_answer(padded),
                })
    length_spans = {}
    for key in {(row["canonical_family"], row["item_seed"])
                for row in rows}:
        values = [row["prompt_token_count"] for row in rows
                  if (row["canonical_family"], row["item_seed"]) == key]
        length_spans[f"{key[0]}:seed-{key[1]:02d}"] = max(values) - min(values)
    shortcuts = shortcut_audit(
        rows, maximum_excess=float(config["shortcut_audit"][
            "maximum_excess_over_load_chance"]))
    condition_counts = {}
    for row in rows:
        key = f"{row['load']}:{row['derivation']}:{row['redundancy']}"
        condition_counts[key] = condition_counts.get(key, 0) + 1
    target_position_counts = {}
    for row in rows:
        counts = target_position_counts.setdefault(
            row["load"], {str(index): 0 for index in range(row["load_n"])})
        counts[str(row["target_position"])] += 1
    target_positions_balanced = all(
        max(counts.values()) == min(counts.values())
        for counts in target_position_counts.values())
    development_reference = validate_development_reference(config, rows)
    audit = {
        "schema_version": 1,
        "n_superfamilies": len(set(row["superfamily"] for row in rows)),
        "n_families": len(set(row["canonical_family"] for row in rows)),
        "n_item_seeds": len({
            (row["canonical_family"], row["item_seed"]) for row in rows}),
        "n_rows": len(rows),
        "condition_counts": condition_counts,
        "partition_counts": {side: len(families)
                             for side, families in partition.items()},
        "partition_row_payload_sha256": {
            side: object_sha256([
                row for row in rows if row["partition"] == side])
            for side in ("development", "confirmatory", "replication")
        },
        "maximum_within_seed_prompt_token_span": max(length_spans.values()),
        "length_match_pass": bool(max(length_spans.values()) <= int(
            config["length_matching"][
                "maximum_pair_difference_tokens"])),
        "answer_in_query_count": sum(row["answer_in_query"] for row in rows),
        "shortcut_audit": shortcuts,
        "target_position_counts": target_position_counts,
        "target_positions_balanced": target_positions_balanced,
        "all_axes_fully_crossed": all(
            value == int(config["families_per_superfamily"])
            * len(config["superfamilies"])
            * int(config["seeds_per_family"])
            for value in condition_counts.values()),
        "programmatic_solution_trace_present": all(
            row["solution_trace"] for row in rows),
        "family_id_is_template_not_seed": all(
            "seed" not in row["canonical_family"] for row in rows),
        "partition_family_disjoint": not (
            set(partition["development"]) & set(partition["confirmatory"])
            or set(partition["development"]) & set(partition["replication"])
            or set(partition["confirmatory"]) & set(partition["replication"])),
        "capability_guard": dict(config["capability_guard"]),
        "primary": dict(config["primary"]),
        "power_evidence_id": config.get("power_evidence_id"),
        "power_binding": power_binding,
        "development_rows_reference": development_reference,
        "bank_rows_sha256": object_sha256(rows),
        "partition_sha256": object_sha256(partition),
        "freeze_ready": False,
        "freeze_blocker": config.get(
            "freeze_blocker",
            "Model-specific baseline capability and common-support gates, "
            "power/SESOI, and independent review remain pending."),
    }
    required = [
        audit["n_superfamilies"] == 6,
        audit["n_families"] == 72,
        audit["length_match_pass"],
        audit["answer_in_query_count"] == 0,
        audit["shortcut_audit"]["all_pass"],
        (audit["target_positions_balanced"]
         if config["shortcut_audit"].get(
             "require_balanced_target_positions", False) else True),
        audit["all_axes_fully_crossed"],
        audit["programmatic_solution_trace_present"],
        audit["family_id_is_template_not_seed"],
        audit["partition_family_disjoint"],
        audit["partition_counts"] == requested_partition,
        (power_binding is not None
         if config.get("power_result") is not None else True),
        (development_reference is not None
         if config.get("development_rows_reference") is not None else True),
    ]
    if not all(required):
        raise RuntimeError("Bank W authoring audit failed: "
                           + json.dumps(audit, sort_keys=True))
    return rows, partition, audit


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    outputs = {key: Path(value) for key, value in config["outputs"].items()}
    if arguments.generate:
        import transformers
        model_path = resolve_uri(config["model_uri"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
        rows, partition, audit = build_candidate(config, tokenizer=tokenizer)
        audit.update({
            "evidence_id": config["evidence_id"],
            "generation_code_commit": clean["code_commit"],
            "config_sha256": file_sha256(config_path),
            "tokenizer_name_or_path": str(model_path),
        })
        _write_jsonl(outputs["bank"], rows)
        atomic_json(outputs["partition"], {
            "schema_version": 1, "payload": partition,
            "payload_sha256": object_sha256(partition),
        })
        atomic_json(outputs["audit"], audit)
        print(json.dumps({
            "status": "generated-unregistered-candidate",
            "audit": audit,
        }, indent=1))
        return
    required = [outputs[name] for name in ("bank", "partition", "audit")]
    if not all(path.exists() for path in required):
        raise RuntimeError("Bank W candidate outputs are incomplete")
    audit = json.loads(outputs["audit"].read_text())
    command = (
        "python -m jspace_phase4.experiments.p4_author_bank_w "
        f"--config {arguments.config} --register-existing")
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            f"Outcome-blind Bank W candidate: {audit['n_families']} "
            f"template families across six superfamilies and "
            f"{audit['n_rows']} fully crossed load × derivation × "
            "redundancy rows; length and shortcut audits pass, while the "
            "model-specific capability gate remains a freeze blocker."),
        command=command, outputs=required,
        inputs={
            "config": file_sha256(config_path),
            "bank_rows": audit["bank_rows_sha256"],
            "partition": audit["partition_sha256"],
            **({"power_result": audit["power_binding"]["sha256"]}
               if audit.get("power_binding") else {}),
            **({"development_reference": audit[
                "development_rows_reference"]["file_sha256"]}
               if audit.get("development_rows_reference") else {}),
        },
    )
    print(json.dumps({
        "status": "registered", "evidence_id": config["evidence_id"],
    }, indent=1))


if __name__ == "__main__":
    main()
