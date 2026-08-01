"""Prospectively revise the registered Bank B v1 metadata candidate.

The v1 candidate and its registered independent audit are immutable inputs.
This producer applies only the outcome-blind country/relation corrections
listed in the v2 configuration, expands genuinely co-valid answer aliases,
rebuilds prompts and exact tokenizer IDs, and writes a new candidate.  It
never reads model or intervention outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from ..manifests import atomic_json, file_sha256, object_sha256, require_clean_tree
from ..paths4 import resolve_uri
from ..registry4 import create, supersede
from .p4_author_bank_b import (
    _prior_entities,
    _relation_prompts,
    _template_hash,
    _token_ids,
    _write_jsonl,
    normalize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--generate", action="store_true")
    modes.add_argument("--register-existing", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def _strip_alias(value: str) -> str:
    return str(value).strip()


def _accepted(specification: Mapping | None, fallback: str) -> list[str]:
    values = (
        specification.get("accepted", [])
        if specification is not None else []
    )
    canonical = (
        str(specification["canonical"])
        if specification is not None else str(fallback)
    )
    ordered: list[str] = []
    for value in [canonical, *values]:
        cleaned = _strip_alias(value)
        if cleaned and normalize(cleaned) not in {
                normalize(existing) for existing in ordered}:
            ordered.append(cleaned)
    return ordered


def _country_spec(config: Mapping, country: str) -> Mapping:
    return config["corrections"].get(country, {})


def corrected_country(config: Mapping, country: str) -> str:
    return str(_country_spec(config, country).get("canonical_name", country))


def corrected_value(
        config: Mapping, country: str, relation: str,
        fallback: str) -> tuple[str, list[str]]:
    relation_spec = _country_spec(config, country).get(
        "relations", {}).get(relation)
    canonical = (
        str(relation_spec["canonical"])
        if relation_spec is not None else str(fallback)
    )
    return canonical, _accepted(relation_spec, fallback)


def _prefixed(values: Iterable[str]) -> list[str]:
    return [f" {value}" for value in values]


def _token_aliases(tokenizer, values: Sequence[str]) -> list[list[int]]:
    return [_token_ids(tokenizer, value) for value in values]


def _prefix_disjoint(alias_ids: Sequence[Sequence[int]]) -> bool:
    for left_index, left in enumerate(alias_ids):
        for right_index, right in enumerate(alias_ids):
            if left_index == right_index:
                continue
            if len(left) <= len(right) and list(left) == list(right[:len(left)]):
                return False
    return True


def revise_row(row: Mapping, *, config: Mapping, tokenizer) -> dict:
    old_bridge = str(row["bridge"])
    bridge = corrected_country(config, old_bridge)
    source, _ = corrected_value(
        config, old_bridge, str(row["source_type"]), str(row["source"]))
    answer, answer_aliases = corrected_value(
        config, old_bridge, str(row["answer_type"]), str(row["answer"]))
    alternate, alternate_aliases = corrected_value(
        config, old_bridge, str(row["alternate_relation"]),
        str(row["alternate_answer"]))

    counterfactuals = []
    for candidate in row["counterfactuals"]:
        old_country = str(candidate["bridge"])
        cf_answer, cf_answer_aliases = corrected_value(
            config, old_country, str(row["answer_type"]),
            str(candidate["answer"]))
        cf_alternate, cf_alternate_aliases = corrected_value(
            config, old_country, str(row["alternate_relation"]),
            str(candidate["alternate_answer"]))
        counterfactuals.append({
            "bridge": corrected_country(config, old_country),
            "answer": cf_answer,
            "accepted_answers": _prefixed(cf_answer_aliases),
            "alternate_answer": cf_alternate,
            "alternate_accepted_answers": _prefixed(cf_alternate_aliases),
        })

    prompts = _relation_prompts(
        str(row["source_type"]), source, bridge,
        str(row["answer_type"]), counterfactuals[0]["bridge"])
    alternate_prompts = _relation_prompts(
        str(row["source_type"]), source, bridge,
        str(row["alternate_relation"]), counterfactuals[0]["bridge"])
    values = [source, bridge, answer, alternate] + [
        value
        for candidate in counterfactuals
        for value in (
            candidate["bridge"], candidate["answer"],
            candidate["alternate_answer"])
    ]
    fact_id = (
        f"bank-b-v2:{row['canonical_family']}:"
        f"{hashlib.sha256(bridge.encode()).hexdigest()[:12]}"
    )
    answer_alias_ids = _token_aliases(tokenizer, answer_aliases)
    alternate_alias_ids = _token_aliases(tokenizer, alternate_aliases)
    counterfactual_answer_alias_ids = [
        _token_aliases(tokenizer, [
            _strip_alias(value) for value in candidate["accepted_answers"]])
        for candidate in counterfactuals
    ]
    counterfactual_alternate_alias_ids = [
        _token_aliases(tokenizer, [
            _strip_alias(value)
            for value in candidate["alternate_accepted_answers"]])
        for candidate in counterfactuals
    ]
    return {
        **dict(row),
        "schema_version": 2,
        "fact_id": fact_id,
        "supersedes_fact_id": row["fact_id"],
        "source": source,
        "bridge": bridge,
        "answer": answer,
        "accepted_answers": _prefixed(answer_aliases),
        "alternate_answer": alternate,
        "alternate_accepted_answers": _prefixed(alternate_aliases),
        "counterfactuals": counterfactuals,
        "unrelated_bridge": corrected_country(
            config, str(row["unrelated_bridge"])),
        "prompts": prompts,
        "alternate_prompts": alternate_prompts,
        "template_hashes": {
            key: _template_hash(prompt, values)
            for key, prompt in prompts.items()
        },
        "token_ids": {
            "source": _token_ids(tokenizer, source),
            "bridge": _token_ids(tokenizer, bridge),
            "answer": _token_ids(tokenizer, answer),
            "answer_aliases": answer_alias_ids,
            "alternate_answer": _token_ids(tokenizer, alternate),
            "alternate_answer_aliases": alternate_alias_ids,
            "counterfactual_bridges": [
                _token_ids(tokenizer, candidate["bridge"])
                for candidate in counterfactuals
            ],
            "counterfactual_answers": [
                _token_ids(tokenizer, candidate["answer"])
                for candidate in counterfactuals
            ],
            "counterfactual_answer_aliases":
                counterfactual_answer_alias_ids,
            "counterfactual_alternate_answer_aliases":
                counterfactual_alternate_alias_ids,
            "unrelated_bridge": _token_ids(
                tokenizer, corrected_country(
                    config, str(row["unrelated_bridge"]))),
        },
        "source_verification_status": (
            "candidate-corrected-pending-independent-reverification"),
    }


def _entity_values(row: Mapping) -> list[str]:
    values = [
        row["source"], row["bridge"], row["answer"],
        row["alternate_answer"], row["unrelated_bridge"],
        *row["accepted_answers"], *row["alternate_accepted_answers"],
    ]
    for candidate in row["counterfactuals"]:
        values.extend([
            candidate["bridge"], candidate["answer"],
            candidate["alternate_answer"],
            *candidate["accepted_answers"],
            *candidate["alternate_accepted_answers"],
        ])
    return [str(value) for value in values]


def build_revision(config: Mapping, *, tokenizer) -> tuple[
        list[dict], list[dict], dict, dict]:
    inputs = config["base_candidate"]
    paths = {
        key: resolve_uri(inputs[f"{key}_uri"])
        for key in ("bank", "sources", "partition", "audit")
    }
    for key, path in paths.items():
        expected = inputs[f"{key}_sha256"]
        if file_sha256(path) != expected:
            raise RuntimeError(f"Bank B v1 {key} hash drift")
    base_rows = load_jsonl(paths["bank"])
    base_sources = load_jsonl(paths["sources"])
    source_by_fact = {row["fact_id"]: row for row in base_sources}
    if len(base_rows) != 160 or len(source_by_fact) != 160:
        raise RuntimeError("Bank B v1 row-count drift")

    rows = [revise_row(row, config=config, tokenizer=tokenizer)
            for row in base_rows]
    old_to_new = {
        row["supersedes_fact_id"]: row["fact_id"] for row in rows}
    source_rows = []
    for row in base_sources:
        source_rows.append({
            **dict(row),
            "schema_version": 2,
            "fact_id": old_to_new[row["fact_id"]],
            "supersedes_fact_id": row["fact_id"],
            "verification_status": (
                "candidate-corrected-pending-independent-reverification"),
            "ambiguity_notes": [
                *row["ambiguity_notes"],
                "Candidate v2 applies only the prospectively enumerated "
                "metadata corrections and accepted-alias expansions from "
                "the registered v1 independent audit.",
                "The v1 audit is a correction input, not a substitute for "
                "the full v2 independent reverification.",
            ],
            "correction_audit": {
                "evidence_id": config["base_verification"]["evidence_id"],
                "rows_sha256": config["base_verification"]["rows_sha256"],
                "result_sha256": config["base_verification"][
                    "result_sha256"],
            },
        })

    partition_envelope = json.loads(paths["partition"].read_text())
    partition = partition_envelope["payload"]
    prior_config = yaml.safe_load(resolve_uri(
        config["base_author_config"]["uri"]).read_text())
    if file_sha256(resolve_uri(config["base_author_config"]["uri"])) != \
            config["base_author_config"]["sha256"]:
        raise RuntimeError("Bank B v1 author config hash drift")
    prior, prior_contract = _prior_entities(prior_config)
    prior_overlap = sorted({
        normalize(value) for row in rows for value in _entity_values(row)
        if normalize(value) in prior
    })
    entity_sets = {
        side: {
            normalize(value)
            for row in rows if row["partition"] == side
            for value in _entity_values(row)
        }
        for side in partition
    }
    cross_partition = {
        f"{left}_vs_{right}": sorted(entity_sets[left] & entity_sets[right])
        for index, left in enumerate(partition)
        for right in list(partition)[index + 1:]
    }
    alias_sets = []
    for row in rows:
        alias_sets.extend([
            row["token_ids"]["answer_aliases"],
            row["token_ids"]["alternate_answer_aliases"],
            *row["token_ids"]["counterfactual_answer_aliases"],
            *row["token_ids"]["counterfactual_alternate_answer_aliases"],
        ])
    audit = {
        "schema_version": 2,
        "evidence_id": config["evidence_id"],
        "supersedes_evidence_id": inputs["evidence_id"],
        "n_families": len({row["canonical_family"] for row in rows}),
        "n_facts": len(rows),
        "partition_counts": {
            side: len(families) for side, families in partition.items()},
        "facts_per_family": {
            family: sum(row["canonical_family"] == family for row in rows)
            for family in sorted({row["canonical_family"] for row in rows})
        },
        "unique_true_bridges": len({row["bridge"] for row in rows}),
        "prior_exact_entity_overlap": prior_overlap,
        "cross_partition_entity_overlap": cross_partition,
        "two_second_hop_relations_per_fact": all(
            row["answer_type"] != row["alternate_relation"] for row in rows),
        "two_counterfactual_bridges_per_fact": all(
            len(row["counterfactuals"]) == 2 for row in rows),
        "all_alias_sets_prefix_disjoint": all(
            _prefix_disjoint(values) for values in alias_sets),
        "corrected_country_count": len(config["corrections"]),
        "correction_countries": sorted(config["corrections"]),
        "source_status_counts": {
            "candidate-corrected-pending-independent-reverification":
                len(source_rows)},
        "independently_reverified_facts": 0,
        "freeze_ready": False,
        "freeze_blocker": (
            "All v2 rows require full independent reverification, a "
            "P4-P1 power/SESOI ruler, and independent protocol review."),
        "outcome_blinding": (
            "Country metadata and registered methods audit only; no model "
            "or intervention outcome was loaded."),
        "prior_contract": prior_contract,
        "bank_rows_sha256": object_sha256(rows),
        "source_rows_sha256": object_sha256(source_rows),
        "partition_sha256": object_sha256(partition),
    }
    if prior_overlap or any(cross_partition.values()):
        raise RuntimeError("Bank B v2 overlap audit failed")
    if audit["n_families"] != 40 or audit["n_facts"] != 160 \
            or audit["unique_true_bridges"] != 160:
        raise RuntimeError("Bank B v2 size/uniqueness invariant failed")
    if not audit["all_alias_sets_prefix_disjoint"]:
        raise RuntimeError("Bank B v2 accepted aliases are not prefix-disjoint")
    return rows, source_rows, partition, audit


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    outputs = {name: Path(path) for name, path in config["outputs"].items()}
    if arguments.generate:
        import transformers

        model_path = resolve_uri(config["model_uri"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
        bank, sources, partition, audit = build_revision(
            config, tokenizer=tokenizer)
        audit.update({
            "generation_code_commit": clean["code_commit"],
            "config_sha256": file_sha256(config_path),
            "tokenizer_name_or_path": str(model_path),
        })
        _write_jsonl(outputs["bank"], bank)
        _write_jsonl(outputs["sources"], sources)
        atomic_json(outputs["partition"], {
            "schema_version": 2,
            "payload": partition,
            "payload_sha256": object_sha256(partition),
            "supersedes": config["base_candidate"]["partition_sha256"],
        })
        atomic_json(outputs["audit"], audit)
        print(json.dumps({
            "status": "generated-unregistered-candidate",
            "evidence_id": config["evidence_id"],
            "audit": audit,
        }, indent=1, ensure_ascii=False))
        return

    required = [outputs[name] for name in (
        "bank", "sources", "partition", "audit", "schema")]
    if not all(path.exists() for path in required):
        raise RuntimeError("Bank B v2 candidate outputs are incomplete")
    audit = json.loads(outputs["audit"].read_text())
    if audit["n_families"] != 40 or audit["n_facts"] != 160:
        raise RuntimeError("Bank B v2 candidate size drift")
    command = (
        "python -m jspace_phase4.experiments.p4_revise_bank_b "
        f"--config {arguments.config} --register-existing")
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            "Outcome-blind Bank B candidate v2: all registered v1 "
            "independent-audit mismatches are prospectively corrected, "
            "co-valid native-name aliases are explicit, and all 40 "
            "two-relation families and frozen partitions are retained; "
            "full v2 reverification and power remain blockers."),
        command=command, outputs=required,
        inputs={
            "config": file_sha256(config_path),
            "base_bank": config["base_candidate"]["bank_sha256"],
            "base_sources": config["base_candidate"]["sources_sha256"],
            "base_partition": config["base_candidate"]["partition_sha256"],
            "base_audit": config["base_candidate"]["audit_sha256"],
            "base_verification_rows": config["base_verification"][
                "rows_sha256"],
            "base_verification_result": config["base_verification"][
                "result_sha256"],
            "bank_rows": audit["bank_rows_sha256"],
            "source_rows": audit["source_rows_sha256"],
            "partition": audit["partition_sha256"],
        },
    )
    supersede(
        config["base_candidate"]["evidence_id"], config["evidence_id"],
        reason=(
            "Registered independent audit found 18 source mismatches and "
            "21 unresolved ambiguity-review rows; v2 prospectively "
            "corrects or explicitly covers every root case."))
    print(json.dumps({
        "status": "registered-and-v1-superseded",
        "evidence_id": config["evidence_id"],
        "outputs": {str(path): file_sha256(path) for path in required},
    }, indent=1))


if __name__ == "__main__":
    main()
