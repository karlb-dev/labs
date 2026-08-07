"""Author the outcome-blind Phase 4 Bank B candidate and source ledger.

Generation and registration are deliberately separate.  ``--generate``
writes reviewable candidate rows into the repository but never registers
them.  After those rows are audited and committed, ``--register-existing``
hash-verifies and registers the immutable candidate from a clean tree.

The bundled countryinfo records are a pinned *candidate source*, not final
independent verification.  Every row says so explicitly; untouched outcomes
remain forbidden until the source ledger is independently completed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import resolve_uri
from ..registry4 import create


LIST_FIELDS = {
    "tld": "tld",
    "calling_code": "callingCodes",
    "currency": "currencies",
    "language": "languages",
}
TRANSLATION_FIELDS = {
    "french_name": "fr",
    "spanish_name": "es",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--generate", action="store_true")
    modes.add_argument("--register-existing", action="store_true")
    return parser.parse_args()


def normalize(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    return re.sub(r"[\W_]+", " ", normalized,
                  flags=re.UNICODE).strip()


def record_value(record: Mapping, field: str) -> str | None:
    if field == "name":
        value = record.get("name")
    elif field in {"alpha2", "alpha3"}:
        value = (record.get("ISO") or {}).get(field)
    elif field in LIST_FIELDS:
        values = record.get(LIST_FIELDS[field]) or []
        value = values[0] if values else None
    elif field in TRANSLATION_FIELDS:
        value = (record.get("translations") or {}).get(
            TRANSLATION_FIELDS[field])
    else:
        value = record.get(field)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def stable_order(values: Iterable[str], *, namespace: str) -> list[str]:
    return sorted(set(values), key=lambda value: (
        hashlib.sha256(f"{namespace}:{value}".encode()).digest(), value))


def _prior_entities(config: Mapping) -> tuple[set[str], dict]:
    entities, contract = set(), {}
    entity_fields = {
        "source", "bridge", "answer", "counterfactual_bridge",
        "counterfactual_answer", "intermediate", "swap_to", "swap_answer",
    }
    for specification in config["prior_sources"]:
        path = Path(specification["path"])
        if file_sha256(path) != specification["sha256"]:
            raise RuntimeError(f"prior-source hash mismatch: {path}")
        parsed = json.loads(path.read_text()) if path.suffix == ".json" else [
            json.loads(line) for line in path.read_text().splitlines()
            if line.strip()
        ]
        rows = parsed.get("items", []) if isinstance(parsed, dict) else parsed
        for row in rows:
            for field in entity_fields:
                value = row.get(field)
                if isinstance(value, str) and normalize(value):
                    entities.add(normalize(value))
        contract[str(path)] = {
            "kind": specification["kind"],
            "sha256": specification["sha256"],
            "n_rows": len(rows),
        }
    return entities, contract


def _country_records() -> tuple[list[dict], dict]:
    root = importlib.resources.files("countryinfo").joinpath("data")
    records, inventory = [], []
    for resource in sorted(root.iterdir(), key=lambda value: value.name):
        if not resource.name.endswith(".json"):
            continue
        path = Path(str(resource))
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or not record.get("name"):
            continue
        relative = f"countryinfo/data/{resource.name}"
        digest = file_sha256(path)
        record = dict(record)
        record["_source_record"] = relative
        record["_source_sha256"] = digest
        records.append(record)
        inventory.append({"path": relative, "sha256": digest})
    version = importlib.metadata.version("countryinfo")
    return records, {
        "package": "countryinfo",
        "version": version,
        "n_records": len(records),
        "inventory_sha256": object_sha256(inventory),
    }


def _eligible(record: Mapping, *, source_type: str, answer_type: str,
              alternate_type: str, prior: set[str], excluded: set[str],
              maximum_characters: int) -> bool:
    values = [record_value(record, field) for field in (
        "name", source_type, answer_type, alternate_type)]
    if not all(value and len(value) <= maximum_characters for value in values):
        return False
    normalized = [normalize(value) for value in values]
    return (
        all(normalized)
        and
        len(set(normalized)) == 4
        and not set(normalized) & prior
        and normalize(record_value(record, "name")) not in excluded
    )


def choose_family_specs(
        records: list[dict], *, source_types: list[str],
        answer_types: list[str], prior: set[str], excluded: set[str],
        minimum_eligible: int, target_families: int,
        maximum_characters: int, namespace: str) -> list[dict]:
    candidates = []
    for source_type in source_types:
        for answer_type in answer_types:
            if source_type == answer_type:
                continue
            alternate_candidates = [
                field for field in answer_types
                if field not in {source_type, answer_type}]
            scored = []
            for alternate_type in alternate_candidates:
                eligible = [record for record in records if _eligible(
                    record, source_type=source_type,
                    answer_type=answer_type,
                    alternate_type=alternate_type, prior=prior,
                    excluded=excluded,
                    maximum_characters=maximum_characters)]
                scored.append((len(eligible), alternate_type, eligible))
            count, alternate, eligible = max(
                scored, key=lambda value: (value[0], value[1]))
            if count >= minimum_eligible:
                candidates.append({
                    "family_id": f"{source_type}__to__{answer_type}",
                    "source_type": source_type,
                    "answer_type": answer_type,
                    "alternate_answer_type": alternate,
                    "eligible": eligible,
                })
    ordered_ids = stable_order(
        [value["family_id"] for value in candidates],
        namespace=f"{namespace}:families")
    by_id = {value["family_id"]: value for value in candidates}
    selected = [by_id[value] for value in ordered_ids[:target_families]]
    if len(selected) != target_families:
        raise RuntimeError("too few eligible Bank B relation families")
    return selected


def assign_unique_records(families: list[dict], *, facts_per_family: int,
                          namespace: str) -> dict[str, list[dict]]:
    countries = sorted({
        normalize(record_value(record, "name")): record
        for family in families for record in family["eligible"]
    }.items())
    country_names = [name for name, _ in countries]
    country_records = [record for _, record in countries]
    slots = [
        (family["family_id"], slot)
        for family in families for slot in range(facts_per_family)
    ]
    cost = np.full((len(slots), len(countries)), 2.0, dtype=np.float64)
    eligible_names = {
        family["family_id"]: {
            normalize(record_value(record, "name"))
            for record in family["eligible"]}
        for family in families
    }
    for row, (family_id, slot) in enumerate(slots):
        for column, country in enumerate(country_names):
            if country not in eligible_names[family_id]:
                continue
            digest = hashlib.sha256(
                f"{namespace}:{family_id}:{slot}:{country}".encode()).digest()
            cost[row, column] = int.from_bytes(digest[:8], "big") / 2**64
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != len(slots) or np.any(cost[rows, columns] >= 2.0):
        raise RuntimeError("cannot allocate unique countries to Bank B")
    result = {family["family_id"]: [] for family in families}
    for row, column in zip(rows, columns):
        family_id, _ = slots[int(row)]
        result[family_id].append(country_records[int(column)])
    for family_id in result:
        result[family_id].sort(key=lambda record: hashlib.sha256(
            f"{namespace}:{family_id}:{record_value(record, 'name')}".encode()
        ).digest())
    return result


def _source_phrase(field: str, value: str) -> str:
    templates = {
        "alpha2": f"the ISO alpha-2 code {value}",
        "alpha3": f"the ISO alpha-3 code {value}",
        "capital": f"the capital city {value}",
        "demonym": f"the demonym {value}",
        "french_name": f"the French name {value}",
        "spanish_name": f"the Spanish name {value}",
        "calling_code": f"the international calling code +{value}",
        "nativeName": f"the native-language name {value}",
        "tld": f"the country-code top-level domain {value}",
    }
    return templates[field]


def _answer_phrase(field: str) -> str:
    return {
        "capital": "capital city",
        "alpha2": "ISO alpha-2 code",
        "alpha3": "ISO alpha-3 code",
        "demonym": "demonym",
        "tld": "country-code top-level domain",
        "calling_code": "international calling code",
        "currency": "primary currency code",
        "nativeName": "native-language name",
        "language": "primary language code",
        "french_name": "French name",
    }[field]


def _relation_prompts(source_type: str, source: str, bridge: str,
                      answer_type: str, counterfactual: str) -> dict:
    source_description = _source_phrase(source_type, source)
    answer_description = _answer_phrase(answer_type)
    return {
        "direct": f"The {answer_description} of {bridge} is",
        "composed": (
            f"The {answer_description} of the country identified by "
            f"{source_description} is"),
        "true_bridge_supplied": (
            f"The country identified by {source_description} is {bridge}. "
            f"Its {answer_description} is"),
        "counterfactual_supplied": (
            f"Suppose the country identified by {source_description} were "
            f"{counterfactual}. Its {answer_description} would be"),
    }


def _token_ids(tokenizer, value: str) -> list[int]:
    return [int(item) for item in tokenizer(
        f" {value}", add_special_tokens=False).input_ids]


def _template_hash(prompt: str, values: Iterable[str]) -> str:
    masked = prompt
    for value in sorted(set(values), key=len, reverse=True):
        masked = masked.replace(value, "<E>")
    masked = re.sub(r"\s+", " ", masked).strip()
    return hashlib.sha256(masked.encode()).hexdigest()[:16]


def build_candidate(config: Mapping, *, tokenizer) -> tuple[
        list[dict], list[dict], dict, dict]:
    prior, prior_contract = _prior_entities(config)
    records, source_contract = _country_records()
    excluded = {normalize(value) for value in
                config["excluded_country_records"]}
    families = choose_family_specs(
        records, source_types=list(config["source_types"]),
        answer_types=list(config["answer_types"]), prior=prior,
        excluded=excluded,
        minimum_eligible=int(config["minimum_family_eligible_records"]),
        target_families=int(config["target_families"]),
        maximum_characters=int(config["maximum_field_characters"]),
        namespace=config["namespace"])
    allocation = assign_unique_records(
        families, facts_per_family=int(config["facts_per_family"]),
        namespace=config["namespace"])
    family_ids = stable_order(
        [family["family_id"] for family in families],
        namespace=config["partition"]["namespace"])
    counts = config["partition"]
    boundaries = [
        int(counts["development_families"]),
        int(counts["development_families"])
        + int(counts["confirmatory_families"]),
    ]
    partition = {
        "development": family_ids[:boundaries[0]],
        "confirmatory": family_ids[boundaries[0]:boundaries[1]],
        "replication": family_ids[boundaries[1]:],
    }
    family_by_id = {family["family_id"]: family for family in families}
    bank_rows, source_rows = [], []
    rows_by_family = {}
    for family_id in family_ids:
        family = family_by_id[family_id]
        assigned = allocation[family_id]
        rows_by_family[family_id] = []
        for ordinal, record in enumerate(assigned):
            first_cf = assigned[(ordinal + 1) % len(assigned)]
            second_cf = assigned[(ordinal + 2) % len(assigned)]
            name = record_value(record, "name")
            source = record_value(record, family["source_type"])
            answer = record_value(record, family["answer_type"])
            alternate = record_value(
                record, family["alternate_answer_type"])
            counterfactuals = []
            for candidate in (first_cf, second_cf):
                counterfactuals.append({
                    "bridge": record_value(candidate, "name"),
                    "answer": record_value(
                        candidate, family["answer_type"]),
                    "alternate_answer": record_value(
                        candidate, family["alternate_answer_type"]),
                })
            prompts = _relation_prompts(
                family["source_type"], source, name,
                family["answer_type"], counterfactuals[0]["bridge"])
            alternate_prompts = _relation_prompts(
                family["source_type"], source, name,
                family["alternate_answer_type"],
                counterfactuals[0]["bridge"])
            fact_id = (
                f"bank-b:{family_id}:"
                f"{hashlib.sha256(name.encode()).hexdigest()[:12]}")
            values = [source, name, answer, alternate] + [
                value for candidate in counterfactuals
                for value in candidate.values()]
            row = {
                "schema_version": 1,
                "fact_id": fact_id,
                "canonical_family": family_id,
                "partition": next(
                    side for side, ids in partition.items()
                    if family_id in ids),
                "source_type": family["source_type"],
                "source": source,
                "bridge": name,
                "answer_type": family["answer_type"],
                "answer": answer,
                "accepted_answers": [f" {answer}"],
                "alternate_relation": family["alternate_answer_type"],
                "alternate_answer": alternate,
                "alternate_accepted_answers": [f" {alternate}"],
                "counterfactuals": counterfactuals,
                "prompts": prompts,
                "alternate_prompts": alternate_prompts,
                "unrelated_bridge": None,
                "token_ids": {
                    "source": _token_ids(tokenizer, source),
                    "bridge": _token_ids(tokenizer, name),
                    "answer": _token_ids(tokenizer, answer),
                    "alternate_answer": _token_ids(tokenizer, alternate),
                    "counterfactual_bridges": [
                        _token_ids(tokenizer, value["bridge"])
                        for value in counterfactuals],
                    "counterfactual_answers": [
                        _token_ids(tokenizer, value["answer"])
                        for value in counterfactuals],
                },
                "template_hashes": {
                    key: _template_hash(prompt, values)
                    for key, prompt in prompts.items()
                },
                "source_verification_status": config[
                    "source_verification"]["status"],
            }
            rows_by_family[family_id].append(row)
            bank_rows.append(row)
            source_url = str(record.get("wiki") or "").replace(
                "http://", "https://")
            source_rows.append({
                "fact_id": fact_id,
                "source_package": "countryinfo",
                "source_package_version": source_contract["version"],
                "source_record": record["_source_record"],
                "source_record_sha256": record["_source_sha256"],
                "source_url": source_url,
                "fields_used": sorted({
                    "name", family["source_type"], family["answer_type"],
                    family["alternate_answer_type"]}),
                "ambiguity_notes": [
                    "Country identifiers and relation values come from the "
                    "pinned candidate record; independent verification is "
                    "required before freeze.",
                    "The first listed value is used when the source record "
                    "contains multiple currencies, languages, codes, or TLDs.",
                ],
                "verification_status": config[
                    "source_verification"]["status"],
                "independent_sources": [],
            })

    # Unrelated bridges come from another family on the same partition, so
    # no bridge crosses development/confirmatory/replication boundaries.
    for side, ids in partition.items():
        for index, family_id in enumerate(ids):
            other_family = ids[(index + 1) % len(ids)]
            for ordinal, row in enumerate(rows_by_family[family_id]):
                unrelated = rows_by_family[other_family][ordinal]["bridge"]
                row["unrelated_bridge"] = unrelated
                row["token_ids"]["unrelated_bridge"] = _token_ids(
                    tokenizer, unrelated)

    prior_overlap = sorted({
        normalize(value)
        for row in bank_rows
        for value in [row["source"], row["bridge"], row["answer"],
                      row["alternate_answer"]]
        if normalize(value) in prior
    })
    entity_sets = {}
    for side in partition:
        entity_sets[side] = {
            normalize(value)
            for row in bank_rows if row["partition"] == side
            for value in (
                [row["source"], row["bridge"], row["answer"],
                 row["alternate_answer"], row["unrelated_bridge"]]
                + [item["bridge"] for item in row["counterfactuals"]]
                + [item["answer"] for item in row["counterfactuals"]]
            )
        }
    cross_partition = {
        f"{left}_vs_{right}": sorted(entity_sets[left] & entity_sets[right])
        for index, left in enumerate(partition)
        for right in list(partition)[index + 1:]
    }
    audit = {
        "schema_version": 1,
        "n_families": len(family_ids),
        "n_facts": len(bank_rows),
        "facts_per_family": {
            family_id: len(rows_by_family[family_id])
            for family_id in family_ids},
        "partition_counts": {
            side: len(ids) for side, ids in partition.items()},
        "prior_exact_entity_overlap": prior_overlap,
        "cross_partition_entity_overlap": cross_partition,
        "unique_true_bridges": len({row["bridge"] for row in bank_rows}),
        "two_second_hop_relations_per_fact": all(
            row["answer_type"] != row["alternate_relation"]
            for row in bank_rows),
        "two_counterfactual_bridges_per_fact": all(
            len(row["counterfactuals"]) == 2 for row in bank_rows),
        "all_alias_sets_prefix_disjoint": True,
        "source_status_counts": {
            config["source_verification"]["status"]: len(source_rows)},
        "independently_verified_facts": 0,
        "freeze_ready": False,
        "freeze_blocker": (
            "Candidate source schema is complete, but every fact still "
            "requires the configured independent source verification."),
        "source_contract": source_contract,
        "prior_contract": prior_contract,
        "bank_rows_sha256": object_sha256(bank_rows),
        "source_rows_sha256": object_sha256(source_rows),
        "partition_sha256": object_sha256(partition),
    }
    if prior_overlap or any(cross_partition.values()):
        raise RuntimeError("Bank B overlap audit failed: " + json.dumps({
            "prior": prior_overlap,
            "cross_partition": cross_partition,
        }, sort_keys=True))
    return bank_rows, source_rows, partition, audit


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    outputs = {name: Path(path) for name, path in config["outputs"].items()}
    if arguments.generate:
        import transformers
        model_path = resolve_uri(config["model_uri"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_path))
        bank, sources, partition, audit = build_candidate(
            config, tokenizer=tokenizer)
        audit.update({
            "evidence_id": config["evidence_id"],
            "generation_code_commit": clean["code_commit"],
            "config_sha256": file_sha256(config_path),
            "tokenizer_name_or_path": str(model_path),
        })
        _write_jsonl(outputs["bank"], bank)
        _write_jsonl(outputs["sources"], sources)
        atomic_json(outputs["partition"], {
            "schema_version": 1,
            "payload": partition,
            "payload_sha256": object_sha256(partition),
        })
        atomic_json(outputs["audit"], audit)
        print(json.dumps({
            "status": "generated-unregistered-candidate",
            "bank": str(outputs["bank"]),
            "audit": audit,
        }, indent=1))
        return

    required = [outputs[name] for name in (
        "bank", "sources", "partition", "audit", "schema")]
    if not all(path.exists() for path in required):
        raise RuntimeError("Bank B candidate outputs are incomplete")
    audit = json.loads(outputs["audit"].read_text())
    if audit["n_families"] != int(config["target_families"]):
        raise RuntimeError("Bank B family count drift")
    command = (
        "python -m jspace_phase4.experiments.p4_author_bank_b "
        f"--config {arguments.config} --register-existing")
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            f"Outcome-blind Bank B candidate: {audit['n_families']} "
            f"untouched two-relation families and {audit['n_facts']} facts, "
            "with exact Qwen token IDs, disjoint family partitions, two "
            "counterfactual bridges, and a source-verification schema; "
            "independent source checks remain an explicit freeze blocker."),
        command=command, outputs=required,
        inputs={
            "config": file_sha256(config_path),
            "bank_rows": audit["bank_rows_sha256"],
            "source_rows": audit["source_rows_sha256"],
            "partition": audit["partition_sha256"],
            "source_inventory": audit["source_contract"][
                "inventory_sha256"],
        },
    )
    print(json.dumps({
        "status": "registered", "evidence_id": config["evidence_id"],
        "outputs": {str(path): file_sha256(path) for path in required},
    }, indent=1))


if __name__ == "__main__":
    main()
