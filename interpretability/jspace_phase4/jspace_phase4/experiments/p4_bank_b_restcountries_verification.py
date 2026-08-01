"""Independently verify Bank B candidate relations against a pinned source.

The source snapshot is the REST Countries v3.1 data file at a pinned upstream
Git revision.  This methods-only producer never loads a model or reads an
intervention outcome.  Exact matches and ambiguity flags are kept separate:
an independently matching but multi-valued relation still requires manual
review before a Bank B freeze candidate can be authored.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import figures_dir, metrics_dir, resolve_uri
from ..provenance4 import Provenance4, write_result4
from ..registry4 import create, resolve
from .p4_qwen_nested_lens_fit import registered_output_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def _unicode_form(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value)).casefold()
    without_marks = "".join(
        character for character in decomposed
        if not unicodedata.category(character).startswith("M"))
    return re.sub(r"[^\w]+", " ", without_marks, flags=re.UNICODE).strip()


def _ascii_form(value: str) -> str:
    folded = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def normalized_forms(value: str) -> set[str]:
    return {
        form for form in (_unicode_form(value), _ascii_form(value)) if form
    }


def _nested(mapping: Mapping, *keys: str):
    value = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def country_name_values(country: Mapping) -> list[str]:
    name = country.get("name", {})
    values = [name.get("common"), name.get("official")]
    values.extend(country.get("altSpellings", []))
    return sorted({str(value) for value in values if value})


def relation_values(country: Mapping, relation: str) -> list[str]:
    if relation == "alpha2":
        values = [country.get("cca2")]
    elif relation == "alpha3":
        values = [country.get("cca3")]
    elif relation == "capital":
        values = list(country.get("capital") or [])
    elif relation == "demonym":
        english = _nested(country, "demonyms", "eng") or {}
        values = [english.get("m"), english.get("f")]
    elif relation == "french_name":
        french = _nested(country, "translations", "fra") or {}
        values = [french.get("common"), french.get("official")]
    elif relation == "spanish_name":
        spanish = _nested(country, "translations", "spa") or {}
        values = [spanish.get("common"), spanish.get("official")]
    elif relation == "nativeName":
        native = _nested(country, "name", "nativeName") or {}
        values = []
        for language in sorted(native):
            values.extend([
                native[language].get("common"),
                native[language].get("official"),
            ])
    elif relation == "tld":
        values = list(country.get("tld") or [])
    else:
        raise ValueError(f"unsupported Bank B relation {relation!r}")
    return sorted({str(value) for value in values if value})


def semantic_answer_values(country: Mapping, relation: str) -> list[str]:
    """Values that are genuinely co-valid answers, not formal-name aliases."""
    if relation == "french_name":
        values = [(_nested(country, "translations", "fra") or {}).get(
            "common")]
    elif relation == "spanish_name":
        values = [(_nested(country, "translations", "spa") or {}).get(
            "common")]
    elif relation == "nativeName":
        native = _nested(country, "name", "nativeName") or {}
        values = [native[language].get("common")
                  for language in sorted(native)]
    else:
        values = relation_values(country, relation)
    unique = {}
    for value in values:
        if value:
            unique.setdefault(tuple(sorted(normalized_forms(value))), value)
    return sorted(str(value) for value in unique.values())


def accepted_alias_coverage(
        accepted: Sequence[str], source_values: Sequence[str]) -> dict:
    """Require every genuinely co-valid source answer to be represented."""
    cleaned = [str(value).strip() for value in accepted if str(value).strip()]
    uncovered = [
        source for source in source_values
        if not any(
            normalized_forms(source) & normalized_forms(candidate)
            for candidate in cleaned)
    ]
    return {
        "candidate_aliases": cleaned,
        "source_semantic_values": list(source_values),
        "uncovered_source_values": uncovered,
        "passes": bool(cleaned) and not uncovered,
    }


class CountryIndex:
    def __init__(self, countries: Sequence[Mapping], *,
                 aliases: Mapping[str, Mapping] | None = None):
        self.countries = list(countries)
        self.by_name: dict[str, set[int]] = {}
        self.by_cca3 = {
            str(country.get("cca3")): index
            for index, country in enumerate(self.countries)
            if country.get("cca3")
        }
        for index, country in enumerate(self.countries):
            for name in country_name_values(country):
                for form in normalized_forms(name):
                    self.by_name.setdefault(form, set()).add(index)
        self.aliases = {}
        for name, specification in (aliases or {}).items():
            cca3 = str(specification["cca3"])
            if cca3 not in self.by_cca3:
                raise RuntimeError(
                    f"configured country alias has unknown cca3 {cca3}")
            for form in normalized_forms(name):
                self.aliases[form] = {
                    **dict(specification), "query_name": str(name)}

    def resolve(self, name: str) -> dict:
        candidates = set()
        matched_forms = []
        alias_specification = None
        for form in normalized_forms(name):
            found = self.by_name.get(form, set())
            if found:
                candidates.update(found)
                matched_forms.append(form)
            configured = self.aliases.get(form)
            if configured is not None:
                candidates.add(self.by_cca3[str(configured["cca3"])])
                matched_forms.append(form)
                alias_specification = configured
        rows = [self.countries[index] for index in sorted(candidates)]
        return {
            "query": name,
            "matched_forms": sorted(matched_forms),
            "n_candidates": len(rows),
            "candidate_cca3": [row.get("cca3") for row in rows],
            "candidate_names": [
                _nested(row, "name", "common") for row in rows],
            "resolution_method": (
                "configured_name_alias" if alias_specification
                else "independent_name" if rows else "unresolved"),
            "alias_manual_review": bool(
                alias_specification
                and alias_specification.get("manual_review")),
            "alias_reason": (
                alias_specification.get("reason")
                if alias_specification else None),
            "country": rows[0] if len(rows) == 1 else None,
        }


def value_match(value: str, country: Mapping | None,
                relation: str) -> dict:
    candidates = relation_values(country, relation) if country else []
    forms = normalized_forms(value)
    matched = [
        candidate for candidate in candidates
        if forms & normalized_forms(candidate)
    ]
    return {
        "value": value,
        "relation": relation,
        "source_values": candidates,
        "matched_values": matched,
        "passes": bool(matched),
    }


def _resolution_summary(result: Mapping) -> dict:
    return {
        key: result[key] for key in (
            "query", "matched_forms", "n_candidates",
            "candidate_cca3", "candidate_names", "resolution_method",
            "alias_manual_review", "alias_reason")
    }


def verify_fact(row: Mapping, source_row: Mapping, *, index: CountryIndex,
                config: Mapping) -> dict:
    verification = config["verification"]
    true_resolution = index.resolve(str(row["bridge"]))
    country = true_resolution["country"]
    checks = {
        "true_bridge_unique": true_resolution["n_candidates"] == 1,
    }
    source_match = value_match(
        str(row["source"]), country, str(row["source_type"]))
    answer_match = value_match(
        str(row["answer"]), country, str(row["answer_type"]))
    alternate_match = value_match(
        str(row["alternate_answer"]), country,
        str(row["alternate_relation"]))
    checks.update({
        "true_source_matches": source_match["passes"],
        "true_answer_matches": answer_match["passes"],
        "true_alternate_answer_matches": alternate_match["passes"],
    })

    counterfactuals = []
    for counterfactual in row["counterfactuals"]:
        resolution = index.resolve(str(counterfactual["bridge"]))
        cf_country = resolution["country"]
        answer = value_match(
            str(counterfactual["answer"]), cf_country,
            str(row["answer_type"]))
        alternate = value_match(
            str(counterfactual["alternate_answer"]), cf_country,
            str(row["alternate_relation"]))
        counterfactuals.append({
            "bridge": counterfactual["bridge"],
            "resolution": _resolution_summary(resolution),
            "answer_match": answer,
            "alternate_answer_match": alternate,
            "passes": bool(
                resolution["n_candidates"] == 1
                and answer["passes"] and alternate["passes"]),
            "semantic_answer_values": (
                semantic_answer_values(
                    cf_country, str(row["answer_type"]))
                if cf_country else []),
        })
    checks["every_counterfactual_matches"] = all(
        item["passes"] for item in counterfactuals)
    unrelated_resolution = index.resolve(str(row["unrelated_bridge"]))
    checks["unrelated_bridge_resolves"] = (
        unrelated_resolution["n_candidates"] == 1)

    manual_reasons = []
    reviewed_reasons = []
    ambiguity_resolutions = []
    for label, resolution in [
            ("true", true_resolution),
            ("unrelated", unrelated_resolution),
            *[(f"counterfactual_{ordinal}", item["resolution"])
              for ordinal, item in enumerate(counterfactuals)]]:
        if resolution.get("alias_manual_review"):
            manual_reasons.append(
                f"{label}_bridge_name_alias:{resolution['alias_reason']}")
    if country is not None and verification[
            "manual_review_multi_valued_answer_relations"]:
        alternatives = semantic_answer_values(
            country, str(row["answer_type"]))
        if len(alternatives) > 1:
            coverage = accepted_alias_coverage(
                row.get("accepted_answers", [row["answer"]]), alternatives)
            if verification.get(
                    "resolve_multi_valued_by_complete_alias_coverage", False) \
                    and coverage["passes"]:
                reason = "true_answer_multi_value_alias_coverage_reviewed"
                reviewed_reasons.append(reason)
                ambiguity_resolutions.append({
                    "reason": reason, "coverage": coverage})
            else:
                manual_reasons.append(
                    "true_answer_relation_has_multiple_valid_values")
    for ordinal, item in enumerate(counterfactuals):
        if len(item["semantic_answer_values"]) > 1:
            source_counterfactual = row["counterfactuals"][ordinal]
            coverage = accepted_alias_coverage(
                source_counterfactual.get(
                    "accepted_answers", [source_counterfactual["answer"]]),
                item["semantic_answer_values"])
            if verification.get(
                    "resolve_multi_valued_by_complete_alias_coverage", False) \
                    and coverage["passes"]:
                reason = (
                    f"counterfactual_{ordinal}_answer_multi_value_"
                    "alias_coverage_reviewed")
                reviewed_reasons.append(reason)
                ambiguity_resolutions.append({
                    "reason": reason, "coverage": coverage})
            else:
                manual_reasons.append(
                    f"counterfactual_{ordinal}_answer_relation_has_multiple_"
                    "valid_values")
    geopolitical = {
        form
        for name in verification["geopolitical_manual_review_names"]
        for form in normalized_forms(name)
    }
    named_bridges = [row["bridge"], row["unrelated_bridge"], *[
        item["bridge"] for item in row["counterfactuals"]]]
    geopolitical_resolutions = {
        form: {"name": name, "resolution": resolution}
        for name, resolution in verification.get(
            "geopolitical_review_resolutions", {}).items()
        for form in normalized_forms(name)
    }
    for bridge in named_bridges:
        if normalized_forms(str(bridge)) & geopolitical:
            configured = next((
                geopolitical_resolutions[form]
                for form in normalized_forms(str(bridge))
                if form in geopolitical_resolutions), None)
            if configured is None:
                manual_reasons.append(
                    f"geopolitical_name_review:{bridge}")
            else:
                reason = f"geopolitical_name_reviewed:{bridge}"
                reviewed_reasons.append(reason)
                ambiguity_resolutions.append({
                    "reason": reason,
                    "resolution": configured["resolution"],
                })
    manual_reasons = sorted(set(manual_reasons))
    reviewed_reasons = sorted(set(reviewed_reasons))

    required = {
        "true_bridge_unique": bool(
            not verification["require_unique_country_resolution"]
            or checks["true_bridge_unique"]),
        "true_source_matches": bool(
            not verification["require_true_source_match"]
            or checks["true_source_matches"]),
        "true_answer_matches": bool(
            not verification["require_true_answer_match"]
            or checks["true_answer_matches"]),
        "true_alternate_answer_matches": bool(
            not verification["require_alternate_answer_match"]
            or checks["true_alternate_answer_matches"]),
        "every_counterfactual_matches": bool(
            not verification["require_every_counterfactual_answer_match"]
            or not verification[
                "require_every_counterfactual_alternate_answer_match"]
            or checks["every_counterfactual_matches"]),
        "unrelated_bridge_resolves": bool(
            not verification["require_unrelated_bridge_resolution"]
            or checks["unrelated_bridge_resolves"]),
    }
    independent_match = all(required.values())
    if not independent_match:
        status = "independent-source-mismatch"
    elif manual_reasons:
        status = "independent-match-manual-ambiguity-review"
    elif reviewed_reasons:
        status = "verified-exact-reviewed-ambiguity"
    else:
        status = "verified-exact-unambiguous"
    return {
        "schema_version": 1,
        "fact_id": row["fact_id"],
        "canonical_family": row["canonical_family"],
        "partition": row["partition"],
        "source_type": row["source_type"],
        "answer_type": row["answer_type"],
        "alternate_relation": row["alternate_relation"],
        "verification_status": status,
        "independent_match": independent_match,
        "manual_review_required": bool(manual_reasons),
        "manual_review_reasons": manual_reasons,
        "reviewed_ambiguity": bool(reviewed_reasons),
        "reviewed_ambiguity_reasons": reviewed_reasons,
        "ambiguity_resolutions": ambiguity_resolutions,
        "checks": checks,
        "required_checks": required,
        "true_country_resolution": _resolution_summary(true_resolution),
        "true_source_match": source_match,
        "true_answer_match": answer_match,
        "true_alternate_answer_match": alternate_match,
        "counterfactuals": counterfactuals,
        "unrelated_bridge_resolution": _resolution_summary(
            unrelated_resolution),
        "existing_candidate_source": {
            "source_record": source_row["source_record"],
            "source_record_sha256": source_row[
                "source_record_sha256"],
            "verification_status": source_row["verification_status"],
        },
        "independent_source": {
            "name": config["independent_source"]["name"],
            "repository": config["independent_source"]["repository"],
            "revision": config["independent_source"]["revision"],
            "source_path": config["independent_source"]["source_path"],
            "snapshot_sha256": config[
                "independent_source"]["snapshot_sha256"],
        },
        "outcome_columns_used": False,
    }


def summarize(rows: Sequence[Mapping], *, config: Mapping) -> dict:
    statuses = [row["verification_status"] for row in rows]
    partitions = sorted({row["partition"] for row in rows})
    checks = sorted(next(iter(rows))["checks"])
    check_failures = {
        check: int(sum(not bool(row["checks"][check]) for row in rows))
        for check in checks
    }
    mismatch_signatures: dict[str, set[str]] = {}
    manual_reason_counts: dict[str, int] = {}
    reviewed_reason_counts: dict[str, int] = {}

    def add_mismatch(signature: str, fact_id: str) -> None:
        mismatch_signatures.setdefault(signature, set()).add(fact_id)

    for row in rows:
        fact_id = str(row["fact_id"])
        for reason in row["manual_review_reasons"]:
            manual_reason_counts[str(reason)] = (
                manual_reason_counts.get(str(reason), 0) + 1)
        for reason in row.get("reviewed_ambiguity_reasons", []):
            reviewed_reason_counts[str(reason)] = (
                reviewed_reason_counts.get(str(reason), 0) + 1)
        if row["true_country_resolution"]["n_candidates"] != 1:
            add_mismatch(
                f"country-resolution:{row['true_country_resolution']['query']}",
                fact_id)
        for label in (
                "true_source_match", "true_answer_match",
                "true_alternate_answer_match"):
            match = row[label]
            if not match["passes"]:
                add_mismatch(
                    f"{match['relation']}:{match['value']}!="
                    f"{match['source_values']}", fact_id)
        for counterfactual in row["counterfactuals"]:
            resolution = counterfactual["resolution"]
            if resolution["n_candidates"] != 1:
                add_mismatch(
                    f"country-resolution:{resolution['query']}", fact_id)
            for label in ("answer_match", "alternate_answer_match"):
                match = counterfactual[label]
                if not match["passes"]:
                    add_mismatch(
                        f"{match['relation']}:{match['value']}!="
                        f"{match['source_values']}", fact_id)
        unrelated = row["unrelated_bridge_resolution"]
        if unrelated["n_candidates"] != 1:
            add_mismatch(
                f"country-resolution:{unrelated['query']}", fact_id)
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "n_facts": len(rows),
        "n_families": len({row["canonical_family"] for row in rows}),
        "status_counts": {
            status: statuses.count(status) for status in sorted(set(statuses))
        },
        "partition_status_counts": {
            partition: {
                status: sum(
                    row["partition"] == partition
                    and row["verification_status"] == status
                    for row in rows)
                for status in sorted(set(statuses))
            }
            for partition in partitions
        },
        "required_check_failure_counts": check_failures,
        "distinct_mismatch_signatures": {
            signature: {
                "n_affected_facts": len(fact_ids),
                "fact_ids": sorted(fact_ids),
            }
            for signature, fact_ids in sorted(mismatch_signatures.items())
        },
        "manual_review_reason_counts": dict(sorted(
            manual_reason_counts.items())),
        "reviewed_ambiguity_reason_counts": dict(sorted(
            reviewed_reason_counts.items())),
        "independent_match_count": int(sum(
            bool(row["independent_match"]) for row in rows)),
        "manual_review_count": int(sum(
            bool(row["manual_review_required"]) for row in rows)),
        "reviewed_ambiguity_count": int(sum(
            bool(row.get("reviewed_ambiguity")) for row in rows)),
        "fully_verified_count": sum(
            status in {
                "verified-exact-unambiguous",
                "verified-exact-reviewed-ambiguity",
            }
            for status in statuses),
        "all_facts_independently_match": all(
            bool(row["independent_match"]) for row in rows),
        "all_facts_free_of_manual_ambiguity": not any(
            bool(row["manual_review_required"]) for row in rows),
        "bank_b_freeze_ready": False,
        "freeze_boundary": (
            "This independent-source audit cannot by itself freeze Bank B. "
            "Every mismatch and manual ambiguity flag must be resolved in a "
            "new candidate, followed by power/SESOI and independent review."),
        "outcome_blinding": (
            "Country metadata only; no model, intervention, confirmatory, "
            "or replication outcome was loaded."),
    }


def _plot(summary: Mapping, *, png_path: Path, pdf_path: Path) -> None:
    statuses = [
        "verified-exact-unambiguous",
        "verified-exact-reviewed-ambiguity",
        "independent-match-manual-ambiguity-review",
        "independent-source-mismatch",
    ]
    labels = ["verified", "reviewed ambiguity", "manual review", "mismatch"]
    colors = ["#009E73", "#56B4E9", "#E69F00", "#D55E00"]
    partitions = ["development", "confirmatory", "replication"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))

    axis = axes[0]
    bottom = np.zeros(len(partitions))
    for status, label, color in zip(statuses, labels, colors):
        values = np.asarray([
            summary["partition_status_counts"].get(partition, {}).get(
                status, 0)
            for partition in partitions])
        axis.bar(partitions, values, bottom=bottom, color=color, label=label)
        bottom += values
    axis.set_ylabel("facts")
    axis.set_title("A · Verification status by frozen partition", loc="left")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    failures = summary["required_check_failure_counts"]
    names = list(failures)
    values = [failures[name] for name in names]
    display = [name.replace("_", "\n", 1) for name in names]
    axis.bar(np.arange(len(names)), values, color="#D55E00")
    axis.set_xticks(np.arange(len(names)), display, rotation=25, ha="right")
    axis.set_ylabel("failed facts")
    axis.set_title("B · Exact independent-check failures", loc="left")

    figure.suptitle(
        "Bank B independent metadata verification · methods only",
        fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(
                row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_parquet(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    flattened = [{
        "fact_id": row["fact_id"],
        "canonical_family": row["canonical_family"],
        "partition": row["partition"],
        "source_type": row["source_type"],
        "answer_type": row["answer_type"],
        "verification_status": row["verification_status"],
        "independent_match": row["independent_match"],
        "manual_review_required": row["manual_review_required"],
        "manual_review_reasons_json": json.dumps(
            row["manual_review_reasons"], ensure_ascii=False),
        "required_checks_json": json.dumps(
            row["required_checks"], sort_keys=True),
    } for row in rows]
    pd.DataFrame(flattened).to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("tier") != "methods":
        raise RuntimeError("Bank B source verification is methods tier")
    existing = registered_output_check(config["evidence_id"])
    if existing is not None:
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return
    clean = require_clean_tree()
    candidate = config["bank_candidate"]
    candidate_event = resolve(candidate["evidence_id"])
    if not candidate_event["live"]:
        raise RuntimeError("Bank B candidate evidence is not live")
    paths = {
        "bank": resolve_uri(candidate["bank_uri"]),
        "source_rows": resolve_uri(candidate["source_rows_uri"]),
        "partition": resolve_uri(candidate["partition_uri"]),
        "audit": resolve_uri(candidate["audit_uri"]),
        "independent_snapshot": resolve_uri(
            config["independent_source"]["snapshot_uri"]),
    }
    expected_hashes = {
        "bank": candidate["bank_sha256"],
        "source_rows": candidate["source_rows_sha256"],
        "partition": candidate["partition_sha256"],
        "audit": candidate["audit_sha256"],
        "independent_snapshot": config[
            "independent_source"]["snapshot_sha256"],
    }
    actual_hashes = {name: file_sha256(path)
                     for name, path in paths.items()}
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            "Bank B verification input hash drift: "
            + json.dumps({name: [expected_hashes[name], actual_hashes[name]]
                          for name in expected_hashes
                          if expected_hashes[name] != actual_hashes[name]},
                         sort_keys=True))

    bank_rows = load_jsonl(paths["bank"])
    source_rows = load_jsonl(paths["source_rows"])
    source_by_fact = {row["fact_id"]: row for row in source_rows}
    if len(bank_rows) != int(candidate["expected_facts"]) \
            or len(source_by_fact) != len(bank_rows):
        raise RuntimeError("Bank B candidate/source row count drift")
    if {row["fact_id"] for row in bank_rows} != set(source_by_fact):
        raise RuntimeError("Bank B source rows do not match candidate facts")
    if len({row["canonical_family"] for row in bank_rows}) != int(
            candidate["expected_families"]):
        raise RuntimeError("Bank B family count drift")
    countries = json.loads(paths["independent_snapshot"].read_text())
    if not isinstance(countries, list) or len(countries) < 200:
        raise RuntimeError("REST Countries snapshot is incomplete")
    index = CountryIndex(
        countries,
        aliases=config["verification"][
            "country_name_resolution_aliases"])
    rows = [
        verify_fact(
            row, source_by_fact[row["fact_id"]], index=index, config=config)
        for row in bank_rows
    ]
    summary = summarize(rows, config=config)
    summary.update({
        "bank_candidate_evidence_id": candidate["evidence_id"],
        "bank_candidate_code_commit": candidate_event["code_commit"],
        "independent_source": dict(config["independent_source"]),
        "input_hashes": actual_hashes,
    })

    output_dir = metrics_dir(config["slug"]) / config["evidence_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "verification_rows.jsonl"
    parquet_path = output_dir / "verification_rows.parquet"
    manifest_path = output_dir / "input_manifest.json"
    result_path = output_dir / "verification_result.json"
    _atomic_jsonl(rows_path, rows)
    _atomic_parquet(parquet_path, rows)
    manifest_payload = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "candidate_event_code_commit": candidate_event["code_commit"],
        "inputs": actual_hashes,
        "independent_source": dict(config["independent_source"]),
        "verification": dict(config["verification"]),
        "country_record_count": len(countries),
    }
    manifest = {
        "schema_version": 1,
        "payload": manifest_payload,
        "payload_sha256": object_sha256(manifest_payload),
    }
    atomic_json(manifest_path, manifest)
    png_path = figures_dir() / f"{config['figure']['stem']}.png"
    pdf_path = figures_dir() / f"{config['figure']['stem']}.pdf"
    _plot(summary, png_path=png_path, pdf_path=pdf_path)
    command = (
        "python -m "
        "jspace_phase4.experiments.p4_bank_b_restcountries_verification "
        f"--config {arguments.config}")
    inputs = {
        "bank_candidate": actual_hashes["bank"],
        "candidate_source_rows": actual_hashes["source_rows"],
        "partition": actual_hashes["partition"],
        "candidate_audit": actual_hashes["audit"],
        "independent_snapshot": actual_hashes["independent_snapshot"],
        "input_manifest": manifest["payload_sha256"],
    }
    write_result4(
        summary, result_path,
        Provenance4(
            evidence_id=config["evidence_id"], tier=config["tier"],
            command=command, inputs=inputs,
            input_manifest_sha256=manifest["payload_sha256"],
            seed_contract=(
                "No randomization; exact normalized relation matching "
                "against a revision- and byte-pinned independent source."),
        ),
    )
    outputs = [
        result_path, manifest_path, rows_path, parquet_path,
        png_path, pdf_path,
    ]
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            "Outcome-blind independent metadata verification of all 160 "
            "Bank B candidate facts, their counterfactual countries, and "
            "unrelated bridges against pinned REST Countries v3.1 data; "
            "exact mismatch and manual ambiguity statuses remain separate."),
        command=command, outputs=outputs, inputs=inputs,
    )
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "status_counts": summary["status_counts"],
        "required_check_failure_counts": summary[
            "required_check_failure_counts"],
        "independent_match_count": summary["independent_match_count"],
        "manual_review_count": summary["manual_review_count"],
        "fully_verified_count": summary["fully_verified_count"],
        "result": str(result_path),
        "figure": str(png_path),
    }, indent=1))


if __name__ == "__main__":
    main()
