"""Recompute the Bank-W common-support gate from imported OLMo evidence.

This producer never loads a model and never reads an intervention outcome.  It
independently rehashes the side-track import, replays the frozen Phase 4
capability analysis from the registered 384-row tables, and applies the
prospective all-eligible-model intersection rule.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from ..import_bundle import validate_import_bundle
from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import resolve_uri
from ..registry4 import create, resolve
from .p4_bank_w_capability import (
    aggregate_model_payloads,
    analyze_model_rows,
)


class JointImportError(RuntimeError):
    """Raised when an imported capability artifact drifts."""


def _require_hash(path: str | Path, expected: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise JointImportError(f"required artifact is absent: {resolved}")
    actual = file_sha256(resolved)
    if actual != expected:
        raise JointImportError(
            f"artifact hash drift: {resolved}; expected {expected}, got "
            f"{actual}")
    return resolved


def _load_envelope(path: str | Path, expected: str) -> tuple[dict, dict]:
    resolved = _require_hash(path, expected)
    envelope = json.loads(resolved.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise JointImportError(f"artifact lacks an envelope payload: {path}")
    if envelope.get("payload_sha256") != object_sha256(payload):
        raise JointImportError(f"artifact payload hash drift: {path}")
    return envelope, payload


def _event_output_inventory(event: Mapping) -> dict[str, str]:
    field = (
        "source_outputs"
        if event.get("event") == "evidence_imported" else "outputs")
    rows = list(event.get(field, []))
    inventory: dict[str, str] = {}
    for row in rows:
        path = str(row.get("path", ""))
        digest = str(row.get("sha256", ""))
        if not path or path in inventory:
            raise JointImportError("registry output inventory is ambiguous")
        inventory[path] = digest
    return inventory


def _require_registered_output(
        inventory: Mapping[str, str], path: str | Path,
        expected: str) -> None:
    text = str(path)
    matches = [
        digest for recorded, digest in inventory.items()
        if recorded == text or Path(recorded).resolve() == Path(text).resolve()
    ]
    if len(matches) != 1 or matches[0] != expected:
        raise JointImportError(
            f"registry does not pin expected output {text} at {expected}")


def validate_import_contract(config: Mapping, *,
                             require_registered: bool) -> dict:
    """Freshly validate the bundle and optionally its mainline import event."""
    specification = config["import_contract"]
    bundle = resolve_uri(specification["bundle_uri"])
    validation = resolve_uri(specification["validation_uri"])
    _require_hash(bundle, specification["bundle_sha256"])
    _require_hash(validation, specification["validation_sha256"])
    recorded = json.loads(validation.read_text())
    recorded_bundle = Path(recorded["bundle_path"])
    if recorded_bundle.resolve() != bundle.resolve():
        raise JointImportError("recorded validation names a foreign bundle")
    current = validate_import_bundle(
        recorded["bundle_path"], allow_existing_target=True)
    if current != recorded:
        raise JointImportError(
            "saved OLMo import validation differs from a fresh validation")
    expected_ids = list(specification["required_source_evidence_ids"])
    observed_ids = [
        row["evidence_id"] for row in current["selected_events"]]
    if observed_ids != expected_ids:
        raise JointImportError("OLMo import selected-event order drift")
    if current["source_commit"] != specification["source_commit"]:
        raise JointImportError("OLMo import source commit drift")
    if current["source_registry"]["sha256"] != specification[
            "source_registry_sha256"]:
        raise JointImportError("OLMo source registry hash drift")
    audit = {
        "bundle_sha256": current["bundle_sha256"],
        "validation_sha256": file_sha256(validation),
        "source_commit": current["source_commit"],
        "source_registry_sha256": current["source_registry"]["sha256"],
        "selected_event_ids": observed_ids,
        "selected_event_ids_sha256": current[
            "selected_event_ids_sha256"],
        "output_inventory_sha256": current["output_inventory_sha256"],
        "n_source_events": len(observed_ids),
        "n_source_outputs": len(current["outputs"]),
        "fresh_validation_equal": True,
        "registered_mainline_import": False,
    }
    inventory = {
        row["path"]: row["sha256"] for row in current["outputs"]}
    if require_registered:
        event = resolve(specification["evidence_id"])
        if not event["live"] or event["event"] != "evidence_imported":
            raise JointImportError("OLMo mainline import event is not live")
        expected_metadata = {
            "source_study": specification["source_study"],
            "source_commit": specification["source_commit"],
            "source_registry_sha256": specification[
                "source_registry_sha256"],
        }
        if any(event.get(key) != value
               for key, value in expected_metadata.items()):
            raise JointImportError("OLMo mainline import metadata drift")
        if list(event.get("source_evidence_ids", [])) != expected_ids:
            raise JointImportError("mainline import source-event list drift")
        registered = _event_output_inventory(event)
        _require_registered_output(
            registered, str(bundle), specification["bundle_sha256"])
        _require_registered_output(
            registered, str(validation), specification["validation_sha256"])
        for path, digest in inventory.items():
            _require_registered_output(registered, path, digest)
        audit["registered_mainline_import"] = True
        audit["import_code_commit"] = event.get("import_code_commit")
    return {"audit": audit, "output_inventory": inventory}


def verify_bank_v3_compatibility(config: Mapping) -> dict:
    """Prove that the consumed v2 development rows are unchanged in v3."""
    specification = config["bank_contract"]
    event = resolve(specification["current_bank_evidence_id"])
    if not event["live"]:
        raise JointImportError("current Bank-W v3 evidence is not live")
    bank = resolve_uri(specification["current_bank_uri"])
    audit_path = resolve_uri(specification["current_audit_uri"])
    _require_hash(bank, specification["current_bank_sha256"])
    _require_hash(audit_path, specification["current_audit_sha256"])
    inventory = _event_output_inventory(event)
    _require_registered_output(
        inventory, str(bank), specification["current_bank_sha256"])
    _require_registered_output(
        inventory, str(audit_path), specification["current_audit_sha256"])
    rows = [
        json.loads(line) for line in bank.read_text().splitlines()
        if line.strip()
    ]
    development = [
        row for row in rows if row.get("partition") == "development"]
    development_sha = object_sha256(development)
    if development_sha != specification["development_payload_sha256"]:
        raise JointImportError("Bank-W v3 development payload drift")
    audit = json.loads(audit_path.read_text())
    reference = audit.get("development_rows_reference", {})
    checks = {
        "byte_identical_payload": reference.get(
            "byte_identical_payload") is True,
        "source_bank_file": reference.get("file_sha256") == specification[
            "source_bank_file_sha256"],
        "development_payload": reference.get(
            "development_rows_sha256") == development_sha,
        "audit_partition_payload": audit.get(
            "partition_row_payload_sha256", {}).get(
                "development") == development_sha,
        "n_development_rows": len(development) == 1536,
    }
    if not all(checks.values()):
        raise JointImportError("Bank-W v2-to-v3 development binding failed")
    return {
        "current_bank_evidence_id": event["evidence_id"],
        "current_bank_sha256": file_sha256(bank),
        "current_audit_sha256": file_sha256(audit_path),
        "development_payload_sha256": development_sha,
        "n_development_rows": len(development),
        "checks": checks,
    }


def _finite_row_audit(rows: Sequence[Mapping], *,
                      aliases: Sequence[str]) -> dict:
    alias_set = set(aliases)
    numeric_fields = (
        "baseline_answer_margin", "true_answer_sequence_lp",
        "prompt_token_count", "answer_token_count",
    )
    checked = 0
    for row in rows:
        if not all(math.isfinite(float(row[field]))
                   for field in numeric_fields):
            raise JointImportError(
                f"non-finite capability endpoint: {row.get('item_id')}")
        scores = json.loads(str(row["candidate_scores_json"]))
        if set(scores) != alias_set or len(scores) != len(alias_set):
            raise JointImportError("incomplete candidate score vector")
        if not all(math.isfinite(float(value)) for value in scores.values()):
            raise JointImportError("non-finite candidate sequence score")
        checked += len(numeric_fields) + len(scores)
    return {
        "all_rows_finite": True,
        "candidate_sequences_per_row": len(alias_set),
        "n_numeric_values_checked": checked,
        "n_rows": len(rows),
        "no_rows_dropped": True,
    }


def _base_analysis(value: Mapping, *, imported: bool,
                   expected_source_hash: str) -> tuple[dict, dict]:
    analysis = dict(value)
    side = {}
    if imported:
        side["side_track_finite_gate"] = analysis.pop(
            "side_track_finite_gate", None)
        side["phase4_function_source_sha256"] = analysis.pop(
            "phase4_function_source_sha256", None)
        if side["phase4_function_source_sha256"] != expected_source_hash:
            raise JointImportError("side-track Phase 4 function hash drift")
        if not isinstance(side["side_track_finite_gate"], dict):
            raise JointImportError("side-track finite audit is absent")
    return analysis, side


def verify_model_artifacts(
        specification: Mapping, *, inventory: Mapping[str, str],
        selection: Mapping, guard: Mapping, aliases: Sequence[str],
        bank_contract: Mapping) -> dict:
    """Verify and independently replay one registered capability table."""
    paths = {
        "input_manifest": specification["input_manifest_path"],
        "rows": specification["rows_path"],
        "result": specification["result_path"],
    }
    hashes = {
        "input_manifest": specification["input_manifest_sha256"],
        "rows": specification["rows_sha256"],
        "result": specification["result_sha256"],
    }
    for name, path in paths.items():
        _require_registered_output(inventory, path, hashes[name])
        _require_hash(path, hashes[name])
    manifest_envelope, manifest = _load_envelope(
        paths["input_manifest"], hashes["input_manifest"])
    _, result = _load_envelope(paths["result"], hashes["result"])
    result_envelope = json.loads(Path(paths["result"]).read_text())
    provenance = result_envelope.get("provenance", {})
    expected_manifest = {
        "experiment_id": specification["evidence_id"],
        "model_id": specification["model_id"],
        "model_revision": specification["model_revision"],
        "tokenizer_manifest_sha256": specification[
            "tokenizer_manifest_sha256"],
        "bank_sha256": bank_contract["development_rows_sha256"],
        "partition_sha256": bank_contract[
            "source_partition_payload_sha256"],
        "scoring_spec_sha256": bank_contract["scoring_spec_sha256"],
    }
    if any(manifest.get(key) != value
           for key, value in expected_manifest.items()):
        raise JointImportError(
            f"input-manifest contract drift for {specification['slug']}")
    if provenance.get("dirty_tree") is not False:
        raise JointImportError("capability result came from a dirty tree")
    if provenance.get("evidence_id") != specification["evidence_id"]:
        raise JointImportError("capability provenance evidence-ID drift")
    if provenance.get("input_manifest_sha256") != manifest_envelope.get(
            "payload_sha256"):
        raise JointImportError("capability input-manifest binding drift")
    expected_model = {
        "model_id": specification["model_id"],
        "revision": specification["model_revision"],
    }
    if result.get("model") != expected_model:
        raise JointImportError("capability result model revision drift")
    if result.get("model_slug") != specification["slug"]:
        raise JointImportError("capability result model slug drift")
    if result.get("interventions_opened", False) is not False:
        raise JointImportError("capability result opened an intervention")
    if result.get("tokenizer_manifest_sha256") != specification[
            "tokenizer_manifest_sha256"]:
        raise JointImportError("capability tokenizer manifest drift")
    protocol_hash = (
        result.get("source_phase4_protocol_sha256")
        if specification["source"] == "imported"
        else result.get("protocol_sha256"))
    if protocol_hash != bank_contract["source_protocol_sha256"]:
        raise JointImportError("capability source protocol hash drift")
    answer_tokens = result.get("answer_token_ids", {})
    sequences = [tuple(answer_tokens.get(alias, [])) for alias in aliases]
    if (any(not row or len(row) > 2 for row in sequences)
            or len(sequences) != len(set(sequences))):
        raise JointImportError("capability answer-token contract drift")

    frame = pd.read_parquet(paths["rows"])
    forbidden_fragments = (
        "intervention", "treatment", "post_intervention", "dose_nats",
    )
    if any(any(fragment in column.lower()
               for fragment in forbidden_fragments)
           for column in frame.columns):
        raise JointImportError("capability table contains intervention columns")
    rows = frame.to_dict(orient="records")
    finite = _finite_row_audit(rows, aliases=aliases)
    recomputed = analyze_model_rows(
        rows, selection=selection, guard=guard)
    stored, side = _base_analysis(
        result["analysis"],
        imported=specification["source"] == "imported",
        expected_source_hash=bank_contract[
            "phase4_function_source_sha256"],
    )
    if object_sha256(stored) != object_sha256(recomputed):
        raise JointImportError(
            f"independent capability replay drift for "
            f"{specification['slug']}")
    if side and side["side_track_finite_gate"] != finite:
        raise JointImportError("side and mainline finite audits disagree")
    return {
        "model_slug": specification["slug"],
        "source": specification["source"],
        "source_evidence_id": specification["evidence_id"],
        "model": expected_model,
        "input_manifest_sha256": hashes["input_manifest"],
        "rows_sha256": hashes["rows"],
        "result_sha256": hashes["result"],
        "analysis_sha256": object_sha256(recomputed),
        "analysis": recomputed,
        "finite_audit": finite,
        "source_side_audit": side,
        "intervention_columns_absent": True,
    }


def build_joint_result(config: Mapping, *,
                       require_registered_import: bool) -> dict:
    imported = validate_import_contract(
        config, require_registered=require_registered_import)
    bank = verify_bank_v3_compatibility(config)
    import_inventory = imported["output_inventory"]
    records = {}
    payloads = {}
    for specification in config["models"]:
        if specification["source"] == "imported":
            inventory = import_inventory
        else:
            event = resolve(specification["evidence_id"])
            if not event["live"]:
                raise JointImportError(
                    f"native capability evidence is not live: "
                    f"{specification['evidence_id']}")
            inventory = _event_output_inventory(event)
        record = verify_model_artifacts(
            specification, inventory=inventory,
            selection=config["selection"],
            guard=config["capability_guard"],
            aliases=config["answer_aliases"],
            bank_contract=config["bank_contract"],
        )
        records[specification["slug"]] = record
        payloads[specification["slug"]] = {
            "analysis": record["analysis"]}
    aggregation_config = {
        "models": [{"slug": row["slug"]} for row in config["models"]],
        "capability_guard": config["capability_guard"],
        "claim_boundary": config["claim_boundary"],
    }
    joint = aggregate_model_payloads(
        payloads, config=aggregation_config)
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "import_audit": imported["audit"],
        "bank_v3_development_compatibility": bank,
        "model_verifications": records,
        "joint": joint,
        "p4p3_disposition": (
            "READY_FOR_FROZEN_INTERVENTION_PRODUCER_REVIEW"
            if joint["p4p3_baseline_capability_ready"] else
            "BLOCKED_PROSPECTIVE_BANK_OR_SUPPORT_REVISION_REQUIRED"),
        "interventions_opened": False,
        "confirmatory_outcomes_opened": False,
        "replication_outcomes_opened": False,
        "independent_review_complete": False,
        "pi_signoff_complete": False,
        "freeze_ready": False,
        "claim_boundary": config["claim_boundary"],
    }


def _plot(result: Mapping, *, png: Path, pdf: Path) -> None:
    records = result["model_verifications"]
    order = list(records)
    labels = [
        value.replace("olmo31-", "OLMo ").replace("qwen36-27b", "Qwen")
        for value in order
    ]
    low = [records[slug]["analysis"]["load_summaries"]["low"][
        "accuracy"] for slug in order]
    high = [records[slug]["analysis"]["load_summaries"]["high"][
        "accuracy"] for slug in order]
    capable = [records[slug]["analysis"]["n_capable_families"]
               for slug in order]
    x = list(range(len(order)))
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    axes[0].bar([value - 0.18 for value in x], low, width=0.36,
                color="#56B4E9", label="low load")
    axes[0].bar([value + 0.18 for value in x], high, width=0.36,
                color="#009E73", label="high load")
    axes[0].axhline(0.70, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("candidate-set accuracy")
    axes[0].set_title("A · Independent capability", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x, capable, color="#009E73")
    minimum = result["joint"]["minimum_joint_common_families"]
    common = result["joint"]["n_joint_common_capable_families"]
    axes[1].axhline(minimum, color="#555555", linestyle="--", linewidth=1,
                    label=f"required common support = {minimum}")
    axes[1].axhline(common, color="#D55E00", linestyle=":", linewidth=1.5,
                    label=f"observed intersection = {common}")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 25)
    axes[1].set_ylabel("families capable at both loads")
    axes[1].set_title("B · Frozen common-support rule", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    status = (
        "PASS" if result["joint"]["p4p3_baseline_capability_ready"]
        else "BLOCKED")
    figure.suptitle(
        "Bank W baseline capability · mainline replay\n"
        f"{status}: three-model intersection {common}/{minimum}",
        fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def generate(config_path: Path, config: Mapping) -> None:
    require_clean_tree()
    result = build_joint_result(config, require_registered_import=True)
    report = resolve_uri(config["outputs"]["report"], must_exist=False)
    png = resolve_uri(config["outputs"]["figure_png"], must_exist=False)
    pdf = resolve_uri(config["outputs"]["figure_pdf"], must_exist=False)
    if any(path.exists() for path in (report, png, pdf)):
        raise FileExistsError("unregistered joint-support output exists")
    atomic_json(report, result)
    _plot(result, png=png, pdf=pdf)
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "p4p3_disposition": result["p4p3_disposition"],
        "n_joint_common_capable_families": result["joint"][
            "n_joint_common_capable_families"],
        "minimum_joint_common_families": result["joint"][
            "minimum_joint_common_families"],
    }, indent=1))


def register(config_path: Path, config: Mapping) -> None:
    require_clean_tree()
    report = resolve_uri(config["outputs"]["report"])
    png = resolve_uri(config["outputs"]["figure_png"])
    pdf = resolve_uri(config["outputs"]["figure_pdf"])
    recorded = json.loads(report.read_text())
    current = build_joint_result(config, require_registered_import=True)
    if recorded != current:
        raise JointImportError("joint-support report differs from fresh replay")
    joint = current["joint"]
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            "Imported and independently replayed OLMo/Qwen Bank-W baseline "
            "capability and frozen common-support gate; no intervention "
            "outcome."),
        command=(
            "python -m jspace_phase4.experiments."
            f"p4_bank_w_joint_imported --config {config_path} --register"),
        outputs=[report, png, pdf],
        inputs={
            "config": file_sha256(config_path),
            config["import_contract"]["evidence_id"]: config[
                "import_contract"]["validation_sha256"],
            **{
                row["evidence_id"]: row["result_sha256"]
                for row in config["models"]
            },
            config["bank_contract"]["current_bank_evidence_id"]: config[
                "bank_contract"]["current_audit_sha256"],
        },
        baseline_capability_ready=bool(
            joint["p4p3_baseline_capability_ready"]),
        n_joint_common_capable_families=int(
            joint["n_joint_common_capable_families"]),
        minimum_joint_common_families=int(
            joint["minimum_joint_common_families"]),
        interventions_opened=False,
        freeze_ready=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--generate", action="store_true")
    action.add_argument("--register", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    if arguments.preflight:
        result = build_joint_result(
            config, require_registered_import=False)
        print(json.dumps({
            "preflight": "PASS",
            "p4p3_disposition": result["p4p3_disposition"],
            "independently_eligible_models": result["joint"][
                "independently_eligible_models"],
            "n_joint_common_capable_families": result["joint"][
                "n_joint_common_capable_families"],
            "minimum_joint_common_families": result["joint"][
                "minimum_joint_common_families"],
        }, indent=1))
    elif arguments.generate:
        generate(config_path, config)
    else:
        register(config_path, config)


if __name__ == "__main__":
    main()
