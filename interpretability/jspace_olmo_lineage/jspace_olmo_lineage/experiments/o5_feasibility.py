"""Bounded identifiability decision for the O5 crossed decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..manifests import (
    atomic_json,
    atomic_text,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths import metrics_dir
from ..registry import create, resolve, resolve_all

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("O5 feasibility config must be a mapping")
    return value


def route_feasibility(availability: dict, required: list[str]) -> dict:
    missing = [name for name in required if not availability.get(name, False)]
    if missing:
        return {
            "decision": "defer-no-identifiable-crossed-intervention-estimand",
            "status": "not-executed-no-proxy-substitution",
            "missing_required_controls": missing,
            "phase5_pilot_eligible": False,
        }
    return {
        "decision": "ready-for-prospective-o5-pilot",
        "status": "ready-not-started",
        "missing_required_controls": [],
        "phase5_pilot_eligible": True,
    }


def _load_registered_json(spec: dict) -> tuple[dict, dict]:
    record = resolve(spec["evidence_id"])
    if not record["live"]:
        raise ValueError(f"upstream is not live: {spec['evidence_id']}")
    matches = [
        row for row in record.get("outputs", [])
        if row["path"].endswith(".json")
        and row["sha256"] == spec["json_sha256"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one pinned JSON for {spec['evidence_id']}")
    output = matches[0]
    path = Path(output["path"])
    if file_sha256(path) != output["sha256"]:
        raise ValueError(f"upstream hash mismatch: {path}")
    raw = json.loads(path.read_text())
    payload = raw.get("payload", raw)
    return ({
        "evidence_id": spec["evidence_id"],
        "path": str(path),
        "sha256": output["sha256"],
        "bytes": output["bytes"],
        "source_commit": record["code_commit"],
    }, payload)


def _render_markdown(result: dict) -> str:
    lines = [
        "# O5 crossed-decomposition feasibility decision",
        "",
        f"Evidence: `{result['evidence_id']}`",
        "",
        f"Decision: `{result['route']['decision']}`",
        "",
        "No O5 intervention was executed and no proxy was substituted.",
        "",
        "## Identifiability audit",
        "",
        "| Requirement | Available | Basis |",
        "|---|---:|---|",
    ]
    for name, row in result["availability"].items():
        lines.append(
            f"| `{name}` | {str(row['available']).lower()} | "
            f"{row['basis']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "O2 estimates sparse capacity and common/own transport-frame "
        "sensitivity; O3 estimates structural operator, token-row, selection, "
        "and readout geometry. Neither is a crossed downstream intervention. "
        "The registered evidence therefore cannot separate activation-state, "
        "transport-map, readout, and downstream-consumer effects.",
        "",
        "The version-1 Bank-W intervention population remains blocked at "
        f"{result['observed']['bank_w_common_support']} < "
        f"{result['observed']['bank_w_required_support']}. The official SFT/DPO "
        "stage wedge is a separate queued H5 task, not an O5 substitute.",
        "",
        "## Phase 5 entry",
        "",
        result["prospective_minimal_design"]["phase5_entry"],
        "",
        "## Claim boundary",
        "",
        result["claim_boundary"],
        "",
    ])
    return "\n".join(lines)


def run(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    source = require_clean_tree(expected_branch=config["branch"])
    output_dir = metrics_dir("o5-feasibility")
    json_path = output_dir / f"{config['evidence_id']}.json"
    markdown_path = output_dir / f"{config['evidence_id']}.md"
    for path in (json_path, markdown_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    inputs = {}
    values = {}
    for name, spec in config["upstream"].items():
        inputs[name], values[name] = _load_registered_json(spec)

    existing_crossed = [
        row["evidence_id"] for row in resolve_all()
        if row["live"] and row["evidence_id"].startswith("ol-o5-crossed-")
    ]
    if existing_crossed:
        raise ValueError(
            "feasibility-only decision is stale; crossed evidence exists: "
            + ", ".join(existing_crossed))

    bank = values["bank_w_gate"]
    capacity = values["capacity"]
    geometry = values["geometry"]
    inventory = values["checkpoint_inventory"]
    if bank["interventions_opened"]:
        raise ValueError("unexpected Bank-W intervention outcome is open")
    if inventory["route"]["model_outcome_opened"]:
        raise ValueError("checkpoint inventory unexpectedly opened a model outcome")

    availability_rows = {
        "registered_capacity_table": {
            "available": capacity["status"] == "complete",
            "basis": capacity["lineage_verdict"],
        },
        "registered_structural_geometry": {
            "available": geometry["router"]["verdict"] is not None,
            "basis": geometry["router"]["verdict"],
        },
        "crossed_activation_model_cells": {
            "available": False,
            "basis": "no registered O5 downstream-intervention rows",
        },
        "crossed_transport_lens_cells": {
            "available": False,
            "basis": "own/common capacity sensitivity is not a causal cross",
        },
        "crossed_readout_cells": {
            "available": False,
            "basis": "O3 readout geometry has no downstream intervention",
        },
        "matched_rank_energy_random_per_dictionary": {
            "available": False,
            "basis": "no crossed-dictionary intervention rows",
        },
        "protected_span_recomputed_per_readout": {
            "available": geometry["selection_margin_audit"][
                "protected_span_overlap"] is not None,
            "basis": geometry["selection_margin_audit"]["boundary"][
                "protected_span_overlap"]["status"],
        },
        "logit_lens_non_j_baseline": {
            "available": False,
            "basis": "not present in registered O2/O3 tables",
        },
        "equal_item_condition_order": {
            "available": False,
            "basis": "cannot be established before crossed rows exist",
        },
    }
    availability = {
        name: row["available"] for name, row in availability_rows.items()
    }
    route = route_feasibility(
        availability, config["required_controls"])
    result = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": config["evidence_id"],
        "tier": config["tier"],
        "code_commit": source["code_commit"],
        "config_sha256": file_sha256(config_path),
        "scientific_import_boundary": config[
            "scientific_import_boundary"],
        "inputs": inputs,
        "observed": {
            "bank_w_service_ready": bank["olmo_phase4_service_ready"],
            "bank_w_common_support": bank[
                "n_joint_common_capable_families"],
            "bank_w_required_support": bank[
                "minimum_joint_common_families"],
            "capacity_verdict": capacity["lineage_verdict"],
            "geometry_verdict": geometry["router"]["verdict"],
            "geometry_protected_span_overlap": geometry[
                "selection_margin_audit"]["protected_span_overlap"],
            "h5_status": inventory["route"]["h5_status"],
            "h5_queue_status": inventory["route"]["minimal_wedge"][
                "status"],
        },
        "availability": availability_rows,
        "required_controls": config["required_controls"],
        "route": route,
        "prospective_minimal_design": config[
            "prospective_minimal_design"],
        "claim_boundary": config["claim_boundary"],
    }
    result["payload_sha256"] = object_sha256(result)
    atomic_json(json_path, result)
    atomic_text(markdown_path, _render_markdown(result))
    command = (
        "python -m jspace_olmo_lineage.experiments.o5_feasibility "
        f"--config {config_path}")
    event = create(
        config["evidence_id"],
        tier=config["tier"],
        what=("Bounded O5 crossed-decomposition identifiability decision; "
              "no model or intervention outcome."),
        command=command,
        outputs=[json_path, markdown_path],
        inputs={name: row["sha256"] for name, row in inputs.items()},
        verdict=route["decision"],
        claim_boundary=config["claim_boundary"],
    )
    return {"status": "registered", "event": event, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PACKAGE_ROOT / "configs/ol_o5_feasibility_v1.yaml"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
