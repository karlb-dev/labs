"""Prospective protocol and pure analyzer for the P4-P2 variance pilot.

Protocol authoring reads only registered methods and an outcome-blind subset.
The pure analyzer is prepared for a later development-only intervention grid,
but this module does not load a model or run an intervention.  Execution is
forbidden until the v2 model baseline passes and the canonical lens is bound.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import resolve_uri
from ..registry4 import create, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--author-protocol", action="store_true")
    group.add_argument("--register-protocol", action="store_true")
    return parser.parse_args()


def _registered_result(specification: Mapping, *, label: str) -> dict:
    path = Path(specification["result"])
    expected = str(specification["result_sha256"])
    if file_sha256(path) != expected:
        raise RuntimeError(f"registered {label} result hash drift")
    event = resolve(str(specification["evidence_id"]))
    if not event["live"]:
        raise RuntimeError(f"{label} evidence is not live")
    registered = {
        row["sha256"] for row in event["outputs"]
        if Path(row["path"]) == path
    }
    if registered != {expected}:
        raise RuntimeError(f"{label} result is absent from its event")
    return json.loads(path.read_text())


def _selection(config: Mapping) -> tuple[dict, list[str], list[str]]:
    specification = config["selection"]
    event = resolve(str(specification["evidence_id"]))
    if not event["live"]:
        raise RuntimeError("mode pilot selection evidence is not live")
    path = resolve_uri(str(specification["manifest_uri"]))
    expected = str(specification["manifest_sha256"])
    if file_sha256(path) != expected:
        raise RuntimeError("mode pilot subset manifest hash drift")
    registered = {
        row["sha256"] for row in event["outputs"]
        if Path(row["path"]) == path
    }
    if registered != {expected}:
        raise RuntimeError("mode pilot subset manifest is absent from event")
    envelope = json.loads(path.read_text())
    if envelope.get("payload_sha256") != specification["payload_sha256"]:
        raise RuntimeError("mode pilot subset payload hash drift")
    payload = envelope["payload"]
    if payload.get("selection_is_outcome_blind") is not True:
        raise RuntimeError("mode pilot subset is not outcome-blind")
    if payload.get("selection_uses_only_consumed_phase3_families") is not True:
        raise RuntimeError("mode pilot subset is not consumed development")
    subset = payload["subset"][specification["subset_key"]]
    facts = [str(value) for value in subset["fact_ids"]]
    families = [value.split(":", 1)[0] for value in facts]
    if len(facts) != int(specification["expected_facts"]):
        raise RuntimeError("mode pilot fact count drift")
    if len(set(families)) != int(specification["expected_families"]):
        raise RuntimeError("mode pilot family count drift")
    if len(families) != len(set(families)):
        raise RuntimeError("mode pilot requires one fact per family")
    return payload, facts, families


def author_protocol(config: Mapping) -> dict:
    methods = _registered_result(config["source_methods"], label="mode methods")
    feasibility = _registered_result(
        config["source_feasibility"], label="mode feasibility")
    baseline = config["source_model_baseline_contract"]
    baseline_path = Path(baseline["config"])
    if file_sha256(baseline_path) != baseline["config_sha256"]:
        raise RuntimeError("mode v2 baseline config hash drift")
    baseline_config = yaml.safe_load(baseline_path.read_text())
    selection, facts, families = _selection(config)
    pilot = config["pilot"]
    expected_cells = (
        len(pilot["modes"])
        * len(pilot["primary_phases"])
        * len(pilot["arms"])
    )
    if expected_cells != 8 or len(pilot["cell_order"]) != 8:
        raise RuntimeError("mode variance pilot must contain eight cells")
    if list(pilot["interaction_coefficients"]) != [
            1, -1, -1, 1, -1, 1, 1, -1]:
        raise RuntimeError("mode pilot interaction coefficients drift")
    gates = {
        "mode_parser_methods_pass": methods.get(
            "all_protocol_gates_pass") is True,
        "feasibility_is_outcome_blind": str(feasibility.get(
            "outcome_blinding", "")).startswith(
                "Uses only registered parser/template methods"),
        "feasibility_does_not_authorize_freeze": feasibility.get(
            "freeze_ready") is False,
        "v2_baseline_contract_matches_methods": baseline_config[
            "methods_gate"]["evidence_id"] == config[
                "source_methods"]["evidence_id"],
        "v2_baseline_uses_same_selection": baseline_config[
            "selection"]["source_evidence_id"] == config[
                "selection"]["evidence_id"],
        "selection_is_consumed_development": selection[
            "selection_uses_only_consumed_phase3_families"] is True,
        "one_fact_per_family": len(facts) == len(set(families)),
        "exact_eight_cell_primary": expected_cells == 8,
        "no_sesoi_or_power_selected": config[
            "variance_summary"]["no_sesoi_or_power_selection"] is True,
        "untouched_families_forbidden": config[
            "conditional_execution"][
                "forbid_confirmatory_and_replication_families"] is True,
    }
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "outcome_blinding": (
            "Protocol authoring reads registered parser/template methods, "
            "the mathematical feasibility envelope, and an outcome-blind "
            "consumed Phase 3 family subset. It reads no v1/v2 model row and "
            "runs no intervention."),
        "source_methods_evidence_id": config[
            "source_methods"]["evidence_id"],
        "source_feasibility_evidence_id": config[
            "source_feasibility"]["evidence_id"],
        "required_passing_baseline_evidence_id": baseline[
            "required_passing_evidence_id"],
        "selection": {
            "evidence_id": config["selection"]["evidence_id"],
            "payload_sha256": config["selection"]["payload_sha256"],
            "n_facts": len(facts),
            "n_families": len(families),
            "fact_ids_sha256": object_sha256(facts),
            "family_ids_sha256": object_sha256(families),
            "consumed_phase3_development_only": True,
        },
        "conditional_execution": dict(config["conditional_execution"]),
        "pilot": dict(pilot),
        "mechanical_gates": dict(config["mechanical_gates"]),
        "variance_summary": dict(config["variance_summary"]),
        "protocol_gate_checks": gates,
        "all_protocol_gates_pass": bool(all(gates.values())),
        "execution_authorized_at_this_boundary": False,
        "execution_blockers": [
            "p4-qwen-mode-gate-dev-v2 must be live and passing",
            "the frozen A250-A500 rule must bind a registered canonical lens",
            "the GPU intervention producer and exact lens binding require review",
        ],
        "analysis_boundary": (
            "The later pilot estimates family-interaction SD only. Its rows "
            "remain development data and cannot enter a primary; it cannot "
            "select the substantive SESOI by observed effect size."),
        "freeze_ready": False,
    }


def analyze_pilot_rows(rows: Sequence[Mapping], protocol: Mapping) -> dict:
    """Analyze a complete future pilot grid without selecting a SESOI."""
    pilot = protocol["pilot"]
    mechanics = protocol["mechanical_gates"]
    expected_families = int(protocol["selection"]["n_families"])
    modes = [str(value) for value in pilot["modes"]]
    phases = [str(value) for value in pilot["primary_phases"]]
    arms = [str(value) for value in pilot["arms"]]
    by_key: dict[tuple[str, str, str, str], Mapping] = {}
    for row in rows:
        key = (
            str(row["canonical_family"]), str(row["mode"]),
            str(row["phase"]), str(row["arm"]),
        )
        if key in by_key:
            raise RuntimeError(f"duplicate mode pilot row: {key}")
        by_key[key] = row
    families = sorted({key[0] for key in by_key})
    complete = (
        len(families) == expected_families
        and all(
            (family, mode, phase, arm) in by_key
            for family in families for mode in modes
            for phase in phases for arm in arms)
    )
    if not complete:
        raise RuntimeError("mode variance pilot lacks its complete family grid")

    wrong_phase = sum(int(row["wrong_phase_hook_fires"]) for row in rows)
    expected_hook_min = min(int(row["expected_phase_hook_fires"]) for row in rows)
    overlap_max = max(int(row["selected_protected_overlap"]) for row in rows)
    rank_exact = all(
        int(row["delivered_rank"]) == int(row["requested_rank"])
        for row in rows)
    energy_max = max(float(row["energy_relative_error"]) for row in rows)
    mechanical_checks = {
        "complete_eight_cell_family_grid": complete,
        "wrong_phase_hook_fires_within_tolerance": wrong_phase <= int(
            mechanics["maximum_wrong_phase_hook_fires"]),
        "expected_phase_hook_fires_present": expected_hook_min >= int(
            mechanics["minimum_expected_phase_hook_fires_per_row"]),
        "selected_protected_overlap_zero": (
            not mechanics["require_zero_selected_protected_overlap"]
            or overlap_max == 0),
        "rank_match_exact": (
            not mechanics["require_exact_rank_match"] or rank_exact),
        "energy_error_within_tolerance": energy_max <= float(
            mechanics["maximum_energy_relative_error"]),
    }

    interactions = []
    parse_failures = 0
    for family in families:
        cells = []
        for cell in pilot["cell_order"]:
            mode, phase, arm = _parse_cell_name(str(cell), modes, phases, arms)
            row = by_key[(family, mode, phase, arm)]
            parse_valid = bool(row["parse_valid"])
            correct = bool(row["correct"])
            if not parse_valid:
                parse_failures += 1
                if correct:
                    raise RuntimeError("parse-failed pilot row cannot be correct")
            cells.append(float(correct))
        interactions.append(float(sum(
            coefficient * value
            for coefficient, value in zip(
                pilot["interaction_coefficients"], cells, strict=True))))
    values = np.asarray(interactions, dtype=np.float64)
    summary = protocol["variance_summary"]
    ddof = int(summary["sample_sd_ddof"])
    sample_sd = float(np.std(values, ddof=ddof))
    generator = np.random.default_rng(int(summary["bootstrap_seed"]))
    indices = generator.integers(
        0, len(values), size=(int(summary["bootstrap_draws"]), len(values)))
    bootstrap_sd = np.std(values[indices], axis=1, ddof=ddof)
    upper = float(np.quantile(
        bootstrap_sd, float(summary["bootstrap_upper_quantile"])))
    return {
        "schema_version": 1,
        "n_families": len(families),
        "n_rows": len(rows),
        "family_interaction_mean_accuracy_points": float(values.mean()),
        "family_interaction_sample_sd": sample_sd,
        "family_interaction_bootstrap_sd_upper": upper,
        "planning_family_sd": max(sample_sd, upper),
        "parse_failure_rows": parse_failures,
        "family_interactions": [float(value) for value in values],
        "mechanical_summary": {
            "wrong_phase_hook_fires": wrong_phase,
            "minimum_expected_phase_hook_fires": expected_hook_min,
            "maximum_selected_protected_overlap": overlap_max,
            "exact_rank_match": rank_exact,
            "maximum_energy_relative_error": energy_max,
        },
        "mechanical_gate_checks": mechanical_checks,
        "pilot_analysis_valid": bool(all(mechanical_checks.values())),
        "outcome_boundary": (
            "Consumed development variance calibration only; the observed "
            "mean cannot select the SESOI and no row enters a primary."),
        "freeze_ready": False,
    }


def _parse_cell_name(cell: str, modes: Sequence[str], phases: Sequence[str],
                     arms: Sequence[str]) -> tuple[str, str, str]:
    matches = []
    for mode in modes:
        for phase in phases:
            for arm in arms:
                if cell == f"{mode}_{phase}_{arm}":
                    matches.append((mode, phase, arm))
    if len(matches) != 1:
        raise RuntimeError(f"invalid or ambiguous pilot cell: {cell}")
    return matches[0]


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    require_clean_tree()
    output = Path(config["outputs"]["protocol_result"])
    if arguments.author_protocol:
        result = author_protocol(config)
        if result["all_protocol_gates_pass"] is not True:
            raise RuntimeError("mode variance pilot protocol gates failed")
        atomic_json(output, result)
        print(json.dumps({
            "status": "protocol-authored-unregistered",
            "evidence_id": config["evidence_id"],
            "execution_authorized": result[
                "execution_authorized_at_this_boundary"],
            "freeze_ready": result["freeze_ready"],
        }, indent=1))
        return
    if not output.exists():
        raise RuntimeError("mode variance pilot protocol result is missing")
    result = json.loads(output.read_text())
    if result.get("all_protocol_gates_pass") is not True:
        raise RuntimeError("mode variance pilot protocol did not pass")
    if result.get("execution_authorized_at_this_boundary") is not False:
        raise RuntimeError("protocol authoring cannot authorize execution")
    if result.get("freeze_ready") is not False:
        raise RuntimeError("mode variance protocol cannot authorize freeze")
    command = (
        "python -m jspace_phase4.experiments.p4_qwen_mode_variance_pilot "
        f"--config {arguments.config} --register-protocol")
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            "Prospective development-only P4-P2 variance-pilot protocol: "
            "the consumed 20-family subset, exact eight-cell interaction, "
            "mechanical gates, and SD upper-bound summary are fixed; model "
            "execution remains blocked on the v2 baseline and canonical lens."),
        command=command, outputs=[output],
        inputs={
            "config": file_sha256(config_path),
            "mode_methods": config["source_methods"]["result_sha256"],
            "mode_feasibility": config[
                "source_feasibility"]["result_sha256"],
            "mode_v2_baseline_config": config[
                "source_model_baseline_contract"]["config_sha256"],
            "selection": config["selection"]["payload_sha256"],
        })
    print(json.dumps({
        "status": "protocol-registered",
        "evidence_id": config["evidence_id"],
    }, indent=1))


if __name__ == "__main__":
    main()
