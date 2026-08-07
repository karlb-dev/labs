"""Mechanical G2.2 route selection over frozen registered inputs."""
from __future__ import annotations


def select_stage1_route(
    calibration: dict,
    decision_config: dict,
    *,
    source_hashes_exact: bool,
) -> dict:
    route = calibration["route"]
    ceilings = calibration["licensed_ceilings"]
    applicable = ceilings.get("gemma")
    historical = float(
        decision_config["historical_source"]["historical_all_slot_relative_error"]
    )
    selected_identical = bool(
        decision_config["historical_source"]["historical_selected_slot_bit_identical"]
    )
    path_ambiguity = route == "path_ambiguity"
    stable = applicable is not None

    conditions = {
        "calibration_route": route,
        "applicable_ceiling": applicable,
        "historical_all_slot_relative_error_declared": historical,
        "historical_error_lte_applicable_ceiling": bool(
            stable and historical <= float(applicable)
        ),
        "historical_error_gt_applicable_ceiling": bool(
            stable and historical > float(applicable)
        ),
        "selected_slot_bit_identical": selected_identical,
        "path_ambiguity": path_ambiguity,
        "stable_applicable_ceiling": stable,
        "source_hashes_exact": bool(source_hashes_exact),
    }
    branch2 = (
        route == "batch_composition_nuisance"
        and conditions["historical_error_gt_applicable_ceiling"]
        and not path_ambiguity
        and source_hashes_exact
    )
    branch1 = (
        conditions["historical_error_lte_applicable_ceiling"]
        and selected_identical
        and not path_ambiguity
        and source_hashes_exact
    )
    if path_ambiguity:
        branch = "branch_3_remains_blocked"
    elif branch2:
        branch = "branch_2_batch1_declared_dose"
    elif branch1:
        branch = "branch_1_relicense_without_recompute"
    else:
        branch = "branch_3_remains_blocked"
    return {
        "branch": branch,
        "evidence_id": decision_config[branch]["evidence_id"],
        "conditions": conditions,
    }
