"""Apply the precommitted G2.2 Stage-1 license router exactly once."""
from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from jspace_gemma.manifests import (
    atomic_json,
    atomic_text,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import PACKAGE_ROOT, resolve_uri, run_root
from jspace_gemma.registry import create, read_events, resolve
from jspace_gemma.stage1_license import select_stage1_route


BRANCH = "interp_jspace_gemma_transport_2"
CONFIG = PACKAGE_ROOT / "configs/gm2_stage1_relicense.yaml"
CANDIDATES = PACKAGE_ROOT / "protocol/G2_STAGE1_CANDIDATE_SENTENCES.md"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _output_with_sha(event: dict, expected_sha: str) -> Path:
    matches = [
        Path(row["path"])
        for row in event.get("outputs", [])
        if row.get("sha256") == expected_sha
    ]
    if len(matches) != 1 or file_sha256(matches[0]) != expected_sha:
        raise RuntimeError(f"registered source hash is absent or ambiguous: {expected_sha}")
    return matches[0]


def main() -> None:
    git = require_clean_tree(branch=BRANCH)
    config = yaml.safe_load(CONFIG.read_text())
    if config.get("status") != "FROZEN_PRE_G2_1_SEPARATE_TARGET_FILE":
        raise RuntimeError("G2.2 decision config is not the frozen target file")
    root = run_root()
    if root.resolve() != Path(config["run_root"]).resolve():
        raise RuntimeError("G2.2 run root differs from frozen config")
    if file_sha256(resolve_uri(config["sentence_source"])) != file_sha256(CANDIDATES):
        raise RuntimeError("precommitted candidate-sentence source drifted")

    calibration = resolve(config["dependency"]["calibration_evidence_id"])
    if (
        not calibration["live"]
        or calibration.get("no_target_read_assertion") is not True
        or calibration.get("ceiling_frozen_before_registry_read") is not True
    ):
        raise RuntimeError("registered target-blind G2.1 dependency is absent")
    threshold_rows = [
        row
        for row in calibration["outputs"]
        if Path(row["path"]).name == "backend_ceiling_frozen.json"
    ]
    if len(threshold_rows) != 1:
        raise RuntimeError("registered G2.1 threshold output is ambiguous")
    threshold_path = Path(threshold_rows[0]["path"])
    if file_sha256(threshold_path) != threshold_rows[0]["sha256"]:
        raise RuntimeError("registered G2.1 threshold hash drifted")
    threshold = json.loads(threshold_path.read_text())
    if (
        threshold.get("status") != "FROZEN_PRE_G2_2"
        or threshold.get("stage1_target_read") is not False
        or threshold.get("source_event_id") != calibration["evidence_id"]
    ):
        raise RuntimeError("G2.1 threshold is not licensed for G2.2")

    source = config["historical_source"]
    stage1 = resolve(source["stage1_evidence_id"])
    parity = resolve(source["backend_parity_evidence_id"])
    if not stage1["live"] or not parity["live"]:
        raise RuntimeError("historical Stage-1 evidence is not live")
    stage1_summary_path = _output_with_sha(stage1, source["stage1_summary_sha256"])
    stage1_rows_path = _output_with_sha(stage1, source["stage1_rows_sha256"])
    parity_artifact_path = _output_with_sha(
        parity, source["backend_parity_artifact_sha256"]
    )
    parity_raw_path = _output_with_sha(parity, source["backend_parity_raw_sha256"])
    source_hashes_exact = all(
        (
            file_sha256(stage1_summary_path) == source["stage1_summary_sha256"],
            file_sha256(stage1_rows_path) == source["stage1_rows_sha256"],
            file_sha256(parity_artifact_path)
            == source["backend_parity_artifact_sha256"],
            file_sha256(parity_raw_path) == source["backend_parity_raw_sha256"],
        )
    )
    stage1_summary = json.loads(stage1_summary_path.read_text())
    parity_artifact = json.loads(parity_artifact_path.read_text())
    comparison = parity_artifact["comparisons"][
        "primary_vs_fallback_tangent_all_slots"
    ]
    selected = parity_artifact["comparisons"][
        "primary_vs_fallback_tangent_selected_slot"
    ]
    historical_exact = float(comparison["relative_error"])
    historical_declared = float(source["historical_all_slot_relative_error"])
    if round(historical_exact, 6) != round(historical_declared, 6):
        raise RuntimeError("historical declared backend disagreement drifted")
    if (
        round(float(comparison["cosine"]), 8)
        != round(float(source["historical_all_slot_cosine"]), 8)
        or float(comparison["max_absolute_error"])
        != float(source["historical_max_absolute_difference"])
        or bool(float(selected["max_absolute_error"]) == 0.0)
        is not bool(source["historical_selected_slot_bit_identical"])
    ):
        raise RuntimeError("historical backend-parity metrics drifted")

    decision = select_stage1_route(
        threshold, config, source_hashes_exact=source_hashes_exact
    )
    evidence_id = decision["evidence_id"]
    known = {
        row["evidence_id"]
        for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
    }
    terminal_ids = {
        config[name]["evidence_id"]
        for name in (
            "branch_1_relicense_without_recompute",
            "branch_2_batch1_declared_dose",
            "branch_3_remains_blocked",
        )
    }
    if known & terminal_ids:
        raise RuntimeError("a G2.2 terminal event already exists")
    if decision["branch"] == "branch_2_batch1_declared_dose":
        raise RuntimeError("precommitted branch 2 requires its declared model replay")

    layer_decisions = stage1_summary["analysis"]["primary_layer_decisions"]
    classifier = [
        {"layer": int(row["source_layer"]), "decision": row["decision"]}
        for row in layer_decisions
    ]
    expected_layers = [22, 30, 37, 44, 52]
    classifier_all_layers = (
        [row["layer"] for row in classifier] == expected_layers
        and all(row["decision"] == "local_tangent_mismatch" for row in classifier)
    )
    if decision["branch"] == "branch_1_relicense_without_recompute" and not classifier_all_layers:
        raise RuntimeError("historical five-layer classifier is not the frozen object")

    ceiling = float(threshold["licensed_ceilings"]["gemma"])
    if decision["branch"] == "branch_1_relicense_without_recompute":
        licensed_sentence = (
            "Under the prospectively calibrated pooled all-frozen-batches backend "
            f"envelope, the registered study-1 all-slot disagreement ({historical_exact:.17g}; "
            f"predeclared rounded value {historical_declared:.6f}) lies within the frozen "
            f"ceiling ({ceiling:.17g}), while the selected scientific slot remains "
            "bit-identical. The unchanged five-layer local_tangent_mismatch classifier "
            "is therefore licensed as a closed exact-JVP finite-scale methods result on "
            "the tested prompts, layers, directions, target map, and doses; it is not a "
            "claim of nondifferentiability, missing information, or workspace absence."
        )
        universal_sentence = (
            "At the tested prompts, layers, directions, and intervention-relevant finite "
            "scales, the prompt-specific first-order tangent of the chosen source-to-target "
            "residual map predicts Gemma's finite response substantially less accurately "
            "than the same estimator predicts the OLMo control; the mismatch changes "
            "character with depth."
        )
    else:
        licensed_sentence = (
            "The study-2 calibration did not supply all requirements for relicensing; "
            "Gemma Stage 1 remains a methods blocker under the precommitted route."
        )
        universal_sentence = None

    payload = {
        "schema_version": 1,
        "evidence_id": evidence_id,
        "tier": "methods",
        "created_utc": _utc(),
        "code_commit": git["code_commit"],
        "decision_config": {"path": str(CONFIG), "sha256": file_sha256(CONFIG)},
        "candidate_sentences": {
            "path": str(CANDIDATES),
            "sha256": file_sha256(CANDIDATES),
            "committed_before_g2_1": True,
        },
        "calibration_dependency": {
            "evidence_id": calibration["evidence_id"],
            "event_code_commit": calibration["code_commit"],
            "threshold_path": str(threshold_path),
            "threshold_sha256": file_sha256(threshold_path),
            "route": threshold["route"],
            "applicable_scope": threshold["applicable_scope"],
            "licensed_ceilings": threshold["licensed_ceilings"],
            "no_target_read_assertion": threshold["no_target_read_assertion"],
        },
        "historical_sources": {
            "stage1_evidence_id": stage1["evidence_id"],
            "stage1_summary": {
                "path": str(stage1_summary_path),
                "sha256": file_sha256(stage1_summary_path),
            },
            "stage1_rows": {
                "path": str(stage1_rows_path),
                "sha256": file_sha256(stage1_rows_path),
                "preserved_without_selection_or_recompute": True,
            },
            "backend_parity_evidence_id": parity["evidence_id"],
            "backend_parity_artifact": {
                "path": str(parity_artifact_path),
                "sha256": file_sha256(parity_artifact_path),
            },
            "backend_parity_raw": {
                "path": str(parity_raw_path),
                "sha256": file_sha256(parity_raw_path),
            },
            "historical_all_slot_relative_error_exact": historical_exact,
            "historical_all_slot_relative_error_declared": historical_declared,
            "historical_selected_slot_bit_identical": True,
            "source_hashes_exact": source_hashes_exact,
        },
        "decision": decision,
        "stage1_classifier": {
            "rows": classifier,
            "all_five_layers_local_tangent_mismatch": classifier_all_layers,
            "promoted_from_operational_diagnostic_to_closed_methods_result": decision[
                "branch"
            ]
            == "branch_1_relicense_without_recompute",
        },
        "licensed_sentence": licensed_sentence,
        "universal_ceiling_sentence": universal_sentence,
        "stronger_claims_forbidden": config["stronger_claims_forbidden"],
        "olmo_h6_export": {
            "licensed": decision["branch"] == "branch_1_relicense_without_recompute",
            "applicable_model": "olmo3_32b_control",
            "ceiling": threshold["licensed_ceilings"]["olmo_control"],
            "source_calibration_evidence_id": calibration["evidence_id"],
            "source_threshold_sha256": file_sha256(threshold_path),
        },
        "g2_2_model_compute_performed": False,
        "historical_outcomes_opened_after_g2_1_registration": True,
        "claim_tier": "methods/development",
    }
    payload["payload_sha256"] = object_sha256(payload)
    output_root = root / "derived" / evidence_id
    output = output_root / "stage1_license_decision.json"
    sentence_output = output_root / "licensed_sentence.md"
    if output.exists() or sentence_output.exists():
        raise FileExistsError("refusing to overwrite an unregistered G2.2 decision")
    atomic_json(output, payload)
    atomic_text(
        sentence_output,
        "# G2.2 licensed sentence\n\n"
        f"Evidence: `{evidence_id}`. Tier: methods/development.\n\n"
        f"{licensed_sentence}\n\n"
        + (
            "## Universal ceiling\n\n" + universal_sentence + "\n"
            if universal_sentence is not None
            else ""
        ),
    )
    create(
        evidence_id,
        tier="methods",
        what=(
            "Mechanical G2.2 relicensing of the immutable Gemma Stage-1 object "
            "under the registered target-blind backend envelope."
            if decision["branch"] == "branch_1_relicense_without_recompute"
            else "Mechanical G2.2 terminal blocked decision under the frozen router."
        ),
        command="python -m jspace_gemma.experiments.gm2_stage1_license",
        outputs=[output, sentence_output],
        inputs={
            "decision_config_sha256": file_sha256(CONFIG),
            "candidate_sentences_sha256": file_sha256(CANDIDATES),
            "calibration_threshold_sha256": file_sha256(threshold_path),
            "stage1_summary_sha256": file_sha256(stage1_summary_path),
            "stage1_rows_sha256": file_sha256(stage1_rows_path),
            "backend_parity_artifact_sha256": file_sha256(parity_artifact_path),
            "backend_parity_raw_sha256": file_sha256(parity_raw_path),
        },
        selected_branch=decision["branch"],
        calibration_route=threshold["route"],
        applicable_ceiling=decision["conditions"]["applicable_ceiling"],
        historical_all_slot_relative_error=historical_exact,
        all_five_layers_local_tangent_mismatch_licensed=bool(
            decision["branch"] == "branch_1_relicense_without_recompute"
            and classifier_all_layers
        ),
        olmo_h6_ceiling_export=payload["olmo_h6_export"],
        model_compute_performed=False,
        historical_rows_preserved=True,
        target_model_opened=False,
    )
    print(
        json.dumps(
            {
                "evidence_id": evidence_id,
                "branch": decision["branch"],
                "applicable_ceiling": decision["conditions"]["applicable_ceiling"],
                "historical_error_exact": historical_exact,
                "all_five_layers_local_tangent_mismatch": classifier_all_layers,
                "output": str(output),
                "output_sha256": file_sha256(output),
                "sentence": str(sentence_output),
                "sentence_sha256": file_sha256(sentence_output),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
