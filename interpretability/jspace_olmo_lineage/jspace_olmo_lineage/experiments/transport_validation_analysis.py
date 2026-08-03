"""Registered joint analysis for the frozen Study-2 H6 transport grid.

This module joins the two mandatory checkpoint results and audits the
registered Phase-3/4 archive for the exact site records required by the
precommitted dose-matching rule.  It never promotes per-item summaries or
protected-subspace energy into total per-position intervention dose.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from ..manifests import (
    InputManifest,
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths import REPO_ROOT, figures_dir, metrics_dir
from ..provenance import Provenance, write_result
from ..registry import RegistryError, create, resolve, resolve_all
from .stage_wedge import configure_run_root

EVIDENCE_ID = "ol2-transport-validation-joint-v1"
MODEL_EVENTS = {
    "base": "ol2-transport-validation-base-v1",
    "olmo31_think": "ol2-transport-validation-olmo31-think-v1",
}
MODEL_SCOPES = {"olmo3-base", "olmo31-think"}
EXTERNAL_REGISTRIES = {
    "phase3": REPO_ROOT / "interpretability/jspace_phase3/reports/evidence_events.jsonl",
    "phase4": REPO_ROOT / "interpretability/jspace_phase4/reports/evidence_events.jsonl",
}
SITE_IDENTITY_FIELDS = {"item_id", "layer", "position"}
TOTAL_REMOVED_ENERGY_FIELDS = {"removed_energy_frac"}
RESIDUAL_RATIO_FIELDS = {
    "local_to_transport_residual_norm_ratio",
    "local_residual_norm_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _verified_event(evidence_id: str) -> tuple[dict, dict[str, Path]]:
    event = resolve(evidence_id)
    if not event["live"]:
        raise RuntimeError(f"required H6 event is not live: {evidence_id}")
    outputs = {}
    for row in event["outputs"]:
        path = Path(row["path"])
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"registered H6 output mismatch: {path}")
        outputs[path.name] = path
    return event, outputs


def _load_model_result(model_key: str, config: Mapping) -> dict:
    event, outputs = _verified_event(MODEL_EVENTS[model_key])
    result = json.loads(outputs["transport_result.json"].read_text())
    claimed_hash = result.pop("payload_sha256")
    if object_sha256(result) != claimed_hash:
        raise RuntimeError(f"H6 summary payload hash mismatch for {model_key}")
    result["payload_sha256"] = claimed_hash
    rows = pd.read_parquet(outputs["transport_rows.parquet"])
    expected = (
        len(config["prompt_ids"])
        * len(config["layers_zero_indexed"])
        * len(config["directions"]["families"])
        * len(config["relative_epsilon_ladder"])
    )
    if len(rows) != expected or int(result["rows"]) != expected:
        raise RuntimeError(f"H6 row count mismatch for {model_key}")
    specification = config["models"][model_key]
    if (
        result["model_id"] != specification["model_id"]
        or result["model_revision"] != specification["revision"]
        or event["model_revision"] != specification["revision"]
    ):
        raise RuntimeError(f"H6 model identity mismatch for {model_key}")
    return {
        "event": event,
        "outputs": outputs,
        "result": result,
        "rows": rows,
    }


def qualify_dose_candidate(
    columns: Sequence[str],
    *,
    model_scopes: Sequence[str],
) -> dict:
    """Classify whether a registered table can execute the frozen dose join."""
    names = set(map(str, columns))
    scopes = set(map(str, model_scopes))
    site_identity = SITE_IDENTITY_FIELDS <= names
    total_energy = bool(TOTAL_REMOVED_ENERGY_FIELDS & names)
    residual_ratio = bool(RESIDUAL_RATIO_FIELDS & names)
    relevant_model = bool(scopes & MODEL_SCOPES)
    reasons = []
    if not relevant_model:
        reasons.append("wrong_model_scope")
    if not site_identity:
        reasons.append("missing_exact_item_layer_position_identity")
    if not total_energy:
        if "removed_energy_in_prot_frac" in names:
            reasons.append("protected_subspace_energy_is_not_total_removed_energy")
        elif {"removed_energy_mean", "removed_energy_max"} & names:
            reasons.append("per_item_removed_energy_aggregate_is_not_site_record")
        else:
            reasons.append("missing_total_removed_energy_fraction")
    if not residual_ratio:
        reasons.append("missing_frozen_local_residual_norm_ratio")
    usable = relevant_model and site_identity and total_energy and residual_ratio
    return {
        "model_scope_relevant": relevant_model,
        "has_exact_site_identity": site_identity,
        "has_total_removed_energy_fraction": total_energy,
        "has_local_residual_norm_ratio": residual_ratio,
        "usable_for_frozen_dose_join": usable,
        "rejection_reasons": reasons,
    }


def _model_scopes(event: Mapping, path: Path, columns: Sequence[str]) -> list[str]:
    evidence_id = str(event["evidence_id"])
    scopes = set()
    if "olmo3-base" in evidence_id:
        scopes.add("olmo3-base")
    if "olmo31-think" in evidence_id:
        scopes.add("olmo31-think")
    if "model" in columns:
        values = pd.read_parquet(path, columns=["model"])["model"].dropna().unique()
        scopes.update(str(value) for value in values)
    if "model_key" in columns:
        values = pd.read_parquet(path, columns=["model_key"])[
            "model_key"
        ].dropna().unique()
        scopes.update(str(value).replace("_", "-") for value in values)
    return sorted(scopes)


def _nested_json_keys(path: Path, columns: Sequence[str]) -> list[str]:
    json_columns = [name for name in columns if str(name).endswith("_json")]
    if not json_columns:
        return []
    frame = pd.read_parquet(path, columns=json_columns).head(1)
    found = set()

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                found.add(str(key))
                walk(nested)
        elif isinstance(value, list):
            for nested in value[:3]:
                walk(nested)

    for name in json_columns:
        value = frame.iloc[0][name]
        if isinstance(value, str) and value:
            try:
                walk(json.loads(value))
            except json.JSONDecodeError:
                found.add(f"unparseable:{name}")
    return sorted(found)


def audit_registered_dose_sources(config: Mapping) -> dict:
    candidates = []
    registry_hashes = {}
    for registry_name, registry_path in EXTERNAL_REGISTRIES.items():
        if not registry_path.is_file():
            raise RuntimeError(f"registered source registry absent: {registry_path}")
        registry_hashes[registry_name] = {
            "path": str(registry_path),
            "sha256": file_sha256(registry_path),
        }
        for event in resolve_all(path=registry_path):
            if not event["live"]:
                continue
            for output in event.get("outputs", []):
                path = Path(output["path"])
                if path.suffix != ".parquet" or not path.is_file():
                    continue
                columns = list(pq.ParquetFile(path).schema.names)
                scopes = _model_scopes(event, path, columns)
                relevant_scope = bool(set(scopes) & MODEL_SCOPES)
                dose_named = any(
                    token in name.lower()
                    for name in columns
                    for token in (
                        "removed_energy", "residual_norm", "position",
                        "overlap_summary",
                    )
                )
                lineage_named = "lineage-grid" in str(event["evidence_id"])
                if not relevant_scope or not (dose_named or lineage_named):
                    continue
                observed = file_sha256(path)
                if observed != output["sha256"]:
                    raise RuntimeError(f"registered dose candidate hash mismatch: {path}")
                qualification = qualify_dose_candidate(
                    columns, model_scopes=scopes)
                candidates.append({
                    "source_registry": registry_name,
                    "evidence_id": event["evidence_id"],
                    "tier": event.get("tier"),
                    "path": str(path),
                    "sha256": observed,
                    "rows": int(pq.ParquetFile(path).metadata.num_rows),
                    "columns": columns,
                    "nested_json_keys_first_row": _nested_json_keys(path, columns),
                    "model_scopes": scopes,
                    "qualification": qualification,
                })
    candidates.sort(key=lambda row: (
        row["source_registry"], row["evidence_id"], row["path"]))
    usable = [
        row for row in candidates
        if row["qualification"]["usable_for_frozen_dose_join"]
    ]
    route = (
        "exact_registered_site_records_available"
        if usable else "unresolved_missing_registered_site_dose_records"
    )
    return {
        "schema_version": 1,
        "frozen_mapping": config["dose_matching"]["effective_relative_epsilon"],
        "required_join_key": list(config["dose_matching"]["join_key"]),
        "coverage_floor": float(
            config["dose_matching"]["coverage_floor_for_in_band_pass"]),
        "source_registries": registry_hashes,
        "n_relevant_registered_tables_audited": len(candidates),
        "n_usable_exact_site_tables": len(usable),
        "candidates": candidates,
        "route": route,
        "causal_assay_epsilon_distribution": None,
        "coverage_by_checkpoint_and_layer": None,
        "epsilon_0_10_above_typical_causal_dose": None,
        "missing_values_are_not_zero": True,
        "claim_boundary": (
            "Only exact registered item/layer/position total-dose records with "
            "the frozen residual-norm conversion may populate coverage. Per-item "
            "means/maxima and protected-subspace energy fractions are ineligible."
        ),
    }


def _nonnull_numeric_values_finite(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return all(
        np.isfinite(numeric[column].dropna().to_numpy(dtype=float)).all()
        for column in numeric
    )


def _model_summary(model_key: str, model: Mapping) -> dict:
    result = model["result"]
    rows = model["rows"]
    numeric = rows.select_dtypes(include=[np.number])
    null_counts = {
        str(column): int(numeric[column].isna().sum())
        for column in numeric
        if bool(numeric[column].isna().any())
    }
    return {
        "model_key": model_key,
        "evidence_id": result["evidence_id"],
        "intrinsic_transport_route": result["intrinsic_transport_route"],
        "valid_epsilons_by_layer": result["valid_epsilons_by_layer"],
        "common_assay_valid_epsilons": result[
            "common_assay_valid_epsilons"],
        "late_anchor_valid_epsilons": result["late_anchor_valid_epsilons"],
        "transport_rows": len(rows),
        "measurement_eligible_rows": int(rows.measurement_eligible.sum()),
        "decision_eligible_rows": int(rows.decision_eligible.sum()),
        "passing_rows": int(rows.transport_row_passed.sum()),
        "backend_gate_passing_rows": int(rows.backend_gate_passed.sum()),
        "maximum_backend_tangent_relative_error": float(
            rows.backend_tangent_relative_error.max()),
        "all_nonnull_numeric_values_finite": _nonnull_numeric_values_finite(rows),
        "undefined_metric_null_counts": null_counts,
        "null_semantics": (
            "Undefined forward/central metrics below the evaluator's numerical "
            "measurability conditions are missing, not nonfinite measurements."
        ),
        "passage": result["passage"],
    }


def joint_route(
    checkpoint_summaries: Mapping[str, Mapping],
    dose_audit: Mapping,
) -> dict:
    common = {
        key: list(value["common_assay_valid_epsilons"])
        for key, value in checkpoint_summaries.items()
    }
    late = {
        key: list(value["late_anchor_valid_epsilons"])
        for key, value in checkpoint_summaries.items()
    }
    any_common = any(common.values())
    any_late = any(late.values())
    if any_common:
        intrinsic = "checkpoint_specific_in_band_regime_measured"
    elif any_late:
        intrinsic = "h6_fail_in_band_with_checkpoint_specific_late_anchor"
    else:
        intrinsic = "h6_fail_no_licensed_regime_measured"
    dose_available = dose_audit["route"] == (
        "exact_registered_site_records_available")
    return {
        "intrinsic_joint_route": intrinsic,
        "checkpoint_common_assay_valid_epsilons": common,
        "checkpoint_late_anchor_valid_epsilons": late,
        "in_band_pass_on_frozen_ladder": any_common,
        "late_anchor_pass_at_any_checkpoint": any_late,
        "relevant_dose_route": dose_audit["route"],
        "intervention_distribution_coverage": None if not dose_available else (
            "requires_exact_join"),
        "h6_pass_in_band_at_relevant_doses": None if not dose_available else False,
        "h6_scale_limited": None if not dose_available else False,
        "licensed_in_band_wording": (
            "Across both mandatory checkpoints, the average or prompt-specific "
            "first-order transport approximation does not meet the frozen "
            "finite-dose gate at L24/L32/L40 on the tested ladder. This does "
            "not invalidate paired ablation effects."
        ),
        "licensed_late_anchor_wording": (
            "OLMo-3.1 Think passes the frozen transport gate only at the L56 "
            "late anchor at epsilon 0.10; Base does not meet the 0.90 passage "
            "floor there."
        ),
        "dose_wording": (
            "Intervention-relevant dose coverage is unresolved because the "
            "registered OLMo archive lacks the exact site records required by "
            "the frozen mapping; missing coverage is not zero coverage."
        ),
    }


def make_figure(
    checkpoint_summaries: Mapping[str, Mapping],
    dose_audit: Mapping,
    config: Mapping,
    png_path: Path,
    pdf_path: Path,
) -> None:
    colors = {24: "#005f73", 32: "#0a9396", 40: "#ee9b00", 56: "#ae2012"}
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    labels = {"base": "Base", "olmo31_think": "OLMo-3.1 Think"}
    floor = float(config["transport_gate"]["row_passage_floor"])
    for axis, model_key in zip(axes, ("base", "olmo31_think")):
        passage = pd.DataFrame(checkpoint_summaries[model_key]["passage"])
        for layer in config["layers_zero_indexed"]:
            selected = passage[passage.source_layer == int(layer)].sort_values(
                "relative_epsilon")
            axis.plot(
                selected.relative_epsilon,
                selected.passage_fraction,
                marker="o",
                linewidth=1.8,
                color=colors[int(layer)],
                label=f"L{layer}",
            )
        axis.axhline(floor, color="#444444", linestyle="--", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_xlim(0.0008, 0.13)
        axis.set_ylim(-0.04, 1.05)
        axis.set_xlabel("Relative epsilon (frozen ladder)")
        axis.set_ylabel("Fraction passing frozen row gate")
        axis.set_title(labels[model_key])
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, ncol=2)
    figure.suptitle(
        "OLMo Study 2 H6: no in-band licensed regime on the frozen ladder",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.005,
        (
            "Dashed line: 0.90 passage floor. Think passes only L56 at "
            "epsilon 0.10. Registered site-dose coverage: "
            f"{dose_audit['route'].replace('_', ' ')}."
        ),
        ha="center",
        fontsize=8.8,
        color="#444444",
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)


def registered_replay() -> dict | None:
    try:
        event = resolve(EVIDENCE_ID)
    except RegistryError:
        return None
    failures = []
    for row in event["outputs"]:
        path = Path(row["path"])
        actual = file_sha256(path) if path.is_file() else None
        if actual != row["sha256"]:
            failures.append({
                "path": str(path), "actual": actual, "expected": row["sha256"]})
    if failures:
        raise RuntimeError(json.dumps(failures, sort_keys=True))
    return {"already_registered": True, "n_outputs_verified": len(event["outputs"])}


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise TypeError("transport config must be a mapping")
    configure_run_root(config)
    clean = require_clean_tree(expected_branch=config["branch"])
    replay = registered_replay()
    if replay is not None:
        print(json.dumps(replay, indent=1))
        return
    if config.get("status") != "FROZEN_PRE_TRANSPORT_DATA":
        raise RuntimeError("transport thresholds are not frozen")
    models = {
        key: _load_model_result(key, config)
        for key in ("base", "olmo31_think")
    }
    checkpoint_summaries = {
        key: _model_summary(key, model) for key, model in models.items()
    }
    dose_audit = audit_registered_dose_sources(config)
    route = joint_route(checkpoint_summaries, dose_audit)
    combined = pd.concat(
        [models[key]["rows"].assign(joint_model_key=key) for key in models],
        ignore_index=True,
    )
    output_dir = metrics_dir("transport-validation-joint") / EVIDENCE_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "transport_joint_rows.parquet"
    combined.to_parquet(rows_path, index=False, compression="zstd")
    audit_path = output_dir / "registered_dose_source_audit.json"
    atomic_json(audit_path, dose_audit)
    png_path = figures_dir() / "ol2_transport_validation_joint.png"
    pdf_path = figures_dir() / "ol2_transport_validation_joint.pdf"
    make_figure(checkpoint_summaries, dose_audit, config, png_path, pdf_path)
    upstream = {
        f"{key}:{name}": file_sha256(path)
        for key, model in models.items()
        for name, path in model["outputs"].items()
    }
    upstream.update({
        f"dose_registry:{key}": value["sha256"]
        for key, value in dose_audit["source_registries"].items()
    })
    manifest = InputManifest(
        experiment_id=EVIDENCE_ID,
        config_sha256=file_sha256(config_path),
        model_id="allenai/Olmo-3-1125-32B + allenai/Olmo-3.1-32B-Think",
        model_revision=object_sha256({
            key: config["models"][key]["revision"] for key in models}),
        tokenizer_manifest_sha256=object_sha256({
            key: model["result"]["input_manifest_sha256"]
            for key, model in models.items()}),
        lens_sha256=object_sha256({"transport_lens": "not_applicable"}),
        bank_sha256=file_sha256(
            REPO_ROOT / "interpretability/jspace_gemma/data/g1_prompts_v1.jsonl"),
        partition_sha256=object_sha256({
            "prompt_ids": config["prompt_ids"],
            "layers": config["layers_zero_indexed"],
            "directions": config["directions"],
            "epsilons": config["relative_epsilon_ladder"],
        }),
        scoring_spec_sha256=object_sha256({
            "delivery": config["delivery"],
            "response_snr": config["response_snr"],
            "transport_gate": config["transport_gate"],
            "dose_matching": config["dose_matching"],
        }),
        upstream=upstream,
        code_commit=clean["code_commit"],
    )
    manifest_path = output_dir / "input_manifest.json"
    atomic_json(manifest_path, manifest.envelope())
    payload = {
        "schema_version": 1,
        "tier": "methods",
        "analysis": "registered Base/Think H6 joint router and dose-source audit",
        "thresholds_frozen_before_transport_data": True,
        "config_sha256": file_sha256(config_path),
        "checkpoint_summaries": checkpoint_summaries,
        "dose_audit": dose_audit,
        "router": route,
        "total_rows": len(combined),
        "all_nonnull_numeric_values_finite": _nonnull_numeric_values_finite(
            combined),
        "claim_boundary": (
            "Methods-tier finite-dose validation on four prompts, three frozen "
            "directions, four layers, and the frozen epsilon ladder. It neither "
            "invalidates paired ablations nor identifies a training-objective "
            "effect. Dose coverage remains unavailable unless exact registered "
            "site records are later imported without changing this result."
        ),
    }
    result_path = output_dir / "transport_joint_result.json"
    command = (
        "python -m jspace_olmo_lineage.experiments.transport_validation_analysis "
        f"--config {config_path}"
    )
    write_result(
        payload,
        result_path,
        Provenance(
            evidence_id=EVIDENCE_ID,
            tier="methods",
            command=command,
            inputs={"input_manifest": file_sha256(manifest_path), **upstream},
            input_manifest_sha256=manifest.sha256(),
            model=None,
            seed_contract="no stochastic joint analysis",
        ),
    )
    event = create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "Joint frozen H6 analysis: Base fails the tested ladder; Think is "
            "late-anchor-only; exact registered intervention-dose coverage is "
            "unresolved rather than zero."
        ),
        command=command,
        outputs=[
            result_path, rows_path, audit_path, manifest_path, png_path, pdf_path],
        inputs={
            "checkpoint_events": MODEL_EVENTS,
            "input_manifest": file_sha256(manifest_path),
            "source_registry_hashes": {
                key: value["sha256"]
                for key, value in dose_audit["source_registries"].items()
            },
        },
        route=route["intrinsic_joint_route"],
        relevant_dose_route=route["relevant_dose_route"],
        thresholds_changed_after_data=False,
        paired_ablation_invalidated=False,
    )
    print(json.dumps({"payload": payload, "event": event}, indent=1))


if __name__ == "__main__":
    main()
