"""Registered joint analysis and mechanical router for the Study-2 wedge."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ..manifests import (
    InputManifest,
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths import figures_dir, metrics_dir
from ..provenance import Provenance, write_result
from ..registry import RegistryError, create, resolve
from .stage_wedge import configure_run_root

EVIDENCE_ID = "ol2-stage-wedge-joint-analysis-v1"
STAGES = ("think_sft", "think_dpo")
FRAMES = ("base-lens-common", "olmo3-think-endpoint-own")
PAIR_KEYS = ("item_id", "fact_id", "canonical_family", "variant", "bank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _verified_event(evidence_id: str) -> tuple[dict, dict[str, Path]]:
    event = resolve(evidence_id)
    if not event["live"]:
        raise RuntimeError(f"required stage event is not live: {evidence_id}")
    paths = {}
    for row in event["outputs"]:
        path = Path(row["path"])
        observed = file_sha256(path) if path.is_file() else None
        if observed != row["sha256"]:
            raise RuntimeError(f"registered output mismatch for {evidence_id}: {path}")
        paths[path.name] = path
    return event, paths


def _envelope(path: Path) -> dict:
    value = json.loads(path.read_text())
    if object_sha256(value["payload"]) != value["payload_sha256"]:
        raise RuntimeError(f"payload hash mismatch: {path}")
    return value


def load_stage(config: Mapping, key: str) -> dict:
    specification = config["checkpoints"][key]
    event, paths = _verified_event(specification["evidence_id"])
    stage_result = _envelope(paths["stage_result.json"])
    g5_result = _envelope(paths["g5_capability.json"])
    cohort = _envelope(paths["cohort_manifest.json"])
    manifest = _envelope(paths["input_manifest.json"])
    if stage_result["provenance"]["evidence_id"] != specification["evidence_id"]:
        raise RuntimeError(f"stage result evidence mismatch for {key}")
    source = manifest["payload"]
    if source["model_id"] != specification["model_id"]:
        raise RuntimeError(f"stage model ID mismatch for {key}")
    if source["model_revision"] != specification["revision"]:
        raise RuntimeError(f"stage model revision mismatch for {key}")
    g5 = pd.read_parquet(paths["g5_capability.parquet"])
    if len(g5) != int(config["g5_capability"]["battery_rows_expected"]):
        raise RuntimeError(f"G5 row count mismatch for {key}")
    frame_paths = {
        frame: paths.get(f"tier1_{frame.replace('-', '_')}.parquet") for frame in FRAMES
    }
    frames = {
        frame: pd.read_parquet(path)
        for frame, path in frame_paths.items()
        if path is not None
    }
    passed = bool(cohort["payload"]["capability_gate_passed"])
    if passed != (stage_result["payload"]["status"] == "complete"):
        raise RuntimeError(f"stage/cohort gate mismatch for {key}")
    if passed and set(frames) != set(FRAMES):
        raise RuntimeError(f"capable stage lacks both frames: {key}")
    if not passed and frames:
        raise RuntimeError(f"gated stage unexpectedly has intervention rows: {key}")
    return {
        "key": key,
        "specification": specification,
        "event": event,
        "paths": paths,
        "stage_result": stage_result,
        "g5_result": g5_result,
        "cohort": cohort,
        "manifest": manifest,
        "g5": g5,
        "frames": frames,
        "capability_gate_passed": passed,
    }


def capability_transition(stages: Mapping[str, Mapping]) -> tuple[pd.DataFrame, dict]:
    left = stages["think_sft"]["g5"].copy()
    right = stages["think_dpo"]["g5"].copy()
    required = set(PAIR_KEYS) | {
        "capable_generation",
        "generation",
        "score_aggregate",
        "alias_set_hash",
    }
    for name, frame in (("SFT", left), ("DPO", right)):
        missing = required - set(frame)
        if missing:
            raise RuntimeError(f"{name} G5 lacks {sorted(missing)}")
        if frame.duplicated(list(PAIR_KEYS)).any():
            raise RuntimeError(f"{name} G5 has duplicate item keys")
    paired = left[
        list(PAIR_KEYS)
        + [
            "capable_generation",
            "generation",
            "score_aggregate",
            "alias_set_hash",
        ]
    ].merge(
        right[
            list(PAIR_KEYS)
            + [
                "capable_generation",
                "generation",
                "score_aggregate",
                "alias_set_hash",
            ]
        ],
        on=list(PAIR_KEYS),
        suffixes=("_sft", "_dpo"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not bool((paired._merge == "both").all()):
        raise RuntimeError("SFT/DPO G5 batteries do not pair exactly")
    paired = paired.drop(columns="_merge")
    if not bool((paired.alias_set_hash_sft == paired.alias_set_hash_dpo).all()):
        raise RuntimeError("SFT/DPO alias token manifests differ")
    paired["capability_transition"] = np.select(
        [
            paired.capable_generation_sft & paired.capable_generation_dpo,
            (~paired.capable_generation_sft) & paired.capable_generation_dpo,
            paired.capable_generation_sft & (~paired.capable_generation_dpo),
        ],
        ["capable_both", "onset_at_dpo", "lost_at_dpo"],
        default="incapable_both",
    )
    bank_variant = {}
    for bank in sorted(paired.bank.unique()):
        for variant in sorted(paired.variant.unique()):
            rows = paired[(paired.bank == bank) & (paired.variant == variant)]
            if rows.empty:
                continue
            bank_variant[f"{bank}:{variant}"] = {
                "n_items": len(rows),
                "sft_rate": float(rows.capable_generation_sft.mean()),
                "dpo_rate": float(rows.capable_generation_dpo.mean()),
                "dpo_minus_sft": float(
                    rows.capable_generation_dpo.mean()
                    - rows.capable_generation_sft.mean()
                ),
                "transition_counts": {
                    str(key): int(value)
                    for key, value in rows.capability_transition.value_counts().items()
                },
            }
    summary = {
        "n_paired_items": len(paired),
        "alias_manifests_identical": True,
        "by_bank_variant": bank_variant,
        "overall_transition_counts": {
            str(key): int(value)
            for key, value in paired.capability_transition.value_counts().items()
        },
        "stage_gate": {
            key: {
                "passed": bool(stage["capability_gate_passed"]),
                "n_bank_s_direct_composed_facts": int(
                    stage["cohort"]["payload"]["n_facts"]
                ),
                "n_bank_s_families": int(stage["cohort"]["payload"]["n_families"]),
            }
            for key, stage in stages.items()
        },
    }
    return paired, summary


def add_effects(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["j_effect"] = result.lp_meanJ_span_safe - result.lp_baseline
    result["matched_effect"] = (
        result.lp_instant_rank_energy_matched - result.lp_baseline
    )
    result["specific"] = result.j_effect - result.matched_effect
    result["label_effect"] = result.lp_meanJ_label_protected - result.lp_baseline
    result["protected_energy_effect"] = (
        result.lp_protected_energy_matched - result.lp_baseline
    )
    result["label_specific"] = result.label_effect - result.protected_energy_effect
    result["mechanics_effect"] = result.lp_mechanics_random - result.lp_baseline
    result["logit_effect"] = result.lp_logit_label_protected - result.lp_baseline
    return result


def bootstrap_family(
    rows: pd.DataFrame,
    column: str,
    *,
    draws: int,
    seed: int,
) -> dict:
    means = (
        rows.groupby("canonical_family", sort=True)[column].mean().to_numpy(dtype=float)
    )
    if len(means) < 3 or not np.isfinite(means).all():
        raise RuntimeError("family bootstrap needs at least three finite families")
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(0, len(means), size=(draws, len(means)))
    distribution = means[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.05, 0.95])
    return {
        "estimate": float(means.mean()),
        "ci90": [float(low), float(high)],
        "n_families": len(means),
        "n_bootstrap": draws,
        "bootstrap_seed": seed,
        "distribution_sha256": hashlib.sha256(
            np.asarray(distribution, dtype="<f8").tobytes()
        ).hexdigest(),
        "method": "equal-family-resampling-percentile",
    }


def summarize_effects(
    stages: Mapping[str, Mapping],
    router: Mapping,
) -> tuple[dict, dict]:
    if not all(stage["capability_gate_passed"] for stage in stages.values()):
        reason = (
            "At least one frozen stage capability gate failed; intervention "
            "effects and adjacent effect contrasts are missing, not zero."
        )
        return {
            "estimable": False,
            "reason": reason,
            "checkpoint_effects": None,
            "adjacent_contrasts": None,
            "leave_one_family_out": None,
            "baseline_capability_adjustment": None,
        }, {
            "sft_think_like_both_frames": None,
            "sft_base_like_both_frames": None,
            "dpo_minus_sft_resolved_negative_both_frames": None,
            "frame_agreement_passed": None,
        }

    draws = int(router["bootstrap_draws"])
    seed = int(router["bootstrap_seed"])
    summaries = {}
    direct_family = {}
    for stage_offset, (stage_key, stage) in enumerate(stages.items()):
        summaries[stage_key] = {}
        direct_family[stage_key] = {}
        for frame_offset, frame_name in enumerate(FRAMES):
            effects = add_effects(stage["frames"][frame_name])
            direct = effects[effects.variant == "direct"]
            composed = effects[effects.variant == "composed"]
            pivot = effects.pivot(
                index=["fact_id", "canonical_family"],
                columns="variant",
                values="specific",
            ).reset_index()
            pivot["composition"] = pivot.composed - pivot.direct
            summaries[stage_key][frame_name] = {
                "direct": bootstrap_family(
                    direct,
                    "specific",
                    draws=draws,
                    seed=seed + 100 * stage_offset + 10 * frame_offset,
                ),
                "composed": bootstrap_family(
                    composed,
                    "specific",
                    draws=draws,
                    seed=seed + 100 * stage_offset + 10 * frame_offset + 1,
                ),
                "composition": bootstrap_family(
                    pivot,
                    "composition",
                    draws=draws,
                    seed=seed + 100 * stage_offset + 10 * frame_offset + 2,
                ),
            }
            direct_family[stage_key][frame_name] = direct.groupby(
                "canonical_family", sort=True
            ).specific.mean()

    increments = {}
    for frame_offset, frame_name in enumerate(FRAMES):
        left = add_effects(stages["think_sft"]["frames"][frame_name])
        right = add_effects(stages["think_dpo"]["frames"][frame_name])
        left = left[left.variant == "direct"]
        right = right[right.variant == "direct"]
        paired = left[[*PAIR_KEYS, "specific"]].merge(
            right[[*PAIR_KEYS, "specific"]],
            on=list(PAIR_KEYS),
            suffixes=("_sft", "_dpo"),
            how="inner",
            validate="one_to_one",
        )
        paired["increment"] = paired.specific_dpo - paired.specific_sft
        increments[frame_name] = bootstrap_family(
            paired,
            "increment",
            draws=draws,
            seed=seed + 1000 + frame_offset,
        )

    think = router["think_like"]
    base = router["base_like"]
    increment_rule = router["resolved_adjacent_negative_increment"]
    sft_think = all(
        summaries["think_sft"][frame]["direct"]["estimate"]
        <= float(think["direct_estimate_lte"])
        and summaries["think_sft"][frame]["direct"]["ci90"][1]
        <= float(think["direct_interval_upper_lte"])
        for frame in FRAMES
    )
    sft_base = all(
        abs(summaries["think_sft"][frame]["direct"]["estimate"])
        <= float(base["absolute_direct_estimate_lte"])
        and summaries["think_sft"][frame]["direct"]["ci90"][0]
        <= 0
        <= summaries["think_sft"][frame]["direct"]["ci90"][1]
        for frame in FRAMES
    )
    resolved = all(
        increments[frame]["estimate"] <= float(increment_rule["estimate_lte"])
        and increments[frame]["ci90"][1] < float(increment_rule["interval_upper_lt"])
        for frame in FRAMES
    )
    frame_direct_difference = max(
        abs(
            summaries[stage][FRAMES[0]]["direct"]["estimate"]
            - summaries[stage][FRAMES[1]]["direct"]["estimate"]
        )
        for stage in STAGES
    )
    signs_agree = all(
        np.sign(summaries[stage][FRAMES[0]]["direct"]["estimate"])
        == np.sign(summaries[stage][FRAMES[1]]["direct"]["estimate"])
        for stage in STAGES
    )
    agreement = (
        frame_direct_difference
        <= float(router["frame_agreement"]["maximum_absolute_direct_difference"])
        and signs_agree
    )
    diagnostics = {
        "sft_think_like_both_frames": sft_think,
        "sft_base_like_both_frames": sft_base,
        "dpo_minus_sft_resolved_negative_both_frames": resolved,
        "frame_agreement_passed": agreement,
        "maximum_absolute_direct_frame_difference": frame_direct_difference,
        "same_sign_both_stages": signs_agree,
    }
    return {
        "estimable": True,
        "checkpoint_effects": summaries,
        "adjacent_contrasts": {"dpo_minus_sft_direct": increments},
        "leave_one_family_out": "reported in per-family side table when estimable",
        "baseline_capability_adjustment": "not implemented in this bounded router",
    }, diagnostics


def mechanical_route(
    capability: Mapping,
    diagnostics: Mapping,
    router: Mapping,
) -> dict:
    sft_pass = bool(capability["stage_gate"]["think_sft"]["passed"])
    dpo_pass = bool(capability["stage_gate"]["think_dpo"]["passed"])
    checks = {
        "capability_onset": (not sft_pass) and dpo_pass,
        "transition_by_sft_boundary": False,
        "transition_across_sft_to_dpo": False,
        "distributed_across_sft_dpo": False,
        "coordinate_only": False,
        "null_or_unresolved": False,
    }
    if sft_pass and dpo_pass:
        think = bool(diagnostics["sft_think_like_both_frames"])
        base = bool(diagnostics["sft_base_like_both_frames"])
        resolved = bool(diagnostics["dpo_minus_sft_resolved_negative_both_frames"])
        agreement = bool(diagnostics["frame_agreement_passed"])
        checks["transition_by_sft_boundary"] = think and not resolved and agreement
        checks["transition_across_sft_to_dpo"] = base and resolved and agreement
        checks["coordinate_only"] = not agreement
        checks["distributed_across_sft_dpo"] = (
            agreement
            and not checks["transition_by_sft_boundary"]
            and not checks["transition_across_sft_to_dpo"]
        )
    checks["null_or_unresolved"] = not any(
        checks[name] for name in checks if name != "null_or_unresolved"
    )
    order = list(router["routes_in_order"])
    if order != list(checks):
        raise RuntimeError("implemented route order differs from frozen config")
    selected = next(name for name in order if checks[name])
    wording = {
        "capability_onset": (
            "The available assay localizes capability onset, not an "
            "intervention-specific causal transition."
        ),
        "transition_by_sft_boundary": (
            "The tested J-space causal recruitment is installed by the "
            "SFT-stage boundary of the official Think recipe."
        ),
        "transition_across_sft_to_dpo": (
            "The tested causal recruitment appears across the SFT-to-DPO interval."
        ),
        "distributed_across_sft_dpo": (
            "Recruitment is distributed across the tested SFT/DPO wedge."
        ),
        "coordinate_only": (
            "The stage pattern is visible in only one frozen lens frame and "
            "is classified as coordinate finding, not recruitment."
        ),
        "null_or_unresolved": (
            "The available official SFT/DPO checkpoints do not localize the "
            "first-release transition under the frozen assay; later recipe "
            "stages, checkpoint differences, capability limits, or "
            "measurement variation remain open."
        ),
    }
    return {
        "route": selected,
        "route_checks_in_order": checks,
        "frozen_route_order": order,
        "licensed_wording": wording[selected],
        "capability_gated_effects_are_missing_not_zero": not (sft_pass and dpo_pass),
        "no_significance_comparison": bool(router["no_significance_comparison"]),
    }


def make_figure(payload: Mapping, path: Path) -> None:
    cells = ["F direct", "F composed", "S direct", "S composed"]
    keys = ["F:direct", "F:composed", "S:direct", "S:composed"]
    sft = [
        payload["capability_transition"]["by_bank_variant"][key]["sft_rate"]
        for key in keys
    ]
    dpo = [
        payload["capability_transition"]["by_bank_variant"][key]["dpo_rate"]
        for key in keys
    ]
    x = np.arange(len(cells))
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    axes[0].bar(x - 0.18, sft, width=0.36, label="Think-SFT", color="#0a9396")
    axes[0].bar(x + 0.18, dpo, width=0.36, label="Think-DPO", color="#ca6702")
    axes[0].axhline(1.0, color="#555555", linewidth=0.7)
    axes[0].set_xticks(x, cells, rotation=18, ha="right")
    axes[0].set_ylabel("Exact generation-capability rate")
    axes[0].set_ylim(0, max(0.04, 1.15 * max([*sft, *dpo])))
    axes[0].set_title("a  Frozen 972-row G5 battery")
    axes[0].legend(frameon=False)

    gate = payload["capability_transition"]["stage_gate"]
    facts = [gate[stage]["n_bank_s_direct_composed_facts"] for stage in STAGES]
    axes[1].bar([0, 1], facts, color=["#0a9396", "#ca6702"])
    axes[1].axhline(72, color="#9b2226", linestyle="--", label="Frozen fact floor")
    axes[1].set_xticks([0, 1], ["Think-SFT", "Think-DPO"])
    axes[1].set_ylabel("Bank-S facts capable on direct + composed")
    axes[1].set_title("b  Intervention cohort gate")
    axes[1].legend(frameon=False)
    figure.suptitle(
        "OLMo Study 2 SFT/DPO wedge: capability-gated joint route",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.005,
        "Development-tier natural experiment; gated effects are missing, not zero.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
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
            failures.append(
                {"path": str(path), "actual": actual, "expected": row["sha256"]}
            )
    if failures:
        raise RuntimeError(json.dumps(failures, sort_keys=True))
    return {"already_registered": True, "n_outputs_verified": len(event["outputs"])}


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise TypeError("stage-wedge config must be a mapping")
    configure_run_root(config)
    clean = require_clean_tree(expected_branch=config["branch"])
    replay = registered_replay()
    if replay is not None:
        print(json.dumps(replay, indent=1))
        return
    stages = {key: load_stage(config, key) for key in STAGES}
    paired, capability = capability_transition(stages)
    effects, diagnostics = summarize_effects(stages, config["stage_router"])
    route = mechanical_route(capability, diagnostics, config["stage_router"])
    payload = {
        "schema_version": 1,
        "tier": config["tier"],
        "analysis": "registered SFT/DPO stage-wedge joint router",
        "capability_definition": config["g5_capability"]["capability_definition"],
        "capability_transition": capability,
        "effects": effects,
        "router_diagnostics": diagnostics,
        "router": route,
        "predictions_frozen_before_stage_data": True,
        "natural_experiment_qualification": config["ancestry"]["qualification"],
        "claim_boundary": (
            "Development-tier official-checkpoint interval; no randomized "
            "training-objective attribution."
        ),
    }
    output_dir = metrics_dir("stage-wedge-joint") / EVIDENCE_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "capability_transition.parquet"
    paired.to_parquet(paired_path, index=False, compression="zstd")
    input_hashes = {
        str(path): file_sha256(path)
        for stage in stages.values()
        for path in stage["paths"].values()
    }
    manifest = InputManifest(
        experiment_id=EVIDENCE_ID,
        config_sha256=file_sha256(config_path),
        model_id="allenai/Olmo-3-32B-Think-SFT + Think-DPO",
        model_revision=object_sha256(
            {key: config["checkpoints"][key]["revision"] for key in STAGES}
        ),
        tokenizer_manifest_sha256=config["tokenizer_contract"][
            "semantic_fingerprint_sha256"
        ],
        lens_sha256=object_sha256(
            {
                key: config["inputs"][key]["sha256"]
                for key in ("frozen_base_lens", "frozen_olmo3_think_lens")
            }
        ),
        bank_sha256=object_sha256(
            {key: config["inputs"][key]["sha256"] for key in ("bank_f", "bank_s")}
        ),
        partition_sha256=object_sha256(
            {key: stages[key]["cohort"]["payload_sha256"] for key in STAGES}
        ),
        scoring_spec_sha256=stages["think_sft"]["manifest"]["payload"][
            "scoring_spec_sha256"
        ],
        upstream=input_hashes,
        code_commit=clean["code_commit"],
    )
    manifest_path = output_dir / "input_manifest.json"
    atomic_json(manifest_path, manifest.envelope())
    figure_path = figures_dir() / "ol2_stage_wedge_capability_route.png"
    make_figure(payload, figure_path)
    result_path = output_dir / "stage_wedge_joint_analysis.json"
    command = (
        "python -m jspace_olmo_lineage.experiments.stage_wedge_analysis "
        f"--config {config_path}"
    )
    write_result(
        payload,
        result_path,
        Provenance(
            evidence_id=EVIDENCE_ID,
            tier=config["tier"],
            command=command,
            inputs={
                "input_manifest": file_sha256(manifest_path),
                **input_hashes,
            },
            input_manifest_sha256=manifest.sha256(),
            model=None,
            seed_contract=(
                f"PCG64 family bootstrap; {config['stage_router']['bootstrap_draws']} "
                f"draws; seed {config['stage_router']['bootstrap_seed']}"
            ),
        ),
    )
    event = create(
        EVIDENCE_ID,
        tier=config["tier"],
        what=(
            "Registered OLMo SFT/DPO joint stage router: "
            f"{route['route']}; capability-gated effects remain missing, not zero."
        ),
        command=command,
        outputs=[result_path, paired_path, manifest_path, figure_path],
        inputs={
            "stage_events": {
                key: stages[key]["specification"]["evidence_id"] for key in STAGES
            },
            "input_manifest": file_sha256(manifest_path),
        },
        route=route["route"],
    )
    print(json.dumps({"payload": payload, "event": event}, indent=1))


if __name__ == "__main__":
    main()
