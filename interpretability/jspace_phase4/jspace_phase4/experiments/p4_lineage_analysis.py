"""Deterministic development analysis for one OLMo lineage checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ..manifests import (
    InputManifest,
    atomic_json,
    file_sha256,
    require_clean_tree,
)
from ..paths4 import figures_dir, metrics_dir, resolve_uri
from ..provenance4 import Provenance4, write_result4
from ..registry4 import create
from ..seeds import SEED_CONTRACT
from ..stats4 import exact_signflip, family_bootstrap_percentile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def add_effects(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["J_effect"] = (
        result.lp_meanJ_span_safe - result.lp_baseline)
    result["control_effect"] = (
        result.lp_ss_matched - result.lp_baseline)
    result["specific"] = result.J_effect - result.control_effect
    result["label_effect"] = (
        result.lp_meanJ_label_protected - result.lp_baseline)
    result["protected_energy_effect"] = (
        result.lp_prot_energy_matched - result.lp_baseline)
    result["label_specific"] = (
        result.label_effect - result.protected_energy_effect)
    result["mechanics_effect"] = (
        result.lp_mechanics_random - result.lp_baseline)
    result["logit_effect"] = (
        result.lp_logit_label_protected - result.lp_baseline)
    return result


def bootstrap(rows: pd.DataFrame, column: str, *,
              draws: int, seed: int) -> dict:
    return family_bootstrap_percentile(
        rows,
        column,
        draws=draws,
        seed=seed,
    )


def effect_block(rows: pd.DataFrame, *,
                 draws: int, seed: int) -> dict:
    blocks = {
        column: bootstrap(
            rows, column, draws=draws, seed=seed + offset)
        for offset, column in enumerate(
            ("J_effect", "control_effect", "specific"))
    }
    family_specific = (
        rows.groupby("canonical_family").specific.mean().to_numpy())
    signflip = (
        exact_signflip(family_specific)
        if 3 <= len(family_specific) <= 22
        else None
    )
    return {
        "n_items": int(len(rows)),
        "n_facts": int(rows.fact_id.nunique()),
        "n_families": int(rows.canonical_family.nunique()),
        "family_weighted": blocks,
        "specific_exact_signflip_descriptive": signflip,
        "tail_rate": {
            str(threshold): {
                "J": float((rows.J_effect < threshold).mean()),
                "control": float(
                    (rows.control_effect < threshold).mean()),
            }
            for threshold in (-0.5, -1.0, -1.5, -2.0, -3.0)
        },
        "item_weighted": {
            column: float(rows[column].mean())
            for column in (
                "J_effect",
                "control_effect",
                "specific",
            )
        },
    }


def composition_rows(effects: pd.DataFrame) -> pd.DataFrame:
    pivot = effects.pivot(
        index=["bank", "fact_id", "canonical_family"],
        columns="variant",
        values="specific",
    ).reset_index()
    if "direct" not in pivot or "composed" not in pivot:
        raise ValueError("lineage grid lacks paired direct/composed rows")
    pivot["composition"] = pivot.composed - pivot.direct
    return pivot


def geometry_block(frame: pd.DataFrame) -> dict:
    span_safe = []
    label = []
    clean_ranks = []
    rank_match = []
    energy_error = []
    for row in frame.itertuples():
        overlaps = json.loads(row.overlap_summary_json)
        for by_condition in overlaps.values():
            span_safe.append(by_condition["meanJ_span_safe"])
            label.append(by_condition["meanJ_label_protected"])
        clean_ranks.append(
            json.loads(row.clean_rank_metadata_json)["min_rank"])
        matched = json.loads(row.matched_summary_json)
        for by_condition in matched.values():
            for summary in by_condition.values():
                rank_match.append(summary["rank_match_frac"])
                value = summary["energy_rel_err_max"]
                if value is not None:
                    energy_error.append(value)
    return {
        "n_alias_realizations": int(len(span_safe)),
        "span_safe_projector_overlap_max": float(max(
            row["projector_overlap_max"] for row in span_safe)),
        "span_safe_lost_rank_mean": float(np.mean([
            row["lost_rank_mean"] for row in span_safe])),
        "label_projector_overlap_mean": float(np.mean([
            row["projector_overlap_mean"] for row in label])),
        "clean_first_rank": {
            "median": float(np.median(clean_ranks)),
            "p90": float(np.quantile(clean_ranks, 0.9)),
            "max": int(max(clean_ranks)),
        },
        "matched_rank_match_min": float(min(rank_match)),
        "matched_energy_rel_error_max": float(max(energy_error)),
    }


def capability_block(g5: pd.DataFrame) -> dict:
    return {
        "overall": float(g5.capable_generation.mean()),
        "by_bank": {
            str(key): float(value)
            for key, value in g5.groupby(
                "bank").capable_generation.mean().items()
        },
        "by_bank_variant": {
            f"{bank}:{variant}": float(value)
            for (bank, variant), value in g5.groupby(
                ["bank", "variant"]).capable_generation.mean().items()
        },
        "fully_capable_direct_composed_facts_by_bank": {
            str(bank): int(count)
            for bank, count in (
                g5[g5.variant.isin(["direct", "composed"])]
                .groupby(["bank", "fact_id"])
                .capable_generation.all()
                .groupby(level=0).sum().items()
            )
        },
    }


def interval_points(blocks: dict, field: str) -> tuple[list, list, list]:
    estimates = []
    low = []
    high = []
    for block in blocks.values():
        summary = block["family_weighted"][field]
        estimates.append(summary["estimate"])
        low.append(summary["ci95"][0])
        high.append(summary["ci95"][1])
    return estimates, low, high


def make_figure(payload: dict, *, png_path: Path,
                pdf_path: Path) -> None:
    labels = ["F direct", "F composed", "S direct", "S composed"]
    keys = ["F:direct", "F:composed", "S:direct", "S:composed"]
    blocks = {key: payload["effects_by_bank_variant"][key]
              for key in keys}
    colors = {
        "J": "#9b2226",
        "control": "#3a6ea5",
        "specific": "#6a4c93",
        "F": "#ca6702",
        "S": "#0a9396",
    }
    figure, axes = plt.subplots(
        2, 2, figsize=(11.2, 7.8), constrained_layout=True)

    capability = payload["capability"]["by_bank_variant"]
    variants = ["direct", "composed", "bridge_supplied"]
    x = np.arange(len(variants))
    width = 0.36
    axes[0, 0].bar(
        x - width / 2,
        [capability[f"F:{variant}"] for variant in variants],
        width,
        color=colors["F"],
        label="Bank F",
    )
    axes[0, 0].bar(
        x + width / 2,
        [capability[f"S:{variant}"] for variant in variants],
        width,
        color=colors["S"],
        label="Bank S",
    )
    axes[0, 0].set_xticks(x, ["Direct", "Composed", "Bridge supplied"])
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Generation capability")
    axes[0, 0].set_title("a  Prospective G5 capability")
    axes[0, 0].legend(frameon=False, ncols=2)

    x = np.arange(len(labels))
    for offset, (field, label, color) in enumerate((
            ("J_effect", "Span-safe J", colors["J"]),
            ("control_effect", "Rank/energy control", colors["control"]))):
        estimate, low, high = interval_points(blocks, field)
        position = x + (-0.10 if offset == 0 else 0.10)
        axes[0, 1].errorbar(
            position,
            estimate,
            yerr=[
                np.asarray(estimate) - np.asarray(low),
                np.asarray(high) - np.asarray(estimate),
            ],
            fmt="o",
            capsize=3,
            color=color,
            label=label,
        )
    axes[0, 1].axhline(0, color="#555555", linewidth=0.8)
    axes[0, 1].set_xticks(x, labels, rotation=18, ha="right")
    axes[0, 1].set_ylabel("Δ answer log-probability (nats)")
    axes[0, 1].set_title("b  Family-weighted intervention effects")
    axes[0, 1].legend(frameon=False)

    estimate, low, high = interval_points(blocks, "specific")
    axes[1, 0].errorbar(
        x,
        estimate,
        yerr=[
            np.asarray(estimate) - np.asarray(low),
            np.asarray(high) - np.asarray(estimate),
        ],
        fmt="o",
        capsize=4,
        color=colors["specific"],
    )
    axes[1, 0].axhline(0, color="#555555", linewidth=0.8)
    axes[1, 0].set_xticks(x, labels, rotation=18, ha="right")
    axes[1, 0].set_ylabel("J − control effect (nats)")
    axes[1, 0].set_title("c  Span-safe J-specific effect")

    composition = payload["composition_by_bank"]
    for index, bank in enumerate(("F", "S")):
        summary = composition[bank]["family_weighted"]
        estimate = summary["estimate"]
        low, high = summary["ci95"]
        axes[1, 1].errorbar(
            index,
            estimate,
            yerr=[[estimate - low], [high - estimate]],
            fmt="o",
            capsize=4,
            color=colors[bank],
        )
    axes[1, 1].axhline(0, color="#555555", linewidth=0.8)
    axes[1, 1].set_xticks([0, 1], ["Bank F", "Bank S"])
    axes[1, 1].set_ylabel("Composed − direct specificity (nats)")
    axes[1, 1].set_title("d  Within-fact composition contrast")

    figure.suptitle(
        "OLMo-3 32B Think: Phase 4 development trajectory point",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.002,
        "Known development banks; family-resampling 95% intervals; "
        "not confirmatory evidence.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    grid_path = resolve_uri(config["grid_parquet_uri"])
    grid_result_path = resolve_uri(config["grid_result_uri"])
    grid_manifest_path = resolve_uri(config["grid_manifest_uri"])
    g5_path = resolve_uri(config["g5_parquet_uri"])
    grid_result = json.loads(grid_result_path.read_text())
    grid_manifest = json.loads(grid_manifest_path.read_text())
    if (
            grid_result["provenance"]["evidence_id"]
            != config["grid_evidence_id"]):
        raise RuntimeError("unexpected lineage-grid evidence")
    frame = pd.read_parquet(grid_path)
    effects = add_effects(frame)
    g5 = pd.read_parquet(g5_path)
    draws = int(config["bootstrap_draws"])
    seed = int(config["bootstrap_seed"])

    effects_by_bank_variant = {}
    ordered_cells = (
        ("F", "direct"),
        ("F", "composed"),
        ("S", "direct"),
        ("S", "composed"),
    )
    for offset, (bank, variant) in enumerate(ordered_cells):
        subset = effects[
            (effects.bank == bank) & (effects.variant == variant)]
        effects_by_bank_variant[f"{bank}:{variant}"] = effect_block(
            subset,
            draws=draws,
            seed=seed + 10 * offset,
        )
    compositions = composition_rows(effects)
    composition_by_bank = {}
    for offset, bank in enumerate(("F", "S")):
        subset = compositions[compositions.bank == bank]
        family_values = (
            subset.groupby("canonical_family").composition.mean()
            .to_numpy()
        )
        composition_by_bank[bank] = {
            "n_facts": int(len(subset)),
            "family_weighted": bootstrap(
                subset,
                "composition",
                draws=draws,
                seed=seed + 100 + offset,
            ),
            "exact_signflip_descriptive": exact_signflip(
                family_values),
        }

    overall_columns = {}
    for offset, column in enumerate((
            "J_effect",
            "control_effect",
            "specific",
            "label_effect",
            "protected_energy_effect",
            "label_specific",
            "mechanics_effect",
            "logit_effect",
    )):
        overall_columns[column] = bootstrap(
            effects,
            column,
            draws=draws,
            seed=seed + 200 + offset,
        )
    payload = {
        "schema_version": 1,
        "tier": config["tier"],
        "development_only": True,
        "claim_guard": (
            "Known Phase 3 banks and a development cohort; estimates "
            "localize trajectory patterns but cannot establish a binary "
            "lineage claim."),
        "effect_definition": {
            "J_effect": "lp_meanJ_span_safe - lp_baseline",
            "control_effect": "lp_ss_matched - lp_baseline",
            "specific": "J_effect - control_effect",
            "composition": "specific_composed - specific_direct",
        },
        "capability": capability_block(g5),
        "effects_by_bank_variant": effects_by_bank_variant,
        "composition_by_bank": composition_by_bank,
        "overall_family_weighted": overall_columns,
        "geometry_and_mechanics": geometry_block(frame),
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "unit": "canonical_family",
            "method": "family-resampling-percentile",
        },
    }

    output_dir = (
        metrics_dir(config["slug"])
        / "lineage_analysis"
        / config["evidence_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = grid_manifest["payload"]
    upstream = {
        str(config["grid_parquet_uri"]): file_sha256(grid_path),
        str(config["grid_result_uri"]): file_sha256(grid_result_path),
        str(config["grid_manifest_uri"]): file_sha256(
            grid_manifest_path),
        str(config["g5_parquet_uri"]): file_sha256(g5_path),
    }
    input_manifest = InputManifest(
        experiment_id=config["evidence_id"],
        config_sha256=file_sha256(config_path),
        model_id=source_manifest["model_id"],
        model_revision=source_manifest["model_revision"],
        tokenizer_manifest_sha256=source_manifest[
            "tokenizer_manifest_sha256"],
        lens_sha256=source_manifest["lens_sha256"],
        bank_sha256=source_manifest["bank_sha256"],
        partition_sha256=source_manifest["partition_sha256"],
        scoring_spec_sha256=source_manifest[
            "scoring_spec_sha256"],
        upstream=upstream,
        code_commit=clean["code_commit"],
    )
    manifest_path = output_dir / "input_manifest.json"
    atomic_json(manifest_path, input_manifest.envelope())
    result_path = output_dir / (
        f"lineage_analysis_{config['slug']}.json")
    figure_stem = (
        figures_dir() / "p4f01_olmo3_think_development")
    png_path = figure_stem.with_suffix(".png")
    pdf_path = figure_stem.with_suffix(".pdf")
    make_figure(payload, png_path=png_path, pdf_path=pdf_path)
    command = (
        "python -m jspace_phase4.experiments.p4_lineage_analysis "
        f"--config {arguments.config}")
    inputs = {
        "input_manifest": file_sha256(manifest_path),
        **upstream,
    }
    write_result4(
        payload,
        result_path,
        Provenance4(
            evidence_id=config["evidence_id"],
            tier=config["tier"],
            command=command,
            inputs=inputs,
            input_manifest_sha256=input_manifest.sha256(),
            model={
                "model_id": source_manifest["model_id"],
                "revision": source_manifest["model_revision"],
            },
            seed_contract=SEED_CONTRACT,
        ),
    )
    bank_s_direct = effects_by_bank_variant[
        "S:direct"]["family_weighted"]["specific"]
    bank_s_composition = composition_by_bank[
        "S"]["family_weighted"]
    create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "Phase 4 OLMo-3 Think development analysis: Bank S direct "
            f"J-specific {bank_s_direct['estimate']:.4f}; Bank S "
            "composed-minus-direct "
            f"{bank_s_composition['estimate']:.4f}; known-bank "
            "development evidence only."),
        command=command,
        outputs=[
            result_path,
            manifest_path,
            png_path,
            pdf_path,
        ],
        inputs=inputs,
    )
    print(json.dumps({
        "bank_s_direct_specific": bank_s_direct,
        "bank_s_composition": bank_s_composition,
        "geometry_and_mechanics": payload[
            "geometry_and_mechanics"],
        "figure": str(png_path),
    }, indent=1))


if __name__ == "__main__":
    main()
