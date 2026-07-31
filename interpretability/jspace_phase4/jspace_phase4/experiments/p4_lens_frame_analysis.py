"""Paired own-lens versus common-lens OLMo development analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .p4_lineage_analysis import add_effects, geometry_block
from ..manifests import (
    InputManifest,
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import figures_dir, metrics_dir, resolve_uri
from ..provenance4 import Provenance4, write_result4
from ..registry4 import create, resolve
from ..seeds import SEED_CONTRACT
from ..stats4 import exact_signflip, family_bootstrap_percentile

PAIR_KEYS = [
    "item_id",
    "bank",
    "fact_id",
    "canonical_family",
    "variant",
]
EFFECT_COLUMNS = [
    "J_effect",
    "control_effect",
    "specific",
    "label_effect",
    "protected_energy_effect",
    "label_specific",
    "mechanics_effect",
    "logit_effect",
]
CELLS = [
    ("F", "direct"),
    ("F", "composed"),
    ("S", "direct"),
    ("S", "composed"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def validate_envelope(path: Path) -> dict:
    envelope = json.loads(path.read_text())
    if object_sha256(envelope["payload"]) != envelope["payload_sha256"]:
        raise RuntimeError(f"payload hash mismatch: {path}")
    return envelope


def result_seed_namespace(result: dict) -> str:
    """Resolve the scientific RNG namespace, including older envelopes."""
    namespace = result.get("payload", {}).get(
        "scientific_seed_namespace")
    if namespace is None:
        namespace = result.get("provenance", {}).get("evidence_id")
    namespace = str(namespace or "").strip()
    if not namespace:
        raise ValueError("grid result lacks a scientific seed namespace")
    return namespace


def validate_shared_seed_namespace(
        own_result: dict,
        common_result: dict,
        own: pd.DataFrame,
        common: pd.DataFrame,
) -> str:
    """Require lens-frame comparisons to share every scientific RNG stream."""
    namespaces = {
        "own": result_seed_namespace(own_result),
        "common": result_seed_namespace(common_result),
    }
    for name, frame in (("own", own), ("common", common)):
        if "scientific_seed_namespace" not in frame:
            # Early Phase 4 grids predate the redundant per-row field.
            # Their producer used the evidence ID as the namespace, which
            # result_seed_namespace recovers from provenance.
            continue
        observed = {
            str(value).strip()
            for value in frame.scientific_seed_namespace.unique()
        }
        if observed != {namespaces[name]}:
            raise ValueError(
                f"{name} grid/result scientific seed namespace "
                f"mismatch: table={sorted(observed)!r}, "
                f"result={namespaces[name]!r}")
    if namespaces["own"] != namespaces["common"]:
        raise ValueError(
            "own/common scientific seed namespace mismatch: "
            f"{namespaces['own']!r} != {namespaces['common']!r}; "
            "paired frame inference would mix lens change with RNG change")
    return namespaces["own"]


def pair_effects(
        own: pd.DataFrame,
        common: pd.DataFrame,
        *,
        baseline_tolerance: float,
) -> pd.DataFrame:
    own_effects = add_effects(own)
    common_effects = add_effects(common)
    required = set(PAIR_KEYS + ["lp_baseline"] + EFFECT_COLUMNS)
    for name, frame in (("own", own_effects), ("common", common_effects)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} grid lacks {sorted(missing)}")
        if frame.duplicated(PAIR_KEYS).any():
            raise ValueError(f"{name} grid has duplicate paired keys")
    columns = PAIR_KEYS + ["lp_baseline"] + EFFECT_COLUMNS
    paired = own_effects[columns].merge(
        common_effects[columns],
        on=PAIR_KEYS,
        how="outer",
        suffixes=("_own", "_common"),
        validate="one_to_one",
        indicator=True,
    )
    if not bool((paired["_merge"] == "both").all()):
        counts = paired["_merge"].value_counts().to_dict()
        raise ValueError(f"own/common cohort mismatch: {counts}")
    paired = paired.drop(columns="_merge")
    baseline_delta = np.abs(
        paired.lp_baseline_own - paired.lp_baseline_common)
    if not np.isfinite(baseline_delta).all():
        raise ValueError("non-finite paired baseline")
    if float(baseline_delta.max()) > baseline_tolerance:
        raise ValueError(
            "own/common baseline drift exceeds tolerance: "
            f"{float(baseline_delta.max()):.9g} > "
            f"{baseline_tolerance:.9g}")
    for column in EFFECT_COLUMNS:
        paired[f"{column}_delta"] = (
            paired[f"{column}_common"]
            - paired[f"{column}_own"]
        )
    return paired


def composition_frame(paired: pd.DataFrame) -> pd.DataFrame:
    index = ["bank", "fact_id", "canonical_family"]
    values = ["specific_own", "specific_common", "specific_delta"]
    pivot = paired.pivot(
        index=index,
        columns="variant",
        values=values,
    )
    variants = set(pivot.columns.get_level_values("variant"))
    if variants != {"direct", "composed"}:
        raise ValueError(
            f"paired grid needs direct/composed only, got {variants}")
    result = pivot.reset_index()
    result.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple) else str(column)
        for column in result.columns
    ]
    for frame in ("own", "common", "delta"):
        result[f"composition_{frame}"] = (
            result[f"specific_{frame}_composed"]
            - result[f"specific_{frame}_direct"]
        )
    return result


def bootstrap(
        rows: pd.DataFrame,
        column: str,
        *,
        draws: int,
        seed: int,
) -> dict:
    return family_bootstrap_percentile(
        rows,
        column,
        draws=draws,
        seed=seed,
    )


def safe_correlation(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 3 or left.std() == 0 or right.std() == 0:
        return None
    value = float(left.corr(right))
    return value if np.isfinite(value) else None


def effect_cell(
        rows: pd.DataFrame,
        *,
        draws: int,
        seed: int,
) -> dict:
    metrics = {}
    for metric_offset, metric in enumerate(
            ("J_effect", "control_effect", "specific")):
        columns = {
            "own": f"{metric}_own",
            "common": f"{metric}_common",
            "common_minus_own": f"{metric}_delta",
        }
        metrics[metric] = {
            frame: bootstrap(
                rows,
                column,
                draws=draws,
                seed=seed + 10 * metric_offset + frame_offset,
            )
            for frame_offset, (frame, column) in enumerate(
                columns.items())
        }
    family_delta = (
        rows.groupby("canonical_family").specific_delta.mean().to_numpy())
    return {
        "n_items": int(len(rows)),
        "n_facts": int(rows.fact_id.nunique()),
        "n_families": int(rows.canonical_family.nunique()),
        "metrics": metrics,
        "specific_common_minus_own_exact_signflip_descriptive":
            exact_signflip(family_delta),
        "specific_agreement": {
            "item_pearson": safe_correlation(
                rows.specific_own, rows.specific_common),
            "item_sign_agreement": float(
                (np.sign(rows.specific_own)
                 == np.sign(rows.specific_common)).mean()),
            "item_mean_absolute_difference": float(
                rows.specific_delta.abs().mean()),
        },
    }


def composition_block(
        rows: pd.DataFrame,
        *,
        draws: int,
        seed: int,
) -> dict:
    summaries = {
        frame: bootstrap(
            rows,
            f"composition_{frame}",
            draws=draws,
            seed=seed + offset,
        )
        for offset, frame in enumerate(("own", "common", "delta"))
    }
    family_delta = (
        rows.groupby("canonical_family").composition_delta.mean()
        .to_numpy()
    )
    return {
        "n_facts": int(len(rows)),
        "n_families": int(rows.canonical_family.nunique()),
        "own": summaries["own"],
        "common": summaries["common"],
        "common_minus_own": summaries["delta"],
        "common_minus_own_exact_signflip_descriptive":
            exact_signflip(family_delta),
    }


def interval(
        block: dict,
        metric: str,
        frame: str,
) -> tuple[float, float, float]:
    summary = block["metrics"][metric][frame]
    return (
        summary["estimate"],
        summary["ci95"][0],
        summary["ci95"][1],
    )


def draw_interval(
        axis,
        x: float,
        summary: dict,
        *,
        color: str,
        label: str | None = None,
        marker: str = "o",
) -> None:
    estimate = summary["estimate"]
    low, high = summary["ci95"]
    axis.errorbar(
        x,
        estimate,
        yerr=[[estimate - low], [high - estimate]],
        fmt=marker,
        capsize=3,
        color=color,
        label=label,
    )


def make_figure(
        payload: dict,
        paired: pd.DataFrame,
        *,
        display_name: str,
        png_path: Path,
        pdf_path: Path,
) -> None:
    labels = ["F direct", "F composed", "S direct", "S composed"]
    keys = ["F:direct", "F:composed", "S:direct", "S:composed"]
    cells = payload["effects_by_bank_variant"]
    colors = {
        "own": "#9b2226",
        "common": "#3a6ea5",
        "delta": "#6a4c93",
        "F": "#ca6702",
        "S": "#0a9396",
    }
    figure, axes = plt.subplots(
        2, 2, figsize=(11.4, 8.0), constrained_layout=True)
    x = np.arange(len(keys))

    for index, key in enumerate(keys):
        for offset, frame in ((-0.10, "own"), (0.10, "common")):
            summary = cells[key]["metrics"]["specific"][frame]
            draw_interval(
                axes[0, 0],
                index + offset,
                summary,
                color=colors[frame],
                label=(
                    "Own lens" if index == 0 and frame == "own"
                    else "Common base lens"
                    if index == 0 else None
                ),
            )
    axes[0, 0].axhline(0, color="#555555", linewidth=0.8)
    axes[0, 0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0, 0].set_ylabel("J − control effect (nats)")
    axes[0, 0].set_title("a  Specificity in two coordinate frames")
    axes[0, 0].legend(frameon=False)

    for index, key in enumerate(keys):
        summary = cells[key]["metrics"]["specific"][
            "common_minus_own"]
        draw_interval(
            axes[0, 1],
            index,
            summary,
            color=colors["delta"],
        )
    axes[0, 1].axhline(0, color="#555555", linewidth=0.8)
    axes[0, 1].set_xticks(x, labels, rotation=18, ha="right")
    axes[0, 1].set_ylabel("Common − own specificity (nats)")
    axes[0, 1].set_title("b  Paired coordinate-frame contrast")

    for index, bank in enumerate(("F", "S")):
        block = payload["composition_by_bank"][bank]
        for offset, frame in ((-0.10, "own"), (0.10, "common")):
            draw_interval(
                axes[1, 0],
                index + offset,
                block[frame],
                color=colors[frame],
            )
    axes[1, 0].axhline(0, color="#555555", linewidth=0.8)
    axes[1, 0].set_xticks([0, 1], ["Bank F", "Bank S"])
    axes[1, 0].set_ylabel(
        "Composed − direct specificity (nats)")
    axes[1, 0].set_title("c  Within-fact composition contrast")

    family = (
        paired.groupby(
            ["bank", "variant", "canonical_family"],
            as_index=False,
        )[["specific_own", "specific_common"]].mean()
    )
    markers = {"direct": "o", "composed": "s"}
    for bank in ("F", "S"):
        for variant in ("direct", "composed"):
            subset = family[
                (family.bank == bank) & (family.variant == variant)]
            axes[1, 1].scatter(
                subset.specific_own,
                subset.specific_common,
                s=30,
                alpha=0.78,
                color=colors[bank],
                marker=markers[variant],
                label=f"Bank {bank} {variant}",
            )
    values = np.concatenate([
        family.specific_own.to_numpy(),
        family.specific_common.to_numpy(),
    ])
    low, high = np.quantile(values, [0.01, 0.99])
    padding = max((high - low) * 0.08, 0.02)
    axes[1, 1].plot(
        [low - padding, high + padding],
        [low - padding, high + padding],
        color="#555555",
        linewidth=0.8,
        linestyle="--",
    )
    axes[1, 1].set_xlim(low - padding, high + padding)
    axes[1, 1].set_ylim(low - padding, high + padding)
    axes[1, 1].set_xlabel("Own-lens family mean (nats)")
    axes[1, 1].set_ylabel("Common-lens family mean (nats)")
    axes[1, 1].set_title("d  Family-level frame agreement")
    axes[1, 1].legend(frameon=False, fontsize=8, ncols=2)

    figure.suptitle(
        f"{display_name}: own-lens versus common-base-lens view",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.002,
        f"Same {payload['pairing']['n_facts']} paired facts; "
        "known development banks; "
        "family-resampling 95% intervals; not confirmatory evidence.",
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
    uri_keys = (
        "own_grid_parquet_uri",
        "own_grid_result_uri",
        "own_grid_manifest_uri",
        "common_grid_parquet_uri",
        "common_grid_result_uri",
        "common_grid_manifest_uri",
    )
    paths = {key: resolve_uri(config[key]) for key in uri_keys}
    own_result = validate_envelope(paths["own_grid_result_uri"])
    common_result = validate_envelope(paths["common_grid_result_uri"])
    own_manifest = validate_envelope(paths["own_grid_manifest_uri"])
    common_manifest = validate_envelope(paths["common_grid_manifest_uri"])
    evidence_pairs = (
        (own_result, config["own_grid_evidence_id"]),
        (common_result, config["common_grid_evidence_id"]),
    )
    for result, expected in evidence_pairs:
        if result["provenance"]["evidence_id"] != expected:
            raise RuntimeError(f"unexpected grid evidence for {expected}")
        if not resolve(expected)["live"]:
            raise RuntimeError(f"grid evidence is not live: {expected}")
    own_source = own_manifest["payload"]
    common_source = common_manifest["payload"]
    invariant_fields = (
        "model_id",
        "model_revision",
        "tokenizer_manifest_sha256",
        "bank_sha256",
        "partition_sha256",
        "scoring_spec_sha256",
    )
    mismatches = {
        field: {
            "own": own_source[field],
            "common": common_source[field],
        }
        for field in invariant_fields
        if own_source[field] != common_source[field]
    }
    if mismatches:
        raise RuntimeError(
            "own/common input-manifest mismatch: "
            + json.dumps(mismatches, sort_keys=True))

    own = pd.read_parquet(paths["own_grid_parquet_uri"])
    common = pd.read_parquet(paths["common_grid_parquet_uri"])
    shared_seed_namespace = validate_shared_seed_namespace(
        own_result,
        common_result,
        own,
        common,
    )
    baseline_tolerance = float(config["baseline_tolerance"])
    paired = pair_effects(
        own,
        common,
        baseline_tolerance=baseline_tolerance,
    )
    draws = int(config["bootstrap_draws"])
    seed = int(config["bootstrap_seed"])
    effects_by_bank_variant = {}
    for offset, (bank, variant) in enumerate(CELLS):
        subset = paired[
            (paired.bank == bank) & (paired.variant == variant)]
        effects_by_bank_variant[f"{bank}:{variant}"] = effect_cell(
            subset,
            draws=draws,
            seed=seed + 100 * offset,
        )
    compositions = composition_frame(paired)
    composition_by_bank = {}
    for offset, bank in enumerate(("F", "S")):
        composition_by_bank[bank] = composition_block(
            compositions[compositions.bank == bank],
            draws=draws,
            seed=seed + 1000 + 100 * offset,
        )
    baseline_delta = np.abs(
        paired.lp_baseline_own - paired.lp_baseline_common)
    family_agreement = (
        paired.groupby(
            ["bank", "variant", "canonical_family"],
            as_index=False,
        )[["specific_own", "specific_common"]].mean()
    )
    payload = {
        "schema_version": 1,
        "tier": config["tier"],
        "development_only": True,
        "claim_guard": (
            "The paired grids use known Phase 3 banks and a development "
            "cohort. Frame agreement or disagreement is localization "
            "evidence, not a binary lineage claim."),
        "inputs": {
            "own_grid_evidence_id": config["own_grid_evidence_id"],
            "common_grid_evidence_id":
                config["common_grid_evidence_id"],
            "own_lens_sha256": own_source["lens_sha256"],
            "common_lens_sha256": common_source["lens_sha256"],
            "shared_scientific_seed_namespace":
                shared_seed_namespace,
        },
        "pairing": {
            "n_items": int(len(paired)),
            "n_facts": int(paired.fact_id.nunique()),
            "n_families": int(
                paired.canonical_family.nunique()),
            "baseline_tolerance": baseline_tolerance,
            "baseline_max_absolute_drift":
                float(baseline_delta.max()),
            "pair_keys": PAIR_KEYS,
        },
        "effect_definition": {
            "specific": (
                "(lp_meanJ_span_safe - lp_baseline) - "
                "(lp_ss_matched - lp_baseline)"
            ),
            "frame_delta": "specific_common - specific_own",
            "composition": (
                "specific_composed - specific_direct"),
        },
        "effects_by_bank_variant": effects_by_bank_variant,
        "composition_by_bank": composition_by_bank,
        "specific_frame_agreement": {
            "item_pearson": safe_correlation(
                paired.specific_own, paired.specific_common),
            "family_pearson": safe_correlation(
                family_agreement.specific_own,
                family_agreement.specific_common,
            ),
            "item_mean_absolute_difference": float(
                paired.specific_delta.abs().mean()),
            "item_max_absolute_difference": float(
                paired.specific_delta.abs().max()),
        },
        "geometry": {
            "own": geometry_block(own),
            "common": geometry_block(common),
        },
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "unit": "canonical_family",
            "method": "family-resampling-percentile",
        },
    }

    output_dir = (
        metrics_dir(config["slug"])
        / "lens_frame_analysis"
        / config["evidence_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream = {
        str(config[key]): file_sha256(path)
        for key, path in paths.items()
    }
    combined_lens_sha256 = object_sha256({
        "own": own_source["lens_sha256"],
        "common": common_source["lens_sha256"],
    })
    input_manifest = InputManifest(
        experiment_id=config["evidence_id"],
        config_sha256=file_sha256(config_path),
        model_id=own_source["model_id"],
        model_revision=own_source["model_revision"],
        tokenizer_manifest_sha256=own_source[
            "tokenizer_manifest_sha256"],
        lens_sha256=combined_lens_sha256,
        bank_sha256=own_source["bank_sha256"],
        partition_sha256=own_source["partition_sha256"],
        scoring_spec_sha256=own_source["scoring_spec_sha256"],
        upstream=upstream,
        code_commit=clean["code_commit"],
    )
    manifest_path = output_dir / "input_manifest.json"
    atomic_json(manifest_path, input_manifest.envelope())
    result_path = output_dir / (
        f"lens_frame_analysis_{config['slug']}.json")
    figure_stem = figures_dir() / config["figure_stem"]
    png_path = figure_stem.with_suffix(".png")
    pdf_path = figure_stem.with_suffix(".pdf")
    display_name = str(config.get(
        "display_name", "OLMo-3 32B Think"))
    make_figure(
        payload,
        paired,
        display_name=display_name,
        png_path=png_path,
        pdf_path=pdf_path,
    )
    command = (
        "python -m jspace_phase4.experiments.p4_lens_frame_analysis "
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
                "model_id": own_source["model_id"],
                "revision": own_source["model_revision"],
            },
            seed_contract=SEED_CONTRACT,
        ),
    )
    f_direct_delta = effects_by_bank_variant[
        "F:direct"]["metrics"]["specific"]["common_minus_own"]
    f_composed_delta = effects_by_bank_variant[
        "F:composed"]["metrics"]["specific"]["common_minus_own"]
    s_direct_common = effects_by_bank_variant[
        "S:direct"]["metrics"]["specific"]["common"]
    create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            f"Paired {display_name} own/common-lens development "
            "analysis: "
            f"Bank F common-minus-own specificity "
            f"{f_direct_delta['estimate']:.4f} direct and "
            f"{f_composed_delta['estimate']:.4f} composed; Bank S "
            f"common-lens direct {s_direct_common['estimate']:.4f}; "
            "known-bank development evidence only."),
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
        "pairing": payload["pairing"],
        "bank_f_direct_common_minus_own": f_direct_delta,
        "bank_f_composed_common_minus_own": f_composed_delta,
        "bank_s_direct_common": s_direct_common,
        "specific_frame_agreement":
            payload["specific_frame_agreement"],
        "figure": str(png_path),
    }, indent=1))


if __name__ == "__main__":
    main()
