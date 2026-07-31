"""Registered four-checkpoint OLMo development-trajectory synthesis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .p4_lineage_analysis import add_effects, composition_rows
from .p4_lens_frame_analysis import (
    pair_effects,
    validate_envelope,
    validate_shared_seed_namespace,
)
from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import figures_dir, metrics_dir, resolve_uri
from ..provenance4 import Provenance4, write_result4
from ..registry4 import create, resolve
from ..seeds import SEED_CONTRACT
from ..stats4 import family_bootstrap_percentile

CELLS = (
    ("F", "direct"),
    ("F", "composed"),
    ("S", "direct"),
    ("S", "composed"),
)
COMPOSITION_BANKS = ("F", "S")
FRAMES = ("own", "common")
PANEL_METRICS = (
    ("F:direct", "Bank F direct"),
    ("F:composed", "Bank F composed"),
    ("S:direct", "Bank S direct"),
    ("S:composed", "Bank S composed"),
    ("F:composition", "Bank F composed − direct"),
    ("S:composition", "Bank S composed − direct"),
)
FIGURE_LAYOUT_RECT = (0.0, 0.075, 1.0, 0.93)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _bootstrap(
        rows: pd.DataFrame,
        column: str,
        *,
        draws: int,
        seed: int,
) -> dict:
    summary = family_bootstrap_percentile(
        rows,
        column,
        draws=draws,
        seed=seed,
    )
    return {**summary, "bootstrap_seed": int(seed)}


def independent_family_bootstrap(
        rows: pd.DataFrame,
        column: str,
        *,
        draws: int,
        seed: int,
) -> dict:
    """Reconstruct a family bootstrap without the production helper."""
    means = (
        rows.groupby("canonical_family", sort=True)[column]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(means) < 3:
        raise ValueError("family bootstrap requires at least 3 families")
    if not np.isfinite(means).all():
        raise ValueError("family bootstrap received non-finite means")
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(
        0,
        len(means),
        size=(draws, len(means)),
    )
    distribution = np.take(means, indices).mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "estimate": float(means.mean()),
        "ci95": [float(low), float(high)],
        "n_families": int(len(means)),
        "n_bootstrap": int(draws),
        "method": "family-resampling-percentile",
        "distribution_sha256": hashlib.sha256(
            np.asarray(distribution, dtype="<f8").tobytes()
        ).hexdigest(),
        "bootstrap_seed": int(seed),
    }


def _summary_row(
        *,
        checkpoint: dict,
        checkpoint_index: int,
        frame: str,
        metric_key: str,
        bank: str,
        variant: str,
        summary: dict,
        n_items: int,
        n_facts: int,
        source_grid_evidence_id: str,
) -> dict:
    return {
        "checkpoint_key": checkpoint["key"],
        "checkpoint_label": checkpoint["label"],
        "checkpoint_index": int(checkpoint_index),
        "lineage_role": checkpoint["lineage_role"],
        "frame": frame,
        "metric_key": metric_key,
        "bank": bank,
        "variant": variant,
        "estimate": summary["estimate"],
        "ci95_low": summary["ci95"][0],
        "ci95_high": summary["ci95"][1],
        "n_families": summary["n_families"],
        "n_bootstrap": summary["n_bootstrap"],
        "bootstrap_seed": summary["bootstrap_seed"],
        "distribution_sha256": summary["distribution_sha256"],
        "n_items": int(n_items),
        "n_facts": int(n_facts),
        "source_grid_evidence_id": source_grid_evidence_id,
        "common_is_own": bool(checkpoint.get("common_is_own", False)),
    }


def summarize_checkpoint(
        own: pd.DataFrame,
        common: pd.DataFrame,
        *,
        checkpoint: dict,
        checkpoint_index: int,
        draws: int,
        seed: int,
) -> tuple[list[dict], dict[str, tuple[pd.DataFrame, str]]]:
    """Create the twelve own/common trajectory rows for one checkpoint."""
    common_is_own = bool(checkpoint.get("common_is_own", False))
    frames = {
        "own": add_effects(own),
        "common": add_effects(own if common_is_own else common),
    }
    source_ids = {
        "own": checkpoint["own_grid_evidence_id"],
        "common": (
            checkpoint["own_grid_evidence_id"]
            if common_is_own
            else checkpoint["common_grid_evidence_id"]
        ),
    }
    output: list[dict] = []
    validation: dict[str, tuple[pd.DataFrame, str]] = {}

    for cell_offset, (bank, variant) in enumerate(CELLS):
        metric_key = f"{bank}:{variant}"
        metric_seed = seed + cell_offset
        own_subset = frames["own"][
            (frames["own"].bank == bank)
            & (frames["own"].variant == variant)
        ]
        if own_subset.empty:
            raise ValueError(
                f"{checkpoint['key']} lacks {metric_key} rows")
        own_summary = _bootstrap(
            own_subset,
            "specific",
            draws=draws,
            seed=metric_seed,
        )
        for frame in FRAMES:
            subset = frames[frame][
                (frames[frame].bank == bank)
                & (frames[frame].variant == variant)
            ]
            if len(subset) != len(own_subset):
                raise ValueError(
                    f"{checkpoint['key']} frame cohort size mismatch "
                    f"for {metric_key}")
            summary = (
                own_summary
                if common_is_own and frame == "common"
                else _bootstrap(
                    subset,
                    "specific",
                    draws=draws,
                    seed=metric_seed,
                )
            )
            key = f"{checkpoint['key']}:{frame}:{metric_key}"
            output.append(_summary_row(
                checkpoint=checkpoint,
                checkpoint_index=checkpoint_index,
                frame=frame,
                metric_key=metric_key,
                bank=bank,
                variant=variant,
                summary=summary,
                n_items=len(subset),
                n_facts=subset.fact_id.nunique(),
                source_grid_evidence_id=source_ids[frame],
            ))
            validation[key] = (subset, "specific")

    compositions = {
        frame: composition_rows(frame_rows)
        for frame, frame_rows in frames.items()
    }
    for bank_offset, bank in enumerate(COMPOSITION_BANKS):
        metric_key = f"{bank}:composition"
        metric_seed = seed + len(CELLS) + bank_offset
        own_subset = compositions["own"][
            compositions["own"].bank == bank
        ]
        own_summary = _bootstrap(
            own_subset,
            "composition",
            draws=draws,
            seed=metric_seed,
        )
        for frame in FRAMES:
            subset = compositions[frame][
                compositions[frame].bank == bank
            ]
            summary = (
                own_summary
                if common_is_own and frame == "common"
                else _bootstrap(
                    subset,
                    "composition",
                    draws=draws,
                    seed=metric_seed,
                )
            )
            key = f"{checkpoint['key']}:{frame}:{metric_key}"
            output.append(_summary_row(
                checkpoint=checkpoint,
                checkpoint_index=checkpoint_index,
                frame=frame,
                metric_key=metric_key,
                bank=bank,
                variant="composed_minus_direct",
                summary=summary,
                n_items=0,
                n_facts=len(subset),
                source_grid_evidence_id=source_ids[frame],
            ))
            validation[key] = (subset, "composition")
    return output, validation


def independently_validate_rows(
        rows: list[dict],
        validation: dict[str, tuple[pd.DataFrame, str]],
) -> dict:
    """Require an exact second reconstruction of every summary."""
    reconstructed_hashes = []
    for row in rows:
        key = (
            f"{row['checkpoint_key']}:{row['frame']}:"
            f"{row['metric_key']}"
        )
        source, column = validation[key]
        reconstructed = independent_family_bootstrap(
            source,
            column,
            draws=int(row["n_bootstrap"]),
            seed=int(row["bootstrap_seed"]),
        )
        expected = {
            field: row[field]
            for field in (
                "estimate",
                "n_families",
                "n_bootstrap",
                "bootstrap_seed",
                "distribution_sha256",
            )
        }
        expected["ci95"] = [row["ci95_low"], row["ci95_high"]]
        expected["method"] = "family-resampling-percentile"
        if reconstructed != expected:
            raise RuntimeError(
                f"independent trajectory bootstrap mismatch: {key}")
        reconstructed_hashes.append(
            reconstructed["distribution_sha256"])
    return {
        "all_exact": True,
        "n_summaries_reconstructed": len(rows),
        "n_unique_distributions": len(set(reconstructed_hashes)),
        "distribution_hash_set_sha256": object_sha256(
            sorted(set(reconstructed_hashes))
        ),
        "method": (
            "independent direct NumPy PCG64 family-resampling "
            "reconstruction"
        ),
    }


def validate_shared_contract(sources: list[dict]) -> dict:
    """Require the same banks and scoring contract across checkpoints."""
    required_equal = ("bank_sha256", "scoring_spec_sha256")
    result = {}
    for field in required_equal:
        values = {str(source[field]) for source in sources}
        if len(values) != 1:
            raise RuntimeError(
                f"cross-checkpoint {field} mismatch: {sorted(values)}")
        result[field] = next(iter(values))
    return result


def _check_condition_order(
        own: pd.DataFrame,
        common: pd.DataFrame,
        *,
        checkpoint_key: str,
) -> None:
    if "condition_order_json" not in own or "condition_order_json" not in common:
        raise ValueError("paired grids lack condition_order_json")
    columns = ["item_id", "condition_order_json"]
    paired = own[columns].merge(
        common[columns],
        on="item_id",
        how="outer",
        suffixes=("_own", "_common"),
        validate="one_to_one",
        indicator=True,
    )
    if not bool((paired["_merge"] == "both").all()):
        raise ValueError(f"{checkpoint_key} condition-order cohort mismatch")
    if not bool((
            paired.condition_order_json_own
            == paired.condition_order_json_common
    ).all()):
        raise ValueError(f"{checkpoint_key} condition orders differ")


def load_checkpoint(
        checkpoint: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Load and validate one own/common grid pair."""
    own_paths = {
        field: resolve_uri(checkpoint[f"own_grid_{field}_uri"])
        for field in ("parquet", "result", "manifest")
    }
    own_result = validate_envelope(own_paths["result"])
    own_manifest = validate_envelope(own_paths["manifest"])
    own_evidence_id = checkpoint["own_grid_evidence_id"]
    if own_result["provenance"]["evidence_id"] != own_evidence_id:
        raise RuntimeError(
            f"unexpected own-grid evidence for {checkpoint['key']}")
    if not resolve(own_evidence_id)["live"]:
        raise RuntimeError(f"own grid is not live: {own_evidence_id}")
    own = pd.read_parquet(own_paths["parquet"])

    common_is_own = bool(checkpoint.get("common_is_own", False))
    upstream = {
        str(checkpoint[f"own_grid_{field}_uri"]):
            file_sha256(path)
        for field, path in own_paths.items()
    }
    if common_is_own:
        return own, own, own_manifest["payload"], upstream

    common_paths = {
        field: resolve_uri(checkpoint[f"common_grid_{field}_uri"])
        for field in ("parquet", "result", "manifest")
    }
    common_result = validate_envelope(common_paths["result"])
    common_manifest = validate_envelope(common_paths["manifest"])
    common_evidence_id = checkpoint["common_grid_evidence_id"]
    if common_result["provenance"]["evidence_id"] != common_evidence_id:
        raise RuntimeError(
            f"unexpected common-grid evidence for {checkpoint['key']}")
    if not resolve(common_evidence_id)["live"]:
        raise RuntimeError(f"common grid is not live: {common_evidence_id}")
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
        field: {"own": own_source[field], "common": common_source[field]}
        for field in invariant_fields
        if own_source[field] != common_source[field]
    }
    if mismatches:
        raise RuntimeError(
            f"{checkpoint['key']} frame manifest mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    common = pd.read_parquet(common_paths["parquet"])
    validate_shared_seed_namespace(
        own_result,
        common_result,
        own,
        common,
    )
    pair_effects(
        own,
        common,
        baseline_tolerance=float(checkpoint["baseline_tolerance"]),
    )
    _check_condition_order(
        own,
        common,
        checkpoint_key=checkpoint["key"],
    )
    upstream.update({
        str(checkpoint[f"common_grid_{field}_uri"]):
            file_sha256(path)
        for field, path in common_paths.items()
    })
    return own, common, own_source, upstream


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def make_figure(
        table: pd.DataFrame,
        *,
        display_name: str,
        png_path: Path,
        pdf_path: Path,
) -> None:
    colors = {"own": "#9b2226", "common": "#3a6ea5"}
    labels = (
        table[["checkpoint_index", "checkpoint_label"]]
        .drop_duplicates()
        .sort_values("checkpoint_index")
        .checkpoint_label.tolist()
    )
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.2, 8.8),
        constrained_layout=True,
    )
    figure.get_layout_engine().set(rect=FIGURE_LAYOUT_RECT)
    for panel_index, (metric_key, title) in enumerate(PANEL_METRICS):
        axis = axes.flat[panel_index]
        for frame in FRAMES:
            subset = table[
                (table.metric_key == metric_key)
                & (table.frame == frame)
            ].sort_values("checkpoint_index")
            if len(subset) != 4:
                raise RuntimeError(
                    f"trajectory figure lacks four {frame} {metric_key} "
                    "points"
                )
            offset = -0.055 if frame == "own" else 0.055
            x = subset.checkpoint_index.to_numpy(dtype=float) + offset
            estimate = subset.estimate.to_numpy(dtype=float)
            low = subset.ci95_low.to_numpy(dtype=float)
            high = subset.ci95_high.to_numpy(dtype=float)
            # The two 3.1 endpoints are siblings. Connect only the
            # Base -> 3.0 Think -> 3.1 Think path.
            axis.errorbar(
                x[:3],
                estimate[:3],
                yerr=[estimate[:3] - low[:3], high[:3] - estimate[:3]],
                fmt="o-",
                capsize=3,
                linewidth=1.4,
                color=colors[frame],
                label=(
                    "Own lens"
                    if frame == "own"
                    else "Common base lens"
                ),
            )
            axis.errorbar(
                x[3],
                estimate[3],
                yerr=[
                    [estimate[3] - low[3]],
                    [high[3] - estimate[3]],
                ],
                fmt="s",
                capsize=3,
                markersize=6,
                color=colors[frame],
            )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.axvspan(2.55, 3.45, color="#667085", alpha=0.055)
        axis.set_xlim(-0.35, 3.35)
        axis.set_xticks(range(4), labels)
        axis.tick_params(axis="x", labelrotation=15)
        axis.set_ylabel("J − matched-control effect (nats)")
        axis.set_title(
            f"{chr(ord('a') + panel_index)}  {title}")
        if panel_index == 0:
            axis.legend(frameon=False, fontsize=9)
    figure.suptitle(display_name, fontsize=15)
    figure.text(
        0.5,
        0.028,
        "Lines connect Base → 3.0 Think → 3.1 Think; squares mark the "
        "3.1 Instruct sibling endpoint (not a temporal continuation).",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.text(
        0.5,
        0.009,
        "Known development banks; checkpoint-specific capability "
        "cohorts; family-resampling 95% intervals; cross-checkpoint "
        "differences are unpaired.",
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
    checkpoints = list(config["checkpoints"])
    if [row["key"] for row in checkpoints] != [
            "olmo3-base",
            "olmo3-think",
            "olmo31-think",
            "olmo31-instruct",
    ]:
        raise ValueError("trajectory checkpoint order is not frozen")

    draws = int(config["bootstrap_draws"])
    base_seed = int(config["bootstrap_seed"])
    rows: list[dict] = []
    validation = {}
    upstream = {}
    sources = []
    checkpoint_sources = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        own, common, source, checkpoint_upstream = load_checkpoint(
            checkpoint)
        checkpoint_rows, checkpoint_validation = summarize_checkpoint(
            own,
            common,
            checkpoint=checkpoint,
            checkpoint_index=checkpoint_index,
            draws=draws,
            seed=base_seed + 100 * checkpoint_index,
        )
        rows.extend(checkpoint_rows)
        validation.update(checkpoint_validation)
        upstream.update(checkpoint_upstream)
        sources.append(source)
        checkpoint_sources.append({
            "key": checkpoint["key"],
            "label": checkpoint["label"],
            "lineage_role": checkpoint["lineage_role"],
            "model_id": source["model_id"],
            "model_revision": source["model_revision"],
            "tokenizer_manifest_sha256":
                source["tokenizer_manifest_sha256"],
            "own_lens_sha256": source["lens_sha256"],
            "own_grid_evidence_id":
                checkpoint["own_grid_evidence_id"],
            "common_grid_evidence_id": (
                checkpoint["own_grid_evidence_id"]
                if checkpoint.get("common_is_own", False)
                else checkpoint["common_grid_evidence_id"]
            ),
            "common_is_own": bool(
                checkpoint.get("common_is_own", False)),
            "n_items": int(len(own)),
            "n_facts": int(own.fact_id.nunique()),
            "n_families": int(own.canonical_family.nunique()),
        })
    shared_contract = validate_shared_contract(sources)
    independent_validation = independently_validate_rows(
        rows,
        validation,
    )
    table = pd.DataFrame(rows).sort_values([
        "metric_key",
        "frame",
        "checkpoint_index",
    ]).reset_index(drop=True)
    payload = {
        "schema_version": 1,
        "tier": config["tier"],
        "development_only": True,
        "claim_guard": (
            "Known Phase 3 banks and checkpoint-specific prospective "
            "capability cohorts. Cross-checkpoint estimates are not "
            "paired causal contrasts and cannot establish a binary "
            "lineage claim."
        ),
        "checkpoint_relationship": {
            "think_path": [
                "olmo3-base",
                "olmo3-think",
                "olmo31-think",
            ],
            "sibling_endpoint": "olmo31-instruct",
            "plot_rule": (
                "Do not connect 3.1 Think to 3.1 Instruct as a temporal "
                "step."
            ),
        },
        "frames": {
            "own": "checkpoint-specific fitted lens",
            "common": "frozen OLMo-3 base lens",
            "base_common_is_own": True,
        },
        "effect_definition": {
            "specific": (
                "(lp_meanJ_span_safe - lp_baseline) - "
                "(lp_ss_matched - lp_baseline)"
            ),
            "composition": (
                "specific_composed - specific_direct"),
        },
        "checkpoints": checkpoint_sources,
        "shared_contract": shared_contract,
        "bootstrap": {
            "draws": draws,
            "base_seed": base_seed,
            "checkpoint_seed_stride": 100,
            "unit": "canonical_family",
            "method": "family-resampling-percentile",
            "paired_frame_seed_rule": (
                "own and common frames use the same per-checkpoint "
                "metric seed"
            ),
        },
        "independent_validation": independent_validation,
        "trajectory_rows": rows,
    }

    output_dir = (
        metrics_dir(config["slug"])
        / "trajectory_analysis"
        / config["evidence_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "schema_version": 1,
        "experiment_id": config["evidence_id"],
        "config_sha256": file_sha256(config_path),
        "analysis_family": "olmo-four-checkpoint-trajectory",
        "checkpoints": checkpoint_sources,
        "shared_contract": shared_contract,
        "upstream": upstream,
        "bootstrap": payload["bootstrap"],
        "code_commit": clean["code_commit"],
    }
    manifest_path = output_dir / "input_manifest.json"
    atomic_json(manifest_path, {
        "schema_version": 1,
        "payload": manifest_payload,
        "payload_sha256": object_sha256(manifest_payload),
    })
    result_path = output_dir / (
        f"trajectory_analysis_{config['slug']}.json")
    table_path = output_dir / (
        f"trajectory_table_{config['slug']}.csv")
    atomic_csv(table_path, table)
    figure_stem = figures_dir() / config["figure_stem"]
    png_path = figure_stem.with_suffix(".png")
    pdf_path = figure_stem.with_suffix(".pdf")
    make_figure(
        table,
        display_name=config["display_name"],
        png_path=png_path,
        pdf_path=pdf_path,
    )
    command = (
        "python -m "
        "jspace_phase4.experiments.p4_lineage_trajectory_analysis "
        f"--config {arguments.config}"
    )
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
            input_manifest_sha256=object_sha256(manifest_payload),
            model={
                "family": "OLMo-3/3.1 32B",
                "checkpoints": [
                    {
                        "model_id": source["model_id"],
                        "revision": source["model_revision"],
                    }
                    for source in checkpoint_sources
                ],
            },
            seed_contract=SEED_CONTRACT,
        ),
    )
    create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "Four-checkpoint OLMo own/common-lens trajectory synthesis "
            "over known-bank development cohorts; 48 frame/metric "
            "summaries independently reconstructed."
        ),
        command=command,
        outputs=[
            result_path,
            manifest_path,
            table_path,
            png_path,
            pdf_path,
        ],
        inputs=inputs,
    )
    print(json.dumps({
        "checkpoints": checkpoint_sources,
        "independent_validation": independent_validation,
        "table": str(table_path),
        "figure": str(png_path),
    }, indent=1))


if __name__ == "__main__":
    main()
