"""Post-hoc common-support closure for the OLMo Phase 4 trajectory.

The registered trajectory intentionally uses checkpoint-specific prospective
capability cohorts.  This development-only producer asks the narrower
question requested by Phase 4.2: does the pattern persist when every compared
checkpoint is restricted to the same direct-and-composed-capable facts?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .p4_lens_frame_analysis import validate_envelope
from .p4_lineage_analysis import add_effects
from .p4_lineage_trajectory_analysis import (
    atomic_csv,
    load_checkpoint,
    validate_shared_contract,
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
from ..seeds import SEED_CONTRACT, stable_seed


CHECKPOINT_ORDER = (
    "olmo3-base",
    "olmo3-think",
    "olmo31-think",
    "olmo31-instruct",
)
FRAMES = ("own", "common")
BANKS = ("F", "S")
METRICS = ("direct", "composed", "composition")
POPULATION_SPECS = (
    {
        "key": "all_four",
        "label": "All-four-checkpoint intersection",
        "checkpoints": CHECKPOINT_ORDER,
        "contrasts": (
            ("olmo3-base", "olmo3-think"),
            ("olmo3-think", "olmo31-think"),
            ("olmo31-think", "olmo31-instruct"),
        ),
    },
    {
        "key": "base_vs_30think",
        "label": "Base versus 3.0 Think intersection",
        "checkpoints": ("olmo3-base", "olmo3-think"),
        "contrasts": (("olmo3-base", "olmo3-think"),),
    },
    {
        "key": "30think_vs_31think",
        "label": "3.0 versus 3.1 Think intersection",
        "checkpoints": ("olmo3-think", "olmo31-think"),
        "contrasts": (("olmo3-think", "olmo31-think"),),
    },
    {
        "key": "31think_vs_31instruct",
        "label": "3.1 Think versus 3.1 Instruct intersection",
        "checkpoints": ("olmo31-think", "olmo31-instruct"),
        "contrasts": (("olmo31-think", "olmo31-instruct"),),
    },
)
FIGURE_LAYOUT_RECT = (0.0, 0.075, 1.0, 0.93)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _complete_fact_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Return membership metadata without consulting any outcome column."""
    required = {
        "fact_id",
        "canonical_family",
        "bank",
        "variant",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"lineage grid lacks membership fields: {missing}")
    membership = frame[[
        "fact_id", "canonical_family", "bank", "variant"
    ]].copy()
    if membership.duplicated(["fact_id", "variant"]).any():
        raise ValueError("lineage grid repeats a fact/variant row")
    counts = membership.groupby("fact_id").variant.agg(
        lambda values: tuple(sorted(values)))
    valid = counts[counts == ("composed", "direct")].index
    if len(valid) != membership.fact_id.nunique():
        raise ValueError("every cohort fact must have direct and composed rows")
    metadata = (
        membership[membership.fact_id.isin(valid)]
        [["fact_id", "canonical_family", "bank"]]
        .drop_duplicates()
        .sort_values("fact_id")
        .reset_index(drop=True)
    )
    if metadata.fact_id.duplicated().any():
        raise ValueError("fact metadata is inconsistent within a grid")
    return metadata


def construct_populations(
    checkpoint_frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict], list[dict]]:
    """Freeze common-support populations from IDs before reading outcomes."""
    if tuple(checkpoint_frames) != CHECKPOINT_ORDER:
        raise ValueError("checkpoint order is not frozen")
    metadata = {
        key: _complete_fact_metadata(frame)
        for key, frame in checkpoint_frames.items()
    }
    by_key = {
        key: table.set_index("fact_id")
        for key, table in metadata.items()
    }
    populations: dict[str, dict] = {}
    manifest_rows: list[dict] = []
    for specification in POPULATION_SPECS:
        checkpoint_keys = tuple(specification["checkpoints"])
        fact_sets = [set(metadata[key].fact_id) for key in checkpoint_keys]
        fact_ids = sorted(set.intersection(*fact_sets))
        if not fact_ids:
            raise ValueError(f"empty common cohort: {specification['key']}")
        reference = by_key[checkpoint_keys[0]].loc[fact_ids]
        for key in checkpoint_keys[1:]:
            candidate = by_key[key].loc[fact_ids]
            if not reference[["canonical_family", "bank"]].equals(
                    candidate[["canonical_family", "bank"]]):
                raise ValueError(
                    f"fact metadata changes across {specification['key']}")
        bank_counts = {
            bank: int((reference.bank == bank).sum())
            for bank in BANKS
        }
        family_counts = {
            bank: int(reference[reference.bank == bank]
                      .canonical_family.nunique())
            for bank in BANKS
        }
        record = {
            "key": specification["key"],
            "label": specification["label"],
            "checkpoints": list(checkpoint_keys),
            "contrasts": [list(pair) for pair in specification["contrasts"]],
            "fact_ids": fact_ids,
            "n_facts": len(fact_ids),
            "n_items": 2 * len(fact_ids),
            "n_families": int(reference.canonical_family.nunique()),
            "facts_by_bank": bank_counts,
            "families_by_bank": family_counts,
            "fact_id_set_sha256": object_sha256(fact_ids),
            "membership_rule": (
                "intersection of fact IDs with exactly one direct and one "
                "composed row at every named checkpoint"
            ),
        }
        populations[specification["key"]] = record
        manifest_rows.append({
            key: value for key, value in record.items()
            if key != "fact_ids"
        })
    return populations, manifest_rows


def _load_g5(
    checkpoint: dict,
    grid_source: dict,
) -> tuple[pd.DataFrame, dict[str, str]]:
    paths = {
        field: resolve_uri(checkpoint[f"g5_{field}_uri"])
        for field in ("parquet", "result", "manifest")
    }
    result = validate_envelope(paths["result"])
    manifest = validate_envelope(paths["manifest"])
    evidence_id = checkpoint["g5_evidence_id"]
    if result["provenance"]["evidence_id"] != evidence_id:
        raise RuntimeError(f"unexpected G5 evidence for {checkpoint['key']}")
    if not resolve(evidence_id)["live"]:
        raise RuntimeError(f"G5 evidence is not live: {evidence_id}")
    source = manifest["payload"]
    invariants = (
        "model_id",
        "model_revision",
        "tokenizer_manifest_sha256",
        "bank_sha256",
        "scoring_spec_sha256",
    )
    mismatches = {
        field: {"grid": grid_source[field], "g5": source[field]}
        for field in invariants
        if grid_source[field] != source[field]
    }
    if mismatches:
        raise RuntimeError(
            f"G5/grid contract mismatch for {checkpoint['key']}: "
            + json.dumps(mismatches, sort_keys=True)
        )
    frame = pd.read_parquet(paths["parquet"])
    frame = frame[frame.variant.isin(["direct", "composed"])].copy()
    if frame.duplicated(["fact_id", "variant"]).any():
        raise ValueError("G5 repeats a direct/composed fact row")
    frame["capability_margin"] = (
        frame.score_aggregate - frame.counterfactual_score_aggregate)
    if not np.isfinite(frame.capability_margin.to_numpy(dtype=float)).all():
        raise ValueError("G5 capability margins must be finite")
    return frame[[
        "fact_id",
        "variant",
        "canonical_family",
        "bank",
        "capability_margin",
    ]], {
        str(checkpoint[f"g5_{field}_uri"]): file_sha256(path)
        for field, path in paths.items()
    }


def enrich_effects(grid: pd.DataFrame, g5: pd.DataFrame) -> pd.DataFrame:
    effects = add_effects(grid)
    joined = effects.merge(
        g5,
        on=["fact_id", "variant", "canonical_family", "bank"],
        how="left",
        validate="one_to_one",
    )
    if joined.capability_margin.isna().any():
        raise ValueError("lineage fact lacks its G5 capability margin")
    return joined


def metric_fact_rows(
    effects: pd.DataFrame,
    *,
    bank: str,
    metric: str,
) -> pd.DataFrame:
    subset = effects[effects.bank == bank]
    if metric in {"direct", "composed"}:
        rows = subset[subset.variant == metric][[
            "fact_id",
            "canonical_family",
            "specific",
            "lp_baseline",
            "capability_margin",
        ]].copy()
        return rows.rename(columns={"specific": "value"})
    if metric != "composition":
        raise ValueError(metric)
    pivot = subset.pivot(
        index=["fact_id", "canonical_family"],
        columns="variant",
        values=["specific", "lp_baseline", "capability_margin"],
    )
    required = {
        (field, variant)
        for field in ("specific", "lp_baseline", "capability_margin")
        for variant in ("direct", "composed")
    }
    if not required.issubset(set(pivot.columns)):
        raise ValueError("composition rows are not direct/composed paired")
    result = pd.DataFrame(index=pivot.index).reset_index()
    result["value"] = (
        pivot[("specific", "composed")]
        - pivot[("specific", "direct")]
    ).to_numpy()
    result["lp_baseline"] = (
        pivot[("lp_baseline", "composed")]
        - pivot[("lp_baseline", "direct")]
    ).to_numpy()
    result["capability_margin"] = (
        pivot[("capability_margin", "composed")]
        - pivot[("capability_margin", "direct")]
    ).to_numpy()
    return result


def _seed(
    evidence_id: str,
    components: Iterable[str],
    *,
    base_seed: int,
) -> int:
    return stable_seed(
        experiment_id=evidence_id,
        item_id="|".join(str(value) for value in components),
        condition="common-cohort-bootstrap",
        base_seed=base_seed,
    )


def bootstrap_vector(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict:
    vector = np.asarray(values, dtype=float)
    if len(vector) < 3 or not np.isfinite(vector).all():
        raise ValueError("bootstrap requires at least three finite units")
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(0, len(vector), size=(draws, len(vector)))
    distribution = vector[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "estimate": float(vector.mean()),
        "ci95": [float(low), float(high)],
        "n_units": int(len(vector)),
        "n_bootstrap": int(draws),
        "bootstrap_seed": int(seed),
        "distribution_sha256": hashlib.sha256(
            np.asarray(distribution, dtype="<f8").tobytes()
        ).hexdigest(),
    }


def summarize_views(
    rows: pd.DataFrame,
    value_column: str,
    *,
    draws: int,
    evidence_id: str,
    seed_components: Iterable[str],
    base_seed: int,
) -> dict[str, dict]:
    fact_values = rows[value_column].to_numpy(dtype=float)
    family_values = (
        rows.groupby("canonical_family", sort=True)[value_column]
        .mean()
        .to_numpy(dtype=float)
    )
    return {
        "equal_family": bootstrap_vector(
            family_values,
            draws=draws,
            seed=_seed(
                evidence_id,
                [*seed_components, "equal_family"],
                base_seed=base_seed,
            ),
        ),
        "fact_weighted": bootstrap_vector(
            fact_values,
            draws=draws,
            seed=_seed(
                evidence_id,
                [*seed_components, "fact_weighted"],
                base_seed=base_seed,
            ),
        ),
    }


def family_leave_one_out(
    rows: pd.DataFrame,
    value_column: str,
) -> tuple[dict, list[dict]]:
    means = rows.groupby("canonical_family", sort=True)[value_column].mean()
    if len(means) < 3:
        raise ValueError("leave-one-family-out needs at least three families")
    overall = float(means.mean())
    details = []
    for family in means.index:
        estimate = float(means.drop(index=family).mean())
        details.append({
            "family_dropped": str(family),
            "estimate_without_family": estimate,
            "shift_from_full": estimate - overall,
        })
    most = max(details, key=lambda row: abs(row["shift_from_full"]))
    return {
        "full_estimate": overall,
        "minimum": min(row["estimate_without_family"] for row in details),
        "maximum": max(row["estimate_without_family"] for row in details),
        "max_abs_shift": abs(most["shift_from_full"]),
        "most_influential_family": most["family_dropped"],
        "n_families": len(details),
    }, details


def _ols_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if len(x_values) != len(y_values) or len(x_values) < 3:
        raise ValueError("adjusted sensitivity needs paired finite vectors")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("adjusted sensitivity received non-finite values")
    centered = x_values - x_values.mean()
    denominator = float(np.square(centered).sum())
    slope = (
        0.0
        if denominator <= 1e-15
        else float((centered * (y_values - y_values.mean())).sum()
                   / denominator)
    )
    intercept = float(y_values.mean() - slope * x_values.mean())
    return intercept, slope


def regression_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    draws: int,
    seed: int,
    chunk_size: int = 5000,
) -> dict:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    intercept, slope = _ols_intercept(x_values, y_values)
    generator = np.random.Generator(np.random.PCG64(seed))
    distribution = np.empty(draws, dtype=float)
    for start in range(0, draws, chunk_size):
        stop = min(draws, start + chunk_size)
        indices = generator.integers(
            0, len(x_values), size=(stop - start, len(x_values)))
        sampled_x = x_values[indices]
        sampled_y = y_values[indices]
        mean_x = sampled_x.mean(axis=1)
        mean_y = sampled_y.mean(axis=1)
        centered_x = sampled_x - mean_x[:, None]
        denominator = np.square(centered_x).sum(axis=1)
        numerator = (
            centered_x * (sampled_y - mean_y[:, None])
        ).sum(axis=1)
        sampled_slope = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-15,
        )
        distribution[start:stop] = mean_y - sampled_slope * mean_x
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "estimate": intercept,
        "slope": slope,
        "ci95": [float(low), float(high)],
        "n_units": int(len(x_values)),
        "n_bootstrap": int(draws),
        "bootstrap_seed": int(seed),
        "distribution_sha256": hashlib.sha256(
            np.asarray(distribution, dtype="<f8").tobytes()
        ).hexdigest(),
        "estimand": "intercept at zero baseline-LP change",
    }


def summarize_adjusted_views(
    rows: pd.DataFrame,
    *,
    draws: int,
    evidence_id: str,
    seed_components: Iterable[str],
    base_seed: int,
) -> dict[str, dict]:
    family = rows.groupby("canonical_family", sort=True)[
        ["baseline_lp_delta", "specific_delta"]
    ].mean()
    return {
        "equal_family": regression_bootstrap(
            family.baseline_lp_delta.to_numpy(dtype=float),
            family.specific_delta.to_numpy(dtype=float),
            draws=draws,
            seed=_seed(
                evidence_id,
                [*seed_components, "adjusted", "equal_family"],
                base_seed=base_seed,
            ),
        ),
        "fact_weighted": regression_bootstrap(
            rows.baseline_lp_delta.to_numpy(dtype=float),
            rows.specific_delta.to_numpy(dtype=float),
            draws=draws,
            seed=_seed(
                evidence_id,
                [*seed_components, "adjusted", "fact_weighted"],
                base_seed=base_seed,
            ),
        ),
    }


def _summary_table_rows(
    identity: dict,
    value_kind: str,
    summaries: dict[str, dict],
) -> list[dict]:
    output = []
    for weighting, summary in summaries.items():
        output.append({
            **identity,
            "value_kind": value_kind,
            "weighting": weighting,
            "estimate": summary["estimate"],
            "ci95_low": summary["ci95"][0],
            "ci95_high": summary["ci95"][1],
            "n_units": summary["n_units"],
            "n_bootstrap": summary["n_bootstrap"],
            "bootstrap_seed": summary["bootstrap_seed"],
            "distribution_sha256": summary["distribution_sha256"],
            "adjustment_slope": summary.get("slope"),
        })
    return output


def paired_contrast(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "fact_id",
        "canonical_family",
        "value",
        "lp_baseline",
        "capability_margin",
    ]
    paired = left[columns].merge(
        right[columns],
        on=["fact_id", "canonical_family"],
        how="outer",
        suffixes=("_left", "_right"),
        validate="one_to_one",
        indicator=True,
    )
    if not bool((paired._merge == "both").all()):
        raise ValueError("checkpoint contrast is not fact paired")
    paired["specific_delta"] = paired.value_right - paired.value_left
    paired["baseline_lp_delta"] = (
        paired.lp_baseline_right - paired.lp_baseline_left)
    paired["capability_margin_delta"] = (
        paired.capability_margin_right - paired.capability_margin_left)
    return paired.drop(columns="_merge")


def make_figure(
    checkpoint_table: pd.DataFrame,
    contrast_table: pd.DataFrame,
    *,
    display_name: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    colors = {"own": "#9b2226", "common": "#3a6ea5"}
    labels = ["Base", "3.0 Think", "3.1 Think", "3.1 Instruct"]
    figure, axes = plt.subplots(
        2, 2, figsize=(12.8, 8.5), constrained_layout=True)
    figure.get_layout_engine().set(rect=FIGURE_LAYOUT_RECT)
    for axis_index, metric in enumerate(("direct", "composition")):
        axis = axes[0, axis_index]
        for frame in FRAMES:
            subset = checkpoint_table[
                (checkpoint_table.population == "all_four")
                & (checkpoint_table.frame == frame)
                & (checkpoint_table.bank == "S")
                & (checkpoint_table.metric == metric)
                & (checkpoint_table.value_kind == "specific")
                & (checkpoint_table.weighting == "equal_family")
            ].set_index("checkpoint").loc[list(CHECKPOINT_ORDER)].reset_index()
            x = np.arange(4) + (-0.045 if frame == "own" else 0.045)
            estimate = subset.estimate.to_numpy(dtype=float)
            low = subset.ci95_low.to_numpy(dtype=float)
            high = subset.ci95_high.to_numpy(dtype=float)
            axis.errorbar(
                x[:3], estimate[:3],
                yerr=[estimate[:3] - low[:3], high[:3] - estimate[:3]],
                fmt="o-", capsize=3, color=colors[frame],
                label="Own lens" if frame == "own" else "Common base lens",
            )
            axis.errorbar(
                x[3], estimate[3],
                yerr=[[estimate[3] - low[3]], [high[3] - estimate[3]]],
                fmt="s", capsize=3, color=colors[frame],
            )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xticks(range(4), labels, rotation=12)
        axis.set_ylabel("J − matched-control effect (nats)")
        axis.set_title(
            f"{chr(ord('a') + axis_index)}  Bank S "
            + ("direct" if metric == "direct" else "composed − direct")
        )
        if axis_index == 0:
            axis.legend(frameon=False)

    pair_population = {
        "olmo3-base->olmo3-think": "Base → 3.0 Think",
        "olmo3-think->olmo31-think": "3.0 → 3.1 Think",
        "olmo31-think->olmo31-instruct": "3.1 Think → Instruct",
    }
    dedicated = {
        "olmo3-base->olmo3-think": "base_vs_30think",
        "olmo3-think->olmo31-think": "30think_vs_31think",
        "olmo31-think->olmo31-instruct": "31think_vs_31instruct",
    }
    for axis_offset, metric in enumerate(("direct", "composition"), start=2):
        axis = axes.flat[axis_offset]
        for frame in FRAMES:
            rows = []
            for contrast, population in dedicated.items():
                row = contrast_table[
                    (contrast_table.population == population)
                    & (contrast_table.contrast == contrast)
                    & (contrast_table.frame == frame)
                    & (contrast_table.bank == "S")
                    & (contrast_table.metric == metric)
                    & (contrast_table.value_kind == "specific_delta")
                    & (contrast_table.weighting == "equal_family")
                ]
                if len(row) != 1:
                    raise RuntimeError("figure contrast row is not unique")
                rows.append(row.iloc[0])
            estimate = np.asarray([row.estimate for row in rows])
            low = np.asarray([row.ci95_low for row in rows])
            high = np.asarray([row.ci95_high for row in rows])
            x = np.arange(3) + (-0.055 if frame == "own" else 0.055)
            axis.errorbar(
                x, estimate, yerr=[estimate - low, high - estimate],
                fmt="o", capsize=3, color=colors[frame],
                label="Own lens" if frame == "own" else "Common base lens",
            )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xticks(
            range(3), [pair_population[key] for key in dedicated], rotation=12)
        axis.set_ylabel("Paired checkpoint difference (nats)")
        axis.set_title(
            f"{chr(ord('a') + axis_offset)}  Pair-specific Bank S "
            + ("direct" if metric == "direct" else "composition")
        )
    figure.suptitle(display_name, fontsize=15)
    figure.text(
        0.5, 0.028,
        "Top: all-four fact intersection. Bottom: each pair's larger "
        "intersection. Lines connect only the Think path; the Instruct "
        "endpoint is a sibling.",
        ha="center", fontsize=9, color="#444444",
    )
    figure.text(
        0.5, 0.009,
        "Post-hoc Phase 4 development sensitivity; fact-paired within each "
        "contrast; equal-family bootstrap intervals; not a causal training "
        "effect.",
        ha="center", fontsize=9, color="#444444",
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
    if tuple(row["key"] for row in checkpoints) != CHECKPOINT_ORDER:
        raise ValueError("common-cohort checkpoint order is not frozen")

    grids: dict[str, dict[str, pd.DataFrame]] = {}
    sources = []
    upstream: dict[str, str] = {}
    checkpoint_manifest = []
    for checkpoint in checkpoints:
        own, common, source, grid_upstream = load_checkpoint(checkpoint)
        g5, g5_upstream = _load_g5(checkpoint, source)
        grids[checkpoint["key"]] = {
            "own": enrich_effects(own, g5),
            "common": enrich_effects(
                own if checkpoint.get("common_is_own", False) else common,
                g5,
            ),
        }
        sources.append(source)
        upstream.update(grid_upstream)
        upstream.update(g5_upstream)
        checkpoint_manifest.append({
            "key": checkpoint["key"],
            "label": checkpoint["label"],
            "lineage_role": checkpoint["lineage_role"],
            "model_id": source["model_id"],
            "model_revision": source["model_revision"],
            "own_grid_evidence_id": checkpoint["own_grid_evidence_id"],
            "common_grid_evidence_id": (
                checkpoint["own_grid_evidence_id"]
                if checkpoint.get("common_is_own", False)
                else checkpoint["common_grid_evidence_id"]
            ),
            "g5_evidence_id": checkpoint["g5_evidence_id"],
        })
    shared_contract = validate_shared_contract(sources)
    populations, population_manifest = construct_populations({
        key: grids[key]["own"] for key in CHECKPOINT_ORDER
    })
    draws = int(config["bootstrap_draws"])
    base_seed = int(config["bootstrap_seed"])
    evidence_id = str(config["evidence_id"])
    checkpoint_rows: list[dict] = []
    contrast_rows: list[dict] = []
    loo_rows: list[dict] = []
    loo_summaries: list[dict] = []

    for population_key, population in populations.items():
        fact_ids = set(population["fact_ids"])
        for frame in FRAMES:
            metric_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
            for checkpoint_key in population["checkpoints"]:
                effects = grids[checkpoint_key][frame]
                restricted = effects[effects.fact_id.isin(fact_ids)]
                if restricted.fact_id.nunique() != population["n_facts"]:
                    raise ValueError("population restriction lost a fact")
                for bank in BANKS:
                    for metric in METRICS:
                        metric_rows = metric_fact_rows(
                            restricted, bank=bank, metric=metric)
                        expected = int(population["facts_by_bank"][bank])
                        if len(metric_rows) != expected:
                            raise ValueError(
                                f"{population_key} {bank} {metric} has "
                                f"{len(metric_rows)} rows, expected {expected}")
                        metric_cache[(checkpoint_key, bank, metric)] = metric_rows
                        identity = {
                            "population": population_key,
                            "frame": frame,
                            "checkpoint": checkpoint_key,
                            "bank": bank,
                            "metric": metric,
                        }
                        summaries = summarize_views(
                            metric_rows,
                            "value",
                            draws=draws,
                            evidence_id=evidence_id,
                            seed_components=[
                                population_key, frame, checkpoint_key,
                                bank, metric, "checkpoint",
                            ],
                            base_seed=base_seed,
                        )
                        checkpoint_rows.extend(_summary_table_rows(
                            identity, "specific", summaries))
                        loo_summary, details = family_leave_one_out(
                            metric_rows, "value")
                        loo_summaries.append({
                            **identity,
                            "value_kind": "specific",
                            **loo_summary,
                        })
                        loo_rows.extend({
                            **identity,
                            "value_kind": "specific",
                            **detail,
                        } for detail in details)

            for left_key, right_key in population["contrasts"]:
                contrast_key = f"{left_key}->{right_key}"
                for bank in BANKS:
                    for metric in METRICS:
                        paired = paired_contrast(
                            metric_cache[(left_key, bank, metric)],
                            metric_cache[(right_key, bank, metric)],
                        )
                        identity = {
                            "population": population_key,
                            "frame": frame,
                            "contrast": contrast_key,
                            "left_checkpoint": left_key,
                            "right_checkpoint": right_key,
                            "bank": bank,
                            "metric": metric,
                        }
                        for value_kind in (
                            "specific_delta",
                            "baseline_lp_delta",
                            "capability_margin_delta",
                        ):
                            summaries = summarize_views(
                                paired,
                                value_kind,
                                draws=draws,
                                evidence_id=evidence_id,
                                seed_components=[
                                    population_key, frame, contrast_key,
                                    bank, metric, value_kind,
                                ],
                                base_seed=base_seed,
                            )
                            contrast_rows.extend(_summary_table_rows(
                                identity, value_kind, summaries))
                            loo_summary, details = family_leave_one_out(
                                paired, value_kind)
                            loo_summaries.append({
                                **identity,
                                "value_kind": value_kind,
                                **loo_summary,
                            })
                            loo_rows.extend({
                                **identity,
                                "value_kind": value_kind,
                                **detail,
                            } for detail in details)
                        adjusted = summarize_adjusted_views(
                            paired,
                            draws=draws,
                            evidence_id=evidence_id,
                            seed_components=[
                                population_key, frame, contrast_key,
                                bank, metric,
                            ],
                            base_seed=base_seed,
                        )
                        contrast_rows.extend(_summary_table_rows(
                            identity,
                            "baseline_lp_adjusted_specific_delta",
                            adjusted,
                        ))

    checkpoint_table = pd.DataFrame(checkpoint_rows).sort_values([
        "population", "frame", "bank", "metric", "checkpoint", "weighting"
    ]).reset_index(drop=True)
    contrast_table = pd.DataFrame(contrast_rows).sort_values([
        "population", "frame", "bank", "metric", "contrast",
        "value_kind", "weighting",
    ]).reset_index(drop=True)
    loo_table = pd.DataFrame(loo_rows).sort_values([
        "population", "frame", "bank", "metric", "value_kind",
        "family_dropped",
    ]).reset_index(drop=True)

    output_dir = (
        metrics_dir(config["slug"])
        / "common_cohort_analysis"
        / evidence_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    population_path = output_dir / "population_manifest.json"
    checkpoint_path = output_dir / "checkpoint_estimates.csv"
    contrast_path = output_dir / "adjacent_contrasts.csv"
    loo_path = output_dir / "family_leave_one_out.csv"
    manifest_path = output_dir / "input_manifest.json"
    result_path = output_dir / f"common_cohort_analysis_{config['slug']}.json"
    atomic_json(population_path, {
        "schema_version": 1,
        "constructed_before_outcome_analysis": True,
        "populations": population_manifest,
    })
    atomic_csv(checkpoint_path, checkpoint_table)
    atomic_csv(contrast_path, contrast_table)
    atomic_csv(loo_path, loo_table)
    manifest_payload = {
        "schema_version": 1,
        "experiment_id": evidence_id,
        "config_sha256": file_sha256(config_path),
        "analysis_family": "olmo-common-support-development-sensitivity",
        "checkpoints": checkpoint_manifest,
        "population_manifest_sha256": file_sha256(population_path),
        "shared_contract": shared_contract,
        "upstream": upstream,
        "bootstrap": {
            "draws": draws,
            "base_seed": base_seed,
            "seed_contract": SEED_CONTRACT,
            "units": ["canonical_family", "fact_id"],
        },
        "code_commit": clean["code_commit"],
    }
    atomic_json(manifest_path, {
        "schema_version": 1,
        "payload": manifest_payload,
        "payload_sha256": object_sha256(manifest_payload),
    })
    figure_stem = figures_dir() / config["figure_stem"]
    png_path = figure_stem.with_suffix(".png")
    pdf_path = figure_stem.with_suffix(".pdf")
    make_figure(
        checkpoint_table,
        contrast_table,
        display_name=config["display_name"],
        png_path=png_path,
        pdf_path=pdf_path,
    )
    payload = {
        "schema_version": 1,
        "tier": config["tier"],
        "development_only": True,
        "post_hoc_common_support": True,
        "claim_guard": (
            "Common-support estimates remove capability-cohort composition "
            "as one explanation but do not identify a causal effect of "
            "training or mode. Known Phase 3 banks remain development-only."
        ),
        "effect_definition": {
            "specific": (
                "(lp_meanJ_span_safe - lp_baseline) - "
                "(lp_ss_matched - lp_baseline)"
            ),
            "composition": "specific_composed - specific_direct",
            "checkpoint_delta": "right checkpoint - left checkpoint",
            "capability_margin": (
                "G5 original-answer aggregate LP minus counterfactual-answer "
                "aggregate LP"
            ),
            "baseline_adjusted_sensitivity": (
                "OLS intercept of paired specificity change on paired "
                "baseline-answer-LP change, evaluated at zero LP change"
            ),
        },
        "checkpoint_relationship": {
            "think_path": list(CHECKPOINT_ORDER[:3]),
            "sibling_endpoint": CHECKPOINT_ORDER[3],
        },
        "populations": population_manifest,
        "checkpoints": checkpoint_manifest,
        "shared_contract": shared_contract,
        "bootstrap": manifest_payload["bootstrap"],
        "checkpoint_estimates": checkpoint_rows,
        "adjacent_contrasts": contrast_rows,
        "family_leave_one_out_summaries": loo_summaries,
    }
    command = (
        "python -m "
        "jspace_phase4.experiments.p4_lineage_common_cohort_analysis "
        f"--config {arguments.config}"
    )
    inputs = {
        "input_manifest": file_sha256(manifest_path),
        "population_manifest": file_sha256(population_path),
        **upstream,
    }
    write_result4(
        payload,
        result_path,
        Provenance4(
            evidence_id=evidence_id,
            tier=config["tier"],
            command=command,
            inputs=inputs,
            input_manifest_sha256=object_sha256(manifest_payload),
            model={
                "family": "OLMo-3/3.1 32B",
                "checkpoints": [
                    {
                        "model_id": row["model_id"],
                        "revision": row["model_revision"],
                    }
                    for row in checkpoint_manifest
                ],
            },
            seed_contract=SEED_CONTRACT,
        ),
    )
    outputs = [
        result_path,
        manifest_path,
        population_path,
        checkpoint_path,
        contrast_path,
        loo_path,
        png_path,
        pdf_path,
    ]
    create(
        evidence_id,
        tier=config["tier"],
        what=(
            "Post-hoc all-four and adjacent-pair OLMo common-support "
            "development analysis in own/common lens frames, with paired "
            "fact/family bootstraps and baseline-LP sensitivity."
        ),
        command=command,
        outputs=outputs,
        inputs=inputs,
    )
    print(json.dumps({
        "populations": population_manifest,
        "checkpoint_rows": len(checkpoint_rows),
        "contrast_rows": len(contrast_rows),
        "leave_one_out_rows": len(loo_rows),
        "result": str(result_path),
        "figure": str(png_path),
    }, indent=1))


if __name__ == "__main__":
    main()
