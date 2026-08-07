"""Outcome-blind power ruler for the two-component P4-P1 endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..manifests import atomic_json, file_sha256, require_clean_tree
from ..registry4 import create
from .p4_bank_w_power import _studentized_from_means, _wilson_interval, stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--simulate", action="store_true")
    group.add_argument("--register-existing", action="store_true")
    return parser.parse_args()


def _signs(n_families: int, *, draws: int, seed: int,
           exact_max_families: int) -> tuple[np.ndarray, bool]:
    if n_families <= exact_max_families:
        patterns = np.arange(2 ** n_families, dtype=np.uint64)[:, None]
        bits = ((patterns >> np.arange(
            n_families, dtype=np.uint64)) & 1).astype(np.int8)
        return 1 - 2 * bits, True
    generator = np.random.default_rng(seed)
    return 2 * generator.integers(
        0, 2, size=(draws, n_families), dtype=np.int8) - 1, False


def batch_intersection_union_pvalues(
        values: np.ndarray, signs: np.ndarray, *, exact: bool,
        permutation_chunk_size: int = 256) -> np.ndarray:
    """Return max(component p-values) under shared family sign flips."""
    cube = np.asarray(values, dtype=np.float64)
    sign_matrix = np.asarray(signs, dtype=np.int8)
    if cube.ndim != 3 or cube.shape[2] != 2:
        raise ValueError("P4-P1 simulations must be simulation x family x 2")
    n_simulations, n_families, _ = cube.shape
    if sign_matrix.ndim != 2 or sign_matrix.shape[1] != n_families:
        raise ValueError("sign matrix family dimension mismatch")
    means = cube.mean(axis=1)
    sum_squares = np.square(cube).sum(axis=1)
    observed = _studentized_from_means(means, sum_squares, n_families)
    extreme = np.zeros((n_simulations, 2), dtype=np.int64)
    for start in range(0, len(sign_matrix), permutation_chunk_size):
        selected = sign_matrix[start:start + permutation_chunk_size]
        permuted_means = np.einsum(
            "bf,sfe->sbe", selected, cube, optimize=True) / n_families
        permuted_t = _studentized_from_means(
            permuted_means, sum_squares[:, None, :], n_families)
        extreme += np.count_nonzero(
            permuted_t >= observed[:, None, :] - 1e-14, axis=1)
    denominator = len(sign_matrix) if exact else len(sign_matrix) + 1
    component = (
        extreme / denominator if exact else (extreme + 1) / denominator)
    return np.max(component, axis=1)


def simulate_iut_rejection_rate(
        *, n_simulations: int, n_families: int, effects: np.ndarray,
        family_sd: float, correlation: float, distribution: str,
        heavy_tail_df: int, signs: np.ndarray, exact_signs: bool,
        seed: int, alpha: float, simulation_batch_size: int = 50,
        permutation_chunk_size: int = 256) -> dict:
    effects = np.asarray(effects, dtype=np.float64)
    if effects.shape != (2,):
        raise ValueError("P4-P1 needs semantic and bridge-specific effects")
    if not -1 < correlation < 1:
        raise ValueError("endpoint correlation must be strictly within (-1,1)")
    covariance = family_sd ** 2 * np.asarray([
        [1.0, correlation], [correlation, 1.0]])
    generator = np.random.default_rng(seed)
    rejections = 0
    completed = 0
    while completed < n_simulations:
        count = min(simulation_batch_size, n_simulations - completed)
        cube = generator.multivariate_normal(
            np.zeros(2), covariance, size=(count, n_families),
            check_valid="raise")
        if distribution == "student_t":
            if heavy_tail_df <= 2:
                raise ValueError("Student t calibration needs df > 2")
            cube *= np.sqrt(
                (heavy_tail_df - 2) / generator.chisquare(
                    heavy_tail_df, size=(count, n_families, 1)))
        elif distribution != "normal":
            raise ValueError(distribution)
        cube += effects[None, None, :]
        pvalues = batch_intersection_union_pvalues(
            cube, signs, exact=exact_signs,
            permutation_chunk_size=permutation_chunk_size)
        rejections += int(np.count_nonzero(pvalues <= alpha))
        completed += count
    rate = rejections / n_simulations
    return {
        "n_families": int(n_families),
        "effects_nats": effects.tolist(),
        "family_sd_nats": float(family_sd),
        "endpoint_correlation": float(correlation),
        "distribution": distribution,
        "heavy_tail_df": heavy_tail_df if distribution == "student_t" else None,
        "rejections": int(rejections),
        "n_simulations": int(n_simulations),
        "rejection_rate": float(rate),
        "monte_carlo_se": float(math.sqrt(rate * (1 - rate) / n_simulations)),
        "wilson_ci95": _wilson_interval(rejections, n_simulations),
        "alpha": float(alpha),
        "sign_patterns_or_draws": int(len(signs)),
        "sign_method": "exact" if exact_signs else "monte-carlo-plus-one",
    }


def _phase3_event(config: Mapping) -> dict:
    calibration = config["development_calibration"]
    registry = Path(calibration["source_registry"])
    if file_sha256(registry) != calibration["source_registry_sha256"]:
        raise RuntimeError("Phase 3 source registry hash drift")
    events = [json.loads(line) for line in registry.read_text().splitlines()
              if line.strip()]
    matches = [row for row in events
               if row.get("event") == "evidence_created"
               and row.get("evidence_id") == calibration["evidence_id"]]
    if len(matches) != 1:
        raise RuntimeError("expected one Phase 3 calibration event")
    return matches[0]


def derive_calibration(config: Mapping) -> dict:
    calibration = config["development_calibration"]
    source = Path(calibration["paired_rows"])
    actual = file_sha256(source)
    if actual != calibration["paired_rows_sha256"]:
        raise RuntimeError("Phase 3 bridge calibration rows hash drift")
    event = _phase3_event(config)
    if actual not in {row["sha256"] for row in event["outputs"]}:
        raise RuntimeError("Phase 3 calibration rows are not registered")
    columns = list(calibration["endpoint_columns"])
    if len(columns) != 2:
        raise RuntimeError("P4-P1 calibration needs exactly two endpoints")
    rows = pd.read_parquet(source)
    family = rows.groupby("canonical_family")[columns].mean().dropna()
    if len(family) < 10:
        raise RuntimeError("too few known development families for calibration")
    standard_deviations = family.std(ddof=1)
    rounding = float(calibration["conservative_sd_rounding_nats"])
    conservative_sd = (
        math.ceil(float(standard_deviations.max()) / rounding - 1e-12)
        * rounding)
    correlation = float(family.corr().iloc[0, 1])
    return {
        "source_role": (
            "registered consumed Phase 3 development variability proxy; "
            "no Bank B outcome is read"),
        "evidence_id": calibration["evidence_id"],
        "paired_rows": str(source),
        "paired_rows_sha256": actual,
        "n_items": int(len(rows)),
        "n_families": int(len(family)),
        "endpoint_columns": columns,
        "family_mean_nats": {
            column: float(family[column].mean()) for column in columns},
        "family_sd_nats": {
            column: float(standard_deviations[column]) for column in columns},
        "family_correlation": correlation,
        "conservative_common_sd_nats": float(conservative_sd),
        "conservative_rule": (
            "larger endpoint family SD rounded upward to "
            f"{rounding:g} nat"),
    }


def run_simulation(config: Mapping) -> dict:
    simulation = config["simulation"]
    randomization = config["primary_randomization"]
    calibration = derive_calibration(config)
    sd = float(calibration["conservative_common_sd_nats"])
    correlation = float(calibration["family_correlation"])
    n_simulations = int(simulation["n_simulations"])
    alpha = float(randomization["alpha"])
    root_seed = int(simulation["seed"])
    family_counts = [int(value) for value in simulation["family_counts"]]
    signs_by_n = {}
    for n_families in family_counts:
        signs_by_n[n_families] = _signs(
            n_families, draws=int(simulation["permutation_draws"]),
            seed=stable_seed(root_seed, "signs", n_families),
            exact_max_families=int(randomization["exact_max_families"]))

    def scenario(name: str, n_families: int, effects: list[float], *,
                 distribution: str = "student_t",
                 endpoint_correlation: float | None = None) -> dict:
        signs, exact = signs_by_n[n_families]
        result = simulate_iut_rejection_rate(
            n_simulations=n_simulations, n_families=n_families,
            effects=np.asarray(effects), family_sd=sd,
            correlation=(correlation if endpoint_correlation is None
                         else endpoint_correlation),
            distribution=distribution,
            heavy_tail_df=int(simulation["heavy_tail_df"]),
            signs=signs, exact_signs=exact,
            seed=stable_seed(root_seed, "scenario", name), alpha=alpha,
            simulation_batch_size=int(simulation["simulation_batch_size"]),
            permutation_chunk_size=int(
                simulation["permutation_chunk_size"]))
        result["scenario"] = name
        return result

    current_n = int(simulation["current_confirmatory_families"])
    strong = sd * float(simulation["composite_null_other_endpoint_sd"])
    type_i = []
    for distribution in ("normal", "student_t"):
        for label, effects in (
                ("both-boundary", [0.0, 0.0]),
                ("semantic-boundary", [0.0, strong]),
                ("bridge-boundary", [strong, 0.0])):
            type_i.append(scenario(
                f"null-{label}-{distribution}", current_n, effects,
                distribution=distribution))
    bounds = [float(value) for value in
              simulation["type_i_acceptance_interval"]]
    for row in type_i:
        if "both-boundary" in row["scenario"]:
            row["type_i_pass"] = row["rejection_rate"] <= bounds[1]
        else:
            row["type_i_pass"] = bool(
                bounds[0] <= row["rejection_rate"] <= bounds[1])

    sesoi = float(simulation["candidate_joint_sesoi_nats"])
    power_at_sesoi = [scenario(
        f"joint-sesoi-n{n_families}", n_families, [sesoi, sesoi])
        for n_families in family_counts]
    effect_grid = [float(value) for value in simulation["effect_grid_nats"]]
    power_curve = [scenario(
        f"joint-grid-{effect:g}-n{n_families}", n_families,
        [effect, effect])
        for n_families in family_counts
        for effect in effect_grid]
    power_target = float(simulation["power_target"])
    mde_by_n = {}
    for n_families in family_counts:
        eligible = [row for row in power_curve
                    if row["n_families"] == n_families
                    and row["rejection_rate"] >= power_target]
        mde_by_n[str(n_families)] = (
            min(row["effects_nats"][0] for row in eligible)
            if eligible else None)
    current_power = next(
        row["rejection_rate"] for row in power_at_sesoi
        if row["n_families"] == current_n)
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "outcome_blinding": (
            "No Bank B intervention, confirmatory, or replication outcome "
            "was read."),
        "calibration": calibration,
        "primary": {
            "components": ["semantic", "bridge_specific"],
            "p_value": "max(one-sided component sign-flip p-values)",
            "intersection_union": True,
            "alpha": alpha,
            "unit": "canonical_family",
        },
        "candidate_joint_sesoi_nats": sesoi,
        "type_i": type_i,
        "power_at_sesoi": power_at_sesoi,
        "power_curve": power_curve,
        "mde_at_power_target_by_n": mde_by_n,
        "power_target": power_target,
        "current_confirmatory_families": current_n,
        "current_power_at_candidate_sesoi": current_power,
        "candidate_power_ready": bool(current_power >= power_target),
        "freeze_boundary": (
            "Power is a candidate design ruler only. Bank B cannot freeze "
            "until the candidate clears the target and PI/independent "
            "protocol review approve the SESOI and design."),
    }


def plot_power(result: Mapping, *, png: Path, pdf: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    rows = result["power_curve"]
    for n_families in sorted({row["n_families"] for row in rows}):
        selected = sorted(
            (row for row in rows if row["n_families"] == n_families),
            key=lambda row: row["effects_nats"][0])
        axes[0].plot(
            [row["effects_nats"][0] for row in selected],
            [row["rejection_rate"] for row in selected], marker="o",
            label=f"n={n_families}")
    axes[0].axhline(result["power_target"], color="#D55E00", linestyle="--")
    axes[0].axvline(
        result["candidate_joint_sesoi_nats"], color="#555555", linestyle=":")
    axes[0].set(xlabel="equal component effect (nats)", ylabel="joint power",
                title="A · Intersection-union power")
    axes[0].legend(frameon=False, fontsize=8)

    sesoi = sorted(result["power_at_sesoi"], key=lambda row: row["n_families"])
    axes[1].plot(
        [row["n_families"] for row in sesoi],
        [row["rejection_rate"] for row in sesoi], marker="o", color="#0072B2")
    axes[1].axhline(result["power_target"], color="#D55E00", linestyle="--")
    axes[1].set(xlabel="confirmatory families", ylabel="joint power",
                title="B · Power at candidate SESOI")
    for axis in axes:
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Bank B P4-P1 outcome-blind power ruler")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    require_clean_tree()
    outputs = {key: Path(value) for key, value in config["outputs"].items()}
    if arguments.simulate:
        result = run_simulation(config)
        atomic_json(outputs["result"], result)
        plot_power(result, png=outputs["figure_png"],
                   pdf=outputs["figure_pdf"])
        print(json.dumps({
            "status": "simulated-unregistered",
            "candidate_power_ready": result["candidate_power_ready"],
            "current_power_at_candidate_sesoi": result[
                "current_power_at_candidate_sesoi"],
            "mde_at_power_target_by_n": result[
                "mde_at_power_target_by_n"],
        }, indent=1))
        return
    if not all(path.exists() for path in outputs.values()):
        raise RuntimeError("Bank B power outputs are incomplete")
    result = json.loads(outputs["result"].read_text())
    command = (
        "python -m jspace_phase4.experiments.p4_bank_b_power "
        f"--config {arguments.config} --register-existing")
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            "Outcome-blind P4-P1 intersection-union power calibration from "
            "registered consumed Phase 3 bridge-swap variability; reports "
            "the current Bank B design verdict without opening Bank B "
            "intervention outcomes."),
        command=command, outputs=outputs.values(),
        inputs={
            "config": file_sha256(config_path),
            "bank_b_audit": file_sha256(config["bank_b_audit"]),
            "calibration_rows": config["development_calibration"][
                "paired_rows_sha256"],
            "phase3_registry": config["development_calibration"][
                "source_registry_sha256"],
        })
    print(json.dumps({
        "status": "registered", "evidence_id": config["evidence_id"],
        "candidate_power_ready": result["candidate_power_ready"],
    }, indent=1))


if __name__ == "__main__":
    main()
