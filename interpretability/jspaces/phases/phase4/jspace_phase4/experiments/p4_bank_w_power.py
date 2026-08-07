"""Calibrate the shared-family max-T primary for Phase 4 Bank W.

The future Bank W outcomes are never opened here. Variability and an
empirical dependence sensitivity come only from registered, known-bank
development rows. The planning ruler uses the largest common-family SD,
rounded upward, for every target model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

def _find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError("cannot locate git repository root")

import subprocess
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from ..manifests import atomic_json, file_sha256, require_clean_tree
from ..registry4 import create, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--simulate", action="store_true")
    group.add_argument("--register-existing", action="store_true")
    return parser.parse_args()


def stable_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.sha256(payload.encode()).digest()[:8], "big")


def _validate_values(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("max-T values must be family x model")
    if matrix.shape[0] < 3 or matrix.shape[1] < 1:
        raise ValueError("max-T requires at least three families and one model")
    if not np.isfinite(matrix).all():
        raise ValueError("max-T values must all be finite on common support")
    if np.any(np.std(matrix, axis=0, ddof=1) <= 1e-14):
        raise ValueError("each max-T model column must have nonzero variance")
    return matrix


def _studentized_from_means(
        means: np.ndarray, sum_squares: np.ndarray, n: int) -> np.ndarray:
    centered = np.maximum(sum_squares - n * np.square(means), 0.0)
    standard_errors = np.sqrt(centered / (n - 1) / n)
    return np.divide(
        means, standard_errors,
        out=np.where(means > 0, np.inf,
                     np.where(means < 0, -np.inf, 0.0)),
        where=standard_errors > 1e-14)


def shared_family_max_t(
        values: np.ndarray, *, model_slugs: list[str] | None = None,
        draws: int = 100_000, seed: int = 48_151_623,
        exact_max_families: int = 20, chunk_size: int = 8192) -> dict:
    """One-sided max-T with the same family sign applied to every model."""
    matrix = _validate_values(values)
    n_families, n_models = matrix.shape
    names = model_slugs or [f"model-{index}" for index in range(n_models)]
    if len(names) != n_models or len(set(names)) != n_models:
        raise ValueError("model slugs must uniquely name every matrix column")
    means = matrix.mean(axis=0)
    sum_squares = np.square(matrix).sum(axis=0)
    observed_t = _studentized_from_means(means, sum_squares, n_families)
    observed_max = float(np.max(observed_t))
    exact = n_families <= exact_max_families
    n_patterns = 2 ** n_families if exact else int(draws)
    if n_patterns < 1:
        raise ValueError("draws must be positive")
    generator = np.random.default_rng(seed)
    extreme = 0
    digest = hashlib.sha256()
    for start in range(0, n_patterns, chunk_size):
        count = min(chunk_size, n_patterns - start)
        if exact:
            patterns = np.arange(start, start + count, dtype=np.uint64)[:, None]
            bits = ((patterns >> np.arange(
                n_families, dtype=np.uint64)) & 1).astype(np.int8)
            signs = 1 - 2 * bits
        else:
            signs = 2 * generator.integers(
                0, 2, size=(count, n_families), dtype=np.int8) - 1
        permuted_means = (signs @ matrix) / n_families
        permuted_t = _studentized_from_means(
            permuted_means, sum_squares, n_families)
        null_max = np.max(permuted_t, axis=1)
        digest.update(np.asarray(null_max, dtype="<f8").tobytes())
        extreme += int(np.count_nonzero(
            null_max >= observed_max - 1e-14))
    p_value = (extreme / n_patterns if exact
               else (extreme + 1) / (n_patterns + 1))
    return {
        "estimate_by_model": {
            name: float(value) for name, value in zip(names, means)},
        "t_by_model": {
            name: float(value) for name, value in zip(names, observed_t)},
        "observed_max_t": observed_max,
        "diagnostic_argmax_model": names[int(np.argmax(observed_t))],
        "p": float(p_value),
        "alternative": "greater",
        "n_families": int(n_families),
        "n_models_jointly_tested": int(n_models),
        "model_slugs_in_frozen_order": list(names),
        "method": ("exact-shared-family-signflip-max-t" if exact else
                   "monte-carlo-shared-family-signflip-max-t-plus-one"),
        "n_patterns_or_draws": int(n_patterns),
        "seed": None if exact else int(seed),
        "null_max_t_sha256": digest.hexdigest(),
        "model_selection_from_intervention_outcomes": False,
    }


def _random_signs(draws: int, families: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return 2 * generator.integers(
        0, 2, size=(draws, families), dtype=np.int8) - 1


def batch_max_t_pvalues(
        values: np.ndarray, signs: np.ndarray, *,
        permutation_chunk_size: int = 512) -> np.ndarray:
    """Monte Carlo plus-one max-T p-values for simulation replicates."""
    cube = np.asarray(values, dtype=np.float64)
    sign_matrix = np.asarray(signs, dtype=np.int8)
    if cube.ndim != 3:
        raise ValueError("simulation values must be simulation x family x model")
    n_simulations, n_families, _ = cube.shape
    if sign_matrix.ndim != 2 or sign_matrix.shape[1] != n_families:
        raise ValueError("sign matrix family dimension does not match")
    means = cube.mean(axis=1)
    sum_squares = np.square(cube).sum(axis=1)
    observed = np.max(_studentized_from_means(
        means, sum_squares, n_families), axis=1)
    extreme = np.zeros(n_simulations, dtype=np.int64)
    for start in range(0, len(sign_matrix), permutation_chunk_size):
        selected = sign_matrix[start:start + permutation_chunk_size]
        # Shared signs are contracted over the family dimension and therefore
        # preserve every cross-model family pairing.
        permuted_means = np.einsum(
            "bf,sfm->sbm", selected, cube, optimize=True) / n_families
        permuted_t = _studentized_from_means(
            permuted_means, sum_squares[:, None, :], n_families)
        null_max = np.max(permuted_t, axis=2)
        extreme += np.count_nonzero(
            null_max >= observed[:, None] - 1e-14, axis=1)
    return (extreme + 1) / (len(sign_matrix) + 1)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / total
        + z * z / (4 * total * total)) / denominator
    return [float(max(0.0, center - half)),
            float(min(1.0, center + half))]


def simulate_rejection_rate(
        *, n_simulations: int, n_families: int, effects: np.ndarray,
        family_sd: float, correlation: np.ndarray, distribution: str,
        heavy_tail_df: int, signs: np.ndarray, seed: int, alpha: float,
        simulation_batch_size: int = 50,
        permutation_chunk_size: int = 512) -> dict:
    effects = np.asarray(effects, dtype=np.float64)
    correlation = np.asarray(correlation, dtype=np.float64)
    n_models = len(effects)
    if correlation.shape != (n_models, n_models):
        raise ValueError("correlation matrix does not match effect vector")
    eigenvalues = np.linalg.eigvalsh(correlation)
    if eigenvalues.min() < -1e-10:
        raise ValueError("correlation matrix must be positive semidefinite")
    generator = np.random.default_rng(seed)
    rejections = 0
    completed = 0
    covariance = correlation * family_sd ** 2
    while completed < n_simulations:
        count = min(simulation_batch_size, n_simulations - completed)
        cube = generator.multivariate_normal(
            np.zeros(n_models), covariance,
            size=(count, n_families), check_valid="raise")
        if distribution == "student_t":
            if heavy_tail_df <= 2:
                raise ValueError("finite-variance Student t requires df > 2")
            scales = np.sqrt(
                (heavy_tail_df - 2)
                / generator.chisquare(
                    heavy_tail_df, size=(count, n_families, 1)))
            cube *= scales
        elif distribution != "normal":
            raise ValueError(distribution)
        cube += effects[None, None, :]
        pvalues = batch_max_t_pvalues(
            cube, signs,
            permutation_chunk_size=permutation_chunk_size)
        rejections += int(np.count_nonzero(pvalues <= alpha))
        completed += count
    rate = rejections / n_simulations
    return {
        "rejections": int(rejections),
        "n_simulations": int(n_simulations),
        "rejection_rate": float(rate),
        "monte_carlo_se": float(math.sqrt(rate * (1 - rate)
                                          / n_simulations)),
        "wilson_ci95": _wilson_interval(rejections, n_simulations),
        "n_families": int(n_families),
        "effects_nats": effects.tolist(),
        "family_sd_nats": float(family_sd),
        "distribution": distribution,
        "heavy_tail_df": (int(heavy_tail_df)
                          if distribution == "student_t" else None),
        "alpha": float(alpha),
        "permutation_draws": int(len(signs)),
    }


def derive_development_calibration(config: Mapping) -> dict:
    calibration = config["development_calibration"]
    frames = []
    sources = []
    for source in calibration["sources"]:
        path = Path(source["path"])
        actual_sha = file_sha256(path)
        if actual_sha != source["sha256"]:
            raise RuntimeError(
                f"development calibration hash mismatch for {source['label']}")
        event = resolve(source["evidence_id"])
        registered = {row["sha256"] for row in event["outputs"]}
        if actual_sha not in registered:
            raise RuntimeError(
                f"calibration source is not registered: {source['label']}")
        rows = pd.read_parquet(path)
        rows = rows[rows["bank"].eq(calibration["bank"])].copy()
        rows["specific"] = (
            rows["lp_meanJ_span_safe"] - rows["lp_ss_matched"])
        pivot = rows.pivot_table(
            index=["canonical_family", "fact_id"], columns="variant",
            values="specific", aggfunc="mean")
        for variant in calibration["variants"]:
            if variant not in pivot:
                raise RuntimeError(
                    f"calibration source lacks {variant}: {source['label']}")
        pivot[source["label"]] = pivot["composed"] - pivot["direct"]
        frames.append(pivot[[source["label"]]])
        sources.append({
            "label": source["label"],
            "evidence_id": source["evidence_id"],
            "path": str(path), "sha256": actual_sha,
        })
    common_items = pd.concat(frames, axis=1, join="inner").dropna()
    family_means = common_items.groupby(level="canonical_family").mean()
    if len(family_means) < 12:
        raise RuntimeError("too few common development families for calibration")
    standard_deviations = family_means.std(ddof=1)
    rounding = float(calibration["conservative_sd_rounding_nats"])
    conservative_sd = (
        math.ceil(float(standard_deviations.max()) / rounding - 1e-12)
        * rounding)
    return {
        "source_role": (
            "known-bank development variability proxy; no Bank W outcome "
            "is read"),
        "estimand": calibration["estimand"],
        "sources": sources,
        "n_common_items": int(len(common_items)),
        "n_common_families": int(len(family_means)),
        "family_sd_nats_by_source": {
            key: float(value) for key, value in standard_deviations.items()},
        "family_correlation_by_source": {
            row: {column: float(value)
                  for column, value in values.items()}
            for row, values in family_means.corr().to_dict(
                orient="index").items()},
        "conservative_common_sd_nats": float(conservative_sd),
        "conservative_rule": (
            "largest common-family SD across registered development sources, "
            f"rounded upward to {rounding:g} nat"),
    }


def _correlation_matrix(calibration: Mapping) -> np.ndarray:
    labels = list(calibration["family_sd_nats_by_source"])
    return np.asarray([
        [calibration["family_correlation_by_source"][row][column]
         for column in labels] for row in labels], dtype=np.float64)


def run_simulation(config: Mapping) -> dict:
    randomization = config["primary_randomization"]
    simulation = config["simulation"]
    model_slugs = list(config["target_model_slugs"])
    n_models = len(model_slugs)
    if n_models != 3:
        raise RuntimeError("P4-P3 max-T is frozen across exactly three models")
    calibration = derive_development_calibration(config)
    empirical = _correlation_matrix(calibration)
    independent = np.eye(n_models, dtype=np.float64)
    family_sd = float(calibration["conservative_common_sd_nats"])
    n_simulations = int(simulation["n_simulations"])
    permutation_draws = int(simulation["permutation_draws"])
    alpha = float(randomization["alpha"])
    root_seed = int(simulation["seed"])
    df = int(simulation["heavy_tail_df"])
    batch_size = int(simulation["simulation_batch_size"])
    permutation_chunk = int(simulation["permutation_chunk_size"])
    signs_by_n = {
        int(n): _random_signs(
            permutation_draws, int(n), stable_seed(root_seed, "signs", n))
        for n in simulation["family_counts"]}

    def scenario(*, name: str, n_families: int, effects: list[float],
                 correlation_name: str, distribution: str) -> dict:
        correlation = (independent if correlation_name == "independent"
                       else empirical)
        result = simulate_rejection_rate(
            n_simulations=n_simulations, n_families=n_families,
            effects=np.asarray(effects), family_sd=family_sd,
            correlation=correlation, distribution=distribution,
            heavy_tail_df=df, signs=signs_by_n[n_families],
            seed=stable_seed(root_seed, "scenario", name), alpha=alpha,
            simulation_batch_size=batch_size,
            permutation_chunk_size=permutation_chunk)
        result.update({
            "scenario": name,
            "correlation": correlation_name,
            "correlation_matrix": correlation.tolist(),
        })
        return result

    confirmatory_n = int(simulation["confirmatory_families"])
    type_i = []
    for correlation_name in ("independent", "empirical"):
        for distribution in ("normal", "student_t"):
            type_i.append(scenario(
                name=f"null-{correlation_name}-{distribution}",
                n_families=confirmatory_n, effects=[0.0] * n_models,
                correlation_name=correlation_name,
                distribution=distribution))
    bounds = [float(value) for value in
              simulation["type_i_acceptance_interval"]]
    for row in type_i:
        row["type_i_pass"] = bool(
            bounds[0] <= row["rejection_rate"] <= bounds[1])
        row["nominal_alpha_in_wilson_ci95"] = bool(
            row["wilson_ci95"][0] <= alpha <= row["wilson_ci95"][1])

    low = float(simulation["low_load"])
    high = float(simulation["high_load"])
    slope_sesoi = float(simulation["sesoi_slope_nats_per_doubling"])
    endpoint_sesoi = slope_sesoi * math.log2(high / low)
    power_at_sesoi = []
    for n_families in map(int, simulation["family_counts"]):
        for model_index in range(n_models):
            effects = [0.0] * n_models
            effects[model_index] = endpoint_sesoi
            power_at_sesoi.append(scenario(
                name=f"sesoi-single-model-{model_index}-n{n_families}",
                n_families=n_families, effects=effects,
                correlation_name="independent", distribution="student_t"))
    for model_index in range(n_models):
        effects = [0.0] * n_models
        effects[model_index] = endpoint_sesoi
        power_at_sesoi.append(scenario(
            name=f"sesoi-normal-single-model-{model_index}-n{confirmatory_n}",
            n_families=confirmatory_n, effects=effects,
            correlation_name="independent", distribution="normal"))
    power_at_sesoi.append(scenario(
        name=f"sesoi-all-models-n{confirmatory_n}",
        n_families=confirmatory_n,
        effects=[endpoint_sesoi] * n_models,
        correlation_name="independent", distribution="student_t"))

    effect_grid = sorted(set(
        [float(value) for value in simulation["effect_grid_nats"]]
        + [endpoint_sesoi]))
    power_curve = []
    for effect in effect_grid:
        power_curve.append(scenario(
            name=f"grid-single-model-0-effect-{effect:.6f}",
            n_families=confirmatory_n,
            effects=[effect, 0.0, 0.0],
            correlation_name="independent", distribution="student_t"))

    power_target = float(simulation["power_target"])
    minimum_power_by_n = {}
    for n_families in map(int, simulation["family_counts"]):
        rows = [row for row in power_at_sesoi
                if row["n_families"] == n_families
                and row["distribution"] == "student_t"
                and row["scenario"].startswith("sesoi-single-model")]
        minimum_power_by_n[str(n_families)] = min(
            row["rejection_rate"] for row in rows)
    minimum_common = next((
        int(n) for n in simulation["family_counts"]
        if minimum_power_by_n[str(n)] >= power_target), None)
    mde80 = next((
        float(row["effects_nats"][0]) for row in power_curve
        if row["rejection_rate"] >= power_target), None)
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "outcome_blinding": (
            "No Bank W intervention, capability, confirmatory, or replication "
            "outcome was read."),
        "calibration": calibration,
        "primary": {
            **dict(randomization),
            "model_slugs_in_frozen_order": model_slugs,
            "statistic": "max_m mean(D_m)/(sd(D_m)/sqrt(n_families))",
            "family_pairing": (
                "one shared sign per canonical family across all models"),
            "model_selection_from_intervention_outcomes": False,
        },
        "sesoi": {
            "slope_nats_per_doubling": slope_sesoi,
            "low_load": low, "high_load": high,
            "endpoint_high_minus_low_nats": endpoint_sesoi,
        },
        "simulation": {
            "n_simulations_per_scenario": n_simulations,
            "permutation_draws_per_simulation": permutation_draws,
            "seed": root_seed, "power_target": power_target,
            "type_i_acceptance_interval": bounds,
            "heavy_tail_df": df,
        },
        "type_i_calibration": type_i,
        "power_at_sesoi": power_at_sesoi,
        "power_curve_single_model_conservative": power_curve,
        "decision": {
            "all_type_i_scenarios_pass": all(
                row["type_i_pass"] for row in type_i),
            "minimum_power_at_sesoi_by_common_family_count":
                minimum_power_by_n,
            "minimum_common_families_for_power_target": minimum_common,
            "confirmatory_family_count": confirmatory_n,
            "confirmatory_count_meets_power_target": bool(
                minimum_power_by_n[str(confirmatory_n)] >= power_target),
            "grid_bounded_mde80_nats": mde80,
            "candidate_freeze_ready": False,
            "remaining_freeze_blockers": [
                "model-specific Bank W baseline capability gates",
                "independent protocol review and PI sign-off",
            ],
        },
    }


def make_figure(result: Mapping, png_path: Path, pdf_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = result["power_curve_single_model_conservative"]
    by_n = result["decision"][
        "minimum_power_at_sesoi_by_common_family_count"]
    target = float(result["simulation"]["power_target"])
    endpoint = float(result["sesoi"]["endpoint_high_minus_low_nats"])
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(
        [row["effects_nats"][0] for row in curve],
        [row["rejection_rate"] for row in curve],
        color="#315a9a", marker="o", linewidth=2)
    axes[0].axhline(target, color="#9b2c2c", linestyle="--", linewidth=1.4,
                    label=f"power target {target:.0%}")
    axes[0].axvline(endpoint, color="#447a45", linestyle=":", linewidth=1.8,
                    label=f"SESOI {endpoint:.3f} nat")
    axes[0].set(xlabel="True high-minus-low effect (nat)",
                ylabel="Joint max-T power", ylim=(0, 1.02),
                title="Conservative one-model alternative")
    axes[0].legend(frameon=False, fontsize=9)
    family_counts = [int(value) for value in by_n]
    axes[1].plot(
        family_counts, [by_n[str(value)] for value in family_counts],
        color="#7a4e9b", marker="s", linewidth=2)
    axes[1].axhline(target, color="#9b2c2c", linestyle="--", linewidth=1.4)
    axes[1].set(xlabel="Common canonical families",
                ylabel="Minimum power over active model", ylim=(0, 1.02),
                xticks=family_counts,
                title="Power at the frozen SESOI")
    figure.suptitle(
        "Bank W shared-family max-T calibration\n"
        "independent models, symmetric Student-t(5) family effects",
        fontsize=12)
    figure.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        png_path, dpi=180, bbox_inches="tight",
        metadata={"Software": "jspace_phase4.p4_bank_w_power"})
    figure.savefig(
        pdf_path, bbox_inches="tight",
        metadata={"Creator": "jspace_phase4.p4_bank_w_power",
                  "CreationDate": None, "ModDate": None})
    plt.close(figure)


def _generation_protocol_unchanged(
        generation_commit: str, config_path: Path) -> bool:
    """Allow an output-only commit without allowing protocol drift."""
    repository = _find_repo_root()
    module_path = Path(__file__).resolve().relative_to(repository)
    config_relative = config_path.resolve().relative_to(repository)
    ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor",
         generation_commit, "HEAD"], check=False).returncode == 0
    unchanged = subprocess.run(
        ["git", "-C", str(repository), "diff", "--quiet",
         generation_commit, "HEAD", "--", str(module_path),
         str(config_relative)], check=False).returncode == 0
    return ancestor and unchanged


def registration_summary(decision: Mapping) -> str:
    """Describe a power verdict without rounding a near miss into a pass."""
    confirmatory_count = int(decision["confirmatory_family_count"])
    confirmatory_power = decision[
        "minimum_power_at_sesoi_by_common_family_count"][
            str(confirmatory_count)]
    return (
        "Outcome-blind Bank W shared-family max-T calibration: all "
        f"type-I scenarios pass; minimum conservative power at "
        f"n={confirmatory_count} is {confirmatory_power:.4f} "
        f"(target met: "
        f"{decision['confirmatory_count_meets_power_target']}); "
        f"minimum powered common-family count is "
        f"{decision['minimum_common_families_for_power_target']}.")


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    outputs = {key: Path(value) for key, value in config["outputs"].items()}
    if arguments.simulate:
        result = run_simulation(config)
        result.update({
            "generation_code_commit": clean["code_commit"],
            "config_sha256": file_sha256(config_path),
            "bank_w_audit_sha256": file_sha256(config["bank_w_audit"]),
        })
        atomic_json(outputs["result"], result)
        make_figure(result, outputs["figure_png"], outputs["figure_pdf"])
        print(json.dumps({
            "status": "simulated-unregistered",
            "decision": result["decision"],
            "type_i": [row["rejection_rate"]
                       for row in result["type_i_calibration"]],
        }, indent=1))
        return
    required = [outputs[key] for key in
                ("result", "figure_png", "figure_pdf")]
    if not all(path.exists() for path in required):
        raise RuntimeError("Bank W power outputs are incomplete")
    result = json.loads(outputs["result"].read_text())
    if not _generation_protocol_unchanged(
            result["generation_code_commit"], config_path):
        raise RuntimeError(
            "Bank W power protocol changed after output generation")
    decision = result["decision"]
    inputs = {
        "config": file_sha256(config_path),
        "bank_w_audit": file_sha256(config["bank_w_audit"]),
    }
    inputs.update({
        f"development_calibration:{source['evidence_id']}": source["sha256"]
        for source in config["development_calibration"]["sources"]})
    command = (
        "python -m jspace_phase4.experiments.p4_bank_w_power "
        f"--config {arguments.config} --register-existing")
    create(
        config["evidence_id"], tier=config["tier"], command=command,
        what=registration_summary(decision),
        outputs=required, inputs=inputs)
    print(json.dumps({
        "status": "registered", "evidence_id": config["evidence_id"]},
        indent=1))


if __name__ == "__main__":
    main()
