"""Plan a future two-OLMo Bank-W primary without opening outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths import resolve_uri
from ..registry import create, resolve

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]


def stable_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _studentized_from_means(
    means: np.ndarray,
    sum_squares: np.ndarray,
    n: int,
) -> np.ndarray:
    centered = np.maximum(sum_squares - n * np.square(means), 0.0)
    standard_errors = np.sqrt(centered / (n - 1) / n)
    return np.divide(
        means,
        standard_errors,
        out=np.where(means > 0, np.inf, np.where(means < 0, -np.inf, 0.0)),
        where=standard_errors > 1e-14,
    )


def shared_family_max_t(
    values: np.ndarray,
    *,
    model_slugs: list[str],
    draws: int = 100_000,
    seed: int = 48_151_623,
    exact_max_families: int = 20,
    chunk_size: int = 8192,
) -> dict:
    """One-sided max-T with a shared family sign across both models."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] != 2:
        raise ValueError("pair max-T requires family x exactly-two-model values")
    if not np.isfinite(matrix).all():
        raise ValueError("pair max-T values must be finite")
    if len(model_slugs) != 2 or len(set(model_slugs)) != 2:
        raise ValueError("two unique model slugs are required")
    if np.any(np.std(matrix, axis=0, ddof=1) <= 1e-14):
        raise ValueError("each model column must have nonzero variance")
    n_families = matrix.shape[0]
    means = matrix.mean(axis=0)
    sums = np.square(matrix).sum(axis=0)
    observed_t = _studentized_from_means(means, sums, n_families)
    observed_max = float(np.max(observed_t))
    exact = n_families <= exact_max_families
    patterns = 2**n_families if exact else int(draws)
    generator = np.random.default_rng(seed)
    extreme = 0
    digest = hashlib.sha256()
    for start in range(0, patterns, chunk_size):
        count = min(chunk_size, patterns - start)
        if exact:
            integers = np.arange(start, start + count, dtype=np.uint64)[:, None]
            bits = ((integers >> np.arange(n_families, dtype=np.uint64)) & 1).astype(
                np.int8
            )
            signs = 1 - 2 * bits
        else:
            signs = (
                2 * generator.integers(0, 2, size=(count, n_families), dtype=np.int8)
                - 1
            )
        permuted_means = (signs @ matrix) / n_families
        permuted_t = _studentized_from_means(permuted_means, sums, n_families)
        null_max = np.max(permuted_t, axis=1)
        digest.update(np.asarray(null_max, dtype="<f8").tobytes())
        extreme += int(np.count_nonzero(null_max >= observed_max - 1e-14))
    p_value = extreme / patterns if exact else (extreme + 1) / (patterns + 1)
    return {
        "estimate_by_model": dict(zip(model_slugs, map(float, means))),
        "t_by_model": dict(zip(model_slugs, map(float, observed_t))),
        "observed_max_t": observed_max,
        "p": float(p_value),
        "n_families": int(n_families),
        "n_models_jointly_tested": 2,
        "method": (
            "exact-shared-family-signflip-max-t"
            if exact
            else "monte-carlo-shared-family-signflip-max-t-plus-one"
        ),
        "n_patterns_or_draws": int(patterns),
        "null_max_t_sha256": digest.hexdigest(),
    }


def _random_signs(draws: int, families: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return 2 * generator.integers(0, 2, size=(draws, families), dtype=np.int8) - 1


def batch_max_t_pvalues(
    values: np.ndarray,
    signs: np.ndarray,
    *,
    permutation_chunk_size: int = 512,
) -> np.ndarray:
    cube = np.asarray(values, dtype=np.float64)
    sign_matrix = np.asarray(signs, dtype=np.int8)
    if cube.ndim != 3 or cube.shape[2] != 2:
        raise ValueError("simulation values must be simulation x family x 2")
    n_simulations, n_families, _ = cube.shape
    if sign_matrix.ndim != 2 or sign_matrix.shape[1] != n_families:
        raise ValueError("sign matrix family dimension does not match")
    means = cube.mean(axis=1)
    sum_squares = np.square(cube).sum(axis=1)
    observed = np.max(_studentized_from_means(means, sum_squares, n_families), axis=1)
    extreme = np.zeros(n_simulations, dtype=np.int64)
    for start in range(0, len(sign_matrix), permutation_chunk_size):
        selected = sign_matrix[start : start + permutation_chunk_size]
        permuted_means = (
            np.einsum("bf,sfm->sbm", selected, cube, optimize=True) / n_families
        )
        permuted_t = _studentized_from_means(
            permuted_means, sum_squares[:, None, :], n_families
        )
        null_max = np.max(permuted_t, axis=2)
        extreme += np.count_nonzero(null_max >= observed[:, None] - 1e-14, axis=1)
    return (extreme + 1) / (len(sign_matrix) + 1)


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def simulate_rejection_rate(
    *,
    n_simulations: int,
    n_families: int,
    effects: np.ndarray,
    family_sd: float,
    correlation: np.ndarray,
    distribution: str,
    heavy_tail_df: int,
    signs: np.ndarray,
    seed: int,
    alpha: float,
    simulation_batch_size: int,
    permutation_chunk_size: int,
) -> dict:
    effects = np.asarray(effects, dtype=np.float64)
    correlation = np.asarray(correlation, dtype=np.float64)
    if effects.shape != (2,) or correlation.shape != (2, 2):
        raise ValueError("pair simulation requires two effects and 2x2 correlation")
    if np.linalg.eigvalsh(correlation).min() < -1e-10:
        raise ValueError("correlation matrix must be positive semidefinite")
    generator = np.random.default_rng(seed)
    covariance = correlation * family_sd**2
    rejections = 0
    completed = 0
    while completed < n_simulations:
        count = min(simulation_batch_size, n_simulations - completed)
        cube = generator.multivariate_normal(
            np.zeros(2), covariance, size=(count, n_families), check_valid="raise"
        )
        if distribution == "student_t":
            if heavy_tail_df <= 2:
                raise ValueError("finite-variance Student t requires df > 2")
            cube *= np.sqrt(
                (heavy_tail_df - 2)
                / generator.chisquare(heavy_tail_df, size=(count, n_families, 1))
            )
        elif distribution != "normal":
            raise ValueError(distribution)
        cube += effects[None, None, :]
        pvalues = batch_max_t_pvalues(
            cube, signs, permutation_chunk_size=permutation_chunk_size
        )
        rejections += int(np.count_nonzero(pvalues <= alpha))
        completed += count
    rate = rejections / n_simulations
    return {
        "rejections": int(rejections),
        "n_simulations": int(n_simulations),
        "rejection_rate": float(rate),
        "monte_carlo_se": float(math.sqrt(rate * (1 - rate) / n_simulations)),
        "wilson_ci95": _wilson(rejections, n_simulations),
        "n_families": int(n_families),
        "effects_nats": effects.tolist(),
        "family_sd_nats": float(family_sd),
        "distribution": distribution,
        "heavy_tail_df": heavy_tail_df if distribution == "student_t" else None,
        "alpha": float(alpha),
        "permutation_draws": len(signs),
    }


def _registered_creation(path: Path, evidence_id: str) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    matches = [
        row
        for row in rows
        if row.get("event") in {"evidence_created", "evidence_imported"}
        and row.get("evidence_id") == evidence_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"source registry lacks one origin for {evidence_id}")
    return matches[0]


def _verify_variance_ruler(config: Mapping) -> dict:
    specification = config["registered_variance_ruler"]
    result_path = REPO_ROOT / specification["result_path"]
    config_path = REPO_ROOT / specification["config_path"]
    registry_path = REPO_ROOT / specification["registry_path"]
    if file_sha256(result_path) != specification["result_sha256"]:
        raise RuntimeError("registered variance-ruler result hash drift")
    if file_sha256(config_path) != specification["config_sha256"]:
        raise RuntimeError("registered variance-ruler config hash drift")
    event = _registered_creation(registry_path, specification["evidence_id"])
    if specification["result_sha256"] not in {
        row.get("sha256") for row in event.get("outputs", [])
    }:
        raise RuntimeError("variance-ruler result is not registered")
    result = json.loads(result_path.read_text())
    calibration = result["calibration"]
    if (
        calibration["conservative_common_sd_nats"]
        != specification["conservative_common_sd_nats"]
    ):
        raise RuntimeError("conservative SD differs from frozen import")
    if calibration["conservative_rule"] != specification["conservative_rule"]:
        raise RuntimeError("conservative SD rule differs from frozen import")
    return {
        "evidence_id": specification["evidence_id"],
        "result_sha256": specification["result_sha256"],
        "config_sha256": specification["config_sha256"],
        "conservative_common_sd_nats": calibration["conservative_common_sd_nats"],
        "conservative_rule": calibration["conservative_rule"],
        "family_correlation_by_source": calibration["family_correlation_by_source"],
    }


def _load_capability(specification: Mapping) -> dict:
    event = resolve(specification["evidence_id"])
    if not event["live"]:
        raise RuntimeError(f"capability evidence is not live: {event['evidence_id']}")
    path = Path(specification["result_path"])
    if file_sha256(path) != specification["result_sha256"]:
        raise RuntimeError(f"capability result hash drift: {path}")
    registered = {
        (row["path"], row["sha256"])
        for row in event["effective_metadata"].get("outputs", [])
    }
    if (str(path), specification["result_sha256"]) not in registered:
        raise RuntimeError("capability result path/hash is not registered")
    envelope = json.loads(path.read_text())
    if envelope.get("payload_sha256") != object_sha256(envelope.get("payload")):
        raise RuntimeError("capability result envelope hash drift")
    analysis = envelope["payload"]["analysis"]
    return {
        "slug": specification["slug"],
        "evidence_id": specification["evidence_id"],
        "result_sha256": specification["result_sha256"],
        "independently_capability_eligible": bool(
            analysis["independently_capability_eligible"]
        ),
        "n_capable_families": int(analysis["n_capable_families"]),
        "capable_family_ids": sorted(analysis["capable_family_ids"]),
    }


def derive_pair_support(config: Mapping) -> dict:
    rows = [_load_capability(row) for row in config["target_models"]]
    common = sorted(
        set(rows[0]["capable_family_ids"]) & set(rows[1]["capable_family_ids"])
    )
    return {
        "models": rows,
        "all_independently_capability_eligible": all(
            row["independently_capability_eligible"] for row in rows
        ),
        "shared_capable_family_ids": common,
        "shared_capable_family_ids_sha256": object_sha256(common),
        "n_shared_capable_families": len(common),
    }


def pair_decision(
    *,
    all_independently_eligible: bool,
    all_type_i_pass: bool,
    shared_families: int,
    minimum_powered_families: int | None,
    power_at_shared_support: float,
    power_target: float,
) -> dict:
    worthwhile = bool(
        all_independently_eligible
        and all_type_i_pass
        and minimum_powered_families is not None
        and shared_families >= minimum_powered_families
        and power_at_shared_support >= power_target
    )
    return {
        "future_pair_worthwhile_at_current_support": worthwhile,
        "route": (
            "powered-planning-candidate-no-intervention-authority"
            if worthwhile
            else "not-powered-at-current-support"
        ),
        "minimum_common_families_for_power_target": minimum_powered_families,
        "power_at_shared_capable_support": power_at_shared_support,
        "power_target": power_target,
        "intervention_authorized": False,
    }


def run_simulation(config: Mapping) -> dict:
    support = derive_pair_support(config)
    ruler = _verify_variance_ruler(config)
    simulation = config["simulation"]
    randomization = config["primary_randomization"]
    slugs = [row["slug"] for row in config["target_models"]]
    labels = [row["source_label"] for row in config["target_models"]]
    correlation_rows = ruler["family_correlation_by_source"]
    empirical = np.asarray(
        [[correlation_rows[a][b] for b in labels] for a in labels],
        dtype=np.float64,
    )
    independent = np.eye(2, dtype=np.float64)
    family_sd = float(ruler["conservative_common_sd_nats"])
    n_simulations = int(simulation["n_simulations"])
    permutation_draws = int(simulation["permutation_draws"])
    alpha = float(randomization["alpha"])
    root_seed = int(simulation["seed"])
    df = int(simulation["heavy_tail_df"])
    batch_size = int(simulation["simulation_batch_size"])
    permutation_chunk = int(simulation["permutation_chunk_size"])
    observed_support = int(support["n_shared_capable_families"])
    family_counts = sorted(
        set(map(int, simulation["family_counts"])) | {observed_support}
    )
    if observed_support < 3:
        raise RuntimeError("too few shared capable families for pair planning")
    signs_by_n = {
        count: _random_signs(
            permutation_draws, count, stable_seed(root_seed, "signs", count)
        )
        for count in family_counts
    }

    def scenario(
        *,
        name: str,
        n_families: int,
        effects: list[float],
        correlation_name: str,
        distribution: str,
    ) -> dict:
        correlation = independent if correlation_name == "independent" else empirical
        result = simulate_rejection_rate(
            n_simulations=n_simulations,
            n_families=n_families,
            effects=np.asarray(effects),
            family_sd=family_sd,
            correlation=correlation,
            distribution=distribution,
            heavy_tail_df=df,
            signs=signs_by_n[n_families],
            seed=stable_seed(root_seed, "scenario", name),
            alpha=alpha,
            simulation_batch_size=batch_size,
            permutation_chunk_size=permutation_chunk,
        )
        result.update(
            {
                "scenario": name,
                "correlation": correlation_name,
                "correlation_matrix": correlation.tolist(),
            }
        )
        return result

    type_i = []
    for correlation_name in ("independent", "empirical"):
        for distribution in ("normal", "student_t"):
            type_i.append(
                scenario(
                    name=f"null-{correlation_name}-{distribution}",
                    n_families=observed_support,
                    effects=[0.0, 0.0],
                    correlation_name=correlation_name,
                    distribution=distribution,
                )
            )
    bounds = list(map(float, simulation["type_i_acceptance_interval"]))
    for row in type_i:
        row["type_i_pass"] = bool(bounds[0] <= row["rejection_rate"] <= bounds[1])
        row["nominal_alpha_in_wilson_ci95"] = bool(
            row["wilson_ci95"][0] <= alpha <= row["wilson_ci95"][1]
        )

    low = float(simulation["low_load"])
    high = float(simulation["high_load"])
    slope_sesoi = float(simulation["sesoi_slope_nats_per_doubling"])
    endpoint_sesoi = slope_sesoi * math.log2(high / low)
    power_rows = [
        scenario(
            name=f"sesoi-one-active-model-n{count}",
            n_families=count,
            effects=[endpoint_sesoi, 0.0],
            correlation_name="independent",
            distribution="student_t",
        )
        for count in family_counts
    ]
    power_by_n = {str(row["n_families"]): row["rejection_rate"] for row in power_rows}
    power_target = float(simulation["power_target"])
    minimum_powered = next(
        (count for count in family_counts if power_by_n[str(count)] >= power_target),
        None,
    )
    effect_grid = sorted(
        set(map(float, simulation["effect_grid_nats"])) | {endpoint_sesoi}
    )
    power_curve = [
        scenario(
            name=f"grid-one-active-model-effect-{effect:.9f}",
            n_families=observed_support,
            effects=[effect, 0.0],
            correlation_name="independent",
            distribution="student_t",
        )
        for effect in effect_grid
    ]
    decision = pair_decision(
        all_independently_eligible=support["all_independently_capability_eligible"],
        all_type_i_pass=all(row["type_i_pass"] for row in type_i),
        shared_families=observed_support,
        minimum_powered_families=minimum_powered,
        power_at_shared_support=power_by_n[str(observed_support)],
        power_target=power_target,
    )
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "outcome_blinding": (
            "No Bank-W intervention, confirmatory, or replication outcome was read."
        ),
        "capability_support": support,
        "registered_variance_ruler": ruler,
        "primary": {
            **dict(randomization),
            "model_slugs_in_frozen_order": slugs,
            "family_pairing": "one shared sign per canonical family across models",
            "single_active_model_power_is_column-symmetric_under_frozen_independent_covariance": True,
        },
        "sesoi": {
            "slope_nats_per_doubling": slope_sesoi,
            "low_load": low,
            "high_load": high,
            "endpoint_high_minus_low_nats": endpoint_sesoi,
        },
        "simulation": {
            "n_simulations_per_scenario": n_simulations,
            "permutation_draws_per_simulation": permutation_draws,
            "seed": root_seed,
            "family_counts": family_counts,
            "power_target": power_target,
            "type_i_acceptance_interval": bounds,
            "heavy_tail_df": df,
        },
        "type_i_calibration": type_i,
        "power_at_sesoi": power_rows,
        "power_curve_at_observed_support": power_curve,
        "decision": decision,
        "claim_boundary": config["claim_boundary"],
    }


def make_figure(result: Mapping, png_path: Path, pdf_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = result["power_curve_at_observed_support"]
    power = result["power_at_sesoi"]
    target = float(result["simulation"]["power_target"])
    endpoint = float(result["sesoi"]["endpoint_high_minus_low_nats"])
    support = int(result["capability_support"]["n_shared_capable_families"])
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(
        [row["effects_nats"][0] for row in curve],
        [row["rejection_rate"] for row in curve],
        color="#315a9a",
        marker="o",
        linewidth=2,
    )
    axes[0].axhline(target, color="#9b2c2c", linestyle="--", linewidth=1.4)
    axes[0].axvline(endpoint, color="#447a45", linestyle=":", linewidth=1.8)
    axes[0].set(
        xlabel="True high-minus-low effect (nat)",
        ylabel="Two-model max-T power",
        ylim=(0, 1.02),
        title=f"Observed shared support: n={support}",
    )
    axes[1].plot(
        [row["n_families"] for row in power],
        [row["rejection_rate"] for row in power],
        color="#7a4e9b",
        marker="s",
        linewidth=2,
    )
    axes[1].axhline(target, color="#9b2c2c", linestyle="--", linewidth=1.4)
    axes[1].axvline(support, color="#447a45", linestyle=":", linewidth=1.8)
    axes[1].set(
        xlabel="Common capable families",
        ylabel="Conservative one-active-model power",
        ylim=(0, 1.02),
        title="Power at the frozen SESOI",
    )
    figure.suptitle(
        "Future OLMo Think/Instruct Bank-W pair planning\n"
        "shared-family max-T; no intervention authorization",
        fontsize=12,
    )
    figure.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        png_path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "jspace_olmo_lineage.bank_w_pair_power"},
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Creator": "jspace_olmo_lineage.bank_w_pair_power",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def _generation_protocol_unchanged(
    generation_commit: str,
    config_path: Path,
) -> bool:
    module_path = Path(__file__).resolve().relative_to(REPO_ROOT)
    config_relative = config_path.resolve().relative_to(REPO_ROOT)
    ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                generation_commit,
                "HEAD",
            ],
            check=False,
        ).returncode
        == 0
    )
    unchanged = (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "diff",
                "--quiet",
                generation_commit,
                "HEAD",
                "--",
                str(module_path),
                str(config_relative),
            ],
            check=False,
        ).returncode
        == 0
    )
    return ancestor and unchanged


def _outputs(config: Mapping) -> dict[str, Path]:
    return {
        name: resolve_uri(uri, must_exist=False)
        for name, uri in config["outputs"].items()
    }


def simulate(config_path: Path, config: Mapping) -> dict:
    clean = require_clean_tree(expected_branch=config["branch"])
    outputs = _outputs(config)
    occupied = [str(path) for path in outputs.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"unregistered pair-power output exists: {occupied}")
    result = run_simulation(config)
    result.update(
        {
            "generation_code_commit": clean["code_commit"],
            "config_sha256": file_sha256(config_path),
        }
    )
    atomic_json(outputs["result"], result)
    make_figure(result, outputs["figure_png"], outputs["figure_pdf"])
    return {"status": "simulated-unregistered", "decision": result["decision"]}


def register_existing(config_path: Path, config: Mapping) -> dict:
    require_clean_tree(expected_branch=config["branch"])
    outputs = _outputs(config)
    for path in outputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    result = json.loads(outputs["result"].read_text())
    if result["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("pair-power config hash drift after simulation")
    if not _generation_protocol_unchanged(
        result["generation_code_commit"], config_path
    ):
        raise RuntimeError("pair-power protocol changed after simulation")
    if result["capability_support"] != derive_pair_support(config):
        raise RuntimeError("pair capability support drift after simulation")
    ruler = _verify_variance_ruler(config)
    if result["registered_variance_ruler"] != ruler:
        raise RuntimeError("pair variance ruler drift after simulation")
    decision = result["decision"]
    support = result["capability_support"]
    command = (
        "python -m jspace_olmo_lineage.experiments.bank_w_pair_power "
        f"--config {config_path} --register-existing"
    )
    event = create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "Outcome-blind OLMo Think/Instruct Bank-W pair planning: "
            f"{support['n_shared_capable_families']} shared capable families, "
            f"power {decision['power_at_shared_capable_support']:.4f} at the "
            f"frozen SESOI, route {decision['route']}; no intervention authority."
        ),
        command=command,
        outputs=list(outputs.values()),
        inputs={
            "config": file_sha256(config_path),
            "registered_variance_ruler": ruler["result_sha256"],
            **{
                f"capability:{row['evidence_id']}": row["result_sha256"]
                for row in support["models"]
            },
        },
        interventions_opened=False,
        bank_w_outcomes_opened=False,
        future_pair_worthwhile_at_current_support=decision[
            "future_pair_worthwhile_at_current_support"
        ],
        n_shared_capable_families=support["n_shared_capable_families"],
        power_at_shared_capable_support=decision["power_at_shared_capable_support"],
        minimum_common_families_for_power_target=decision[
            "minimum_common_families_for_power_target"
        ],
        route=decision["route"],
        claim_boundary=config["claim_boundary"],
    )
    return {"status": "registered", "event": event}


def verify(config: Mapping) -> dict:
    outputs = _outputs(config)
    event = resolve(config["evidence_id"])
    if not event["live"] or event["effective_tier"] != config["tier"]:
        raise RuntimeError("pair-power event is not live methods evidence")
    registered = {
        row["path"]: (row["sha256"], int(row["bytes"]))
        for row in event["effective_metadata"].get("outputs", [])
    }
    for path in outputs.values():
        expected = registered.get(str(path))
        if expected is None:
            raise RuntimeError(f"pair-power output is not registered: {path}")
        if expected != (file_sha256(path), path.stat().st_size):
            raise RuntimeError(f"pair-power output drift: {path}")
    result = json.loads(outputs["result"].read_text())
    return {
        "ok": True,
        "evidence_id": config["evidence_id"],
        "decision": result["decision"],
        "outputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in outputs.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--simulate", action="store_true")
    action.add_argument("--register-existing", action="store_true")
    action.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    if arguments.simulate:
        result = simulate(config_path, config)
    elif arguments.register_existing:
        result = register_existing(config_path, config)
    else:
        result = verify(config)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
