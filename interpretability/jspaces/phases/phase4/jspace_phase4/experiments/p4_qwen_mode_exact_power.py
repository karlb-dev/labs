"""Exact/Monte Carlo sign-flip power ruler for the P4-P2 accuracy endpoint.

The source is the consumed-family variance pilot.  Its signed mean is a
development outcome and is therefore centered away before any planning
calculation.  The retained information is the reflection-invariant residual
shape and the prospectively registered conservative planning SD.  Small
family counts use complete sign enumeration; larger counts use a fixed
Monte Carlo sign matrix with plus-one p-values.
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
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import resolve_uri
from ..registry4 import create, resolve
from .p4_bank_w_power import _wilson_interval, stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--simulate", action="store_true")
    action.add_argument("--register-existing", action="store_true")
    return parser.parse_args()


def _complete_sign_matrix(width: int) -> np.ndarray:
    if not 1 <= width <= 22:
        raise ValueError("complete Rademacher width must lie in 1..22")
    patterns = np.arange(2 ** width, dtype=np.uint64)[:, None]
    bits = ((patterns >> np.arange(
        width, dtype=np.uint64)) & 1).astype(np.int8)
    return 1 - 2 * bits


def sign_matrix(
        n_families: int, *, draws: int, seed: int,
        exact_max_families: int) -> tuple[np.ndarray, bool]:
    """Return complete or fixed Monte Carlo Rademacher sign patterns."""
    if n_families < 3:
        raise ValueError("family sign flip requires at least three families")
    if not 3 <= exact_max_families <= 22:
        raise ValueError("exact sign-flip ceiling must lie in 3..22")
    if n_families <= exact_max_families:
        return _complete_sign_matrix(n_families), True
    if draws < 1:
        raise ValueError("Monte Carlo sign draws must be positive")
    generator = np.random.default_rng(seed)
    return 2 * generator.integers(
        0, 2, size=(draws, n_families), dtype=np.int8) - 1, False


def batch_signflip_pvalues(
        values: np.ndarray, signs: np.ndarray, *, exact: bool,
        chunk_size: int = 512) -> np.ndarray:
    """One-sided equal-family sign-flip p-values for simulation rows.

    The unstudentized mean and the usual studentized mean have the same
    ordering over sign patterns because the sum of squares is invariant and
    the studentized statistic is monotone in the signed mean.  Using sums here
    is therefore the exact frozen mean-test randomization calculation while
    avoiding unstable divisions at degenerate simulated samples.
    """
    matrix = np.asarray(values, dtype=np.float64)
    sign_array = np.asarray(signs, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError("values must be simulation x family")
    if not np.isfinite(matrix).all():
        raise ValueError("sign-flip values must be finite")
    if sign_array.ndim != 2 or sign_array.shape[1] != matrix.shape[1]:
        raise ValueError("sign matrix family dimension mismatch")
    if len(sign_array) < 1 or chunk_size < 1:
        raise ValueError("sign patterns and chunk size must be positive")
    observed_sums = matrix.sum(axis=1)
    if exact:
        expected_patterns = 2 ** matrix.shape[1]
        if len(sign_array) != expected_patterns:
            raise ValueError(
                "exact sign matrix does not contain the complete pattern count")
        if np.any((sign_array != -1) & (sign_array != 1)):
            raise ValueError("exact sign matrix contains non-Rademacher values")
        # Complete enumeration at n=20 has 1,048,576 rows.  Repeating a dense
        # full-pattern matrix product for every simulation would be wasteful.
        # Meet in the middle counts all 2^n signed sums exactly using two
        # matrices of at most 2^10 rows and a binary search over their sums.
        split = matrix.shape[1] // 2
        left_signs = _complete_sign_matrix(split)
        right_width = matrix.shape[1] - split
        right_signs = _complete_sign_matrix(right_width)
        left_sums = np.asarray(left_signs, dtype=np.float64) @ matrix[
            :, :split].T
        right_sums = np.asarray(right_signs, dtype=np.float64) @ matrix[
            :, split:].T
        right_sums.sort(axis=0)
        tolerance = 1e-13 * np.maximum(1.0, np.abs(observed_sums))
        counts = np.empty(len(matrix), dtype=np.int64)
        for index in range(len(matrix)):
            thresholds = (
                observed_sums[index] - tolerance[index]
                - left_sums[:, index])
            insertion = np.searchsorted(
                right_sums[:, index], thresholds, side="left")
            counts[index] = int(np.sum(len(right_signs) - insertion))
        return counts / expected_patterns
    extreme = np.zeros(len(matrix), dtype=np.int64)
    # Matrix multiplication is substantially faster than materializing a
    # sign x simulation x family cube and retains the exact float64 sums.
    transposed = matrix.T
    for start in range(0, len(sign_array), chunk_size):
        selected = np.asarray(
            sign_array[start:start + chunk_size], dtype=np.float64)
        null_sums = selected @ transposed
        tolerance = 1e-13 * np.maximum(1.0, np.abs(observed_sums))
        extreme += np.count_nonzero(
            null_sums >= observed_sums[None, :] - tolerance[None, :],
            axis=0)
    return (extreme + 1) / (len(sign_array) + 1)


def symmetrized_residual_calibration(
        family_interactions: Sequence[float], *,
        planning_sd: float) -> dict:
    """Center, reflect, and scale pilot residual magnitudes for planning.

    Only reflection-invariant magnitudes leave this function.  Consequently,
    adding any constant to every pilot interaction produces byte-identical
    calibration magnitudes and cannot change the power result.
    """
    values = np.asarray(family_interactions, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("pilot needs at least three finite family interactions")
    if not math.isfinite(planning_sd) or planning_sd <= 0:
        raise ValueError("planning SD must be finite and positive")
    centered = values - values.mean()
    sample_sd = float(np.std(centered, ddof=1))
    if sample_sd <= 1e-14:
        raise ValueError("pilot interactions have zero centered variance")
    magnitudes = np.abs(centered)
    rms = float(np.sqrt(np.mean(np.square(magnitudes))))
    if rms <= 1e-14:
        raise ValueError("pilot centered residual RMS is zero")
    scaled = magnitudes * (planning_sd / rms)
    # A fresh random sign gives a zero-mean planning law whose population SD
    # is exactly the registered planning SD.
    if not math.isclose(
            float(np.sqrt(np.mean(np.square(scaled)))), planning_sd,
            rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("planning residual scaling failed")
    rounded = [float(value) for value in np.round(np.sort(scaled), 15)]
    return {
        "n_families": int(len(values)),
        "source_centered_sample_sd": sample_sd,
        "source_centered_residual_rms": rms,
        "planning_sd": float(planning_sd),
        "scale_factor": float(planning_sd / rms),
        "scaled_residual_magnitudes": np.asarray(scaled, dtype=np.float64),
        "scaled_residual_magnitudes_sha256": object_sha256(rounded),
        "signed_pilot_mean_retained": False,
        "signed_family_interactions_retained": False,
    }


def simulate_rejection_rate(
        *, residual_magnitudes: np.ndarray, n_families: int,
        effect: float, n_simulations: int, signs: np.ndarray, exact: bool,
        seed: int, alpha: float, simulation_batch_size: int = 50,
        sign_chunk_size: int = 512) -> dict:
    """Power/type-I rate under the symmetrized empirical planning law."""
    magnitudes = np.asarray(residual_magnitudes, dtype=np.float64)
    if magnitudes.ndim != 1 or len(magnitudes) < 3:
        raise ValueError("residual magnitudes must be a one-dimensional pilot")
    if np.any(magnitudes < 0) or not np.isfinite(magnitudes).all():
        raise ValueError("residual magnitudes must be finite and nonnegative")
    if n_simulations < 1 or simulation_batch_size < 1:
        raise ValueError("simulation counts must be positive")
    if not 0 < alpha < 0.5 or not math.isfinite(effect):
        raise ValueError("invalid alpha or effect")
    generator = np.random.default_rng(seed)
    rejections = 0
    completed = 0
    while completed < n_simulations:
        count = min(simulation_batch_size, n_simulations - completed)
        indices = generator.integers(
            0, len(magnitudes), size=(count, n_families))
        residual_signs = 2 * generator.integers(
            0, 2, size=(count, n_families), dtype=np.int8) - 1
        values = magnitudes[indices] * residual_signs + float(effect)
        pvalues = batch_signflip_pvalues(
            values, signs, exact=exact, chunk_size=sign_chunk_size)
        rejections += int(np.count_nonzero(pvalues <= alpha))
        completed += count
    rate = rejections / n_simulations
    return {
        "n_families": int(n_families),
        "effect_accuracy_points": float(effect),
        "rejections": int(rejections),
        "n_simulations": int(n_simulations),
        "rejection_rate": float(rate),
        "monte_carlo_se": float(math.sqrt(
            rate * (1.0 - rate) / n_simulations)),
        "wilson_ci95": _wilson_interval(rejections, n_simulations),
        "alpha": float(alpha),
        "sign_method": (
            "exact-complete-family-signflip" if exact
            else "monte-carlo-family-signflip-plus-one"),
        "sign_patterns_or_draws": int(len(signs)),
    }


def _registered_pilot(config: Mapping) -> tuple[dict, Path, str]:
    specification = config["source_pilot"]
    expected = str(specification["result_sha256"])
    if expected.startswith("BIND_") or len(expected) != 64:
        raise RuntimeError("P4-P2 pilot result hash is not bound")
    path = resolve_uri(specification["result_uri"])
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError("registered P4-P2 pilot result hash drift")
    event = resolve(str(specification["evidence_id"]))
    if not event["live"]:
        raise RuntimeError("P4-P2 pilot evidence is not live")
    registered = {
        row["sha256"] for row in event["outputs"]
        if Path(row["path"]) == path
    }
    if registered != {expected}:
        raise RuntimeError("P4-P2 pilot result is absent from its event")
    result = json.loads(path.read_text())
    if result.get("evidence_id") != specification["evidence_id"]:
        raise RuntimeError("P4-P2 pilot evidence ID drift")
    if specification["require_valid_mechanical_gates"] \
            and result.get("pilot_analysis_valid") is not True:
        raise RuntimeError("P4-P2 pilot mechanical gates did not pass")
    if result.get("n_families") != int(specification["expected_families"]):
        raise RuntimeError("P4-P2 pilot family count drift")
    if result.get("pilot_mean_used_for_sesoi_selection") is not False:
        raise RuntimeError("P4-P2 pilot improperly used its signed mean")
    if result.get("freeze_ready") is not False:
        raise RuntimeError("P4-P2 variance pilot cannot authorize freeze")
    return result, path, actual


def _calibration(config: Mapping) -> tuple[dict, np.ndarray, dict]:
    pilot, _path, pilot_sha = _registered_pilot(config)
    sesoi = config["prospective_sesoi"]
    memo = resolve_uri(sesoi["memo_uri"])
    if file_sha256(memo) != sesoi["memo_sha256"]:
        raise RuntimeError("P4-P2 prospective SESOI memo hash drift")
    if sesoi["fixed_before_pilot"] is not True \
            or sesoi["pilot_mean_may_not_select_or_revise_sesoi"] is not True:
        raise RuntimeError("P4-P2 SESOI is not prospectively protected")
    source_sesoi = float(pilot["substantive_sesoi_accuracy_points"])
    if not math.isclose(
            source_sesoi, float(sesoi["accuracy_points"]),
            rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("pilot and prospective SESOI differ")
    values = np.asarray(pilot["family_interactions"], dtype=np.float64)
    if len(values) != int(config["source_pilot"]["expected_families"]):
        raise RuntimeError("pilot family-interaction vector length drift")
    sample_sd = float(pilot["family_interaction_sample_sd"])
    upper = float(pilot["family_interaction_bootstrap_sd_upper"])
    planning_sd = float(pilot["planning_family_sd"])
    if not math.isclose(
            planning_sd, max(sample_sd, upper), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("pilot planning SD does not follow the frozen rule")
    if not math.isclose(
            float(np.std(values, ddof=1)), sample_sd,
            rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("pilot family-interaction SD drift")
    centered = symmetrized_residual_calibration(
        values, planning_sd=planning_sd)
    magnitudes = centered.pop("scaled_residual_magnitudes")
    public = {
        **centered,
        "source_evidence_id": config["source_pilot"]["evidence_id"],
        "source_result_sha256": pilot_sha,
        "source_sample_sd": sample_sd,
        "source_bootstrap_90pct_upper_sd": upper,
        "planning_sd_rule": "max(sample SD, family-bootstrap 90% upper SD)",
        "calibration_law": config["simulation"]["calibration_law"],
        "centering_is_shift_invariant": True,
        "signed_pilot_mean_reported": False,
        "signed_pilot_mean_used": False,
        "signed_family_interactions_reported": False,
    }
    continuous = pilot.get("continuous_answer_lp_interaction", {})
    continuous_public = {
        "n_complete_families": continuous.get("n_complete_families"),
        "sample_sd": continuous.get("sample_sd"),
        "signed_mean_reported": False,
        "family_values_reported": False,
        "status": (
            "named variance sensitivity only; no prospective SESOI and no "
            "power, primary, split, or freeze authorization"),
    }
    return public, magnitudes, continuous_public


def run_simulation(config: Mapping) -> dict:
    calibration, magnitudes, continuous = _calibration(config)
    randomization = config["primary_randomization"]
    simulation = config["simulation"]
    familywise = float(randomization["familywise_alpha"])
    primary_count = int(randomization["conservative_holm_primary_count"])
    alpha = float(randomization["alpha"])
    if not math.isclose(
            alpha, familywise / primary_count,
            rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("P4-P2 alpha is not the prospective Holm bound")
    if simulation["center_before_any_simulation"] is not True \
            or simulation["never_report_or_use_signed_pilot_mean"] is not True:
        raise RuntimeError("P4-P2 power calibration does not mask the mean")
    counts = [int(value) for value in simulation["family_counts"]]
    if counts != sorted(set(counts)) or counts[0] < 3:
        raise RuntimeError("P4-P2 family-count grid must be sorted and unique")
    n_simulations = int(simulation["n_simulations"])
    root_seed = int(simulation["simulation_seed"])
    sign_root = int(randomization["sign_seed"])
    exact_max = int(randomization["exact_max_families"])
    sign_draws = int(simulation["power_pvalue_sign_draws"])
    if int(randomization["eventual_outcome_monte_carlo_sign_draws"]) \
            < 100_000:
        raise RuntimeError("eventual P4-P2 outcome test needs >=100000 signs")
    signs_by_count = {
        count: sign_matrix(
            count, draws=sign_draws,
            seed=stable_seed(sign_root, "p4p2-signs", count),
            exact_max_families=exact_max)
        for count in counts
    }

    def scenario(count: int, effect: float, label: str) -> dict:
        signs, exact = signs_by_count[count]
        row = simulate_rejection_rate(
            residual_magnitudes=magnitudes, n_families=count,
            effect=effect, n_simulations=n_simulations,
            signs=signs, exact=exact,
            seed=stable_seed(root_seed, label, count), alpha=alpha,
            simulation_batch_size=int(simulation["simulation_batch_size"]),
            sign_chunk_size=int(randomization["sign_chunk_size"]))
        row["scenario"] = label
        row["sign_matrix_sha256"] = hashlib.sha256(
            np.asarray(signs, dtype=np.int8).tobytes()).hexdigest()
        return row

    type_i = [scenario(count, 0.0, "null") for count in counts]
    bounds = [float(value) for value in
              simulation["type_i_acceptance_interval"]]
    for row in type_i:
        row["type_i_pass"] = bool(
            bounds[0] <= row["rejection_rate"] <= bounds[1])
        row["nominal_alpha_in_wilson_ci95"] = bool(
            row["wilson_ci95"][0] <= alpha <= row["wilson_ci95"][1])
    sesoi = float(config["prospective_sesoi"]["accuracy_points"])
    power = [scenario(count, sesoi, "sesoi") for count in counts]
    target = float(simulation["power_target"])
    for row in power:
        row["point_power_meets_target"] = bool(
            row["rejection_rate"] >= target)
        row["wilson_lower_meets_target"] = bool(
            row["wilson_ci95"][0] >= target)
    minimum_point = next((
        row["n_families"] for row in power
        if row["point_power_meets_target"]), None)
    minimum_wilson = next((
        row["n_families"] for row in power
        if row["wilson_lower_meets_target"]), None)
    type_i_pass = bool(all(row["type_i_pass"] for row in type_i))
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "source_boundary": (
            "Consumes only the registered 20-family development variance "
            "pilot. No untouched, confirmatory, or replication row is read."),
        "calibration": calibration,
        "primary": {
            **dict(randomization),
            "sesoi_accuracy_points": sesoi,
            "power_target": target,
            "test_statistic_equivalence": (
                "sign-flip ordering of the equal-family mean is identical "
                "to its studentized form for each fixed family vector"),
        },
        "simulation": {
            **dict(simulation),
            "signed_pilot_mean_reported": False,
            "signed_pilot_mean_used": False,
        },
        "type_i_calibration": type_i,
        "power_at_prospective_sesoi": power,
        "continuous_answer_lp_sensitivity": continuous,
        "decision": {
            "all_type_i_counts_pass": type_i_pass,
            "minimum_family_count_with_point_power_target": minimum_point,
            "minimum_family_count_with_wilson_lower_at_target": minimum_wilson,
            "accuracy_endpoint_power_feasible_on_grid": bool(
                type_i_pass and minimum_wilson is not None),
            "family_bank_or_split_authorized": False,
            "primary_or_freeze_authorized": False,
            "remaining_requirements": [
                "independent review and PI approval of the substantive SESOI",
                "author and audit enough untouched common-support families",
                "freeze a hash-pinned confirmatory/replication family split",
                "final Holm-family reconciliation before Phase 4 freeze",
            ],
        },
        "claim_boundary": config["claim_boundary"],
        "freeze_ready": False,
    }


def make_figure(result: Mapping, *, png: Path, pdf: Path) -> None:
    power = result["power_at_prospective_sesoi"]
    type_i = result["type_i_calibration"]
    counts = [row["n_families"] for row in power]
    target = float(result["primary"]["power_target"])
    alpha = float(result["primary"]["alpha"])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(
        counts, [row["rejection_rate"] for row in power],
        marker="o", color="#0072B2", label="power estimate")
    axes[0].fill_between(
        counts,
        [row["wilson_ci95"][0] for row in power],
        [row["wilson_ci95"][1] for row in power],
        color="#56B4E9", alpha=0.25, label="95% Wilson interval")
    axes[0].axhline(target, color="#D55E00", linestyle="--")
    axes[0].set(
        xscale="log", xlabel="independent canonical families",
        ylabel="power", ylim=(0, 1.02),
        title="A · Power at prospective 0.20 SESOI")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(
        counts, [row["rejection_rate"] for row in type_i],
        marker="s", color="#009E73")
    axes[1].axhline(alpha, color="#D55E00", linestyle="--",
                    label=f"planning alpha {alpha:.4f}")
    bounds = result["simulation"]["type_i_acceptance_interval"]
    axes[1].axhspan(bounds[0], bounds[1], color="#E69F00", alpha=0.18)
    axes[1].set(
        xscale="log", xlabel="independent canonical families",
        ylabel="null rejection rate", ylim=(0, max(0.04, bounds[1] * 1.25)),
        title="B · Sign-flip type-I calibration")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "P4-P2 exact/Monte Carlo family sign-flip planning\n"
        "Signed pilot mean centered away before simulation")
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def _generation_protocol_unchanged(
        generation_commit: str, config_path: Path) -> bool:
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
        })
        atomic_json(outputs["result"], result)
        make_figure(
            result, png=outputs["figure_png"], pdf=outputs["figure_pdf"])
        print(json.dumps({
            "status": "simulated-unregistered",
            "decision": result["decision"],
            "signed_pilot_mean_reported": False,
        }, indent=1))
        return
    required = [outputs[key] for key in ("result", "figure_png", "figure_pdf")]
    if not all(path.exists() for path in required):
        raise RuntimeError("P4-P2 exact-power outputs are incomplete")
    result = json.loads(outputs["result"].read_text())
    if result.get("freeze_ready") is not False:
        raise RuntimeError("P4-P2 power calibration cannot authorize freeze")
    if result["decision"].get("family_bank_or_split_authorized") is not False:
        raise RuntimeError("P4-P2 power result improperly authorizes a split")
    if not _generation_protocol_unchanged(
            result["generation_code_commit"], config_path):
        raise RuntimeError("P4-P2 power protocol changed after simulation")
    _pilot, _path, pilot_sha = _registered_pilot(config)
    command = (
        "python -m jspace_phase4.experiments.p4_qwen_mode_exact_power "
        f"--config {arguments.config} --register-existing")
    decision = result["decision"]
    create(
        config["evidence_id"], tier=config["tier"], command=command,
        what=(
            "Consumed-development P4-P2 exact/Monte Carlo family sign-flip "
            "power calibration at the prospective 0.20 accuracy SESOI; "
            "signed pilot mean centered away; minimum count whose Wilson "
            f"lower bound reaches 0.80 is "
            f"{decision['minimum_family_count_with_wilson_lower_at_target']}; "
            "no bank or split is authorized."),
        outputs=required,
        inputs={
            "config": file_sha256(config_path),
            "variance_pilot": pilot_sha,
            "prospective_sesoi_memo": config[
                "prospective_sesoi"]["memo_sha256"],
        },
        untouched_families_opened=False,
        confirmatory_or_replication_outcomes_opened=False,
        freeze_ready=False)
    print(json.dumps({
        "status": "registered", "evidence_id": config["evidence_id"]},
        indent=1))


if __name__ == "__main__":
    main()
