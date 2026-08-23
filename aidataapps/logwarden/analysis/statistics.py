#!/usr/bin/env python3
"""Deterministic paired, scenario-grouped power simulation for LogWarden."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="config/scenarios/standard-v1.json")
    parser.add_argument("--output")
    parser.add_argument("--dev-rows")
    parser.add_argument("--seed", type=int, default=590_003)
    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--holm-family", type=int, default=10)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--base-accuracy", type=float, default=0.80)
    parser.add_argument("--effect", type=float, default=0.10)
    parser.add_argument("--paired-latent-correlation", type=float, default=0.50)
    parser.add_argument("--within-group-icc", type=float, default=0.10)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def catalog_design(path: Path) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, Any]] = {}
    family_roles: Counter[tuple[str, str]] = Counter()
    role_counts: Counter[str] = Counter()
    for scenario in catalog["scenarios"]:
        variants = scenario.get("variants", [])
        roles = {variant["splitRole"] for variant in variants}
        if len(roles) != 1:
            raise ValueError(f"scenario group crosses roles: {scenario['groupId']}")
        role = next(iter(roles))
        group_id = scenario["groupId"]
        if group_id in groups:
            raise ValueError(f"duplicate scenario group: {group_id}")
        groups[group_id] = {
            "family": scenario["family"],
            "role": role,
            "episodes": len(variants),
        }
        role_counts[role] += len(variants)
        family_roles[(scenario["family"], role)] += len(variants)
    test_groups = [value for value in groups.values() if value["role"] == "test_id"]
    if not test_groups:
        raise ValueError("catalog has no test_id groups")
    group_sizes = [int(value["episodes"]) for value in test_groups]
    median_group_size = int(median(group_sizes))
    family_floor = {
        family: family_roles[(family, "test_id")] + family_roles[(family, "test_variant_holdout")]
        for family in sorted({value["family"] for value in groups.values()})
    }
    return {
        "catalogId": catalog["catalogId"],
        "totalEpisodes": sum(role_counts.values()),
        "roleCounts": dict(sorted(role_counts.items())),
        "testIdGroups": len(test_groups),
        "testIdEpisodes": sum(group_sizes),
        "testIdGroupSizes": group_sizes,
        "medianTestIdGroupSize": median_group_size,
        "familyTestAndVariantCounts": family_floor,
        "familiesMeetingThirtyEpisodeFloor": all(count >= 30 for count in family_floor.values()),
    }


def read_dev_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        rows = parsed if isinstance(parsed, list) else parsed.get("rows", [])
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        group = row.get("scenario_group_id", row.get("scenarioGroupId"))
        baseline = row.get("baseline_success", row.get("baselineSuccess"))
        treatment = row.get("treatment_success", row.get("treatmentSuccess"))
        if not isinstance(group, str) or not group:
            raise ValueError(f"dev row {index} has no scenario group")
        try:
            baseline_value, treatment_value = int(baseline), int(treatment)
        except (TypeError, ValueError) as error:
            raise ValueError(f"dev row {index} has invalid paired outcomes") from error
        if baseline_value not in (0, 1) or treatment_value not in (0, 1):
            raise ValueError(f"dev row {index} outcomes must be binary")
        output.append({"group": group, "baseline": baseline_value, "treatment": treatment_value})
    if len(output) < 20 or len({row["group"] for row in output}) < 4:
        raise ValueError("dev calibration requires at least 20 paired rows and four scenario groups")
    return output


def empirical_calibration(rows: list[dict[str, Any]], fallback_pair: float, fallback_icc: float) -> dict[str, Any]:
    baseline = [row["baseline"] for row in rows]
    treatment = [row["treatment"] for row in rows]
    pair = pearson(baseline, treatment)
    differences = [right - left for left, right in zip(baseline, treatment)]
    icc = one_way_icc([(row["group"], diff) for row, diff in zip(rows, differences)])
    return {
        "source": "empirical_dev_pairs",
        "rowCount": len(rows),
        "groupCount": len({row["group"] for row in rows}),
        "observedBaselineAccuracy": mean(baseline),
        "observedTreatmentAccuracy": mean(treatment),
        "pairedLatentCorrelation": clamp(pair if math.isfinite(pair) else fallback_pair, -0.90, 0.90),
        "withinGroupIcc": clamp(icc if math.isfinite(icc) else fallback_icc, 0.0, 0.80),
    }


def pearson(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right))
    return numerator / denominator if denominator else float("nan")


def one_way_icc(rows: list[tuple[str, float]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for group, value in rows:
        grouped[group].append(value)
    values = [value for _, value in rows]
    grand = mean(values)
    group_count, row_count = len(grouped), len(values)
    if group_count < 2 or row_count <= group_count:
        return float("nan")
    between = sum(len(group) * (mean(group) - grand) ** 2 for group in grouped.values()) / (group_count - 1)
    within = sum(sum((value - mean(group)) ** 2 for value in group) for group in grouped.values()) / (row_count - group_count)
    mean_size = row_count / group_count
    denominator = between + (mean_size - 1) * within
    return (between - within) / denominator if denominator else float("nan")


def simulate_group_differences(
    rng: random.Random,
    groups: int,
    group_size: int,
    base_accuracy: float,
    treatment_accuracy: float,
    paired_correlation: float,
    icc: float,
) -> list[float]:
    base_threshold = normal_inverse_cdf(base_accuracy)
    treatment_threshold = normal_inverse_cdf(treatment_accuracy)
    group_scale, residual_scale = math.sqrt(icc), math.sqrt(1 - icc)
    pair_residual = math.sqrt(max(0.0, 1 - paired_correlation**2))
    output: list[float] = []
    for _ in range(groups):
        common = rng.gauss(0, 1)
        difference_sum = 0
        for _ in range(group_size):
            left_noise, independent = rng.gauss(0, 1), rng.gauss(0, 1)
            right_noise = paired_correlation * left_noise + pair_residual * independent
            left_latent = group_scale * common + residual_scale * left_noise
            right_latent = group_scale * common + residual_scale * right_noise
            difference_sum += int(right_latent <= treatment_threshold) - int(left_latent <= base_threshold)
        output.append(difference_sum / group_size)
    return output


def grouped_bootstrap_lower(
    rng: random.Random, group_differences: list[float], replicates: int, tail_probability: float
) -> float:
    group_count = len(group_differences)
    estimates = [
        sum(rng.choice(group_differences) for _ in range(group_count)) / group_count
        for _ in range(replicates)
    ]
    estimates.sort()
    index = max(0, min(replicates - 1, math.floor(tail_probability * replicates)))
    return estimates[index]


def estimate_power(
    seed: int,
    groups: int,
    group_size: int,
    simulations: int,
    bootstrap_replicates: int,
    base_accuracy: float,
    effect: float,
    paired_correlation: float,
    icc: float,
    corrected_alpha: float,
) -> float:
    detected = 0
    for simulation in range(simulations):
        data_rng = random.Random(stable_seed(seed, groups, simulation, 1))
        bootstrap_rng = random.Random(stable_seed(seed, groups, simulation, 2))
        differences = simulate_group_differences(
            data_rng, groups, group_size, base_accuracy, base_accuracy + effect, paired_correlation, icc
        )
        lower = grouped_bootstrap_lower(bootstrap_rng, differences, bootstrap_replicates, corrected_alpha / 2)
        detected += int(lower > 0)
    return detected / simulations


def run_power(args: argparse.Namespace) -> dict[str, Any]:
    if not (0 < args.base_accuracy < 1 and 0 < args.base_accuracy + args.effect < 1):
        raise ValueError("base accuracy and effect must imply probabilities within (0,1)")
    if args.simulations < 50 or args.bootstrap_replicates < 200:
        raise ValueError("use at least 50 simulations and 200 grouped bootstrap replicates")
    if args.holm_family < 1:
        raise ValueError("Holm family must be positive")
    design = catalog_design(Path(args.catalog))
    calibration: dict[str, Any]
    if args.dev_rows:
        calibration = empirical_calibration(
            read_dev_rows(Path(args.dev_rows)), args.paired_latent_correlation, args.within_group_icc
        )
    else:
        calibration = {
            "source": "conservative_design_assumption",
            "rowCount": 0,
            "groupCount": 0,
            "pairedLatentCorrelation": args.paired_latent_correlation,
            "withinGroupIcc": args.within_group_icc,
        }
    group_size = design["medianTestIdGroupSize"]
    corrected_alpha = args.alpha / args.holm_family
    pair_corr = calibration["pairedLatentCorrelation"]
    icc = calibration["withinGroupIcc"]
    candidate_groups = sorted(set(list(range(10, 66, 5)) + [design["testIdGroups"]]))
    curve = []
    for groups in candidate_groups:
        power = estimate_power(
            args.seed, groups, group_size, args.simulations, args.bootstrap_replicates,
            args.base_accuracy, args.effect, pair_corr, icc, corrected_alpha,
        )
        curve.append({"groups": groups, "episodes": groups * group_size, "estimatedPower": power})
    qualified = [row for row in curve if row["estimatedPower"] >= args.target_power]
    required = min(qualified, key=lambda row: row["groups"]) if qualified else None
    actual = next(row for row in curve if row["groups"] == design["testIdGroups"])
    return {
        "schemaVersion": 1,
        "algorithm": "paired-latent-bernoulli-grouped-percentile-bootstrap-v1",
        "seed": args.seed,
        "assumptions": {
            "baseAccuracy": args.base_accuracy,
            "absoluteEffect": args.effect,
            "targetPower": args.target_power,
            "alpha": args.alpha,
            "holmFamilySize": args.holm_family,
            "holmWorstCaseAlpha": corrected_alpha,
            "twoSidedBootstrapTail": corrected_alpha / 2,
            "simulations": args.simulations,
            "bootstrapReplicatesPerSimulation": args.bootstrap_replicates,
        },
        "calibration": calibration,
        "catalog": design,
        "powerCurve": curve,
        "requiredDesign": required,
        "actualTestIdDesign": actual,
        "passesPowerTarget": actual["estimatedPower"] >= args.target_power,
        "passesFamilyFloor": design["familiesMeetingThirtyEpisodeFloor"],
        "empiricallyCalibrated": calibration["source"] == "empirical_dev_pairs",
        "disposition": (
            "PASS" if actual["estimatedPower"] >= args.target_power
            and design["familiesMeetingThirtyEpisodeFloor"]
            and calibration["source"] == "empirical_dev_pairs"
            else "DESIGN_ONLY" if calibration["source"] != "empirical_dev_pairs"
            else "STOP_POWER"
        ),
    }


def mean(values: list[float] | list[int]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def median(values: list[int]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def normal_inverse_cdf(probability: float) -> float:
    """Acklam's rational approximation for the standard-normal quantile."""
    if not 0 < probability < 1:
        raise ValueError("normal quantile probability must be within (0,1)")
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416)
    lower = 0.02425
    if probability < lower:
        q = math.sqrt(-2 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if probability > 1 - lower:
        q = math.sqrt(-2 * math.log(1 - probability))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = probability - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def stable_seed(seed: int, groups: int, simulation: int, stream: int) -> int:
    value = seed & ((1 << 64) - 1)
    for part in (groups, simulation, stream):
        value ^= part + 0x9E3779B97F4A7C15 + ((value << 6) & ((1 << 64) - 1)) + (value >> 2)
        value &= (1 << 64) - 1
    return value


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def self_test() -> None:
    assert stable_seed(1, 2, 3, 4) == stable_seed(1, 2, 3, 4)
    assert pearson([0, 1, 0, 1], [0, 1, 0, 1]) == 1
    rng = random.Random(1)
    values = simulate_group_differences(rng, 4, 10, 0.8, 0.9, 0.5, 0.1)
    assert len(values) == 4 and all(-1 <= value <= 1 for value in values)
    lower = grouped_bootstrap_lower(random.Random(2), [0.1, 0.2, 0.3], 200, 0.025)
    assert 0.09 <= lower <= 0.31
    print(json.dumps({"selfTest": "PASS"}))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    result = run_power(args)
    if args.output:
        atomic_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
