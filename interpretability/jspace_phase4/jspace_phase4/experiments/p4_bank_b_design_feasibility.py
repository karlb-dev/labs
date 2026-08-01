"""Outcome-blind feasibility envelope for the blocked Bank B design.

The registered heavy-tailed IUT simulation is the power evidence of record.
This successor adds an intentionally optimistic necessary-condition bound:
even a known-SD Gaussian test of only one component needs a minimum number of
families. The real two-component intersection-union test cannot need fewer.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Mapping

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..manifests import atomic_json, file_sha256, require_clean_tree
from ..registry4 import create, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--analyze", action="store_true")
    group.add_argument("--register-existing", action="store_true")
    return parser.parse_args()


def optimistic_minimum_families(*, effect: float, family_sd: float,
                                alpha: float, power: float) -> int:
    """Known-SD one-sided Gaussian n; a lower bound for the two-test IUT."""
    if effect <= 0 or family_sd <= 0:
        raise ValueError("effect and family SD must be positive")
    if not 0 < alpha < 0.5 or not 0.5 < power < 1:
        raise ValueError("invalid alpha or power target")
    normal = NormalDist()
    z_sum = normal.inv_cdf(1 - alpha) + normal.inv_cdf(power)
    return int(math.ceil((z_sum * family_sd / effect) ** 2))


def optimistic_mde(*, n_families: int, family_sd: float,
                   alpha: float, power: float) -> float:
    if n_families < 2 or family_sd <= 0:
        raise ValueError("invalid family count or SD")
    normal = NormalDist()
    z_sum = normal.inv_cdf(1 - alpha) + normal.inv_cdf(power)
    return float(z_sum * family_sd / math.sqrt(n_families))


def _load_source(config: Mapping) -> dict:
    specification = config["source_power"]
    path = Path(specification["result"])
    if file_sha256(path) != specification["result_sha256"]:
        raise RuntimeError("registered Bank B power result hash drift")
    event = resolve(specification["evidence_id"])
    if not event["live"]:
        raise RuntimeError("Bank B power evidence is not live")
    registered = {
        row["sha256"] for row in event["outputs"]
        if Path(row["path"]) == path
    }
    if registered != {specification["result_sha256"]}:
        raise RuntimeError("Bank B power result is absent from its event")
    result = json.loads(path.read_text())
    if result.get("outcome_blinding") != (
            "No Bank B intervention, confirmatory, or replication outcome "
            "was read."):
        raise RuntimeError("source power result lacks outcome-blinding gate")
    return result


def analyze(config: Mapping) -> dict:
    source = _load_source(config)
    bound = config["optimistic_bound"]
    alpha = float(bound["alpha"])
    target = float(bound["power_target"])
    family_sd = float(source["calibration"][
        "conservative_common_sd_nats"])
    effects = [float(value) for value in bound["effects_nats"]]
    family_counts = [int(value) for value in
                     bound["available_family_counts"]]
    minimum_n = [
        {
            "effect_nats": effect,
            "minimum_families": optimistic_minimum_families(
                effect=effect, family_sd=family_sd,
                alpha=alpha, power=target),
        }
        for effect in effects
    ]
    mde = [
        {
            "n_families": n,
            "optimistic_single_component_mde_nats": optimistic_mde(
                n_families=n, family_sd=family_sd,
                alpha=alpha, power=target),
            "registered_heavy_tail_iut_mde_nats": source[
                "mde_at_power_target_by_n"].get(str(n)),
        }
        for n in family_counts
    ]
    candidate_effect = float(source["candidate_joint_sesoi_nats"])
    bridge_mean = float(source["calibration"]["family_mean_nats"][
        "cf_vs_cf_answer"])
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "outcome_blinding": (
            "Uses only the registered consumed Phase 3 variability summary; "
            "no Bank B outcome is opened."),
        "source_power_evidence_id": config["source_power"]["evidence_id"],
        "family_sd_nats": family_sd,
        "alpha": alpha,
        "power_target": target,
        "bound": (
            "Optimistic known-SD Gaussian one-sided single-component test. "
            "It ignores SD estimation, heavy tails, and the second required "
            "IUT component, so it is a necessary lower bound, not a new "
            "power claim."),
        "minimum_families_by_effect": minimum_n,
        "mde_by_available_family_count": mde,
        "candidate_joint_sesoi_nats": candidate_effect,
        "candidate_sesoi_optimistic_minimum_families":
            optimistic_minimum_families(
                effect=candidate_effect, family_sd=family_sd,
                alpha=alpha, power=target),
        "consumed_phase3_bridge_specific_mean_nats": bridge_mean,
        "phase3_mean_optimistic_minimum_families":
            optimistic_minimum_families(
                effect=bridge_mean, family_sd=family_sd,
                alpha=alpha, power=target),
        "existing_bank_total_families": 40,
        "existing_split": {
            "development": 20, "confirmatory": 10, "replication": 10},
        "feasibility_verdict": (
            "Reallocating or using all 40 existing families cannot power the "
            "0.25-nat SESOI. The design needs a substantively new estimand "
            "or must leave P4-P1 outside the Phase 4 confirmatory family; "
            "raising the SESOI solely to obtain power is not licensed."),
        "decision_options_requiring_review": [
            "Retain Bank B sealed and move P4-P1 to estimation-only future "
            "work, removing it prospectively from the Phase 4 primary family.",
            "Design an answer-direction-orthogonal bridge intervention and "
            "obtain a new outcome-blind variability/power ruler before use.",
            "Expand to a genuinely adequate number of independent families; "
            "the lower bound rules out a split-only repair.",
        ],
        "freeze_ready": False,
    }


def plot(result: Mapping, *, png: Path, pdf: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    minimum = result["minimum_families_by_effect"]
    axes[0].plot(
        [row["effect_nats"] for row in minimum],
        [row["minimum_families"] for row in minimum],
        marker="o", color="#0072B2")
    for count, color in ((10, "#CC79A7"), (20, "#E69F00"),
                         (40, "#009E73")):
        axes[0].axhline(count, color=color, linestyle="--", linewidth=1,
                        label=f"available n={count}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("equal component effect (nats)")
    axes[0].set_ylabel("optimistic minimum families")
    axes[0].set_title("A · Single-test lower bound", loc="left")
    axes[0].legend(frameon=False, fontsize=7)

    mde = result["mde_by_available_family_count"]
    x = [str(row["n_families"]) for row in mde]
    optimistic = [
        row["optimistic_single_component_mde_nats"] for row in mde]
    heavy = [
        row["registered_heavy_tail_iut_mde_nats"] for row in mde]
    positions = list(range(len(x)))
    axes[1].bar(
        [value - 0.18 for value in positions], optimistic, width=0.36,
        color="#56B4E9", label="optimistic single test")
    heavy_values = [5.0 if value is None else value for value in heavy]
    heavy_bars = axes[1].bar(
        [value + 0.18 for value in positions], heavy_values, width=0.36,
        color="#D55E00", label="registered heavy-tail IUT")
    for bar, value in zip(heavy_bars, heavy, strict=True):
        if value is None:
            bar.set_facecolor("none")
            bar.set_edgecolor("#D55E00")
            bar.set_hatch("///")
            axes[1].text(
                bar.get_x() + bar.get_width() / 2, 5.03, ">5 grid",
                ha="center", va="bottom", fontsize=7, color="#D55E00")
    axes[1].axhline(
        result["candidate_joint_sesoi_nats"], color="#555555",
        linestyle=":", label="candidate SESOI")
    axes[1].set_xticks(positions, x)
    axes[1].set_xlabel("families")
    axes[1].set_ylabel("80% minimum detectable effect (nats)")
    axes[1].set_title("B · Existing-bank feasibility", loc="left")
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Bank B design feasibility · optimistic bound versus registered IUT")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
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
    if arguments.analyze:
        result = analyze(config)
        atomic_json(outputs["result"], result)
        plot(result, png=outputs["figure_png"], pdf=outputs["figure_pdf"])
        print(json.dumps({
            "status": "analyzed-unregistered",
            "candidate_sesoi_optimistic_minimum_families": result[
                "candidate_sesoi_optimistic_minimum_families"],
            "phase3_mean_optimistic_minimum_families": result[
                "phase3_mean_optimistic_minimum_families"],
            "freeze_ready": result["freeze_ready"],
        }, indent=1))
        return
    if not all(path.exists() for path in outputs.values()):
        raise RuntimeError("Bank B feasibility outputs are incomplete")
    result = json.loads(outputs["result"].read_text())
    if result.get("freeze_ready") is not False:
        raise RuntimeError("feasibility diagnostic cannot authorize freeze")
    command = (
        "python -m jspace_phase4.experiments.p4_bank_b_design_feasibility "
        f"--config {arguments.config} --register-existing")
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            "Outcome-blind Bank B feasibility envelope: an optimistic "
            "single-component Gaussian lower bound demonstrates that "
            "reallocating the existing 40 families cannot repair the "
            "registered 0.25-nat IUT power failure."),
        command=command, outputs=outputs.values(),
        inputs={
            "config": file_sha256(config_path),
            "source_power_result": config["source_power"]["result_sha256"],
        })
    print(json.dumps({
        "status": "registered", "evidence_id": config["evidence_id"]},
        indent=1))


if __name__ == "__main__":
    main()
