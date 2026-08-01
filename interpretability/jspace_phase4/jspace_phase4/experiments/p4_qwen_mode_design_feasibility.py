"""Outcome-blind P4-P2 family-count and variance feasibility envelope.

The model-backed v2 baseline is deliberately not consumed here.  P4-P2's
primary family statistic is an eight-cell interaction of binary answer
quality.  Before an SESOI or untouched-family split can be chosen, this
producer shows the family counts implied by a grid of substantive effects
and family-level interaction SDs.  The known-SD Gaussian calculation is a
planning envelope, not power evidence for the eventual sign-flip test.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..manifests import atomic_json, file_sha256, require_clean_tree
from ..registry4 import create, resolve


CELL_ORDER = (
    "thinking_on_final_control",
    "thinking_on_final_j",
    "thinking_on_prefill_control",
    "thinking_on_prefill_j",
    "thinking_off_final_control",
    "thinking_off_final_j",
    "thinking_off_prefill_control",
    "thinking_off_prefill_j",
)
CELL_COEFFICIENTS = (1, -1, -1, 1, -1, 1, 1, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--analyze", action="store_true")
    group.add_argument("--register-existing", action="store_true")
    return parser.parse_args()


def family_accuracy_interaction(cells: Sequence[float]) -> float:
    """Return the preregistered final-vs-prefill by mode interaction."""
    if len(cells) != len(CELL_COEFFICIENTS):
        raise ValueError("P4-P2 interaction requires exactly eight cells")
    values = [float(value) for value in cells]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("accuracy cells must lie in [0, 1]")
    return float(sum(
        coefficient * value
        for coefficient, value in zip(
            CELL_COEFFICIENTS, values, strict=True)))


def binary_interaction_sd_upper_bound() -> float:
    """Distribution-free SD bound from the interaction support [-4, 4]."""
    positive = sum(max(0, value) for value in CELL_COEFFICIENTS)
    negative = sum(min(0, value) for value in CELL_COEFFICIENTS)
    support_width = float(positive - negative)
    # Popoviciu: variance <= (max - min)^2 / 4.
    return support_width / 2.0


def gaussian_minimum_families(*, effect: float, family_sd: float,
                              alpha: float, power: float) -> int:
    """Known-SD one-sided Gaussian family count for a planning scenario."""
    if effect <= 0 or family_sd <= 0:
        raise ValueError("effect and family SD must be positive")
    if not 0 < alpha < 0.5 or not 0.5 < power < 1:
        raise ValueError("invalid alpha or power target")
    normal = NormalDist()
    z_sum = normal.inv_cdf(1 - alpha) + normal.inv_cdf(power)
    return int(math.ceil((z_sum * family_sd / effect) ** 2))


def gaussian_mde(*, n_families: int, family_sd: float,
                 alpha: float, power: float) -> float:
    if n_families < 2 or family_sd <= 0:
        raise ValueError("invalid family count or SD")
    if not 0 < alpha < 0.5 or not 0.5 < power < 1:
        raise ValueError("invalid alpha or power target")
    normal = NormalDist()
    z_sum = normal.inv_cdf(1 - alpha) + normal.inv_cdf(power)
    return float(z_sum * family_sd / math.sqrt(n_families))


def exact_sign_flip_resolution_families(alpha: float) -> int:
    """Smallest tie-free n whose minimum one-sided exact p can reach alpha."""
    if not 0 < alpha < 0.5:
        raise ValueError("invalid alpha")
    return int(math.ceil(math.log2(1.0 / alpha)))


def _load_methods(config: Mapping) -> dict:
    specification = config["source_methods"]
    path = Path(specification["result"])
    if file_sha256(path) != specification["result_sha256"]:
        raise RuntimeError("registered mode methods result hash drift")
    event = resolve(specification["evidence_id"])
    if not event["live"]:
        raise RuntimeError("mode parser methods evidence is not live")
    registered = {
        row["sha256"] for row in event["outputs"]
        if Path(row["path"]) == path
    }
    if registered != {specification["result_sha256"]}:
        raise RuntimeError("mode parser result is absent from its event")
    result = json.loads(path.read_text())
    if result.get("all_protocol_gates_pass") is not True:
        raise RuntimeError("mode parser methods gate did not pass")
    if result.get("freeze_ready") is not False:
        raise RuntimeError("source mode methods unexpectedly authorize freeze")
    return result


def analyze(config: Mapping) -> dict:
    methods = _load_methods(config)
    planning = config["planning_envelope"]
    familywise_alpha = float(planning["familywise_alpha"])
    primary_hypotheses = int(planning["primary_hypotheses"])
    alpha = float(planning["conservative_endpoint_alpha"])
    if not math.isclose(
            alpha, familywise_alpha / primary_hypotheses,
            rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("endpoint alpha is not the conservative Holm bound")
    target = float(planning["power_target"])
    effects = [float(value) for value in
               planning["candidate_sesoi_accuracy_points"]]
    sds = [float(value) for value in
           planning["family_interaction_sd_scenarios"]]
    counts = [int(value) for value in planning["family_count_grid"]]
    bound = binary_interaction_sd_upper_bound()
    if not math.isclose(max(sds), bound):
        raise RuntimeError("variance grid must include the support-based SD bound")

    minimum_rows = [
        {
            "candidate_sesoi_accuracy_points": effect,
            "family_interaction_sd": family_sd,
            "minimum_families": gaussian_minimum_families(
                effect=effect, family_sd=family_sd,
                alpha=alpha, power=target),
        }
        for effect in effects for family_sd in sds
    ]
    mde_rows = [
        {
            "n_families": count,
            "family_interaction_sd": family_sd,
            "gaussian_mde_accuracy_points": gaussian_mde(
                n_families=count, family_sd=family_sd,
                alpha=alpha, power=target),
        }
        for count in counts for family_sd in sds
    ]
    return {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "outcome_blinding": (
            "Uses only registered parser/template methods and mathematical "
            "bounds. It reads no v1/v2 model baseline row and no P4-P2 "
            "intervention, confirmatory, or replication outcome."),
        "source_methods_evidence_id": config[
            "source_methods"]["evidence_id"],
        "source_methods_parser_version": methods["parser_version"],
        "primary_family_statistic": {
            "cell_order": list(CELL_ORDER),
            "coefficients": list(CELL_COEFFICIENTS),
            "support": [-4.0, 4.0],
            "distribution_free_family_sd_upper_bound": bound,
            "quality_endpoint": "normalized exact accepted-alias accuracy",
        },
        "multiplicity_planning": {
            "familywise_alpha": familywise_alpha,
            "primary_hypotheses": primary_hypotheses,
            "conservative_endpoint_alpha": alpha,
            "rule": (
                "Uses alpha/3 so P4-P2 is powered even at the most "
                "conservative Holm rank; this does not replace the frozen "
                "Holm analysis."),
        },
        "power_target": target,
        "calculation_boundary": (
            "Known-SD one-sided Gaussian approximation. It is a scenario "
            "envelope, not exact sign-flip power and not a selected SESOI."),
        "minimum_families_by_sesoi_and_sd": minimum_rows,
        "mde_by_family_count_and_sd": mde_rows,
        "minimum_families_for_exact_sign_flip_resolution":
            exact_sign_flip_resolution_families(alpha),
        "design_conclusion": (
            "No P4-P2 SESOI, family count, or split is selected. Family "
            "requirements vary quadratically with the unmeasured family-"
            "interaction SD; therefore a consumed-development intervention "
            "variance pilot plus substantive review is required before an "
            "untouched bank can be sized and split."),
        "next_required_methods": [
            "Pass the prospectively frozen v2 model-backed baseline gate.",
            "Run a development-only phase-intervention variance pilot on "
            "already consumed families after the canonical lens is fixed.",
            "Have independent review select a substantive accuracy SESOI, "
            "then author and hash a sufficiently large untouched family bank.",
            "Simulate the exact frozen family sign-flip analysis under the "
            "selected bank and variance ruler before freeze.",
        ],
        "freeze_ready": False,
    }


def plot(result: Mapping, *, png: Path, pdf: Path) -> None:
    minimum = result["minimum_families_by_sesoi_and_sd"]
    effects = sorted({
        row["candidate_sesoi_accuracy_points"] for row in minimum})
    sds = sorted({row["family_interaction_sd"] for row in minimum})
    lookup = {
        (row["candidate_sesoi_accuracy_points"],
         row["family_interaction_sd"]): row["minimum_families"]
        for row in minimum
    }
    matrix = np.asarray([
        [lookup[(effect, family_sd)] for effect in effects]
        for family_sd in sds], dtype=np.float64)

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    image = axes[0].imshow(
        np.log10(matrix), aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_xticks(range(len(effects)), [f"{value:.2f}" for value in effects])
    axes[0].set_yticks(range(len(sds)), [f"{value:g}" for value in sds])
    axes[0].set_xlabel("candidate SESOI (accuracy points)")
    axes[0].set_ylabel("family-interaction SD scenario")
    axes[0].set_title("A · Gaussian minimum families", loc="left")
    for y, family_sd in enumerate(sds):
        for x, effect in enumerate(effects):
            value = lookup[(effect, family_sd)]
            axes[0].text(
                x, y, f"{value:,}", ha="center", va="center", fontsize=6.5,
                color="white" if np.log10(value) > np.median(
                    np.log10(matrix)) else "black")
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("log10 minimum families")

    mde = result["mde_by_family_count_and_sd"]
    counts = sorted({row["n_families"] for row in mde})
    mde_lookup = {
        (row["n_families"], row["family_interaction_sd"]):
            row["gaussian_mde_accuracy_points"]
        for row in mde
    }
    colors = plt.cm.plasma(np.linspace(0.05, 0.9, len(sds)))
    for color, family_sd in zip(colors, sds, strict=True):
        axes[1].plot(
            counts, [mde_lookup[(count, family_sd)] for count in counts],
            marker="o", markersize=3, color=color, label=f"SD={family_sd:g}")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("independent canonical families")
    axes[1].set_ylabel("80% Gaussian MDE (accuracy points)")
    axes[1].set_title("B · Family-count sensitivity", loc="left")
    axes[1].grid(alpha=0.2, which="both")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=7, ncol=2)

    figure.suptitle(
        "P4-P2 design feasibility · scenario envelope, no SESOI selected")
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
    if arguments.analyze:
        result = analyze(config)
        atomic_json(outputs["result"], result)
        plot(result, png=outputs["figure_png"], pdf=outputs["figure_pdf"])
        print(json.dumps({
            "status": "analyzed-unregistered",
            "minimum_exact_resolution_families": result[
                "minimum_families_for_exact_sign_flip_resolution"],
            "freeze_ready": result["freeze_ready"],
        }, indent=1))
        return
    if not all(path.exists() for path in outputs.values()):
        raise RuntimeError("mode feasibility outputs are incomplete")
    result = json.loads(outputs["result"].read_text())
    if result.get("freeze_ready") is not False:
        raise RuntimeError("mode feasibility diagnostic cannot authorize freeze")
    command = (
        "python -m "
        "jspace_phase4.experiments.p4_qwen_mode_design_feasibility "
        f"--config {arguments.config} --register-existing")
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            "Outcome-blind P4-P2 design-feasibility envelope across "
            "candidate accuracy-interaction SESOIs, family-level SD "
            "scenarios, and conservative Holm planning alpha; no SESOI, "
            "family count, or split is selected."),
        command=command, outputs=outputs.values(),
        inputs={
            "config": file_sha256(config_path),
            "source_methods_result": config[
                "source_methods"]["result_sha256"],
        })
    print(json.dumps({
        "status": "registered", "evidence_id": config["evidence_id"]},
        indent=1))


if __name__ == "__main__":
    main()
