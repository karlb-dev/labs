"""Generate the Phase 3 release-audit summary figure from live evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from jspace_part2.lib import sha256_file
from ..paths3 import figures_dir, run_root
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)

EVIDENCE_ID = "p3-release-summary-figure-v1"
TIER = "methods"
SEED = 4242


def _payload(path: Path) -> dict:
    value = json.loads(path.read_text())
    return value.get("payload", value)


def source_paths(root: Path) -> dict[str, Path]:
    return {
        "inference": (
            root / "metrics/cross_model/release_audit/"
            "p3_inference_audit.json"),
        "protected": (
            root / "metrics/qwen36-27b/release_audit/protected_answer/"
            "p3_protected_answer_audit.json"),
        "boundary": (
            root / "metrics/cross_model/release_audit/"
            "alias_cohort_sensitivity_v2/"
            "p3_alias_cohort_sensitivity.json"),
        "geometry": (
            root / "metrics/qwen36-27b/release_audit/"
            "bridge_geometry_v2/p3_bridge_geometry_qwen36-27b.json"),
        "swap": (
            root / "metrics/qwen36-27b/release_audit/"
            "bridge_swap_endpoint/p3_bridge_swap_endpoint_qwen36-27b.json"),
        "alias": (
            root / "metrics/cross_model/release_audit/alias_endpoint/"
            "p3_alias_endpoint_cross_model.json"),
    }


def collect_stats(sources: dict[str, dict]) -> dict:
    inference = sources["inference"]
    protected = sources["protected"]
    boundary = sources["boundary"]
    geometry = sources["geometry"]
    swap = sources["swap"]
    alias = sources["alias"]

    seed_views = boundary[
        "control_seed_cohort_sensitivity"]["confirmatory"]
    seed_values = {
        seed: views["historical_unsafe_strict"]["P3-P1"][
            "estimate_equal_family"]
        for seed, views in seed_views.items()
    }
    boundary_state = seed_views["31337"][
        "boundary_safe_strict"]["P3-P1"]["estimate_equal_family"]

    def protected_p2(side: str, view: str) -> dict:
        return protected["sides"][side][view][
            "threshold_curve"]["-1.0"]

    p2 = {
        "confirmatory_all": protected_p2("confirmatory", "all_items"),
        "confirmatory_exact": protected_p2(
            "confirmatory", "exact_scored_alias_protected"),
        "replication_all": protected_p2("replication", "all_items"),
        "replication_exact": protected_p2(
            "replication", "exact_scored_alias_protected"),
        "boundary_confirmatory": boundary[
            "cohort_sensitivity"]["confirmatory"][
                "boundary_safe_strict"]["P3-P2"],
    }
    p3 = {
        "raw": geometry["raw_rescue"],
        "geometry_residual": geometry[
            "residualized_semantic_contrast"],
        "strict_geometry": geometry[
            "exact_geometry_matched_subset"]["rescue"],
        "boundary": boundary[
            "cohort_sensitivity"]["confirmatory"][
                "boundary_safe_strict"]["P3-P3"],
    }
    semantic = {
        "from_baseline": swap["primary_cf_preference_shift"],
        "vs_unrelated": swap[
            "primary_cf_vs_unrelated_matched_injection"],
        "vs_answer_direction": swap["control_contrasts"][
            "cf_swap_minus_cf_answer_direction"],
    }
    alias_change = {
        key: value["estimate"]
        for key, value in alias[
            "P3-P1_alias_change_vs_stable_first"].items()
    }
    return {
        "schema_version": 1,
        "P3-P1_control_seed_estimates": seed_values,
        "P3-P1_boundary_state_seed_31337": boundary_state,
        "P3-P1_historical_frozen": inference[
            "confirmatory"]["P3-P1"][
                "estimate_family_weighted"],
        "P3-P2": p2,
        "P3-P3": p3,
        "semantic_swap": semantic,
        "alias_change_vs_stable_first": alias_change,
        "figure_contract": {
            "intervals": (
                "family-resampling percentile intervals for P3-P2; "
                "family bootstrap intervals for P3-P3 and semantic swap"),
            "P3-P1_seed_points": (
                "five explicit Qwen control realizations; OLMo rows remain "
                "the historical frozen controls"),
            "tiers": {
                "P3-P2": "phase3-confirmatory and phase3-replication",
                "P3-P3": "phase3-confirmatory plus methods sensitivities",
                "semantic_swap": "phase3-development",
            },
        },
    }


def _interval(result: dict) -> list[float] | None:
    if "effect_size_interval" in result:
        return result["effect_size_interval"]["ci95"]
    return result.get("ci95_family_bootstrap")


def _estimate(result: dict) -> float:
    return float(result.get(
        "estimate_equal_family", result.get("estimate")))


def _bar_panel(ax, entries: list[tuple[str, dict]], *,
               ylabel: str, title: str, colors: list[str]) -> None:
    values = [_estimate(result) for _, result in entries]
    intervals = [_interval(result) for _, result in entries]
    x = np.arange(len(entries))
    ax.bar(x, values, color=colors, alpha=0.9)
    for index, (value, interval) in enumerate(zip(values, intervals)):
        if interval is None:
            continue
        ax.errorbar(
            [index], [value],
            yerr=[[value - interval[0]], [interval[1] - value]],
            fmt="none", ecolor="black", capsize=4, lw=1)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xticks(x, [label for label, _ in entries], rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def make_figure(stats: dict, png: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    blue = "#355c7d"
    teal = "#2a9d8f"
    gold = "#e9c46a"
    coral = "#e76f51"
    gray = "#8d99ae"
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))

    seed_items = sorted(
        stats["P3-P1_control_seed_estimates"].items(),
        key=lambda pair: int(pair[0]))
    x = np.arange(len(seed_items))
    y = [value for _, value in seed_items]
    colors = [coral if seed == "31337" else blue
              for seed, _ in seed_items]
    axes[0, 0].scatter(x, y, s=65, c=colors, zorder=3)
    axes[0, 0].plot(x, y, color=gray, lw=1, alpha=0.7)
    axes[0, 0].scatter(
        [len(x)], [stats["P3-P1_boundary_state_seed_31337"]],
        marker="*", s=150, color=teal, zorder=3)
    axes[0, 0].axhline(0, color="black", lw=1, ls="--")
    axes[0, 0].set_xticks(
        list(x) + [len(x)],
        [seed for seed, _ in seed_items] + ["31337\nboundary"])
    axes[0, 0].set_ylabel("P3-P1 estimate (nats)")
    axes[0, 0].set_title("Control-seed and boundary sensitivity")

    _bar_panel(
        axes[0, 1],
        [
            ("confirm\nall", stats["P3-P2"]["confirmatory_all"]),
            ("confirm\nrank≤10", stats["P3-P2"]["confirmatory_exact"]),
            ("confirm\nboundary", stats["P3-P2"][
                "boundary_confirmatory"]),
            ("replicate\nall", stats["P3-P2"]["replication_all"]),
            ("replicate\nrank≤10", stats["P3-P2"][
                "replication_exact"]),
        ],
        ylabel="Tail-rate excess at −1 nat",
        title="P3-P2 survives all release gates",
        colors=[blue, teal, gold, coral, gray],
    )
    _bar_panel(
        axes[1, 0],
        [
            ("raw", stats["P3-P3"]["raw"]),
            ("geometry\nresidual", stats["P3-P3"][
                "geometry_residual"]),
            ("strict\nmatch", stats["P3-P3"]["strict_geometry"]),
            ("boundary", stats["P3-P3"]["boundary"]),
        ],
        ylabel="True − distractor rescue (nats)",
        title="P3-P3 geometry and cohort checks",
        colors=[blue, teal, gold, gray],
    )
    _bar_panel(
        axes[1, 1],
        [
            ("from\nbaseline", stats["semantic_swap"]["from_baseline"]),
            ("vs\nunrelated", stats["semantic_swap"]["vs_unrelated"]),
            ("vs answer\ndirection", stats["semantic_swap"][
                "vs_answer_direction"]),
        ],
        ylabel="Counterfactual preference shift (nats)",
        title="Semantic movement and its key limitation",
        colors=[coral, teal, gray],
    )
    fig.suptitle("Phase 3 release-audit state of record", fontsize=16)
    fig.tight_layout()
    for path in (png, pdf):
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        fig.savefig(
            tmp, dpi=180, bbox_inches="tight",
            format=path.suffix[1:])
        os.replace(tmp, path)
    plt.close(fig)


def main() -> None:
    require_clean_tree(False)
    root = run_root()
    paths = source_paths(root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing release figure inputs: {missing}")
    sources = {name: _payload(path) for name, path in paths.items()}
    inputs = {name: sha256_file(path) for name, path in paths.items()}
    stats = collect_stats(sources)
    out_dir = figures_dir()
    png = out_dir / "p3f06_phase3_release_audit.png"
    pdf = out_dir / "p3f06_phase3_release_audit.pdf"
    make_figure(stats, png, pdf)
    result = (
        root / "metrics/cross_model/release_audit/"
        "p3_release_summary_figure.json")
    command = (
        "python -m "
        "jspace_phase3.experiments.p3_release_summary_figure")
    write_result3(
        stats, result,
        Provenance3(
            evidence_id=EVIDENCE_ID, tier=TIER,
            command=command, inputs=inputs, seed=SEED))
    register(
        EVIDENCE_ID,
        tier=TIER,
        command=command,
        what=(
            "Final Phase 3 release-audit synthesis figure from live "
            "inference, protection, seed, boundary, alias, geometry, and "
            "semantic-swap evidence."),
        outputs=[result, png, pdf],
        inputs=inputs)
    print(json.dumps({
        "result": str(result), "png": str(png), "pdf": str(pdf),
    }, indent=1))


if __name__ == "__main__":
    main()
