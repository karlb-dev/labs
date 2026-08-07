"""Campaign-side comparison of the already-sealed N8-P3-L1 report."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from jspace_part2.lib import sha256_file

from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)

EVIDENCE_ID = "p3-n8-p3-level1-v1"
TIER = "methods"
TOLERANCE = 1e-10


def get(mapping: dict, path: tuple[str, ...]):
    value = mapping
    for key in path:
        value = value[key]
    return value


def main() -> None:
    require_clean_tree("--allow-dirty" in sys.argv)
    report_arg = (
        sys.argv[sys.argv.index("--report") + 1]
        if "--report" in sys.argv else os.environ.get("N8_P3_REPORT"))
    if not report_arg:
        raise RuntimeError("pass --report or set N8_P3_REPORT")
    report_path = Path(report_arg)
    sealed = json.loads(report_path.read_text())
    inference = json.loads((
        metrics_dir("cross_model") / "release_audit"
        / "p3_inference_audit.json").read_text())["payload"]
    protected = json.loads((
        metrics_dir("qwen36-27b") / "release_audit"
        / "protected_answer"
        / "p3_protected_answer_audit.json").read_text())["payload"]
    locked = {
        "confirmatory": json.loads((
            metrics_dir("cross_model")
            / "phase3_locked_analysis.json").read_text())["payload"],
        "replication": json.loads((
            metrics_dir("cross_model")
            / "phase3_locked_analysis_replication.json"
        ).read_text())["payload"],
    }
    comparisons = []

    def compare(side: str, quantity: str, observed: float,
                target: float) -> None:
        comparisons.append({
            "side": side, "quantity": quantity,
            "sealed": float(observed), "campaign": float(target),
            "abs_error": abs(float(observed) - float(target)),
        })

    for side in ("confirmatory", "replication"):
        actual = sealed["results"][side]
        expected = inference[side]
        paths = {
            "P3-P1.estimate": (
                ("P3-P1", "exact_randomization", "estimate"),
                ("P3-P1", "exact_randomization", "estimate")),
            "P3-P1.p": (
                ("P3-P1", "exact_randomization", "p"),
                ("P3-P1", "exact_randomization", "p")),
            "P3-P1.item_weighted": (
                ("P3-P1", "item_weighted"),
                ("P3-P1", "sensitivity_item_weighted")),
            "P3-P1.relation_group_weighted": (
                ("P3-P1", "relation_group_weighted"),
                ("P3-P1", "sensitivity_relation_group_weighted")),
            "P3-P1.median_family_mean": (
                ("P3-P1", "median_family_mean"),
                ("P3-P1", "median_of_family_means")),
            "P3-P2.all.estimate": (
                ("P3-P2", "all_items", "estimate"),
                ("P3-P2_all_items_threshold_curve", "-1.0", "estimate")),
            "P3-P2.all.p": (
                ("P3-P2", "all_items", "p_plus_one"),
                ("P3-P2_all_items_threshold_curve", "-1.0",
                 "p_plus_one")),
        }
        if side == "confirmatory":
            paths |= {
                "P3-P3.estimate": (
                    ("P3-P3", "estimate"), ("P3-P3", "estimate")),
                "P3-P3.p": (
                    ("P3-P3", "p_plus_one"),
                    ("P3-P3", "p_plus_one")),
            }
        for name, (actual_path, expected_path) in paths.items():
            compare(
                side, name, get(actual, actual_path),
                get(expected, expected_path))
        for hypothesis, target in expected[
                "holm_with_plus_one_mc"].items():
            compare(
                side, f"holm.{hypothesis}",
                actual["holm"][hypothesis], target)
        for view, protected_name in (
            ("exact_scored_alias_protected",
             "exact_scored_alias_protected"),
            ("any_accepted_alias_protected",
             "any_accepted_alias_protected"),
        ):
            protected_target = protected["sides"][side][protected_name][
                "threshold_curve"]["-1.0"]
            compare(
                side, f"P3-P2.{view}.estimate",
                actual["P3-P2"][view]["estimate"],
                protected_target["estimate"])
            compare(
                side, f"P3-P2.{view}.p",
                actual["P3-P2"][view]["p_plus_one"],
                protected_target["p_plus_one"])
        target_estimands = locked[side]["estimation_targets"]
        actual_estimands = actual["estimation_targets"]
        for name, actual_name, target_name in (
            ("think_minus_instruct", "think_minus_instruct_thick",
             "think_vs_instruct_thick"),
        ):
            for field in ("estimate", "n_families"):
                compare(
                    side, f"estimand.{name}.{field}",
                    actual_estimands[actual_name][field],
                    target_estimands[target_name][field])
            for index, endpoint in enumerate(("lo", "hi")):
                compare(
                    side, f"estimand.{name}.ci95.{endpoint}",
                    actual_estimands[actual_name]["ci95"][index],
                    target_estimands[target_name]["ci95"][index])
        for model in ("olmo31-think", "olmo31-instruct", "qwen36-27b"):
            actual_target = actual_estimands[
                "bank_s_composition_by_model"][model]
            campaign_target = target_estimands[
                "bank_s_composition_by_model"][model]
            for field in ("estimate", "n_families"):
                compare(
                    side, f"estimand.bank_s.{model}.{field}",
                    actual_target[field], campaign_target[field])
            for index, endpoint in enumerate(("lo", "hi")):
                compare(
                    side, f"estimand.bank_s.{model}.ci95.{endpoint}",
                    actual_target["ci95"][index],
                    campaign_target["ci95"][index])
    worst = max(row["abs_error"] for row in comparisons)
    passed = worst <= TOLERANCE
    payload = {
        "level": "N8-P3-L1",
        "sealed_report_sha256": sha256_file(report_path),
        "sealed_payload_sha256": sealed["payload_sha256"],
        "n_comparisons": len(comparisons),
        "comparisons": comparisons,
        "worst_abs_error": worst,
        "tolerance": TOLERANCE,
        "pass": passed,
        "blindness_contract": sealed["analysis_contract"],
    }
    out = (
        metrics_dir("cross_model") / "release_audit"
        / "p3_n8_phase3_level1_comparison.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        "python -m jspace_phase3.experiments.p3_n8_phase3_compare "
        f"--report {report_path}"
    )
    write_result3(payload, out, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd, seed=4242,
        inputs={"sealed_report": sha256_file(report_path)}))
    markdown = report_path.with_suffix(".md")
    register(
        EVIDENCE_ID, tier=TIER, command=cmd,
        what=(
            f"N8-P3-L1 expected-value-blind analysis reproduction: "
            f"{len(comparisons)} quantities, worst |error|={worst:.3g} "
            f"(tol {TOLERANCE}) — {'PASS' if passed else 'FAIL'}"
        ),
        outputs=[out, report_path, markdown],
        inputs={"sealed_report": sha256_file(report_path)},
    )
    print(json.dumps(payload, indent=1))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
