"""Register the OR1-A Qwen boundary events (plan §4.5 boundaries 3-5).

Run only after the lane driver completed and the tree is clean; verifies
stage outputs exist before appending events.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .. import registry
from ..lens_eval import EVAL_SETS
from ..paths import DRIVE_ROOT, REPORTS, STUDY_ROOT

LANE = DRIVE_ROOT / "qwen_lane"


def _require(path: Path) -> Path:
    if not path.exists():
        raise RuntimeError(f"missing stage output: {path}")
    return path


def main() -> None:
    transcript = REPORTS / "conformance_pytest_transcript.txt"
    result = subprocess.run(
        ["python", "-m", "pytest", str(STUDY_ROOT / "tests"), "-q"],
        capture_output=True, text=True,
    )
    transcript.write_text(result.stdout[-4000:] + result.stderr[-1000:])
    if result.returncode != 0:
        raise RuntimeError("conformance pytest failed at registration time")

    admission = _require(LANE / "qwen_admission.json")
    admission_data = json.loads(admission.read_text())
    assert admission_data["readout_parity"]["ok"]
    assert admission_data["noop"]["ok"]

    registry.create(
        "or1-conformance-v1", tier="methods",
        what=("OR1.1 conformance: 31-test CPU suite (swap algebra, fit "
              "identity vs upstream, resume, render goldens) plus Qwen "
              "model-backed readout parity (exact), alpha-0 no-op (exact), "
              "directional hook check, g-folding audit (material, D9), "
              "repeatability."),
        command="python -m pytest sidelines/official_repro/tests -q; qwen_lane admission",
        outputs=[transcript, admission],
        inputs={"g_folding_min_cosine": admission_data["g_folding"]["min_cosine"]},
    )
    registry.create(
        "or1-qwen-lens-admission-v1", tier="methods",
        what=("Qwen lane admission: pinned model+lens identity, 48 GDN "
              "blocks, ten readout sentinels (boot probe reads Italy@39 -> "
              "euro@58), exact repeatability."),
        command="python -m jspace_official_repro.experiments.qwen_lane",
        outputs=[admission],
    )
    eval_outputs = []
    for prefix in ("eval_", "eval_alllayers_"):
        for set_name in EVAL_SETS:
            eval_outputs.append(_require(LANE / f"{prefix}{set_name}.json"))
    registry.create(
        "or1-qwen-lens-evals-v1", tier="development",
        what=("Six released lens evaluations on Qwen, paper-grid primary "
              "(24 source layers) + all-source-layer sensitivity; J-lens vs "
              "logit lens pass@{1,5,20}, token-valid primary denominators."),
        command="python -m jspace_official_repro.experiments.qwen_lane",
        outputs=eval_outputs,
    )
    registry.create(
        "or1-qwen-verbal-report-v1", tier="development",
        what=("Verbal report on Qwen: release-literal candidate rule, D1 "
              "generation-boundary scoring, alpha=1 paper band, "
              "paper-text top-10 exclusion as recorded sensitivity."),
        command="python -m jspace_official_repro.experiments.qwen_lane",
        outputs=[_require(LANE / "verbal_report_qwen.json"),
                 _require(LANE / "verbal_report_qwen_raw.jsonl")],
    )
    registry.create(
        "or1-qwen-flexible-generalization-v1", tier="development",
        what=("Flexible generalization on Qwen: 192 ordered swaps, alpha 1 "
              "primary + alpha 2 full sensitivity, capability-conditioned "
              "primary with three-population accounting."),
        command="python -m jspace_official_repro.experiments.qwen_lane",
        outputs=[_require(LANE / "flexible_generalization_qwen.json"),
                 _require(LANE / "flexible_generalization_qwen_raw.jsonl")],
    )
    registry.create(
        "or1-qwen-probe-swap-v1", tier="development",
        what=("Probe-swap raw J-lens token arm on Qwen "
              "(prompt_exact_representation_adapted_raw_jlens); official "
              "probe arm NOT_IDENTIFIED_FROM_RELEASE."),
        command="python -m jspace_official_repro.experiments.qwen_lane",
        outputs=[_require(LANE / "probe_swap_qwen.json"),
                 _require(LANE / "probe_swap_qwen_raw.jsonl")],
    )
    print("registered 6 Qwen-boundary events")


if __name__ == "__main__":
    main()
