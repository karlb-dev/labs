"""Qwen follow-up block: verbal-report v2 rerun (INCIDENT or1-001) and
the Qwen instrument cross-over. One Qwen process; runs after the OLMo
lane blocks so lanes never share a process."""
from __future__ import annotations

from ..paths import DRIVE_ROOT
from ..readout import lens_to_device
from . import admission as admission_module
from . import verbal_report
from .crossover import run_lane


def main() -> None:
    model, hf_model, tokenizer = admission_module.load_qwen()
    lens = admission_module.load_qwen_lens()
    lens_to_device(lens, "cuda:0")
    lane_dir = DRIVE_ROOT / "qwen_lane"
    if not (lane_dir / "verbal_report_qwen_v2.json").exists():
        result = verbal_report.run(
            model, lens, lane="qwen", out_dir=lane_dir,
            common={"study_id": "jspace-official-repro-1",
                    "model_id": "Qwen/Qwen3.6-27B",
                    "incident": "or1-001 v2 scoring"},
            suffix="_v2",
        )
        print("VR v2:", result["category_equal_top1"],
              result["category_equal_top5"])
    if not (DRIVE_ROOT / "crossover_qwen" / "crossover_qwen.json").exists():
        summary = run_lane(model, hf_model, lens, lane="qwen")
        print("crossover qwen:", summary["paper_swap_top1_primary"],
              summary["protected_answer_degradation_median"],
              summary["matched_answer_degradation_median"])


if __name__ == "__main__":
    main()
