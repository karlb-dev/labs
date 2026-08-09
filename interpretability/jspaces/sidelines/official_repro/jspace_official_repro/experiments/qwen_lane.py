"""OR1-A Qwen lane driver: admission -> six evals -> verbal report ->
flexible generalization -> probe-swap raw arm.

Stages are idempotent by output existence (immutable files); a resumed
session re-runs only what is missing. Registration happens from the
control session after each boundary — this driver only produces outputs.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from ..layers import PAPER_BAND, PAPER_GRID_SOURCES
from ..lens_eval import EVAL_SETS, aggregate_pass_at_k, run_eval_set, save_result
from ..manifests import write_json
from ..paths import DRIVE_ROOT
from ..readout import lens_to_device
from . import admission as admission_module
from . import flexible_generalization, probe_swap, verbal_report

LANE_DIR = DRIVE_ROOT / "qwen_lane"


def run(*, smoke: bool = False, lane_dir: Path | None = None) -> None:
    lane_dir = Path(lane_dir) if lane_dir else LANE_DIR
    lane_dir.mkdir(parents=True, exist_ok=True)
    log_path = lane_dir / "driver_log.jsonl"

    def log(stage: str, **fields) -> None:
        with log_path.open("a") as handle:
            handle.write(json.dumps(
                {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "stage": stage, **fields}) + "\n")

    log("start", smoke=smoke)
    model = lens = None
    try:
        # ---------------------------------------------------- admission
        admission_path = lane_dir / "qwen_admission.json"
        model, hf_model, tokenizer = admission_module.load_qwen()
        lens = admission_module.load_qwen_lens()
        lens_to_device(lens, "cuda:0")
        if not admission_path.exists():
            result = admission_module.run_admission_with(
                model, hf_model, tokenizer, lens, lane="qwen",
                out_dir=lane_dir)
            log("admission", ok=True,
                parity=result["readout_parity"]["max_abs_diff"],
                g_fold_min=result["g_folding"]["min_cosine"],
                noop=result["noop"]["ok"])
        else:
            log("admission", skipped=True)
        common = {"study_id": "jspace-official-repro-1",
                  "model_id": "Qwen/Qwen3.6-27B"}

        # -------------------------------------------------------- evals
        max_items = 3 if smoke else None
        for set_name in EVAL_SETS:
            out = lane_dir / f"eval_{set_name}.json"
            if out.exists():
                log("eval", set=set_name, skipped=True)
                continue
            result = run_eval_set(model, lens, set_name,
                                  source_layers=PAPER_GRID_SOURCES,
                                  max_items=max_items)
            result["aggregate_jlens"] = aggregate_pass_at_k(result, which="jlens")
            result["aggregate_logit"] = aggregate_pass_at_k(result, which="logit")
            save_result(result, out)
            log("eval", set=set_name,
                jlens_pass20=result["aggregate_jlens"]["token_valid"]["pass@20"],
                logit_pass20=result["aggregate_logit"]["token_valid"]["pass@20"],
                wall=result["wall_seconds"])
        # All-source-layer sensitivity (Qwen only, named secondary).
        for set_name in EVAL_SETS:
            out = lane_dir / f"eval_alllayers_{set_name}.json"
            if out.exists():
                continue
            if smoke:
                continue
            result = run_eval_set(model, lens, set_name,
                                  source_layers=list(range(63)),
                                  max_items=max_items)
            result["aggregate_jlens"] = aggregate_pass_at_k(result, which="jlens")
            result["aggregate_logit"] = aggregate_pass_at_k(result, which="logit")
            save_result(result, out)
            log("eval_alllayers", set=set_name,
                jlens_pass20=result["aggregate_jlens"]["token_valid"]["pass@20"])

        # ------------------------------------------------------- core
        if not (lane_dir / "verbal_report_qwen.json").exists():
            result = verbal_report.run(model, lens, lane="qwen",
                                       out_dir=lane_dir, common=common)
            log("verbal_report",
                cat_equal_top1=result["category_equal_top1"],
                top5=result["category_equal_top5"],
                wall=result["wall_seconds"])
        if not smoke:
            if not (lane_dir / "flexible_generalization_qwen.json").exists():
                result = flexible_generalization.run(
                    model, lens, lane="qwen", out_dir=lane_dir, common=common)
                log("flexible_generalization",
                    capable_a1=result["category_equal_capable_top1_alpha1"],
                    capable_a2=result["category_equal_capable_top1_alpha2"],
                    wall=result["wall_seconds"])
            if not (lane_dir / "probe_swap_qwen.json").exists():
                result = probe_swap.run(model, lens, lane="qwen",
                                        out_dir=lane_dir, common=common)
                log("probe_swap", capable_top1=result["capable_top1"],
                    diagnostic_top1=result["diagnostic_top1"],
                    wall=result["wall_seconds"])
        log("done")
    except Exception as exc:  # noqa: BLE001 — driver must record the failure
        log("error", error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-2000:])
        raise


if __name__ == "__main__":
    import sys

    run(smoke="--smoke" in sys.argv,
        lane_dir=(sys.argv[sys.argv.index("--dir") + 1]
                  if "--dir" in sys.argv else None))
