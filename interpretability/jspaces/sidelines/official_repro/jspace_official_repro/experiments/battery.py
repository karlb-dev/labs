"""OR1.5 extended-battery driver for one lane (complete groups only).

Group A: selectivity-language, selectivity-linecount.
Group B: verbal-introspection, directed-modulation (math+topic),
         dual-task, line-break (R2, last in group).
Group C: capacity, ignition, top-down-summoning.
Idempotent by output existence; bank a complete group before starting
the next (plan §10/§14)."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from ..paths import DRIVE_ROOT
from ..readout import lens_to_device
from . import admission as admission_module
from .introspection import run as run_introspection
from .linebreak import run as run_linebreak
from .modulation import run_directed_modulation, run_dual_task
from .selectivity import run_language, run_linecount
from .structure import run_capacity, run_ignition, run_top_down


def run(lane: str, *, lane_dir: Path | None = None) -> None:
    lane_dir = Path(lane_dir) if lane_dir else DRIVE_ROOT / f"{lane}_battery"
    lane_dir.mkdir(parents=True, exist_ok=True)
    log_path = lane_dir / "battery_log.jsonl"

    def log(stage: str, **fields) -> None:
        with log_path.open("a") as handle:
            handle.write(json.dumps(
                {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "stage": stage, **fields}) + "\n")

    log("start", lane=lane)
    try:
        if lane == "qwen":
            model, hf_model, tokenizer = admission_module.load_qwen()
            lens = admission_module.load_qwen_lens()
        else:
            from .olmo_lane import load_or1_lens

            model, hf_model, tokenizer = admission_module.load_olmo()
            lens = load_or1_lens("merged")
        lens_to_device(lens, "cuda:0")

        groups = [
            ("A", [
                (f"selectivity_language_{lane}.json",
                 lambda: run_language(model, lens, lane=lane, out_dir=lane_dir)),
                (f"selectivity_linecount_{lane}.json",
                 lambda: run_linecount(model, lens, lane=lane, out_dir=lane_dir)),
            ]),
            ("B", [
                (f"verbal_introspection_{lane}.json",
                 lambda: run_introspection(model, lens, lane=lane,
                                           out_dir=lane_dir)),
                (f"directed_modulation_{lane}.json",
                 lambda: run_directed_modulation(model, lens, lane=lane,
                                                 out_dir=lane_dir)),
                (f"dual_task_{lane}.json",
                 lambda: run_dual_task(model, lens, lane=lane,
                                       out_dir=lane_dir)),
                (f"linebreak_{lane}.json",
                 lambda: run_linebreak(model, lens, lane=lane,
                                       out_dir=lane_dir)),
            ]),
            ("C", [
                (f"capacity_{lane}.json",
                 lambda: run_capacity(model, lens, lane=lane,
                                      out_dir=lane_dir)),
                (f"ignition_{lane}.json",
                 lambda: run_ignition(model, lens, lane=lane,
                                      out_dir=lane_dir)),
                (f"top_down_{lane}.json",
                 lambda: run_top_down(model, lens, lane=lane,
                                      out_dir=lane_dir)),
            ]),
        ]
        for group_name, cells in groups:
            log("group_start", group=group_name)
            for output_name, runner in cells:
                if (lane_dir / output_name).exists():
                    log("cell", output=output_name, skipped=True)
                    continue
                result = runner()
                log("cell", output=output_name,
                    wall=result.get("wall_seconds"))
            log("group_done", group=group_name)
        log("done")
    except Exception as exc:  # noqa: BLE001
        log("error", error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-2000:])
        raise


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "qwen")
