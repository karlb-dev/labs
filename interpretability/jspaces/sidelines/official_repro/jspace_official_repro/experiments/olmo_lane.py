"""OR1-C OLMo lane driver: admission -> lens-eval comparator grid ->
core (verbal report, flexible generalization, probe-swap raw arm).

Lens views (plan §9.1): merged OR1 lens on the 24-layer paper grid
(primary); half A / half B stability views (same subset); frozen campaign
lens on its full 21-layer set (secondary historical) and on the 9-layer
paper-grid intersection (fair concordance); logit lens baseline.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import torch

from jlens.lens import JacobianLens

from ..layers import (
    CAMPAIGN_OLMO_LENS_LAYERS,
    PAPER_BAND,
    PAPER_CAMPAIGN_INTERSECTION,
    PAPER_GRID_SOURCES,
)
from ..lens_eval import EVAL_SETS, aggregate_pass_at_k, run_eval_set, save_result
from ..manifests import file_sha256
from ..paths import CAMPAIGN_OLMO_LENS, CAMPAIGN_OLMO_LENS_SHA256, DRIVE_ROOT
from ..readout import lens_to_device
from . import admission as admission_module
from . import flexible_generalization, probe_swap, verbal_report

LANE_DIR = DRIVE_ROOT / "olmo_lane"
FIT_DIR = DRIVE_ROOT / "olmo_fit"


def load_or1_lens(name: str) -> JacobianLens:
    path = FIT_DIR / f"olmo_or1_{name}.pt"
    lens = JacobianLens.load(str(path))
    for layer, J in lens.jacobians.items():
        if not torch.isfinite(J).all():
            raise RuntimeError(f"non-finite lens layer {layer} in {name}")
    return lens


def load_campaign_lens() -> JacobianLens:
    if file_sha256(CAMPAIGN_OLMO_LENS) != CAMPAIGN_OLMO_LENS_SHA256:
        raise RuntimeError("frozen campaign lens hash mismatch")
    lens = JacobianLens.load(str(CAMPAIGN_OLMO_LENS))
    if lens.source_layers != CAMPAIGN_OLMO_LENS_LAYERS:
        raise RuntimeError(
            f"campaign lens layers {lens.source_layers} != pinned")
    return lens


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
    try:
        model, hf_model, tokenizer = admission_module.load_olmo()
        merged = load_or1_lens("merged")
        lens_to_device(merged, "cuda:0")
        # ---------------------------------------------------- admission
        admission_path = lane_dir / "olmo_admission.json"
        if not admission_path.exists():
            result = admission_module.run_admission_with(
                model, hf_model, tokenizer, merged, lane="olmo",
                out_dir=lane_dir)
            log("admission", ok=True,
                parity=result["readout_parity"]["max_abs_diff"],
                g_fold_min=result["g_folding"]["min_cosine"],
                noop=result["noop"]["ok"])
        else:
            log("admission", skipped=True)
        common = {"study_id": "jspace-official-repro-1",
                  "model_id": "allenai/Olmo-3.1-32B-Instruct"}
        max_items = 3 if smoke else None

        # ------------------------------------- lens-eval comparator grid
        views = [("merged", "eval_", None, PAPER_GRID_SOURCES)]
        if not smoke:
            views += [
                ("halfA", "eval_halfA_", "half_A", PAPER_GRID_SOURCES),
                ("halfB", "eval_halfB_", "half_B", PAPER_GRID_SOURCES),
                ("campaign", "eval_campaign_", "campaign",
                 CAMPAIGN_OLMO_LENS_LAYERS),
                ("merged9", "eval_merged9_", None, PAPER_CAMPAIGN_INTERSECTION),
                ("campaign9", "eval_campaign9_", "campaign",
                 PAPER_CAMPAIGN_INTERSECTION),
            ]
        loaded: dict[str, JacobianLens] = {"merged": merged}
        for view_name, prefix, lens_key, layers in views:
            if lens_key == "half_A" and "half_A" not in loaded:
                loaded["half_A"] = load_or1_lens("half_A")
                lens_to_device(loaded["half_A"], "cuda:0")
            if lens_key == "half_B" and "half_B" not in loaded:
                loaded["half_B"] = load_or1_lens("half_B")
                lens_to_device(loaded["half_B"], "cuda:0")
            if lens_key == "campaign" and "campaign" not in loaded:
                loaded["campaign"] = load_campaign_lens()
                lens_to_device(loaded["campaign"], "cuda:0")
            lens = loaded[lens_key] if lens_key else merged
            usable = [l for l in layers if l in lens.source_layers]
            if usable != list(layers):
                log("view_layers_clipped", view=view_name,
                    missing=[l for l in layers if l not in usable])
            for set_name in EVAL_SETS:
                out = lane_dir / f"{prefix}{set_name}.json"
                if out.exists():
                    continue
                result = run_eval_set(model, lens, set_name,
                                      source_layers=usable,
                                      max_items=max_items)
                result["view"] = view_name
                result["aggregate_jlens"] = aggregate_pass_at_k(result, which="jlens")
                result["aggregate_logit"] = aggregate_pass_at_k(result, which="logit")
                save_result(result, out)
                log("eval", view=view_name, set=set_name,
                    jlens_pass20=result["aggregate_jlens"]["token_valid"]["pass@20"],
                    logit_pass20=result["aggregate_logit"]["token_valid"]["pass@20"])
            # Free half-lens GPU memory once its view is done.
            if lens_key in ("half_A", "half_B") and lens_key in loaded:
                lens_to_device(loaded[lens_key], "cpu")
                torch.cuda.empty_cache()

        # ------------------------------------------------------- core
        if not (lane_dir / "verbal_report_olmo.json").exists():
            result = verbal_report.run(model, merged, lane="olmo",
                                       out_dir=lane_dir, common=common)
            log("verbal_report", cat_equal_top1=result["category_equal_top1"],
                top5=result["category_equal_top5"])
        if not smoke:
            if not (lane_dir / "flexible_generalization_olmo.json").exists():
                result = flexible_generalization.run(
                    model, merged, lane="olmo", out_dir=lane_dir, common=common)
                log("flexible_generalization",
                    capable_a1=result["category_equal_capable_top1_alpha1"],
                    capable_a2=result["category_equal_capable_top1_alpha2"])
            if not (lane_dir / "probe_swap_olmo.json").exists():
                result = probe_swap.run(model, merged, lane="olmo",
                                        out_dir=lane_dir, common=common)
                log("probe_swap", capable_top1=result["capable_top1"],
                    diagnostic_top1=result["diagnostic_top1"])
            from .slice_example import capture

            capture(model, merged, out_dir=lane_dir, lane="olmo")
            log("slices", ok=True)
        log("done")
    except Exception as exc:  # noqa: BLE001
        log("error", error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-2000:])
        raise


if __name__ == "__main__":
    import sys

    run(smoke="--smoke" in sys.argv,
        lane_dir=(sys.argv[sys.argv.index("--dir") + 1]
                  if "--dir" in sys.argv else None))
