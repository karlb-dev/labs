"""OR1-B OLMo fit block driver (OLMO_FIT_CONTRACT).

Stages: timing/dim_batch sentinels -> frozen route -> half A -> half B ->
split-half audit inputs -> merged lens. Every stage durable and
idempotent; the route file freezes before fit prompt 0.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from jlens.lens import JacobianLens

from ..adapters import load_olmo
from ..fitting import (
    SENTINEL_MAX_REL_DIFF,
    HalfFit,
    runtime_sentinel,
    sentinel_rel_diff,
)
from ..layers import FINAL_LAYER, OLMO_FIT_SOURCE_LAYERS
from ..manifests import file_sha256, runtime_fingerprint, write_json
from ..paths import DRIVE_ROOT, FIT_DATA, LOCAL_WORK

FIT_DIR = DRIVE_ROOT / "olmo_fit"
LOCAL_FIT = LOCAL_WORK / "olmo_fit"
CEILING_SECONDS = 18 * 3600
ROUTES = [  # (n_per_half, required s/prompt for 2*n within the ceiling)
    (500, 64.8),
    (250, 129.6),
    (125, 259.2),
]


def _load_fit_rows() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in
            (FIT_DATA / "wikitext_first1000_min600.jsonl").read_text().splitlines()]
    sentinels = [json.loads(line) for line in
                 (FIT_DATA / "wikitext_sentinels_after1000.jsonl").read_text().splitlines()]
    return rows, sentinels


def timing_gate(model, sentinel_rows: list[dict], *, log) -> dict:
    """Benchmark dim_batch candidates on excluded prompts; freeze route."""
    route_path = FIT_DIR / "fit_route.json"
    if route_path.exists():
        return json.loads(route_path.read_text())
    candidates = [8, 4, 2, 1]
    free_bytes = torch.cuda.mem_get_info()[0]
    headroom_note = {"free_bytes_before_timing": int(free_bytes),
                     "consider_16": free_bytes > (15 << 30)}
    if headroom_note["consider_16"]:
        candidates = [16, *candidates]
    results = []
    chosen = None
    for dim_batch in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            first = runtime_sentinel(model, sentinel_rows[0]["text"],
                                     OLMO_FIT_SOURCE_LAYERS,
                                     target_layer=FINAL_LAYER,
                                     dim_batch=dim_batch)
            peak = int(torch.cuda.max_memory_allocated())
            results.append({"dim_batch": dim_batch,
                            "wall_seconds": first["wall_seconds"],
                            "peak_vram_bytes": peak})
            log("timing_probe", dim_batch=dim_batch,
                wall=first["wall_seconds"], peak_gb=peak / 2**30)
            if chosen is None or first["wall_seconds"] < chosen["wall_seconds"]:
                chosen = {"dim_batch": dim_batch, **first, "peak_vram_bytes": peak}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"dim_batch": dim_batch, "oom": True})
            log("timing_probe", dim_batch=dim_batch, oom=True)
    # Two repeats at the chosen value must agree within tolerance.
    repeat = runtime_sentinel(model, sentinel_rows[0]["text"],
                              OLMO_FIT_SOURCE_LAYERS,
                              target_layer=FINAL_LAYER,
                              dim_batch=chosen["dim_batch"])
    rel = sentinel_rel_diff(chosen, repeat)
    if rel > SENTINEL_MAX_REL_DIFF:
        raise RuntimeError(f"sentinel repeats disagree: rel diff {rel}")
    # Wall-time estimate from a second, different prompt (guards against a
    # single-prompt fluke); use the max of the two as the planning number.
    second = runtime_sentinel(model, sentinel_rows[1]["text"],
                              OLMO_FIT_SOURCE_LAYERS,
                              target_layer=FINAL_LAYER,
                              dim_batch=chosen["dim_batch"])
    per_prompt = max(chosen["wall_seconds"], repeat["wall_seconds"],
                     second["wall_seconds"])
    tier = None
    for n_half, budget in ROUTES:
        if per_prompt <= budget:
            tier = n_half
            break
    route = {
        "dim_batch": chosen["dim_batch"],
        "sentinel_norms_sha256": chosen["norms_sha256"],
        "sentinel_repeat_rel_diff": rel,
        "measured_s_per_prompt": per_prompt,
        "probe_results": results,
        "headroom": headroom_note,
        "n_per_half": tier,
        "merged_n": tier * 2 if tier else None,
        "blocked": tier is None,
        "ceiling_seconds": CEILING_SECONDS,
        "projected_total_seconds": (per_prompt * tier * 2) if tier else None,
        "runtime": runtime_fingerprint(),
        "sentinel": {k: chosen[k] for k in ("norms", "seq_len", "n_valid",
                                            "norms_sha256")},
    }
    write_json(route_path, route)
    log("route_frozen", **{k: route[k] for k in
                           ("dim_batch", "measured_s_per_prompt", "n_per_half",
                            "blocked")})
    return route


def run() -> None:
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_FIT.mkdir(parents=True, exist_ok=True)
    log_path = FIT_DIR / "fit_driver_log.jsonl"

    def log(stage: str, **fields) -> None:
        with log_path.open("a") as handle:
            handle.write(json.dumps(
                {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "stage": stage, **fields}) + "\n")

    log("start")
    model, hf_model, tokenizer = load_olmo()
    if model.d_model != 5120:
        log("d_model", value=model.d_model)  # recorded fact; shapes derive
    rows, sentinel_rows = _load_fit_rows()
    route = timing_gate(model, sentinel_rows, log=log)
    if route["blocked"]:
        log("blocked", why="no tier fits the 18 GPU-hour ceiling")
        return
    n_half = route["n_per_half"]
    sentinel = {"norms_sha256": route["sentinel_norms_sha256"],
                **route["sentinel"]}
    halves = {"A": [r for r in rows if r["half"] == "A"][:n_half],
              "B": [r for r in rows if r["half"] == "B"][:n_half]}
    lenses = {}
    for half_name in ("A", "B"):
        marker = FIT_DIR / f"half_{half_name}_complete.json"
        fitter = HalfFit(
            half=half_name, prompts=halves[half_name],
            source_layers=OLMO_FIT_SOURCE_LAYERS, target_layer=FINAL_LAYER,
            dim_batch=route["dim_batch"], local_dir=LOCAL_FIT,
            drive_dir=FIT_DIR, sentinel=sentinel,
            milestones=(125, 250, 500),
        )
        if not marker.exists():
            log("half_start", half=half_name, n=n_half)
            summary = fitter.run(model)
            lens = fitter.final_lens()
            lens_path = LOCAL_FIT / f"olmo_or1_half_{half_name}.pt"
            lens.save(str(lens_path))
            drive_lens = FIT_DIR / f"olmo_or1_half_{half_name}.pt"
            import shutil

            shutil.copy2(lens_path, drive_lens)
            if file_sha256(drive_lens) != file_sha256(lens_path):
                raise RuntimeError("half lens Drive copy hash mismatch")
            write_json(marker, {**summary,
                                "lens_sha256": file_sha256(lens_path),
                                "lens_path": str(drive_lens)})
            log("half_done", half=half_name, n_done=summary["n_done"],
                skipped=len(summary["skipped"]))
        lenses[half_name] = FIT_DIR / f"olmo_or1_half_{half_name}.pt"
    # ------------------------------------------------------------- merge
    merged_marker = FIT_DIR / "merged_complete.json"
    if not merged_marker.exists():
        half_a = JacobianLens.load(str(lenses["A"]))
        half_b = JacobianLens.load(str(lenses["B"]))
        merged = JacobianLens.merge([half_a, half_b])
        merged_path = LOCAL_FIT / "olmo_or1_merged.pt"
        merged.save(str(merged_path))
        import shutil

        drive_merged = FIT_DIR / "olmo_or1_merged.pt"
        shutil.copy2(merged_path, drive_merged)
        if file_sha256(drive_merged) != file_sha256(merged_path):
            raise RuntimeError("merged lens Drive copy hash mismatch")
        write_json(merged_marker, {
            "n_prompts": merged.n_prompts,
            "source_layers": merged.source_layers,
            "lens_sha256": file_sha256(merged_path),
            "lens_path": str(drive_merged),
        })
        log("merged", n_prompts=merged.n_prompts)
    log("done")


if __name__ == "__main__":
    run()
