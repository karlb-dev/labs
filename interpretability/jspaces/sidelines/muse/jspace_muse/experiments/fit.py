"""120-prompt WikiText J-lens fit (Phase-2 recipe, Muse-retargeted layers)."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch

from ..adapters import load_muse
from ..paths import (
    DRIVE_ROOT,
    FINAL_LAYER,
    FIT_CORPUS,
    FIT_SOURCE_LAYERS,
    LOCAL_WORK,
    ensure_dirs,
)
from ..registry import register
from ..util import atomic_write_json, log, runtime_fingerprint, sha256_file, utc_now

N_PROMPTS = 120
SLICE_SIZE = 30
SYNC_EVERY = 10
MAX_SEQ = 128
SKIP_FIRST = 16


def _load_corpus() -> list[dict]:
    rows = [json.loads(l) for l in FIT_CORPUS.read_text().splitlines()]
    if len(rows) < N_PROMPTS:
        raise RuntimeError(f"corpus has {len(rows)} rows; need {N_PROMPTS}")
    return rows[:N_PROMPTS]


def timing_route(model, prompts: list[str], dim_candidates=(16, 8, 4, 2)) -> dict:
    """Benchmark dim_batch on one short prompt; pick fastest non-OOM."""
    from jlens.fitting import jacobian_for_prompt

    route_path = DRIVE_ROOT / "metrics" / "fit_route.json"
    if route_path.exists():
        return json.loads(route_path.read_text())

    prompt = prompts[0]
    results = []
    chosen = None
    for db in dim_candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            Js, seq_len, n_valid = jacobian_for_prompt(
                model, prompt, FIT_SOURCE_LAYERS,
                target_layer=FINAL_LAYER, dim_batch=db,
                max_seq_len=MAX_SEQ, skip_first=SKIP_FIRST,
            )
            wall = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated()
            row = {
                "dim_batch": db,
                "wall_seconds": wall,
                "peak_vram_bytes": int(peak),
                "seq_len": seq_len,
                "n_valid": n_valid,
                "n_layers": len(Js),
            }
            results.append(row)
            log(f"timing dim_batch={db}: {wall:.1f}s peak={peak/1e9:.1f}GB")
            if chosen is None or wall < chosen["wall_seconds"]:
                chosen = row
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"dim_batch": db, "oom": True})
            log(f"timing dim_batch={db}: OOM")
    if chosen is None:
        raise RuntimeError("all dim_batch candidates OOM")
    route = {
        "dim_batch": chosen["dim_batch"],
        "measured_s_per_prompt": chosen["wall_seconds"],
        "projected_total_seconds": chosen["wall_seconds"] * N_PROMPTS,
        "probe_results": results,
        "source_layers": FIT_SOURCE_LAYERS,
        "target_layer": FINAL_LAYER,
        "max_seq_len": MAX_SEQ,
        "skip_first": SKIP_FIRST,
        "n_prompts": N_PROMPTS,
        "runtime": runtime_fingerprint(),
        "utc": utc_now(),
    }
    atomic_write_json(route, route_path)
    register({
        "evidence_id": "muse-fit-route-v1",
        "what": f"Fit route frozen dim_batch={route['dim_batch']} "
                f"~{route['measured_s_per_prompt']:.1f}s/prompt",
        "command": "python -m jspace_muse.experiments.fit",
        "outputs": [route_path],
    })
    return route


def slice_paths(k: int) -> tuple[Path, Path, Path]:
    return (
        LOCAL_WORK / f"fit_muse_slice{k}.ckpt",
        DRIVE_ROOT / "lens" / f"fit_muse_slice{k}.ckpt",
        DRIVE_ROOT / "lens" / f"muse_slice{k}.pt",
    )


def fit_slice(model, k: int, prompts: list[str], dim_batch: int, metrics: dict):
    import jlens
    from jlens import JacobianLens

    local_ckpt, drive_ckpt, slice_lens = slice_paths(k)
    if slice_lens.exists():
        log(f"slice {k}: already have {slice_lens.name}")
        return JacobianLens.load(str(slice_lens))
    if not local_ckpt.exists() and drive_ckpt.exists():
        log(f"slice {k}: restoring ckpt from Drive")
        shutil.copy2(drive_ckpt, local_ckpt)

    t0, lens = time.time(), None
    for end in range(SYNC_EVERY, SLICE_SIZE + 1, SYNC_EVERY):
        lens = jlens.fit(
            model, prompts[:end],
            source_layers=FIT_SOURCE_LAYERS,
            target_layer=FINAL_LAYER,
            dim_batch=dim_batch,
            max_seq_len=MAX_SEQ,
            skip_first=SKIP_FIRST,
            checkpoint_path=str(local_ckpt),
            checkpoint_every=1,
            resume=True,
        )
        shutil.copy2(local_ckpt, drive_ckpt)
        peak = torch.cuda.max_memory_allocated() / 1e9
        log(f"slice {k}: {end}/{SLICE_SIZE}; peak VRAM {peak:.1f}GB; "
            f"elapsed {time.time()-t0:.0f}s")
        metrics["slices"].setdefault(str(k), {}).update(
            prompts_done=end,
            peak_vram_gb=round(peak, 1),
            elapsed_s=round(time.time() - t0),
        )
        atomic_write_json(metrics, DRIVE_ROOT / "metrics" / "fit.json")

    lens.save(str(slice_lens))
    metrics["slices"][str(k)].update(
        lens_file=slice_lens.name, n_prompts=lens.n_prompts
    )
    atomic_write_json(metrics, DRIVE_ROOT / "metrics" / "fit.json")
    log(f"slice {k}: done -> {slice_lens.name}")
    return lens


def run(dim_batch: int | None = None, slices: list[int] | None = None) -> dict:
    ensure_dirs()
    LOCAL_WORK.mkdir(parents=True, exist_ok=True)
    merged_path = DRIVE_ROOT / "lens" / "muse_glimmer_lens.pt"
    metrics_path = DRIVE_ROOT / "metrics" / "fit.json"

    if merged_path.exists():
        log(f"{merged_path} exists; fit complete")
        return json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    rows = _load_corpus()
    corpus_sha = sha256_file(FIT_CORPUS)
    prompts_all = [r["text"] if "text" in r else r.get("prompt", "") for r in rows]
    # some corpora use different keys
    if not prompts_all[0]:
        prompts_all = [r[list(r.keys())[0]] for r in rows]

    model, hf_model, tokenizer = load_muse()
    log(f"model ready: {model.n_layers}x{model.d_model}")

    if dim_batch is None:
        route = timing_route(model, prompts_all)
        dim_batch = int(route["dim_batch"])
    else:
        route = {"dim_batch": dim_batch, "note": "user override"}

    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"slices": {}}
    metrics.update({
        "model_id": "meta-models/Muse-Glimmer-30B",
        "corpus": str(FIT_CORPUS),
        "corpus_sha256": corpus_sha,
        "n_prompts": N_PROMPTS,
        "source_layers": FIT_SOURCE_LAYERS,
        "target_layer": FINAL_LAYER,
        "dim_batch": dim_batch,
        "max_seq_len": MAX_SEQ,
        "skip_first": SKIP_FIRST,
        "route": route,
        "runtime": runtime_fingerprint(),
        "utc": utc_now(),
    })
    atomic_write_json(metrics, metrics_path)

    which = slices if slices is not None else [0, 1, 2, 3]
    for k in which:
        chunk = prompts_all[k * SLICE_SIZE:(k + 1) * SLICE_SIZE]
        if len(chunk) < SLICE_SIZE:
            raise RuntimeError(f"slice {k} has only {len(chunk)} prompts")
        fit_slice(model, k, chunk, dim_batch, metrics)

    from jlens import JacobianLens
    slice_paths_pt = sorted(DRIVE_ROOT.glob("lens/muse_slice*.pt"))
    if len(slice_paths_pt) < 4:
        log(f"{len(slice_paths_pt)}/4 slices present; merge deferred")
        return metrics

    merged = JacobianLens.merge([JacobianLens.load(str(p)) for p in slice_paths_pt])
    merged.save(str(merged_path))
    metrics["merged"] = {
        "n_prompts": merged.n_prompts,
        "file": merged_path.name,
        "from": [p.name for p in slice_paths_pt],
        "sha256": sha256_file(merged_path),
        "source_layers": sorted(merged.jacobians.keys()),
    }
    atomic_write_json(metrics, metrics_path)
    register({
        "evidence_id": "muse-fit-wikitext120-v1",
        "what": f"Muse Glimmer 120-prompt WikiText J-lens "
                f"({len(FIT_SOURCE_LAYERS)} sources -> L{FINAL_LAYER})",
        "command": "python -m jspace_muse.experiments.fit",
        "outputs": [merged_path, metrics_path],
    })
    log(f"MERGED n={merged.n_prompts} -> {merged_path}")
    return metrics


if __name__ == "__main__":
    import sys
    db = None
    if "--dim-batch" in sys.argv:
        db = int(sys.argv[sys.argv.index("--dim-batch") + 1])
    sl = None
    if "--slices" in sys.argv:
        sl = [int(x) for x in sys.argv[sys.argv.index("--slices") + 1].split(",")]
    run(dim_batch=db, slices=sl)
