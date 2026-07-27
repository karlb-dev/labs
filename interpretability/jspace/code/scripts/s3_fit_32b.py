# Phase 1b: fit the Jacobian lens on Olmo-3-32B-Think, chunked + resumable.
#
# 4 slices x 30 WikiText prompts. Within a slice, jlens.fit checkpoints after
# every prompt to fast local disk; the local checkpoint is copied to Drive
# every SYNC_EVERY prompts, so a dead VM loses at most SYNC_EVERY prompts.
# On a fresh VM the Drive checkpoint is pulled back local and fit resumes.
# Each finished slice is saved as its own fp16 lens on Drive; the final lens
# is the n-weighted merge. Usage:
#   python scripts/s3_fit_32b.py [--slices 0,1,2,3] [--dim-batch 8]
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (LOCAL_WORK, RUN_DIR, SOURCE_LAYERS_32B,
                        atomic_write_json, die, ensure_dirs, gpu_mem_gb,
                        load_model, log, read_json, seed_all)

import torch
import jlens
from jlens import JacobianLens

CORPUS = RUN_DIR / "config" / "prompts" / "fitting_corpus.jsonl"
MERGED = RUN_DIR / "lens" / "olmo32bthink_lens.pt"
METRICS = RUN_DIR / "metrics" / "fit_32b.json"
SLICE_SIZE, SYNC_EVERY = 30, 10
TARGET_LAYER, MAX_SEQ, SKIP_FIRST = 63, 128, 16


def arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def slice_paths(k: int) -> tuple[Path, Path, Path]:
    return (LOCAL_WORK / f"fit32b_slice{k}.ckpt",
            RUN_DIR / "lens" / f"fit32b_slice{k}.ckpt",
            RUN_DIR / "lens" / f"olmo32bthink_slice{k}.pt")


def load_metrics() -> dict:
    return read_json(METRICS) if METRICS.exists() else {"slices": {}}


def fit_slice(model, k: int, prompts: list[str], dim_batch: int,
              metrics: dict) -> JacobianLens:
    local_ckpt, drive_ckpt, slice_lens_path = slice_paths(k)
    if slice_lens_path.exists():
        log(f"slice {k}: {slice_lens_path.name} already on Drive; loading")
        return JacobianLens.load(str(slice_lens_path))
    if not local_ckpt.exists() and drive_ckpt.exists():
        log(f"slice {k}: pulling Drive checkpoint back to local disk")
        shutil.copy2(drive_ckpt, local_ckpt)

    t0 = time.time()
    lens = None
    for end in range(SYNC_EVERY, SLICE_SIZE + 1, SYNC_EVERY):
        lens = jlens.fit(
            model, prompts[:end],
            source_layers=SOURCE_LAYERS_32B, target_layer=TARGET_LAYER,
            dim_batch=dim_batch, max_seq_len=MAX_SEQ, skip_first=SKIP_FIRST,
            checkpoint_path=str(local_ckpt), checkpoint_every=1, resume=True,
        )
        shutil.copy2(local_ckpt, drive_ckpt)
        used, total = gpu_mem_gb()
        peak = torch.cuda.max_memory_allocated() / 1e9
        log(f"slice {k}: {end}/{SLICE_SIZE} prompts; ckpt -> Drive; "
            f"VRAM now {used:.1f}GB peak {peak:.1f}GB")
        metrics["slices"].setdefault(str(k), {})["prompts_done"] = end
        metrics["slices"][str(k)]["peak_vram_gb"] = round(peak, 1)
        metrics["slices"][str(k)]["elapsed_s"] = round(time.time() - t0)
        atomic_write_json(metrics, METRICS)

    lens.save(str(slice_lens_path))  # fp16 on Drive
    metrics["slices"][str(k)]["lens_file"] = slice_lens_path.name
    metrics["slices"][str(k)]["n_prompts"] = lens.n_prompts
    atomic_write_json(metrics, METRICS)
    log(f"slice {k}: done in {time.time()-t0:.0f}s -> {slice_lens_path.name}")
    return lens


def main() -> None:
    ensure_dirs()
    seed_all()
    slices = [int(s) for s in arg("--slices", "0,1,2,3").split(",")]
    dim_batch = int(arg("--dim-batch", "8"))
    jlens.configure_logging()  # per-prompt seconds + convergence to stderr

    if MERGED.exists() and "--force" not in sys.argv:
        log(f"{MERGED} exists; nothing to do")
        return
    if not CORPUS.exists():
        die(f"missing {CORPUS}; run s2_corpus.py first")
    rows = [json.loads(l) for l in CORPUS.read_text().splitlines()]
    fit_cfg = read_json(RUN_DIR / "config" / "fit_config.json")["fit_geometry"]
    if fit_cfg["source_layers"] != SOURCE_LAYERS_32B:
        die("fit_config.json disagrees with SOURCE_LAYERS_32B")

    metrics = load_metrics()
    metrics.update({"model": "allenai/Olmo-3-32B-Think",
                    "source_layers": SOURCE_LAYERS_32B,
                    "target_layer": TARGET_LAYER, "dim_batch": dim_batch,
                    "max_seq_len": MAX_SEQ, "skip_first": SKIP_FIRST})
    atomic_write_json(metrics, METRICS)

    model, hf, tok = load_model("main")
    if model.n_layers != 64 or model.d_model != 5120:
        die(f"expected 64 layers x 5120, got {model.n_layers} x {model.d_model}")

    done = []
    for k in slices:
        prompts = [r["text"] for r in rows[k * SLICE_SIZE:(k + 1) * SLICE_SIZE]]
        done.append(fit_slice(model, k, prompts, dim_batch, metrics))

    # Merge whatever slices exist on Drive (supports partial multi-run fits).
    all_slices = sorted(RUN_DIR.glob("lens/olmo32bthink_slice*.pt"))
    lenses = [JacobianLens.load(str(p)) for p in all_slices]
    merged = JacobianLens.merge(lenses)
    if len(all_slices) == 4:
        merged.save(str(MERGED))
        metrics["merged"] = {"n_prompts": merged.n_prompts,
                             "file": MERGED.name,
                             "from": [p.name for p in all_slices]}
        atomic_write_json(metrics, METRICS)
        log(f"MERGED {len(all_slices)} slices ({merged.n_prompts} prompts) -> {MERGED}")
    else:
        log(f"{len(all_slices)}/4 slices present; merge deferred")


if __name__ == "__main__":
    main()
