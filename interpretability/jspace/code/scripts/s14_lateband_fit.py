# v2 Priority 3a: fit the lens on the LATE band {46, 50, 54, 58, 62}.
#
# v1 fitted 33-70% depth; the paper's workspace band runs ~38-92%, and v1's
# suppressed-CoT profile shows the answer's rank still collapsing L44->L60 —
# into exactly the unfitted region. Same recipe as s3 (v1 corpus, 4x30
# prompts, target L63, dim_batch 8, seq 128, skip 16), same chunked
# checkpointing; backward graphs only span L46->L63 so per-prompt cost is
# far below v1's 155s. Outputs to the v2 run dir:
#   lens/olmo32bthink_late_slice{k}.pt, lens/olmo32bthink_late.pt (merged)
#   metrics/fit_late.json
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (LOCAL_WORK, RUN_DIR, RUN_DIR_V2, atomic_write_json,
                        die, gpu_mem_gb, load_model, log, read_json, seed_all)

import torch
import jlens
from jlens import JacobianLens

LATE_LAYERS = [46, 50, 54, 58, 62]
CORPUS = RUN_DIR / "config" / "prompts" / "fitting_corpus.jsonl"
MERGED = RUN_DIR_V2 / "lens" / "olmo32bthink_late.pt"
METRICS = RUN_DIR_V2 / "metrics" / "fit_late.json"
SLICE_SIZE, SYNC_EVERY = 30, 10
TARGET_LAYER, MAX_SEQ, SKIP_FIRST = 63, 128, 16


def arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def slice_paths(k: int) -> tuple[Path, Path, Path]:
    return (LOCAL_WORK / f"fitlate_slice{k}.ckpt",
            RUN_DIR_V2 / "lens" / f"fitlate_slice{k}.ckpt",
            RUN_DIR_V2 / "lens" / f"olmo32bthink_late_slice{k}.pt")


def fit_slice(model, k: int, prompts: list[str], dim_batch: int,
              metrics: dict) -> JacobianLens:
    local_ckpt, drive_ckpt, slice_lens_path = slice_paths(k)
    if slice_lens_path.exists():
        log(f"late slice {k}: already on Drive; loading")
        return JacobianLens.load(str(slice_lens_path))
    if not local_ckpt.exists() and drive_ckpt.exists():
        log(f"late slice {k}: pulling Drive checkpoint local")
        shutil.copy2(drive_ckpt, local_ckpt)
    t0 = time.time()
    lens = None
    for end in range(SYNC_EVERY, SLICE_SIZE + 1, SYNC_EVERY):
        lens = jlens.fit(
            model, prompts[:end],
            source_layers=LATE_LAYERS, target_layer=TARGET_LAYER,
            dim_batch=dim_batch, max_seq_len=MAX_SEQ, skip_first=SKIP_FIRST,
            checkpoint_path=str(local_ckpt), checkpoint_every=1, resume=True,
        )
        shutil.copy2(local_ckpt, drive_ckpt)
        used, _ = gpu_mem_gb()
        peak = torch.cuda.max_memory_allocated() / 1e9
        log(f"late slice {k}: {end}/{SLICE_SIZE}; VRAM {used:.1f} "
            f"peak {peak:.1f}GB")
        metrics["slices"].setdefault(str(k), {})["prompts_done"] = end
        metrics["slices"][str(k)]["peak_vram_gb"] = round(peak, 1)
        metrics["slices"][str(k)]["elapsed_s"] = round(time.time() - t0)
        atomic_write_json(metrics, METRICS)
    lens.save(str(slice_lens_path))
    metrics["slices"][str(k)]["lens_file"] = slice_lens_path.name
    metrics["slices"][str(k)]["n_prompts"] = lens.n_prompts
    atomic_write_json(metrics, METRICS)
    log(f"late slice {k}: done in {time.time() - t0:.0f}s")
    return lens


def main() -> None:
    seed_all()
    (RUN_DIR_V2 / "lens").mkdir(parents=True, exist_ok=True)
    (RUN_DIR_V2 / "metrics").mkdir(parents=True, exist_ok=True)
    LOCAL_WORK.mkdir(parents=True, exist_ok=True)
    slices = [int(s) for s in arg("--slices", "0,1,2,3").split(",")]
    dim_batch = int(arg("--dim-batch", "8"))
    jlens.configure_logging()
    if MERGED.exists() and "--force" not in sys.argv:
        log(f"{MERGED} exists; nothing to do")
        return
    if not CORPUS.exists():
        die(f"missing {CORPUS}")
    rows = [json.loads(l) for l in CORPUS.read_text().splitlines()]
    metrics = read_json(METRICS) if METRICS.exists() else {"slices": {}}
    metrics.update({"model": "allenai/Olmo-3-32B-Think",
                    "source_layers": LATE_LAYERS,
                    "target_layer": TARGET_LAYER, "dim_batch": dim_batch,
                    "max_seq_len": MAX_SEQ, "skip_first": SKIP_FIRST,
                    "corpus": "v1 fitting_corpus.jsonl (same 4x30 slices)"})
    atomic_write_json(metrics, METRICS)

    model, hf, tok = load_model("main")
    if model.n_layers != 64 or model.d_model != 5120:
        die("unexpected model shape")
    for k in slices:
        prompts = [r["text"] for r in rows[k * SLICE_SIZE:(k + 1) * SLICE_SIZE]]
        fit_slice(model, k, prompts, dim_batch, metrics)

    all_slices = sorted((RUN_DIR_V2 / "lens").glob("olmo32bthink_late_slice*.pt"))
    lenses = [JacobianLens.load(str(p)) for p in all_slices]
    merged = JacobianLens.merge(lenses)
    if len(all_slices) == 4:
        merged.save(str(MERGED))
        metrics["merged"] = {"n_prompts": merged.n_prompts,
                             "file": MERGED.name,
                             "from": [p.name for p in all_slices]}
        atomic_write_json(metrics, METRICS)
        log(f"MERGED late lens ({merged.n_prompts} prompts) -> {MERGED}")
    else:
        log(f"{len(all_slices)}/4 late slices present; merge deferred")


if __name__ == "__main__":
    main()
