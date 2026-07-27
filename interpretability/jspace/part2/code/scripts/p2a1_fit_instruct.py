# A1 phase 1: fit a 120-prompt J-lens on Olmo-3.1-32B-Instruct.
#
# Triggered by the preregistered A0 FAIL rule (transferred Think lens read
# probes + pass@5/@20 at donor level but multihop pass@1 0.217 < 0.2408
# floor). Fitting the sibling's own lens disambiguates: bridge readout
# recovers → the miss was J-map drift; stays ~0.22 → Instruct genuinely
# resolves the bridge more weakly (an H1-flavored capacity datum).
#
# Identical recipe to part-1 s3 (same WikiText corpus file, same slices,
# same geometry: 21 source layers, target 63, dim_batch 8, seq 128, skip 16)
# so dictionaries stay commensurable and B1's fit-size scaling can reuse the
# slices. Chunked + resumable exactly like s3: local ckpt every prompt,
# Drive copy every 10, per-slice fp16 lenses, n-weighted merge.
#
# Usage: python scripts/p2a1_fit_instruct.py [--slices 0,1,2,3] [--dim-batch 8]
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import (LOCAL_WORK, RUN_DIR, RUN_DIR_P2, SOURCE_LAYERS_32B,
                        atomic_write_json, die, gpu_mem_gb, log,
                        p2_load_model, p2_metrics_dir, read_json, seed_all)

import torch
import jlens
from jlens import JacobianLens

CORPUS = RUN_DIR / "config" / "prompts" / "fitting_corpus.jsonl"  # part-1 file, reused verbatim
MERGED = RUN_DIR_P2 / "lens" / "olmo31instruct_lens.pt"
METRICS = None  # set in main via p2_metrics_dir
SLICE_SIZE, SYNC_EVERY = 30, 10
TARGET_LAYER, MAX_SEQ, SKIP_FIRST = 63, 128, 16


def arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def slice_paths(k: int) -> tuple[Path, Path, Path]:
    return (LOCAL_WORK / f"fit31i_slice{k}.ckpt",
            RUN_DIR_P2 / "lens" / f"fit31i_slice{k}.ckpt",
            RUN_DIR_P2 / "lens" / f"olmo31instruct_slice{k}.pt")


def fit_slice(model, k: int, prompts: list[str], dim_batch: int,
              metrics: dict, metrics_path: Path) -> JacobianLens:
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
        atomic_write_json(metrics, metrics_path)

    lens.save(str(slice_lens_path))
    metrics["slices"][str(k)]["lens_file"] = slice_lens_path.name
    metrics["slices"][str(k)]["n_prompts"] = lens.n_prompts
    atomic_write_json(metrics, metrics_path)
    log(f"slice {k}: done in {time.time()-t0:.0f}s -> {slice_lens_path.name}")
    return lens


def main() -> None:
    seed_all()
    metrics_path = p2_metrics_dir("olmo31-instruct") / "fit.json"
    LOCAL_WORK.mkdir(parents=True, exist_ok=True)
    slices = [int(s) for s in arg("--slices", "0,1,2,3").split(",")]
    dim_batch = int(arg("--dim-batch", "8"))
    jlens.configure_logging()

    if MERGED.exists() and "--force" not in sys.argv:
        log(f"{MERGED} exists; nothing to do")
        return
    if not CORPUS.exists():
        die(f"missing part-1 corpus {CORPUS}")
    rows = [json.loads(l) for l in CORPUS.read_text().splitlines()]

    metrics = read_json(metrics_path) if metrics_path.exists() else {"slices": {}}
    metrics.update({"model": "allenai/Olmo-3.1-32B-Instruct",
                    "corpus": "part-1 fitting_corpus.jsonl (WikiText-103, seed 0, first 120)",
                    "source_layers": SOURCE_LAYERS_32B,
                    "target_layer": TARGET_LAYER, "dim_batch": dim_batch,
                    "max_seq_len": MAX_SEQ, "skip_first": SKIP_FIRST})
    atomic_write_json(metrics, metrics_path)

    model, hf, tok = p2_load_model("olmo31-instruct")
    if model.n_layers != 64 or model.d_model != 5120:
        die(f"expected 64 layers x 5120, got {model.n_layers} x {model.d_model}")

    for k in slices:
        prompts = [r["text"] for r in rows[k * SLICE_SIZE:(k + 1) * SLICE_SIZE]]
        fit_slice(model, k, prompts, dim_batch, metrics, metrics_path)

    all_slices = sorted(RUN_DIR_P2.glob("lens/olmo31instruct_slice*.pt"))
    lenses = [JacobianLens.load(str(p)) for p in all_slices]
    merged = JacobianLens.merge(lenses)
    if len(all_slices) == 4:
        merged.save(str(MERGED))
        metrics["merged"] = {"n_prompts": merged.n_prompts, "file": MERGED.name,
                             "from": [p.name for p in all_slices]}
        atomic_write_json(metrics, metrics_path)
        log(f"MERGED {len(all_slices)} slices ({merged.n_prompts} prompts) -> {MERGED}")
    else:
        log(f"{len(all_slices)}/4 slices present; merge deferred")


if __name__ == "__main__":
    main()
