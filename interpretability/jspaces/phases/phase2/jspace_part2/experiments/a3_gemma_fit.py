# A3 — full 120-prompt J-lens fit on Gemma-4-31B-it.
#
# The deep-band sweep (a3-gemma-deepband-logit-v1) found the output basis
# opaque across the whole mid-band (median answer rank 23k-90k at L24-38
# of 60) with a sharp readability transition at L42-44 (352 -> 23; <=10
# by L48; rank 1 from L52). The paper's relative depths (37/50/62% ->
# L22/30/37) all sit in the opaque zone. This fit gives the JACOBIAN
# transport its honest chance to rescue mid-band readability where the
# vanilla logit lens fails (the 2-prompt micro-fit's rank-52k reading is
# not that test): source layers span paper-relative depths (22, 30, 37),
# the transition (40, 42, 44), and the readable zone (48, 52).
#
# Recipe otherwise IDENTICAL to the part-1/part-2 OLMo fits (same WikiText
# corpus file, same slicing, target = last layer, dim_batch 8, seq 128,
# skip 16) so dictionaries stay commensurable. Checkpoint/resume contract:
# local ckpt every prompt, Drive ckpt copy every 10 prompts, per-slice
# fp16 lenses on Drive, n-weighted merge; same command resumes.
#
# Usage: python -m jspace_part2.experiments.a3_gemma_fit \
#          [--slices 0,1,2,3] [--dim-batch 8] [--allow-dirty]
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

MODEL = "/content/models/gemma4-31b-it"
RUN_DIR_P1 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "2026-07-25_1726")
RUN_DIR_P2 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "part2_20260727")
CORPUS = RUN_DIR_P1 / "config" / "prompts" / "fitting_corpus.jsonl"
LOCAL_WORK = Path("/content/sl1_work")
MERGED = RUN_DIR_P2 / "lens" / "gemma431_lens.pt"
METRICS = RUN_DIR_P2 / "metrics" / "gemma4-31b" / "fit.json"
SOURCE_LAYERS = [22, 30, 37, 40, 42, 44, 48, 52]
TARGET_LAYER = 59
SLICE_SIZE, SYNC_EVERY = 30, 10
MAX_SEQ, SKIP_FIRST = 128, 16


def arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def slice_paths(k: int) -> tuple[Path, Path, Path]:
    return (LOCAL_WORK / f"fitg4_slice{k}.ckpt",
            RUN_DIR_P2 / "lens" / f"fitg4_slice{k}.ckpt",
            RUN_DIR_P2 / "lens" / f"gemma431_slice{k}.pt")


def fit_slice(model, k, prompts, dim_batch, metrics):
    import jlens
    from jlens import JacobianLens
    local_ckpt, drive_ckpt, slice_lens_path = slice_paths(k)
    if slice_lens_path.exists():
        log(f"slice {k}: {slice_lens_path.name} already on Drive; loading")
        return JacobianLens.load(str(slice_lens_path))
    if not local_ckpt.exists() and drive_ckpt.exists():
        log(f"slice {k}: pulling Drive checkpoint back to local disk")
        shutil.copy2(drive_ckpt, local_ckpt)

    t0, lens = time.time(), None
    for end in range(SYNC_EVERY, SLICE_SIZE + 1, SYNC_EVERY):
        lens = jlens.fit(
            model, prompts[:end],
            source_layers=SOURCE_LAYERS, target_layer=TARGET_LAYER,
            dim_batch=dim_batch, max_seq_len=MAX_SEQ, skip_first=SKIP_FIRST,
            checkpoint_path=str(local_ckpt), checkpoint_every=1, resume=True)
        shutil.copy2(local_ckpt, drive_ckpt)
        peak = torch.cuda.max_memory_allocated() / 1e9
        log(f"slice {k}: {end}/{SLICE_SIZE} prompts; ckpt -> Drive; "
            f"peak VRAM {peak:.1f}GB")
        metrics["slices"].setdefault(str(k), {}).update(
            prompts_done=end, peak_vram_gb=round(peak, 1),
            elapsed_s=round(time.time() - t0))
        atomic_write_json(metrics, METRICS)

    lens.save(str(slice_lens_path))
    metrics["slices"][str(k)].update(lens_file=slice_lens_path.name,
                                     n_prompts=lens.n_prompts)
    atomic_write_json(metrics, METRICS)
    log(f"slice {k}: done in {time.time()-t0:.0f}s -> {slice_lens_path.name}")
    return lens


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    torch.manual_seed(0)
    import transformers
    import jlens
    from jlens import JacobianLens

    if MERGED.exists() and "--force" not in sys.argv:
        log(f"{MERGED} exists; nothing to do")
        return
    slices = [int(s) for s in arg("--slices", "0,1,2,3").split(",")]
    dim_batch = int(arg("--dim-batch", "8"))
    LOCAL_WORK.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in CORPUS.read_text().splitlines()]

    metrics = (json.loads(METRICS.read_text()) if METRICS.exists()
               else {"slices": {}})
    metrics.update({"model": MODEL, "corpus": str(CORPUS),
                    "corpus_sha256": sha256_file(CORPUS),
                    "source_layers": SOURCE_LAYERS,
                    "target_layer": TARGET_LAYER, "dim_batch": dim_batch,
                    "max_seq_len": MAX_SEQ, "skip_first": SKIP_FIRST,
                    "band_rationale": "a3-gemma-deepband-logit-v1"})
    atomic_write_json(metrics, METRICS)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    if model.n_layers != 60 or model.d_model != 5376:
        raise SystemExit(f"expected 60 x 5376, got "
                         f"{model.n_layers} x {model.d_model}")

    for k in slices:
        prompts = [r["text"] for r in rows[k * SLICE_SIZE:(k + 1) * SLICE_SIZE]]
        fit_slice(model, k, prompts, dim_batch, metrics)

    all_slices = sorted(RUN_DIR_P2.glob("lens/gemma431_slice*.pt"))
    if len(all_slices) < 4:
        log(f"{len(all_slices)}/4 slices present; merge deferred")
        return
    merged = JacobianLens.merge([JacobianLens.load(str(p))
                                 for p in all_slices])
    merged.save(str(MERGED))
    metrics["merged"] = {"n_prompts": merged.n_prompts, "file": MERGED.name,
                         "from": [p.name for p in all_slices]}
    atomic_write_json(metrics, METRICS)
    log(f"MERGED {len(all_slices)} slices ({merged.n_prompts} prompts) "
        f"-> {MERGED}")
    prov = Provenance(
        evidence_id="a3-gemma-fullfit-v1", tier="pilot",
        command="python -m jspace_part2.experiments.a3_gemma_fit",
        inputs={"corpus": sha256_file(CORPUS)},
        model=resolve_model(MODEL), seed=0)
    write_result({"metrics": metrics},
                 RUN_DIR_P2 / "metrics" / "gemma4-31b" / "fit_provenance.json",
                 prov)
    registry_append({
        "evidence_id": "a3-gemma-fullfit-v1", "tier": "pilot",
        "what": (f"Gemma-4-31B-it full 120-prompt J-lens (band "
                 f"{SOURCE_LAYERS} spanning paper-relative depths + "
                 f"readability transition; recipe commensurable with OLMo "
                 f"fits)"),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(MERGED), "sha256": sha256_file(MERGED)}]})
    log("fit registered")


if __name__ == "__main__":
    main()
