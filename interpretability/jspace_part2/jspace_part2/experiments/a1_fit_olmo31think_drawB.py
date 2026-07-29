# A1 drawB — INDEPENDENT 120-prompt J-lens for Olmo-3.1-32B-Think
# (corpus draw B, disjoint from draw A; discharges HP3's "reproduce
# under an independently fitted lens" clause on the replication
# partition). Derived from a1_fit_olmo31think:
# allenai/Olmo-3.1-32B-Think (freeze-blocking condition 3; the 3.1-Instruct
# lens already exists as a1-ownlens-regate-olmo31instruct-v1 /
# b1-fitB-independent-lens-olmo31instruct-v1).
#
# Recipe IDENTICAL to the 3.1-Instruct draw-A fit (p2a1_fit_instruct.py)
# and the part-1 s3 fits: same WikiText corpus file (draw A rows 0-119),
# 21 source layers, target 63, dim_batch 8, seq 128, skip 16 — so the
# primary pair's dictionaries are commensurable with each other and with
# every pilot lens. Checkpoint/resume contract: local ckpt every prompt,
# Drive ckpt copy every 10 prompts, per-slice fp16 lenses on Drive,
# n-weighted merge; the same command resumes.
#
# Usage: python -m jspace_part2.experiments.a1_fit_olmo31think \
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

MODEL = ("/content/hf_local/models--allenai--Olmo-3.1-32B-Think/"
         "snapshots/832c3f543499af8fe68b88359501de9cb7840544")
HUB_ID = "allenai/Olmo-3.1-32B-Think"
HUB_REVISION = "832c3f543499af8fe68b88359501de9cb7840544"
RUN_DIR_P1 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "2026-07-25_1726")
RUN_DIR_P2 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "part2_20260727")
CORPUS = RUN_DIR_P2 / "config" / "prompts" / "fitting_corpus_drawB.jsonl"
LOCAL_WORK = Path("/content/sl1_work")
MERGED = RUN_DIR_P2 / "lens" / "olmo31think_lensB.pt"
METRICS = RUN_DIR_P2 / "metrics" / "olmo31-think" / "fit_drawB.json"
SOURCE_LAYERS = [4, 8, 12, 16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                 42, 44, 48, 52, 56, 60]
TARGET_LAYER = 63
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
    return (LOCAL_WORK / f"fit31tb_slice{k}.ckpt",
            RUN_DIR_P2 / "lens" / f"fit31tb_slice{k}.ckpt",
            RUN_DIR_P2 / "lens" / f"olmo31think_lensB_slice{k}.pt")


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
    metrics.update({"model": HUB_ID, "hub_revision": HUB_REVISION,
                    "local_path": MODEL, "corpus": str(CORPUS),
                    "corpus_sha256": sha256_file(CORPUS),
                    "source_layers": SOURCE_LAYERS,
                    "target_layer": TARGET_LAYER, "dim_batch": dim_batch,
                    "max_seq_len": MAX_SEQ, "skip_first": SKIP_FIRST,
                    "recipe_commensurable_with":
                        "olmo31-instruct fit.json (draw A) / part-1 s3"})
    atomic_write_json(metrics, METRICS)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    if model.n_layers != 64 or model.d_model != 5120:
        raise SystemExit(f"expected 64 x 5120, got "
                         f"{model.n_layers} x {model.d_model}")

    for k in slices:
        prompts = [r["text"] for r in rows[k * SLICE_SIZE:(k + 1) * SLICE_SIZE]]
        fit_slice(model, k, prompts, dim_batch, metrics)

    all_slices = sorted(RUN_DIR_P2.glob("lens/olmo31think_lensB_slice*.pt"))
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
        evidence_id="a1-fitB-independent-lens-olmo31think-v1", tier="pilot",
        command="python -m jspace_part2.experiments.a1_fit_olmo31think_drawB",
        inputs={"corpus": sha256_file(CORPUS)},
        model=resolve_model(MODEL), seed=0)
    write_result({"metrics": metrics},
                 RUN_DIR_P2 / "metrics" / "olmo31-think" / "fit_provenance.json",
                 prov)
    registry_append({
        "evidence_id": "a1-fitB-independent-lens-olmo31think-v1", "tier": "pilot",
        "what": (f"Second INDEPENDENT 120-prompt lens for Olmo-3.1-32B-Think "
                 f"(corpus draw B, disjoint from draw A; same recipe: "
                 f"{len(SOURCE_LAYERS)} source layers, target "
                 f"{TARGET_LAYER}; HP3 independent-lens clause)"),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(MERGED), "sha256": sha256_file(MERGED)}]})
    log("fit registered")


if __name__ == "__main__":
    main()
