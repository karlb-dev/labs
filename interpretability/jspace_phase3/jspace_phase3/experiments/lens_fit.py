# Phase 3 J-lens fitter — the Part-2 A1 recipe, parameterized (R5
# lineage leg: the OLMo-3 BASE lens is the first new fit).
#
# Recipe IDENTICAL to every campaign lens (part-1 s3 / part-2 A1): the
# frozen draw-A WikiText corpus rows 0..119, 21 source layers, target
# 63, dim_batch 8, seq 128, skip 16 — so lineage dictionaries are
# commensurable with the primaries'. Checkpoint contract: local ckpt
# every prompt, Drive ckpt every 10, per-slice fp16 lenses on Drive,
# n-weighted merge; the same command resumes after any interruption.
#
# Usage:
#   python -m jspace_phase3.experiments.lens_fit --slug olmo3-base \
#       --model-uri model://allenai/Olmo-3-1125-32B@<rev> \
#       [--slices 0,1,2,3] [--dim-batch 8]
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from jspace_part2.lib import sha256_file
from jspace_part2.paths import resolve as resolve_uri
from ..paths3 import lens_dir, local_work, metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)

TIER = "phase3-development"
DRAW_A = Path("/content/drive/MyDrive/interpret/special-lab-1/"
              "2026-07-25_1726/config/prompts/fitting_corpus.jsonl")
# the Part-1 fit_config pins the CONCATENATED prompt texts, not the
# jsonl file (verified: concat-texts sha matches its "sha256" exactly)
DRAW_A_TEXTS_SHA = ("481be0c66a9733cc44af969e9053b2463ca85e759"
                    "a65e5e7d1e9b635c99339d0")
SOURCE_LAYERS = [4, 8, 12, 16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                 42, 44, 48, 52, 56, 60]
TARGET_LAYER = 63
SLICE_SIZE, SYNC_EVERY = 30, 10
MAX_SEQ, SKIP_FIRST = 128, 16


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():  # noqa: C901
    require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--slug")
    model_uri = arg("--model-uri")
    slices = [int(s) for s in (arg("--slices") or "0,1,2,3").split(",")]
    dim_batch = int(arg("--dim-batch") or 8)
    corpus = Path(arg("--corpus") or DRAW_A)
    if str(corpus) == str(DRAW_A):
        import hashlib
        rows0 = [json.loads(l) for l in corpus.read_text().splitlines()]
        got = hashlib.sha256("".join(
            r["text"] for r in rows0).encode()).hexdigest()
        if got != DRAW_A_TEXTS_SHA:
            raise SystemExit(f"draw-A concat-texts sha mismatch: {got}")

    out_lens = lens_dir() / f"{slug}_lens.pt"
    metrics_p = metrics_dir(slug) / "lens_fit.json"
    metrics = (json.loads(metrics_p.read_text()) if metrics_p.exists()
               else {"slices": {}})

    import transformers
    import jlens
    from jlens import JacobianLens
    model_path = str(resolve_uri(model_uri, must_exist=True))
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    if model.n_layers != 64 or model.d_model != 5120:
        raise SystemExit(f"expected 64 x 5120, got "
                         f"{model.n_layers} x {model.d_model}")
    torch.manual_seed(0)

    rows = [json.loads(l) for l in corpus.read_text().splitlines()]
    metrics.update({"model_uri": model_uri, "local_path": model_path,
                    "corpus": str(corpus),
                    "corpus_sha256": sha256_file(corpus),
                    "source_layers": SOURCE_LAYERS,
                    "target_layer": TARGET_LAYER, "dim_batch": dim_batch,
                    "max_seq_len": MAX_SEQ, "skip_first": SKIP_FIRST,
                    "recipe": "part2-A1-commensurable"})

    def atomic(obj, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(obj, indent=2))
        os.replace(tmp, path)

    atomic(metrics, metrics_p)

    def fit_slice(k, prompts):
        local_ckpt = local_work() / f"fit_{slug}_slice{k}.ckpt"
        drive_ckpt = lens_dir() / f"fit_{slug}_slice{k}.ckpt"
        slice_lens = lens_dir() / f"{slug}_lens_slice{k}.pt"
        if slice_lens.exists():
            log(f"slice {k}: already on Drive; loading")
            return JacobianLens.load(str(slice_lens))
        if not local_ckpt.exists() and drive_ckpt.exists():
            shutil.copy2(drive_ckpt, local_ckpt)
        t0, lens = time.time(), None
        for end in range(SYNC_EVERY, SLICE_SIZE + 1, SYNC_EVERY):
            lens = jlens.fit(
                model, prompts[:end],
                source_layers=SOURCE_LAYERS, target_layer=TARGET_LAYER,
                dim_batch=dim_batch, max_seq_len=MAX_SEQ,
                skip_first=SKIP_FIRST, checkpoint_path=str(local_ckpt),
                checkpoint_every=1, resume=True)
            shutil.copy2(local_ckpt, drive_ckpt)
            peak = torch.cuda.max_memory_allocated() / 1e9
            log(f"slice {k}: {end}/{SLICE_SIZE}; ckpt->Drive; "
                f"peak {peak:.1f}GB")
            metrics["slices"].setdefault(str(k), {}).update(
                prompts_done=end, peak_vram_gb=round(peak, 1),
                elapsed_s=round(time.time() - t0))
            atomic(metrics, metrics_p)
        lens.save(str(slice_lens))
        metrics["slices"][str(k)].update(lens_file=slice_lens.name,
                                         n_prompts=lens.n_prompts)
        atomic(metrics, metrics_p)
        log(f"slice {k}: done in {time.time() - t0:.0f}s")
        return lens

    for k in slices:
        prompts = [r["text"]
                   for r in rows[k * SLICE_SIZE:(k + 1) * SLICE_SIZE]]
        fit_slice(k, prompts)

    all_slices = sorted(lens_dir().glob(f"{slug}_lens_slice*.pt"))
    if len(all_slices) < 4:
        log(f"{len(all_slices)}/4 slices; merge deferred")
        return
    if not out_lens.exists():
        merged = JacobianLens.merge([JacobianLens.load(str(p))
                                     for p in all_slices])
        merged.save(str(out_lens))
        metrics["merged"] = {"n_prompts": merged.n_prompts,
                             "file": out_lens.name}
        atomic(metrics, metrics_p)
        log(f"MERGED -> {out_lens}")
    eid = f"p3-lens-{slug}-v1"
    cmd = (f"python -m jspace_phase3.experiments.lens_fit --slug {slug} "
           f"--model-uri {model_uri}")
    write_result3({"metrics": metrics}, metrics_dir(slug) /
                  "lens_fit_provenance.json", Provenance3(
                      evidence_id=eid, tier=TIER, command=cmd,
                      inputs={"corpus": sha256_file(corpus)},
                      model=resolve_model(model_path), seed=0))
    register(eid, tier=TIER, command=cmd,
             what=(f"120-prompt J-lens for {slug} under the frozen A1 "
                   f"recipe (draw-A corpus, {len(SOURCE_LAYERS)} source "
                   f"layers, target {TARGET_LAYER}) — lineage leg R5"),
             outputs=[out_lens,
                      metrics_dir(slug) / "lens_fit_provenance.json"])
    log("fit registered")


if __name__ == "__main__":
    main()
