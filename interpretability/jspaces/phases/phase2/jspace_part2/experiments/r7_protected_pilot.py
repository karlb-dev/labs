# R7 pilot — THE Stage-2 decision experiment: does the paper's
# output-protected dynamic J ablation behave differently from part-1's
# unprotected variant on Olmo-3-32B-Think?
#
# Conditions (single changed variable between the two dyn-J arms):
#   none               baseline (clean pass)
#   dynJ_protected     nonneg top-k J deflation per position, clean-top-10
#                      output tokens protected  (THE PAPER'S PROTOCOL)
#   dynJ_unprotected   identical mechanics, no protection mask
#   dynR_protected     identical mechanics + protection on a seeded
#                      vocab-sized random unit dictionary (matched control)
# Tasks: twohop full-seq lp + first-token acc (n<=60), onehop lp + acc
# (n=30), prose NLL (n=20), 3-sample generation audit per condition.
#
# Per-item rows -> parquet + JSON summary with provenance; resumable per
# (condition, task); refuses dirty git tree (repro contract).
#
# Usage: python -m jspace_part2.experiments.r7_protected_pilot \
#          --config configs/r7_protected_pilot.yaml [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..battery import (answer_variants, onehop_items, prose_items,
                       seq_lp_from_logits, twohop_items)
from ..dictionaries import build_j_dictionaries
from ..lib import sha256_file
from ..protected_dynamic import (ProtectedDynamicAblator, protected_generate,
                                 protected_teacher_forced)
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def load_model(path_or_id):
    import transformers
    import jlens
    tok = transformers.AutoTokenizer.from_pretrained(path_or_id)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        path_or_id, dtype=torch.bfloat16).to("cuda").eval()
    return jlens.from_hf(hf, tok), hf, tok


def random_dictionary(vocab, d, seed, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(vocab, d, generator=g)
    return torch.nn.functional.normalize(R, dim=1).to(device, torch.float16)


def main():
    cfg_path = arg("--config", "configs/r7_protected_pilot.yaml")
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    git = require_clean_tree("--allow-dirty" in sys.argv)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "r7_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else \
        {"cells": {}, "rows": []}

    from jlens import JacobianLens
    lens = JacobianLens.load(cfg["lens_path"])
    band = cfg["band"]
    model, hf, tok = load_model(cfg["model_path"])
    jd = build_j_dictionaries(hf, lens, band)
    rd = {l: random_dictionary(jd[band[0]].shape[0], model.d_model,
                               cfg["rand_seed"]) for l in [band[0]]}
    rd = {l: rd[band[0]] for l in band}   # one shared tensor
    k, pk = cfg["k"], cfg["protect_top_k"]

    conds = {
        "none": None,
        "dynJ_protected": {"dicts": jd, "protected": True},
        "dynJ_unprotected": {"dicts": jd, "protected": False},
        "dynR_protected": {"dicts": rd, "protected": True},
    }
    tasks = {
        "twohop": twohop_items(cfg["n_twohop"]),
        "onehop": onehop_items(),
        "prose": prose_items(cfg["prose_corpus"]),
    }
    ab = ProtectedDynamicAblator(model.layers, band)

    def lp_and_acc(spec, it):
        """Full-seq answer lp (max over frozen variants) + first-token acc."""
        best, first_ok = None, 0.0
        for v in answer_variants(it["answer"]):
            text = it["prompt"].rstrip() + v
            n_prompt = model.encode(it["prompt"].rstrip(),
                                    max_length=512).shape[1]
            if spec is None:
                ab.mode = None
                ids = model.encode(text, max_length=512)
                logits = hf(input_ids=ids, use_cache=False).logits[0]\
                    .float().cpu()
            else:
                ids, logits = protected_teacher_forced(
                    hf, model.encode, ab, spec["dicts"], text, k=k,
                    protect=pk, protected=spec["protected"])
            lp = seq_lp_from_logits(ids, logits, n_prompt)
            if best is None or lp > best:
                best = lp
                vid = tok(v, add_special_tokens=False).input_ids[0]
                first_ok = float(int(logits[n_prompt - 1].argmax()) == vid)
        return best, first_ok

    with ab:
        for cname, spec in conds.items():
            for tname, items in tasks.items():
                cell = f"{cname}/{tname}"
                if cell in state["cells"]:
                    continue
                t0 = time.time()
                for it in items:
                    if tname == "prose":
                        if spec is None:
                            ab.mode = None
                            ids = model.encode(it["text"], max_length=256)
                            logits = hf(input_ids=ids, use_cache=False)\
                                .logits[0].float().cpu()
                        else:
                            ids, logits = protected_teacher_forced(
                                hf, model.encode, ab, spec["dicts"],
                                it["text"], k=k, protect=pk,
                                protected=spec["protected"], max_length=256)
                        lp = torch.log_softmax(logits[:-1], -1)
                        tgt = ids[0, 1:].cpu()
                        score = float(-lp[torch.arange(len(tgt)), tgt].mean())
                        row = {"condition": cname, "task": tname,
                               "item_id": it["item_id"],
                               "family": it["family"], "score": score,
                               "metric": "nll"}
                    else:
                        lp, acc = lp_and_acc(spec, it)
                        row = {"condition": cname, "task": tname,
                               "item_id": it["item_id"],
                               "family": it["family"], "score": lp,
                               "metric": "answer_seq_lp", "first_tok_acc": acc}
                    state["rows"].append(row)
                # generation audit (3 samples, twohop prompts)
                audit = []
                if tname == "twohop":
                    for it in items[:3]:
                        if spec is None:
                            ab.mode = None
                            ids = tok(it["prompt"], return_tensors="pt")\
                                .input_ids.cuda()
                            g = hf.generate(ids, max_new_tokens=32,
                                            do_sample=False,
                                            pad_token_id=tok.eos_token_id)
                            txt = tok.decode(g[0, ids.shape[1]:],
                                             skip_special_tokens=True)
                        else:
                            txt, _ = protected_generate(
                                hf, tok, ab, spec["dicts"], it["prompt"],
                                k=k, protect=pk, max_new=32,
                                protected=spec["protected"])
                        audit.append({"prompt": it["prompt"][:60],
                                      "gen": txt})
                state["cells"][cell] = {
                    "seconds": round(time.time() - t0),
                    "audit": audit,
                    "hook_log": {"steps": ab.log.n_steps,
                                 "blocked": ab.log.protected_hits_blocked,
                                 "mean_removed_energy": (
                                     round(float(np.mean(ab.log.removed_energy)), 5)
                                     if ab.log.removed_energy else None)}}
                ab.log.__init__()
                state_path.write_text(json.dumps(state))
                print(f"[{time.strftime('%H:%M:%S')}] {cell} done "
                      f"({state['cells'][cell]['seconds']}s)", flush=True)

    # ---- summary: paired per-item deltas vs baseline
    import pandas as pd
    df = pd.DataFrame(state["rows"])
    df.to_parquet(out_dir / "r7_per_item.parquet")
    summ = {}
    for tname in tasks:
        base = df[(df.condition == "none") & (df.task == tname)]\
            .set_index("item_id")["score"]
        for cname in conds:
            if cname == "none":
                continue
            s = df[(df.condition == cname) & (df.task == tname)]\
                .set_index("item_id")["score"]
            delta = (s - base).dropna()
            summ[f"{cname}/{tname}"] = {
                "n": int(len(delta)),
                "mean_delta": round(float(delta.mean()), 4),
                "median_delta": round(float(delta.median()), 4)}
    prov = Provenance(
        evidence_id=cfg["evidence_id"], tier=cfg["tier"],
        command=f"python -m jspace_part2.experiments.r7_protected_pilot --config {cfg_path}",
        config_path=cfg_path,
        inputs={"lens": sha256_file(cfg["lens_path"]),
                "probe_swap": sha256_file(
                    "/content/jacobian-lens/data/experiments/probe-swap.json")},
        model=resolve_model(cfg["model_path"]),
        seed=cfg["rand_seed"], allow_dirty="--allow-dirty" in sys.argv)
    write_result({"config": cfg, "summary": summ,
                  "cells": state["cells"]},
                 out_dir / "r7_summary.json", prov)
    registry_append({
        "evidence_id": cfg["evidence_id"], "tier": cfg["tier"],
        "what": f"R7 pilot protected-vs-unprotected dynamic grid: {summ}",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(out_dir / "r7_summary.json"),
                     "sha256": sha256_file(out_dir / "r7_summary.json")},
                    {"path": str(out_dir / "r7_per_item.parquet"),
                     "sha256": sha256_file(out_dir / "r7_per_item.parquet")}]})
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
