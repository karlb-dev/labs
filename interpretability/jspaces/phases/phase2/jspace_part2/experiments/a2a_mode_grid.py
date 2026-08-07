# A2a cell 2 — protected dynamic grid BY OFFICIAL MODE on Qwen3.6-27B
# (H1b causal): same weights/lens/items, chat-rendered with
# enable_thinking True/False; conditions none / dynJ_protected /
# dynR_protected.
#
# Scoring semantics differ BY DESIGN and are labeled as different metrics:
#   think_off: answer follows the auto-closed think block -> the natural
#              continuation -> metric = answer_seq_lp (task-performance-like)
#   think_on:  answer teacher-forced right after the OPEN <think> tag ->
#              metric = precot_answer_lp (pre-CoT answer availability,
#              NOT task performance; part-1 found this signal weak on
#              OLMo -- here we ask how ablation moves it on Qwen)
#
# H1b question: does the protected-ablation effect differ by requested
# mode on identical content?
# Usage: python -m jspace_part2.experiments.a2a_mode_grid [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ..battery import answer_variants, onehop_items, seq_lp_from_logits, twohop_items
from ..dictionaries import build_j_dictionaries
from ..lib import sha256_file
from ..protected_dynamic import ProtectedDynamicAblator, protected_teacher_forced
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

MODEL = "/content/hf_local/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
LENS = "/content/hf_local/models--neuronpedia--jacobian-lens/snapshots/a4114d7752d11eb546e6cf372213d7e75526d3a1/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
OUT_DIR = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
               "metrics/qwen36-27b/a2a_mode_grid")
BAND = list(range(20, 45, 2))
K, PK = 10, 10
RAND_SEED = 4242


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state_p = OUT_DIR / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else \
        {"rows": [], "cells": {}}

    import transformers
    import jlens
    from jlens import JacobianLens

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.load(LENS)
    jd = build_j_dictionaries(hf, lens, BAND)
    V, d = jd[BAND[0]].shape
    g = torch.Generator().manual_seed(RAND_SEED)
    R = torch.nn.functional.normalize(torch.randn(V, d, generator=g),
                                      dim=1).to("cuda", torch.float16)
    rd = {l: R for l in BAND}
    conds = {"none": None, "dynJ_protected": jd, "dynR_protected": rd}
    tasks = {"twohop": twohop_items(60), "onehop": onehop_items()}
    ab = ProtectedDynamicAblator(model.layers, BAND)

    def render(q, thinking):
        return tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True, enable_thinking=thinking)

    with ab:
        for mode, flag in (("think_off", False), ("think_on", True)):
            metric = "answer_seq_lp" if mode == "think_off" else "precot_answer_lp"
            for cname, dicts in conds.items():
                for tname, items in tasks.items():
                    cell = f"{mode}/{cname}/{tname}"
                    if cell in state["cells"]:
                        continue
                    t0 = time.time()
                    for it in items:
                        prompt = render(it["prompt"].rstrip(), flag)
                        n_prompt = tok(prompt, return_tensors="pt",
                                       truncation=True, max_length=640
                                       ).input_ids.shape[1]
                        best = None
                        for v in answer_variants(it["answer"]):
                            text = prompt + v.lstrip() if prompt.endswith("\n") else prompt + v
                            if dicts is None:
                                ab.mode = None
                                ids = tok(text, return_tensors="pt",
                                          truncation=True, max_length=640
                                          ).input_ids.cuda()
                                logits = hf(input_ids=ids, use_cache=False
                                            ).logits[0].float().cpu()
                            else:
                                ids, logits = protected_teacher_forced(
                                    hf, lambda t, max_length=640:
                                        tok(t, return_tensors="pt",
                                            truncation=True,
                                            max_length=max_length
                                            ).input_ids.cuda(),
                                    ab, dicts, text, k=K, protect=PK,
                                    protected=True, max_length=640)
                            lp = seq_lp_from_logits(ids, logits, n_prompt)
                            best = lp if best is None or lp > best else best
                        state["rows"].append({
                            "mode": mode, "condition": cname, "task": tname,
                            "item_id": it["item_id"], "family": it["family"],
                            "metric": metric, "score": best})
                    state["cells"][cell] = {"seconds": round(time.time() - t0)}
                    state_p.write_text(json.dumps(state))
                    print(f"[{time.strftime('%H:%M:%S')}] {cell} "
                          f"({state['cells'][cell]['seconds']}s)", flush=True)

    import pandas as pd
    df = pd.DataFrame(state["rows"])
    df.to_parquet(OUT_DIR / "a2a_per_item.parquet")
    summ = {}
    for mode in ("think_off", "think_on"):
        for tname in tasks:
            base = df[(df["mode"] == mode) & (df.condition == "none")
                      & (df.task == tname)].set_index("item_id")["score"]
            for cname in ("dynJ_protected", "dynR_protected"):
                s = df[(df["mode"] == mode) & (df.condition == cname)
                       & (df.task == tname)].set_index("item_id")["score"]
                delta = (s - base).dropna()
                summ[f"{mode}/{cname}/{tname}"] = {
                    "n": int(len(delta)),
                    "mean_delta": round(float(delta.mean()), 4),
                    "median_delta": round(float(delta.median()), 4)}
    prov = Provenance(
        evidence_id="a2a-mode-grid-qwen-v1", tier="pilot",
        command="python -m jspace_part2.experiments.a2a_mode_grid",
        inputs={"lens": sha256_file(LENS)}, model=resolve_model(MODEL),
        seed=RAND_SEED)
    write_result({"summary": summ, "cells": state["cells"], "band": BAND,
                  "k": K, "protect_top_k": PK}, OUT_DIR / "a2a_summary.json",
                 prov)
    registry_append({
        "evidence_id": "a2a-mode-grid-qwen-v1", "tier": "pilot",
        "what": f"H1b causal: protected dyn-J grid by OFFICIAL mode: {summ}",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT_DIR / "a2a_summary.json"),
                     "sha256": sha256_file(OUT_DIR / "a2a_summary.json")},
                    {"path": str(OUT_DIR / "a2a_per_item.parquet"),
                     "sha256": sha256_file(OUT_DIR / "a2a_per_item.parquet")}]})
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
