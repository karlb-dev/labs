# A2a cell 1 — Qwen3.6-27B within-checkpoint mode contrast (H1b):
# paper-defined occupancy + excess at PRE-RESPONSE positions under the
# OFFICIAL thinking toggle (enable_thinking=True/False in the chat
# template; raw completions are neither mode — addendum §5.11).
#
# Same weights, same lens, same questions; the only difference is the
# rendered mode. Positions: the last `n_positions` prompt tokens (template
# close + question tail), i.e. the state from which the model is about to
# (a) begin thinking or (b) answer directly.
#
# H1b prediction (addendum §8): internal workspace load differs by mode
# pre-response; weakened if modes differ only in surface tokens.
#
# Usage: python -m jspace_part2.experiments.a2a_mode_occupancy [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ..battery import twohop_items
from ..lib import sha256_file
from ..occupancy import occupancy_and_excess
from ..dictionaries import build_j_dictionaries
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

MODEL = "/content/hf_local/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
LENS = "/content/hf_local/models--neuronpedia--jacobian-lens/snapshots/a4114d7752d11eb546e6cf372213d7e75526d3a1/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
OUT = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
           "metrics/qwen36-27b/a2a_mode_occupancy.json")
LAYERS = [24, 32, 40]
K_MAX = 50
N_POS = 8
RAND_SEEDS = [11, 12, 13]


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    if OUT.exists() and "--force" not in sys.argv:
        print("exists; skipping")
        return
    import transformers
    import jlens
    from jlens import JacobianLens
    from jlens.hooks import ActivationRecorder

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.load(LENS)
    items = twohop_items(60)

    acts = {m: {l: [] for l in LAYERS} for m in ("think_on", "think_off")}
    with torch.no_grad():
        for it in items:
            q = it["prompt"].rstrip()
            for mode, flag in (("think_on", True), ("think_off", False)):
                text = tok.apply_chat_template(
                    [{"role": "user", "content": q}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=flag)
                ids = tok(text, return_tensors="pt",
                          truncation=True, max_length=512).input_ids.cuda()
                with ActivationRecorder(model.layers, at=LAYERS) as rec:
                    model.forward(ids)
                for l in LAYERS:
                    h = rec.activations[l][0].float()
                    acts[mode][l].append(h[-N_POS:])
    jd = build_j_dictionaries(hf, lens, LAYERS)
    V, d = jd[LAYERS[0]].shape
    res = {"per_mode": {}, "n_items": len(items), "n_pos": N_POS,
           "layers": LAYERS}
    for mode in ("think_on", "think_off"):
        res["per_mode"][mode] = {}
        for l in LAYERS:
            H = torch.cat(acts[mode][l]).cuda()
            rands = []
            for s in RAND_SEEDS:
                g = torch.Generator().manual_seed(s)
                rands.append(torch.nn.functional.normalize(
                    torch.randn(V, d, generator=g), dim=1)
                    .to("cuda", torch.float16))
            out = occupancy_and_excess(H, jd[l], rands, K_MAX, H.mean(0))
            occ = out.pop("occupancy")
            res["per_mode"][mode][str(l)] = {
                "occ_median": float(occ.float().median()),
                "occ_q25": float(occ.float().quantile(0.25)),
                "occ_q75": float(occ.float().quantile(0.75)),
                "excess_share": out["excess_share"],
                "censored_frac": out["censored_frac"],
            }
            del rands
            torch.cuda.empty_cache()
            print(f"{mode} L{l}: occ {res['per_mode'][mode][str(l)]['occ_median']}"
                  f" excess {out['excess_share']:.4f}", flush=True)

    prov = Provenance(
        evidence_id="a2a-mode-occupancy-qwen-v1", tier="pilot",
        command="python -m jspace_part2.experiments.a2a_mode_occupancy",
        inputs={"lens": sha256_file(LENS)},
        model=resolve_model(MODEL), seed=RAND_SEEDS[0])
    write_result(res, OUT, prov)
    summ = "; ".join(
        f"{m} L40 occ {res['per_mode'][m]['40']['occ_median']} excess "
        f"{res['per_mode'][m]['40']['excess_share']:.4f}"
        for m in res["per_mode"])
    registry_append({
        "evidence_id": "a2a-mode-occupancy-qwen-v1", "tier": "pilot",
        "what": f"H1b: pre-response occupancy by OFFICIAL thinking mode "
                f"(same weights/lens/questions): {summ}",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print("A2a mode-occupancy done")


if __name__ == "__main__":
    main()
