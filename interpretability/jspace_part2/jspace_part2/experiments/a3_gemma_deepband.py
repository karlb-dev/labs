# A3 follow-up — Gemma-4-31B-it DEEP-BAND logit-lens depth sweep.
#
# The adaptation gate found mid-band output-basis opacity: at L24/30 (of
# 60) BOTH the 2-prompt micro jlens AND the vanilla logit lens rank the
# known answer ~52k/69k of 262k, while the final head ranks it 1. Before
# any family verdict we need the depth profile: WHERE (if anywhere) does
# Gemma's residual stream become readable in the output basis? This sweep
# records the vanilla logit-lens rank of the answer's first token at the
# final prompt position for every even layer, using the reference
# `model.unembed` (final norm + tied head + 30.0 softcap, wrapper-aware).
# The result picks the band for the full 120-prompt fit (next queue item)
# and is an architecture datum on its own.
#
# Tier: pilot. Cheap: one recorded forward per probe.
# Usage: python -m jspace_part2.experiments.a3_gemma_deepband [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ..battery import answer_variants, onehop_items, twohop_items
from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

MODEL = "/content/models/gemma4-31b-it"
OUT = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
           "metrics/gemma4-31b/a3_deepband_logit.json")


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    if OUT.exists() and "--force" not in sys.argv:
        print("exists; skipping")
        return
    import transformers
    import jlens
    from jlens.hooks import ActivationRecorder

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    layers = list(range(2, model.n_layers - 1, 2)) + [model.n_layers - 1]

    probes = onehop_items() + twohop_items(10)
    rows = []
    with torch.no_grad():
        for it in probes:
            ids = model.encode(it["prompt"].rstrip(), max_length=128)
            with ActivationRecorder(model.layers, at=layers) as r:
                out = model.forward(ids)
            final_logits = (out.logits if hasattr(out, "logits") else out)[0, -1].float()
            cand = [tok(v, add_special_tokens=False).input_ids[0]
                    for v in answer_variants(it["answer"])]
            final_rank = min(int((final_logits > final_logits[c]).sum()) + 1
                             for c in cand)
            per_layer = {}
            for l in layers:
                h = r.activations[l][0, -1]
                lg = model.unembed(h.reshape(1, 1, -1)).reshape(-1).float()
                per_layer[l] = min(int((lg > lg[c]).sum()) + 1 for c in cand)
            rows.append({"item_id": it["item_id"], "final_rank": final_rank,
                         "ranks": per_layer})
            print(f"  {it['item_id']:24s} final={final_rank:6d} "
                  f"L24={per_layer.get(24):7d} L40={per_layer.get(40):7d} "
                  f"L52={per_layer.get(52):7d} L58={per_layer.get(58):7d}",
                  flush=True)

    known = [r for r in rows if r["final_rank"] <= 3]
    med = {l: int(sorted(r["ranks"][l] for r in known)[len(known) // 2])
           for l in layers}
    def first_depth(thr):
        hit = [l for l in layers if med[l] <= thr]
        return hit[0] if hit else None
    summ = {"n_probes": len(rows), "n_known": len(known),
            "median_rank_by_layer": med,
            "first_layer_median_le_100": first_depth(100),
            "first_layer_median_le_10": first_depth(10),
            "seconds": round(time.time() - t0)}
    prov = Provenance(
        evidence_id="a3-gemma-deepband-logit-v1", tier="pilot",
        command="python -m jspace_part2.experiments.a3_gemma_deepband",
        model=resolve_model(MODEL))
    write_result({"summary": summ, "rows": rows}, OUT, prov)
    registry_append({
        "evidence_id": "a3-gemma-deepband-logit-v1", "tier": "pilot",
        "what": (f"Gemma deep-band logit-lens depth sweep (n_known="
                 f"{len(known)}): first layer median rank<=100: "
                 f"{summ['first_layer_median_le_100']}, <=10: "
                 f"{summ['first_layer_median_le_10']}; mid-band L24 median "
                 f"{med.get(24)}, L30 {med.get(30)}, L40 {med.get(40)}, "
                 f"L52 {med.get(52)}, L58 {med.get(58)}"),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
