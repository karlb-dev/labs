# B1/G2 GPU pass — dictionary-level agreement between the two INDEPENDENT
# 120-prompt Instruct lenses (disjoint corpora), in the units the B1
# decision rule actually wants:
#   (i)  per-token dictionary-row cosine distribution per band layer
#        (rows of (W_U⊙g)@J_A vs @J_B, both normalized);
#   (ii) frozen top-10 selection Jaccard per item per layer on the 60
#        two-hop prompts (part-1 selection rule: summed |corr| over prompt
#        positions, first 4 skipped).
# Usage: python -m jspace_part2.experiments.b1_dict_stability [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ..battery import twohop_items
from ..dictionaries import build_j_dictionaries
from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")
MODEL = "/content/hf_local/models--allenai--Olmo-3.1-32B-Instruct/snapshots/ac0587e4a7744a551c059d8cd17ba220bc940dae"
LENS_A = RUN / "lens" / "olmo31instruct_lens.pt"
LENS_B = RUN / "lens" / "olmo31instruct_lensB.pt"
BAND = list(range(20, 45, 2))
OUT = RUN / "metrics" / "olmo31-instruct" / "b1_dict_stability.json"
K, SKIP = 10, 4


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
    A = JacobianLens.load(str(LENS_A))
    B = JacobianLens.load(str(LENS_B))

    items = twohop_items(60)
    acts = {l: [] for l in BAND}
    with torch.no_grad():
        for it in items:
            ids = model.encode(it["prompt"].rstrip(), max_length=512)
            with ActivationRecorder(model.layers, at=BAND) as rec:
                model.forward(ids)
            for l in BAND:
                h = rec.activations[l][0].float()
                acts[l].append(h[min(SKIP, h.shape[0] - 1):])

    per_layer = {}
    t0 = time.time()
    for l in BAND:
        DA = build_j_dictionaries(hf, A, [l])[l]
        DB = build_j_dictionaries(hf, B, [l])[l]
        cos = (DA.float() * DB.float()).sum(dim=1).cpu()   # rows unit-norm
        jacc = []
        for h in acts[l]:
            sA = (h.half() @ DA.T).abs().sum(0).topk(K).indices
            sB = (h.half() @ DB.T).abs().sum(0).topk(K).indices
            inter = len(set(sA.tolist()) & set(sB.tolist()))
            jacc.append(inter / (2 * K - inter))
        per_layer[str(l)] = {
            "row_cos_median": round(float(cos.median()), 5),
            "row_cos_q05": round(float(cos.quantile(0.05)), 5),
            "row_cos_q01": round(float(cos.quantile(0.01)), 5),
            "sel_jaccard_mean": round(sum(jacc) / len(jacc), 4),
            "sel_jaccard_min": round(min(jacc), 4),
        }
        del DA, DB
        torch.cuda.empty_cache()
        print(f"L{l}: row-cos med {per_layer[str(l)]['row_cos_median']} "
              f"q05 {per_layer[str(l)]['row_cos_q05']} | sel-Jaccard "
              f"{per_layer[str(l)]['sel_jaccard_mean']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    med_cos = sorted(v["row_cos_median"] for v in per_layer.values())[len(per_layer)//2]
    med_j = sorted(v["sel_jaccard_mean"] for v in per_layer.values())[len(per_layer)//2]
    res = {"per_layer": per_layer, "median_row_cos": med_cos,
           "median_sel_jaccard": med_j,
           "decision_rule_note": "B1 heuristics were cos>0.9, Jaccard>=0.7 "
                                 "(prereg v1; margins to be pilot-calibrated)"}
    prov = Provenance(
        evidence_id="b1-dict-stability-olmo31instruct-v1", tier="pilot",
        command="python -m jspace_part2.experiments.b1_dict_stability",
        inputs={"lens_A": sha256_file(LENS_A), "lens_B": sha256_file(LENS_B)},
        model=resolve_model(MODEL))
    write_result(res, OUT, prov)
    registry_append({
        "evidence_id": "b1-dict-stability-olmo31instruct-v1", "tier": "pilot",
        "what": f"dict-level lens stability across independent corpus draws: "
                f"median row-cos {med_cos}, median frozen-selection Jaccard "
                f"{med_j} (band L20-44, n=60 items)",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(f"B1 dict stability done: cos {med_cos}, Jaccard {med_j}")


if __name__ == "__main__":
    main()
