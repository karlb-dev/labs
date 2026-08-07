# R7 follow-up: log each battery item's CLEAN answer-token rank at the
# scoring position on Olmo-3-32B-Think. Tests the tail mechanization:
# protected-condition damage should concentrate on items whose answer
# first-token rank > protect_top_k (unprotected by construction).
# Cheap: one clean prefill per item. Tier: pilot.
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

OUT = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
           "metrics/olmo3-think/r7_pilot/r7_cleanrank.json")
MODEL = "/content/models/olmo3-think"


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    if OUT.exists() and "--force" not in sys.argv:
        print("exists; skipping")
        return
    import transformers
    import jlens
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    rows = []
    with torch.no_grad():
        for it in twohop_items(60) + onehop_items():
            ids = model.encode(it["prompt"].rstrip(), max_length=512)
            logits = hf(input_ids=ids, use_cache=False).logits[0, -1].float()
            best = None
            for v in answer_variants(it["answer"]):
                fid = tok(v, add_special_tokens=False).input_ids[0]
                rank = int((logits > logits[fid]).sum()) + 1
                best = rank if best is None or rank < best else best
            rows.append({"item_id": it["item_id"], "clean_rank": best,
                         "protected_by_top10": best <= 10})
    frac = sum(r["protected_by_top10"] for r in rows) / len(rows)
    prov = Provenance(
        evidence_id="r7-cleanrank-think-v1", tier="pilot",
        command="python -m jspace_part2.experiments.r7_cleanrank",
        model=resolve_model(MODEL))
    write_result({"rows": rows, "frac_protected": round(frac, 3)}, OUT, prov)
    registry_append({
        "evidence_id": "r7-cleanrank-think-v1", "tier": "pilot",
        "what": f"clean answer first-token ranks for R7 battery items "
                f"({frac:.0%} inside protect-top-10)",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(f"done; {frac:.0%} of items have clean rank <= 10")


if __name__ == "__main__":
    main()
