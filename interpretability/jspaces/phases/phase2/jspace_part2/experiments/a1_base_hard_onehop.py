# Base-leg de-confound: the reverse dissociation (onehop -0.81, twohop
# null) used near-ceiling capitals. Rerun {none, dynJ_protected,
# dynR_protected} on the 41-item HARD one-hop dev set (difficulty-matched
# to two-hop on Think). If hard one-hop is also hit -> the base's one-hop
# vulnerability is channel structure, not ceiling.
# Usage: python -m jspace_part2.experiments.a1_base_hard_onehop [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ..battery import answer_variants, seq_lp_from_logits
from ..dictionaries import build_j_dictionaries
from ..lib import sha256_file
from ..protected_dynamic import ProtectedDynamicAblator, protected_teacher_forced
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

MODEL = "/content/hf_local/models--allenai--Olmo-3-1125-32B/snapshots/c2b61dae89a1ad10e4ad5653d0e46b590902607b"
LENS = ("/content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726/"
        "lens/olmo32bthink_lens.pt")   # transfer-gate PASSED at donor level
ITEMS = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
             "config/prompts/hard_onehop_dev.jsonl")
OUT = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
           "metrics/olmo3-base/a1_hard_onehop.json")
BAND = list(range(20, 45, 2))
K, PK = 10, 10


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    if OUT.exists() and "--force" not in sys.argv:
        print("exists; skipping")
        return
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
    g = torch.Generator().manual_seed(4242)
    R = torch.nn.functional.normalize(torch.randn(V, d, generator=g),
                                      dim=1).to("cuda", torch.float16)
    rd = {l: R for l in BAND}
    items = [json.loads(l) for l in ITEMS.read_text().splitlines()]
    ab = ProtectedDynamicAblator(model.layers, BAND)
    rows = []
    with ab:
        for it in items:
            row = {"item": it["answer"], "lp_think_cal": it["lp"]}
            for cname, dicts in (("none", None), ("dynJ_protected", jd),
                                 ("dynR_protected", rd)):
                best = None
                for v in answer_variants(it["answer"]):
                    text = it["prompt"].rstrip() + v
                    n_p = model.encode(it["prompt"].rstrip(),
                                       max_length=512).shape[1]
                    if dicts is None:
                        ab.mode = None
                        ids = model.encode(text, max_length=512)
                        logits = hf(input_ids=ids, use_cache=False)\
                            .logits[0].float().cpu()
                    else:
                        ids, logits = protected_teacher_forced(
                            hf, model.encode, ab, dicts, text, k=K,
                            protect=PK, protected=True)
                    lp = seq_lp_from_logits(ids, logits, n_p)
                    best = lp if best is None or lp > best else best
                row[cname] = round(best, 4)
            rows.append(row)
    n = len(rows)
    dj = sum(r["dynJ_protected"] - r["none"] for r in rows) / n
    dr = sum(r["dynR_protected"] - r["none"] for r in rows) / n
    med = sorted(r["dynJ_protected"] - r["none"] for r in rows)[n // 2]
    summ = {"n": n, "dynJ_protected_mean_delta": round(dj, 3),
            "dynJ_protected_median_delta": round(med, 3),
            "dynR_protected_mean_delta": round(dr, 3)}
    prov = Provenance(
        evidence_id="a1-base-hard-onehop-v1", tier="pilot",
        command="python -m jspace_part2.experiments.a1_base_hard_onehop",
        inputs={"lens": sha256_file(LENS), "items": sha256_file(ITEMS)},
        model=resolve_model(MODEL), seed=4242)
    write_result({"summary": summ, "rows": rows}, OUT, prov)
    registry_append({
        "evidence_id": "a1-base-hard-onehop-v1", "tier": "pilot",
        "what": f"base reverse-dissociation ceiling check on HARD one-hop "
                f"(n={n} dev set): {summ}",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
