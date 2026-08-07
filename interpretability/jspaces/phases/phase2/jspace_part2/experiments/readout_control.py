# POSITIVE CONTROL for the readout comparison instrument.
#
# The A3 verdict (`a3-gemma-readout-verdict-v1`) reported something
# stronger than "no rescue": on Gemma the fitted J-lens is WORSE than the
# vanilla logit lens at every identified layer (gain 0.03-0.67, all < 1).
# That is a surprising claim, because the Jacobian BEATING the logit lens
# is the premise of the whole method — so before it is believed, the
# comparison instrument itself must be shown to work.
#
# This runs the identical code path (lens.apply with use_jacobian True vs
# False, same probes, same rank statistic) on a model where the answer is
# known: Olmo-3-32B-Think with its own part-1 120-prompt lens. Part 1
# found J readout superior there.
#
#   PASS  -> median gain (logit-rank / J-rank) > 1 somewhere in OLMo's
#            band: the instrument can detect a J advantage, so Gemma's
#            gain < 1 is a real property of that model/recipe pairing.
#   FAIL  -> the instrument reports no J advantage even on OLMo, so the
#            Gemma "J is worse" finding is an artifact of THIS comparison
#            and must be withdrawn, not published.
#
# Either outcome is banked. Tier: pilot.
# Usage: python -m jspace_part2.experiments.readout_control [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..battery import answer_variants, onehop_items, twohop_items
from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

MODEL = "/content/models/olmo3-think"
LENS = ("/content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726/"
        "lens/olmo32bthink_lens.pt")
RUN_DIR_P2 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "part2_20260727")
OUT = RUN_DIR_P2 / "metrics" / "olmo3-think" / "readout_control.json"
# OLMo's paper-relative band (64 layers): 37/50/62% -> ~L24/32/40
BAND_OF_INTEREST = [24, 32, 40]


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    if OUT.exists() and "--force" not in sys.argv:
        print("exists; skipping")
        return
    import transformers
    import jlens
    from jlens import JacobianLens

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.load(LENS)
    layers = sorted(lens.jacobians.keys())

    probes = onehop_items() + twohop_items(20)
    rows = []
    with torch.no_grad():
        for it in probes:
            prompt = it["prompt"].rstrip()
            cand = [tok(v, add_special_tokens=False).input_ids[0]
                    for v in answer_variants(it["answer"])]
            jl, ml, _ = lens.apply(model, prompt, positions=[-1])
            ll, _, _ = lens.apply(model, prompt, positions=[-1],
                                  use_jacobian=False)
            final = ml[0].float()
            final_rank = min(int((final > final[c]).sum()) + 1 for c in cand)
            jr, lr = {}, {}
            for l in layers:
                a, b = jl[l][0].float(), ll[l][0].float()
                jr[l] = min(int((a > a[c]).sum()) + 1 for c in cand)
                lr[l] = min(int((b > b[c]).sum()) + 1 for c in cand)
            rows.append({"item_id": it["item_id"], "final_rank": final_rank,
                         "j_ranks": jr, "logit_ranks": lr})
            print(f"  {it['item_id']:24s} final={final_rank:5d} "
                  f"J@24={jr.get(24):6d} log@24={lr.get(24):6d} "
                  f"J@40={jr.get(40):6d} log@40={lr.get(40):6d}", flush=True)

    known = [r for r in rows if r["final_rank"] <= 3]
    med_j = {l: float(np.median([r["j_ranks"][l] for r in known]))
             for l in layers}
    med_l = {l: float(np.median([r["logit_ranks"][l] for r in known]))
             for l in layers}
    gain = {l: (round(med_l[l] / med_j[l], 3) if med_j[l] else None)
            for l in layers}
    band = [l for l in BAND_OF_INTEREST if l in layers]
    band_gain = {l: gain[l] for l in band}
    best = max((gain[l] or 0) for l in layers)
    best_band = max((gain[l] or 0) for l in band) if band else 0
    verdict = ("READOUT_CONTROL_PASS" if best_band > 1.0
               else "READOUT_CONTROL_FAIL")

    summ = {
        "verdict": verdict, "model": "olmo3-think",
        "n_probes": len(rows), "n_known": len(known),
        "median_rank_jacobian": med_j, "median_rank_logit": med_l,
        "gain_logit_over_jacobian": gain,
        "paper_band_layers": band, "paper_band_gain": band_gain,
        "best_gain_any_layer": best, "best_gain_in_band": best_band,
        "reading": (
            f"{verdict}: on OLMo-3-32B-Think with its own 120-prompt lens, "
            f"the SAME comparison code gives J-over-logit gain "
            f"{band_gain} in the paper-relative band (best any layer "
            f"{best:.2f}). "
            + ("The instrument CAN detect a Jacobian advantage, so the "
               "Gemma result (gain < 1 everywhere) is a real property of "
               "that model/recipe pairing, not a bug in this comparison."
               if best_band > 1.0 else
               "The instrument reports NO Jacobian advantage even on OLMo, "
               "where part 1 found one. The Gemma 'J is worse than logit' "
               "finding is therefore NOT SAFE to publish — it is confounded "
               "with this comparison's construction (rank-at-final-position, "
               "single readout convention) and must be withdrawn pending a "
               "corrected instrument.")),
        "seconds": round(time.time() - t0)}
    prov = Provenance(
        evidence_id="readout-control-olmo3think-v1", tier="pilot",
        command="python -m jspace_part2.experiments.readout_control",
        inputs={"lens": sha256_file(LENS)}, model=resolve_model(MODEL))
    write_result({"summary": summ, "rows": rows}, OUT, prov)
    registry_append({
        "evidence_id": "readout-control-olmo3think-v1", "tier": "pilot",
        "what": f"Positive control for the readout comparison: {summ['reading']}",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(json.dumps({k: v for k, v in summ.items()}, indent=2))


if __name__ == "__main__":
    main()
