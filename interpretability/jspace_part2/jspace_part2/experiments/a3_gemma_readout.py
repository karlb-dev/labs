# A3 VERDICT — does the Jacobian transport rescue Gemma's mid-band
# readability where the vanilla logit lens cannot?
#
# The state of the question:
#   * `a3-gemma-gate-v1`  — a 2-prompt MICRO-fit read the answer at rank
#     ~52k of 262k mid-band. Too small a fit to conclude anything.
#   * `a3-gemma-deepband-logit-v1` — the VANILLA logit lens is opaque
#     through the whole mid-band (median rank 73k @L24 ... 10.7k @L40),
#     resolving abruptly at L42-44. The paper's relative depths (37-62%,
#     L22-L37) sit entirely inside the opaque zone.
#   * fit-health observation (slice 0) — ‖J‖/√d rises 64× across the band
#     (0.124 @L22 -> 7.94 @L52), where OLMo's fits sat flat at ≈0.94.
# So the honest question left open is whether a PROPERLY FITTED J-lens
# (120 prompts, same recipe as the OLMo lenses) reads what the logit lens
# cannot. This script answers it on the merged lens, and the answer is
# the A3 family datum either way.
#
# Method: for each probe, apply the lens at every source layer with
# use_jacobian=True (J transport) and False (vanilla logit lens) at the
# final prompt position, and record the answer token's rank under each.
# Paired per-item, so the J-vs-logit contrast is within-probe. The
# decision rule is PREREGISTERED HERE, before the merged lens exists:
#
#   RESCUE      if median J-rank <= 100 at any layer <= L37 (a paper-band
#               depth) AND J beats the logit lens there by >= 10x
#   PARTIAL     if J beats logit by >= 10x mid-band but stays > 100
#   NO_RESCUE   otherwise -> the opacity is a property of the model's
#               computation, not of the readout basis, and Gemma-4 is
#               reported as a boundary case for the paper's method
#
# Tier: pilot. Cheap: one recorded forward per probe.
# Usage: python -m jspace_part2.experiments.a3_gemma_readout [--allow-dirty]
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

MODEL = "/content/models/gemma4-31b-it"
RUN_DIR_P2 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "part2_20260727")
LENS = RUN_DIR_P2 / "lens" / "gemma431_lens.pt"
OUT = RUN_DIR_P2 / "metrics" / "gemma4-31b" / "a3_readout_verdict.json"
PAPER_BAND_MAX = 37       # deepest layer inside the paper's relative band
RANK_OK = 100
GAIN_OK = 10.0


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    if not LENS.exists():
        raise SystemExit(f"merged lens not present yet: {LENS}")
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
    lens = JacobianLens.load(str(LENS))
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
                a = jl[l][0].float()
                b = ll[l][0].float()
                jr[l] = min(int((a > a[c]).sum()) + 1 for c in cand)
                lr[l] = min(int((b > b[c]).sum()) + 1 for c in cand)
            rows.append({"item_id": it["item_id"], "final_rank": final_rank,
                         "j_ranks": jr, "logit_ranks": lr})
            print(f"  {it['item_id']:24s} final={final_rank:6d} "
                  f"J@22={jr.get(22):7d} log@22={lr.get(22):7d} "
                  f"J@37={jr.get(37):7d} J@44={jr.get(44):7d}", flush=True)

    known = [r for r in rows if r["final_rank"] <= 3]
    med_j = {l: float(np.median([r["j_ranks"][l] for r in known]))
             for l in layers}
    med_l = {l: float(np.median([r["logit_ranks"][l] for r in known]))
             for l in layers}
    gain = {l: round(med_l[l] / med_j[l], 3) if med_j[l] else None
            for l in layers}

    band = [l for l in layers if l <= PAPER_BAND_MAX]
    j_ok = [l for l in band if med_j[l] <= RANK_OK]
    g_ok = [l for l in band if (gain[l] or 0) >= GAIN_OK]
    if j_ok and g_ok:
        verdict = "GEMMA_RESCUE"
    elif g_ok:
        verdict = "GEMMA_PARTIAL_RESCUE"
    else:
        verdict = "GEMMA_NO_RESCUE"

    summ = {
        "verdict": verdict, "n_probes": len(rows), "n_known": len(known),
        "layers": layers,
        "median_rank_jacobian": med_j, "median_rank_logit": med_l,
        "gain_logit_over_jacobian": gain,
        "decision_rule": {
            "paper_band_max_layer": PAPER_BAND_MAX, "rank_ok": RANK_OK,
            "gain_ok": GAIN_OK,
            "note": "rule fixed in code BEFORE the merged lens existed"},
        "layers_in_band_with_rank_ok": j_ok,
        "layers_in_band_with_gain_ok": g_ok,
        "reading": (
            f"{verdict}: with a full 120-prompt lens, J transport "
            f"{'reads' if j_ok else 'does NOT read'} the answer inside the "
            f"paper's band (layers<={PAPER_BAND_MAX}); median J-rank at L22 "
            f"{med_j.get(22):.0f} vs logit {med_l.get(22):.0f}, at L37 "
            f"{med_j.get(37):.0f} vs {med_l.get(37):.0f}. "
            + ("The Jacobian recovers what the output basis hides — the "
               "earlier opacity was a readout-basis artifact after all."
               if j_ok else
               "The opacity survives a properly fitted Jacobian lens, so it "
               "is a property of where Gemma-4 puts answer information, not "
               "of the readout basis. Gemma-4 is a BOUNDARY CASE for the "
               "paper's method: its workspace band, if any, is not "
               "output-token-aligned at the depths the paper studies.")),
        "seconds": round(time.time() - t0)}
    prov = Provenance(
        evidence_id="a3-gemma-readout-verdict-v1", tier="pilot",
        command="python -m jspace_part2.experiments.a3_gemma_readout",
        inputs={"lens": sha256_file(LENS)},
        model=resolve_model(MODEL))
    write_result({"summary": summ, "rows": rows}, OUT, prov)
    registry_append({
        "evidence_id": "a3-gemma-readout-verdict-v1", "tier": "pilot",
        "what": f"A3 verdict on the full 120-prompt Gemma lens: {summ['reading']}",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
