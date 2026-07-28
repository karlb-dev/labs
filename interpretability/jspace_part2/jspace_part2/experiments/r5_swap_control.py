# R5/G4 — the released probe-swap experiment as the assay's POSITIVE
# CONTROL (addendum §11.3): can the instrument move the model's answer in
# a PREDICTED direction by swapping the silent bridge in J-space?
#
# Per item (probe-swap fields: intermediate e.g. Brazil, swap_to e.g.
# Mexico, answer/swap_answer e.g. Portuguese/Spanish): at each band layer,
# project out the bridge token's J-dictionary direction and inject the
# swap token's direction (alpha_rel * ||h|| per position, prefill-only).
# Conditions: none / swap_j / swap_rand (bridge removed + matched random
# unit injection — controls for remove+perturb without swap content).
# Endpoints: Δ(answer_seq_lp) for BOTH candidate answers + first-token
# argmax flip rate between them.
#
# Calibration (recorded, addendum-sanctioned for positive controls): the
# first 10 items sweep alpha_rel ∈ {0.05, 0.1, 0.2}; the chosen dose runs
# on the remaining 50 items as the measured cells.
# Usage: python -m jspace_part2.experiments.r5_swap_control [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ..battery import answer_variants, seq_lp_from_logits
from ..dictionaries import build_j_dictionaries
from ..lib import sha256_file
from ..protected_dynamic import ProtectedDynamicAblator
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

MODEL_DIR = "/content/models/olmo3-think"   # plain dir (Drive rsync)
LENS = ("/content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726/"
        "lens/olmo32bthink_lens.pt")
PROBE_SWAP = Path("/content/jacobian-lens/data/experiments/probe-swap.json")
OUT_DIR = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
               "metrics/olmo3-think/r5_swap")
BAND = list(range(20, 45, 2))
ALPHAS = (0.05, 0.1, 0.2)
N_CAL = 10


def first_id(tok, word):
    return tok(f" {word.strip()}", add_special_tokens=False).input_ids[0]


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "r5_swap.json"
    if out_json.exists() and "--force" not in sys.argv:
        print("exists; skipping")
        return
    import transformers
    import jlens
    from jlens import JacobianLens

    snap = Path(MODEL_DIR)
    tok = transformers.AutoTokenizer.from_pretrained(str(snap))
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        str(snap), dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.load(LENS)
    jd = build_j_dictionaries(hf, lens, BAND)
    items = json.loads(PROBE_SWAP.read_text())["items"][:60]
    g = torch.Generator().manual_seed(777)
    rand_dir = torch.nn.functional.normalize(
        torch.randn(model.d_model, generator=g), dim=0).cuda()
    ab = ProtectedDynamicAblator(model.layers, BAND)

    def score_pair(it, mode):
        """Returns (lp_answer, lp_swap_answer, argmax_pick)."""
        out = {}
        for tag, ans in (("orig", it["answer"]), ("swap", it["swap_answer"])):
            best = None
            for v in answer_variants(ans):
                text = it["prompt"].rstrip() + v
                n_prompt = model.encode(it["prompt"].rstrip(),
                                        max_length=512).shape[1]
                ab.mode = mode
                ids = model.encode(text, max_length=512)
                logits = hf(input_ids=ids, use_cache=False).logits[0]\
                    .float().cpu()
                ab.mode = None
                lp = seq_lp_from_logits(ids, logits, n_prompt)
                best = lp if best is None or lp > best else best
                if tag == "orig" and v == answer_variants(ans)[0]:
                    last = logits[n_prompt - 1]
            out[tag] = best
        a, s = first_id(tok, it["answer"]), first_id(tok, it["swap_answer"])
        out["pick_swap"] = float(last[s] > last[a])
        return out

    def make_mode(it, alpha, inject_rand):
        rm, inj = {}, {}
        for l in BAND:
            b = jd[l][first_id(tok, it["intermediate"])].float()
            rm[l] = (b / b.norm()).reshape(-1, 1).cuda()
            inj[l] = (rand_dir if inject_rand
                      else jd[l][first_id(tok, it["swap_to"])].float().cuda())
        return {"inject": inj, "remove": rm, "alpha_rel": alpha}

    # ---- calibration on the first N_CAL items
    cal = {}
    with ab:
        for alpha in ALPHAS:
            flips, dlp = 0, 0.0
            for it in items[:N_CAL]:
                base = score_pair(it, None)
                swp = score_pair(it, make_mode(it, alpha, False))
                flips += swp["pick_swap"]
                dlp += (swp["swap"] - base["swap"])
            cal[alpha] = {"flip_rate": flips / N_CAL,
                          "mean_dlp_swap": round(dlp / N_CAL, 3)}
            print(f"alpha {alpha}: {cal[alpha]}", flush=True)
    alpha_star = max(ALPHAS, key=lambda a: cal[a]["flip_rate"])

    # ---- measured cells on items[N_CAL:]
    rows = []
    with ab:
        for it in items[N_CAL:]:
            r = {"item": it["name"]}
            r["none"] = score_pair(it, None)
            r["swap_j"] = score_pair(it, make_mode(it, alpha_star, False))
            r["swap_rand"] = score_pair(it, make_mode(it, alpha_star, True))
            rows.append(r)
    n = len(rows)
    summ = {"alpha_star": alpha_star, "calibration": cal, "n": n}
    for cond in ("none", "swap_j", "swap_rand"):
        summ[cond] = {
            "flip_rate": round(sum(r[cond]["pick_swap"] for r in rows) / n, 3),
            "mean_lp_swapans": round(sum(r[cond]["swap"] for r in rows) / n, 3),
            "mean_lp_origans": round(sum(r[cond]["orig"] for r in rows) / n, 3)}
    prov = Provenance(
        evidence_id="r5-swap-positive-control-think-v1", tier="pilot",
        command="python -m jspace_part2.experiments.r5_swap_control",
        inputs={"lens": sha256_file(LENS),
                "probe_swap": sha256_file(PROBE_SWAP)},
        model=resolve_model(str(snap)), seed=777)
    write_result({"summary": summ, "rows": rows}, out_json, prov)
    registry_append({
        "evidence_id": "r5-swap-positive-control-think-v1", "tier": "pilot",
        "what": f"G4 positive control (released probe-swap, remove-bridge+"
                f"inject-swap in J-space, alpha {alpha_star}): {summ['swap_j']}"
                f" vs rand {summ['swap_rand']} vs none {summ['none']}",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(out_json), "sha256": sha256_file(out_json)}]})
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
