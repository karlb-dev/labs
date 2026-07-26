# v2 Priority 5: does externalized reasoning rescue the frozen-J recall
# deletion?
#
# P2 found the one real causal handle: per-item frozen top-10 J-ablation
# deletes the retrieved fact in no-think mode (recall 0.58->0.23, answer
# logprob -2.9 nats). The paper's identity claim predicts CoT partially
# rescues workspace ablation (externalization); v1's suppression asymmetry
# (think acc >= suppressed acc) points the same way. Test: the SAME frozen
# per-item projectors, but with think-mode prompting (open <think>, up to
# 400 reasoning tokens) — can the model re-derive or re-retrieve the fact
# out loud when its J-space content is removed?
#
# Conditions: none, frozen_j10, frozen_rand10 (mechanism control),
# jspace_k20 (static-span null control). Items: twohop 30, onehop 15,
# arithmetic 15, sql 15 (v2 subset; per-item resumable). Scoring: answer
# string present in the POST-</think> segment (headline) and anywhere
# (secondary); think-segment hit recorded separately.
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (RUN_DIR, RUN_DIR_V2, atomic_write_json, load_model,
                        log, read_json, seed_all)

import numpy as np
import torch
from jlens import JacobianLens

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7_ablation as s7  # noqa: E402
from s12_frozen_ablation import build_dicts, frozen_projectors  # noqa: E402

BAND = s7.BAND
OUT = RUN_DIR_V2 / "metrics" / "cot_rescue.json"
MAX_THINK = 400
# PLAN_v3 capped grid (VM5, 3h window): decision-relevant cells only —
# the two frozen conditions under open-<think>. `none` reference = v1 s8
# think-mode baselines + frozen_rand10 (=baseline everywhere in P2); the
# static-span rescue cell is dead compute (P1: static does nothing).
# Banked 4-cond cells from VM4 (items 0-3) stay in the JSON untouched.
CONDS = ("frozen_j10", "frozen_rand10")


def build_items():
    rng = np.random.default_rng(0)
    tasks = s7.build_tasks(rng)
    items = []
    for it in tasks["twohop"][:30]:
        items.append({"kind": "twohop", "user": it["prompt"].strip()
                      + " ...?\nAnswer with the missing word.",
                      "sel_prompt": it["prompt"], "answer": it["answer"]})
    for it in tasks["onehop"][:15]:
        items.append({"kind": "onehop", "user": it["prompt"].strip()
                      + " ...?\nAnswer with the missing word.",
                      "sel_prompt": it["prompt"],
                      "answer": it["answer"].strip()})
    for it in tasks["arithmetic"][:15]:
        q = it["prompt"].split("\nA:")[0].replace("Q: ", "")
        items.append({"kind": "arithmetic", "user": q,
                      "sel_prompt": it["prompt"], "answer": it["answer"]})
    # PLAN_v3: SQL dropped from the capped grid (flaky 3-schema cell; iids
    # unchanged because sql occupied the tail of the list).
    for i, it in enumerate(items):
        it["iid"] = i
    return items


@torch.no_grad()
def gen_think(hf, tok, rendered, max_new):
    ids = tok(rendered, return_tensors="pt").input_ids.cuda()
    out = hf.generate(ids, max_new_tokens=max_new, do_sample=False,
                      pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)


def score(it, text) -> dict:
    post = text.split("</think>")[-1] if "</think>" in text else ""
    think = text.split("</think>")[0]
    if it["kind"] == "sql":
        ok_post = all(re.search(c, post) for c in it["checks"])
        ok_any = all(re.search(c, text) for c in it["checks"])
        ok_think = all(re.search(c, think) for c in it["checks"])
    else:
        a = it["answer"].strip().lower()
        ok_post = a in post.lower()
        ok_any = a in text.lower()
        ok_think = a in think.lower()
    return {"post": float(ok_post), "any": float(ok_any),
            "think": float(ok_think),
            "closed_think": float("</think>" in text)}


def main() -> None:
    seed_all()
    (RUN_DIR_V2 / "metrics").mkdir(parents=True, exist_ok=True)
    lens = JacobianLens.load(str(RUN_DIR / "lens" / "olmo32bthink_lens.pt"))
    model, hf, tok = load_model("main")
    jd, rd = build_dicts(lens, hf)
    # static-span control (same construction as s11's integrity cell)
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    span20 = {}
    for l in BAND:
        st = torch.load(RUN_DIR / "metrics" / "layer_state" / f"layer_{l}.pt",
                        weights_only=True)
        D = torch.nn.functional.normalize(
            (W_U[st["top_dir_ids"][:20]] * g[None, :])
            @ lens.jacobians[l].to(W_U.device), dim=1)
        Q, _ = torch.linalg.qr(D.T)
        span20[l] = Q.contiguous()
    del W_U

    items = build_items()
    res = read_json(OUT) if OUT.exists() else {
        "conds": list(CONDS), "max_think": MAX_THINK, "items": {}}
    ab = s7.Ablator(model.layers)
    with ab:
        for it in items:
            key = str(it["iid"])
            res["items"].setdefault(key, {"kind": it["kind"]})
            row = res["items"][key]
            for cond in CONDS:
                if cond in row and "--force" not in sys.argv:
                    continue
                t0 = time.time()
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": it["user"]}],
                    tokenize=False, add_generation_prompt=True)
                ab.mode = None
                if cond == "none":
                    pass
                elif cond == "jspace_k20":
                    ab.mode = ("static", span20)
                else:
                    dicts = jd if cond == "frozen_j10" else rd
                    ab.mode = ("static",
                               frozen_projectors(model, dicts, rendered))
                text = gen_think(hf, tok, rendered, MAX_THINK)
                ab.mode = None
                row[cond] = score(it, text)
                row[cond]["seconds"] = round(time.time() - t0)
                atomic_write_json(res, OUT)
            if it["iid"] % 5 == 0:
                log(f"item {it['iid']:>2} ({it['kind']:>10}): " + " ".join(
                    f"{c}={row[c]['post']:.0f}" for c in CONDS if c in row))

    # aggregate: per (kind, cond) accuracy + paired rescue stats
    agg = {}
    for kind in ("twohop", "onehop", "arithmetic", "sql"):
        ks = [r for r in res["items"].values() if r["kind"] == kind]
        if not ks:
            continue
        agg[kind] = {c: {
            "post": float(np.mean([r[c]["post"] for r in ks])),
            "any": float(np.mean([r[c]["any"] for r in ks])),
            "closed": float(np.mean([r[c]["closed_think"] for r in ks])),
            "n": len(ks)} for c in CONDS if all(c in r for r in ks)}
    res["agg"] = agg
    # headline rescue comparison vs the no-think P2 numbers (from metrics)
    fr = read_json(RUN_DIR_V2 / "metrics" / "frozen_ablation.json")
    res["nothink_reference"] = {
        "frozen_j10_twohop": fr["conditions"]["frozen_j10"]["twohop"]["mean"],
        "none_twohop": fr["conditions"]["none"]["twohop"]["mean"]}
    atomic_write_json(res, OUT)
    log("agg: " + json.dumps({k: {c: round(v["post"], 2)
                                  for c, v in d.items()}
                              for k, d in agg.items()}))
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
