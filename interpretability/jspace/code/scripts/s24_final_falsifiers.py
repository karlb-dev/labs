# Final falsifier pass (VM5, post-completion bonus window): one OLMo
# load, three blocks, each discharging a caveat the ledger records.
#
#   A pool_size   [SL1-C2 falsifier] frozen selection from a VOCAB-SIZED
#                 random dictionary (100k rows, same mechanism/dose). If
#                 it reproduces frozen-J's deletion, the P2 effect was
#                 alignment-depth, not content. n=30 twohop readouts +
#                 prose NLL.
#   B filler      [SL1-C6 falsifier] length-matched NON-reasoning think
#                 padding (400 filler tokens, then forced </think>). If
#                 recall recovers like real CoT did (0.23->0.80), the P5
#                 rescue was length/compute, not externalized reasoning.
#   C fanout      [SL1-C3 falsifier] s6's broadcast fan-out, re-run with
#                 the v2 ENERGY-matched pools: are top-J directions read
#                 by more downstream components than energy-matched non-J
#                 / random directions? (v1 only had variance-UNmatched
#                 controls here.)
#
# Resumable per block; -> metrics/final_falsifiers.json
import json
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
OUT = RUN_DIR_V2 / "metrics" / "final_falsifiers.json"
FILLER_TOKENS = 400
ANSWER_TOKENS = 32
FAN_SOURCES = (24, 32, 40)
Z_THRESH = 3.0


def res_io():
    return read_json(OUT) if OUT.exists() else {}


# ---------------------------------------------------------------- block A
def block_a_pool_size(model, hf, tok, lens, tasks):
    res = res_io()
    if "pool_size" in res and "--force" not in sys.argv:
        log("block A (pool_size) already done")
        return
    W = hf.lm_head.weight
    V, d = W.shape
    rdv = {}
    for l in BAND:
        rng = np.random.default_rng(4000 + l)
        # vocab-sized random dictionary, built in fp16 chunks to bound RAM
        rows = []
        for s in range(0, V, 32768):
            n = min(32768, V - s)
            R = torch.tensor(rng.standard_normal((n, d)),
                             dtype=torch.float32, device="cuda")
            rows.append(torch.nn.functional.normalize(R, dim=1).half())
            del R
        rdv[l] = torch.cat(rows)
        del rows
        torch.cuda.empty_cache()
    log(f"block A: vocab-sized ({V} rows) random dictionaries built")
    ab = s7.Ablator(model.layers)
    out = {"n_rows": int(V), "conditions": {}}
    with ab:
        for tname in ("twohop", "twohop_lp", "prose_nll"):
            t0 = time.time()
            key = {"twohop_lp": "twohop"}.get(tname, tname)
            scores = []
            for it in tasks[key]:
                ab.mode = None
                p = it.get("prompt") or it.get("text")
                ab.mode = ("static", frozen_projectors(model, rdv, p))
                scores += s7.run_task(tname, "frozen_rand_vocab10", model,
                                      hf, tok, {key: [it]})
                ab.mode = None
            e = s7.boot_ci(scores)
            e["seconds"] = round(time.time() - t0)
            out["conditions"][tname] = e
            log(f"A frozen_rand_vocab10 {tname:>10}: {e['mean']:.3f} "
                f"[{e['ci_lo']:.3f},{e['ci_hi']:.3f}] ({e['seconds']}s)")
    del rdv
    torch.cuda.empty_cache()
    fr = read_json(RUN_DIR_V2 / "metrics" / "frozen_ablation.json")
    out["reference"] = {c: {t: fr["conditions"][c][t]["mean"]
                            for t in ("twohop", "twohop_lp", "prose_nll")}
                        for c in ("none", "frozen_j10", "frozen_rand10")}
    res = res_io()
    res["pool_size"] = out
    atomic_write_json(res, OUT)


# ---------------------------------------------------------------- block B
def block_b_filler(model, hf, tok, lens, tasks):
    res = res_io()
    if "filler" in res and "--force" not in sys.argv:
        log("block B (filler) already done")
        return
    jd, rd = build_dicts(lens, hf)
    phrase = ("Hmm. Let me take a moment and think about this carefully, "
              "going slowly, one small step at a time. ")
    ids = tok(phrase * 60, add_special_tokens=False).input_ids[:FILLER_TOKENS]
    filler = tok.decode(ids)
    out = {"filler_tokens": len(ids), "n": 0, "conditions": {}}
    ab = s7.Ablator(model.layers)
    with ab:
        for cond in ("none", "frozen_j10", "frozen_rand10"):
            t0 = time.time()
            hits = []
            for it in tasks["twohop"]:
                user = it["prompt"].strip() + \
                    " ...?\nAnswer with the missing word."
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": user}], tokenize=False,
                    add_generation_prompt=True)
                # Olmo-3-Think's generation prompt ends with an open
                # <think>; pad it with non-reasoning filler, then force
                # closure — same construction as s8's suppressed mode,
                # plus length matched to the P5 rescue cap.
                prompt = rendered + "\n" + filler + "\n</think>\n\n"
                ab.mode = None
                if cond != "none":
                    dicts = jd if cond == "frozen_j10" else rd
                    ab.mode = ("static",
                               frozen_projectors(model, dicts, rendered))
                gen = s7.generate(hf, tok, prompt, ANSWER_TOKENS)
                ab.mode = None
                hits.append(float(it["answer"].strip().lower()
                                  in gen.lower()))
            e = s7.boot_ci(hits)
            e["seconds"] = round(time.time() - t0)
            out["conditions"][cond] = e
            out["n"] = len(hits)
            log(f"B {cond:>14}+filler400: {e['mean']:.3f} "
                f"[{e['ci_lo']:.3f},{e['ci_hi']:.3f}] ({e['seconds']}s)")
            res = res_io()
            res.setdefault("filler", out)
            res["filler"] = out
            atomic_write_json(res, OUT)
    cr = read_json(RUN_DIR_V2 / "metrics" / "cot_rescue.json")
    out["reference"] = {
        "rescue_frozen_j10_any": 0.80, "rescue_frozen_rand10_any": 0.93,
        "nothink": cr.get("nothink_reference", {})}
    res = res_io()
    res["filler"] = out
    atomic_write_json(res, OUT)
    del jd, rd
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- block C
def block_c_fanout(hf, lens):
    res = res_io()
    if "fanout" in res and "--force" not in sys.argv:
        log("block C (fanout) already done")
        return
    em = read_json(RUN_DIR_V2 / "metrics" / "energy_match.json")
    plan = {int(l): v for l, v in em["plan"].items()}
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    d = W_U.shape[1]
    n_layers = hf.config.num_hidden_layers
    rng = np.random.default_rng(7)
    out = {"z_thresh": Z_THRESH, "source_layers": {}}
    for l_src in FAN_SOURCES:
        t0 = time.time()
        st = torch.load(RUN_DIR / "metrics" / "layer_state"
                        / f"layer_{l_src}.pt", weights_only=True)
        J = lens.jacobians[l_src].to(W_U.device)
        ids = st["top_dir_ids"]
        D_j = torch.nn.functional.normalize(
            (W_U[ids[:20]] * g[None, :]) @ J, dim=1)
        # v2 energy-matched pools, reconstructed with the s11/s17 recipes
        S = torch.nn.functional.normalize(
            (W_U[ids[:256]] * g[None, :]) @ J, dim=1)
        Qs, _ = torch.linalg.qr(S.T)
        P = st["pca_evecs"].T.to(W_U.device)
        P_perp = P - (P @ Qs) @ Qs.T
        keep = P_perp.norm(dim=1) > 0.5
        Qn, _ = torch.linalg.qr(torch.nn.functional.normalize(
            P_perp[keep], dim=1).T)
        rng1 = np.random.default_rng(1000 + l_src)
        R512 = torch.tensor(rng1.standard_normal((512, d)),
                            dtype=torch.float32, device=W_U.device)
        Qr, _ = torch.linalg.qr(R512.T)
        groups = {
            "J_top20": D_j,
            "vmatch_nonJ": Qn[:, plan[l_src]["vmatch_nonJ_k20"]].T,
            "vmatch_rand": Qr[:, plan[l_src]["vmatch_rand_k20"]].T,
        }
        R_base = torch.nn.functional.normalize(
            torch.tensor(rng.standard_normal((192, d)),
                         dtype=torch.float32, device=W_U.device), dim=1)
        counts = {k: torch.zeros(v.shape[0], device=W_U.device)
                  for k, v in groups.items()}
        n_comp = 0
        for lc in range(l_src + 1, n_layers):
            blk = hf.model.layers[lc]
            for W in (blk.self_attn.q_proj.weight, blk.self_attn.k_proj.weight,
                      blk.self_attn.v_proj.weight, blk.mlp.gate_proj.weight,
                      blk.mlp.up_proj.weight):
                Wf = W.detach().float()
                base = (Wf @ R_base.T).norm(dim=0)
                mu, sd = base.mean(), base.std()
                for k, D in groups.items():
                    z = ((Wf @ D.float().T).norm(dim=0) - mu) / sd
                    counts[k] += (z > Z_THRESH).float()
                n_comp += 1
                del Wf
        row = {"n_components": n_comp}
        for k, c in counts.items():
            cl = c.cpu().tolist()
            row[k] = {"mean": float(np.mean(cl)), "median": float(np.median(cl)),
                      "min": float(np.min(cl)), "max": float(np.max(cl)),
                      "n_dirs": len(cl), "counts": [int(x) for x in cl]}
        out["source_layers"][str(l_src)] = row
        log(f"C L{l_src}: J {row['J_top20']['mean']:.1f} | nonJ(m) "
            f"{row['vmatch_nonJ']['mean']:.1f} | rand(m) "
            f"{row['vmatch_rand']['mean']:.1f} of {n_comp} comps "
            f"({time.time()-t0:.0f}s)")
    v1 = read_json(RUN_DIR / "metrics" / "broadcast.json")
    out["v1_reference_note"] = ("v1 groups (unmatched): J 73-94, random 0, "
                                "nonJ-top-variance 77-125 readers")
    del W_U
    res = res_io()
    res["fanout"] = out
    atomic_write_json(res, OUT)


def main() -> None:
    seed_all()
    (RUN_DIR_V2 / "metrics").mkdir(parents=True, exist_ok=True)
    lens = JacobianLens.load(str(RUN_DIR / "lens" / "olmo32bthink_lens.pt"))
    model, hf, tok = load_model("main")
    rng = np.random.default_rng(0)
    tasks = s7.build_tasks(rng)
    tasks["twohop"] = tasks["twohop"][:30]
    block_a_pool_size(model, hf, tok, lens, tasks)
    block_b_filler(model, hf, tok, lens, tasks)
    block_c_fanout(hf, lens)
    log(f"s24 done -> {OUT}")


if __name__ == "__main__":
    main()
