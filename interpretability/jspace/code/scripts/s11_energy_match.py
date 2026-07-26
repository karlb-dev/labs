# v2 Priority 1: variance-matched controls for the causal battery.
#
# v1's non-J PCA control removed far more activation energy than the J-span
# at equal rank (top PCs are fat; J-dirs are thin), so "non-J does the real
# damage" could be an energy artifact. Here we measure, per band layer, the
# raw activation energy the k-dim J-span projection actually removes, then
# select control subspaces whose removed energy MATCHES it:
#   vmatch_rand_k{k}  prefix of a 512-dim random orthonormal pool
#   vmatch_nonJ_k{k}  prefix (or contiguous window, if a prefix can't get
#                     within [0.8, 1.25]x) of the v1-constructed non-J PC
#                     pool, orthonormalized in PC order
# Energy accounting is raw (uncentered) h — exactly what the ablation hook
# projects out — over the s5 descriptive prompt set. Verification per layer
# lands in metrics/energy_match.json BEFORE the battery starts.
#
# Battery: identical to s7 (same tasks, same seed 0 => byte-identical item
# sets, same 2000-resample bootstrap). `none` and `jspace_k*` are REUSED
# from v1 (identical protocol; one integrity cell is re-run and must agree
# exactly). New cells are resumable per (condition, task).
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (RUN_DIR, RUN_DIR_V2, atomic_write_json, die,
                        load_model, log, read_json, seed_all)

import numpy as np
import torch
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7_ablation as s7  # noqa: E402  (build_tasks, Ablator, run_task, ...)

BAND = s7.BAND
DOSES = s7.DOSES
EM_OUT = RUN_DIR_V2 / "metrics" / "energy_match.json"
AB_OUT = RUN_DIR_V2 / "metrics" / "ablation_v2.json"
STATE_DIR = RUN_DIR / "metrics" / "layer_state"
PROMPTS = RUN_DIR / "config" / "prompts" / "descriptive_prompts.jsonl"
RAND_POOL = 512


def build_bases(lens, hf):
    """Per band layer: J-span prefixes, non-J PC pool (v1 construction,
    orthonormalized in PC order), random pool. All fp32 on GPU."""
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    out = {}
    for l in BAND:
        st = torch.load(STATE_DIR / f"layer_{l}.pt", weights_only=True)
        J = lens.jacobians[l].to(W_U.device)
        ids = st["top_dir_ids"]
        D_top = torch.nn.functional.normalize(
            (W_U[ids[:max(DOSES)]] * g[None, :]) @ J, dim=1)
        Qj, _ = torch.linalg.qr(D_top.T)                    # [d, 40]
        S = torch.nn.functional.normalize(
            (W_U[ids[:256]] * g[None, :]) @ J, dim=1)
        Qs, _ = torch.linalg.qr(S.T)
        P = st["pca_evecs"].T.to(W_U.device)                # [64, d]
        P_perp = P - (P @ Qs) @ Qs.T
        keep = P_perp.norm(dim=1) > 0.5
        P_ok = torch.nn.functional.normalize(P_perp[keep], dim=1)
        Qn, _ = torch.linalg.qr(P_ok.T)                     # [d, <=64] PC order
        rng = np.random.default_rng(1000 + l)
        R = torch.tensor(rng.standard_normal((RAND_POOL, P.shape[1])),
                         dtype=torch.float32, device=W_U.device)
        Qr, _ = torch.linalg.qr(R.T)                        # [d, 512]
        out[l] = {"J": Qj.contiguous(), "nonJ": Qn.contiguous(),
                  "rand": Qr.contiguous()}
    del W_U
    return out


@torch.no_grad()
def energy_pass(model, bases):
    """Stream raw h over the descriptive prompts; per layer accumulate
    ||h||^2, ||Qj_k^T h||^2 per dose, and per-direction energies of both
    control pools."""
    recs = [json.loads(x) for x in PROMPTS.read_text().splitlines()]
    texts = [r.get("prompt") or r.get("text") for r in recs]
    acc = {l: {"n": 0, "h2": 0.0,
               "J": np.zeros(len(DOSES)),
               "nonJ": np.zeros(bases[l]["nonJ"].shape[1]),
               "rand": np.zeros(RAND_POOL)} for l in BAND}
    t0 = time.time()
    for i, text in enumerate(texts):
        ids = model.encode(text, max_length=128)
        with ActivationRecorder(model.layers, at=BAND) as rec:
            model.forward(ids)
        for l in BAND:
            h = rec.activations[l][0].float()               # [T, d]
            a = acc[l]
            a["n"] += h.shape[0]
            a["h2"] += float((h * h).sum())
            for j, k in enumerate(DOSES):
                p = h @ bases[l]["J"][:, :k]
                a["J"][j] += float((p * p).sum())
            for pool in ("nonJ", "rand"):
                p = h @ bases[l][pool]
                a[pool] += (p * p).sum(0).cpu().numpy()
        if i % 50 == 49:
            log(f"energy pass {i + 1}/{len(texts)} ({time.time() - t0:.0f}s)")
    return acc


def match(acc):
    """Choose matched ranks; return (em_report, plan) where plan[l][cond]
    is the column index list into the pool."""
    em, plan = {}, {l: {} for l in BAND}
    for l in BAND:
        a = acc[l]
        em[l] = {"n_pos": a["n"], "mean_h2": a["h2"] / a["n"], "doses": {}}
        for j, k in enumerate(DOSES):
            tgt = a["J"][j] / a["n"]
            d = {"E_jspan": tgt, "share_of_h2": tgt / em[l]["mean_h2"]}
            for pool in ("nonJ", "rand"):
                e = a[pool] / a["n"]
                cum = np.cumsum(e)
                m = int(np.argmin(np.abs(cum - tgt))) + 1
                ratio = float(cum[m - 1] / tgt)
                method, cols = "prefix", list(range(m))
                if pool == "nonJ" and not (0.8 <= ratio <= 1.25):
                    best = (1e9, None)
                    for s in range(len(e)):
                        c = np.cumsum(e[s:])
                        j2 = int(np.argmin(np.abs(np.log(c / tgt))))
                        r2 = float(c[j2] / tgt)
                        key = abs(np.log(r2))
                        if key < best[0] - 1e-12:
                            best = (key, (s, s + j2 + 1, r2))
                    s, t, r2 = best[1]
                    method, cols, ratio = f"window[{s}:{t}]", \
                        list(range(s, t)), r2
                d[f"{pool}_m"] = len(cols)
                d[f"{pool}_ratio"] = ratio
                d[f"{pool}_method"] = method
                plan[l][f"vmatch_{pool}_k{k}"] = cols
            em[l]["doses"][k] = d
    return em, plan


def main() -> None:
    seed_all()
    (RUN_DIR_V2 / "metrics").mkdir(parents=True, exist_ok=True)
    lens = JacobianLens.load(str(RUN_DIR / "lens" / "olmo32bthink_lens.pt"))
    model, hf, tok = load_model("main")
    bases = build_bases(lens, hf)
    log(f"bases built for {len(BAND)} layers")

    if EM_OUT.exists() and "--force" not in sys.argv:
        saved = read_json(EM_OUT)
        em = {int(l): v for l, v in saved["per_layer"].items()}
        plan = {int(l): v for l, v in saved["plan"].items()}
        log("energy match loaded from existing file")
    else:
        acc = energy_pass(model, bases)
        em, plan = match(acc)
        atomic_write_json(
            {"band": BAND, "doses": list(DOSES), "rand_pool": RAND_POOL,
             "corpus": "v1 descriptive_prompts.jsonl (raw h, all positions)",
             "per_layer": em, "plan": plan}, EM_OUT)
        for l in BAND[:4] + BAND[-1:]:
            d = em[l]["doses"][20]
            log(f"L{l} k20: E_J={d['E_jspan']:.1f} "
                f"nonJ m={d['nonJ_m']} r={d['nonJ_ratio']:.2f} "
                f"({d['nonJ_method']}) rand m={d['rand_m']} "
                f"r={d['rand_ratio']:.2f}")
        log(f"wrote {EM_OUT}")

    # ---- assemble condition subspaces
    subs = {}
    for k in DOSES:
        for pool in ("nonJ", "rand"):
            cond = f"vmatch_{pool}_k{k}"
            subs[cond] = {l: bases[l][pool][:, plan[l][cond]].contiguous()
                          for l in BAND}

    # ---- battery (reusing s7 machinery; v1 rows imported)
    rng = np.random.default_rng(0)
    tasks = s7.build_tasks(rng)
    v1 = read_json(RUN_DIR / "metrics" / "ablation_results.json")
    res = read_json(AB_OUT) if AB_OUT.exists() else {
        "band": BAND, "doses": list(DOSES), "conditions": {},
        "reused_from_v1": ["none", "jspace_k10", "jspace_k20", "jspace_k40"]}
    for c in res["reused_from_v1"]:
        res["conditions"][c] = v1["conditions"][c]

    ab = s7.Ablator(model.layers)
    task_names = ["twohop", "onehop", "arithmetic", "sql", "prose_nll",
                  "grammar", "twohop_lp", "samples", "arithmetic_v2"]
    with ab:
        # integrity cell: deterministic greedy readout must match v1 exactly
        if "integrity" not in res:
            ab.mode = ("static", {l: bases[l]["J"][:, :10] for l in BAND})
            sc = s7.run_task("onehop", "jspace_k10", model, hf, tok, tasks)
            ab.mode = None
            mine, theirs = float(np.mean(sc)), \
                v1["conditions"]["jspace_k10"]["onehop"]["mean"]
            if abs(mine - theirs) > 1e-9:
                die(f"integrity cell mismatch: {mine} vs v1 {theirs}")
            res["integrity"] = {"cell": "jspace_k10/onehop", "mean": mine,
                                "matches_v1": True}
            atomic_write_json(res, AB_OUT)
            log(f"integrity cell OK ({mine:.3f} == v1)")
        for cond in subs:
            res["conditions"].setdefault(cond, {})
            for tname in task_names:
                if tname in res["conditions"][cond] and "--force" not in sys.argv:
                    continue
                t0 = time.time()
                ab.mode = ("static", subs[cond])
                extra = {}
                scores = s7.run_task(tname, cond, model, hf, tok, tasks, extra)
                ab.mode = None
                entry = s7.boot_ci(scores)
                entry["seconds"] = round(time.time() - t0)
                entry.update(extra)
                res["conditions"][cond][tname] = entry
                atomic_write_json(res, AB_OUT)
                log(f"{cond:>16} {tname:>13}: {entry['mean']:.3f} "
                    f"[{entry['ci_lo']:.3f},{entry['ci_hi']:.3f}] "
                    f"({entry['seconds']}s)")
    log(f"wrote {AB_OUT}")


if __name__ == "__main__":
    main()
