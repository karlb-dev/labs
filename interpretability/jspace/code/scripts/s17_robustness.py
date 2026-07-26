# v2 Priority 6: robustness — second seed + doubled n on the weakest cells.
#
# Reruns the decisive P1/P2 conditions with (a) seed-1 task generation
# (fresh arithmetic items), (b) arithmetic and SQL n 30->60 (same 3 SQL
# schemas — a diversity limit, recorded), (c) a FRESH two-hop set
# (probe-swap items [60:120] when available, else the v1 [:60] reused,
# recorded), (d) seed-1 random pools for the matched-random and
# frozen-random controls. Non-J pools are deterministic (PCA + J-span), so
# their seed-1 variation is items+bootstrap only. Matched ranks m are
# reused from energy_match.json (random-direction energies concentrate
# tightly; assumption recorded).
#
# Conditions: none, jspace_k20, vmatch_rand_k20, vmatch_nonJ_k20,
# frozen_j10, frozen_rand10. Resumable per (condition, task).
# Output: metrics/robustness_seed1.json with a seed0-vs-seed1 comparison.
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
from s12_frozen_ablation import frozen_projectors  # noqa: E402

BAND = s7.BAND
OUT = RUN_DIR_V2 / "metrics" / "robustness_seed1.json"
SEED = 1
# PLAN_v3 P6 trim (VM5): the marquee frozen grid gets the second seed
# first, on the FRESH twohop set, n=30; the static energy-matched cells
# ride along because they are pure readouts (seconds each). Generation
# tasks (arith/sql) and n->60 doubling dropped — cross-seed beats
# bigger-n for credibility per GPU-minute.
CONDS = ("none", "frozen_j10", "frozen_rand10", "jspace_k20",
         "vmatch_rand_k20", "vmatch_nonJ_k20")
TASKS = ("twohop", "twohop_lp")


def build_tasks_v2(rng) -> tuple[dict, dict]:
    base = s7.build_tasks(np.random.default_rng(0))  # onehop/grammar/prose
    tasks = {"onehop": base["onehop"], "grammar": base["grammar"],
             "prose_nll": base["prose_nll"]}
    notes = {}
    ps = json.loads(s7.PROBE_SWAP.read_text())["items"]
    if len(ps) > 60:
        tasks["twohop"] = [{"prompt": it["prompt"], "answer": it["answer"]}
                           for it in ps[60:]]
        notes["twohop"] = f"fresh probe-swap items [60:{len(ps)}] (n={len(ps) - 60})"
    else:
        tasks["twohop"] = base["twohop"]
        notes["twohop"] = f"probe-swap has only {len(ps)} items; v1 set reused"
    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
           ("*", lambda a, b: a * b)]
    arith = []
    for _ in range(60):
        a, b, c, d = [int(x) for x in rng.integers(3, 40, size=4)]
        (o1, f1) = ops[rng.integers(3)]
        (o2, f2) = ops[rng.integers(2)]
        val = f2(f1(a, b), c) - d
        arith.append({"prompt": f"Q: What is (({a} {o1} {b}) {o2} {c}) - {d}?"
                                f" Work step by step, then give the answer."
                                f"\nA: ({a} {o1} {b}) =",
                      "answer": str(val)})
    tasks["arithmetic"] = arith
    schemas = [s for s in [
        ("users(id, name, city_id)", "cities(id, city_name, country)",
         "orders(id, user_id, total)", "Total order value per country.",
         [r"o\.user_id\s*=\s*u\.id|u\.id\s*=\s*o\.user_id",
          r"u\.city_id\s*=\s*c\.id|c\.id\s*=\s*u\.city_id"]),
        ("products(pid, pname, cat_id)", "categories(cat_id, cat_name)",
         "sales(sid, pid, qty, price)", "Revenue by category name.",
         [r"s\.pid\s*=\s*p\.pid|p\.pid\s*=\s*s\.pid",
          r"p\.cat_id\s*=\s*(c|cat)\.cat_id|(c|cat)\.cat_id\s*=\s*p\.cat_id"]),
        ("employees(eid, ename, dept_id)", "departments(dept_id, dname)",
         "salaries(eid, amount, year)", "Total salary per department name.",
         [r"(s|sal)\.eid\s*=\s*e\.eid|e\.eid\s*=\s*(s|sal)\.eid",
          r"e\.dept_id\s*=\s*d\.dept_id|d\.dept_id\s*=\s*e\.dept_id"])]]
    sql = []
    for i in range(60):
        t1, t2, t3, q, checks = schemas[i % 3]
        sql.append({"prompt": f"-- Tables:\n-- {t1}\n-- {t2}\n-- {t3}\n"
                              f"-- Task: {q}\nSELECT", "checks": checks})
    tasks["sql"] = sql
    notes["sql"] = "n=60 over the same 3 schemas (diversity limit)"
    return tasks, notes


def build_subspaces(lens, hf):
    """seed-1 pools; matched ranks reused from v2 energy_match plan."""
    em = read_json(RUN_DIR_V2 / "metrics" / "energy_match.json")
    plan = {int(l): v for l, v in em["plan"].items()}
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    span20, vrand, vnonj, jd, rd = {}, {}, {}, {}, {}
    for l in BAND:
        st = torch.load(RUN_DIR / "metrics" / "layer_state" / f"layer_{l}.pt",
                        weights_only=True)
        J = lens.jacobians[l].to(W_U.device)
        ids = st["top_dir_ids"]
        D20 = torch.nn.functional.normalize(
            (W_U[ids[:20]] * g[None, :]) @ J, dim=1)
        span20[l], _ = torch.linalg.qr(D20.T)
        S = torch.nn.functional.normalize(
            (W_U[ids[:256]] * g[None, :]) @ J, dim=1)
        Qs, _ = torch.linalg.qr(S.T)
        P = st["pca_evecs"].T.to(W_U.device)
        P_perp = P - (P @ Qs) @ Qs.T
        keep = P_perp.norm(dim=1) > 0.5
        Qn, _ = torch.linalg.qr(torch.nn.functional.normalize(
            P_perp[keep], dim=1).T)
        vnonj[l] = Qn[:, plan[l]["vmatch_nonJ_k20"]].contiguous()
        rng = np.random.default_rng(5000 + l)
        R = torch.tensor(rng.standard_normal((512, P.shape[1])),
                         dtype=torch.float32, device=W_U.device)
        Qr, _ = torch.linalg.qr(R.T)
        vrand[l] = Qr[:, plan[l]["vmatch_rand_k20"]].contiguous()
        jd[l] = torch.nn.functional.normalize(
            (W_U * g[None, :]) @ J.cuda(), dim=1).half()
        rng2 = np.random.default_rng(6000 + l)
        R2 = torch.tensor(rng2.standard_normal((5120, P.shape[1])),
                          device="cuda", dtype=torch.float32)
        rd[l] = torch.nn.functional.normalize(R2, dim=1).half()
    del W_U
    return {"jspace_k20": span20, "vmatch_rand_k20": vrand,
            "vmatch_nonJ_k20": vnonj}, jd, rd


def main() -> None:
    seed_all(SEED)
    (RUN_DIR_V2 / "metrics").mkdir(parents=True, exist_ok=True)
    lens = JacobianLens.load(str(RUN_DIR / "lens" / "olmo32bthink_lens.pt"))
    model, hf, tok = load_model("main")
    subs, jd, rd = build_subspaces(lens, hf)
    rng = np.random.default_rng(SEED)
    tasks, notes = build_tasks_v2(rng)
    tasks["twohop"] = tasks["twohop"][:30]
    notes["twohop"] += "; trimmed to n=30 (PLAN_v3 P6)"
    res = read_json(OUT) if OUT.exists() else {
        "seed": SEED, "notes": notes, "conditions": {}}
    ab = s7.Ablator(model.layers)
    with ab:
        for cond in CONDS:
            res["conditions"].setdefault(cond, {})
            for tname in TASKS:
                if tname in res["conditions"][cond] and "--force" not in sys.argv:
                    continue
                t0 = time.time()
                extra = {}
                if cond in ("frozen_j10", "frozen_rand10"):
                    dicts = jd if cond == "frozen_j10" else rd
                    key = {"twohop_lp": "twohop",
                           "arithmetic_v2": "arithmetic"}.get(tname, tname)
                    scores = []
                    for it in tasks[key]:
                        ab.mode = None
                        p = (it.get("prompt") or it.get("text")
                             or it.get("good"))
                        ab.mode = ("static",
                                   frozen_projectors(model, dicts, p))
                        one = {key: [it]}
                        scores += s7.run_task(tname, cond, model, hf, tok,
                                              one)
                        ab.mode = None
                else:
                    ab.mode = None if cond == "none" else \
                        ("static", subs[cond])
                    scores = s7.run_task(tname, cond, model, hf, tok,
                                         tasks, extra)
                    ab.mode = None
                entry = s7.boot_ci(scores)
                entry["seconds"] = round(time.time() - t0)
                res["conditions"][cond][tname] = entry
                atomic_write_json(res, OUT)
                log(f"{cond:>16} {tname:>13}: {entry['mean']:.3f} "
                    f"[{entry['ci_lo']:.3f},{entry['ci_hi']:.3f}] "
                    f"({entry['seconds']}s)")
    # seed0 comparison block
    cmp = {}
    for src, name in ((RUN_DIR_V2 / "metrics" / "ablation_v2.json", "s11"),
                      (RUN_DIR_V2 / "metrics" / "frozen_ablation.json",
                       "s12")):
        if src.exists():
            s0 = read_json(src)["conditions"]
            for c in CONDS:
                if c in s0:
                    cmp[c] = {t: {"seed0": s0[c][t]["mean"],
                                  "seed1": res["conditions"][c][t]["mean"]}
                              for t in TASKS if t in s0[c]
                              and t in res["conditions"].get(c, {})}
    res["seed_comparison"] = cmp
    atomic_write_json(res, OUT)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
