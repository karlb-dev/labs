# Core Battery energy-matched static grid for a matrix model — the paper's
# causal claim under clean instruments:
#   none / jspace_k{10,20,40} / vmatch_rand_k{...} / vmatch_nonJ_k{...}
# Controls are matched on REMOVED RAW-h ENERGY per layer/dose (part-1 P1
# machinery: prefix or windowed rank selection into 512-dim random and
# ≤64-dim non-J PC pools), measured over the shared v1 descriptive prompt
# set. Task list per preregistration for statics: twohop, twohop_lp,
# onehop, prose_nll (all prefill-only — fast). Resumable per (cond, task).
#
# Prerequisite: p2core_descriptive.py has produced
#   metrics/<slug>/layer_state/layer_<l>.pt  (top_dir_ids + pca_evecs).
#
# Usage: python scripts/p2core_energy_match.py --model olmo31-instruct \
#          --lens <path/to/lens.pt> [--force]
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import (atomic_write_json, die, log, p2_load_model,
                        p2_metrics_dir, read_json, seed_all)

import numpy as np
import torch
from jlens import JacobianLens

PART1_SCRIPTS = Path(__file__).resolve().parents[1] / "part1" / "scripts"
sys.path.insert(0, str(PART1_SCRIPTS))
import s7_ablation as s7  # noqa: E402
import s11_energy_match as s11  # noqa: E402  (energy_pass, match — shared prompt set)

BAND, DOSES, RAND_POOL = s7.BAND, s7.DOSES, s11.RAND_POOL
TASKS = ["twohop", "twohop_lp", "onehop", "prose_nll"]


def arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def build_bases(lens, hf, state_dir: Path):
    """s11.build_bases with the layer-state directory parameterized (the
    part-1 original reads v1's). Same construction, seeds, pool sizes."""
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    out = {}
    for l in BAND:
        st = torch.load(state_dir / f"layer_{l}.pt", weights_only=True)
        J = lens.jacobians[l].to(W_U.device)
        ids = st["top_dir_ids"]
        D_top = torch.nn.functional.normalize(
            (W_U[ids[:max(DOSES)]] * g[None, :]) @ J, dim=1)
        Qj, _ = torch.linalg.qr(D_top.T)
        S = torch.nn.functional.normalize(
            (W_U[ids[:256]] * g[None, :]) @ J, dim=1)
        Qs, _ = torch.linalg.qr(S.T)
        P = st["pca_evecs"].T.to(W_U.device)
        P_perp = P - (P @ Qs) @ Qs.T
        keep = P_perp.norm(dim=1) > 0.5
        P_ok = torch.nn.functional.normalize(P_perp[keep], dim=1)
        Qn, _ = torch.linalg.qr(P_ok.T)
        rng = np.random.default_rng(1000 + l)
        R = torch.tensor(rng.standard_normal((RAND_POOL, P.shape[1])),
                         dtype=torch.float32, device=W_U.device)
        Qr, _ = torch.linalg.qr(R.T)
        out[l] = {"J": Qj.contiguous(), "nonJ": Qn.contiguous(),
                  "rand": Qr.contiguous()}
    del W_U
    return out


def main() -> None:
    seed_all()
    slug = arg("--model", "olmo31-instruct")
    lens_path = Path(arg("--lens", ""))
    if not lens_path.exists():
        die(f"--lens required and must exist (got {lens_path!r})")
    mdir = p2_metrics_dir(slug)
    state_dir = mdir / "layer_state"
    if not (state_dir / f"layer_{BAND[0]}.pt").exists():
        die(f"layer state missing in {state_dir}; run p2core_descriptive first")
    em_out = mdir / "energy_match.json"
    ab_out = mdir / "core_static_grid.json"

    lens = JacobianLens.load(str(lens_path))
    model, hf, tok = p2_load_model(slug)
    bases = build_bases(lens, hf, state_dir)
    log(f"bases built for {len(BAND)} layers")

    if em_out.exists() and "--force" not in sys.argv:
        saved = read_json(em_out)
        em = {int(l): v for l, v in saved["per_layer"].items()}
        plan = {int(l): v for l, v in saved["plan"].items()}
        log("energy match loaded from existing file")
    else:
        acc = s11.energy_pass(model, bases)     # shared v1 descriptive prompts
        em, plan = s11.match(acc)
        atomic_write_json(
            {"model_slug": slug, "band": BAND, "doses": list(DOSES),
             "rand_pool": RAND_POOL,
             "corpus": "v1 descriptive_prompts.jsonl (raw h, all positions)",
             "per_layer": em, "plan": plan}, em_out)
        bad = [(l, k, d) for l in BAND for k, d in em[l]["doses"].items()
               for pool in ("nonJ", "rand")
               if not 0.8 <= d[f"{pool}_ratio"] <= 1.25]
        if bad:
            log(f"WARNING: {len(bad)} (layer,dose,pool) cells outside "
                f"[0.8,1.25] energy ratio — recorded, review before claims")
        log(f"wrote {em_out}")

    subs = {}
    for k in DOSES:
        subs[f"jspace_k{k}"] = {l: bases[l]["J"][:, :k].contiguous()
                                for l in BAND}
        for pool in ("nonJ", "rand"):
            cond = f"vmatch_{pool}_k{k}"
            subs[cond] = {l: bases[l][pool][:, plan[l][cond]].contiguous()
                          for l in BAND}

    rng = np.random.default_rng(0)
    tasks = s7.build_tasks(rng)
    res = read_json(ab_out) if ab_out.exists() else {
        "model_slug": slug, "lens": lens_path.name, "band": BAND,
        "doses": list(DOSES), "conditions": {}}
    conds = ["none"] + list(subs)
    ab = s7.Ablator(model.layers)
    with ab:
        for cond in conds:
            res["conditions"].setdefault(cond, {})
            for tname in TASKS:
                if tname in res["conditions"][cond] and "--force" not in sys.argv:
                    continue
                t0 = time.time()
                ab.mode = None if cond == "none" else ("static", subs[cond])
                scores = s7.run_task(tname, cond, model, hf, tok, tasks)
                ab.mode = None
                entry = s7.boot_ci(scores)
                entry["seconds"] = round(time.time() - t0)
                res["conditions"][cond][tname] = entry
                atomic_write_json(res, ab_out)
                log(f"{cond:>17} {tname:>9}: {entry['mean']:.3f} "
                    f"[{entry['ci_lo']:.3f},{entry['ci_hi']:.3f}] "
                    f"({entry['seconds']}s)")
    c = res["conditions"]
    res["summary"] = {
        cond: {"twohop_lp_delta": round(c[cond]["twohop_lp"]["mean"]
                                        - c["none"]["twohop_lp"]["mean"], 3)}
        for cond in conds[1:] if "twohop_lp" in c.get(cond, {})
        and "twohop_lp" in c.get("none", {})}
    atomic_write_json(res, ab_out)
    log(f"wrote {ab_out}; summary {res['summary']}")


if __name__ == "__main__":
    main()
