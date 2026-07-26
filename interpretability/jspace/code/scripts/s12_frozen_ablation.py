# v2 Priority 2: frozen prompt-selected top-k ablation — the confound-free
# variant of the paper's (confirmed live per-token) intervention.
#
# Per item: one clean forward over the prompt; per band layer, rank the full
# J-dictionary by summed |corr| over prompt positions (first 4 skipped);
# FREEZE the top-10, orthonormalize, and generate with that per-item static
# projector. Dynamic selection, static application: can see position-
# specific workspace structure without deleting live computation.
#
# Conditions (metrics/frozen_ablation.json):
#   frozen_j10    per-item frozen top-10 J-directions
#   frozen_rand10 same mechanism on a 5120-row random dictionary (seed
#                 2000+layer) — matched selection+application control
#   live_rand10   per-token top-10 of the random dictionary via the v1 dyn
#                 mechanism — the matched-live control v1's dyn10 lacked
#   live_j10      = v1 jspace_dyn10, copied for side-by-side
#   none          = v1 baseline, copied
# Same battery, seed, bootstrap as s7/s11. Resumable per (condition, task).
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
from jlens.hooks import ActivationRecorder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7_ablation as s7  # noqa: E402

BAND = s7.BAND
OUT = RUN_DIR_V2 / "metrics" / "frozen_ablation.json"
SKIP = 4
K = 10


def build_dicts(lens, hf):
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    jd, rd = {}, {}
    for l in BAND:
        jd[l] = torch.nn.functional.normalize(
            (W_U * g[None, :]) @ lens.jacobians[l].cuda(), dim=1).half()
        rng = np.random.default_rng(2000 + l)
        R = torch.tensor(rng.standard_normal((5120, W_U.shape[1])),
                         device="cuda", dtype=torch.float32)
        rd[l] = torch.nn.functional.normalize(R, dim=1).half()
    del W_U
    return jd, rd


@torch.no_grad()
def frozen_projectors(model, dicts, prompt: str) -> dict:
    """One clean pass; per band layer, QR of the top-K dictionary rows by
    summed |corr| over prompt positions (skipping the first SKIP)."""
    ids = model.encode(prompt, max_length=512)
    with ActivationRecorder(model.layers, at=BAND) as rec:
        model.forward(ids)
    out = {}
    for l in BAND:
        h = rec.activations[l][0].float()          # [T, d]
        h = h[min(SKIP, h.shape[0] - 1):]
        score = (h.half() @ dicts[l].T).abs().sum(0)   # [V]
        top = score.topk(K).indices
        Q, _ = torch.linalg.qr(dicts[l][top].float().T)
        out[l] = Q.contiguous()
    return out


def item_prompt(tname: str, it: dict) -> str:
    if tname == "prose_nll":
        return it["text"]
    if tname == "grammar":
        return it["good"]
    return it["prompt"]


def run_frozen(cond, dicts, ab, model, hf, tok, tasks, tname, extra):
    """Per-item selection then per-item scoring via s7.run_task on
    singleton task dicts (scorer identical to v1)."""
    src = "arithmetic" if tname in ("samples",) else tname
    base_items = tasks["arithmetic"][:3] + tasks["sql"][:2] \
        if tname == "samples" else tasks[src if src in tasks else tname]
    if tname == "samples":
        # selection from the first arithmetic prompt only; samples is an
        # audit artifact, not a statistic
        ab.mode = ("static", frozen_projectors(model, dicts,
                                               tasks["arithmetic"][0]["prompt"]))
        sc = s7.run_task("samples", cond, model, hf, tok, tasks, extra)
        ab.mode = None
        return sc
    key = "twohop" if tname == "twohop_lp" else tname
    key = "arithmetic" if tname == "arithmetic_v2" else key
    scores = []
    for it in tasks[key]:
        ab.mode = None
        proj = frozen_projectors(model, dicts, item_prompt(key, it))
        ab.mode = ("static", proj)
        one = {key: [it]} if tname not in ("twohop_lp", "arithmetic_v2") \
            else {"twohop": [it]} if tname == "twohop_lp" \
            else {"arithmetic": [it]}
        scores += s7.run_task(tname, cond, model, hf, tok, one)
        ab.mode = None
    return scores


def main() -> None:
    seed_all()
    (RUN_DIR_V2 / "metrics").mkdir(parents=True, exist_ok=True)
    lens = JacobianLens.load(str(RUN_DIR / "lens" / "olmo32bthink_lens.pt"))
    model, hf, tok = load_model("main")
    jd, rd = build_dicts(lens, hf)
    log("dictionaries built (J + random) for band")

    rng = np.random.default_rng(0)
    tasks = s7.build_tasks(rng)
    v1 = read_json(RUN_DIR / "metrics" / "ablation_results.json")
    res = read_json(OUT) if OUT.exists() else {
        "band": BAND, "k": K, "skip": SKIP, "conditions": {},
        "reused_from_v1": {"none": "none", "live_j10": "jspace_dyn10"}}
    for new, old in res["reused_from_v1"].items():
        res["conditions"][new] = v1["conditions"][old]

    task_names = ["twohop", "onehop", "arithmetic", "sql", "prose_nll",
                  "grammar", "twohop_lp", "samples", "arithmetic_v2"]
    ab = s7.Ablator(model.layers)
    with ab:
        for cond in ("frozen_j10", "frozen_rand10", "live_rand10"):
            res["conditions"].setdefault(cond, {})
            for tname in task_names:
                if tname in res["conditions"][cond] and "--force" not in sys.argv:
                    continue
                t0 = time.time()
                extra = {}
                if cond == "live_rand10":
                    ab.mode = ("dyn", rd, K)
                    scores = s7.run_task(tname, cond, model, hf, tok,
                                         tasks, extra)
                    ab.mode = None
                else:
                    dicts = jd if cond == "frozen_j10" else rd
                    scores = run_frozen(cond, dicts, ab, model, hf, tok,
                                        tasks, tname, extra)
                entry = s7.boot_ci(scores)
                entry["seconds"] = round(time.time() - t0)
                entry.update(extra)
                res["conditions"][cond][tname] = entry
                atomic_write_json(res, OUT)
                log(f"{cond:>14} {tname:>13}: {entry['mean']:.3f} "
                    f"[{entry['ci_lo']:.3f},{entry['ci_hi']:.3f}] "
                    f"({entry['seconds']}s)")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
