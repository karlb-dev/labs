# B3 — frozen-logit control on Olmo-3-32B-Think: the identical frozen
# per-item mechanism as part-1's marquee instrument (s12), but the dictionary
# is plain (W_U ⊙ g) rows — no Jacobian pullback. Preregistered reading:
#   frozen_logit10 ≈ frozen_j10 on fact deletion  → the pullback is not doing
#     causal work on this model; the method claim narrows honestly.
#   frozen_logit10 markedly weaker                → the J-space framing earns
#     its name causally, not just as readout.
# Pool sizes are inherently matched (both dictionaries are vocab-sized), so
# the s24 pool-size caveat does not apply to this comparison.
#
# Mechanism, band, K, SKIP, battery, scorers, bootstrap: imported unchanged
# from part-1 (s7/s12). Comparison cells (none / frozen_j10 / frozen_rand10)
# are copied from v2 frozen_ablation.json — same items, same code path.
# Output: part2 metrics/olmo3-think/b3_frozen_logit.json. Resumable per task.
#
# Usage: python scripts/p2b3_frozen_logit.py [--force]
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import (RUN_DIR_V2, atomic_write_json, log, p2_load_model,
                        p2_metrics_dir, read_json, seed_all)

import numpy as np
import torch

PART1_SCRIPTS = Path(__file__).resolve().parents[1] / "part1" / "scripts"
sys.path.insert(0, str(PART1_SCRIPTS))
import s7_ablation as s7  # noqa: E402
import s12_frozen_ablation as s12  # noqa: E402

OUT = p2_metrics_dir("olmo3-think") / "b3_frozen_logit.json"
COND = "frozen_logit10"
TASKS = ["twohop", "onehop", "arithmetic", "sql", "prose_nll",
         "grammar", "twohop_lp", "samples", "arithmetic_v2"]


def build_logit_dict(hf):
    """Vocab-sized (W_U ⊙ g) dictionary; one shared tensor for every band
    layer (rows are layer-independent without the J pullback — selection
    still differs per layer through h_l)."""
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    D = torch.nn.functional.normalize((W_U * g[None, :]), dim=1).half().cuda()
    del W_U
    return {l: D for l in s7.BAND}


def main() -> None:
    seed_all()
    res = read_json(OUT) if OUT.exists() else {
        "model_slug": "olmo3-think", "band": s7.BAND, "k": s12.K,
        "skip": s12.SKIP, "conditions": {},
        "evidence_tier": "exploratory-pilot",
        "tier_note": "part-1 mechanics (raw QR, first-token scoring, "
                     "both-phase hooks) — internally consistent J-vs-logit "
                     "comparison; confirmatory version = R3 control family",
        "reused_from_v2": ["none", "frozen_j10", "frozen_rand10"]}
    v2 = read_json(RUN_DIR_V2 / "metrics" / "frozen_ablation.json")
    for c in res["reused_from_v2"]:
        res["conditions"][c] = v2["conditions"][c]
    done = res["conditions"].get(COND, {})
    if all(t in done for t in TASKS) and "--force" not in sys.argv:
        log(f"{OUT} complete; skipping")
        return

    model, hf, tok = p2_load_model("olmo3-think")
    ld = build_logit_dict(hf)
    log("logit dictionary built (vocab-sized, shared across band)")
    rng = np.random.default_rng(0)
    tasks = s7.build_tasks(rng)

    ab = s7.Ablator(model.layers)
    with ab:
        res["conditions"].setdefault(COND, {})
        for tname in TASKS:
            if tname in res["conditions"][COND] and "--force" not in sys.argv:
                continue
            t0 = time.time()
            extra = {}
            scores = s12.run_frozen(COND, ld, ab, model, hf, tok, tasks,
                                    tname, extra)
            entry = s7.boot_ci(scores)
            entry["seconds"] = round(time.time() - t0)
            entry.update(extra)
            res["conditions"][COND][tname] = entry
            atomic_write_json(res, OUT)
            log(f"{COND:>15} {tname:>13}: {entry['mean']:.3f} "
                f"[{entry['ci_lo']:.3f},{entry['ci_hi']:.3f}] "
                f"({entry['seconds']}s)")

    # summary block: deltas vs baseline for the two vocab-sized dictionaries
    c = res["conditions"]
    summ = {}
    for cond in (COND, "frozen_j10", "frozen_rand10"):
        if "twohop_lp" in c.get(cond, {}) and "twohop_lp" in c.get("none", {}):
            summ[cond] = {
                "twohop_lp_delta": round(c[cond]["twohop_lp"]["mean"]
                                         - c["none"]["twohop_lp"]["mean"], 3),
                "twohop_acc": c[cond].get("twohop", {}).get("mean"),
                "onehop_acc": c[cond].get("onehop", {}).get("mean"),
                "prose_nll": c[cond].get("prose_nll", {}).get("mean"),
            }
    res["summary"] = summ
    atomic_write_json(res, OUT)
    log(f"wrote {OUT}")
    log(f"summary: {summ}")


if __name__ == "__main__":
    main()
