# C3 — build the hard one-hop set: 60 single-hop items difficulty-matched to
# the two-hop battery by baseline answer logprob on the anchor model
# (Olmo-3-32B-Think), then FROZEN for the whole matrix (one item set for
# every model, matching model recorded). Kills the ceiling confound behind
# every 1-hop/2-hop asymmetry claim (Qwen's spared one-hop, A1/A3 patterns).
#
# Phase "score": one prefill per item — answer logprob at the final prompt
#   position, max over case/space variant first tokens (the battery's
#   convention) — for (a) the 113 curated hard-fact candidates,
#   (b) the 60 two-hop battery items (probe-swap [0:60], the target
#   distribution). → metrics/olmo3-think/c3_scores.json
# Phase "match": quantile-match candidates to the two-hop lp distribution;
#   exclusions: lp > -0.5 (ceiling) or lp < -9 (model doesn't know the
#   fact). → config/prompts/hard_onehop.jsonl (60 items, frozen) +
#   metrics/olmo3-think/c3_match.json (summary + audit trail).
#
# Usage: python scripts/p2c3_hard_onehop.py [--model olmo3-think] [--force]
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import (LAB_DIR_P2, RUN_DIR_P2, atomic_write_json, die, log,
                        p2_load_model, p2_metrics_dir, read_json, seed_all,
                        variant_first_ids)

import torch

CANDIDATES = LAB_DIR_P2 / "config_c3_candidates.jsonl"
PROBE_SWAP = Path("/content/jacobian-lens/data/experiments/probe-swap.json")
FROZEN_OUT = RUN_DIR_P2 / "config" / "prompts" / "hard_onehop.jsonl"
CEILING_LP, UNKNOWN_LP = -0.5, -9.0


def arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


@torch.no_grad()
def answer_lp(model, hf, tok, prompt: str, answer: str) -> float:
    ids = model.encode(prompt.rstrip(), max_length=512)
    logits = hf(input_ids=ids, use_cache=False).logits[0, -1].float()
    lsm = torch.log_softmax(logits, dim=-1)
    return max(lsm[i].item() for i in variant_first_ids(tok, answer))


def score_phase(slug: str, scores_path: Path) -> dict:
    cands = [json.loads(l) for l in CANDIDATES.read_text().splitlines() if l.strip()]
    swap = json.load(open(PROBE_SWAP))["items"][:60]
    model, hf, tok = p2_load_model(slug)
    out = {"model_slug": slug, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "candidates": [], "twohop": []}
    for i, it in enumerate(cands):
        lp = answer_lp(model, hf, tok, it["prompt"], it["answer"])
        out["candidates"].append(it | {"lp": round(lp, 4)})
        if (i + 1) % 20 == 0:
            log(f"  candidates {i+1}/{len(cands)}")
    for it in swap:
        lp = answer_lp(model, hf, tok, it["prompt"], it["answer"])
        out["twohop"].append({"name": it["name"], "answer": it["answer"],
                              "lp": round(lp, 4)})
    atomic_write_json(out, scores_path)
    log(f"wrote {scores_path}")
    return out


def match_phase(scores: dict, slug: str) -> None:
    cands = [c for c in scores["candidates"]
             if UNKNOWN_LP <= c["lp"] <= CEILING_LP]
    excluded = [c for c in scores["candidates"] if c not in cands]
    targets = sorted((t["lp"] for t in scores["twohop"]))
    chosen, pool = [], sorted(cands, key=lambda c: c["lp"])
    interim = len(cands) < 60
    if interim:
        # Honest fallback (2026-07-28): the curated pool is mostly ceiling
        # (68/113 items at lp > -0.5 on Think — a 32B knows "obscure" facts).
        # Freeze a DEV-TIER interim set: candidate-driven pairing to nearest
        # unused targets, coverage gaps recorded. Confirmatory C3 = Stage-3
        # pool expansion; do NOT dilute hardness by raising the ceiling.
        global FROZEN_OUT
        FROZEN_OUT = FROZEN_OUT.with_name("hard_onehop_dev.jsonl")
        log(f"INTERIM MODE: n={len(cands)} dev-tier set -> {FROZEN_OUT.name}")
        tpool = list(targets)
        for c in pool:
            t = min(tpool, key=lambda x: abs(x - c["lp"]))
            tpool.remove(t)
            chosen.append(c | {"matched_twohop_lp": round(t, 4),
                               "match_gap": round(abs(c["lp"] - t), 4)})
        targets = [c["matched_twohop_lp"] for c in chosen]
    if not interim:
        for tlp in targets:
            best = min(pool, key=lambda c: abs(c["lp"] - tlp))
            pool.remove(best)
            chosen.append(best | {"matched_twohop_lp": round(tlp, 4),
                                  "match_gap": round(abs(best["lp"] - tlp), 4)})
    FROZEN_OUT.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_OUT.write_text("\n".join(json.dumps(c) for c in chosen) + "\n")
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    summ = {
        "model_slug": slug, "n": len(chosen),
        "onehop_lp_mean": round(mean([c["lp"] for c in chosen]), 3),
        "twohop_lp_mean": round(mean(targets), 3),
        "interim_dev_tier": interim,
        "onehop_lp_median": round(sorted(c["lp"] for c in chosen)[len(chosen)//2], 3),
        "twohop_lp_median": round(sorted(targets)[len(targets)//2], 3),
        "worst_match_gap": max(c["match_gap"] for c in chosen),
        "n_excluded_ceiling": sum(1 for c in excluded if c["lp"] > CEILING_LP),
        "n_excluded_unknown": sum(1 for c in excluded if c["lp"] < UNKNOWN_LP),
        "excluded": [{"answer": c["answer"], "lp": c["lp"]} for c in excluded],
        "frozen_file": str(FROZEN_OUT),
    }
    atomic_write_json(summ, p2_metrics_dir(slug) / "c3_match.json")
    log(f"FROZEN {len(chosen)} hard one-hop items -> {FROZEN_OUT}")
    log(f"lp mean one-hop {summ['onehop_lp_mean']} vs two-hop "
        f"{summ['twohop_lp_mean']}; worst gap {summ['worst_match_gap']}")


def main() -> None:
    seed_all()
    slug = arg("--model", "olmo3-think")
    scores_path = p2_metrics_dir(slug) / "c3_scores.json"
    if FROZEN_OUT.exists() and "--force" not in sys.argv:
        log(f"{FROZEN_OUT} exists; set is frozen — nothing to do")
        return
    if scores_path.exists() and "--force" not in sys.argv:
        scores = read_json(scores_path)
        log("scores exist; matching only")
    else:
        scores = score_phase(slug, scores_path)
    match_phase(scores, slug)


if __name__ == "__main__":
    main()
