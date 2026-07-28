# R7 tail mechanization — what does the protected ablation actually
# DELETE on the items that lose >1 nat despite protection?
#
# The pilot established the tail behaviorally (protected median ~0, but a
# hard-item cohort loses >1 nat; reproduces across independent lenses,
# corr 0.989). The competing readings:
#   (H-content)  the deflated directions carry the two-hop BRIDGE entity
#                — deleting internal content the model still needs;
#   (H-output)   they are answer-adjacent output directions that the
#                clean-top-10 protection failed to cover (e.g. the answer
#                sits at clean rank > 10, so it was never protected);
#   (H-nothing)  selection is item-generic — the tail is unrelated to
#                what got deleted.
# This script decides between them from the selected-id logs
# (`r7-selected-ids-*`) joined to the per-item protected deltas
# (`r7_per_item.parquet`) and the clean ranks (`r7_cleanrank.json`).
#
# Endpoints (all per-item, family-clustered where tested):
#   bridge_hit    : any bridge-entity token id among the selected rows at
#                   the prompt-final positions (two-hop items only)
#   answer_hit    : any answer-token id among the selected rows
#   protect_miss  : answer's clean rank > protect_top_k (unprotectable)
#   selection_overlap: Jaccard of selected id sets, tail vs non-tail items
# Contrast: tail (delta < -1 nat) vs non-tail, protected arm; the
# unprotected arm is the reference for what protection removed.
#
# Tier: pilot (descriptive mechanism). CPU-only.
# Usage: python -m jspace_part2.experiments.r7_tail_mechanism \
#          [--model olmo3-think] [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          write_result)

RUN_DIR_P2 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "part2_20260727")
TAIL_THR = -1.0
PROMPT_FINAL_WINDOW = 2      # positions before the scoring position


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def permutation_bridge_null(rows, n_perm=2000, seed=4242):
    """THE control the raw bridge-hit rate needs: the selected set is
    large (hundreds of distinct ids per item), so a high hit rate may be
    chance. Reassign each item's bridge-token set to a DIFFERENT item and
    recompute — the gap between observed and permuted is the item-specific
    signal. Returns (observed, permuted_mean, permuted_p95, p_value)."""
    rng = np.random.default_rng(seed)
    pairs = [(set(r["sel_ids_pf"]), set(r["_bridge_ids"])) for r in rows
             if r["_bridge_ids"]]
    if len(pairs) < 3:
        return None
    obs = float(np.mean([bool(s & b) for s, b in pairs]))
    sels = [s for s, _ in pairs]
    brs = [b for _, b in pairs]
    perm = np.empty(n_perm)
    for i in range(n_perm):
        order = rng.permutation(len(brs))
        # derangement-ish: retry indices that map to themselves
        order = np.array([o if o != j else (o + 1) % len(brs)
                          for j, o in enumerate(order)])
        perm[i] = np.mean([bool(sels[j] & brs[order[j]])
                           for j in range(len(sels))])
    return {"observed": round(obs, 4),
            "permuted_mean": round(float(perm.mean()), 4),
            "permuted_p95": round(float(np.percentile(perm, 95)), 4),
            "p_value": round(float((perm >= obs).mean()), 4),
            "n_items": len(pairs)}


def cluster_boot_ci(values, families, n_boot=4000, seed=4242):
    """Family-clustered bootstrap CI of a mean."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"v": values, "f": families})
    groups = [g["v"].to_numpy() for _, g in df.groupby("f")]
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        means[b] = np.concatenate([groups[i] for i in idx]).mean()
    return [round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--model", "olmo3-think")
    sel_dir = RUN_DIR_P2 / "metrics" / slug / "r7_selected_ids"
    out = sel_dir / "r7_tail_mechanism.json"
    t0 = time.time()

    sel = pd.read_parquet(sel_dir / "r7sel_selected.parquet")
    items = pd.read_parquet(sel_dir / "r7sel_items.parquet")
    per_item = pd.read_parquet(RUN_DIR_P2 / "metrics" / slug /
                               "r7_pilot" / "r7_per_item.parquet")

    # per-item protected delta (the tail definition), from the pilot grid
    base = per_item[per_item.condition == "none"].set_index("item_id")["score"]
    prot = per_item[per_item.condition == "dynJ_protected"]\
        .set_index("item_id")["score"]
    delta = (prot - base).dropna()

    ranks = {}
    rank_path = RUN_DIR_P2 / "metrics" / slug / "r7_pilot" / "r7_cleanrank.json"
    if rank_path.exists():
        ranks = {r["item_id"]: r["clean_rank"]
                 for r in json.loads(rank_path.read_text())["rows"]}

    meta_p = items[items.condition == "dynJ_protected"]
    rows = []
    for _, m in meta_p.iterrows():
        d = delta.get(m.item_id, np.nan)
        s_all = sel[(sel.condition == "dynJ_protected") &
                    (sel.item_id == m.item_key)]
        s_pf = s_all[(s_all.pos >= m.n_prompt - PROMPT_FINAL_WINDOW) &
                     (s_all.pos < m.n_prompt)]
        sel_ids_pf = set(s_pf.token_id.tolist())
        bridge = set(m.bridge_token_ids or [])
        rows.append({
            "item_id": m.item_id, "task": m.task, "family": m.family,
            "delta": float(d) if d == d else None,
            "is_tail": bool(d < TAIL_THR) if d == d else None,
            "n_selected_pf": int(len(sel_ids_pf)),
            "bridge_hit": (bool(bridge & sel_ids_pf) if bridge else None),
            "clean_rank": ranks.get(m.item_id),
            "protect_miss": (ranks.get(m.item_id, 0) > 10
                             if m.item_id in ranks else None),
            "sel_ids_pf": sorted(sel_ids_pf),
            "_bridge_ids": sorted(bridge)})

    df = pd.DataFrame(rows)
    scored = df[df.delta.notna()]
    tail = scored[scored.is_tail == True]           # noqa: E712
    nontail = scored[scored.is_tail == False]       # noqa: E712

    def rate(sub, col):
        v = sub[col].dropna()
        return round(float(v.mean()), 4) if len(v) else None

    two = scored[scored.task == "twohop"]
    two_t, two_n = two[two.is_tail == True], two[two.is_tail == False]  # noqa

    # selection overlap: how item-specific is the selected set?
    def mean_pairwise_jacc(sub, cap=40):
        sets = [set(s) for s in sub.sel_ids_pf.tolist()[:cap]]
        vals = [jaccard(sets[i], sets[j])
                for i in range(len(sets)) for j in range(i + 1, len(sets))]
        return round(float(np.mean(vals)), 4) if vals else None

    bridge_ci = None
    if len(two_t.bridge_hit.dropna()) >= 3:
        sub = two_t.dropna(subset=["bridge_hit"])
        bridge_ci = cluster_boot_ci(sub.bridge_hit.astype(float).to_numpy(),
                                    sub.family.to_numpy())

    two_rows = [r for r in rows if r["task"] == "twohop"]
    summ = {
        "model": slug, "tail_threshold_nats": TAIL_THR,
        "n_items_scored": int(len(scored)),
        "n_tail": int(len(tail)), "n_nontail": int(len(nontail)),
        "bridge_hit_rate_twohop": {
            "tail": rate(two_t, "bridge_hit"),
            "nontail": rate(two_n, "bridge_hit"),
            "tail_ci95_clustered": bridge_ci},
        "bridge_permutation_control": {
            "all_twohop": permutation_bridge_null(two_rows),
            "tail_only": permutation_bridge_null(
                [r for r in two_rows if r["is_tail"]]),
            "note": ("observed vs bridge-sets-reassigned-to-other-items; a "
                     "gap is item-specific bridge selection, no gap means "
                     "the selected set is simply large")},
        "protect_miss_rate": {"tail": rate(tail, "protect_miss"),
                              "nontail": rate(nontail, "protect_miss")},
        "clean_rank_median": {
            "tail": (float(np.median(tail.clean_rank.dropna()))
                     if tail.clean_rank.notna().any() else None),
            "nontail": (float(np.median(nontail.clean_rank.dropna()))
                        if nontail.clean_rank.notna().any() else None)},
        "mean_pairwise_selection_jaccard": {
            "tail": mean_pairwise_jacc(tail),
            "nontail": mean_pairwise_jacc(nontail)},
        "n_selected_promptfinal_mean": {
            "tail": rate(tail, "n_selected_pf"),
            "nontail": rate(nontail, "n_selected_pf")},
        "reading": None, "seconds": round(time.time() - t0)}

    bh_t = summ["bridge_hit_rate_twohop"]["tail"]
    bh_n = summ["bridge_hit_rate_twohop"]["nontail"]
    pm_t = summ["protect_miss_rate"]["tail"]
    pm_n = summ["protect_miss_rate"]["nontail"]
    parts = []
    if bh_t is not None and bh_n is not None:
        parts.append(f"bridge-token selection tail {bh_t:.2f} vs non-tail "
                     f"{bh_n:.2f}")
    if pm_t is not None and pm_n is not None:
        parts.append(f"protection-miss (clean rank>10) tail {pm_t:.2f} vs "
                     f"non-tail {pm_n:.2f}")
    summ["reading"] = ("; ".join(parts) +
                       " — H-content favored if the bridge gap is large and "
                       "the protect-miss gap is not; H-output favored if the "
                       "reverse; H-nothing if neither separates.")

    prov = Provenance(
        evidence_id=f"r7-tail-mechanism-{slug}-v1", tier="pilot",
        command=("python -m jspace_part2.experiments.r7_tail_mechanism "
                 f"--model {slug}"),
        inputs={"selected": sha256_file(sel_dir / "r7sel_selected.parquet"),
                "items": sha256_file(sel_dir / "r7sel_items.parquet"),
                "per_item": sha256_file(RUN_DIR_P2 / "metrics" / slug /
                                        "r7_pilot" / "r7_per_item.parquet")},
        seed=4242)
    write_result({"summary": summ,
                  "rows": df.drop(columns=["sel_ids_pf"]).to_dict("records")},
                 out, prov)
    registry_append({
        "evidence_id": f"r7-tail-mechanism-{slug}-v1", "tier": "pilot",
        "what": f"R7 tail mechanization ({slug}): {summ['reading']}",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(out), "sha256": sha256_file(out)}]})
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
