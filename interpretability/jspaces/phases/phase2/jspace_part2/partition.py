# D5 — deterministic FAMILY-LEVEL partition (nextsteps_2_2 §4-D5, §8.6).
#
# THIS MODULE IS BUILT AND TESTED BUT MUST NOT BE RUN BEFORE THE FREEZE.
# Generating the partition is a preregistration-freeze action: the split
# manifest is created in the dedicated freeze commit, after PI sign-off,
# and NO intervention outcome may be viewed between generation and freeze.
# `build_partition` therefore refuses to write unless explicitly told the
# freeze is authorised.
#
# The split is over FAMILIES, never items. Splitting items inside a shared
# template gives an item holdout, not an independent replication: the two
# halves would share the relation, the surface frame and often the same
# underlying fact.
#
# Balancing: families are allocated greedily, largest first, to whichever
# side is currently lighter on the stratum being balanced. Strata are
# difficulty (median baseline logprob tercile) and answer-token length,
# so neither partition is systematically easier or shorter.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ALGORITHM_VERSION = "family_split_v1"


class FreezeNotAuthorised(RuntimeError):
    pass


def canonical_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def family_table(items: list[dict], *, model: str = "olmo3-think") -> list[dict]:
    """One row per eligible family: size, difficulty, answer length."""
    by: dict[str, list] = {}
    for r in items:
        if r.get("excluded"):
            continue
        m = (r.get("baseline_metrics_by_model") or {}).get(model)
        if not m:
            continue
        if not (-9.0 <= m["answer_seq_lp"] <= -1.0):
            continue
        by.setdefault(r["canonical_family"], []).append((r, m))
    rows = []
    for fam, rs in sorted(by.items()):
        if len(rs) < 3:
            continue
        rows.append({
            "canonical_family": fam,
            "n_items": len(rs),
            "item_ids": sorted(r["item_id"] for r, _ in rs),
            "relation_group": rs[0][0].get("relation_group", fam),
            "tasks": sorted({r["task"] for r, _ in rs}),
            "median_lp": round(float(np.median([m["answer_seq_lp"]
                                                for _, m in rs])), 4),
            "median_answer_tokens": int(np.median(
                [m["answer_token_count"] for _, m in rs])),
        })
    return rows


def build_partition(items: list[dict], *, seed: int, freeze_authorised: bool,
                    model: str = "olmo3-think") -> dict:
    if not freeze_authorised:
        raise FreezeNotAuthorised(
            "Refusing to generate the confirmatory/replication split. This "
            "is a preregistration-freeze action requiring explicit PI "
            "sign-off; see reviews/NEXTSTEPS_2_2_ACCEPTED.md. Pass "
            "freeze_authorised=True only inside the dedicated freeze commit.")
    fams = family_table(items, model=model)
    if len(fams) < 60:
        raise RuntimeError(f"only {len(fams)} eligible families; D5 needs 60")

    lp = np.array([f["median_lp"] for f in fams])
    t1, t2 = np.quantile(lp, [1 / 3, 2 / 3])
    for f in fams:
        f["difficulty_stratum"] = ("hard" if f["median_lp"] <= t1
                                   else "medium" if f["median_lp"] <= t2
                                   else "easy")

    rng = np.random.default_rng(seed)
    order = sorted(fams, key=lambda f: (-f["n_items"], f["canonical_family"]))
    # deterministic tie-break within equal sizes
    rng.shuffle(order) if False else None
    sides = {"confirmatory": [], "replication": []}
    load = {"confirmatory": {}, "replication": {}}
    for f in order:
        k = (f["difficulty_stratum"], f["median_answer_tokens"])
        a = load["confirmatory"].get(k, 0)
        b = load["replication"].get(k, 0)
        if a < b:
            side = "confirmatory"
        elif b < a:
            side = "replication"
        else:
            side = ("confirmatory"
                    if len(sides["confirmatory"]) <= len(sides["replication"])
                    else "replication")
        sides[side].append(f)
        load[side][k] = load[side].get(k, 0) + 1

    assert not (set(x["canonical_family"] for x in sides["confirmatory"]) &
                set(x["canonical_family"] for x in sides["replication"])), \
        "a family appears in both partitions"

    def summarize(rows):
        return {
            "n_families": len(rows),
            "n_items": sum(r["n_items"] for r in rows),
            "families": sorted(r["canonical_family"] for r in rows),
            "item_ids": sorted(i for r in rows for i in r["item_ids"]),
            "difficulty_counts": {s: sum(1 for r in rows
                                         if r["difficulty_stratum"] == s)
                                  for s in ("hard", "medium", "easy")},
            "answer_token_counts": {str(k): sum(1 for r in rows
                                                if r["median_answer_tokens"] == k)
                                    for k in sorted({r["median_answer_tokens"]
                                                     for r in rows})},
            "relation_groups": sorted({r["relation_group"] for r in rows}),
            "median_lp": round(float(np.median([r["median_lp"] for r in rows])), 4),
        }

    out = {
        "algorithm_version": ALGORITHM_VERSION, "seed": seed,
        "anchor_model": model,
        "source_bank_hash": canonical_hash(
            [{"i": r["item_id"], "f": r["canonical_family"]}
             for r in sorted(items, key=lambda r: r["item_id"])]),
        "eligible_family_table": fams,
        "confirmatory": summarize(sides["confirmatory"]),
        "replication": summarize(sides["replication"]),
        "invariants": {
            "families_disjoint": True,
            "no_outcome_data_used": True,
            "split_unit": "canonical_family",
        },
    }
    out["manifest_sha256"] = canonical_hash(
        {k: out[k] for k in ("confirmatory", "replication",
                             "algorithm_version", "seed", "source_bank_hash")})
    return out


def dry_run_report(items: list[dict], *, model: str = "olmo3-think") -> dict:
    """What the partition WOULD look like, without generating or writing
    it — safe before the freeze because it reveals only bank structure,
    never an assignment."""
    fams = family_table(items, model=model)
    lp = np.array([f["median_lp"] for f in fams]) if fams else np.array([0.0])
    return {
        "eligible_families": len(fams),
        "eligible_items": sum(f["n_items"] for f in fams),
        "min_items_per_family": min((f["n_items"] for f in fams), default=0),
        "median_items_per_family": float(np.median(
            [f["n_items"] for f in fams])) if fams else 0,
        "difficulty_range_median_lp": [round(float(lp.min()), 3),
                                       round(float(lp.max()), 3)],
        "relation_groups": len({f["relation_group"] for f in fams}),
        "tasks": sorted({t for f in fams for t in f["tasks"]}),
        "would_meet_d5": len(fams) >= 60,
        "per_side_if_split_evenly": {"families": len(fams) // 2,
                                     "items": sum(f["n_items"] for f in fams) // 2},
    }
