# Offline G5 regrade — no GPU, no new generations.
#
# Two instrument fixes invalidated the first G5 grading pass:
#   1. ScoringSpec.normalize DELETED whitespace runs instead of spacing
#      them ("the\nBaht" -> "thebaht"), failing prefix grading on
#      correct newline-led generations;
#   2. currency alias sets lacked adjectival forms the models actually
#      produce ("the Thai baht") — Bank F v7 appends them (alias[0]
#      unchanged, so every stored lp_canonical stays valid).
# The G5 parquets store each item's greedy generation (first 80 chars —
# ample for prefix grading), so capability regrades OFFLINE from the
# stored text. lp columns are untouched.
#
# Usage: python -m jspace_phase3.experiments.g5_regrade
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from ..bank import load_bank
from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)
from ..scoring import DEFAULT_SPEC

TIER = "phase3-development"
REPO_DATA = Path(__file__).resolve().parents[2] / "data"
BANKS = ["bank_f_v7.jsonl", "bank_s_v3.jsonl"]
PREV = {"olmo31-think": "v3", "olmo31-instruct": "v2", "qwen36-27b": "v2"}


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    aliases = {}
    for bank in BANKS:
        for b in load_bank(REPO_DATA / bank):
            for it in b.as_items():
                aliases[it["item_id"]] = it["accepted_answers"]

    norm = DEFAULT_SPEC.normalize
    for slug, prev in PREV.items():
        d = metrics_dir(slug) / "g5_bank"
        df = pd.read_parquet(d / f"g5_bank_{slug}.parquet")

        def grade_prefix(row):
            gnorm = norm(row.generation)
            return any(gnorm.startswith(norm(a))
                       for a in aliases[row.item_id] if norm(a))

        def grade_contains(row):
            g = f" {norm(row.generation)}"
            return any(f" {norm(a)}" in g
                       for a in aliases[row.item_id] if norm(a))
        old = df.capable_generation.copy()
        # THE inclusion predicate: answer appears anywhere in the 8-token
        # greedy continuation. The endpoints are teacher-forced logprobs,
        # so completion-mode prefixing is not required for a measurable
        # item; prefix capability is kept as a reported column (prereg §3
        # discloses both counts).
        df["capable_prefix"] = df.apply(grade_prefix, axis=1)
        df["capable_generation"] = df.apply(grade_contains, axis=1)
        flips = int((df.capable_generation != old).sum())
        pq = d / f"g5_bank_{slug}_regraded.parquet"
        df.to_parquet(pq)
        summary = {
            "n_items": int(len(df)), "flips": flips,
            "capable_prefix_rate": round(
                float(df.capable_prefix.mean()), 4),
            "capable_rate": round(float(df.capable_generation.mean()), 4),
            "capable_by_variant": {k: round(float(v), 4) for k, v in
                                   df.groupby("variant")
                                   .capable_generation.mean().items()},
            "capable_by_bank": {k: round(float(v), 4) for k, v in
                                df.groupby("bank")
                                .capable_generation.mean().items()}}
        eid_prev = f"p3-g5-bank-{slug}-{prev}"
        n_new = int(prev[1]) + 1
        eid = f"p3-g5-bank-{slug}-v{n_new}"
        cmd = "python -m jspace_phase3.experiments.g5_regrade"
        out = d / f"g5_bank_{slug}_regraded.json"
        write_result3({"summary": summary}, out, Provenance3(
            evidence_id=eid, tier=TIER, command=cmd, seed=0))
        register(eid, tier=TIER, command=cmd, supersedes=eid_prev,
                 what=(f"G5 predicate v2 on {slug}: capable_generation = "
                       f"answer CONTAINED in the 8-token greedy "
                       f"continuation (endpoints are teacher-forced lp; "
                       f"prefix rate kept as column): rate "
                       f"{summary['capable_rate']} vs prefix "
                       f"{summary['capable_prefix_rate']}; {flips} flips; "
                       f"lp columns untouched"),
                 outputs=[out, pq])
        print(slug, json.dumps(summary))


if __name__ == "__main__":
    main()
