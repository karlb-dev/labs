# C3 expansion — score the Stage-3 hard one-hop candidate pool on the
# anchor model (Olmo-3-32B-Think) and report the difficulty distribution
# by family. Input pool: jspace_part2.c3_pool (committed data module,
# family-first authoring). Endpoint: FULL-ANSWER-SEQUENCE conditional
# logprob (R4 rule; the v1 dev set used first-token lp — recorded as a
# deliberate change, and both are emitted so the sets stay comparable).
#
# THIS SCRIPT DOES NOT PARTITION. It produces a scored candidate pool
# plus the pass/fail counts against the difficulty window. The confirmatory
# /replication partition is a preregistration-freeze action (user-gated,
# hashed before outcomes are viewed) — prereg §Item pools.
#
# Tier: dev (pool construction, not a claim about models).
# Usage: python -m jspace_part2.experiments.c3_pool_score \
#          [--pool v2|v3|all] [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ..battery import answer_variants, seq_lp_from_logits
from ..c3_pool import pool_rows, summary as pool_summary
from ..c3_pool_v4 import rows_v4
from ..c3_pool_v5 import rows_v5
from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

MODEL = "/content/models/olmo3-think"
RUN_DIR_P2 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "part2_20260727")
LO, HI = -9.0, -1.0          # difficulty window (v1 rule, seq-lp scale)


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    version = arg("--pool", "v2")
    if version in ("v4", "v5"):
        # v4 lives in its own data module (34 NEW canonical families for
        # the D5 family-disjoint expansion); everything else is unchanged.
        def pool_rows(_v=None):          # noqa: F811
            return rows_v4() if version == "v4" else rows_v5()

        def pool_summary(_v=None):       # noqa: F811
            import collections
            r = rows_v4() if version == "v4" else rows_v5()
            c = collections.Counter(x["family"] for x in r)
            return {"pool": version, "n_items": len(r), "n_families": len(c),
                    "items_per_family": dict(c)}
    OUT = (RUN_DIR_P2 / "metrics" / "olmo3-think" /
           f"c3_pool_{version}_scores.json")
    POOL_OUT = (RUN_DIR_P2 / "config" / "prompts" /
                f"c3_pool_{version}_scored.jsonl")
    if OUT.exists() and "--force" not in sys.argv:
        print("exists; skipping")
        return
    import transformers
    import jlens

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)

    rows = pool_rows(version)
    scored = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            best_seq, best_first = None, None
            for v in answer_variants(r["answer"]):
                text = r["prompt"].rstrip() + v
                n_p = model.encode(r["prompt"].rstrip(),
                                   max_length=512).shape[1]
                ids = model.encode(text, max_length=512)
                logits = hf(input_ids=ids, use_cache=False).logits[0]\
                    .float().cpu()
                lp = seq_lp_from_logits(ids, logits, n_p)
                fid = tok(v, add_special_tokens=False).input_ids[0]
                flp = float(torch.log_softmax(logits[n_p - 1], -1)[fid])
                if best_seq is None or lp > best_seq:
                    best_seq, best_first = lp, flp
            scored.append({**r, "lp": round(best_seq, 4),
                           "first_tok_lp": round(best_first, 4),
                           "in_window": bool(LO <= best_seq <= HI)})
            if (i + 1) % 25 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(rows)}",
                      flush=True)

    POOL_OUT.parent.mkdir(parents=True, exist_ok=True)
    POOL_OUT.write_text("\n".join(json.dumps(s) for s in scored) + "\n")

    keep = [s for s in scored if s["in_window"]]
    fams = {}
    for s in scored:
        f = fams.setdefault(s["family"], {"n": 0, "n_window": 0})
        f["n"] += 1
        f["n_window"] += int(s["in_window"])
    fams_ok = [f for f, v in fams.items() if v["n_window"] >= 2]
    lps = sorted(s["lp"] for s in scored)
    summ = {
        "pool": pool_summary(version),
        "n_scored": len(scored), "n_in_window": len(keep),
        "window": [LO, HI],
        "n_families_with_ge2_in_window": len(fams_ok),
        "n_ceiling_above_hi": sum(1 for s in scored if s["lp"] > HI),
        "n_unknown_below_lo": sum(1 for s in scored if s["lp"] < LO),
        "lp_quartiles": [round(lps[len(lps) // 4], 2),
                         round(lps[len(lps) // 2], 2),
                         round(lps[3 * len(lps) // 4], 2)],
        "by_family": fams,
        "meets_prereg_floor_n90_fams30": bool(len(keep) >= 90 and
                                              len(fams_ok) >= 30),
        "seconds": round(time.time() - t0)}
    prov = Provenance(
        evidence_id=f"c3-pool-{version}-scored-think-v1", tier="dev",
        command=("python -m jspace_part2.experiments.c3_pool_score "
                 f"--pool {version}"),
        inputs={"pool_module": sha256_file(
            Path(__file__).resolve().parents[1] / "c3_pool.py")},
        model=resolve_model(MODEL))
    write_result({"summary": summ, "rows": scored}, OUT, prov)
    registry_append({
        "evidence_id": f"c3-pool-{version}-scored-think-v1", "tier": "dev",
        "what": (f"Stage-3 hard one-hop pool {version} scored on Think: "
                 f"{len(keep)}/{len(scored)} in window [{LO},{HI}] across "
                 f"{len(fams_ok)} families with >=2 (prereg floor n>=90 & "
                 f">=30 families: {summ['meets_prereg_floor_n90_fams30']}); "
                 f"ceiling {summ['n_ceiling_above_hi']}, unknown "
                 f"{summ['n_unknown_below_lo']}. NOT partitioned — freeze "
                 f"action."),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)},
                    {"path": str(POOL_OUT), "sha256": sha256_file(POOL_OUT)}]})
    print(json.dumps({k: v for k, v in summ.items() if k != "by_family"},
                     indent=2))


if __name__ == "__main__":
    main()
