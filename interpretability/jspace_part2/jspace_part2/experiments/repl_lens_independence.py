# HP3's independent-lens clause, discharged on the replication partition:
# compare per-item meanJ_protected deltas between the draw-A and draw-B
# lenses for each primary checkpoint. High per-item correlation + tail
# reproduction = the protected-tail phenomenon is a MODEL property, not
# a fit artifact (the pilot's Instruct result, now at confirmatory tier
# on held-out families).
#
# Usage: python -m jspace_part2.experiments.repl_lens_independence \
#          [--slugs olmo31-think,olmo31-instruct] [--allow-dirty]
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          write_result_v2)

RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")
THRESH = -1.0


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def deltas(slug: str, suffix: str) -> pd.Series:
    pq = RUN / "metrics" / slug / f"n6_grid{suffix}" / \
        f"n6_per_item_{slug}.parquet"
    df = pd.read_parquet(pq)
    df = df[df.task != "prose"]        # v2: real items only (v1 included
    #                                    20 near-zero prose pairs)
    base = df[df.condition == "baseline"].set_index("item_id")["lp_logsumexp"]
    j = df[df.condition == "meanJ_protected"].set_index("item_id")[
        "lp_logsumexp"]
    common = j.index.intersection(base.index)
    return (j.loc[common] - base.loc[common])


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    slugs = arg("--slugs", "olmo31-think,olmo31-instruct").split(",")
    out = {}
    for slug in slugs:
        a, b = deltas(slug, "_repl"), deltas(slug, "_repl_lensB")
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        ta, tb = a < THRESH, b < THRESH
        out[slug] = {
            "n_items": int(len(common)),
            "pearson_delta": round(float(np.corrcoef(a, b)[0, 1]), 4),
            "spearman_delta": round(float(
                pd.Series(a).corr(pd.Series(b), method="spearman")), 4),
            "tail_n_lensA": int(ta.sum()), "tail_n_lensB": int(tb.sum()),
            "tail_jaccard": round(float((ta & tb).sum() /
                                        max((ta | tb).sum(), 1)), 4),
            "mean_delta_lensA": round(float(a.mean()), 4),
            "mean_delta_lensB": round(float(b.mean()), 4),
        }
        print(slug, json.dumps(out[slug]))
    p = RUN / "metrics" / "cross_model" / "repl_lens_independence.json"
    prov = Provenance(
        evidence_id="n6-repl-lens-independence-v2", tier="confirmatory",
        command=("python -m jspace_part2.experiments.repl_lens_independence "
                 f"--slugs {','.join(slugs)}"),
        inputs={f"{s}_{sfx}": sha256_file(
            RUN / "metrics" / s / f"n6_grid{sfx}" / f"n6_per_item_{s}.parquet")
            for s in slugs for sfx in ("_repl", "_repl_lensB")},
        model={"note": "analysis over banked replication parquets"}, seed=0)
    write_result_v2(out, p, prov)
    registry_append({
        "evidence_id": "n6-repl-lens-independence-v2", "tier": "confirmatory",
        "what": ("HP3 independent-lens clause on the replication partition: "
                 + "; ".join(
                     f"{s} r={out[s]['pearson_delta']} tail-Jaccard "
                     f"{out[s]['tail_jaccard']}" for s in slugs)),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(p), "sha256": sha256_file(p)}]})


if __name__ == "__main__":
    main()
