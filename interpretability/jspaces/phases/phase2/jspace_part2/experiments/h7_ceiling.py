# N3 follow-up — is the in-band faithfulness gap an ESTIMATOR failure or a
# LINEARITY CEILING?
#
# `h7-context-j-olmo3-think-v2` returned a decisive null: conditioning the
# Jacobian on position (leave-one-out across prompts) does NOT improve
# response DIRECTION at any band layer (gain -0.05, -0.04, -0.02, +0.02).
# So the campaign's H7 — "averaging over positions and prompts is what
# discards the accuracy" — is refuted for direction.
#
# But the same run measured, per cell, the local linearity of the ground
# truth itself: r(2d)/r(d), where 2.00 is exactly linear. Those ratios are
# NOT uniform (median 1.65 at L24 rising to 1.96 at L56), and they order
# the layers exactly as the cosines do. If cosine tracks the linearity
# ratio cell by cell, then what limits in-band faithfulness is not the
# estimator at all: it is that a single-position perturbation — the kind
# the dynamic ablation actually applies — is genuinely nonlinear at
# shallow band depths, so NO first-order model can do better there.
#
# This module tests that correspondence on the banked payload. No GPU, no
# model load; it is a pure function of registered data.
#
# Tier: pilot (methods). Usage:
#   python -m jspace_part2.experiments.h7_ceiling [--model olmo3-think]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

from ..lib import sha256_file
from ..paths import to_uri
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          write_result_v2)

RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--model", "olmo3-think")
    src = RUN / "metrics" / slug / "h7_context_j.json"
    d = json.loads(src.read_text())["payload"]
    t0 = time.time()

    lin = {(c["layer"], c["pos"], c["prompt"]): c["scale_ratio_median"]
           for c in d["linearity_check"]["cells"]}
    # cell-level median cosine per estimator
    cells = {}
    for r in d["records"]:
        k = (r["layer"], r["pos"], r["prompt"])
        cells.setdefault(k, {}).setdefault(r["estimator"], []).append(r["cos"])
    rows = []
    for k, v in cells.items():
        if k not in lin:
            continue
        rows.append({"layer": k[0], "pos": k[1], "prompt": k[2],
                     "linearity_ratio": lin[k],
                     "cos_meanJ": float(np.median(v["campaign_meanJ"])),
                     "cos_posJ": float(np.median(v["position_J_loo"]))})

    x = np.array([r["linearity_ratio"] for r in rows])
    out = {"n_cells": len(rows)}
    for est in ("cos_meanJ", "cos_posJ"):
        y = np.array([r[est] for r in rows])
        pear = stats.pearsonr(x, y)
        spear = stats.spearmanr(x, y)
        out[est] = {"pearson_r": round(float(pear.statistic), 4),
                    "pearson_p": float(pear.pvalue),
                    "spearman_rho": round(float(spear.statistic), 4),
                    "spearman_p": float(spear.pvalue)}
        # within-layer, so the correlation is not merely a depth trend
        within = {}
        for L in sorted({r["layer"] for r in rows}):
            sub = [r for r in rows if r["layer"] == L]
            if len(sub) > 4:
                sx = np.array([r["linearity_ratio"] for r in sub])
                sy = np.array([r[est] for r in sub])
                if sx.std() > 1e-9:
                    pr = stats.pearsonr(sx, sy)
                    within[str(L)] = {"r": round(float(pr.statistic), 3),
                                      "p": round(float(pr.pvalue), 5),
                                      "n": len(sub)}
        out[est]["within_layer"] = within

    # how much of the cosine is "explained" by the ceiling: a cell whose
    # ground truth is only ~82% linear cannot be predicted with cosine 1.
    by_layer = {}
    for L in sorted({r["layer"] for r in rows}):
        sub = [r for r in rows if r["layer"] == L]
        by_layer[str(L)] = {
            "median_linearity_ratio": round(float(np.median(
                [r["linearity_ratio"] for r in sub])), 3),
            "median_cos_meanJ": round(float(np.median(
                [r["cos_meanJ"] for r in sub])), 3),
            "median_cos_posJ": round(float(np.median(
                [r["cos_posJ"] for r in sub])), 3)}

    payload = {
        "source_evidence": f"h7-context-j-{slug}-v2",
        "question": ("does in-band linearization faithfulness track the "
                     "measured LINEARITY of the ground truth, rather than "
                     "the estimator's conditioning?"),
        "correlations": out, "by_layer": by_layer, "cells": rows,
        "reading": (
            "A strong positive correlation means the first-order model is "
            "about as good as first-order CAN be for that cell, and the "
            "in-band shortfall is a property of the network's response to "
            "position-specific perturbation, not of how the Jacobian was "
            "estimated. WITHIN-layer correlations matter most: a purely "
            "across-layer correlation could just be the trivial "
            "closer-to-target-is-more-linear depth effect."),
    }
    outp = RUN / "metrics" / slug / "h7_ceiling.json"
    prov = Provenance(
        evidence_id=f"h7-linearity-ceiling-{slug}-v1", tier="pilot",
        command=f"python -m jspace_part2.experiments.h7_ceiling --model {slug}",
        inputs={"h7_context_j": sha256_file(src)})
    env = write_result_v2(payload, outp, prov)
    registry_append({
        "evidence_id": f"h7-linearity-ceiling-{slug}-v1", "tier": "pilot",
        "what": (f"Tests whether the in-band linearization gap is an "
                 f"estimator failure or a linearity CEILING, on {len(rows)} "
                 f"cells from h7-context-j-{slug}-v2. Correlation between a "
                 f"cell's measured linearity ratio r(2d)/r(d) and its "
                 f"response cosine: mean-J Pearson "
                 f"{out['cos_meanJ']['pearson_r']} "
                 f"(p={out['cos_meanJ']['pearson_p']:.2g}), position-J "
                 f"{out['cos_posJ']['pearson_r']}. Within-layer: " +
                 json.dumps(out["cos_meanJ"]["within_layer"]) + "."),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "input_uris": {"h7_context_j": to_uri(str(src))},
        "inputs": {"h7_context_j": sha256_file(src)},
        "outputs": [{"path": str(outp), "uri": to_uri(str(outp)),
                     "sha256": sha256_file(outp),
                     "payload_sha256": env["payload_sha256"]}]})
    print(json.dumps({"correlations": out, "by_layer": by_layer,
                      "seconds": round(time.time() - t0)}, indent=1))


if __name__ == "__main__":
    main()
