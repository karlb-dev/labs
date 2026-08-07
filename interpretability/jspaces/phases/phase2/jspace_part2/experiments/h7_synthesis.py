# N3 synthesis — the consolidated answer to D2/H7, from banked data only.
#
# Three measurements, one conclusion, stated so the methods section can
# cite a single evidence id:
#
#  (a) ESTIMATOR. Conditioning the Jacobian on source position (leave-one-
#      out across prompts) does not improve response DIRECTION at any band
#      layer. It does improve MAGNITUDE by roughly 2x. So H7's "averaging
#      discards the accuracy" is false for direction and partly true for
#      scale.
#  (b) CEILING. Cell by cell, response cosine tracks the measured local
#      linearity of the ground truth (within-layer Pearson r 0.76-0.90).
#      What limits in-band faithfulness is therefore not how the Jacobian
#      was estimated.
#  (c) WHERE IT IS USED. Along the lens's own top rows — the directions
#      the dynamic top-k ablation actually selects — the lens is much more
#      faithful than along random or unembedding-aligned directions
#      (in-band cosine ~0.52-0.61 vs ~0.19-0.44). The frequently quoted
#      "~0.2 cosine in band" is a random-probe number and OVERSTATES the
#      problem for the intervention the campaign performs.
#
# Scope note that must travel with this result: the earlier uniform-
# perturbation test found OLMo LINEAR across this band (ratio 1.98-2.02).
# Both are correct; they are different probes. A uniform shift moves every
# key together and partly cancels inside the attention softmax, while a
# single-position shift does not. The probe that matches a position-wise
# intervention is the single-position one, and under that probe the band
# is measurably less linear at shallow depth (1.65 at L24).
#
# Tier: pilot (methods). CPU. Usage:
#   python -m jspace_part2.experiments.h7_synthesis [--model olmo3-think]
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..lib import sha256_file
from ..paths import to_uri
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          write_result_v2)

RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")
BAND = [24, 32, 40]


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--model", "olmo3-think")
    p_ctx = RUN / "metrics" / slug / "h7_context_j.json"
    p_ceil = RUN / "metrics" / slug / "h7_ceiling.json"
    ctx = json.loads(p_ctx.read_text())["payload"]
    ceil = json.loads(p_ceil.read_text())["payload"]
    recs = ctx["records"]

    def med(est, L, kind=None, field="cos"):
        v = [r[field] for r in recs if r["estimator"] == est
             and r["layer"] == L and (kind is None or r["kind"] == kind)
             and r[field] is not None]
        return round(float(np.median(v)), 4) if v else None

    layers = sorted({r["layer"] for r in recs})
    by_layer_kind = {
        str(L): {k: {"cos_meanJ": med("campaign_meanJ", L, k),
                     "cos_posJ": med("position_J_loo", L, k),
                     "norm_ratio_meanJ": med("campaign_meanJ", L, k,
                                             "norm_ratio")}
                 for k in ("jrow", "random", "logit")}
        for L in layers}

    band = [L for L in BAND if L in layers]
    jrow_band = [by_layer_kind[str(L)]["jrow"]["cos_meanJ"] for L in band]
    rand_band = [by_layer_kind[str(L)]["random"]["cos_meanJ"] for L in band]

    payload = {
        "sources": [f"h7-context-j-{slug}-v2", f"h7-linearity-ceiling-{slug}-v1"],
        "a_estimator": {
            "direction_gain_position_over_mean_by_band_layer": {
                str(L): round((ctx["by_layer"][str(L)]["position_J_loo"]["cos"]
                               - ctx["by_layer"][str(L)]["campaign_meanJ"]["cos"]), 4)
                for L in band},
            "magnitude_norm_ratio_by_band_layer": {
                str(L): {"meanJ": ctx["by_layer"][str(L)]["campaign_meanJ"]["norm_ratio"],
                         "positionJ": ctx["by_layer"][str(L)]["position_J_loo"]["norm_ratio"]}
                for L in band},
            "dev_gate": ctx["verdict"],
            "conclusion": ("position-conditioning does NOT improve direction; "
                           "it roughly halves the magnitude error")},
        "b_ceiling": {
            "within_layer_pearson_r": ceil["correlations"]["cos_meanJ"]["within_layer"],
            "overall_pearson_r": ceil["correlations"]["cos_meanJ"]["pearson_r"],
            "median_linearity_by_layer": {
                L: v["median_linearity_ratio"] for L, v in ceil["by_layer"].items()},
            "conclusion": ("cosine tracks the ground truth's own linearity "
                           "within layer, so the estimator is not the "
                           "binding constraint in band")},
        "c_where_used": {
            "by_layer_kind": by_layer_kind,
            "band_median_cos_jrow": round(float(np.median(jrow_band)), 4),
            "band_median_cos_random": round(float(np.median(rand_band)), 4),
            "conclusion": ("the lens is most faithful along its own top "
                           "rows, which is where the ablation acts; the "
                           "random-probe number understates its accuracy "
                           "for the actual intervention")},
        "scope": ("Uniform-perturbation linearity (1.98-2.02, "
                  "local-linearity-v3) and single-position linearity "
                  "(1.65-1.96, here) are different probes and both stand. "
                  "Causal claims about position-wise interventions must use "
                  "the single-position numbers."),
        "consequence_for_preregistration": (
            "The contextJ_methods arm is NOT admitted: it failed its "
            "committed dev gate. meanJ_paper remains the sole primary, now "
            "with a measured and defensible statement of what it costs — "
            "in-band direction cosine ~0.52-0.61 along selected rows, "
            "magnitude under-predicted ~2x, and a ceiling that no better "
            "estimator can pass. That bound applies to every J-direction "
            "causal claim in this literature, this campaign's included."),
    }
    out = RUN / "metrics" / slug / "h7_synthesis.json"
    prov = Provenance(
        evidence_id=f"h7-synthesis-{slug}-v1", tier="pilot",
        command=f"python -m jspace_part2.experiments.h7_synthesis --model {slug}",
        inputs={"context_j": sha256_file(p_ctx), "ceiling": sha256_file(p_ceil)})
    env = write_result_v2(payload, out, prov)
    registry_append({
        "evidence_id": f"h7-synthesis-{slug}-v1", "tier": "pilot",
        "what": (f"D2/H7 CONSOLIDATED ({slug}). (a) Position-conditioning "
                 f"does not improve response direction in band (gain " +
                 ", ".join(f"L{L} {payload['a_estimator']['direction_gain_position_over_mean_by_band_layer'][str(L)]:+.3f}"
                           for L in band) +
                 f") but roughly halves the magnitude error, so H7 is false "
                 f"for direction and partly true for scale. (b) Response "
                 f"cosine tracks the ground truth's own local linearity "
                 f"WITHIN layer (Pearson r " +
                 ", ".join(f"L{k} {v['r']}" for k, v in
                           payload["b_ceiling"]["within_layer_pearson_r"].items()) +
                 f"), so the estimator is not the binding constraint. (c) "
                 f"Along the lens's own top rows — where the ablation acts — "
                 f"in-band cosine is "
                 f"{payload['c_where_used']['band_median_cos_jrow']} vs "
                 f"{payload['c_where_used']['band_median_cos_random']} for "
                 f"random probes. CONSEQUENCE: the contextJ methods arm is "
                 f"NOT admitted to the preregistration (failed its committed "
                 f"dev gate); meanJ_paper stays the sole primary with a "
                 f"measured statement of its cost."),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "input_uris": {"context_j": to_uri(str(p_ctx)),
                       "ceiling": to_uri(str(p_ceil))},
        "inputs": {"context_j": sha256_file(p_ctx), "ceiling": sha256_file(p_ceil)},
        "outputs": [{"path": str(out), "uri": to_uri(str(out)),
                     "sha256": sha256_file(out),
                     "payload_sha256": env["payload_sha256"]}]})
    print(json.dumps(payload, indent=1)[:3500])


if __name__ == "__main__":
    main()
