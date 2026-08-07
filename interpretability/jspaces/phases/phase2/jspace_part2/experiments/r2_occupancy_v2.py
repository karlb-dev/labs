# R2 v2 — capacity with the estimand mismatch repaired (nextsteps_2_2 §2.2).
#
# v1 computed `hc = h - global_mean` and then never used it: both shares
# were RAW-energy shares, while the report and handout described the
# result as the paper's globally CENTERED excess variance. v2 reports
# three separately named quantities (occupancy_v2.py) so the label can
# never drift again, and adds the uncertainty v1 lacked: a bootstrap over
# PROMPTS for the capacity share, plus random-dictionary seed sensitivity
# and crossing-rule persistence sensitivity.
#
# The occupancy (crossing) number is expected to be unchanged; it never
# depended on the share definition. The capacity share is expected to
# MOVE, possibly a lot: on synthetic data a reconstruction capturing only
# the corpus mean scores 0.998 raw and 0.000 centered (tests/[4]).
#
# Usage: python -m jspace_part2.experiments.r2_occupancy_v2 \
#          --config configs/r2_occupancy_think.yaml [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..dictionaries import build_j_dictionaries
from ..lib import sha256_file
from ..occupancy_v2 import centered_shares, occupancy_and_excess_v2
from ..paths import to_uri
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result_v2)

SKIP_FIRST, MAX_SEQ = 16, 256
N_BOOT_PROMPT = 400


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    cfg_path = arg("--config")
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    git = require_clean_tree("--allow-dirty" in sys.argv)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "r2_occupancy_v2.json"
    eid = cfg["evidence_id"].replace("-v1", "-v2")
    if out_json.exists() and "--force" not in sys.argv:
        print(f"{out_json} exists; skipping")
        return

    import transformers
    import jlens
    from jlens import JacobianLens
    from jlens.hooks import ActivationRecorder

    lens = JacobianLens.load(cfg["lens_path"])
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model_path"])
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    layers = cfg["layers"]
    prompts = [json.loads(l) for l in
               Path(cfg["prompt_set"]).read_text().splitlines()][:cfg["n_prompts"]]

    # ---- capture residuals, KEEPING the prompt index so the capacity
    # share can be bootstrapped over prompts (v1 reported a point estimate
    # with no uncertainty at all)
    acts = {l: [] for l in layers}
    owner = []
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = model.encode(p["text"], max_length=MAX_SEQ)
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(ids)
            P = ids.shape[1]
            lo = min(SKIP_FIRST, max(P - 8, 1))
            n = max(0, (P - 1) - lo)
            for l in layers:
                acts[l].append(rec.activations[l][0, lo:P - 1].float())
            owner.extend([i] * n)
            if (i + 1) % 20 == 0:
                print(f"  capture {i+1}/{len(prompts)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    H = {l: torch.cat(acts[l]) for l in layers}
    owner = np.array(owner)
    del acts
    print(f"captured: {[tuple(H[l].shape) for l in layers]} "
          f"({len(set(owner.tolist()))} prompts)", flush=True)
    assert len(owner) == H[layers[0]].shape[0], "owner index misaligned"

    jd = build_j_dictionaries(hf, lens, layers)
    V, d = jd[layers[0]].shape
    res = {"per_layer": {}, "config": cfg,
           "n_positions": {str(l): int(H[l].shape[0]) for l in layers}}

    for l in layers:
        rand_dicts = []
        for s in cfg["rand_seeds"]:
            g = torch.Generator().manual_seed(s)
            R = torch.nn.functional.normalize(
                torch.randn(V, d, generator=g), dim=1).to("cuda", torch.float16)
            rand_dicts.append(R)
        Hl = H[l].cuda()
        gm = Hl.mean(0)
        t1 = time.time()
        out = occupancy_and_excess_v2(Hl, jd[l], rand_dicts, cfg["k_max"], gm)
        occ = out.pop("occupancy_crossing_k")

        # ---- prompt-level bootstrap of the CONFIRMATORY capacity share
        pj = out.pop("_pj", None)
        boots = []
        rng = np.random.default_rng(4242)
        uniq = np.unique(owner)
        # recompute reconstructions once at the median K for bootstrapping
        from ..occupancy_v2 import gradient_pursuit_v2
        K = out["occupancy_median"]
        rj = gradient_pursuit_v2(Hl, jd[l], K, keep_recons=True).recons_by_k[K]
        rr = [gradient_pursuit_v2(Hl, R, K, keep_recons=True).recons_by_k[K]
              for R in rand_dicts]
        for _ in range(N_BOOT_PROMPT):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(owner == p)[0] for p in pick])
            ti = torch.as_tensor(idx, device=Hl.device)
            hb = Hl[ti]
            cj = centered_shares(hb, rj[ti], hb.mean(0))["centered_r2_B"]
            cr = float(np.mean([centered_shares(hb, r[ti], hb.mean(0))
                                ["centered_r2_B"] for r in rr]))
            boots.append(cj - cr)
        lo_, hi_ = np.percentile(boots, [2.5, 97.5])

        res["per_layer"][str(l)] = {
            **{k: (v.tolist() if torch.is_tensor(v) else v)
               for k, v in out.items() if k != "definitions"},
            "occ_median": float(occ.float().median()),
            "occ_q25": float(occ.float().quantile(0.25)),
            "occ_q75": float(occ.float().quantile(0.75)),
            "occ_hist": torch.bincount(occ, minlength=cfg["k_max"] + 1).tolist(),
            "centered_excess_ci": {"low": round(float(lo_), 5),
                                   "high": round(float(hi_), 5),
                                   "n_boot": N_BOOT_PROMPT,
                                   "resample": "prompts"},
            "seconds": round(time.time() - t1),
        }
        v = res["per_layer"][str(l)]
        print(f"L{l}: occ med {v['occ_median']} IQR "
              f"[{v['occ_q25']},{v['occ_q75']}] | RAW excess "
              f"{v['raw_reconstruction_excess']:.4f} | CENTERED excess "
              f"{v['centered_variance_explained_excess']:.4f} "
              f"[{lo_:.4f},{hi_:.4f}] | censored "
              f"{v['occupancy_censored_frac']:.2f}", flush=True)
        del rand_dicts, Hl, rj, rr
        torch.cuda.empty_cache()

    res["definitions"] = {
        "occupancy_crossing_k": "sparse-support crossing vs random controls",
        "raw_reconstruction_excess": "the v1 quantity (NOT variance)",
        "centered_variance_explained_excess": "centered R^2, CONFIRMATORY",
        "centered_variance_share_excess_A": "sensitivity only",
    }
    prov = Provenance(
        evidence_id=eid, tier=cfg["tier"],
        command=(f"python -m jspace_part2.experiments.r2_occupancy_v2 "
                 f"--config {cfg_path}"),
        config_path=cfg_path,
        inputs={"lens": sha256_file(cfg["lens_path"]),
                "prompt_set": sha256_file(cfg["prompt_set"])},
        model=resolve_model(cfg["model_path"]),
        seed=cfg["rand_seeds"][0], allow_dirty="--allow-dirty" in sys.argv)
    env = write_result_v2(res, out_json, prov)
    registry_append({
        "evidence_id": eid, "tier": cfg["tier"],
        "what": ("Capacity with the estimand mismatch REPAIRED (supersedes "
                 f"{cfg['evidence_id']}, which reported a raw-energy share "
                 "under the label 'centered excess variance'): " +
                 "; ".join(f"L{l}: occ_med {v['occ_median']}, RAW excess "
                           f"{v['raw_reconstruction_excess']:.4f}, CENTERED "
                           f"excess {v['centered_variance_explained_excess']:.4f} "
                           f"[{v['centered_excess_ci']['low']:.4f},"
                           f"{v['centered_excess_ci']['high']:.4f}]"
                           for l, v in res["per_layer"].items())),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "input_uris": {"lens": to_uri(cfg["lens_path"]),
                       "prompt_set": to_uri(cfg["prompt_set"])},
        "inputs": {"lens": sha256_file(cfg["lens_path"]),
                   "prompt_set": sha256_file(cfg["prompt_set"])},
        "outputs": [{"path": str(out_json), "uri": to_uri(str(out_json)),
                     "sha256": sha256_file(out_json),
                     "payload_sha256": env["payload_sha256"]}]})
    from .. import registry as reg
    try:
        reg.supersede(cfg["evidence_id"], eid,
                      reason="raw-energy share reported as centered variance")
    except Exception as e:
        print(f"  (supersede: {e})")
    print("R2 v2 done")


if __name__ == "__main__":
    main()
