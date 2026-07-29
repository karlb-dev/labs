# Dev validation of the geometry-matched primary control
# (dyn_energy_rank_matched_random) — freeze-blocking condition 2.
#
# GATES, COMMITTED BEFORE THE RUN (mechanical only — deliberately NO
# behavioural gate, so the control cannot be tuned on outcomes):
#   MC1  rank match:        achieved rank == J rank at 100% of positions
#   MC2  energy match:      median relative error <= 0.5%, max <= 5%
#                           (clamped positions excluded from MC2, bounded
#                           by MC3)
#   MC3  clamp rate:        <= 1% of positions may hit the reachable-energy
#                           clamp
#   MC4  protection:        max |basis . protected_row| <= 1e-3
# Direction freedom (the control span shares nothing with the J span
# beyond the h-alignment the energy match forces) is a property of the
# CONSTRUCTION, verified synthetically in tests/test_matched_control.py
# (test_randomness_vs_reference_span); there is nothing further for a GPU
# run to establish about the sampler, so it is deliberately not a gate
# here.
# Behavioural deltas (J arm vs matched control) are reported DESCRIPTIVELY
# for the record; they gate nothing here. Items are the pilot one-hop
# battery + the first 12 released two-hop items — all with long-exposed
# pilot outcomes, none of them freeze-partition material until the split
# runs inside the freeze commit.
#
# Usage: python -m jspace_part2.experiments.mc_dev_validation \
#          --model-path <dir> --lens-path <file> [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ..battery import onehop_items, seq_lp_from_logits, twohop_items
from ..dictionaries import build_j_dictionaries
from ..lib import sha256_file
from ..matched_control import teacher_forced_matched_pair_v2
from ..protected_dynamic_v2 import ProtectedDynamicAblatorV2
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

BAND = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]
K, PROTECT, SEED = 10, 10, 4242
RUN_DIR_P2 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "part2_20260727")

GATES = {"MC1_rank_match_frac": 1.0,
         "MC2_energy_rel_err_median": 0.005,
         "MC2_energy_rel_err_max": 0.05,
         "MC3_clamped_frac": 0.01,
         "MC4_max_protected_cos": 1e-3}


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    model_path = arg("--model-path")
    lens_path = arg("--lens-path")
    slug = arg("--slug", Path(model_path).name)
    out_dir = RUN_DIR_P2 / "metrics" / slug / "mc_dev_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    import transformers
    import jlens
    from jlens import JacobianLens
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.load(lens_path)
    dicts = build_j_dictionaries(hf, lens, BAND)
    layers = model.layers

    def encode(text, max_length=512):
        return tok(text, return_tensors="pt", truncation=True,
                   max_length=max_length).input_ids.to("cuda")

    items = twohop_items(12) + onehop_items()[:10]
    rows = []
    j_summary_acc, c_summary_acc = [], []
    for i, it in enumerate(items):
        text = it["prompt"] + " " + it["answer"].strip()
        ids, clean, abl_j, abl_c, j_log, c_log = \
            teacher_forced_matched_pair_v2(
                hf, encode, layers, BAND, dicts, text,
                k=K, protect=PROTECT, seed_base=SEED + i)
        n_prompt = len(encode(it["prompt"])[0])
        lp_clean = seq_lp_from_logits(ids, clean, n_prompt)
        lp_j = seq_lp_from_logits(ids, abl_j, n_prompt)
        lp_c = seq_lp_from_logits(ids, abl_c, n_prompt)
        ms = c_log.matched_summary()
        rows.append({"item_id": it["item_id"],
                     "delta_j": lp_j - lp_clean, "delta_c": lp_c - lp_clean,
                     **{f"mc_{k_}": v for k_, v in ms.items()}})
        c_summary_acc.append(ms)
        j_summary_acc.append(j_log.summary())
        if (i + 1) % 5 == 0:
            log(f"{i+1}/{len(items)} items")

    import statistics as st
    agg = {
        "rank_match_frac": min(s["rank_match_frac"] for s in c_summary_acc),
        "energy_rel_err_median": st.median(
            s["energy_rel_err_median"] for s in c_summary_acc
            if s["energy_rel_err_median"] is not None),
        "energy_rel_err_max": max(
            s["energy_rel_err_max"] for s in c_summary_acc
            if s["energy_rel_err_max"] is not None),
        "clamped_frac": st.mean(s["clamped_frac"] for s in c_summary_acc),
        "max_protected_cos": max(s["max_protected_cos"]
                                 for s in c_summary_acc),
    }
    verdict = {
        "MC1": agg["rank_match_frac"] >= GATES["MC1_rank_match_frac"],
        "MC2": (agg["energy_rel_err_median"] <=
                GATES["MC2_energy_rel_err_median"]
                and agg["energy_rel_err_max"] <=
                GATES["MC2_energy_rel_err_max"]),
        "MC3": agg["clamped_frac"] <= GATES["MC3_clamped_frac"],
        "MC4": agg["max_protected_cos"] <= GATES["MC4_max_protected_cos"],
    }

    deltas_j = [r["delta_j"] for r in rows]
    deltas_c = [r["delta_c"] for r in rows]
    payload = {
        "gates": GATES, "aggregate": agg, "verdict": verdict,
        "pass": all(v for v in verdict.values() if v is not None),
        "n_items": len(rows), "band": BAND, "k": K, "protect": PROTECT,
        "descriptive_deltas": {
            "j_median": st.median(deltas_j), "c_median": st.median(deltas_c),
            "j_mean": st.mean(deltas_j), "c_mean": st.mean(deltas_c)},
        "rows": rows,
    }
    ev_id = f"mc-dev-validation-{slug}-v1"
    prov = Provenance(
        evidence_id=ev_id, tier="dev",
        command=(f"python -m jspace_part2.experiments.mc_dev_validation "
                 f"--model-path {model_path} --lens-path {lens_path} "
                 f"--slug {slug}"),
        inputs={"lens": sha256_file(Path(lens_path))},
        model=resolve_model(model_path), seed=SEED)
    out_path = out_dir / "mc_dev_validation.json"
    write_result(payload, out_path, prov)
    registry_append({
        "evidence_id": ev_id, "tier": "dev",
        "what": (f"Geometry-matched primary control "
                 f"(dyn_energy_rank_matched_random) dev validation on "
                 f"{slug}: gates MC1-MC4 {'PASS' if payload['pass'] else 'FAIL'} "
                 f"(rank match {agg['rank_match_frac']:.3f}, energy rel-err "
                 f"med {agg['energy_rel_err_median']:.2e} max "
                 f"{agg['energy_rel_err_max']:.2e}, clamped "
                 f"{agg['clamped_frac']:.4f}, prot-cos "
                 f"{agg['max_protected_cos']:.1e}); descriptive medians "
                 f"J {st.median(deltas_j):+.3f} vs C "
                 f"{st.median(deltas_c):+.3f} nats"),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(out_path), "sha256": sha256_file(out_path)}]})
    log(f"gates: {verdict} -> {'PASS' if payload['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
