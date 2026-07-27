# Phase 0: verify VM1's smoke artifacts still load and reproduce on VM2.
#
# VM1 fit lens/smoke7b_lens.pt on Olmo-3-7B-Instruct and recorded an 11-probe
# boot-battery in metrics/smoke_7b.json (J-lens 9/11 hits@20 mid-layers,
# median best mid rank 3). The fitting code died with the VM; this re-runs
# the exact stored battery with the stored lens and the rebuilt pipeline.
# Pass = pipeline + artifacts proven end-to-end; only then touch the 32B.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (RUN_DIR, answer_rank, atomic_write_json, die,
                        ensure_dirs, first_token_id, load_model, log,
                        read_json, seed_all)

import torch
from jlens import JacobianLens

OUT = RUN_DIR / "metrics" / "smoke_verify_vm2.json"


def main() -> None:
    ensure_dirs()
    seed_all()
    if OUT.exists() and "--force" not in sys.argv:
        log(f"{OUT} exists; skipping (use --force to redo)")
        return

    ref = read_json(RUN_DIR / "metrics" / "smoke_7b.json")
    lens = JacobianLens.load(str(RUN_DIR / "lens" / "smoke7b_lens.pt"))
    log(f"loaded VM1 lens: {lens!r}")
    if lens.source_layers != ref["sources"]:
        die(f"lens layers {lens.source_layers} != recorded {ref['sources']}")

    model, hf, tok = load_model(ref["model"])
    mid_layers = ref["boot_probe"]["mid_layers"]

    rows, hits20 = [], 0
    for item in ref["boot_probe"]["rows"]:
        prompt, answer = item["prompt"], item["answer"]
        ans_id = first_token_id(tok, answer)
        jl, ml, _ = lens.apply(model, prompt, positions=[-1])
        j_ranks = {l: answer_rank(jl[l][0], ans_id) for l in lens.source_layers}
        mid_min = min(j_ranks[l] for l in mid_layers)
        hits20 += mid_min <= 20
        rows.append({
            "prompt": prompt, "answer": answer,
            "jlens_rank_by_layer_vm2": j_ranks,
            "jlens_mid_min_vm2": mid_min,
            "jlens_mid_min_vm1": item["jlens_mid_min"],
            "final_rank_vm2": answer_rank(ml[0], ans_id),
        })
        log(f"  {answer!r:14} mid_min vm2={mid_min:>5}  vm1={item['jlens_mid_min']:>5}")

    med = sorted(r["jlens_mid_min_vm2"] for r in rows)[len(rows) // 2]
    out = {
        "model": ref["model"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_probes": len(rows),
        "jlens_hits_at_20_vm2": hits20,
        "jlens_hits_at_20_vm1": ref["boot_probe"]["jlens_hits_at_20"],
        "median_mid_min_vm2": med,
        "median_mid_min_vm1": ref["boot_probe"]["median_jlens_mid_min"],
        "rows": rows,
    }
    atomic_write_json(out, OUT)
    log(f"wrote {OUT}")

    # Self-checks: same lens + same prompts must reproduce VM1's picture.
    if hits20 < ref["boot_probe"]["jlens_hits_at_20"] - 1:
        die(f"hits@20 regressed: vm2={hits20} vm1={ref['boot_probe']['jlens_hits_at_20']}")
    if med > 2 * max(1, ref["boot_probe"]["median_jlens_mid_min"]):
        die(f"median mid rank regressed: vm2={med} vm1={ref['boot_probe']['median_jlens_mid_min']}")
    del hf, model
    torch.cuda.empty_cache()
    log(f"SMOKE VERIFY PASS: {hits20}/{len(rows)} hits@20 (vm1: "
        f"{ref['boot_probe']['jlens_hits_at_20']}), median {med} (vm1 "
        f"{ref['boot_probe']['median_jlens_mid_min']})")


if __name__ == "__main__":
    main()
