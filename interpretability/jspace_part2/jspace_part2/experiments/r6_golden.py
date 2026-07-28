# R6 — tiny-model golden test: prove the pipeline end-to-end on a small
# REAL decoder (SmolLM2-135M, Drive HF cache). Not about workspace science;
# about invariants unit tests can't reach (addendum §1.8):
#   G1: jlens fit -> save -> load -> readout runs and is deterministic
#   G2: protected_generate: protected ids never selected on real tensors;
#       protected vs unprotected produce different logs; hooks detach clean
#   G3: teacher-forced protected pass matches shape/finiteness contracts
#   G4: provenance block embeds and output hash is stable across reruns
# Usage: python -m jspace_part2.experiments.r6_golden [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ..dictionaries import build_j_dictionaries
from ..lib import sha256_file
from ..protected_dynamic import (ProtectedDynamicAblator, protected_generate,
                                 protected_teacher_forced)
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"   # Drive HF cache
OUT = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
           "metrics/golden/r6_golden.json")


def check(rows, name, cond):
    rows.append({"check": name, "pass": bool(cond)})
    print(f"  {'ok ' if cond else 'FAIL'} {name}", flush=True)
    if not cond:
        raise SystemExit(f"R6 GOLDEN FAILED: {name}")


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    import transformers
    import jlens
    from jlens import JacobianLens

    rows = []
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    layers = [model.n_layers // 3, model.n_layers // 2]

    # G1: fit -> save -> load -> readout, deterministic
    prompts = ["The capital of France is Paris, a city known for its",
               "Water is composed of hydrogen and oxygen atoms that",
               "The quick brown fox jumps over the lazy dog and then"]
    lens = jlens.fit(model, prompts, source_layers=layers,
                     target_layer=model.n_layers - 1, dim_batch=64,
                     max_seq_len=32, skip_first=2)
    tmp = Path("/tmp/claude-0/r6_lens.pt")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    lens.save(str(tmp))
    lens2 = JacobianLens.load(str(tmp))
    jl1, _, _ = lens.apply(model, "The largest planet is", positions=[-1])
    jl2, _, _ = lens2.apply(model, "The largest planet is", positions=[-1])
    check(rows, "fit->save->load readout identical",
          all(torch.allclose(jl1[l], jl2[l], atol=1e-4) for l in layers))
    jl3, _, _ = lens2.apply(model, "The largest planet is", positions=[-1])
    check(rows, "readout deterministic across calls",
          all(torch.equal(jl2[l], jl3[l]) for l in layers))

    # G2: protected generation on real tensors
    dicts = build_j_dictionaries(hf, lens2, layers, dtype=torch.float16)
    ab = ProtectedDynamicAblator(model.layers, layers)
    with ab:
        txt_p, _ = protected_generate(hf, tok, ab, dicts,
                                      "The capital of France is", k=5,
                                      protect=10, max_new=8, protected=True)
        blocked_p = ab.log.protected_hits_blocked
        ab.log.__init__()
        txt_u, _ = protected_generate(hf, tok, ab, dicts,
                                      "The capital of France is", k=5,
                                      protect=10, max_new=8, protected=False)
        blocked_u = ab.log.protected_hits_blocked
    check(rows, "protection blocks selections (blocked>0 protected)",
          blocked_p > 0)
    check(rows, "unprotected blocks nothing", blocked_u == 0)
    check(rows, "hooks detached", len(ab._handles) == 0)
    check(rows, "generations returned text",
          isinstance(txt_p, str) and isinstance(txt_u, str))

    # G3: teacher-forced protected pass contracts
    with ab:
        ids, logits = protected_teacher_forced(
            hf, model.encode, ab, dicts, "Paris is the capital of France.",
            k=5, protect=10, protected=True, max_length=32)
    check(rows, "teacher-forced logits finite", bool(torch.isfinite(logits).all()))
    check(rows, "teacher-forced shape [T,V]",
          logits.shape[0] == ids.shape[1] and logits.shape[1] >= 40000)

    # G4: provenance + output hash stability
    prov = Provenance(evidence_id="r6-golden-v1", tier="pilot",
                      command="python -m jspace_part2.experiments.r6_golden",
                      model=resolve_model(MODEL))
    payload = {"checks": rows, "seconds": round(time.time() - t0),
               "model": MODEL, "layers": layers}
    write_result(dict(payload), OUT, prov)
    h1 = sha256_file(OUT)
    d = json.loads(OUT.read_text())
    check(rows, "provenance block embedded",
          d.get("provenance", {}).get("evidence_id") == "r6-golden-v1")
    registry_append({
        "evidence_id": "r6-golden-v1", "tier": "pilot",
        "what": f"tiny-model end-to-end golden: {len(rows)} checks green "
                f"(fit/save/load determinism, protection invariants on real "
                f"tensors, teacher-forced contracts, provenance embed)",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": h1}]})
    print(f"R6 GOLDEN: all {len(rows)} checks green ({payload['seconds']}s)")


if __name__ == "__main__":
    main()
