# A3 — Gemma-4-31B-it instrument-adaptation gate (addendum §7.3/§9-6.1).
# Static findings (config): Gemma4ForConditionalGeneration (multimodal
# wrapper), final_logit_softcapping=30.0, tied embeddings, 60L x d5376,
# vocab 262k, 5:1 sliding/full attention. jlens supports the wrapper
# (model.language_model layout) and applies the softcap in unembed —
# reference-faithful; pre-cap-vs-capped Jacobian target recorded as caveat.
# This gate adds the weights-level checks: wrap, hooks on both attention
# types, micro-fit, readout sanity. Verdict either way is the A3 result.
# Usage: python -m jspace_part2.experiments.a3_gemma_gate [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

MODEL = "/content/hf_local/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"
OUT = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
           "metrics/gemma4-31b/a3_gate.json")
FIT_PROMPTS = [
    "The history of astronomy begins with the earliest civilizations that "
    "tracked the motions of the sun, moon, and visible planets across",
    "In modern computing, the distinction between memory and storage has "
    "shaped how programs are written, because fast volatile memory",
]


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    checks, t0 = [], time.time()

    def rec(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)[:200]})
        print(f"  {'ok ' if ok else 'FAIL'} {name} {detail}", flush=True)
        return ok

    import transformers
    import jlens

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    rec("from_hf wraps Gemma4ForConditionalGeneration", True,
        f"n_layers={model.n_layers} d_model={model.d_model}")
    softcap = getattr(model, "_logit_softcap", None)
    rec("softcap detected by jlens", softcap == 30.0, f"cap={softcap}")

    from jlens.hooks import ActivationRecorder
    ids = model.encode("Paris is the capital of France.", max_length=32)
    test_layers = [28, 29]  # sliding + full attention neighbors
    with ActivationRecorder(model.layers, at=test_layers) as r:
        with torch.no_grad():
            model.forward(ids)
    ok_shapes = all(r.activations[l].shape[-1] == model.d_model
                    for l in test_layers)
    rec("hooks fire on sliding AND full attention layers", ok_shapes,
        {l: tuple(r.activations[l].shape) for l in test_layers})

    lens = jlens.fit(model, FIT_PROMPTS, source_layers=[24, 30],
                     target_layer=model.n_layers - 1, dim_batch=32,
                     max_seq_len=64, skip_first=8)
    peak = torch.cuda.max_memory_allocated() / 1e9
    rec("micro-fit completes", True, f"peak VRAM {peak:.1f}GB")
    rec("jacobians finite", all(torch.isfinite(lens.jacobians[l]).all()
                                for l in [24, 30]))

    jl, ml, _ = lens.apply(model, "The capital of France is", positions=[-1])
    ll, _, _ = lens.apply(model, "The capital of France is", positions=[-1],
                          use_jacobian=False)
    pid = tok(" Paris", add_special_tokens=False).input_ids[0]
    ranks = {}
    for name, out_ in (("jlens", jl), ("logit", ll)):
        ranks[name] = min(int((out_[l][0] > out_[l][0][pid]).sum()) + 1
                          for l in [24, 30])
    final_rank = int((ml[0] > ml[0][pid]).sum()) + 1
    rec("model knows the probe (final rank<=3)", final_rank <= 3, final_rank)
    rec("micro-lens readout sane (jlens rank<=100 mid-band on 2-prompt fit)",
        ranks["jlens"] <= 100, ranks)

    passed = all(c["pass"] for c in checks)
    verdict = "GEMMA_GATE_PASS" if passed else \
        "GEMMA_BLOCKED_" + ",".join(c["check"] for c in checks if not c["pass"])
    res = {"verdict": verdict, "checks": checks,
           "config_findings": {"softcapping": 30.0, "wrapper":
                               "Gemma4ForConditionalGeneration (language_model layout)",
                               "tied_embeddings": True, "vocab": 262144,
                               "d_model": model.d_model, "n_layers": model.n_layers,
                               "attention": "5:1 sliding(1024)/full"},
           "caveats": ["Jacobian fits target CAPPED logits (reference-"
                       "faithful; pre-cap variant = recorded follow-up)",
                       "dictionary memory: 262k x 5376 fp16 = 2.8GB/layer "
                       "-> band grids need chunked build + layer batching"],
           "seconds": round(time.time() - t0)}
    prov = Provenance(evidence_id="a3-gemma-gate-v1", tier="pilot",
                      command="python -m jspace_part2.experiments.a3_gemma_gate",
                      model=resolve_model(MODEL))
    write_result(res, OUT, prov)
    registry_append({
        "evidence_id": "a3-gemma-gate-v1", "tier": "pilot",
        "what": f"A3 adaptation gate: {verdict} ({len(checks)} checks; "
                f"micro-fit peak {peak:.0f}GB; probe ranks {ranks})",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(f"A3 GATE: {verdict} ({res['seconds']}s)")


if __name__ == "__main__":
    main()
