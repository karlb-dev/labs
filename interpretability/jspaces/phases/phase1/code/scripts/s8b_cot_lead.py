# Phase 4b (CPU): squeeze the saved CoT traces for three report numbers.
#   1. Lead time — on traced items, gap between the answer entering the
#      workspace top-8 (answer_emerge_step, from s8) and the answer first
#      appearing in the generated text. Positive = workspace leads text.
#   2. Pre-CoT anticipation layer profile — median rank of the answer per
#      source layer (think prompt and suppressed prompt), J-lens vs logit.
#   3. Top-20 divergence events across items, ranked by run length, with
#      item context — the mission's verbatim examples.
# No GPU, no model forward: tokenizer + saved JSON/traces only.
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import RUN_DIR, atomic_write_json, log

import numpy as np

COT = RUN_DIR / "metrics" / "cot_results.json"
TRACE_DIR = RUN_DIR / "metrics" / "cot_traces"
OUT = RUN_DIR / "metrics" / "cot_lead.json"


def main() -> None:
    import transformers
    res = json.loads(COT.read_text())
    items = res["items"]
    tok = transformers.AutoTokenizer.from_pretrained("allenai/Olmo-3-32B-Think")

    # ---- 1. lead time on traced items
    leads = []
    for f in sorted(TRACE_DIR.glob("item_*.json.gz")):
        iid = f.stem.split("_")[1].split(".")[0]
        it = items.get(iid)
        if it is None or it["answer_emerge_step"] is None:
            continue
        steps = json.loads(gzip.open(f, "rt").read())["steps"]
        ans = it["answer"].strip().lower().replace(" ", "")
        cum = ""
        text_step = None
        for i, s in enumerate(steps):
            cum += tok.decode([s["tok"]], skip_special_tokens=False)
            if ans in cum.lower().replace(" ", ""):
                text_step = i
                break
        if text_step is None:
            continue
        leads.append({"iid": int(iid), "kind": it["kind"],
                      "emerge_step": it["answer_emerge_step"],
                      "text_step": text_step,
                      "lead": text_step - it["answer_emerge_step"]})
    lead_vals = [x["lead"] for x in leads]
    lead_summary = {
        "n_traced_scored": len(leads),
        "median_lead_steps": float(np.median(lead_vals)) if leads else None,
        "frac_workspace_leads": float(np.mean([v > 0 for v in lead_vals]))
        if leads else None,
        "per_kind_median": {k: float(np.median([x["lead"] for x in leads
                                                if x["kind"] == k]))
                            for k in ("twohop", "arithmetic", "sql")
                            if any(x["kind"] == k for x in leads)},
        "note": "text_step = first step whose cumulative decoded text "
                "contains the space-stripped lowercased answer; emerge_step "
                "from s8 (first appearance in any read layer's top-8).",
    }

    # ---- 2. per-layer anticipation profile (all items)
    some = next(iter(items.values()))
    layers = sorted(int(l) for l in
                    some["pre_cot"]["answer"]["jlens_rank_by_layer"])
    prof = {}
    for name, path in (("think_jlens", ("pre_cot", "answer",
                                        "jlens_rank_by_layer")),
                       ("think_logit", ("pre_cot", "answer",
                                        "logit_rank_by_layer")),
                       ("suppressed_jlens", ("suppressed", "pre", "answer",
                                             "jlens_rank_by_layer")),
                       ("suppressed_logit", ("suppressed", "pre", "answer",
                                             "logit_rank_by_layer"))):
        by_layer = {}
        for l in layers:
            vals = []
            for it in items.values():
                d = it
                for k in path:
                    d = d[k]
                vals.append(d[str(l)])
            by_layer[l] = float(np.median(vals))
        prof[name] = by_layer

    # ---- 3. top-20 divergence events
    ev = []
    for iid, it in items.items():
        for e in it["divergence_events"]:
            ev.append({**e, "iid": int(iid), "kind": it["kind"],
                       "question": it["question"], "answer": it["answer"],
                       "think_correct": it["think_correct"]})
    ev.sort(key=lambda e: -e["run_len"])
    out = {"lead": lead_summary, "leads_detail": leads,
           "anticipation_rank_by_layer_median": prof,
           "top20_divergence_events": ev[:20],
           "n_divergence_events_total": len(ev)}
    atomic_write_json(out, OUT)
    log(f"wrote {OUT} (traced {len(leads)}, events {len(ev)})")


if __name__ == "__main__":
    main()
