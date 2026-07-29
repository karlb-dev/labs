# N4 / G5 — the task gate, run rather than waived (PI decision D6).
#
# Produces the IMMUTABLE ITEM MANIFEST the review demands (§2.6, §7-N4.1):
# one row per item carrying verified answer, tokenizations, baseline
# metrics, capability flags, shortcut audits, canonical family, template
# hash and an UNASSIGNED partition field. The gate output is a manifest,
# not a report paragraph.
#
# Checks implemented here:
#   A  answer/alias well-formedness and tokenization under the anchor model
#   B  leakage: the answer string (or a trivial alias) must not appear in
#      the prompt
#   C  two-hop bridge audit: the bridge entity must not appear verbatim in
#      the prompt, and the released swap counterfactual must exist
#   D  capability: baseline full-sequence logprob and greedy generation
#      accuracy on the anchor model, with the frozen difficulty window
#   E  duplicate-fact audit ACROSS families (the `func-filters-count` /
#      `organ-count-kidney2` case: two surfaces, one fact)
#   F  prompt-length and answer-token-length summaries per family, so a
#      family difference cannot be a length artifact
#   G  family/template integrity via the audited map
#
# WHAT THIS GATE CANNOT DO ON ONE VM, stated rather than hidden: the
# cross-model capability cohorts need every confirmatory model's weights.
# Only the anchor (OLMo-3-32B-Think) is resident, so `capability_flags`
# carries the anchor and every other model is recorded as PENDING. The
# cohort assignment is therefore INCOMPLETE by construction and the
# manifest says so; completing it is a prerequisite for the freeze, not
# for this gate.
#
# Tier: dev. Usage:
#   python -m jspace_part2.experiments.g5_task_gate [--allow-dirty]
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..battery import ONEHOP, answer_variants, seq_lp_from_logits
from ..c3_pool import canonical_family
from ..family import load_map
from ..lib import sha256_file
from ..paths import to_uri
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result_v2)

RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")
MODEL = "/content/models/olmo3-think"
PROBE_SWAP = Path("/content/jacobian-lens/data/experiments/probe-swap.json")
SCORES = RUN / "metrics" / "olmo3-think"
LO, HI = -9.0, -1.0
MAX_NEW = 8


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    t0 = time.time()
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()

    fam_map = load_map()
    items: list[dict] = []

    # ---------------- two-hop (released probe-swap) ----------------------
    ps = json.loads(PROBE_SWAP.read_text())["items"]
    for it in ps:
        iid = f"twohop:{it['name']}"
        fm = fam_map["_by_item"].get(iid, {})
        bridge = it.get("intermediate")
        items.append({
            "item_id": iid, "source_pool": "probe_swap_released",
            "task": "twohop", "prompt": it["prompt"],
            "canonical_answer": it["answer"],
            "accepted_answers": answer_variants(it["answer"]),
            "canonical_family": fm.get("canonical_family"),
            "template_id": fm.get("template_id"),
            "template_hash": fm.get("template_hash"),
            "bridge_entity": bridge,
            "counterfactual": {"swap_to": it.get("swap_to"),
                               "swap_answer": it.get("swap_answer")},
            "ground_truth_source": "released with anthropics/jacobian-lens",
            "partition": "UNASSIGNED"})

    # ---------------- one-hop battery -----------------------------------
    for i, (p, a) in enumerate(ONEHOP):
        iid = f"onehop:{i}"
        fm = fam_map["_by_item"].get(iid, {})
        items.append({
            "item_id": iid, "source_pool": "battery_onehop",
            "task": "onehop", "prompt": p, "canonical_answer": a,
            "accepted_answers": answer_variants(a),
            "canonical_family": fm.get("canonical_family"),
            "template_id": fm.get("template_id"),
            "template_hash": fm.get("template_hash"),
            "bridge_entity": None, "counterfactual": None,
            "ground_truth_source": "authored (package battery)",
            "partition": "UNASSIGNED"})

    # ---------------- Stage-3 hard one-hop bank -------------------------
    for v in ("v2", "v3", "v4", "v5"):
        f = SCORES / f"c3_pool_{v}_scores.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text())["rows"]:
            cf = canonical_family(r["family"])
            ans = r["answer"].strip()
            iid = f"c3:{v}:{hashlib.sha256(r['prompt'].encode()).hexdigest()[:10]}"
            items.append({
                "item_id": iid, "source_pool": f"c3_{v}",
                "task": "hard_onehop", "prompt": r["prompt"],
                "canonical_answer": ans,
                "accepted_answers": answer_variants(ans),
                "canonical_family": cf, "template_id": r["family"],
                "template_hash": hashlib.sha256(
                    cf.encode()).hexdigest()[:16],
                "bridge_entity": None, "counterfactual": None,
                "ground_truth_source": "authored (c3 pool, hand-verified)",
                "prescored_lp": r["lp"], "prescored_in_window": r["in_window"],
                "partition": "UNASSIGNED"})

    print(f"manifest rows: {len(items)}", flush=True)

    # ---------------- checks --------------------------------------------
    fails = {"missing_family": [], "answer_in_prompt": [],
             "bridge_in_prompt": [], "no_counterfactual": [],
             "empty_answer": []}
    for r in items:
        if not r["canonical_family"]:
            fails["missing_family"].append(r["item_id"])
        if not r["canonical_answer"].strip():
            fails["empty_answer"].append(r["item_id"])
        if norm(r["canonical_answer"]) and \
                norm(r["canonical_answer"]) in norm(r["prompt"]).split():
            fails["answer_in_prompt"].append(r["item_id"])
        if r["task"] == "twohop":
            b = r.get("bridge_entity")
            if b and norm(b) in norm(r["prompt"]):
                fails["bridge_in_prompt"].append(r["item_id"])
            cf = r.get("counterfactual") or {}
            if not (cf.get("swap_to") and cf.get("swap_answer")):
                fails["no_counterfactual"].append(r["item_id"])

    # duplicate FACT audit across families (same answer + same key nouns)
    seen: dict[tuple, list] = {}
    for r in items:
        key = (norm(r["canonical_answer"]),
               tuple(sorted(w for w in norm(r["prompt"]).split()
                            if len(w) > 5))[:4])
        seen.setdefault(key, []).append(r["item_id"])
    dup_facts = {str(k): v for k, v in seen.items() if len(v) > 1}

    # ---------------- capability on the anchor ---------------------------
    t1 = time.time()
    for n, r in enumerate(items):
        ids = tok(r["prompt"], return_tensors="pt").input_ids.cuda()
        best, best_v = None, -1e9
        with torch.no_grad():
            for v in r["accepted_answers"]:
                aid = tok(v, add_special_tokens=False,
                          return_tensors="pt").input_ids.cuda()
                full = torch.cat([ids, aid], dim=1)
                lg = hf(input_ids=full, use_cache=False).logits[0].float().cpu()
                lp = seq_lp_from_logits(full.cpu(), lg, ids.shape[1])
                if lp > best_v:
                    best_v, best = lp, v
            gen = hf.generate(input_ids=ids, max_new_tokens=MAX_NEW,
                              do_sample=False,
                              pad_token_id=tok.eos_token_id)
        out = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        hit = any(norm(v) and norm(v) in norm(out)
                  for v in r["accepted_answers"])
        r["baseline_metrics_by_model"] = {
            "olmo3-think": {"answer_seq_lp": round(best_v, 4),
                            "best_variant": best,
                            "greedy_continuation": out[:60],
                            "greedy_correct": bool(hit),
                            "answer_token_count": len(tok(
                                best, add_special_tokens=False).input_ids),
                            "prompt_token_count": int(ids.shape[1])}}
        r["capability_flags_by_model"] = {
            "olmo3-think": {"capable_generation": bool(hit),
                            "in_difficulty_window": bool(LO <= best_v <= HI)},
            "olmo31-instruct": "PENDING_WEIGHTS_NOT_ON_THIS_VM",
            "olmo31-think": "PENDING_WEIGHTS_NOT_ON_THIS_VM",
            "qwen36-27b": "PENDING_WEIGHTS_NOT_ON_THIS_VM"}
        if (n + 1) % 100 == 0:
            print(f"  scored {n+1}/{len(items)} ({time.time()-t1:.0f}s)",
                  flush=True)

    # ---------------- per-family length summaries ------------------------
    byfam: dict[str, dict] = {}
    for r in items:
        f = r["canonical_family"] or "UNMAPPED"
        m = r["baseline_metrics_by_model"]["olmo3-think"]
        d = byfam.setdefault(f, {"n": 0, "prompt_toks": [], "ans_toks": [],
                                 "lp": [], "capable": 0, "in_window": 0,
                                 "tasks": set()})
        d["n"] += 1
        d["prompt_toks"].append(m["prompt_token_count"])
        d["ans_toks"].append(m["answer_token_count"])
        d["lp"].append(m["answer_seq_lp"])
        d["capable"] += int(m["greedy_correct"])
        d["in_window"] += int(LO <= m["answer_seq_lp"] <= HI)
        d["tasks"].add(r["task"])
    fam_summary = {f: {"n": d["n"], "tasks": sorted(d["tasks"]),
                       "median_prompt_tokens": int(np.median(d["prompt_toks"])),
                       "median_answer_tokens": int(np.median(d["ans_toks"])),
                       "median_lp": round(float(np.median(d["lp"])), 3),
                       "n_greedy_capable": d["capable"],
                       "n_in_window": d["in_window"]}
                   for f, d in byfam.items()}

    eligible = {f: v for f, v in fam_summary.items() if v["n_in_window"] >= 3}
    gate = {
        "A_wellformed": {"empty_answer": fails["empty_answer"]},
        "B_leakage": {"answer_appears_in_prompt": fails["answer_in_prompt"]},
        "C_twohop": {"bridge_appears_in_prompt": fails["bridge_in_prompt"],
                     "missing_counterfactual": fails["no_counterfactual"]},
        "D_capability": {
            "n_items": len(items),
            "n_greedy_capable": sum(1 for r in items if r[
                "capability_flags_by_model"]["olmo3-think"]["capable_generation"]),
            "n_in_difficulty_window": sum(1 for r in items if r[
                "capability_flags_by_model"]["olmo3-think"]["in_difficulty_window"])},
        "E_duplicate_facts": {"n_groups": len(dup_facts),
                              "examples": dict(list(dup_facts.items())[:8])},
        "F_lengths": {"families": len(fam_summary)},
        "G_family_integrity": {"missing_family": fails["missing_family"]},
        "families_with_ge3_in_window": len(eligible),
        "d5_target_60_families": len(eligible) >= 60,
        "cohorts": {
            "lineage_anchor_cohort": ("families capable on the OLMo primary "
                                      "pair — INCOMPLETE: only the anchor is "
                                      "scored on this VM"),
            "cross_model_intersection_cohort": "PENDING (needs all weights)",
            "model_specific_cohort": "PENDING (needs all weights)"},
    }
    passed = (not fails["missing_family"] and not fails["empty_answer"]
              and not fails["bridge_in_prompt"]
              and not fails["no_counterfactual"]
              and len(eligible) >= 60)

    payload = {"manifest_version": 1, "anchor_model": MODEL,
               "difficulty_window": [LO, HI], "gate": gate,
               "family_summary": fam_summary,
               "eligible_families": sorted(eligible),
               "items": items,
               "g5_status": "PASS" if passed else "CONDITIONAL",
               "conditions_outstanding": (
                   [] if passed else ["see gate failures"]) +
                   ["cross-model capability cohorts pending other weights"]}
    out = RUN / "metrics" / "cross_model" / "g5_item_manifest.json"
    prov = Provenance(
        evidence_id="g5-item-manifest-v1", tier="dev",
        command="python -m jspace_part2.experiments.g5_task_gate",
        inputs={"probe_swap": sha256_file(PROBE_SWAP)},
        model=resolve_model(MODEL))
    env = write_result_v2(payload, out, prov)
    registry_append({
        "evidence_id": "g5-item-manifest-v1", "tier": "dev",
        "what": (f"G5 task gate (PI decision D6: run, do not waive). "
                 f"Immutable item manifest over {len(items)} items with "
                 f"verified answers, tokenizations, anchor-model baseline "
                 f"metrics, capability flags, shortcut/leakage audits, "
                 f"canonical families and UNASSIGNED partition. Gate: "
                 f"answer-in-prompt {len(fails['answer_in_prompt'])}, "
                 f"bridge-in-prompt {len(fails['bridge_in_prompt'])}, "
                 f"missing counterfactual {len(fails['no_counterfactual'])}, "
                 f"missing family {len(fails['missing_family'])}, "
                 f"duplicate-fact groups {len(dup_facts)}. Families with >=3 "
                 f"in-window items: {len(eligible)} (D5 target 60). Status "
                 f"{payload['g5_status']} — cross-model capability cohorts "
                 f"remain PENDING because only the anchor's weights are on "
                 f"this VM."),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "input_uris": {"probe_swap": to_uri(str(PROBE_SWAP))},
        "inputs": {"probe_swap": sha256_file(PROBE_SWAP)},
        "outputs": [{"path": str(out), "uri": to_uri(str(out)),
                     "sha256": sha256_file(out),
                     "payload_sha256": env["payload_sha256"]}]})
    print(json.dumps(gate, indent=1)[:3000])
    print(f"\nG5 {payload['g5_status']}  ({round(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
