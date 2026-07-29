# Freeze-blocking condition 1 — close the CROSS-MODEL CAPABILITY COHORTS.
#
# g5-item-manifest-v3 scored only the anchor (Olmo-3-32B-Think); every
# other confirmatory checkpoint reads PENDING_WEIGHTS_NOT_ON_THIS_VM. A
# bank selected on one model's difficulty smuggles a sampling bias into
# every cross-model claim, so the freeze requires per-model capability on:
#   olmo31-think     allenai/Olmo-3.1-32B-Think      (primary)
#   olmo31-instruct  allenai/Olmo-3.1-32B-Instruct   (primary)
#   qwen36-27b       Qwen/Qwen3.6-27B                (external validation)
#
# DESIGN DECISION (2026-07-29, VM9), stated once and used everywhere:
# the cohort predicate is `capable_generation` — greedy decoding produces
# an accepted alias (frozen alias set, MAX_NEW=8), exactly the anchor's
# G5 predicate. The [-9,-1] difficulty window was a BANK-BUILDING device
# on the anchor and is recorded per model (`in_difficulty_window`) for
# analysis, but it does not enter the cohort predicate: a cross-model
# cohort defined on per-model difficulty windows would re-select the bank
# per model, which is the bias this closes. Cohorts:
#   lineage_anchor_cohort        capable on BOTH primary checkpoints
#   cross_model_intersection     capable on both primaries AND Qwen
#   model_specific_cohort[m]     capable on model m only (within-model use)
#
# Two subcommands:
#   score:     score all non-excluded manifest items on ONE local model
#              (checkpoint-cached every 100 items, resumable)
#   assemble:  refuse unless all three models are scored; emit
#              g5-item-manifest-v4 superseding v3, cohorts assigned
#
# Usage:
#   python -m jspace_part2.experiments.g5_cohorts score \
#       --slug olmo31-think --model-path /content/models/olmo31-think
#   python -m jspace_part2.experiments.g5_cohorts assemble
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch

from ..battery import seq_lp_from_logits
from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result_v2)
from .. import registry as reg

RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")
MANIFEST_V3 = RUN / "metrics" / "cross_model" / "g5_item_manifest.json"
LO, HI = -9.0, -1.0
MAX_NEW = 8
MODELS = ["olmo31-think", "olmo31-instruct", "qwen36-27b"]


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def load_manifest_items() -> tuple[dict, list[dict]]:
    env = json.loads(MANIFEST_V3.read_text())
    payload = env["payload"]
    return payload, [r for r in payload["items"] if not r["excluded"]]


def score_path(slug: str) -> Path:
    sfx = "_bos" if "--bos" in sys.argv else ""
    return RUN / "metrics" / slug / f"g5_capability_scores{sfx}.json"


def cmd_score():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--slug")
    model_path = arg("--model-path")
    if slug not in MODELS:
        raise SystemExit(f"slug {slug!r} not in {MODELS}")
    payload, items = load_manifest_items()
    out_p = score_path(slug)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    scores = (json.loads(out_p.read_text())
              if out_p.exists() else {"model": slug, "items": {}})
    done = scores["items"]

    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    if "--bos" in sys.argv:            # AMENDMENT 1: assay-wide BOS units
        tok.add_bos_token = True
        probe = tok("probe").input_ids
        assert probe and probe[0] == (tok.bos_token_id
                                      if tok.bos_token_id is not None
                                      else probe[0]), "BOS not applied"
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()

    t0, n_new = time.time(), 0
    for n, r in enumerate(items):
        iid = r["item_id"]
        if iid in done:
            continue
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
                              do_sample=False, pad_token_id=tok.eos_token_id)
        out = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        hit = any(norm(v) and norm(v) in norm(out)
                  for v in r["accepted_answers"])
        done[iid] = {"answer_seq_lp": round(best_v, 4), "best_variant": best,
                     "greedy_continuation": out[:60],
                     "greedy_correct": bool(hit),
                     "capable_generation": bool(hit),
                     "in_difficulty_window": bool(LO <= best_v <= HI),
                     "answer_token_count": len(tok(
                         best, add_special_tokens=False).input_ids),
                     "prompt_token_count": int(ids.shape[1])}
        n_new += 1
        if n_new % 100 == 0:
            out_p.write_text(json.dumps(scores))
            log(f"  {len(done)}/{len(items)} scored "
                f"({(time.time()-t0)/max(n_new,1):.2f}s/item)")
    scores["model_resolved"] = resolve_model(model_path)
    scores["code_commit"] = git["code_commit"]
    out_p.write_text(json.dumps(scores))
    cap = sum(1 for v in done.values() if v["capable_generation"])
    log(f"{slug}: {len(done)} items scored, {cap} capable_generation "
        f"({time.time()-t0:.0f}s)")
    sfx = "-bos-v2" if "--bos" in sys.argv else "-v1"
    registry_append({
        "evidence_id": f"g5-capability-scores-{slug}{sfx}", "tier": "dev",
        "what": (f"Cross-model capability scoring for the G5 bank on {slug}: "
                 f"{len(done)} non-excluded items, {cap} capable_generation "
                 f"under the frozen alias set (predicate for the capability "
                 f"cohorts; per-model difficulty window recorded, not part "
                 f"of the predicate)"),
        "command": (f"python -m jspace_part2.experiments.g5_cohorts score "
                    f"--slug {slug} --model-path {model_path}"
                    + (" --bos" if "--bos" in sys.argv else "")),
        "code_commit": git["code_commit"], "rerun": "auto",
        "outputs": [{"path": str(out_p), "sha256": sha256_file(out_p)}]})
    if "--bos" in sys.argv:
        try:
            reg.supersede(f"g5-capability-scores-{slug}-v1",
                          f"g5-capability-scores-{slug}-bos-v2",
                          reason="AMENDMENT 1: rescored in assay-wide BOS "
                                 "units (from_hf force_bos)")
        except Exception as ex:
            log(f"  (supersede: {ex})")


def cmd_assemble():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    payload, items = load_manifest_items()
    per_model = {}
    for slug in MODELS:
        p = score_path(slug)
        if not p.exists():
            raise SystemExit(f"REFUSING to assemble: {slug} not scored "
                             f"({p} missing)")
        per_model[slug] = json.loads(p.read_text())["items"]
        missing = [r["item_id"] for r in items
                   if r["item_id"] not in per_model[slug]]
        if missing:
            raise SystemExit(f"REFUSING to assemble: {slug} missing "
                             f"{len(missing)} items, e.g. {missing[:3]}")

    coh_counts = {"lineage_anchor": 0, "cross_model_intersection": 0,
                  **{f"model_specific:{m}": 0 for m in MODELS}}
    fam_int: dict[str, set] = {}
    for r in payload["items"]:
        if r["excluded"]:
            continue
        iid = r["item_id"]
        for slug in MODELS:
            r["baseline_metrics_by_model"][slug] = {
                k: v for k, v in per_model[slug][iid].items()
                if k not in ("capable_generation", "in_difficulty_window")}
            r["capability_flags_by_model"][slug] = {
                "capable_generation": per_model[slug][iid]["capable_generation"],
                "in_difficulty_window": per_model[slug][iid]["in_difficulty_window"]}
        cap = {m: r["capability_flags_by_model"][m]["capable_generation"]
               for m in MODELS}
        r["cohorts"] = {
            "lineage_anchor": cap["olmo31-think"] and cap["olmo31-instruct"],
            "cross_model_intersection": all(cap.values()),
            "model_specific": [m for m in MODELS if cap[m]],
        }
        for key in ("lineage_anchor", "cross_model_intersection"):
            coh_counts[key] += int(r["cohorts"][key])
        for m in MODELS:
            coh_counts[f"model_specific:{m}"] += int(cap[m])
        if r["cohorts"]["cross_model_intersection"] and r["canonical_family"]:
            fam_int.setdefault(r["canonical_family"], set()).add(iid)

    fam3 = {f: len(s) for f, s in fam_int.items() if len(s) >= 3}
    payload["manifest_version"] = ver
    if bos:
        payload["units"] = "BOS (AMENDMENT_1_BOS_UNITS)"
    payload["cohort_predicate"] = ("capable_generation (greedy, frozen alias "
                                   "set, MAX_NEW=8); difficulty window "
                                   "recorded per model, not in predicate")
    payload["cohort_counts"] = coh_counts
    payload["cross_model_intersection_families_ge3"] = len(fam3)
    payload["conditions_outstanding"] = [
        c for c in payload.get("conditions_outstanding", [])
        if "capability cohorts" not in c]

    bos = "--bos" in sys.argv
    ver = 5 if bos else 4
    out = RUN / "metrics" / "cross_model" / f"g5_item_manifest_v{ver}.json"
    prov = Provenance(
        evidence_id=f"g5-item-manifest-v{ver}", tier="dev",
        command="python -m jspace_part2.experiments.g5_cohorts assemble",
        inputs={"manifest_v3": sha256_file(MANIFEST_V3),
                **{f"scores_{m}": sha256_file(score_path(m))
                   for m in MODELS}},
        model={"note": "assembly step; per-model resolution in score files"},
        seed=0)
    write_result_v2(payload, out, prov)
    registry_append({
        "evidence_id": f"g5-item-manifest-v{ver}", "tier": "dev",
        "what": (f"Item manifest with CLOSED cross-model capability cohorts "
                 f"(freeze condition 1): lineage_anchor "
                 f"{coh_counts['lineage_anchor']}, cross_model_intersection "
                 f"{coh_counts['cross_model_intersection']} items "
                 f"({len(fam3)} families with >=3 intersection items); "
                 f"predicate = capable_generation on the frozen alias set"),
        "command": ("python -m jspace_part2.experiments.g5_cohorts assemble"
                    + (" --bos" if bos else "")),
        "code_commit": git["code_commit"], "rerun": "auto",
        "outputs": [{"path": str(out), "sha256": sha256_file(out)}]})
    try:
        old = f"g5-item-manifest-v{ver-1}"
        reg.supersede(old, f"g5-item-manifest-v{ver}",
                      reason=("AMENDMENT 1: cohorts in BOS units" if bos else
                              "capability cohorts closed on all three "
                              "confirmatory models"))
    except Exception as ex:
        log(f"  (supersede: {ex})")
    log(f"manifest v4 written: {coh_counts}")


if __name__ == "__main__":
    if "score" in sys.argv[1:2]:
        cmd_score()
    elif "assemble" in sys.argv[1:2]:
        cmd_assemble()
    else:
        raise SystemExit("usage: g5_cohorts score|assemble ...")
