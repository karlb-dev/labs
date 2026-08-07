# N6 — THE CONFIRMATORY PROTECTED-DYNAMIC GRID (prereg §4, §7).
#
# Runs one complete model cell on the FROZEN confirmatory partition. Will
# not run unless the partition manifest exists (i.e. the freeze commit has
# happened) and the git tree is clean.
#
# Conditions (prereg §4):
#   baseline                 clean pass
#   meanJ_protected          PRIMARY — paper-faithful averaged-J,
#                            output-protected, v2 ablator
#   matched_control          PRIMARY CONTROL — dyn_energy_rank_matched_random,
#                            consumes the J arm's (rank, energy) profile on
#                            the same item
#   dynR_mechanics_control   isotropic random dictionary, protected
#   meanJ_unprotected        diagnostic only
#   logit_protected          prespecified secondary (output-aligned basis)
#
# ENDPOINT AGGREGATION OVER ALIASES (decided pre-freeze, recorded in the
# prereg §6): the primary answer-sequence logprob is
# logsumexp over the FROZEN accepted-alias set — total probability
# assigned to the answer concept, which cannot let each arm pick a
# different winning surface form. Canonical-answer lp and max-over-alias
# lp are stored per item as prespecified sensitivities (per-alias rows are
# in the parquet, so any aggregation is recomputable).
#
# Condition order (prereg §7): per-item seeded shuffle of
# [baseline, meanJ_protected, dynR, meanJ_unprotected, logit_protected],
# with matched_control ALWAYS immediately after meanJ_protected (it
# consumes the J arm's log — a deterministic mechanical profile, not an
# outcome).
#
# Stop rules implemented (prereg §7): baseline-capability check against
# the manifest's per-model scores (max-alias lp within tolerance) on the
# first 25 items; matched-control mechanical gates per item; sentinel
# determinism rerun at the end; protection invariants enforced by the v2
# ablator itself. Any violation aborts the cell with a STOP_RULE record.
#
# Usage: python -m jspace_part2.experiments.confirmatory_protected_grid \
#          --config configs/n6_grid_<slug>.yaml [--allow-dirty]
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..battery import seq_lp_from_logits
from ..dictionaries import build_j_dictionaries, build_logit_dictionary
from ..lib import sha256_file
from ..matched_control import (MatchedControlAblatorV2, profile_from_log)
from ..protected_dynamic_v2 import (ProtectedDynamicAblatorV2,
                                    protected_teacher_forced_v2)
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result_v2)

RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")
PARTITION = RUN / "metrics" / "cross_model" / "partition_manifest.json"
MANIFEST_V4 = RUN / "metrics" / "cross_model" / "g5_item_manifest_v5.json"  # BOS units (AMENDMENT 1)
BASELINE_TOL_NATS = 0.75
N_BASELINE_CHECK = 25


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_confirmatory_items(side: str) -> list[dict]:
    part = json.loads(PARTITION.read_text())["payload"]
    man = json.loads(MANIFEST_V4.read_text())["payload"]
    ids = set(part[side]["item_ids"])
    items = [r for r in man["items"] if r["item_id"] in ids
             and not r["excluded"]]
    missing = ids - {r["item_id"] for r in items}
    if missing:
        raise SystemExit(f"partition names {len(missing)} items absent from "
                         f"manifest v4, e.g. {sorted(missing)[:3]}")
    return items


def main():
    cfg_path = arg("--config")
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    git = require_clean_tree("--allow-dirty" in sys.argv)
    if not PARTITION.exists():
        raise SystemExit("REFUSING: partition manifest missing — the freeze "
                         "commit has not happened")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "n6_state.json"
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else {"done": {}, "rows": [], "mc_gate_rows": [],
                   "stop_rule_events": [], "baseline_checked": 0})

    import transformers
    import jlens
    from jlens import JacobianLens
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model_path"])
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.load(cfg["lens_path"])
    band = cfg["band"]
    k, pk = cfg["k"], cfg["protect_top_k"]
    slug = cfg["slug"]
    jd = build_j_dictionaries(hf, lens, band)
    ld = build_logit_dictionary(hf, band)
    V, d = jd[band[0]].shape
    g = torch.Generator().manual_seed(cfg["rand_seed"])
    rd_one = torch.nn.functional.normalize(
        torch.randn(V, d, generator=g), dim=1).to("cuda", torch.float16)
    rd = {l: rd_one for l in band}

    items = load_confirmatory_items(cfg.get("partition_side", "confirmatory"))
    items = [r for r in items if r["task"] in ("twohop", "onehop",
                                               "hard_onehop")]
    items.sort(key=lambda r: r["item_id"])
    log(f"{slug}: {len(items)} confirmatory items, band {band}, k={k}, "
        f"protect={pk}")

    def encode(text, max_length=512):
        return tok(text, return_tensors="pt", truncation=True,
                   max_length=max_length).input_ids.to("cuda")

    ab_j = ProtectedDynamicAblatorV2(model.layers, band)

    def score_arm(spec, prompt, aliases):
        """Return per-alias lp dict + intervention summary for one arm.

        AMENDMENT 1 tokenization convention: piecewise concatenation of the
        UN-rstripped prompt ids and the alias ids — exactly the frozen
        gate's estimand (g5 scoring), so the baseline stop rule compares
        like with like. The 5/325 trailing-whitespace items carry a
        double-space tokenization artifact that cancels in paired deltas.
        """
        prompt_ids = encode(prompt)
        n_prompt = prompt_ids.shape[1]
        per_alias, summary = {}, None
        clean_ranks = []
        for v in aliases:
            aid = tok(v, add_special_tokens=False,
                      return_tensors="pt").input_ids.to("cuda")
            full_ids = torch.cat([prompt_ids, aid], dim=1)

            def stub(_t, max_length=512, _f=full_ids):
                return _f
            if spec is None:
                ab_j.mode = None
                ids = full_ids
                logits = hf(input_ids=ids, use_cache=False).logits[0]\
                    .float().cpu()
                j_log = None
                # clean first-token answer rank at the last prompt position
                # (defines the HP3 protected-answer stratum: rank <= pk)
                vid = int(aid[0, 0])
                row_l = logits[n_prompt - 1]
                clean_ranks.append(int((row_l > row_l[vid]).sum()) + 1)
            elif spec["kind"] == "dyn":
                with ab_j:
                    ids, logits, _clean = protected_teacher_forced_v2(
                        hf, stub, ab_j, spec["dicts"], v, k=k,
                        protect=pk, protected=spec["protected"])
                j_log = ab_j.log
                ab_j.log = type(ab_j.log)()
            elif spec["kind"] == "matched":
                # J arm first (deterministic), then matched control
                from ..matched_control import teacher_forced_matched_pair_v2
                ids, _clean, _abl_j, logits, jl, cl = \
                    teacher_forced_matched_pair_v2(
                        hf, stub, model.layers, band, jd, v, k=k,
                        protect=pk,
                        seed_base=cfg["rand_seed"] + spec["item_index"])
                j_log = cl
            per_alias[v] = seq_lp_from_logits(ids, logits, n_prompt)
            if summary is None and j_log is not None:
                summary = (j_log.matched_summary()
                           if hasattr(j_log, "matched_summary")
                           else j_log.summary())
        lps = np.array(list(per_alias.values()))
        agg = {"lp_logsumexp": float(np.logaddexp.reduce(lps)),
               "lp_max": float(lps.max()),
               "clean_first_rank_min": (min(clean_ranks) if clean_ranks
                                        else None)}
        return per_alias, agg, summary

    conds = {
        "baseline": None,
        "meanJ_protected": {"kind": "dyn", "dicts": jd, "protected": True},
        "matched_control": {"kind": "matched"},
        "dynR_mechanics_control": {"kind": "dyn", "dicts": rd,
                                   "protected": True},
        "meanJ_unprotected": {"kind": "dyn", "dicts": jd, "protected": False},
        "logit_protected": {"kind": "dyn", "dicts": ld, "protected": True},
    }
    order_base = ["baseline", "meanJ_protected", "dynR_mechanics_control",
                  "meanJ_unprotected", "logit_protected"]

    t_start = time.time()
    n_done0 = len(state["done"])
    for idx, it in enumerate(items):
        iid = it["item_id"]
        if iid in state["done"]:
            continue
        aliases = it["accepted_answers"]
        canon = it["canonical_answer"]
        rng = np.random.default_rng(cfg["rand_seed"] * 100003 + idx)
        order = list(order_base)
        rng.shuffle(order)
        order.insert(order.index("meanJ_protected") + 1, "matched_control")

        item_rows = []
        for cname in order:
            spec = None if conds[cname] is None else dict(conds[cname])
            if spec is not None and spec.get("kind") == "matched":
                spec["item_index"] = idx
            per_alias, agg, summary = score_arm(spec, it["prompt"], aliases)
            canon_key = next((v for v in aliases
                              if v.strip() == canon.strip()), aliases[0])
            item_rows.append({
                "item_id": iid, "condition": cname, "task": it["task"],
                "canonical_family": it["canonical_family"],
                "relation_group": it.get("relation_group"),
                "cohorts": it.get("cohorts"),
                "lp_logsumexp": agg["lp_logsumexp"],
                "lp_max": agg["lp_max"],
                "lp_canonical": per_alias[canon_key],
                "clean_first_rank_min": agg.get("clean_first_rank_min"),
                "per_alias": per_alias,
                "intervention_summary": summary,
                "order_position": order.index(cname),
            })
            if cname == "matched_control" and summary and summary.get(
                    "n_positions", 0) > 0:
                mc_ok = (summary["rank_match_frac"] == 1.0
                         and (summary["energy_rel_err_max"] or 0) <= 0.05
                         and (summary.get("energy_abs_err_max_below_floor")
                              or 0) <= 1e-4
                         and summary["clamped_frac"] <= 0.01
                         and summary["max_protected_cos"] <= 1e-3)
                state["mc_gate_rows"].append(
                    {"item_id": iid, **summary, "ok": bool(mc_ok)})
                if not mc_ok:
                    state["stop_rule_events"].append(
                        {"rule": "matched_control_gate", "item_id": iid,
                         "summary": summary})
                    state_path.write_text(json.dumps(state))
                    raise SystemExit(f"STOP RULE: matched-control gate "
                                     f"failed on {iid}: {summary}")

        # ---- stop rule: baseline capability vs manifest (first N items)
        if state["baseline_checked"] < N_BASELINE_CHECK:
            man_lp = (it["baseline_metrics_by_model"].get(slug) or {})\
                .get("answer_seq_lp")
            here = next(r for r in item_rows if r["condition"] == "baseline")
            if man_lp is not None and \
                    abs(here["lp_max"] - man_lp) > BASELINE_TOL_NATS:
                state["stop_rule_events"].append(
                    {"rule": "baseline_capability", "item_id": iid,
                     "manifest_lp": man_lp, "measured_lp_max": here["lp_max"]})
                state_path.write_text(json.dumps(state))
                raise SystemExit(
                    f"STOP RULE: baseline mismatch on {iid}: manifest "
                    f"{man_lp} vs measured {here['lp_max']}")
            state["baseline_checked"] += 1

        state["rows"].extend(item_rows)
        state["done"][iid] = round(time.time() - t_start)
        if (len(state["done"]) - n_done0) % 10 == 0:
            state_path.write_text(json.dumps(state))
            rate = (time.time() - t_start) / max(len(state["done"]) - n_done0, 1)
            log(f"{len(state['done'])}/{len(items)} items "
                f"({rate:.1f}s/item, ETA "
                f"{rate*(len(items)-len(state['done']))/60:.0f}m)")
    state_path.write_text(json.dumps(state))

    # ---- prose NLL guard (N6.3): nonspecific-damage check per condition
    if "prose" not in state["done"]:
        from ..battery import prose_items
        prose = prose_items(cfg["prose_corpus"])
        for cname, spec0 in conds.items():
            if cname == "matched_control":
                continue          # profile-dependent; prose guard uses dyn arms
            for it in prose:
                spec = None if spec0 is None else dict(spec0)
                if spec is None:
                    ab_j.mode = None
                    ids = encode(it["text"], max_length=256)
                    logits = hf(input_ids=ids, use_cache=False).logits[0]\
                        .float().cpu()
                else:
                    with ab_j:
                        ids, logits, _c = protected_teacher_forced_v2(
                            hf, encode, ab_j, spec["dicts"], it["text"], k=k,
                            protect=pk, protected=spec["protected"],
                            max_length=256)
                    ab_j.log = type(ab_j.log)()
                lps = torch.log_softmax(logits[:-1], -1)
                tgt = ids[0, 1:].cpu()
                nll = float(-lps[torch.arange(len(tgt)), tgt].mean())
                state["rows"].append({
                    "item_id": it["item_id"], "condition": cname,
                    "task": "prose", "canonical_family": it["family"],
                    "relation_group": None, "cohorts": None,
                    "lp_logsumexp": -nll, "lp_max": -nll,
                    "lp_canonical": -nll, "per_alias": {},
                    "intervention_summary": None, "order_position": -1})
        state["done"]["prose"] = round(time.time() - t_start)
        state_path.write_text(json.dumps(state))
        log("prose guard done")

    # ---- sentinel determinism: rerun the first item's primary arm
    sent = items[0]
    _, agg1, _ = score_arm(dict(conds["meanJ_protected"]), sent["prompt"],
                           sent["accepted_answers"])
    orig = next(r for r in state["rows"]
                if r["item_id"] == sent["item_id"]
                and r["condition"] == "meanJ_protected")
    drift = abs(agg1["lp_logsumexp"] - orig["lp_logsumexp"])
    if drift > 1e-3:
        state["stop_rule_events"].append(
            {"rule": "sentinel_determinism", "drift_nats": drift})
        state_path.write_text(json.dumps(state))
        raise SystemExit(f"STOP RULE: sentinel rerun drifted {drift} nats")
    log(f"sentinel determinism ok (drift {drift:.2e})")

    # ---- bank
    import pandas as pd
    rows = [{k_: v for k_, v in r.items() if k_ != "per_alias"}
            | {"per_alias_json": json.dumps(r["per_alias"]),
               "intervention_summary_json":
                   json.dumps(r["intervention_summary"]),
               "cohorts_json": json.dumps(r["cohorts"])}
            for r in state["rows"]]
    df = pd.DataFrame(rows).drop(columns=["intervention_summary", "cohorts"],
                                 errors="ignore")
    pq = out_dir / f"n6_per_item_{slug}.parquet"
    df.to_parquet(pq)

    mc_all = state["mc_gate_rows"]
    payload = {
        "config": cfg, "n_items": len(items),
        "n_rows": len(state["rows"]),
        "conditions": list(conds.keys()),
        "endpoint_primary": "lp_logsumexp over frozen accepted aliases",
        "matched_control_gates": {
            "n": len(mc_all),
            "all_ok": all(r["ok"] for r in mc_all) if mc_all else None},
        "stop_rule_events": state["stop_rule_events"],
        "sentinel_drift_nats": drift,
        "partition_manifest_sha256": sha256_file(PARTITION),
    }
    prov = Provenance(
        evidence_id=cfg["evidence_id"], tier=cfg["tier"],
        command=(f"python -m jspace_part2.experiments."
                 f"confirmatory_protected_grid --config {cfg_path}"),
        config_path=cfg_path,
        inputs={"lens": sha256_file(cfg["lens_path"]),
                "partition": sha256_file(PARTITION),
                "manifest_v4": sha256_file(MANIFEST_V4)},
        model=resolve_model(cfg["model_path"]), seed=cfg["rand_seed"])
    out_json = out_dir / f"n6_summary_{slug}.json"
    write_result_v2(payload, out_json, prov)
    registry_append({
        "evidence_id": cfg["evidence_id"], "tier": cfg["tier"],
        "what": (f"N6 confirmatory protected-dynamic grid on {slug}: "
                 f"{len(items)} partition items x {len(conds)} conditions, "
                 f"primary endpoint logsumexp-alias full-seq lp, matched "
                 f"control gates "
                 f"{'ALL OK' if payload['matched_control_gates']['all_ok'] else 'VIOLATION'}, "
                 f"no aggregation performed during the run"),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(out_json), "sha256": sha256_file(out_json)},
                    {"path": str(pq), "sha256": sha256_file(pq)}]})
    log("cell banked and registered")


if __name__ == "__main__":
    main()
