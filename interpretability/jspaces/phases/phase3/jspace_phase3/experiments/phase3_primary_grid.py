# Block B — the Phase 3 PRIMARY confirmatory grid (one model per run).
#
# Runs only AFTER the freeze commit: it loads the FROZEN partition and
# refuses to start without `freeze_authorised: true` in the partition
# payload. Discipline mirrors Phase 2's N6:
#   * per-item randomized condition order (seeded per item);
#   * §7-style baseline stop rule: the measured baseline lp_canonical of
#     the first `stop_check_n` items must match the G5 manifest value
#     within tolerance BEFORE any intervention outcome is produced;
#   * per-item checkpoint every 5 items, same-command resume;
#   * NO aggregation is computed or printed here — the locked analysis
#     runs once, from raw parquets, after ALL cells bank.
#
# Conditions (prereg candidate §2):
#   baseline
#   meanJ_span_safe                      (primary arm)
#   ss_matched  = instant rank+energy matched consuming the SPAN-SAFE
#                 profile (primary comparator)
#   meanJ_label_protected                (secondary, Phase 2 continuity)
#   prot_energy_matched                  (leakage comparator, label profile)
#   mechanics_random · logit_label_protected
#   + on composed items of the P3-P3 model only:
#     true_bridge · distractor_bridge    (span-safe base, §6.5)
#
# Usage:
#   python -m jspace_phase3.experiments.phase3_primary_grid \
#       --config interpretability/jspaces/phases/phase3/configs/p3_grid_<slug>.yaml
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from jspace_part2.dictionaries import build_j_dictionaries, build_logit_dictionary
from jspace_part2.lib import sha256_file
from jspace_part2.paths import resolve as resolve_uri
from ..ablator3 import (Phase3JAblator, profile_from_p3log,
                        teacher_forced_matched_arm)
from ..bank import load_bank
from ..gpu import require_cuda_gpu
from ..paths3 import metrics_dir, resolve_uri as resolve3
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..scoring import DEFAULT_SPEC, ScoringSession
from ..seeds import stable_seed

DEFAULT_TIER = "phase3-confirmatory"
REPO_DATA = Path(__file__).resolve().parents[2] / "data"


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    os.replace(tmp, path)


def bridge_piece_ids(tok, bridge: str) -> torch.Tensor:
    b = bridge.removeprefix("the ").removeprefix("The ").strip()
    ids = set()
    for v in {f" {b}", b, f" {b.lower()}", f" {b.title()}"}:
        for t in tok(v, add_special_tokens=False).input_ids:
            ids.add(int(t))
    return torch.tensor(sorted(ids), dtype=torch.long)


@torch.no_grad()
def main():  # noqa: C901
    cfg_path = arg("--config")
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    require_clean_tree("--allow-dirty" in sys.argv)
    slug = cfg["slug"]
    tier = cfg.get("tier", DEFAULT_TIER)

    puri = cfg["partition_uri"]
    ppath = Path(resolve3(puri)) if "://" in str(puri) else Path(puri)
    part = json.loads(ppath.read_text())
    payload = part.get("payload", part)
    if not payload.get("freeze_authorised"):
        raise RuntimeError("partition is not a freeze artifact "
                           "(freeze_authorised missing) — the primary "
                           "grid runs only after jspace-phase3-freeze-v1")
    fams = set(payload[cfg.get("partition_side", "confirmatory")])

    side = cfg.get("partition_side", "confirmatory")
    suffix = "" if side == "confirmatory" else f"_{side}"
    out_dir = metrics_dir(slug) / f"p3_grid{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else {"done": {}, "rows": [], "baseline_checked": 0,
                   "stop_events": []})

    gpu = require_cuda_gpu()
    log("GPU gate PASS: " + json.dumps(gpu, sort_keys=True))
    state["gpu"] = gpu
    atomic_json(state_path, state)

    import transformers
    import jlens
    from jlens import JacobianLens
    model_path = str(resolve_uri(cfg["model_uri"], must_exist=True))
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    if not next(hf.parameters()).is_cuda:
        raise RuntimeError("model parameters are not on CUDA")
    model = jlens.from_hf(hf, tok)
    sess = ScoringSession(tok, DEFAULT_SPEC, device="cuda")
    lens = JacobianLens.load(str(resolve_uri(cfg["lens_uri"])))
    band, k, pk = cfg["band"], cfg["k"], cfg["protect_top_k"]
    jd = build_j_dictionaries(hf, lens, band)
    ld = build_logit_dictionary(hf, band)
    V, d = jd[band[0]].shape
    gtorch = torch.Generator().manual_seed(cfg["rand_seed"])
    rd_one = torch.nn.functional.normalize(
        torch.randn(V, d, generator=gtorch), dim=1).to("cuda", torch.float16)
    rd = {l: rd_one for l in band}
    ab = Phase3JAblator(model.layers, band)

    # cohort: G5 fact-level capability (direct AND composed) on THIS model
    gd = metrics_dir(slug) / "g5_bank"
    gp = gd / f"g5_bank_{slug}_regraded.parquet"
    g5 = pd.read_parquet(gp if gp.exists()
                         else gd / f"g5_bank_{slug}.parquet")
    dc = g5[g5.variant.isin(["direct", "composed"])]
    cap_facts = {fid for fid, sub in dc.groupby("fact_id")
                 if len(sub) == 2 and bool(sub.capable_generation.all())}
    manifest_lp = {r.item_id: r.lp_canonical for r in g5.itertuples()}

    bundles = [b for bank in cfg["banks"] for b in
               load_bank(REPO_DATA / bank)]
    items = []
    for b in bundles:
        if b.canonical_family not in fams or b.fact_id not in cap_facts:
            continue
        for it in b.as_items():
            if it["variant"] in ("direct", "composed"):
                items.append(it)
    items.sort(key=lambda r: r["item_id"])
    requested_ids = set(cfg.get("item_ids", []))
    if requested_ids:
        available_ids = {item["item_id"] for item in items}
        missing_ids = requested_ids - available_ids
        if missing_ids:
            raise RuntimeError(
                "configured item_ids are outside the frozen model cohort: "
                f"{sorted(missing_ids)}")
        items = [
            item for item in items if item["item_id"] in requested_ids]
    p3p3 = bool(cfg.get("bridge_arms", False))
    log(f"{slug}: {len(items)} frozen-cohort items "
        f"({len({i['fact_id'] for i in items})} facts), bridge_arms={p3p3}")

    stop_n = int(cfg.get("stop_check_n", 25))
    stop_tol = float(cfg.get("stop_tol", 0.05))

    def j_arm(ids, psets, span_safe, record=True):
        ab.log = type(ab.log)()
        ab.phase, ab.forward_index = "prefill", 0
        ab.mode = {"dicts": jd, "k": k, "nonneg": True,
                   "protect_sets": psets, "active_phases": {"prefill"},
                   "span_safe": span_safe, "record_overlap": record,
                   "answer_id": None}
        with ab:
            out = hf(input_ids=ids, use_cache=False).logits[0].float()
        ab.mode = None
        return out, ab.log

    def dict_arm(ids, psets, dicts):
        ab.log = type(ab.log)()
        ab.phase, ab.forward_index = "prefill", 0
        ab.mode = {"dicts": dicts, "k": k, "nonneg": True,
                   "protect_sets": psets, "active_phases": {"prefill"},
                   "span_safe": False, "record_overlap": False,
                   "answer_id": None}
        with ab:
            out = hf(input_ids=ids, use_cache=False).logits[0].float()
        ab.mode = None
        return out

    t0 = time.time()
    n0 = len(state["done"])
    for it in items:
        iid = it["item_id"]
        if iid in state["done"]:
            continue
        alias = it["accepted_answers"][0]
        full, n_p = sess.full_ids(it["prompt"], alias)
        T = full.shape[1]

        ab.mode = None
        clean = hf(input_ids=full, use_cache=False).logits[0].float()
        psets = clean.topk(pk, dim=-1).indices
        lp_base = sess.answer_seq_lp(full, clean.cpu(), n_p)

        # §7 stop rule BEFORE any intervention outcome exists
        if state["baseline_checked"] < stop_n:
            man = manifest_lp.get(iid)
            if man is None or abs(lp_base - man) > stop_tol:
                state["stop_events"].append(
                    {"item_id": iid, "measured": lp_base,
                     "manifest": man})
                atomic_json(state_path, state)
                raise RuntimeError(
                    f"BASELINE STOP RULE: {iid} measured {lp_base:.4f} "
                    f"vs manifest {man} (tol {stop_tol}) — instrument "
                    f"mismatch; NO outcome viewed")
            state["baseline_checked"] += 1

        if cfg.get("reduced_arms"):
            conds = ["meanJ_span_safe", "meanJ_label_protected"]
        else:
            conds = ["meanJ_span_safe", "meanJ_label_protected",
                     "mechanics_random", "logit_label_protected"]
        if p3p3 and it["variant"] == "composed" \
                and it.get("counterfactual_bridge"):
            conds += ["true_bridge", "distractor_bridge"]
        order = np.random.default_rng(stable_seed(
            "phase3-primary-condition-order", iid, cfg["rand_seed"]))
        conds = [conds[i] for i in order.permutation(len(conds))]

        row = {"item_id": iid, "fact_id": it["fact_id"],
               "variant": it["variant"], "bank": it["bank"],
               "canonical_family": it["canonical_family"],
               "relation_group": it["relation_group"],
               "lp_baseline": lp_base, "n_tokens": int(T)}
        if p3p3 and it["variant"] == "composed" \
                and it.get("counterfactual_bridge"):
            for name, entity in (
                    ("true", it["bridge_entity"]),
                    ("distractor", it["counterfactual_bridge"])):
                pieces = bridge_piece_ids(tok, entity).to(psets.device)
                overlap = torch.isin(psets, pieces).sum(dim=1).float()
                row[f"{name}_bridge_piece_count"] = int(len(pieces))
                row[f"{name}_bridge_clean_topk_overlap_mean"] = float(
                    overlap.mean().cpu())
                row[f"{name}_bridge_added_rank_mean"] = float(
                    (len(pieces) - overlap).mean().cpu())
        profiles = {}
        for cond in conds:
            if cond == "meanJ_span_safe":
                abl, jlog = j_arm(full, psets, True)
                profiles["ss"] = profile_from_p3log(
                    jlog, overlap_records=jlog.overlap)
                row["overlap_ss_json"] = json.dumps(jlog.overlap_summary())
            elif cond == "meanJ_label_protected":
                abl, jlog = j_arm(full, psets, False)
                profiles["label"] = profile_from_p3log(
                    jlog, overlap_records=jlog.overlap)
            elif cond == "mechanics_random":
                abl = dict_arm(full, psets, rd)
            elif cond == "logit_label_protected":
                abl = dict_arm(full, psets, ld)
            elif cond in ("true_bridge", "distractor_bridge"):
                ent = it["bridge_entity"] if cond == "true_bridge" \
                    else it["counterfactual_bridge"]
                bt = bridge_piece_ids(tok, ent).to(psets.device)
                ps2 = torch.cat([psets, bt.unsqueeze(0).expand(T, -1)],
                                dim=1)
                abl, _ = j_arm(full, ps2, True, record=False)
            row[f"lp_{cond}"] = sess.answer_seq_lp(full, abl.cpu(), n_p)

        # matched controls consume their source-arm profiles (§14.1's
        # C_effect uses the primary comparator = span-safe profile)
        matched_specs = [("instant_rank_energy_matched",
                          "lp_ss_matched", "ss")]
        if not cfg.get("reduced_arms"):
            matched_specs.append(("prot_energy_matched",
                                  "lp_prot_energy_matched", "label"))
        for variant, key, src in matched_specs:
            default_namespace = f"phase3-primary-{variant}"
            namespace = cfg.get(
                "matched_seed_namespaces", {}).get(
                    variant, default_namespace)
            logits, matched_log = teacher_forced_matched_arm(
                hf, model.layers, band, jd, full, profiles[src],
                variant=variant, protect_sets=psets,
                seed_base=stable_seed(
                    namespace, iid, cfg["rand_seed"]))
            row[key] = sess.answer_seq_lp(full, logits, n_p)
            row[f"{key}_summary_json"] = json.dumps(
                matched_log.matched_summary(), sort_keys=True)

        state["rows"].append(row)
        state["done"][iid] = round(time.time() - t0)
        if (len(state["done"]) - n0) % 5 == 0:
            atomic_json(state_path, state)
            rate = (time.time() - t0) / max(len(state["done"]) - n0, 1)
            log(f"{len(state['done'])}/{len(items)} ({rate:.1f}s/item, "
                f"ETA {(len(items) - len(state['done'])) * rate / 60:.0f}m)")
    atomic_json(state_path, state)

    df = pd.DataFrame(state["rows"])
    pq = out_dir / f"p3_grid{suffix}_{slug}.parquet"
    df.to_parquet(pq)
    eid = cfg["evidence_id"]
    cmd = (f"python -m jspace_phase3.experiments.phase3_primary_grid "
           f"--config {cfg_path}"
           + (" --no-register" if "--no-register" in sys.argv else ""))
    out_json = out_dir / f"p3_grid{suffix}_{slug}.json"
    write_result3({"n_items": int(len(df)),
                   "n_facts": int(df.fact_id.nunique()),
                   "n_families": int(df.canonical_family.nunique()),
                   "baseline_checked": state["baseline_checked"],
                   "conditions_note": "raw rows only; locked analysis "
                                      "runs once after all cells bank"},
                  out_json, Provenance3(
                      evidence_id=eid, tier=tier, command=cmd,
                      config_path=cfg_path,
                      inputs={"lens": sha256_file(
                          str(resolve_uri(cfg["lens_uri"])))},
                      model=resolve_model(model_path),
                      seed=cfg["rand_seed"]))
    if "--no-register" not in sys.argv:
        register(eid, tier=tier, command=cmd,
                 what=(f"Phase 3 primary grid cell on {slug}: {len(df)} "
                       f"frozen-cohort items × span-safe primary arm set; "
                       f"stop rule passed on {state['baseline_checked']} "
                       f"baselines; NO aggregation viewed"),
                 outputs=[out_json, pq])
    log(f"CELL BANKED: {len(df)} items")


if __name__ == "__main__":
    main()
