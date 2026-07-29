# §4.1b — THE PROTECTED-SPAN GEOMETRY AUDIT (nextsteps §2.3, first GPU
# job of Phase 3).
#
# What §4.1a could not answer: label protection guarantees a protected row
# is never SELECTED, not that the selected span is orthogonal to the
# protected rows. This measures the geometry directly, per (item, layer,
# position), on the FROZEN Phase 2 items with each model's own primary
# lens — read-only with respect to Phase 2 artifacts, new evidence id:
#
#   * principal cosines between the selected J span and protected span,
#     and the projector overlap trace(P_J P_prot);
#   * survival of every protected row under the J projector;
#   * survival of the ANSWER token direction specifically;
#   * fraction of the REMOVED activation energy that lay inside the
#     protected span;
#   * span-safe rank loss and null-row fraction, i.e. how much of the
#     nominal selection lives inside the protected span;
#   * the same quantities for the span-safe arm (must be ~0 by
#     construction — the audit's own positive control).
#
# It also scores both arms on the primary endpoint, so the decisive
# comparison (does the tail survive span-safe protection?) is available
# on Phase 2 items at DEVELOPMENT tier before the thick bank exists.
#
# Usage:
#   python -m jspace_phase3.experiments.span_audit --config <cfg.yaml>
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from jspace_part2.dictionaries import build_j_dictionaries
from jspace_part2.lib import sha256_file
from jspace_part2.paths import resolve as resolve_uri
from ..ablator3 import (Phase3JAblator, profile_from_p3log,
                        teacher_forced_matched_arm)
from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..scoring import LEGACY_PHASE2_SPEC, ScoringSession

TIER = "phase3-development"


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_items(cfg) -> list[dict]:
    """Phase 2 frozen items, read-only, via logical URIs."""
    man = json.loads(resolve_uri(cfg["manifest_uri"]).read_text())["payload"]
    part = json.loads(resolve_uri(cfg["partition_uri"]).read_text())["payload"]
    ids = set(part[cfg.get("partition_side", "confirmatory")]["item_ids"])
    items = [r for r in man["items"] if r["item_id"] in ids
             and not r["excluded"]
             and r["task"] in ("twohop", "onehop", "hard_onehop")]
    items.sort(key=lambda r: r["item_id"])
    n = cfg.get("n_items")
    if n:
        # deterministic stratified subsample: every task, then round-robin
        by_task: dict = {}
        for r in items:
            by_task.setdefault(r["task"], []).append(r)
        picked, i = [], 0
        while len(picked) < min(n, len(items)):
            added = False
            for t in sorted(by_task):
                if i < len(by_task[t]) and len(picked) < n:
                    picked.append(by_task[t][i])
                    added = True
            if not added:
                break
            i += 1
        items = sorted(picked, key=lambda r: r["item_id"])
    return items


def main():  # noqa: C901
    cfg_path = arg("--config")
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    git = require_clean_tree("--allow-dirty" in sys.argv)
    slug = cfg["slug"]
    out_dir = metrics_dir(slug) / "span_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else {"done": {}, "rows": [], "overlap": []})

    import transformers
    import jlens
    from jlens import JacobianLens

    model_path = str(resolve_uri(cfg["model_uri"], must_exist=True))
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)          # mutates tok -> BOS units
    # Frozen Phase 2 items are scored under the LEGACY spec: Amendment 1
    # used the un-rstripped prompt and 5/325 bank items carry a
    # trailing-space artifact. Rejecting them (the Phase 3 default) would
    # change the estimand of the very items being audited. Count reported.
    sess = ScoringSession(tok, LEGACY_PHASE2_SPEC, device="cuda")
    lens_path = str(resolve_uri(cfg["lens_uri"]))
    lens = JacobianLens.load(lens_path)
    band, k, pk = cfg["band"], cfg["k"], cfg["protect_top_k"]
    jd = build_j_dictionaries(hf, lens, band)

    items = load_items(cfg)
    log(f"{slug}: {len(items)} items, band {band}, k={k}, protect={pk}")

    ab = Phase3JAblator(model.layers, band)
    t0 = time.time()
    n0 = len(state["done"])

    for it in items:
        iid = it["item_id"]
        if iid in state["done"]:
            continue
        alias = it["accepted_answers"][0]
        full_ids, n_prompt = sess.full_ids(it["prompt"], alias)
        answer_id = int(sess.answer_ids(alias)[0, 0])

        # ---- clean
        ab.mode = None
        clean = hf(input_ids=full_ids, use_cache=False).logits[0]
        lp_base = sess.answer_seq_lp(full_ids, clean.float().cpu(), n_prompt)
        psets = clean.topk(pk, dim=-1).indices
        row_l = clean[n_prompt - 1].float().cpu()
        clean_rank = int((row_l > row_l[answer_id]).sum()) + 1

        arm_lp, arm_overlap, arm_profile = {}, {}, {}
        for arm, span_safe in (("meanJ_label_protected", False),
                               ("meanJ_span_safe", True)):
            ab.log = type(ab.log)()
            ab.phase, ab.forward_index = "prefill", 0
            ab.mode = {"dicts": jd, "k": k, "nonneg": True,
                       "protect_sets": psets, "active_phases": {"prefill"},
                       "span_safe": span_safe, "record_overlap": True,
                       "answer_id": answer_id}
            with ab:
                abl = hf(input_ids=full_ids, use_cache=False).logits[0]
            ab.mode = None
            arm_lp[arm] = sess.answer_seq_lp(full_ids, abl.float().cpu(),
                                             n_prompt)
            arm_overlap[arm] = ab.log.overlap_summary()
            arm_profile[arm] = profile_from_p3log(
                ab.log, overlap_records=ab.log.overlap)
            for r in ab.log.overlap:
                if r.layer in cfg.get("overlap_detail_layers", []):
                    state["overlap"].append(
                        {"item_id": iid, "arm": arm, "layer": r.layer,
                         "position": r.position,
                         "rank_selected": r.rank_selected,
                         "rank_protected": r.rank_protected,
                         "projector_overlap": r.projector_overlap,
                         "overlap_normalized": r.overlap_normalized,
                         "answer_dir_survival": r.answer_dir_survival,
                         "removed_energy_in_prot_frac":
                             r.removed_energy_in_prot_frac,
                         "lost_rank": r.lost_rank,
                         "null_row_frac": r.null_row_frac})

        # ---- controls, each consuming ITS OWN arm's profile
        ctl_lp, ctl_sum = {}, {}
        for variant, src_arm in (
                ("instant_rank_energy_matched", "meanJ_label_protected"),
                ("overlap_matched", "meanJ_label_protected"),
                ("persistent_matched", "meanJ_label_protected"),
                ("instant_rank_energy_matched_vs_span_safe",
                 "meanJ_span_safe")):
            v = variant.replace("_vs_span_safe", "")
            logits, mlog = teacher_forced_matched_arm(
                hf, model.layers, band, jd, full_ids, arm_profile[src_arm],
                variant=v, protect_sets=psets,
                seed_base=cfg["rand_seed"] + abs(hash(iid)) % 10_000)
            ctl_lp[variant] = sess.answer_seq_lp(
                full_ids, logits, n_prompt)
            ctl_sum[variant] = mlog.matched_summary()

        state["rows"].append({
            "item_id": iid, "task": it["task"],
            "canonical_family": it["canonical_family"],
            "relation_group": it.get("relation_group"),
            "clean_first_rank": clean_rank,
            "protected_answer": clean_rank <= pk,
            "lp_baseline": lp_base,
            **{f"lp_{a}": v for a, v in arm_lp.items()},
            **{f"lp_{c}": v for c, v in ctl_lp.items()},
            "overlap_label_json": json.dumps(
                arm_overlap["meanJ_label_protected"]),
            "overlap_spansafe_json": json.dumps(
                arm_overlap["meanJ_span_safe"]),
            "control_summaries_json": json.dumps(ctl_sum),
        })
        state["done"][iid] = round(time.time() - t0)
        if (len(state["done"]) - n0) % 5 == 0:
            state_path.write_text(json.dumps(state))
            rate = (time.time() - t0) / max(len(state["done"]) - n0, 1)
            log(f"{len(state['done'])}/{len(items)} ({rate:.1f}s/item, "
                f"ETA {rate * (len(items) - len(state['done'])) / 60:.0f}m)")
    state_path.write_text(json.dumps(state))

    df = pd.DataFrame(state["rows"])
    ov = pd.DataFrame(state["overlap"])
    pq = out_dir / f"span_audit_items_{slug}.parquet"
    pq_ov = out_dir / f"span_audit_overlap_{slug}.parquet"
    df.to_parquet(pq)
    ov.to_parquet(pq_ov)

    # ---- headline summary (development tier: descriptive, no test)
    def d(col):
        return df[col] - df.lp_baseline

    dl = d("lp_meanJ_label_protected")
    ds = d("lp_meanJ_span_safe")
    dm = d("lp_instant_rank_energy_matched")
    dms = d("lp_instant_rank_energy_matched_vs_span_safe")
    summary = {
        "n_items": int(len(df)),
        "n_families": int(df.canonical_family.nunique()),
        "scoring_spec": "LEGACY_PHASE2 (un-rstripped prompts)",
        "n_items_with_trailing_whitespace": int(sum(
            1 for it in items if it["prompt"] != it["prompt"].rstrip())),
        "delta_label_mean": round(float(dl.mean()), 4),
        "delta_span_safe_mean": round(float(ds.mean()), 4),
        "delta_matched_mean": round(float(dm.mean()), 4),
        "delta_matched_vs_span_safe_mean": round(float(dms.mean()), 4),
        "delta_overlap_matched_mean": round(
            float(d("lp_overlap_matched").mean()), 4),
        "delta_persistent_matched_mean": round(
            float(d("lp_persistent_matched").mean()), 4),
        "tail_rate_label": round(float((dl < -1.0).mean()), 4),
        "tail_rate_span_safe": round(float((ds < -1.0).mean()), 4),
        "tail_rate_matched": round(float((dm < -1.0).mean()), 4),
        "specific_tail_label": round(
            float(((dl < -1.0).astype(int) - (dm < -1.0).astype(int)).mean()), 4),
        "specific_tail_span_safe": round(
            float(((ds < -1.0).astype(int) - (dms < -1.0).astype(int)).mean()), 4),
        "per_item_delta_corr_label_vs_span_safe": round(
            float(np.corrcoef(dl, ds)[0, 1]), 4),
        "overlap": {
            arm: {
                key: round(float(pd.Series(
                    [json.loads(r)[key] for r in df[col]
                     if json.loads(r).get(key) is not None]).mean()), 6)
                for key in ("projector_overlap_mean", "overlap_normalized_mean",
                            "answer_dir_survival_mean", "lost_rank_mean",
                            "removed_energy_in_prot_frac_mean")}
            for arm, col in (("label_protected", "overlap_label_json"),
                             ("span_safe", "overlap_spansafe_json"))},
    }
    payload = {"config": cfg, "summary": summary,
               "conditions": ["baseline", "meanJ_label_protected",
                              "meanJ_span_safe",
                              "instant_rank_energy_matched",
                              "overlap_matched", "persistent_matched",
                              "instant_rank_energy_matched_vs_span_safe"]}
    cmd = (f"python -m jspace_phase3.experiments.span_audit "
           f"--config {cfg_path}")
    prov = Provenance3(
        evidence_id=cfg["evidence_id"], tier=TIER, command=cmd,
        config_path=cfg_path,
        inputs={"lens": sha256_file(lens_path)},
        model=resolve_model(model_path), seed=cfg["rand_seed"])
    out_json = out_dir / f"span_audit_{slug}.json"
    write_result3(payload, out_json, prov)
    register(cfg["evidence_id"], tier=TIER, command=cmd,
             what=(f"§4.1b protected-span geometry audit on {slug}: "
                   f"{len(df)} frozen Phase 2 items, label vs span-safe J "
                   f"plus three matched controls; overlap "
                   f"{summary['overlap']['label_protected']['projector_overlap_mean']} "
                   f"label vs "
                   f"{summary['overlap']['span_safe']['projector_overlap_mean']} "
                   f"span-safe; tail {summary['tail_rate_label']} -> "
                   f"{summary['tail_rate_span_safe']}"),
             outputs=[out_json, pq, pq_ov],
             inputs={"lens_sha256": sha256_file(lens_path)})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
