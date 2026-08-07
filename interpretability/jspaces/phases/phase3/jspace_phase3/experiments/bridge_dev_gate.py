# P3-P3 development IDENTIFIABILITY gate (addendum §4.4 + nextsteps
# §6.5): before the freeze, choose which model carries the
# bridge-protection-rescue primary — "based only on development
# identifiability, not effect magnitude."
#
# On the first N capable composed Bank F bundles (deterministic fact_id
# order), three arms per item, span-safe base protection throughout:
#   meanJ_span_safe                     reference damage
#   + true-bridge protection            protect_sets ∪ bridge pieces
#   + distractor-bridge protection      protect_sets ∪ counterfactual
#                                       bridge pieces
#
# GATE METRICS (magnitude-blind by construction — nothing here reports
# a family-level rescue mean):
#   * bridge piece coverage: any-piece / all-piece presence of bridge
#     tokens in the J selection under the reference arm;
#   * per-item rescue contrast r_i = Δ(true) − Δ(distractor):
#     SD(r_i), fraction |r_i| above the determinism floor (one item
#     re-measured to bound it), piece-count mismatch true vs distractor;
#   * verdict: identifiable iff any-piece coverage ≥ 0.5 AND
#     frac(|r_i| > floor) ≥ 0.5.
#
# Usage:
#   python -m jspace_phase3.experiments.bridge_dev_gate --slug <slug> \
#       --model-uri model://...
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from jspace_part2.dictionaries import build_j_dictionaries
from jspace_part2.lib import sha256_file
from jspace_part2.paths import resolve as resolve_uri
from ..ablator3 import Phase3JAblator
from ..bank import load_bank
from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..scoring import DEFAULT_SPEC, ScoringSession

TIER = "phase3-development"
N_ITEMS = 20
REPO_DATA = Path(__file__).resolve().parents[2] / "data"


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bridge_piece_ids(tok, bridge: str) -> torch.Tensor:
    """Union of token ids over surface variants of the bridge string
    (leading-space, bare, title/lower), articles stripped."""
    b = bridge.removeprefix("the ").removeprefix("The ").strip()
    ids = set()
    for v in {f" {b}", b, f" {b.lower()}", f" {b.title()}"}:
        for t in tok(v, add_special_tokens=False).input_ids:
            ids.add(int(t))
    return torch.tensor(sorted(ids), dtype=torch.long)


@torch.no_grad()
def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--slug")
    model_uri = arg("--model-uri")
    out_dir = metrics_dir(slug) / "bridge_dev_gate"
    out_dir.mkdir(parents=True, exist_ok=True)

    import transformers
    import jlens
    from jlens import JacobianLens
    model_path = str(resolve_uri(model_uri, must_exist=True))
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    sess = ScoringSession(tok, DEFAULT_SPEC, device="cuda")
    lens = JacobianLens.load(str(resolve_uri(arg("--lens-uri"))))
    band = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]
    k = pk = 10
    jd = build_j_dictionaries(hf, lens, band)
    ab = Phase3JAblator(model.layers, band)

    # capable composed bundles with counterfactual bridges, G5-gated
    g5 = pd.read_parquet(metrics_dir(slug) / "g5_bank" /
                         f"g5_bank_{slug}.parquet")
    cap = set(g5[(g5.variant == "composed")
                 & g5.capable_generation].fact_id) \
        & set(g5[(g5.variant == "direct") & g5.capable_generation].fact_id)
    bundles = [b for b in load_bank(REPO_DATA / "bank_f_v7.jsonl")
               if b.fact_id in cap and b.counterfactual_bridge]
    bundles = sorted(bundles, key=lambda b: b.fact_id)[:N_ITEMS]
    log(f"{slug}: {len(bundles)} capable composed bundles for the gate")

    def run_arm(ids, psets):
        ab.log = type(ab.log)()
        ab.phase, ab.forward_index = "prefill", 0
        ab.mode = {"dicts": jd, "k": k, "nonneg": True,
                   "protect_sets": psets, "active_phases": {"prefill"},
                   "span_safe": True, "record_overlap": False,
                   "answer_id": None, "record_ids": True}
        with ab:
            out = hf(input_ids=ids, use_cache=False).logits[0].float()
        ab.mode = None
        return out.cpu(), ab.log

    rows = []
    floor_probe = None
    for n, b in enumerate(bundles):
        prompt = b.prompts["composed"]
        alias = b.accepted_answers[0]
        full, n_p = sess.full_ids(prompt, alias)
        T = full.shape[1]
        ab.mode = None
        clean = hf(input_ids=full, use_cache=False).logits[0].float()
        psets = clean.topk(pk, dim=-1).indices
        lp_base = sess.answer_seq_lp(full, clean.cpu(), n_p)

        bt = bridge_piece_ids(tok, b.bridge).to(psets.device)
        dt = bridge_piece_ids(tok, b.counterfactual_bridge).to(psets.device)
        arms = {
            "span_safe": psets,
            "true_bridge": torch.cat(
                [psets, bt.unsqueeze(0).expand(T, -1)], dim=1),
            "distractor_bridge": torch.cat(
                [psets, dt.unsqueeze(0).expand(T, -1)], dim=1)}
        lp, sel_ids = {}, None
        for arm, ps in arms.items():
            logits, alog = run_arm(full, ps)
            lp[arm] = sess.answer_seq_lp(full, logits, n_p)
            if arm == "span_safe":
                sel_ids = set()
                for prec in alog.positions:
                    if prec.selected_ids:
                        sel_ids.update(int(x) for x in prec.selected_ids)
        bt_set = {int(x) for x in bt.cpu()}
        cover_any = bool(sel_ids and (bt_set & sel_ids))
        cover_all = bool(sel_ids and bt_set and bt_set <= sel_ids)
        rows.append({
            "fact_id": b.fact_id, "lp_base": lp_base,
            "d_span_safe": lp["span_safe"] - lp_base,
            "d_true": lp["true_bridge"] - lp_base,
            "d_distractor": lp["distractor_bridge"] - lp_base,
            "rescue_contrast": lp["true_bridge"] - lp["distractor_bridge"],
            "bridge_pieces": int(len(bt)), "distractor_pieces": int(len(dt)),
            "cover_any": cover_any, "cover_all": cover_all})
        if n == 0 and floor_probe is None:
            logits2, _ = run_arm(full, arms["true_bridge"])
            floor_probe = abs(sess.answer_seq_lp(full, logits2, n_p)
                              - lp["true_bridge"])
        log(f"{n + 1}/{len(bundles)} {b.fact_id}")

    df = pd.DataFrame(rows)
    floor = max(float(floor_probe or 0.0), 1e-4)
    r = df.rescue_contrast
    gate = {
        "n_items": int(len(df)),
        "determinism_floor": round(floor, 6),
        "cover_any_rate": round(float(df.cover_any.mean()), 4),
        "cover_all_rate": round(float(df.cover_all.mean()), 4),
        "rescue_sd": round(float(r.std()), 4),
        "frac_above_floor": round(float((r.abs() > 10 * floor).mean()), 4),
        "piece_count_mismatch_mean": round(float(
            (df.bridge_pieces - df.distractor_pieces).abs().mean()), 3),
        "identifiable": bool(float(df.cover_any.mean()) >= 0.5
                             and float((r.abs() > 10 * floor).mean()) >= 0.5),
    }
    eid = f"p3-bridge-dev-gate-{slug}-v1"
    cmd = (f"python -m jspace_phase3.experiments.bridge_dev_gate "
           f"--slug {slug} --model-uri {model_uri} "
           f"--lens-uri {arg('--lens-uri')}")
    pq = out_dir / f"bridge_dev_gate_{slug}.parquet"
    df.to_parquet(pq)
    out_json = out_dir / f"bridge_dev_gate_{slug}.json"
    write_result3({"gate": gate}, out_json, Provenance3(
        evidence_id=eid, tier=TIER, command=cmd,
        model=resolve_model(model_path), seed=0))
    register(eid, tier=TIER, command=cmd,
             what=(f"P3-P3 dev identifiability gate on {slug}: cover_any "
                   f"{gate['cover_any_rate']}, frac>|floor| "
                   f"{gate['frac_above_floor']}, identifiable="
                   f"{gate['identifiable']} (magnitude-blind)"),
             outputs=[out_json, pq])
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
