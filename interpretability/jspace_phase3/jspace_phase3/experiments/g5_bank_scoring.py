# G5 capability scoring of the Phase 3 banks (nextsteps §5.5), one model
# per run — rides each model's GPU window alongside Workstream C.
#
# For every bank item (each variant of each bundle, F and S):
#   * greedy generation, MAX_NEW=8, assay units (ScoringSession — BOS
#     where the tokenizer has one, native otherwise; Amendment-1
#     piecewise concatenation for the lp readout);
#   * deterministic grading: normalized generation starts with a
#     normalized accepted alias (capable_generation, the Phase 2
#     predicate);
#   * baseline sequence logprob of the canonical alias (the §5.5
#     covariate — capability gates inclusion, logprob is a covariate,
#     never a window).
# Counterfactual answers are also scored for lp (bridge-swap experiments
# need them) but do not enter the capability predicate.
#
# Usage:
#   python -m jspace_phase3.experiments.g5_bank_scoring --slug <slug> \
#       --model-uri model://... [--banks bank_f_v5.jsonl,bank_s_v2.jsonl]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from jspace_part2.paths import resolve as resolve_uri
from ..bank import load_bank
from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..scoring import DEFAULT_SPEC, ScoringSession

TIER = "phase3-development"
MAX_NEW = 8
REPO_DATA = Path(__file__).resolve().parents[2] / "data"


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--slug")
    model_uri = arg("--model-uri")
    banks = (arg("--banks") or "bank_f_v5.jsonl,bank_s_v2.jsonl").split(",")
    out_dir = metrics_dir(slug) / "g5_bank"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else {"done": {}, "rows": []})

    import transformers
    import jlens
    model_path = str(resolve_uri(model_uri, must_exist=True))
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    jlens.from_hf(hf, tok)                  # the assay-unit mutation
    sess = ScoringSession(tok, DEFAULT_SPEC, device="cuda")

    items = []
    bank_shas = {}
    for b in banks:
        p = REPO_DATA / b
        from jspace_part2.lib import sha256_file
        bank_shas[b] = sha256_file(p)
        for bundle in load_bank(p):
            items.extend(bundle.as_items())
    items.sort(key=lambda r: r["item_id"])
    log(f"{slug}: {len(items)} bank items ({', '.join(banks)}), "
        f"bos_prefixed={sess.bos_prefixed}")

    t0 = time.time()
    for it in items:
        iid = it["item_id"]
        if iid in state["done"]:
            continue
        ids = sess.prompt_ids(it["prompt"])
        n_prompt = ids.shape[1]
        gen = hf.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, n_prompt:], skip_special_tokens=True)
        gnorm = sess.spec.normalize(text)
        capable = any(gnorm.startswith(sess.spec.normalize(a))
                      for a in it["accepted_answers"]
                      if sess.spec.normalize(a))
        row = {"item_id": iid, "fact_id": it["fact_id"],
               "variant": it["variant"], "bank": it["bank"],
               "canonical_family": it["canonical_family"],
               "relation_group": it["relation_group"],
               "generation": text[:80], "capable_generation": bool(capable)}
        # canonical + counterfactual alias logprobs (piecewise)
        full, n_p = sess.full_ids(it["prompt"], it["accepted_answers"][0])
        logits = hf(input_ids=full, use_cache=False).logits[0].float().cpu()
        row["lp_canonical"] = sess.answer_seq_lp(full, logits, n_p)
        if it.get("counterfactual_accepted"):
            fc, n_c = sess.full_ids(it["prompt"],
                                    it["counterfactual_accepted"][0])
            lc = hf(input_ids=fc, use_cache=False).logits[0].float().cpu()
            row["lp_counterfactual"] = sess.answer_seq_lp(fc, lc, n_c)
        state["rows"].append(row)
        state["done"][iid] = 1
        if len(state["done"]) % 25 == 0:
            state_path.write_text(json.dumps(state))
            rate = (time.time() - t0) / max(len(state["done"]), 1)
            log(f"{len(state['done'])}/{len(items)} "
                f"({rate:.2f}s/item)")
    state_path.write_text(json.dumps(state))

    df = pd.DataFrame(state["rows"])
    pq = out_dir / f"g5_bank_{slug}.parquet"
    df.to_parquet(pq)
    fam_cap = (df[df.variant.isin(["direct", "composed"])]
               .groupby(["bank", "canonical_family"]).capable_generation
               .mean())
    summary = {
        "n_items": int(len(df)),
        "capable_rate": round(float(df.capable_generation.mean()), 4),
        "capable_by_variant": {k: round(float(v), 4) for k, v in
                               df.groupby("variant").capable_generation
                               .mean().items()},
        "capable_by_bank": {k: round(float(v), 4) for k, v in
                            df.groupby("bank").capable_generation
                            .mean().items()},
        "families_fully_capable_direct_composed": int(
            (fam_cap.groupby(level=0).apply(lambda s: (s == 1.0).sum()))
            .sum()),
        "bos_prefixed": sess.bos_prefixed, "banks": bank_shas}
    eid = f"p3-g5-bank-{slug}-v1"
    cmd = (f"python -m jspace_phase3.experiments.g5_bank_scoring "
           f"--slug {slug} --model-uri {model_uri}")
    out_json = out_dir / f"g5_bank_{slug}.json"
    write_result3({"summary": summary}, out_json, Provenance3(
        evidence_id=eid, tier=TIER, command=cmd,
        model=resolve_model(model_path), seed=0,
        inputs=bank_shas))
    register(eid, tier=TIER, command=cmd,
             what=(f"G5 bank capability scoring on {slug}: {len(df)} "
                   f"items, capable rate {summary['capable_rate']}, "
                   f"by variant {summary['capable_by_variant']}"),
             outputs=[out_json, pq], inputs=bank_shas)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
