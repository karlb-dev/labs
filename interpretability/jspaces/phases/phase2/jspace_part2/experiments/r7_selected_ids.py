# R7 follow-up — selected-id logging grids: record, per position x band
# layer, WHICH dictionary rows the dynamic ablator deflates (and which the
# protection mask blocks) in both dyn-J arms. The protected tail
# (>1-nat deletions surviving protection on hard items) is so far a
# behavioral observation; this instrument logs its mechanism so analysis
# can test whether the deflated directions decode to the two-hop BRIDGE
# entity (workspace-content reading), the answer's own tokens
# (output-adjacent reading), or nothing in particular.
#
# Design notes (recorded):
# - Canonical answer variant only (" "+answer as given): selection at
#   prompt positions is variant-independent by causality; the scored
#   region's ids are secondary. lp is logged for cross-checking against
#   the max-over-variants grid, not as an endpoint.
# - dynR arm skipped: random-dictionary row indices carry no token
#   semantics, so an id log has no referent; the null reference for
#   overlap statistics is non-tail items under dyn-J.
# - The protected arm's per-position protect sets (clean top-10) are the
#   same clean pass either arm would see, so the unprotected arm's
#   would-have-been-protected ids come from the protected arm's log.
# Tier: pilot. Resumable per (condition, task) cell; chunk parquets land
# in a LOCAL work dir and concatenate to the Drive out_dir at the end.
#
# Usage: python -m jspace_part2.experiments.r7_selected_ids \
#          --config configs/r7_selected_ids_think.yaml [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..battery import (PROBE_SWAP, onehop_items, seq_lp_from_logits,
                       twohop_items)
from ..dictionaries import build_j_dictionaries
from ..lib import sha256_file
from ..protected_dynamic import ProtectedDynamicAblator, protected_teacher_forced
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def load_model(path_or_id):
    import transformers
    import jlens
    tok = transformers.AutoTokenizer.from_pretrained(path_or_id)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        path_or_id, dtype=torch.bfloat16).to("cuda").eval()
    return jlens.from_hf(hf, tok), hf, tok


def hard_onehop_items(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    return [{"item_id": f"hardonehop:{i}", "prompt": r["prompt"],
             "answer": r["answer"].strip(), "family": r["domain"]}
            for i, r in enumerate(rows)]


def bridge_token_ids(tok, entity: str) -> list[int]:
    """Union of tokenizations over surface forms of the bridge entity."""
    forms = {f" {entity}", entity, f" {entity.lower()}",
             f" {entity.capitalize()}"}
    ids = set()
    for f in forms:
        ids.update(tok(f, add_special_tokens=False).input_ids)
    return sorted(ids)


def unpack_capture(capture):
    """Capture records -> flat column arrays (one entry per logged id)."""
    sel, prot = [], []
    for rec in capture:
        ids = rec["ids"]                       # [T, width]
        T, W = ids.shape
        pos = np.repeat(np.arange(T, dtype=np.int32), W)
        slot = np.tile(np.arange(W, dtype=np.int32), T)
        layer = np.full(T * W, rec["layer"], dtype=np.int32)
        if rec["kind"] == "selected":
            sel.append((layer, pos, slot, ids.reshape(-1),
                        rec["scores"].reshape(-1)))
        else:
            prot.append((layer, pos, slot, ids.reshape(-1),
                         rec["blocked"].reshape(-1)))
    return sel, prot


def cell_frames(rows_sel, rows_prot, meta):
    import pandas as pd
    out = {}
    if rows_sel:
        out["sel"] = pd.DataFrame({
            "item_id": np.concatenate([np.full(len(r[0]), i, dtype=np.int32)
                                       for i, r in rows_sel]),
            "layer": np.concatenate([r[0] for _, r in rows_sel]),
            "pos": np.concatenate([r[1] for _, r in rows_sel]),
            "slot": np.concatenate([r[2] for _, r in rows_sel]),
            "token_id": np.concatenate([r[3] for _, r in rows_sel]),
            "score": np.concatenate([r[4] for _, r in rows_sel])})
    if rows_prot:
        out["prot"] = pd.DataFrame({
            "item_id": np.concatenate([np.full(len(r[0]), i, dtype=np.int32)
                                       for i, r in rows_prot]),
            "layer": np.concatenate([r[0] for _, r in rows_prot]),
            "pos": np.concatenate([r[1] for _, r in rows_prot]),
            "slot": np.concatenate([r[2] for _, r in rows_prot]),
            "token_id": np.concatenate([r[3] for _, r in rows_prot]),
            "blocked": np.concatenate([r[4] for _, r in rows_prot])})
    out["meta"] = pd.DataFrame(meta)
    return out


def main():
    import pandas as pd
    cfg_path = arg("--config", "configs/r7_selected_ids_think.yaml")
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    git = require_clean_tree("--allow-dirty" in sys.argv)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(cfg.get("work_dir", "/content/sl1_work/r7sel_chunks"))
    work.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "r7sel_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else \
        {"cells": {}}

    from jlens import JacobianLens
    lens = JacobianLens.load(cfg["lens_path"])
    band = cfg["band"]
    model, hf, tok = load_model(cfg["model_path"])
    jd = build_j_dictionaries(hf, lens, band)
    k, pk = cfg["k"], cfg["protect_top_k"]

    bridges = {f"twohop:{it['name']}": it["intermediate"]
               for it in json.loads(PROBE_SWAP.read_text())["items"]}
    tasks = {
        "twohop": twohop_items(cfg["n_twohop"]),
        "onehop": onehop_items(),
        "hardonehop": hard_onehop_items(cfg["hard_onehop_items"]),
    }
    item_index = {}          # item_id -> integer key used in the parquets
    for items in tasks.values():
        for it in items:
            item_index[it["item_id"]] = len(item_index)

    ab = ProtectedDynamicAblator(model.layers, band)
    with ab, torch.no_grad():
        for cond in ("dynJ_protected", "dynJ_unprotected"):
            for tname, items in tasks.items():
                cell = f"{cond}/{tname}"
                if state["cells"].get(cell, {}).get("done"):
                    continue
                t0 = time.time()
                rows_sel, rows_prot, meta = [], [], []
                for it in items:
                    ikey = item_index[it["item_id"]]
                    text = it["prompt"].rstrip() + f" {it['answer'].strip()}"
                    n_prompt = model.encode(it["prompt"].rstrip(),
                                            max_length=512).shape[1]
                    capture = []
                    ids, logits = protected_teacher_forced(
                        hf, model.encode, ab, jd, text, k=k, protect=pk,
                        protected=(cond == "dynJ_protected"),
                        capture=capture)
                    lp = seq_lp_from_logits(ids, logits, n_prompt)
                    sel, prot = unpack_capture(capture)
                    rows_sel += [(ikey, r) for r in sel]
                    rows_prot += [(ikey, r) for r in prot]
                    meta.append({
                        "item_key": ikey, "item_id": it["item_id"],
                        "task": tname, "condition": cond,
                        "family": it["family"], "n_prompt": int(n_prompt),
                        "n_total": int(ids.shape[1]),
                        "lp_canonical": round(lp, 4),
                        "answer": it["answer"],
                        "bridge": bridges.get(it["item_id"]),
                        "bridge_token_ids": (
                            bridge_token_ids(tok, bridges[it["item_id"]])
                            if it["item_id"] in bridges else [])})
                frames = cell_frames(rows_sel, rows_prot, meta)
                tag = cell.replace("/", "_")
                for name, df in frames.items():
                    if name != "meta":          # meta already carries both
                        df["condition"] = cond
                        df["task"] = tname
                    df.to_parquet(work / f"{tag}_{name}.parquet")
                state["cells"][cell] = {
                    "done": True, "seconds": round(time.time() - t0),
                    "chunks": sorted(p.name for p in
                                     work.glob(f"{tag}_*.parquet"))}
                state_path.write_text(json.dumps(state))
                print(f"[{time.strftime('%H:%M:%S')}] {cell} done "
                      f"({state['cells'][cell]['seconds']}s)", flush=True)
                ab.log.__init__()

    # ---- concatenate chunks -> Drive out_dir
    final = {}
    for name in ("sel", "prot", "meta"):
        parts = sorted(work.glob(f"*_{name}.parquet"))
        if parts:
            final[name] = pd.concat([pd.read_parquet(p) for p in parts],
                                    ignore_index=True)
    final["sel"].to_parquet(out_dir / "r7sel_selected.parquet")
    final["prot"].to_parquet(out_dir / "r7sel_protect.parquet")
    final["meta"].to_parquet(out_dir / "r7sel_items.parquet")

    # ---- mechanical summary + one prompt-final bridge quick-stat
    meta_df = final["meta"]
    sel_df = final["sel"]
    key_of = meta_df[meta_df.condition == "dynJ_protected"]\
        .set_index("item_key")
    hits, tot = 0, 0
    for ikey, row in key_of.iterrows():
        if row.task != "twohop" or not len(row.bridge_token_ids):
            continue
        m = sel_df[(sel_df.condition == "dynJ_protected") &
                   (sel_df.item_id == ikey) &
                   (sel_df.pos >= row.n_prompt - 2) &
                   (sel_df.pos < row.n_prompt)]
        tot += 1
        if set(row.bridge_token_ids) & set(m.token_id.tolist()):
            hits += 1
    summ = {
        "n_items": int(meta_df.item_key.nunique()),
        "n_selected_rows": int(len(sel_df)),
        "n_protect_rows": int(len(final["prot"])),
        "lp_canonical_mean_by_cell": {
            f"{c}/{t}": round(float(g.lp_canonical.mean()), 3)
            for (c, t), g in meta_df.groupby(["condition", "task"])},
        "twohop_bridge_in_promptfinal_selected_frac": (
            round(hits / tot, 3) if tot else None),
        "note": ("quick-stat = protected arm, prompt-final 2 positions, "
                 "any-layer union; per-arm split + tail contrast is the "
                 "analysis script's job")}
    prov = Provenance(
        evidence_id=cfg["evidence_id"], tier=cfg["tier"],
        command=("python -m jspace_part2.experiments.r7_selected_ids "
                 f"--config {cfg_path}"),
        config_path=cfg_path,
        inputs={"lens": sha256_file(cfg["lens_path"]),
                "probe_swap": sha256_file(PROBE_SWAP),
                "hard_onehop_items": sha256_file(cfg["hard_onehop_items"])},
        model=resolve_model(cfg["model_path"]),
        allow_dirty="--allow-dirty" in sys.argv)
    write_result({"config": cfg, "summary": summ,
                  "cells": state["cells"]},
                 out_dir / "r7sel_summary.json", prov)
    registry_append({
        "evidence_id": cfg["evidence_id"], "tier": cfg["tier"],
        "what": f"selected-id logging grid (per-position deflated/blocked "
                f"dictionary rows, both dyn-J arms): {summ}",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(out_dir / f), "sha256":
                     sha256_file(out_dir / f)}
                    for f in ("r7sel_summary.json", "r7sel_selected.parquet",
                              "r7sel_protect.parquet",
                              "r7sel_items.parquet")]})
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
