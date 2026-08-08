"""Probe-swap raw token-vector arm (plan §7.5):
``prompt_exact_representation_adapted_raw_jlens`` — the paper's own §3.3
headline arm. 90 released prompts; swap intermediate -> swap_to J-lens
token vectors across the band at every prompt position; score
``swap_answer`` at the final position. Official linear-probe arm:
NOT_IDENTIFIED_FROM_RELEASE (registered as a state, never run)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..layers import PAPER_BAND
from ..paths import EXPERIMENTS_DIR
from ..rendering import preferred_token, render_raw
from ..scoring import RawRecordWriter, rank_of
from .core import baseline_answer, swap_trial


def run(model, lens, *, lane: str, out_dir: Path, band=PAPER_BAND,
        alpha: float = 1.0, common: dict | None = None) -> dict:
    data = json.loads((EXPERIMENTS_DIR / "probe-swap.json").read_text())
    tokenizer = model.tokenizer
    start = time.time()
    raw_path = out_dir / f"probe_swap_{lane}_raw.jsonl"
    rows = []
    with RawRecordWriter(raw_path, common={
        **(common or {}), "experiment": "probe-swap", "lane": lane,
        "band": list(band), "alpha": alpha,
        "fidelity": "prompt_exact_representation_adapted_raw_jlens",
    }) as writer:
        for index, item in enumerate(data["items"]):
            rendered = render_raw(model, item["prompt"])
            base = baseline_answer(model, rendered)
            tokens = {
                "intermediate": preferred_token(tokenizer, item["intermediate"]),
                "swap_to": preferred_token(tokenizer, item["swap_to"]),
                "answer": preferred_token(tokenizer, item["answer"]),
                "swap_answer": preferred_token(tokenizer, item["swap_answer"]),
            }
            row = {
                "item_index": index, "name": item["name"],
                "category": item["category"],
                "greedy": base["greedy_token"],
                "baseline_correct": (
                    tokens["answer"] is not None
                    and base["greedy_token_id"] == tokens["answer"]),
                "token_valid": all(tokens[k] is not None
                                   for k in ("intermediate", "swap_to",
                                             "swap_answer")),
            }
            if not row["token_valid"]:
                row["state"] = "TOKENIZATION_GATED"
                row["gated_fields"] = [k for k, v in tokens.items() if v is None]
            else:
                row["swap_answer_rank_before"] = rank_of(
                    base["logits"], tokens["swap_answer"])
                trial = swap_trial(
                    model, lens, rendered, band=list(band),
                    source_token_id=tokens["intermediate"],
                    target_token_id=tokens["swap_to"],
                    score_token_ids=[tokens["swap_answer"]], alpha=alpha,
                    collect_diagnostics=(index % 10 == 0),
                )
                row["state"] = ("GEOMETRY_GATED" if trial["geometry_fully_gated"]
                                else "EXECUTED")
                row["swap_answer_rank_after"] = trial["best_rank"]
                row["top1_success"] = trial["top1_success"]
                row["top1_token"] = trial["top1_token"]
                if "sample_diagnostics" in trial:
                    row["sample_diagnostics"] = trial["sample_diagnostics"]
                writer.write({
                    "item_id": item["name"],
                    "released_category": item["category"],
                    "upstream_item_index": index,
                    "condition": f"raw_jlens_swap_alpha{alpha:g}",
                    "source_token_ids": [tokens["intermediate"]],
                    "target_token_ids": [tokens["swap_to"]],
                    "baseline_capable": row["baseline_correct"],
                    "tokenization_valid": True,
                    "gate_reason": (row["state"] if row["state"] != "EXECUTED"
                                    else None),
                    "target_rank_before": row["swap_answer_rank_before"],
                    "target_rank_after": row["swap_answer_rank_after"],
                    "top1_after": row["top1_success"],
                })
            rows.append(row)
    executed = [r for r in rows if r.get("state") == "EXECUTED"]
    capable = [r for r in executed if r["baseline_correct"]]
    def _top1(subset):
        return (sum(1 for r in subset if r["top1_success"]) / len(subset)
                if subset else None)
    by_category: dict[str, dict] = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"], {"n": 0, "executed": 0,
                                                          "capable": 0,
                                                          "success": 0})
        bucket["n"] += 1
        if row.get("state") == "EXECUTED":
            bucket["executed"] += 1
            if row["baseline_correct"]:
                bucket["capable"] += 1
                if row["top1_success"]:
                    bucket["success"] += 1
    summary = {
        "experiment": "probe-swap", "lane": lane, "alpha": alpha,
        "band": list(band),
        "fidelity": "prompt_exact_representation_adapted_raw_jlens",
        "official_probe_arm": "NOT_IDENTIFIED_FROM_RELEASE",
        "n_items": len(rows),
        "n_tokenization_gated": sum(1 for r in rows
                                    if r.get("state") == "TOKENIZATION_GATED"),
        "n_geometry_gated": sum(1 for r in rows
                                if r.get("state") == "GEOMETRY_GATED"),
        "n_executed": len(executed),
        "n_baseline_capable": len(capable),
        "diagnostic_top1": _top1(executed),
        "capable_top1": _top1(capable),
        "multihop_capable_top1": _top1(
            [r for r in capable if r["category"] == "multihop"]),
        "non_multihop_capable_top1": _top1(
            [r for r in capable if r["category"] != "multihop"]),
        "by_category": by_category,
        "wall_seconds": time.time() - start,
        "rows": rows,
        "raw_records": str(raw_path),
    }
    result_path = out_dir / f"probe_swap_{lane}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    result_path.write_text(json.dumps(summary))
    summary["_output"] = str(result_path)
    return summary
