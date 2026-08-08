"""Verbal report (plan §7.3): 14 categories, release-literal candidate
rule primary; paper-text top-10-output exclusion as named sensitivity.

Prompt: user turn "Think of a {category}. Answer in one word." rendered
at the generation boundary (D1); the greedy next token there is the
swap-out source; each eligible candidate is swapped in across the band at
every prompt position; success = candidate rank 1 at the same position.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..layers import PAPER_BAND
from ..paths import EXPERIMENTS_DIR
from ..rendering import preferred_token, render_chat
from ..scoring import RawRecordWriter, rank_of
from .core import baseline_answer, swap_trial


def _render(model, category: str, *, lane: str):
    messages = [{"role": "user",
                 "content": f"Think of a {category}. Answer in one word."}]
    kwargs = {"enable_thinking": False} if lane == "qwen" else {}
    return render_chat(model, messages, extra_template_kwargs=kwargs)


def run(model, lens, *, lane: str, out_dir: Path, band=PAPER_BAND,
        alpha: float = 1.0, common: dict | None = None) -> dict:
    data = json.loads((EXPERIMENTS_DIR / "verbal-report.json").read_text())
    tokenizer = model.tokenizer
    start = time.time()
    categories = []
    raw_path = out_dir / f"verbal_report_{lane}_raw.jsonl"
    with RawRecordWriter(raw_path, common={
        **(common or {}), "experiment": "verbal-report", "lane": lane,
        "band": list(band), "alpha": alpha,
    }) as writer:
        for category, words in data["candidates"].items():
            rendered = _render(model, category, lane=lane)
            base = baseline_answer(model, rendered)
            answer_token = base["greedy_token_id"]
            answer_text = base["greedy_token"].strip()
            clean_top10_ids = base["logits"].topk(10).indices.tolist()
            trials = []
            for candidate in words[:10]:
                if candidate.strip().lower() == answer_text.lower():
                    trials.append({"candidate": candidate, "state": "SKIP_ANSWER"})
                    continue
                candidate_token = preferred_token(tokenizer, candidate)
                if candidate_token is None:
                    trials.append({"candidate": candidate,
                                   "state": "TOKENIZATION_GATED"})
                    continue
                clean_rank = rank_of(base["logits"], candidate_token)
                trial = swap_trial(
                    model, lens, rendered, band=list(band),
                    source_token_id=answer_token,
                    target_token_id=candidate_token,
                    score_token_ids=[candidate_token], alpha=alpha,
                    collect_diagnostics=True,
                )
                trial.update({
                    "candidate": candidate,
                    "candidate_token_id": candidate_token,
                    "state": ("GEOMETRY_GATED" if trial["geometry_fully_gated"]
                              else "EXECUTED"),
                    "rank_before": clean_rank,
                    "rank_after": trial["score_ranks"][str(candidate_token)],
                    "in_clean_top10": candidate_token in clean_top10_ids,
                })
                trials.append(trial)
                writer.write({
                    "item_id": f"{category}:{candidate}",
                    "released_category": category,
                    "source_token_ids": [answer_token],
                    "target_token_ids": [candidate_token],
                    "condition": f"swap_alpha{alpha}",
                    "baseline_capable": True,
                    "tokenization_valid": True,
                    "gate_reason": (trial["state"]
                                    if trial["state"] != "EXECUTED" else None),
                    "target_rank_before": clean_rank,
                    "target_rank_after": trial["rank_after"],
                    "top1_after": trial["top1_success"],
                })
            executed = [t for t in trials if t.get("state") == "EXECUTED"]
            success = [t for t in executed if t["top1_success"]]
            top5 = [t for t in executed if t["rank_after"] <= 5]
            categories.append({
                "category": category,
                "answer_token": base["greedy_token"],
                "n_candidates": len(words[:10]),
                "n_executed": len(executed),
                "n_skip_answer": sum(1 for t in trials
                                     if t.get("state") == "SKIP_ANSWER"),
                "n_tokenization_gated": sum(1 for t in trials
                                            if t.get("state") == "TOKENIZATION_GATED"),
                "n_geometry_gated": sum(1 for t in trials
                                        if t.get("state") == "GEOMETRY_GATED"),
                "top1_rate": len(success) / len(executed) if executed else None,
                "top5_rate": len(top5) / len(executed) if executed else None,
                "median_rank_after": (sorted(t["rank_after"] for t in executed)
                                      [len(executed) // 2] if executed else None),
                "sensitivity_excl_top10": {
                    "n": len([t for t in executed if not t["in_clean_top10"]]),
                    "top1": (lambda subset: (sum(1 for t in subset
                                                 if t["top1_success"]) / len(subset)
                                             if subset else None))(
                        [t for t in executed if not t["in_clean_top10"]]),
                },
                "trials": trials,
            })
    n_categories = len(categories)
    rates = [c["top1_rate"] for c in categories if c["top1_rate"] is not None]
    summary = {
        "experiment": "verbal-report", "lane": lane, "alpha": alpha,
        "band": list(band), "n_categories": n_categories,
        "category_equal_top1": sum(rates) / len(rates) if rates else None,
        "trial_weighted_top1": (
            sum(c["top1_rate"] * c["n_executed"] for c in categories
                if c["top1_rate"] is not None)
            / max(sum(c["n_executed"] for c in categories), 1)),
        "category_equal_top5": (
            sum(c["top5_rate"] for c in categories if c["top5_rate"] is not None)
            / max(len([c for c in categories if c["top5_rate"] is not None]), 1)),
        "wall_seconds": time.time() - start,
        "categories": categories,
        "raw_records": str(raw_path),
    }
    result_path = out_dir / f"verbal_report_{lane}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    result_path.write_text(json.dumps(summary))
    summary["_output"] = str(result_path)
    return summary
