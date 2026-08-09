"""Flexible generalization (plan §7.4): 4 categories x 4 funcs x 12
ordered arg swaps; raw-completion templates; alpha 1 primary + alpha 2
frozen sensitivity; three-population accounting."""
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
        alphas=(1.0, 2.0), common: dict | None = None) -> dict:
    data = json.loads((EXPERIMENTS_DIR / "flexible-generalization.json").read_text())
    tokenizer = model.tokenizer
    start = time.time()
    raw_path = out_dir / f"flexible_generalization_{lane}_raw.jsonl"
    categories_out = []
    with RawRecordWriter(raw_path, common={
        **(common or {}), "experiment": "flexible-generalization",
        "lane": lane, "band": list(band),
    }) as writer:
        for category in data["categories"]:
            name = category["name"]
            args = category["args"]
            arg_tokens = {a: preferred_token(tokenizer, a) for a in args}
            baselines = {}
            for func in category["funcs"]:
                for arg in args:
                    rendered = render_raw(model, func["template"].format(arg=arg))
                    base = baseline_answer(model, rendered)
                    answer = func["answers"][arg]
                    answer_token = preferred_token(tokenizer, answer)
                    baselines[(func["name"], arg)] = {
                        "rendered": rendered,
                        "answer": answer,
                        "answer_token": answer_token,
                        "greedy": base["greedy_token"],
                        "correct": (answer_token is not None
                                    and base["greedy_token_id"] == answer_token),
                        "logits": base["logits"],
                    }
            cells = []
            for func in category["funcs"]:
                for source_arg in args:
                    for target_arg in args:
                        if source_arg == target_arg:
                            continue  # diagonal no-op sentinels handled below
                        key = (func["name"], source_arg)
                        base = baselines[key]
                        target_answer_token = baselines[
                            (func["name"], target_arg)
                        ]["answer_token"]
                        cell = {
                            "category": name, "function": func["name"],
                            "source_arg": source_arg, "target_arg": target_arg,
                            "source_baseline_correct": base["correct"],
                            "target_baseline_correct": baselines[
                                (func["name"], target_arg)]["correct"],
                        }
                        source_token = arg_tokens[source_arg]
                        target_token = arg_tokens[target_arg]
                        if (source_token is None or target_token is None
                                or target_answer_token is None):
                            cell["state"] = "TOKENIZATION_GATED"
                            cells.append(cell)
                            continue
                        cell["rank_before"] = rank_of(base["logits"],
                                                      target_answer_token)
                        for alpha in alphas:
                            trial = swap_trial(
                                model, lens, base["rendered"], band=list(band),
                                source_token_id=source_token,
                                target_token_id=target_token,
                                score_token_ids=[target_answer_token],
                                alpha=alpha,
                            )
                            state = ("GEOMETRY_GATED"
                                     if trial["geometry_fully_gated"]
                                     else "EXECUTED")
                            cell[f"alpha{alpha:g}"] = {
                                "state": state,
                                "rank_after": trial["best_rank"],
                                "top1": trial["top1_success"],
                                "top1_token": trial["top1_token"],
                            }
                            writer.write({
                                "item_id": f"{name}:{func['name']}:"
                                           f"{source_arg}->{target_arg}",
                                "released_category": name,
                                "condition": f"swap_alpha{alpha:g}",
                                "source_token_ids": [source_token],
                                "target_token_ids": [target_token],
                                "baseline_capable": bool(
                                    base["correct"] and baselines[
                                        (func["name"], target_arg)]["correct"]),
                                "tokenization_valid": True,
                                "gate_reason": (state if state != "EXECUTED"
                                                else None),
                                "target_rank_before": cell["rank_before"],
                                "target_rank_after": trial["best_rank"],
                                "top1_after": trial["top1_success"],
                            })
                        cell["state"] = "EXECUTED"
                        cells.append(cell)
            # Diagonal sentinels: one per function, alpha 1, expect no-op-ish
            diagonal = []
            for func in category["funcs"][:2]:
                arg = args[0]
                base = baselines[(func["name"], arg)]
                if arg_tokens[arg] is None or base["answer_token"] is None:
                    continue
                own_rank_before = rank_of(base["logits"], base["answer_token"])
                diagonal.append({
                    "function": func["name"], "arg": arg,
                    "note": "s==t geometry-gated by contract; recorded as sentinel",
                    "own_answer_rank_clean": own_rank_before,
                })
            executed = [c for c in cells if c.get("state") == "EXECUTED"]
            capable = [c for c in executed
                       if c["source_baseline_correct"]
                       and c["target_baseline_correct"]]
            def _rate(subset, alpha):
                key = f"alpha{alpha:g}"
                done = [c for c in subset if c.get(key, {}).get("state") == "EXECUTED"]
                if not done:
                    return None
                return sum(1 for c in done if c[key]["top1"]) / len(done)
            per_function = {}
            for func in category["funcs"]:
                subset = [c for c in capable if c["function"] == func["name"]]
                per_function[func["name"]] = _rate(subset, 1.0)
            categories_out.append({
                "category": name,
                "n_cells": len(cells),
                "n_executed": len(executed),
                "n_tokenization_gated": sum(1 for c in cells
                                            if c.get("state") == "TOKENIZATION_GATED"),
                "n_capable": len(capable),
                "baseline_correct_fraction": (
                    sum(1 for b in baselines.values() if b["correct"])
                    / len(baselines)),
                "diagnostic_top1_alpha1": _rate(executed, 1.0),
                "capable_top1_alpha1": _rate(capable, 1.0),
                "diagnostic_top1_alpha2": _rate(executed, 2.0),
                "capable_top1_alpha2": _rate(capable, 2.0),
                "per_function_capable_alpha1": per_function,
                "n_functions_redirected_alpha1": sum(
                    1 for rate in per_function.values() if rate and rate > 0),
                "diagonal_sentinels": diagonal,
                "cells": cells,
            })
    def _mean(key):
        values = [c[key] for c in categories_out if c[key] is not None]
        return sum(values) / len(values) if values else None
    summary = {
        "experiment": "flexible-generalization", "lane": lane,
        "band": list(band), "alphas": list(alphas),
        "n_categories": len(categories_out),
        "category_equal_capable_top1_alpha1": _mean("capable_top1_alpha1"),
        "category_equal_diagnostic_top1_alpha1": _mean("diagnostic_top1_alpha1"),
        "category_equal_capable_top1_alpha2": _mean("capable_top1_alpha2"),
        "total_executed_alpha1": sum(c["n_executed"] for c in categories_out),
        "wall_seconds": time.time() - start,
        "categories": categories_out,
        "raw_records": str(raw_path),
    }
    result_path = out_dir / f"flexible_generalization_{lane}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    result_path.write_text(json.dumps(summary))
    summary["_output"] = str(result_path)
    return summary
