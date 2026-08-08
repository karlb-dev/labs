"""OR1.6 bounded instrument cross-over (plan §11).

Qwen conditions on the frozen subsets: baseline · paper coordinate swap
(paper band, unfolded v_t) · campaign output-protected dynamic top-10 J
ablation (campaign band, campaign g-folded normalized dictionaries,
k=10, protect=10 — the frozen campaign contract, driven by the
campaign's own v2 producer) · exact rank/energy matched control.

OLMo adds the lens dimension: new merged OR1 lens vs frozen campaign
lens under both intervention families on the campaign band (paper-band
paper-swap remains the released-material primary for the new lens).

Instrument note for OR-Q4: the campaign dictionaries fold the effective
final-norm gain and row-normalize; the paper-literal swap basis is
unfolded and unnormalized (D9 recorded the folding as material on Qwen).
The cross-over therefore compares the two machineries as frozen, not a
shared basis.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from jspace_part2.dictionaries import build_j_dictionaries
from jspace_part2.matched_control import MatchedControlAblatorV2
from jspace_part2.protected_dynamic_v2 import (
    ProtectedDynamicAblatorV2,
    protected_teacher_forced_v2,
)

from ..layers import CAMPAIGN_BAND, PAPER_BAND
from ..paths import CONFIGS, DRIVE_ROOT, EXPERIMENTS_DIR
from ..rendering import preferred_token, render_raw
from ..scoring import rank_of
from .core import baseline_answer, swap_trial

CAMPAIGN_K = 10
CAMPAIGN_PROTECT = 10


def _subsets() -> dict:
    return json.loads((CONFIGS / "crossover_subset_manifest.json").read_text())


def _campaign_conditions(hf_model, model, band_dicts, ablator_cls, text,
                         score_ids):
    """One campaign-machinery pass; returns rank of each score id in
    ablated logits at the final position plus the clean rank."""
    ab = ablator_cls(model.layers, list(CAMPAIGN_BAND))
    with ab:
        ids, ablated, clean = protected_teacher_forced_v2(
            hf_model, lambda t, max_length=512: model.encode(t, max_length=max_length),
            ab, band_dicts, text, k=CAMPAIGN_K, protect=CAMPAIGN_PROTECT,
        )
    position = ids.shape[1] - 1
    return {
        "clean_ranks": {str(t): rank_of(clean[position], t) for t in score_ids},
        "ablated_ranks": {str(t): rank_of(ablated[position], t) for t in score_ids},
        "log_summary": ab.log.summary(),
    }


def run_lane(model, hf_model, lens, *, lane: str, extra_lenses: dict | None = None,
             out_dir: Path | None = None) -> dict:
    """Cross-over for one lane. ``extra_lenses`` (OLMo): {"campaign": lens}
    adds new-vs-frozen swap/ablation conditions on the campaign band."""
    out_dir = out_dir or (DRIVE_ROOT / f"crossover_{lane}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"crossover_{lane}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    subsets = _subsets()
    tokenizer = model.tokenizer
    start = time.time()
    probe_data = json.loads((EXPERIMENTS_DIR / "probe-swap.json").read_text())
    dictionaries = {
        "primary": build_j_dictionaries(hf_model, lens, list(CAMPAIGN_BAND)),
    }
    if extra_lenses:
        for name, other in extra_lenses.items():
            dictionaries[name] = build_j_dictionaries(
                hf_model, other, list(CAMPAIGN_BAND))
    rows = []
    for item_index in subsets["probe_swap_items"]:
        item = probe_data["items"][item_index]
        tokens = {
            "intermediate": preferred_token(tokenizer, item["intermediate"]),
            "swap_to": preferred_token(tokenizer, item["swap_to"]),
            "answer": preferred_token(tokenizer, item["answer"]),
            "swap_answer": preferred_token(tokenizer, item["swap_answer"]),
        }
        if any(v is None for v in tokens.values()):
            rows.append({"item_index": item_index, "state": "TOKENIZATION_GATED"})
            continue
        rendered = render_raw(model, item["prompt"])
        base = baseline_answer(model, rendered)
        score_ids = [tokens["answer"], tokens["swap_answer"]]
        row = {
            "item_index": item_index, "name": item["name"],
            "category": item["category"],
            "baseline": {
                "greedy": base["greedy_token"],
                "answer_rank": rank_of(base["logits"], tokens["answer"]),
                "swap_answer_rank": rank_of(base["logits"], tokens["swap_answer"]),
            },
        }
        for lens_name, active in [("primary", lens)] + list(
                (extra_lenses or {}).items()):
            swap = swap_trial(
                model, active, rendered,
                band=list(PAPER_BAND if lens_name == "primary" else CAMPAIGN_BAND),
                source_token_id=tokens["intermediate"],
                target_token_id=tokens["swap_to"],
                score_token_ids=[tokens["swap_answer"]], alpha=1.0,
            )
            row[f"paper_swap_{lens_name}"] = {
                "band": "paper" if lens_name == "primary" else "campaign",
                "swap_answer_rank_after": swap["best_rank"],
                "top1_success": swap["top1_success"],
            }
            if lens_name == "primary":
                swap_campaign_band = swap_trial(
                    model, active, rendered, band=list(CAMPAIGN_BAND),
                    source_token_id=tokens["intermediate"],
                    target_token_id=tokens["swap_to"],
                    score_token_ids=[tokens["swap_answer"]], alpha=1.0,
                )
                row["paper_swap_primary_campaign_band"] = {
                    "swap_answer_rank_after": swap_campaign_band["best_rank"],
                    "top1_success": swap_campaign_band["top1_success"],
                }
            row[f"protected_ablation_{lens_name}"] = _campaign_conditions(
                hf_model, model, dictionaries[lens_name],
                ProtectedDynamicAblatorV2, item["prompt"], score_ids)
            row[f"matched_control_{lens_name}"] = _campaign_conditions(
                hf_model, model, dictionaries[lens_name],
                MatchedControlAblatorV2, item["prompt"], score_ids)
        row["state"] = "EXECUTED"
        rows.append(row)
        with (out_dir / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps({"item": item_index,
                                     "utc": time.strftime("%H:%M:%SZ",
                                                          time.gmtime())}) + "\n")
    executed = [r for r in rows if r.get("state") == "EXECUTED"]

    def _answer_degradation(condition_key, lens_name):
        """Median rank change of the released answer under broad ablation."""
        deltas = []
        for r in executed:
            cell = r.get(f"{condition_key}_{lens_name}")
            if not cell:
                continue
            answer_id = None  # ranks keyed by token id string
            clean = cell["clean_ranks"]
            ablated = cell["ablated_ranks"]
            key = next(iter(clean))
            deltas.append(ablated[key] - clean[key])
        deltas.sort()
        return deltas[len(deltas) // 2] if deltas else None

    summary = {
        "experiment": "instrument-crossover", "lane": lane,
        "subsets_sha": "configs/crossover_subset_manifest.json",
        "campaign_contract": {"k": CAMPAIGN_K, "protect": CAMPAIGN_PROTECT,
                              "band": list(CAMPAIGN_BAND),
                              "dicts": "g-folded row-normalized (campaign)"},
        "n_items": len(rows), "n_executed": len(executed),
        "paper_swap_top1_primary": (
            sum(1 for r in executed if r["paper_swap_primary"]["top1_success"])
            / len(executed) if executed else None),
        "paper_swap_top1_primary_campaign_band": (
            sum(1 for r in executed
                if r["paper_swap_primary_campaign_band"]["top1_success"])
            / len(executed) if executed else None),
        "protected_answer_degradation_median": _answer_degradation(
            "protected_ablation", "primary"),
        "matched_answer_degradation_median": _answer_degradation(
            "matched_control", "primary"),
        "wall_seconds": time.time() - start,
        "rows": rows,
    }
    result_path.write_text(json.dumps(summary))
    return summary
