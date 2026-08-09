"""The six released lens evaluations (plan §7.2 / §9.1).

One forward per item records residuals at the requested source layers +
final; J-lens logits (transport + unembed) and logit-lens logits (unembed
raw) come from the same activations. Readout positions per D7: final
token everywhere except poetry (last newline).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from jlens.hooks import ActivationRecorder

from .paths import EVALUATIONS_DIR
from .rendering import Rendered, last_newline_position, render_raw
from .scoring import rank_of
from .targets import intermediate_token_ids

EVAL_SETS = [
    "lens-eval-multihop",
    "lens-eval-multilingual",
    "lens-eval-poetry",
    "lens-eval-typo",
    "lens-eval-order-ops",
    "lens-eval-association",
]
PASS_KS = (1, 5, 20)


@torch.no_grad()
def run_eval_set(
    model,
    lens,
    set_name: str,
    *,
    source_layers: list[int],
    max_items: int | None = None,
) -> dict:
    """Per-item, per-layer ranks for J-lens and logit lens on one set."""
    data = json.loads((EVALUATIONS_DIR / f"{set_name}.json").read_text())
    items = data["items"][:max_items] if max_items else data["items"]
    tokenizer = model.tokenizer
    final_layer = model.n_layers - 1
    record_at = sorted(set(source_layers) | {final_layer})
    rows = []
    start = time.time()
    for index, item in enumerate(items):
        rendered = render_raw(model, item["prompt"])
        if set_name == "lens-eval-poetry":
            position = last_newline_position(rendered, tokenizer)
        else:
            position = rendered.final_position
        with ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward(rendered.input_ids)
            residuals = {
                layer: recorder.activations[layer][0, position].detach()
                for layer in record_at
            }
        intermediates = []
        for key in item["intermediates"]:
            token_ids = intermediate_token_ids(tokenizer, set_name, key)
            entry = {"key": key, "token_ids": token_ids,
                     "tokenization_valid": bool(token_ids)}
            if token_ids:
                per_layer_j = {}
                per_layer_logit = {}
                for layer in source_layers:
                    h = residuals[layer].float()
                    transported = lens.transport(
                        h.unsqueeze(0), layer
                    )[0]
                    j_logits = model.unembed(transported).float().cpu()
                    raw_logits = model.unembed(h).float().cpu()
                    per_layer_j[layer] = min(
                        rank_of(j_logits, t) for t in token_ids
                    )
                    per_layer_logit[layer] = min(
                        rank_of(raw_logits, t) for t in token_ids
                    )
                entry["jlens_rank_per_layer"] = per_layer_j
                entry["logit_rank_per_layer"] = per_layer_logit
                entry["jlens_min_rank"] = min(per_layer_j.values())
                entry["logit_min_rank"] = min(per_layer_logit.values())
            intermediates.append(entry)
        rows.append({
            "item_index": index,
            "name": item.get("name"),
            "position": position,
            "seq_len": rendered.seq_len,
            "intermediates": intermediates,
        })
    return {
        "set": set_name,
        "n_items": len(rows),
        "source_layers": source_layers,
        "wall_seconds": time.time() - start,
        "rows": rows,
    }


def aggregate_pass_at_k(result: dict, *, which: str) -> dict:
    """pass@k over token-valid intermediates (primary) and the released
    denominator (sensitivity, invalid counted as miss)."""
    key = f"{which}_min_rank"
    out = {}
    for denominator in ("token_valid", "released"):
        per_item = []
        for row in result["rows"]:
            valid = [i for i in row["intermediates"] if i["tokenization_valid"]]
            universe = row["intermediates"] if denominator == "released" else valid
            if not universe:
                continue
            fractions = {}
            for k in PASS_KS:
                hits = sum(
                    1 for i in valid if i[key] <= k
                )
                fractions[k] = hits / len(universe)
            per_item.append(fractions)
        out[denominator] = {
            f"pass@{k}": (sum(f[k] for f in per_item) / len(per_item)
                          if per_item else float("nan"))
            for k in PASS_KS
        }
        out[denominator]["n_items"] = len(per_item)
    attrition = sum(
        1 for row in result["rows"]
        for i in row["intermediates"] if not i["tokenization_valid"]
    )
    total = sum(len(row["intermediates"]) for row in result["rows"])
    out["attrition"] = {"gated_intermediates": attrition, "total": total}
    return out


def save_result(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable eval result exists: {path}")
    path.write_text(json.dumps(result))
