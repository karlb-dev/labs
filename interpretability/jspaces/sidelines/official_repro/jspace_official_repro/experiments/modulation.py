"""directed-modulation (math + topic families) and dual-task
(plan §10.B). Carrier-copy template reconstructed from the upstream
``jlens/examples.py`` released examples (D13); deterministic carrier
rotation (D12). Line-break family is a separate R2 module.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from jlens.hooks import ActivationRecorder

from ..layers import PAPER_BAND
from ..paths import EXPERIMENTS_DIR
from ..rendering import find_token_span, render_chat
from ..scoring import rank_of
from ..targets import synonym_token_ids

TAIL = " Don't write anything else."


def _render_carrier_task(model, *, lane: str, carrier: str, instruction: str):
    user = f'Write "{carrier}" {instruction}{TAIL}'
    kwargs = {"enable_thinking": False} if lane == "qwen" else {}
    messages = [{"role": "user", "content": user},
                {"role": "assistant", "content": carrier}]
    rendered = render_chat(model, messages, continue_final=True,
                           extra_template_kwargs=kwargs)
    span = find_token_span(rendered, model.tokenizer, carrier, from_end=True)
    return rendered, span


@torch.no_grad()
def _span_hit(model, lens, rendered, *, span, token_ids, band):
    with ActivationRecorder(model.layers, at=list(band)) as recorder:
        model.forward(rendered.input_ids)
        residuals = {
            layer: recorder.activations[layer][0, span[0]:span[1] + 1].detach()
            for layer in band
        }
    min_rank = None
    for layer in band:
        transported = lens.transport(residuals[layer].float(), layer)
        logits = model.unembed(transported).float().cpu()
        for offset in range(logits.shape[0]):
            rank = min(rank_of(logits[offset], t) for t in token_ids)
            min_rank = rank if min_rank is None else min(min_rank, rank)
    return {"min_rank": min_rank, "hit_rank1": min_rank == 1,
            "hit_rank5": min_rank <= 5}


def run_directed_modulation(model, lens, *, lane: str, out_dir: Path,
                            band=PAPER_BAND) -> dict:
    data = json.loads((EXPERIMENTS_DIR / "directed-modulation.json").read_text())
    tokenizer = model.tokenizer
    start = time.time()
    carriers = data["carrier_sentences"]
    targets = ([("math", p["expr"], [str(p["answer"])], p["tier"])
                for p in data["math_problems"]]
               + [("topic", c["name"], c["members"], None)
                  for c in data["topic_categories"]])
    rows = []
    for phrasing_index, phrasing in enumerate(data["phrasings"]):
        kind = data["group_kind"][phrasing["group"]]
        for target_index, (family, x_value, tracked, tier) in enumerate(targets):
            carrier = carriers[(phrasing_index + target_index) % len(carriers)]
            token_ids = synonym_token_ids(tokenizer, tracked)
            row = {"phrasing": phrasing["name"], "group": phrasing["group"],
                   "kind": kind, "family": family, "x": x_value,
                   "tier": tier, "carrier_index":
                       (phrasing_index + target_index) % len(carriers),
                   "tokenization_valid": bool(token_ids)}
            if not token_ids:
                row["state"] = "TOKENIZATION_GATED"
                rows.append(row)
                continue
            instruction = phrasing["text"].format(x=x_value)
            rendered, span = _render_carrier_task(
                model, lane=lane, carrier=carrier, instruction=instruction)
            row.update(_span_hit(model, lens, rendered, span=span,
                                 token_ids=token_ids, band=list(band)))
            row["state"] = "EXECUTED"
            rows.append(row)
    executed = [r for r in rows if r.get("state") == "EXECUTED"]
    def _rates(subset):
        return {
            "n": len(subset),
            "hit_rank1": (sum(1 for r in subset if r["hit_rank1"]) / len(subset)
                          if subset else None),
            "hit_rank5": (sum(1 for r in subset if r["hit_rank5"]) / len(subset)
                          if subset else None),
        }
    summary = {
        "experiment": "directed-modulation", "lane": lane, "band": list(band),
        "pairing": "carrier rotation (phrasing_index+target_index) % 20 (D12)",
        "n_trials": len(rows), "n_executed": len(executed),
        "by_kind": {kind: _rates([r for r in executed if r["kind"] == kind])
                    for kind in ("focus", "suppress", "control")},
        "by_kind_family": {
            f"{kind}:{family}": _rates([
                r for r in executed
                if r["kind"] == kind and r["family"] == family])
            for kind in ("focus", "suppress", "control")
            for family in ("math", "topic")},
        "wall_seconds": time.time() - start,
        "rows": rows,
    }
    path = out_dir / f"directed_modulation_{lane}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(summary))
    return summary


def run_dual_task(model, lens, *, lane: str, out_dir: Path,
                  band=PAPER_BAND) -> dict:
    data = json.loads((EXPERIMENTS_DIR / "dual-task.json").read_text())
    tokenizer = model.tokenizer
    start = time.time()
    carrier = data["carrier_sentence"]
    pairs = {p["key"]: p for p in data["pairs"]}

    def concept_clause(pair):
        return f"Concentrate on {pair['concept']}"

    def math_clause(pair):
        return (f"try to focus on evaluating "
                f"{pair['base']}^{pair['exp']} - {pair['sub']}")

    def math_tokens(pair):
        answer = pair["base"] ** pair["exp"] - pair["sub"]
        from ..targets import _NUMBER_WORDS

        forms = [str(answer)]
        if str(answer) in _NUMBER_WORDS:
            forms.append(_NUMBER_WORDS[str(answer)])
        return synonym_token_ids(tokenizer, forms)

    def concept_tokens(pair):
        return synonym_token_ids(tokenizer, pair["concept_words"])

    rows = []
    for pair in data["pairs"]:
        conditions = {
            "concept": f"{concept_clause(pair)} while you write the sentence.",
            "math": f"{math_clause(pair).capitalize()} while you write the sentence.",
            "concept_first": (f"{concept_clause(pair)} and "
                              f"{math_clause(pair)} while you write the sentence."),
            "math_first": (f"{math_clause(pair).capitalize()} and "
                           f"concentrate on {pair['concept']} "
                           f"while you write the sentence."),
        }
        for condition_name, instruction in conditions.items():
            rendered, span = _render_carrier_task(
                model, lane=lane, carrier=carrier, instruction=instruction)
            row = {"key": pair["key"], "arm": "concept_math",
                   "condition": condition_name}
            for task_name, token_ids in (("concept", concept_tokens(pair)),
                                         ("math", math_tokens(pair))):
                if not token_ids:
                    row[task_name] = {"state": "TOKENIZATION_GATED"}
                    continue
                hit = _span_hit(model, lens, rendered, span=span,
                                token_ids=token_ids, band=list(band))
                row[task_name] = {**hit, "reachable": hit["hit_rank5"],
                                  "state": "EXECUTED"}
            rows.append(row)
    for a_key, b_key in data["concept_pairs"]:
        pair_a, pair_b = pairs[a_key], pairs[b_key]
        conditions = {
            "a_alone": f"{concept_clause(pair_a)} while you write the sentence.",
            "b_alone": f"{concept_clause(pair_b)} while you write the sentence.",
            "a_first": (f"{concept_clause(pair_a)} and concentrate on "
                        f"{pair_b['concept']} while you write the sentence."),
            "b_first": (f"{concept_clause(pair_b)} and concentrate on "
                        f"{pair_a['concept']} while you write the sentence."),
        }
        for condition_name, instruction in conditions.items():
            rendered, span = _render_carrier_task(
                model, lane=lane, carrier=carrier, instruction=instruction)
            row = {"key": f"{a_key}+{b_key}", "arm": "concept_concept",
                   "condition": condition_name}
            for task_name, token_ids in (("a", concept_tokens(pair_a)),
                                         ("b", concept_tokens(pair_b))):
                if not token_ids:
                    row[task_name] = {"state": "TOKENIZATION_GATED"}
                    continue
                hit = _span_hit(model, lens, rendered, span=span,
                                token_ids=token_ids, band=list(band))
                row[task_name] = {**hit, "reachable": hit["hit_rank5"],
                                  "state": "EXECUTED"}
            rows.append(row)

    def _reach(arm, task, single_conditions, dual_conditions):
        singles = [r[task]["reachable"] for r in rows
                   if r["arm"] == arm and r["condition"] in single_conditions
                   and r.get(task, {}).get("state") == "EXECUTED"]
        duals = [r[task]["reachable"] for r in rows
                 if r["arm"] == arm and r["condition"] in dual_conditions
                 and r.get(task, {}).get("state") == "EXECUTED"]
        return {
            "single_rate": sum(singles) / len(singles) if singles else None,
            "dual_rate": sum(duals) / len(duals) if duals else None,
            "interference": ((sum(singles) / len(singles))
                             - (sum(duals) / len(duals))
                             if singles and duals else None),
        }
    summary = {
        "experiment": "dual-task", "lane": lane, "band": list(band),
        "template_provenance": "reconstructed from released examples.py (D13)",
        "n_rows": len(rows),
        "concept_math": {
            "concept": _reach("concept_math", "concept", {"concept"},
                              {"concept_first", "math_first"}),
            "math": _reach("concept_math", "math", {"math"},
                           {"concept_first", "math_first"}),
        },
        "concept_concept": {
            "a": _reach("concept_concept", "a", {"a_alone"},
                        {"a_first", "b_first"}),
            "b": _reach("concept_concept", "b", {"b_alone"},
                        {"a_first", "b_first"}),
        },
        "wall_seconds": time.time() - start,
        "rows": rows,
    }
    path = out_dir / f"dual_task_{lane}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(summary))
    return summary
