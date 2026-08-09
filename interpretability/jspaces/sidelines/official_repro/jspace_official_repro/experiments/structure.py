"""Group C: capacity, ignition, top-down-summoning (plan §10.C)."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from jlens.hooks import ActivationRecorder

from ..layers import PAPER_BAND
from ..paths import EXPERIMENTS_DIR
from ..rendering import (
    find_token_span,
    preferred_token,
    render_chat,
    render_raw,
)
from ..scoring import forward_logits, rank_of
from ..targets import capacity_canon, synonym_token_ids

CAPACITY_TRIALS = 20
CAPACITY_SEED = 20260808
IGNITION_ALPHAS = [round(0.1 * i, 1) for i in range(11)]


# ------------------------------------------------------------- capacity

def run_capacity(model, lens, *, lane: str, out_dir: Path,
                 band=PAPER_BAND) -> dict:
    data = json.loads((EXPERIMENTS_DIR / "capacity.json").read_text())
    tokenizer = model.tokenizer
    start = time.time()
    canons = {
        pool["name"]: capacity_canon(
            tokenizer, pool["pool"], data["targets_per_family"][pool["name"]])
        for pool in data["candidate_pools"]
    }
    trials = []
    for trial_index in range(CAPACITY_TRIALS):
        rng = random.Random(CAPACITY_SEED + trial_index)
        families = list(data["block_families"])
        rng.shuffle(families)
        words: list[tuple[str, str]] = []
        for family in families:
            words.extend((w, family) for w in rng.sample(canons[family], 20))
        text = ", ".join(w for w, _ in words)
        rendered = render_raw(model, text)
        ids = rendered.input_ids[0].tolist()
        comma_positions = [i for i, t in enumerate(ids)
                           if tokenizer.decode([t]).strip() == ","]
        word_tokens = [preferred_token(tokenizer, w) for w, _ in words]
        with ActivationRecorder(model.layers, at=list(band)) as recorder:
            model.forward(rendered.input_ids)
            residuals = {
                layer: recorder.activations[layer][0].detach()
                for layer in band
            }
        # Band-min rank of every seen word at each comma position.
        curve = []
        for comma_index, position in enumerate(comma_positions):
            n_seen = comma_index + 1
            best = {w: None for w in range(n_seen)}
            for layer in band:
                h = residuals[layer][position].float()
                transported = lens.transport(h.unsqueeze(0), layer)[0]
                logits = model.unembed(transported).float().cpu()
                for w in range(n_seen):
                    rank = rank_of(logits, word_tokens[w])
                    if best[w] is None or rank < best[w]:
                        best[w] = rank
            curve.append({
                "n_seen": n_seen,
                "active_rank1": sum(1 for r in best.values() if r == 1),
                "active_rank5": sum(1 for r in best.values() if r <= 5),
                "active_rank20": sum(1 for r in best.values() if r <= 20),
            })
        trials.append({"trial": trial_index, "block_order": families,
                       "n_words": len(words),
                       "n_comma_positions": len(comma_positions),
                       "curve": curve})
    def _plateau(key):
        finals = [t["curve"][-1][key] for t in trials if t["curve"]]
        return sum(finals) / len(finals) if finals else None
    summary = {
        "experiment": "capacity", "lane": lane, "band": list(band),
        "n_trials": CAPACITY_TRIALS, "seed": CAPACITY_SEED,
        "canon_sizes": {k: len(v) for k, v in canons.items()},
        "note": ("task-level capacity (released design); distinct from the "
                 "campaign sparse-occupancy estimator — never merged"),
        "mean_final_active_rank5": _plateau("active_rank5"),
        "mean_final_active_rank20": _plateau("active_rank20"),
        "wall_seconds": time.time() - start,
        "trials": trials,
    }
    path = out_dir / f"capacity_{lane}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(summary))
    return summary


# ------------------------------------------------------------- ignition

def _embedding_swap_forward(model, rendered, *, position, id_a, id_b, alpha,
                            band):
    """Forward with the {W} position's embedding replaced by
    alpha*emb(A) + (1-alpha)*emb(B); returns per-layer logits at {W}."""
    embed = model._embed_tokens if hasattr(model, "_embed_tokens") else None
    if embed is None:
        embed = model._text_module.embed_tokens if hasattr(
            model, "_text_module") else None
    weight = embed.weight.detach()
    mixed = alpha * weight[id_a] + (1 - alpha) * weight[id_b]

    def hook(module, inputs, output):
        output = output.clone()
        output[:, position, :] = mixed.to(output.dtype)
        return output

    handle = embed.register_forward_hook(hook)
    try:
        with ActivationRecorder(model.layers, at=list(band)) as recorder:
            model.forward(rendered.input_ids)
            residuals = {
                layer: recorder.activations[layer][0, position].detach()
                for layer in band
            }
    finally:
        handle.remove()
    per_layer = {}
    for layer in band:
        transported = lens_holder[0].transport(
            residuals[layer].float().unsqueeze(0), layer)[0]
        per_layer[layer] = model.unembed(transported).float().cpu()
    return per_layer


lens_holder = [None]


def run_ignition(model, lens, *, lane: str, out_dir: Path,
                 band=PAPER_BAND) -> dict:
    lens_holder[0] = lens
    data = json.loads((EXPERIMENTS_DIR / "ignition.json").read_text())
    tokenizer = model.tokenizer
    start = time.time()
    countries = data["countries_12"]
    pair_sets = (
        [("country", a, b) for i, a in enumerate(countries)
         for b in countries[i + 1:]]
        + [("alt", w, "France") for w in data["alt_words"]]
        + [("idiom", a, b) for a, b in data["idiom_pairs"]]
        + [("scrambled", a, b) for a, b in data["scrambled_pairs"]]
    )
    carriers = data["ctx_templates"]
    rows = []
    for pair_index, (family, a, b) in enumerate(pair_sets):
        id_a = preferred_token(tokenizer, a)
        id_b = preferred_token(tokenizer, b)
        row = {"family": family, "a": a, "b": b,
               "tokenization_valid": id_a is not None and id_b is not None}
        if not row["tokenization_valid"]:
            row["state"] = "TOKENIZATION_GATED"
            rows.append(row)
            continue
        carrier = carriers[pair_index % len(carriers)]
        text = carrier.format(W=a)
        rendered = render_raw(model, text)
        span = find_token_span(rendered, tokenizer,
                               tokenizer.decode([id_a]).strip())
        position = span[1]
        shares = {}
        for alpha in IGNITION_ALPHAS:
            per_layer = _embedding_swap_forward(
                model, rendered, position=position, id_a=id_a, id_b=id_b,
                alpha=alpha, band=list(band))
            layer_shares = {}
            for layer, logits in per_layer.items():
                rank_a = rank_of(logits, id_a)
                rank_b = rank_of(logits, id_b)
                share = (1 / rank_a) / ((1 / rank_a) + (1 / rank_b))
                layer_shares[layer] = share
            shares[f"a{alpha:g}"] = layer_shares
        row["carrier_index"] = pair_index % len(carriers)
        row["shares"] = shares
        # Per-layer threshold alpha (share crosses 0.5) and 10->90 width.
        thresholds = {}
        widths = {}
        for layer in band:
            profile = [shares[f"a{alpha:g}"][layer] for alpha in IGNITION_ALPHAS]
            crossing = next((IGNITION_ALPHAS[i] for i in range(len(profile))
                             if profile[i] >= 0.5), None)
            thresholds[layer] = crossing
            lo = next((IGNITION_ALPHAS[i] for i in range(len(profile))
                       if profile[i] >= 0.1), None)
            hi = next((IGNITION_ALPHAS[i] for i in range(len(profile))
                       if profile[i] >= 0.9), None)
            widths[layer] = (hi - lo) if lo is not None and hi is not None else None
        row["threshold_alpha"] = thresholds
        row["transition_width"] = widths
        row["state"] = "EXECUTED"
        rows.append(row)
    executed = [r for r in rows if r.get("state") == "EXECUTED"]
    def _mean_width(subset, layer):
        values = [r["transition_width"][layer] for r in subset
                  if r["transition_width"].get(layer) is not None]
        return sum(values) / len(values) if values else None
    summary = {
        "experiment": "ignition", "lane": lane, "band": list(band),
        "alphas": IGNITION_ALPHAS,
        "n_pairs": len(rows), "n_executed": len(executed),
        "mean_transition_width_by_layer": {
            layer: _mean_width(executed, layer) for layer in band},
        "by_family_width_band_mean": {
            family: _mean_width([r for r in executed if r["family"] == family],
                                band[len(band) // 2])
            for family in ("country", "alt", "idiom", "scrambled")},
        "note": "descriptive nonlinearity/competition result (plan §10.C)",
        "wall_seconds": time.time() - start,
        "rows": rows,
    }
    path = out_dir / f"ignition_{lane}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(summary))
    return summary


# ----------------------------------------------------- top-down summoning

def run_top_down(model, lens, *, lane: str, out_dir: Path,
                 band=PAPER_BAND) -> dict:
    from ..interventions import HookPlan, InterventionSession
    from ..readout import token_vectors

    data = json.loads((EXPERIMENTS_DIR / "top-down-summoning.json").read_text())
    tokenizer = model.tokenizer
    start = time.time()
    rows = []
    for item in data["items"]:
        row = {"key": item["key"]}
        renders = {}
        for question_name, question in (("q1", data["q1"]), ("q2", item["q2"])):
            kwargs = {"enable_thinking": False} if lane == "qwen" else {}
            messages = [{"role": "user",
                         "content": f"{item['stimulus']}\n\n{question}"}]
            rendered = render_chat(model, messages,
                                   extra_template_kwargs=kwargs)
            span = find_token_span(rendered, tokenizer,
                                   item["stimulus"][:120], from_end=False)
            stimulus_span = (span[0], span[0] + len(
                tokenizer(item["stimulus"], add_special_tokens=False).input_ids) - 1)
            renders[question_name] = (rendered, stimulus_span)
        expected_ids = synonym_token_ids(tokenizer, item["expected"])
        foil_ids = synonym_token_ids(tokenizer, item["foil"])
        if not expected_ids or not foil_ids:
            row["state"] = "TOKENIZATION_GATED"
            rows.append(row)
            continue
        for question_name, (rendered, span) in renders.items():
            with ActivationRecorder(model.layers, at=list(band)) as recorder:
                model.forward(rendered.input_ids)
                residuals = {
                    layer: recorder.activations[layer][0, span[0]:span[1] + 1]
                    .detach() for layer in band
                }
            positions_hit = 0
            n_positions = span[1] - span[0] + 1
            for offset in range(n_positions):
                hit = False
                for layer in band:
                    h = residuals[layer][offset].float()
                    transported = lens.transport(h.unsqueeze(0), layer)[0]
                    logits = model.unembed(transported).float().cpu()
                    if min(rank_of(logits, t) for t in expected_ids) <= 5:
                        hit = True
                        break
                positions_hit += hit
            row[f"{question_name}_expected_fraction"] = positions_hit / n_positions
        row["q2_minus_q1"] = (row["q2_expected_fraction"]
                              - row["q1_expected_fraction"])
        # Released causal swaps: label <-> foil at every stimulus position.
        swaps = []
        for label_word, foil_word in item["swaps"]:
            source = preferred_token(tokenizer, label_word)
            target = preferred_token(tokenizer, foil_word)
            if source is None or target is None:
                swaps.append({"pair": [label_word, foil_word],
                              "state": "TOKENIZATION_GATED"})
                continue
            for question_name, (rendered, span) in renders.items():
                vectors = {
                    layer: tuple(token_vectors(lens, model, layer,
                                               [source, target]))
                    for layer in band
                }
                plan = HookPlan(layers=list(band),
                                positions=list(range(span[0], span[1] + 1)))
                with InterventionSession(model.layers, plan, kind="swap",
                                         vectors=vectors) as session:
                    logits, _ = forward_logits(
                        model, rendered.input_ids,
                        positions=[rendered.final_position])
                    session.assert_fires(1)
                top1 = int(logits[0].argmax())
                swaps.append({
                    "pair": [label_word, foil_word],
                    "question": question_name,
                    "answer_after": tokenizer.decode([top1]),
                    "foil_rank_after": min(rank_of(logits[0], t)
                                           for t in foil_ids),
                    "expected_rank_after": min(rank_of(logits[0], t)
                                               for t in expected_ids),
                    "state": "EXECUTED",
                })
        row["swaps"] = swaps
        row["state"] = "EXECUTED"
        rows.append(row)
    executed = [r for r in rows if r.get("state") == "EXECUTED"]
    summary = {
        "experiment": "top-down-summoning", "lane": lane, "band": list(band),
        "n_items": len(rows), "n_executed": len(executed),
        "mean_q2_minus_q1": (sum(r["q2_minus_q1"] for r in executed)
                             / len(executed) if executed else None),
        "wall_seconds": time.time() - start,
        "rows": rows,
    }
    path = out_dir / f"top_down_{lane}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(summary))
    return summary
