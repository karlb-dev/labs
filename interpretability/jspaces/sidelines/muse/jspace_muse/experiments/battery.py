"""Compact promising-battery for Muse (best prior open-model cells)."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from jlens.hooks import ActivationRecorder

from ..adapters import load_muse
from ..paths import DEPTH_GRID, DRIVE_ROOT, FINAL_LAYER, PAPER_BAND, ensure_dirs
from ..readout import lens_to_device, preferred_token, rank_of, token_vectors
from ..registry import register
from ..util import atomic_write_json, log, runtime_fingerprint, utc_now


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _band_logits(model, lens, residual_pos: torch.Tensor, layers: list[int]):
    """For one residual vector (d,), return {layer: logits_cpu} after transport."""
    out = {}
    for L in layers:
        h = residual_pos.float().unsqueeze(0)
        transported = lens.transport(h.to(lens.jacobians[L].device), L)[0]
        out[L] = model.unembed(transported).float().cpu()
    return out


def _best_rank(logits_by_layer: dict, token_id: int) -> int:
    return min(rank_of(logits, token_id) for logits in logits_by_layer.values())


# ---------------------------------------------------------------------------
# 1. J vs logit depth profile
# ---------------------------------------------------------------------------

DEPTH_PROMPTS = [
    ("boot_currency", "Fact: The currency used in the country shaped like a boot is",
     ["euro", "Euro", "EUR"]),
    ("capital_japan", "The capital of Japan is", ["Tokyo", "tokyo"]),
    ("gold", "The chemical symbol for gold is", ["Au", "AU"]),
    ("shakespeare", "The author of Romeo and Juliet is William", ["Shakespeare"]),
    ("hot_opposite", "The opposite of hot is", ["cold", "Cold"]),
    ("planet", "The largest planet in the solar system is", ["Jupiter", "jupiter"]),
    ("rome", "Paris is to France as Rome is to", ["Italy", "italy"]),
    ("two_plus", "2 + 2 =", ["4", "four"]),
]


def run_depth_profile(model, lens, *, out_dir: Path) -> dict:
    path = out_dir / "depth_profile_muse.json"
    if path.exists():
        return json.loads(path.read_text())
    layers_j = [L for L in DEPTH_GRID if L in lens.jacobians]
    layers_all = sorted(set(layers_j) | {FINAL_LAYER})
    lens_to_device(lens, "cuda:0", layers=layers_j)
    t0 = time.time()
    rows = []
    for name, prompt, targets in DEPTH_PROMPTS:
        tok_ids = [preferred_token(model.tokenizer, t) for t in targets]
        tok_ids = [t for t in tok_ids if t is not None]
        if not tok_ids:
            rows.append({"name": name, "state": "TOKENIZATION_GATED"})
            continue
        ids = model.encode(prompt, max_length=128)
        pos = int(ids.shape[1] - 1)
        with ActivationRecorder(model.layers, at=layers_all) as rec:
            model.forward(ids)
            acts = {L: rec.activations[L][0, pos].detach() for L in layers_all}
        per = []
        for L in layers_all:
            # logit lens
            logit_logits = model.unembed(acts[L].float().unsqueeze(0))[0].float().cpu()
            logit_rank = min(rank_of(logit_logits, t) for t in tok_ids)
            logit_top = [model.tokenizer.decode([i]) for i in logit_logits.topk(5).indices]
            # jlens (if available)
            if L in lens.jacobians:
                tr = lens.transport(acts[L].float().unsqueeze(0).to(lens.jacobians[L].device), L)[0]
                j_logits = model.unembed(tr).float().cpu()
                j_rank = min(rank_of(j_logits, t) for t in tok_ids)
                j_top = [model.tokenizer.decode([i]) for i in j_logits.topk(5).indices]
            else:
                j_rank, j_top = None, None
            per.append({
                "layer": L,
                "logit_rank": logit_rank,
                "j_rank": j_rank,
                "logit_top5": logit_top,
                "j_top5": j_top,
            })
        rows.append({"name": name, "prompt": prompt, "targets": targets, "per_layer": per})
    # summary: mean best rank in early/mid/late for j vs logit
    def mean_rank(kind: str, layer_pred):
        vals = []
        for r in rows:
            if "per_layer" not in r:
                continue
            for p in r["per_layer"]:
                if layer_pred(p["layer"]):
                    v = p["j_rank"] if kind == "j" else p["logit_rank"]
                    if v is not None:
                        vals.append(v)
        return sum(vals) / len(vals) if vals else None

    summary = {
        "experiment": "depth_profile",
        "n_prompts": len(rows),
        "mean_logit_rank_early": mean_rank("logit", lambda L: L <= 16),
        "mean_j_rank_early": mean_rank("j", lambda L: L <= 16),
        "mean_logit_rank_band": mean_rank("logit", lambda L: 20 <= L <= 46),
        "mean_j_rank_band": mean_rank("j", lambda L: 20 <= L <= 46),
        "mean_logit_rank_late": mean_rank("logit", lambda L: L >= 48),
        "mean_j_rank_late": mean_rank("j", lambda L: L >= 48),
        "wall_seconds": time.time() - t0,
        "rows": rows,
        "runtime": runtime_fingerprint(),
    }
    # advantage: lower rank is better
    if summary["mean_j_rank_band"] and summary["mean_logit_rank_band"]:
        summary["j_advantage_band"] = (
            summary["mean_logit_rank_band"] - summary["mean_j_rank_band"]
        )
    atomic_write_json(summary, path)
    log(f"depth_profile: j_adv_band={summary.get('j_advantage_band')}")
    return summary


# ---------------------------------------------------------------------------
# 2. Selectivity-language (compact inline stimuli)
# ---------------------------------------------------------------------------

LANG_PASSAGES = [
    {
        "key": "fr1",
        "language": "French",
        "text": (
            "Marie walks through Paris every morning. She buys a croissant "
            "at the bakery near the Seine and greets her neighbors in the street."
        ),
        "labels": ["French", "français", "Francais"],
    },
    {
        "key": "de1",
        "language": "German",
        "text": (
            "Hans arbeitet in Berlin. Jeden Abend trinkt er ein Bier mit "
            "Freunden und spricht über Fußball und die Arbeit."
        ),
        "labels": ["German", "Deutsch", "deutsch"],
    },
    {
        "key": "es1",
        "language": "Spanish",
        "text": (
            "Ana vive en Madrid. Le gusta el sol, el flamenco y comer "
            "tapas con su familia los domingos por la tarde."
        ),
        "labels": ["Spanish", "español", "Espanol"],
    },
    {
        "key": "it1",
        "language": "Italian",
        "text": (
            "Marco abita a Roma. Ogni mattina beve un caffè e legge il "
            "giornale prima di andare al lavoro in centro."
        ),
        "labels": ["Italian", "italiano", "Italiano"],
    },
]


def run_selectivity_language(model, lens, *, out_dir: Path, band=PAPER_BAND) -> dict:
    path = out_dir / "selectivity_language_muse.json"
    if path.exists():
        return json.loads(path.read_text())
    layers = [L for L in band if L in lens.jacobians]
    lens_to_device(lens, "cuda:0", layers=layers)
    t0 = time.time()
    rows = []
    for p in LANG_PASSAGES:
        label_ids = [preferred_token(model.tokenizer, lab) for lab in p["labels"]]
        label_ids = [t for t in label_ids if t is not None]
        entry = {"key": p["key"], "language": p["language"], "labels": p["labels"]}
        if not label_ids:
            entry["state"] = "TOKENIZATION_GATED"
            rows.append(entry)
            continue
        for cond, template in (
            ("explicit", "Passage:\n{text}\n\nQuestion: What language is this passage written in?\nAnswer:"),
            ("automatic", "Passage:\n{text}\n\n"),
        ):
            text = template.format(text=p["text"])
            ids = model.encode(text, max_length=256)
            # score over last 8 tokens of the prompt
            seq = int(ids.shape[1])
            span = list(range(max(0, seq - 8), seq))
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(ids)
                acts = {L: rec.activations[L][0].detach() for L in layers}
            hit = False
            best = 10**9
            for pos in span:
                for L in layers:
                    tr = lens.transport(
                        acts[L][pos].float().unsqueeze(0).to(lens.jacobians[L].device), L
                    )[0]
                    logits = model.unembed(tr).float().cpu()
                    r = min(rank_of(logits, t) for t in label_ids)
                    best = min(best, r)
                    if r == 1:
                        hit = True
            entry[cond] = {"hit_rank1": hit, "best_rank": best}
        rows.append(entry)

    def rate(cond):
        vals = [1.0 if r.get(cond, {}).get("hit_rank1") else 0.0
                for r in rows if cond in r]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "experiment": "selectivity_language",
        "explicit_rate": rate("explicit"),
        "automatic_rate": rate("automatic"),
        "contrast": (rate("explicit") - rate("automatic"))
        if rate("explicit") is not None and rate("automatic") is not None
        else None,
        "n": len(rows),
        "wall_seconds": time.time() - t0,
        "rows": rows,
        "runtime": runtime_fingerprint(),
    }
    atomic_write_json(summary, path)
    log(f"selectivity_language: contrast={summary['contrast']}")
    return summary


# ---------------------------------------------------------------------------
# 3. Directed modulation (white-bear style)
# ---------------------------------------------------------------------------

MOD_CONCEPTS = [
    ("bear", ["bear", "Bear"]),
    ("piano", ["piano", "Piano"]),
    ("ocean", ["ocean", "Ocean"]),
    ("volcano", ["volcano", "Volcano"]),
    ("pyramid", ["pyramid", "Pyramid"]),
    ("galaxy", ["galaxy", "Galaxy"]),
]


def run_modulation(model, lens, *, out_dir: Path, band=PAPER_BAND) -> dict:
    path = out_dir / "directed_modulation_muse.json"
    if path.exists():
        return json.loads(path.read_text())
    layers = [L for L in band if L in lens.jacobians]
    mid = layers[len(layers) // 2]
    lens_to_device(lens, "cuda:0", layers=[mid])
    t0 = time.time()
    rows = []
    for concept, forms in MOD_CONCEPTS:
        tok = preferred_token(model.tokenizer, forms[0])
        if tok is None:
            rows.append({"concept": concept, "state": "TOKENIZATION_GATED"})
            continue
        v = token_vectors(lens, model, mid, [tok])[0]  # cpu
        v = v / (v.norm() + 1e-8)
        base_prompt = "Write a short story about a quiet village.\nStory:"
        ids = model.encode(base_prompt, max_length=64)
        pos = int(ids.shape[1] - 1)

        def score_with_alpha(alpha: float) -> int:
            handle = None

            def hook(module, inp, out):
                if isinstance(out, tuple):
                    h = out[0].clone()
                    h[0, pos] = h[0, pos] + (alpha * v).to(h.device).to(h.dtype)
                    return (h,) + out[1:]
                h = out.clone()
                h[0, pos] = h[0, pos] + (alpha * v).to(h.device).to(h.dtype)
                return h

            handle = model.layers[mid].register_forward_hook(hook)
            try:
                with ActivationRecorder(model.layers, at=[mid]) as rec:
                    model.forward(ids)
                    h = rec.activations[mid][0, pos].detach()
                tr = lens.transport(h.float().unsqueeze(0).to(lens.jacobians[mid].device), mid)[0]
                logits = model.unembed(tr).float().cpu()
                return rank_of(logits, tok)
            finally:
                handle.remove()

        ranks = {f"alpha_{a}": score_with_alpha(float(a)) for a in (0.0, 1.0, 2.0, -1.0)}
        rows.append({"concept": concept, "token_id": tok, "layer": mid, **ranks})

    focus = [r["alpha_2.0"] for r in rows if "alpha_2.0" in r]
    suppress = [r["alpha_-1.0"] for r in rows if "alpha_-1.0" in r]
    base = [r["alpha_0.0"] for r in rows if "alpha_0.0" in r]
    summary = {
        "experiment": "directed_modulation",
        "layer": mid,
        "mean_rank_alpha0": sum(base) / len(base) if base else None,
        "mean_rank_focus_a2": sum(focus) / len(focus) if focus else None,
        "mean_rank_suppress_am1": sum(suppress) / len(suppress) if suppress else None,
        "focus_improves": (
            (sum(focus) / len(focus)) < (sum(base) / len(base))
            if focus and base else None
        ),
        "wall_seconds": time.time() - t0,
        "rows": rows,
        "runtime": runtime_fingerprint(),
    }
    atomic_write_json(summary, path)
    log(f"modulation: focus_improves={summary['focus_improves']} "
        f"a0={summary['mean_rank_alpha0']} a2={summary['mean_rank_focus_a2']}")
    return summary


# ---------------------------------------------------------------------------
# 4. Dual-task interference (compact)
# ---------------------------------------------------------------------------

DUAL_ITEMS = [
    {"math": "Compute 17 + 28.", "concept": "lion", "answer": "45"},
    {"math": "Compute 12 * 4.", "concept": "eagle", "answer": "48"},
    {"math": "Compute 100 - 37.", "concept": "shark", "answer": "63"},
    {"math": "Compute 9 * 7.", "concept": "spider", "answer": "63"},
    {"math": "Compute 15 + 16.", "concept": "piano", "answer": "31"},
    {"math": "Compute 8 * 8.", "concept": "violin", "answer": "64"},
]


def run_dual_task(model, lens, *, out_dir: Path, band=PAPER_BAND) -> dict:
    path = out_dir / "dual_task_muse.json"
    if path.exists():
        return json.loads(path.read_text())
    layers = [L for L in band if L in lens.jacobians]
    mid = layers[len(layers) // 2]
    lens_to_device(lens, "cuda:0", layers=[mid])
    t0 = time.time()
    rows = []
    for item in DUAL_ITEMS:
        ctok = preferred_token(model.tokenizer, item["concept"])
        atok = preferred_token(model.tokenizer, item["answer"])
        if ctok is None or atok is None:
            rows.append({**item, "state": "TOKENIZATION_GATED"})
            continue
        single_m = f"{item['math']}\nAnswer:"
        single_c = f"Name an animal: {item['concept']}\nThe word is"
        dual = f"{item['math']} Also keep the word '{item['concept']}' in mind.\nAnswer:"

        def ranks_for(prompt: str):
            ids = model.encode(prompt, max_length=128)
            pos = int(ids.shape[1] - 1)
            with ActivationRecorder(model.layers, at=[mid]) as rec:
                model.forward(ids)
                h = rec.activations[mid][0, pos].detach()
            tr = lens.transport(h.float().unsqueeze(0).to(lens.jacobians[mid].device), mid)[0]
            logits = model.unembed(tr).float().cpu()
            return {"math_rank": rank_of(logits, atok), "concept_rank": rank_of(logits, ctok)}

        sm, sc, d = ranks_for(single_m), ranks_for(single_c), ranks_for(dual)
        rows.append({
            **item,
            "single_math": sm,
            "single_concept": sc,
            "dual": d,
            "math_interference": d["math_rank"] - sm["math_rank"],
            "concept_interference": d["concept_rank"] - sc["concept_rank"],
        })
    mi = [r["math_interference"] for r in rows if "math_interference" in r]
    ci = [r["concept_interference"] for r in rows if "concept_interference" in r]
    summary = {
        "experiment": "dual_task",
        "mean_math_interference": sum(mi) / len(mi) if mi else None,
        "mean_concept_interference": sum(ci) / len(ci) if ci else None,
        "wall_seconds": time.time() - t0,
        "rows": rows,
        "runtime": runtime_fingerprint(),
    }
    atomic_write_json(summary, path)
    log(f"dual_task: math_if={summary['mean_math_interference']} "
        f"concept_if={summary['mean_concept_interference']}")
    return summary


# ---------------------------------------------------------------------------
# 5. Capacity (compact word-list)
# ---------------------------------------------------------------------------

CAPACITY_POOLS = {
    "animals": ["lion", "tiger", "eagle", "shark", "whale", "otter", "panda", "zebra",
                "moose", "goose", "raven", "cobra", "llama", "bison", "crane", "finch",
                "heron", "hippo", "lemur", "moose"],
    "colors": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white",
               "brown", "pink", "gray", "cyan", "magenta", "teal", "maroon", "navy",
               "gold", "silver", "beige", "ivory"],
    "instruments": ["piano", "violin", "guitar", "flute", "drum", "harp", "cello", "oboe",
                    "tuba", "banjo", "organ", "viola", "bugle", "sitar", "ukulele",
                    "clarinet", "trombone", "trumpet", "bagpipe", "xylophone"],
}


def run_capacity(model, lens, *, out_dir: Path, band=PAPER_BAND, n_trials: int = 8) -> dict:
    path = out_dir / "capacity_muse.json"
    if path.exists():
        return json.loads(path.read_text())
    layers = [L for L in band if L in lens.jacobians]
    lens_to_device(lens, "cuda:0", layers=layers)
    t0 = time.time()
    trials = []
    for trial in range(n_trials):
        rng = random.Random(20260811 + trial)
        words = []
        for fam, pool in CAPACITY_POOLS.items():
            words.extend(rng.sample(pool, 8))
        rng.shuffle(words)
        text = ", ".join(words)
        ids = model.encode(text, max_length=256)
        tok_ids = ids[0].tolist()
        # positions after each comma-separated word: use positions of commas + final
        comma_str_id = preferred_token(model.tokenizer, ",")
        # fallback: score at final position only for a simple active-set estimate
        pos = int(ids.shape[1] - 1)
        with ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(ids)
            acts = {L: rec.activations[L][0, pos].detach() for L in layers}
        word_tokens = []
        for w in words:
            t = preferred_token(model.tokenizer, w)
            if t is not None:
                word_tokens.append((w, t))
        best = {}
        for w, t in word_tokens:
            rbest = 10**9
            for L in layers:
                tr = lens.transport(
                    acts[L].float().unsqueeze(0).to(lens.jacobians[L].device), L
                )[0]
                logits = model.unembed(tr).float().cpu()
                rbest = min(rbest, rank_of(logits, t))
            best[w] = rbest
        trials.append({
            "trial": trial,
            "n_words": len(word_tokens),
            "active_rank1": sum(1 for r in best.values() if r == 1),
            "active_rank5": sum(1 for r in best.values() if r <= 5),
            "active_rank20": sum(1 for r in best.values() if r <= 20),
            "ranks": best,
        })
    summary = {
        "experiment": "capacity",
        "n_trials": n_trials,
        "mean_active_rank5": sum(t["active_rank5"] for t in trials) / len(trials),
        "mean_active_rank20": sum(t["active_rank20"] for t in trials) / len(trials),
        "mean_active_rank1": sum(t["active_rank1"] for t in trials) / len(trials),
        "wall_seconds": time.time() - t0,
        "trials": trials,
        "note": "end-of-list band-min ranks; descriptive working-set estimate",
        "runtime": runtime_fingerprint(),
    }
    atomic_write_json(summary, path)
    log(f"capacity: mean_r5={summary['mean_active_rank5']:.2f} "
        f"mean_r20={summary['mean_active_rank20']:.2f}")
    return summary


# ---------------------------------------------------------------------------
# 6. Ignition (alpha ramp on concept pair)
# ---------------------------------------------------------------------------

IGNITION_PAIRS = [
    ("lion", "tiger"),
    ("piano", "violin"),
    ("ocean", "river"),
    ("paris", "london"),
    ("apple", "orange"),
    ("doctor", "lawyer"),
]


def run_ignition(model, lens, *, out_dir: Path, band=PAPER_BAND) -> dict:
    path = out_dir / "ignition_muse.json"
    if path.exists():
        return json.loads(path.read_text())
    layers = [L for L in band if L in lens.jacobians]
    mid = layers[len(layers) // 2]
    late = layers[-1]
    lens_to_device(lens, "cuda:0", layers=[mid, late])
    alphas = [round(0.2 * i, 1) for i in range(11)]
    t0 = time.time()
    rows = []
    prompt = "The animal in the cage is a"
    ids = model.encode(prompt, max_length=64)
    pos = int(ids.shape[1] - 1)
    for a_word, b_word in IGNITION_PAIRS:
        ta = preferred_token(model.tokenizer, a_word)
        tb = preferred_token(model.tokenizer, b_word)
        if ta is None or tb is None:
            rows.append({"pair": [a_word, b_word], "state": "TOKENIZATION_GATED"})
            continue
        va = token_vectors(lens, model, mid, [ta])[0]
        vb = token_vectors(lens, model, mid, [tb])[0]
        va = va / (va.norm() + 1e-8)
        vb = vb / (vb.norm() + 1e-8)
        curve = []
        for alpha in alphas:
            # inject alpha*va - (1-ish)*vb competition via +alpha va
            delta = alpha * va

            def hook(module, inp, out, d=delta):
                if isinstance(out, tuple):
                    h = out[0].clone()
                    h[0, pos] = h[0, pos] + d.to(h.device).to(h.dtype)
                    return (h,) + out[1:]
                h = out.clone()
                h[0, pos] = h[0, pos] + d.to(h.device).to(h.dtype)
                return h

            handle = model.layers[mid].register_forward_hook(hook)
            try:
                with ActivationRecorder(model.layers, at=[late]) as rec:
                    model.forward(ids)
                    h = rec.activations[late][0, pos].detach()
                tr = lens.transport(h.float().unsqueeze(0).to(lens.jacobians[late].device), late)[0]
                logits = model.unembed(tr).float().cpu()
                ra, rb = rank_of(logits, ta), rank_of(logits, tb)
                curve.append({"alpha": alpha, "rank_a": ra, "rank_b": rb,
                              "winner_a": ra < rb})
            finally:
                handle.remove()
        # sharpness: how fast rank_a drops below 5
        first_hit = next((c["alpha"] for c in curve if c["rank_a"] <= 5), None)
        rows.append({
            "pair": [a_word, b_word], "layer_inject": mid, "layer_read": late,
            "curve": curve, "alpha_rank5": first_hit,
        })
    summary = {
        "experiment": "ignition",
        "n_pairs": len(rows),
        "median_alpha_rank5": sorted(
            c["alpha_rank5"] for c in rows if c.get("alpha_rank5") is not None
        ),
        "wall_seconds": time.time() - t0,
        "rows": rows,
        "runtime": runtime_fingerprint(),
    }
    if summary["median_alpha_rank5"]:
        vals = summary["median_alpha_rank5"]
        summary["median_alpha_rank5"] = vals[len(vals) // 2]
    else:
        summary["median_alpha_rank5"] = None
    atomic_write_json(summary, path)
    log(f"ignition: median_alpha_rank5={summary['median_alpha_rank5']}")
    return summary


# ---------------------------------------------------------------------------
# 7. Verbal report (coordinate swap, small n)
# ---------------------------------------------------------------------------

VR_ITEMS = [
    ("France", "Paris", "Canada", "Ottawa"),
    ("France", "Paris", "China", "Beijing"),
    ("France", "Paris", "Egypt", "Cairo"),
    ("Canada", "Ottawa", "France", "Paris"),
    ("China", "Beijing", "France", "Paris"),
    ("Egypt", "Cairo", "France", "Paris"),
    ("Japan", "Tokyo", "Italy", "Rome"),
    ("Italy", "Rome", "Japan", "Tokyo"),
    ("Germany", "Berlin", "Spain", "Madrid"),
    ("Spain", "Madrid", "Germany", "Berlin"),
]


def run_verbal_report(model, lens, *, out_dir: Path, band=PAPER_BAND) -> dict:
    """Paper-literal-ish α=1 country→capital coordinate swap, small n."""
    path = out_dir / "verbal_report_muse.json"
    if path.exists():
        return json.loads(path.read_text())
    layers = [L for L in band if L in lens.jacobians]
    mid = layers[len(layers) // 2]
    lens_to_device(lens, "cuda:0", layers=[mid])
    t0 = time.time()
    trials = []
    for src_country, src_cap, dst_country, dst_cap in VR_ITEMS:
        t_src = preferred_token(model.tokenizer, src_country)
        t_dst = preferred_token(model.tokenizer, dst_country)
        t_src_cap = preferred_token(model.tokenizer, src_cap)
        t_dst_cap = preferred_token(model.tokenizer, dst_cap)
        if None in (t_src, t_dst, t_src_cap, t_dst_cap):
            trials.append({"src": src_country, "dst": dst_country,
                           "state": "TOKENIZATION_GATED"})
            continue
        # vectors at mid layer
        v_src = token_vectors(lens, model, mid, [t_src])[0]
        v_dst = token_vectors(lens, model, mid, [t_dst])[0]
        delta = v_dst - v_src  # swap direction
        prompt = f"Fact: The capital of {src_country} is"
        ids = model.encode(prompt, max_length=64)
        # find country token position (last occurrence)
        id_list = ids[0].tolist()
        try:
            cpos = len(id_list) - 1 - id_list[::-1].index(t_src)
        except ValueError:
            cpos = int(ids.shape[1] - 2)

        def run_alpha(alpha: float):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    h = out[0].clone()
                    h[0, cpos] = h[0, cpos] + (alpha * delta).to(h.device).to(h.dtype)
                    return (h,) + out[1:]
                h = out.clone()
                h[0, cpos] = h[0, cpos] + (alpha * delta).to(h.device).to(h.dtype)
                return h
            handle = model.layers[mid].register_forward_hook(hook)
            try:
                with ActivationRecorder(model.layers, at=[mid]) as rec:
                    model.forward(ids)
                    # read at final position
                    fpos = int(ids.shape[1] - 1)
                    h = rec.activations[mid][0, fpos].detach()
                tr = lens.transport(h.float().unsqueeze(0).to(lens.jacobians[mid].device), mid)[0]
                logits = model.unembed(tr).float().cpu()
                return {
                    "rank_src_cap": rank_of(logits, t_src_cap),
                    "rank_dst_cap": rank_of(logits, t_dst_cap),
                    "top5": [model.tokenizer.decode([i]) for i in logits.topk(5).indices],
                }
            finally:
                handle.remove()

        base = run_alpha(0.0)
        swapped = run_alpha(1.0)
        trials.append({
            "src": src_country, "dst": dst_country,
            "src_cap": src_cap, "dst_cap": dst_cap,
            "layer": mid, "alpha0": base, "alpha1": swapped,
            "top5_hit_dst": dst_cap.lower().strip() in
            [t.lower().strip() for t in swapped["top5"]]
            or any(dst_cap.lower() in t.lower() for t in swapped["top5"]),
            "rank_dst_improved": swapped["rank_dst_cap"] < base["rank_dst_cap"],
        })
    hits = [t for t in trials if t.get("top5_hit_dst")]
    summary = {
        "experiment": "verbal_report",
        "n": len(trials),
        "top5_hit_rate": len(hits) / len(trials) if trials else None,
        "n_rank_improved": sum(1 for t in trials if t.get("rank_dst_improved")),
        "wall_seconds": time.time() - t0,
        "trials": trials,
        "runtime": runtime_fingerprint(),
    }
    atomic_write_json(summary, path)
    log(f"verbal_report: top5_hit_rate={summary['top5_hit_rate']}")
    return summary


# ---------------------------------------------------------------------------
# 8. Protected J-ablation smoke (direction vs energy-matched random)
# ---------------------------------------------------------------------------

ABLATION_PROMPTS = [
    ("capital_france", "The capital of France is", "Paris"),
    ("currency_japan", "The currency of Japan is the", "yen"),
    ("author_hamlet", "Hamlet was written by William", "Shakespeare"),
    ("element_fe", "The chemical symbol for iron is", "Fe"),
    ("planet_red", "The red planet is", "Mars"),
    ("einstein", "E = mc^2 was proposed by", "Einstein"),
]


def run_protected_ablation(model, lens, *, out_dir: Path, band=PAPER_BAND) -> dict:
    path = out_dir / "protected_ablation_muse.json"
    if path.exists():
        return json.loads(path.read_text())
    layers = [L for L in band if L in lens.jacobians]
    mid = layers[len(layers) // 2]
    lens_to_device(lens, "cuda:0", layers=[mid])
    t0 = time.time()
    rows = []
    rng = random.Random(0)
    for name, prompt, target in ABLATION_PROMPTS:
        tok = preferred_token(model.tokenizer, target)
        if tok is None:
            rows.append({"name": name, "state": "TOKENIZATION_GATED"})
            continue
        v = token_vectors(lens, model, mid, [tok])[0]
        v = v / (v.norm() + 1e-8)
        # random energy-matched direction
        r = torch.randn_like(v)
        r = r / (r.norm() + 1e-8)
        ids = model.encode(prompt, max_length=64)
        pos = int(ids.shape[1] - 1)

        def score(delta: torch.Tensor | None):
            if delta is None:
                with ActivationRecorder(model.layers, at=[mid]) as rec:
                    model.forward(ids)
                    h = rec.activations[mid][0, pos].detach()
            else:
                def hook(module, inp, out):
                    # project-out / ablate: subtract projection onto delta direction
                    if isinstance(out, tuple):
                        h = out[0].clone()
                        vec = delta.to(h.device).to(h.dtype)
                        # remove component along vec
                        coeff = (h[0, pos].float() @ vec.float()).to(h.dtype)
                        h[0, pos] = h[0, pos] - coeff * vec
                        return (h,) + out[1:]
                    h = out.clone()
                    vec = delta.to(h.device).to(h.dtype)
                    coeff = (h[0, pos].float() @ vec.float()).to(h.dtype)
                    h[0, pos] = h[0, pos] - coeff * vec
                    return h
                handle = model.layers[mid].register_forward_hook(hook)
                try:
                    with ActivationRecorder(model.layers, at=[mid]) as rec:
                        model.forward(ids)
                        h = rec.activations[mid][0, pos].detach()
                finally:
                    handle.remove()
            tr = lens.transport(h.float().unsqueeze(0).to(lens.jacobians[mid].device), mid)[0]
            logits = model.unembed(tr).float().cpu()
            return rank_of(logits, tok)

        base = score(None)
        abl_j = score(v)
        abl_r = score(r)
        rows.append({
            "name": name, "target": target, "layer": mid,
            "rank_base": base, "rank_ablate_j": abl_j, "rank_ablate_random": abl_r,
            "j_damage": abl_j - base, "random_damage": abl_r - base,
            "selective": (abl_j - base) > (abl_r - base) + 2,
        })
    j_d = [r["j_damage"] for r in rows if "j_damage" in r]
    r_d = [r["random_damage"] for r in rows if "random_damage" in r]
    summary = {
        "experiment": "protected_ablation",
        "mean_j_damage": sum(j_d) / len(j_d) if j_d else None,
        "mean_random_damage": sum(r_d) / len(r_d) if r_d else None,
        "n_selective": sum(1 for r in rows if r.get("selective")),
        "n": len(rows),
        "wall_seconds": time.time() - t0,
        "rows": rows,
        "runtime": runtime_fingerprint(),
    }
    atomic_write_json(summary, path)
    log(f"ablation: j_dmg={summary['mean_j_damage']} rnd={summary['mean_random_damage']} "
        f"selective={summary['n_selective']}/{summary['n']}")
    return summary


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_all() -> dict:
    ensure_dirs()
    out_dir = DRIVE_ROOT / "metrics"
    lens_path = DRIVE_ROOT / "lens" / "muse_glimmer_lens.pt"
    if not lens_path.exists():
        raise FileNotFoundError(f"need fitted lens at {lens_path}")

    from jlens.lens import JacobianLens

    model, hf_model, tokenizer = load_muse()
    lens = JacobianLens.load(str(lens_path))
    log(f"lens layers={sorted(lens.jacobians.keys())} n_prompts={lens.n_prompts}")

    # post-fit admission first
    from .admission import run_post_fit
    post = run_post_fit(model, lens, out_dir=out_dir)

    cells = [
        ("depth_profile", run_depth_profile),
        ("selectivity_language", run_selectivity_language),
        ("directed_modulation", run_modulation),
        ("dual_task", run_dual_task),
        ("capacity", run_capacity),
        ("ignition", run_ignition),
        ("verbal_report", run_verbal_report),
        ("protected_ablation", run_protected_ablation),
    ]
    results = {"admission_post_fit": post, "utc": utc_now()}
    for name, fn in cells:
        log(f"=== battery cell: {name} ===")
        try:
            results[name] = fn(model, lens, out_dir=out_dir)
        except Exception as e:  # noqa: BLE001
            log(f"CELL FAIL {name}: {type(e).__name__}: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            atomic_write_json(
                results[name], out_dir / f"{name}_muse_ERROR.json"
            )

    summary_path = out_dir / "battery_summary.json"
    headline = {
        "utc": utc_now(),
        "depth_j_advantage_band": results.get("depth_profile", {}).get("j_advantage_band"),
        "selectivity_contrast": results.get("selectivity_language", {}).get("contrast"),
        "modulation_focus_improves": results.get("directed_modulation", {}).get("focus_improves"),
        "dual_math_interference": results.get("dual_task", {}).get("mean_math_interference"),
        "capacity_mean_r5": results.get("capacity", {}).get("mean_active_rank5"),
        "ignition_median_alpha": results.get("ignition", {}).get("median_alpha_rank5"),
        "vr_top5_hit_rate": results.get("verbal_report", {}).get("top5_hit_rate"),
        "ablation_j_damage": results.get("protected_ablation", {}).get("mean_j_damage"),
        "ablation_random_damage": results.get("protected_ablation", {}).get("mean_random_damage"),
        "ablation_n_selective": results.get("protected_ablation", {}).get("n_selective"),
        "g_fold_min_cosine": post.get("g_folding", {}).get("min_cosine"),
        "readout_parity_ok": post.get("readout_parity", {}).get("ok"),
    }
    results["headline"] = headline
    atomic_write_json(results, summary_path)
    register({
        "evidence_id": "muse-battery-v1",
        "what": "Muse compact promising-battery (8 cells + post-fit admission)",
        "command": "python -m jspace_muse.experiments.battery",
        "outputs": [summary_path],
        "headline": headline,
    })
    log(f"BATTERY DONE headline={headline}")
    return results


if __name__ == "__main__":
    run_all()
