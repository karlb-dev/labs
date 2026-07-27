# Core Battery frozen grid for a matrix model — the marquee causal
# instrument, now with the frozen-logit column built in:
#   none / frozen_j10 / frozen_rand10 / frozen_logit10
# on the part-1 battery + the C3 hard one-hop cells (skipped with a log if
# the frozen set isn't built yet). Replicates:
#   --seed 1   redrawn random dictionary (rng 12000+layer) AND fresh two-hop
#              items (probe-swap [60:90]) — the deliberately-harder replicate
#   --temp 0.7 sampled decisive cells (twohop, arithmetic_v2) — closes the
#              greedy-only caveat
# Mechanism, band, K, scorers, bootstrap: part-1 code imported unchanged
# (s7/s12). Resumable per (condition, task).
#
# Usage: python scripts/p2core_frozen_grid.py --model olmo31-instruct \
#          --lens <path/to/lens.pt> [--seed 0] [--temp 0] [--force]
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import (RUN_DIR_P2, atomic_write_json, die, log,
                        p2_load_model, p2_metrics_dir, read_json, seed_all,
                        variant_first_ids)

import numpy as np
import torch
from jlens import JacobianLens

PART1_SCRIPTS = Path(__file__).resolve().parents[1] / "part1" / "scripts"
sys.path.insert(0, str(PART1_SCRIPTS))
import s7_ablation as s7  # noqa: E402
import s12_frozen_ablation as s12  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2b3_frozen_logit import build_logit_dict  # noqa: E402

HARD_ONEHOP = RUN_DIR_P2 / "config" / "prompts" / "hard_onehop.jsonl"
CORE_TASKS = ["twohop", "onehop", "arithmetic_v2", "sql", "prose_nll",
              "grammar", "twohop_lp", "samples"]
HARD_TASKS = ["hard_onehop", "hard_onehop_lp"]


def arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def build_rand_dict(d_model: int, seed_base: int):
    rd = {}
    for l in s7.BAND:
        rng = np.random.default_rng(seed_base + l)
        R = torch.tensor(rng.standard_normal((5120, d_model)),
                         device="cuda", dtype=torch.float32)
        rd[l] = torch.nn.functional.normalize(R, dim=1).half()
    return rd


def run_frozen_items(cond, dicts, ab, model, hf, tok, items, scorer):
    """Per-item frozen selection + scoring for task lists s12 doesn't know
    (the C3 hard one-hop cells). scorer ∈ {'onehop','twohop_lp'} reuses the
    part-1 scorer branches via singleton task dicts."""
    scores = []
    for it in items:
        ab.mode = None
        proj = s12.frozen_projectors(model, dicts, it["prompt"])
        ab.mode = ("static", proj)
        key = "onehop" if scorer == "onehop" else "twohop"
        scores += s7.run_task(scorer, cond, model, hf, tok, {key: [it]})
        ab.mode = None
    return scores


@torch.no_grad()
def sampled_next(model, tok, prompt: str, temp: float, rng) -> int:
    ids = model.encode(prompt, max_length=512)
    logits = model.unembed(s7._last_hidden(model, ids))[0, -1].float()
    p = torch.softmax(logits / temp, dim=-1).cpu().numpy().astype(np.float64)
    return int(rng.choice(len(p), p=p / p.sum()))


@torch.no_grad()
def sampled_gen(hf, tok, prompt: str, max_new: int, temp: float) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    out = hf.generate(ids, max_new_tokens=max_new, do_sample=True,
                      temperature=temp, top_p=1.0,
                      pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def run_task_sampled(name, model, hf, tok, tasks, temp, seed):
    import re
    rng = np.random.default_rng(31337 + seed)
    scores = []
    if name == "twohop":
        for it in tasks["twohop"]:
            nid = sampled_next(model, tok, it["prompt"], temp, rng)
            scores.append(float(nid in variant_first_ids(tok, it["answer"])))
    elif name == "arithmetic_v2":
        torch.manual_seed(31337 + seed)
        for it in tasks["arithmetic"]:
            seg = sampled_gen(hf, tok, it["prompt"], 64, temp).split("\nQ:")[0]
            scores.append(float(it["answer"] in re.findall(r"-?\d+", seg)))
    else:
        die(f"no sampled scorer for {name}")
    return scores


def main() -> None:
    seed_all()
    slug = arg("--model", "olmo31-instruct")
    lens_path = Path(arg("--lens", ""))
    seed = int(arg("--seed", "0"))
    temp = float(arg("--temp", "0"))
    if not lens_path.exists():
        die(f"--lens required and must exist (got {lens_path!r})")
    suffix = ("_seed1" if seed else "") + ("_t07" if temp else "")
    out_path = p2_metrics_dir(slug) / f"core_frozen_grid{suffix}.json"

    task_names = list(CORE_TASKS)
    if temp:
        task_names = ["twohop", "arithmetic_v2"]  # decisive sampled cells
    hard_items = None
    if HARD_ONEHOP.exists() and not temp:
        hard_items = [json.loads(l) for l in
                      HARD_ONEHOP.read_text().splitlines() if l.strip()]
        task_names += HARD_TASKS
    elif not temp:
        log("hard_onehop.jsonl not built yet — C3 cells skipped this run")

    conds = ["none", "frozen_j10", "frozen_rand10", "frozen_logit10"]
    res = read_json(out_path) if out_path.exists() else {
        "model_slug": slug, "lens": lens_path.name, "band": s7.BAND,
        "k": s12.K, "skip": s12.SKIP, "seed": seed, "temp": temp,
        "conditions": {}}
    if all(t in res["conditions"].get(c, {}) for c in conds
           for t in task_names) and "--force" not in sys.argv:
        log(f"{out_path} complete; skipping")
        return

    lens = JacobianLens.load(str(lens_path))
    model, hf, tok = p2_load_model(slug)
    jd, rd = s12.build_dicts(lens, hf)         # J-dict + seed-2000 rand twin
    if seed:                                    # redrawn pool for the replicate
        del rd
        rd = build_rand_dict(model.d_model, 12000)
    ld = build_logit_dict(hf)
    log(f"dictionaries built (J / rand seed-base {12000 if seed else 2000} / logit)")

    rng = np.random.default_rng(0)
    tasks = s7.build_tasks(rng)
    if seed:  # deliberately-harder replicate: fresh two-hop items
        ps = json.loads(s7.PROBE_SWAP.read_text())["items"][60:90]
        tasks["twohop"] = [{"prompt": it["prompt"], "answer": it["answer"]}
                           for it in ps]

    dict_for = {"frozen_j10": jd, "frozen_rand10": rd, "frozen_logit10": ld}
    ab = s7.Ablator(model.layers)
    with ab:
        for cond in conds:
            res["conditions"].setdefault(cond, {})
            for tname in task_names:
                if tname in res["conditions"][cond] and "--force" not in sys.argv:
                    continue
                t0 = time.time()
                extra = {}
                if temp:
                    if cond == "none":
                        ab.mode = None
                        scores = run_task_sampled(tname, model, hf, tok,
                                                  tasks, temp, seed)
                    else:
                        scores = []
                        key = "twohop" if tname == "twohop" else "arithmetic"
                        for it in tasks[key]:
                            ab.mode = None
                            proj = s12.frozen_projectors(
                                model, dict_for[cond],
                                s12.item_prompt(key, it))
                            ab.mode = ("static", proj)
                            scores += run_task_sampled(
                                tname, model, hf, tok, {key: [it],
                                "twohop": [it], "arithmetic": [it]},
                                temp, seed)
                            ab.mode = None
                elif tname in HARD_TASKS:
                    scorer = "onehop" if tname == "hard_onehop" else "twohop_lp"
                    if cond == "none":
                        ab.mode = None
                        key = "onehop" if scorer == "onehop" else "twohop"
                        scores = s7.run_task(scorer, cond, model, hf, tok,
                                             {key: hard_items})
                    else:
                        scores = run_frozen_items(cond, dict_for[cond], ab,
                                                  model, hf, tok, hard_items,
                                                  scorer)
                elif cond == "none":
                    ab.mode = None
                    scores = s7.run_task(tname, cond, model, hf, tok, tasks,
                                         extra)
                else:
                    scores = s12.run_frozen(cond, dict_for[cond], ab, model,
                                            hf, tok, tasks, tname, extra)
                entry = s7.boot_ci(scores)
                entry["seconds"] = round(time.time() - t0)
                entry.update(extra)
                res["conditions"][cond][tname] = entry
                atomic_write_json(res, out_path)
                log(f"{cond:>15} {tname:>14}: {entry['mean']:.3f} "
                    f"[{entry['ci_lo']:.3f},{entry['ci_hi']:.3f}] "
                    f"({entry['seconds']}s)")

    c = res["conditions"]
    summ = {}
    for cond in conds[1:]:
        row = {}
        for t in ("twohop_lp", "hard_onehop_lp"):
            if t in c.get(cond, {}) and t in c.get("none", {}):
                row[f"{t}_delta"] = round(c[cond][t]["mean"]
                                          - c["none"][t]["mean"], 3)
        summ[cond] = row
    res["summary"] = summ
    atomic_write_json(res, out_path)
    log(f"wrote {out_path}; summary {summ}")


if __name__ == "__main__":
    main()
