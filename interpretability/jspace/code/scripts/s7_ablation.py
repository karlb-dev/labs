# Phase 3: causal dissociation — ablate the J-space, watch what breaks.
#
# Paper headline: zeroing the top-10 most-active J-lens directions across
# the workspace band collapses multi-hop reasoning while fluency and
# shallow tasks survive; matched controls do neither.
#
# Conditions
#   none                       baseline
#   jspace_dyn10               paper-faithful: per (token, band layer),
#                              top-10 most-active J-dirs, project out
#   jspace_k{10,20,40}         static corpus-level top-k J-span (dose)
#   random_k{10,20,40}         matched-dim random orthonormal subspace
#   nonJ_pca_k{10,20,40}       matched-dim top residual PCs orthogonal to
#                              the J-span (high-variance, non-verbalizable)
# Band: fitted mid layers 20..44 step 2 (33-70% depth).
#
# Battery
#   multi-step: probe-swap 2-hop factual (greedy next token == answer),
#               chained arithmetic (parse generated int), 3-table SQL
#               (join-key correctness checks)
#   fluency:    held-out prose NLL, single-hop factual, grammatical
#               minimal pairs (logprob comparison)
# Bootstrap 95% CIs (2000 resamples) per (condition, task). Resumable per
# (condition, task) block; metrics/ablation_results.json grows as it goes.
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (RUN_DIR, atomic_write_json, die, ensure_dirs,
                        load_model, log, read_json, seed_all,
                        variant_first_ids)

import numpy as np
import torch
from jlens import JacobianLens

OUT = RUN_DIR / "metrics" / "ablation_results.json"
STATE_DIR = RUN_DIR / "metrics" / "layer_state"
BAND = list(range(20, 45, 2))
DOSES = (10, 20, 40)
BOOT = 2000

PROBE_SWAP = Path("/content/jacobian-lens/data/experiments/probe-swap.json")


# ------------------------------------------------------------- task sets
def build_tasks(rng) -> dict:
    tasks = {}
    ps = json.loads(PROBE_SWAP.read_text())["items"][:60]
    tasks["twohop"] = [{"prompt": it["prompt"], "answer": it["answer"]}
                       for it in ps]
    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
           ("*", lambda a, b: a * b)]
    arith = []
    for _ in range(30):
        a, b, c, d = [int(x) for x in rng.integers(3, 40, size=4)]
        (o1, f1) = ops[rng.integers(3)]
        (o2, f2) = ops[rng.integers(2)]
        val = f2(f1(a, b), c) - d
        arith.append({"prompt": f"Q: What is (({a} {o1} {b}) {o2} {c}) - {d}?"
                                f" Work step by step, then give the answer."
                                f"\nA: ({a} {o1} {b}) =",
                      "answer": str(val)})
    tasks["arithmetic"] = arith
    schemas = [
        ("users(id, name, city_id)", "cities(id, city_name, country)",
         "orders(id, user_id, total)",
         "Total order value per country.",
         [r"o\.user_id\s*=\s*u\.id|u\.id\s*=\s*o\.user_id",
          r"u\.city_id\s*=\s*c\.id|c\.id\s*=\s*u\.city_id"]),
        ("products(pid, pname, cat_id)", "categories(cat_id, cat_name)",
         "sales(sid, pid, qty, price)",
         "Revenue by category name.",
         [r"s\.pid\s*=\s*p\.pid|p\.pid\s*=\s*s\.pid",
          r"p\.cat_id\s*=\s*(c|cat)\.cat_id|(c|cat)\.cat_id\s*=\s*p\.cat_id"]),
        ("employees(eid, ename, dept_id)", "departments(dept_id, dname)",
         "salaries(eid, amount, year)",
         "Total salary per department name.",
         [r"(s|sal)\.eid\s*=\s*e\.eid|e\.eid\s*=\s*(s|sal)\.eid",
          r"e\.dept_id\s*=\s*d\.dept_id|d\.dept_id\s*=\s*e\.dept_id"]),
    ]
    sql = []
    for i in range(30):
        t1, t2, t3, q, checks = schemas[i % 3]
        sql.append({"prompt": f"-- Tables:\n-- {t1}\n-- {t2}\n-- {t3}\n"
                              f"-- Task: {q}\nSELECT",
                    "checks": checks})
    tasks["sql"] = sql

    onehop_facts = [
        ("The capital of France is", " Paris"),
        ("The capital of Japan is", " Tokyo"),
        ("The capital of Italy is", " Rome"),
        ("The capital of Germany is", " Berlin"),
        ("The capital of Spain is", " Madrid"),
        ("The capital of Russia is", " Moscow"),
        ("The capital of England is", " London"),
        ("The capital of Egypt is", " Cairo"),
        ("The capital of Canada is", " Ottawa"),
        ("The capital of China is", " Beijing"),
        ("The largest planet in our solar system is", " Jupiter"),
        ("The chemical symbol for gold is", " Au"),
        ("The author of Romeo and Juliet is William", " Shakespeare"),
        ("The language spoken in Brazil is", " Portuguese"),
        ("Water freezes at zero degrees", " Celsius"),
        ("The opposite of hot is", " cold"),
        ("A spider has eight", " legs"),
        ("The color of the sky on a clear day is", " blue"),
        ("The first month of the year is", " January"),
        ("The number of days in a week is", " seven"),
        ("The star at the center of our solar system is the", " Sun"),
        ("The ocean between America and Europe is the", " Atlantic"),
        ("The currency of the United States is the", " dollar"),
        ("The largest mammal on Earth is the blue", " whale"),
        ("The frozen form of water is called", " ice"),
        ("The organ that pumps blood is the", " heart"),
        ("The planet known as the Red Planet is", " Mars"),
        ("The fastest land animal is the", " cheetah"),
        ("A baby dog is called a", " puppy"),
        ("The season after winter is", " spring"),
    ]
    tasks["onehop"] = [{"prompt": p, "answer": a} for p, a in onehop_facts]

    corpus = [json.loads(l) for l in
              (RUN_DIR / "config" / "prompts" / "fitting_corpus.jsonl")
              .read_text().splitlines()]
    tasks["prose_nll"] = [{"text": r["text"]} for r in corpus[170:190]]

    gram = []
    subs = [("The dogs", "run", "runs"), ("The cat", "sleeps", "sleep"),
            ("My friends", "were", "was"), ("The teacher", "explains", "explain"),
            ("Those birds", "fly", "flies"), ("The child", "plays", "play"),
            ("Both engines", "work", "works"), ("Her brother", "cooks", "cook"),
            ("The students", "study", "studies"), ("This machine", "beeps", "beep")]
    for subj, good, bad in subs:
        for tail in (" every day.", " at night."):
            gram.append({"good": f"{subj} {good}{tail}",
                         "bad": f"{subj} {bad}{tail}"})
    tasks["grammar"] = gram
    return tasks


# ------------------------------------------------------ direction builders
def build_subspaces(lens, hf) -> dict:
    """Per band layer: fp32 orthonormal bases Q [d, k] per condition."""
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    rng = np.random.default_rng(0)
    subs = {f"jspace_k{k}": {} for k in DOSES}
    subs |= {f"random_k{k}": {} for k in DOSES}
    subs |= {f"nonJ_pca_k{k}": {} for k in DOSES}
    dicts = {}
    for l in BAND:
        state = torch.load(STATE_DIR / f"layer_{l}.pt", weights_only=True)
        J = lens.jacobians[l].to(W_U.device)          # lens loads on CPU
        ids = state["top_dir_ids"]
        D_top = torch.nn.functional.normalize(
            (W_U[ids[:max(DOSES)]] * g[None, :]) @ J, dim=1)
        span_ids = ids[:256]
        S = torch.nn.functional.normalize((W_U[span_ids] * g[None, :]) @ J, dim=1)
        Qs, _ = torch.linalg.qr(S.T)
        P = state["pca_evecs"].T.to(W_U.device)
        P_perp = P - (P @ Qs) @ Qs.T
        keep = P_perp.norm(dim=1) > 0.5
        P_ok = torch.nn.functional.normalize(P_perp[keep], dim=1)
        for k in DOSES:
            Qj, _ = torch.linalg.qr(D_top[:k].T)
            subs[f"jspace_k{k}"][l] = Qj.contiguous()
            R = torch.tensor(rng.standard_normal((k, D_top.shape[1])),
                             dtype=torch.float32, device=W_U.device)
            Qr, _ = torch.linalg.qr(R.T)
            subs[f"random_k{k}"][l] = Qr.contiguous()
            Qp, _ = torch.linalg.qr(P_ok[:k].T)
            subs[f"nonJ_pca_k{k}"][l] = Qp.contiguous()
        # full dictionary for the dynamic condition (fp16, GPU)
        dicts[l] = torch.nn.functional.normalize(
            (W_U * g[None, :]) @ J, dim=1).half()
    return subs, dicts


class Ablator:
    """Forward hooks on band-layer blocks projecting out a subspace."""

    def __init__(self, model_layers):
        self._layers = model_layers
        self._handles = []
        self.mode = None          # None | ("static", {l: Q}) | ("dyn", dicts, k)

    def _hook(self, layer_idx):
        def fn(mod, inp, out):
            if self.mode is None:
                return out
            h = out[0] if not torch.is_tensor(out) else out
            kind = self.mode[0]
            if kind == "static":
                Q = self.mode[1][layer_idx]               # [d, k] fp32
                hf32 = h.float()
                hf32 = hf32 - (hf32 @ Q) @ Q.T
                h_new = hf32.to(h.dtype)
            else:
                D, k = self.mode[1][layer_idx], self.mode[2]
                hf32 = h.float()
                B, T, d = hf32.shape
                flat = hf32.reshape(-1, d)
                corr = flat.half() @ D.T                  # [BT, V]
                top = corr.topk(k, dim=1).indices         # [BT, k]
                dirs = D[top].float()                     # [BT, k, d]
                for _ in range(2):                        # 2-pass deflation
                    c = torch.einsum("bd,bkd->bk", flat, dirs)
                    flat = flat - torch.einsum("bk,bkd->bd", c, dirs)
                h_new = flat.reshape(B, T, d).to(h.dtype)
            if torch.is_tensor(out):
                return h_new
            return (h_new, *out[1:])
        return fn

    def __enter__(self):
        for l in BAND:
            self._handles.append(
                self._layers[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


# ------------------------------------------------------------- scoring
def boot_ci(scores: list[float]) -> dict:
    arr = np.array(scores, dtype=float)
    rng = np.random.default_rng(1)
    means = rng.choice(arr, size=(BOOT, len(arr)), replace=True).mean(axis=1)
    return {"mean": float(arr.mean()),
            "ci_lo": float(np.quantile(means, 0.025)),
            "ci_hi": float(np.quantile(means, 0.975)), "n": len(arr)}


@torch.no_grad()
def greedy_next(model, tok, prompt: str) -> int:
    ids = model.encode(prompt, max_length=512)
    logits = model.unembed(_last_hidden(model, ids))
    return int(logits[0, -1].argmax())


def _last_hidden(model, ids):
    from jlens.hooks import ActivationRecorder
    with ActivationRecorder(model.layers, at=[model.n_layers - 1]) as rec:
        model.forward(ids)
    return rec.activations[model.n_layers - 1]


@torch.no_grad()
def generate(hf, tok, prompt: str, max_new: int) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    out = hf.generate(ids, max_new_tokens=max_new, do_sample=False,
                      pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def seq_nll(model, tok, text: str) -> float:
    ids = model.encode(text, max_length=256)
    logits = model.unembed(_last_hidden(model, ids)).float()
    lp = torch.log_softmax(logits[0, :-1], dim=-1)
    tgt = ids[0, 1:]
    return float(-lp[torch.arange(len(tgt)), tgt].mean())


def run_task(name, cond, model, hf, tok, tasks, extra=None) -> list[float]:
    scores = []
    if name in ("twohop", "onehop"):
        for it in tasks[name]:
            nid = greedy_next(model, tok, it["prompt"])
            scores.append(float(nid in variant_first_ids(tok, it["answer"])))
    elif name == "twohop_lp":
        # finer instrument than greedy hit: mean logprob of the answer's
        # best variant first-token at the final position (teacher-free)
        for it in tasks["twohop"]:
            ids = model.encode(it["prompt"], max_length=512)
            logits = model.unembed(_last_hidden(model, ids)).float()
            lp = torch.log_softmax(logits[0, -1], dim=-1)
            best = max(float(lp[a]) for a in
                       variant_first_ids(tok, it["answer"]))
            scores.append(best)
    elif name == "samples":
        # audit trail: verbatim generations for scorer-sanity inspection
        texts = []
        for it in tasks["arithmetic"][:3]:
            texts.append({"prompt": it["prompt"], "answer": it["answer"],
                          "gen": generate(hf, tok, it["prompt"], 48)})
        for it in tasks["sql"][:2]:
            texts.append({"prompt": it["prompt"][-120:],
                          "gen": "SELECT" + generate(hf, tok, it["prompt"], 64)})
        if extra is not None:
            extra["texts"] = texts
        scores = [1.0]
    elif name == "arithmetic":
        for it in tasks[name]:
            text = generate(hf, tok, it["prompt"], 48)
            nums = re.findall(r"-?\d+", text.split("\n")[0] + " " +
                              (text.split("\n")[1] if "\n" in text else ""))
            scores.append(float(it["answer"] in nums))
    elif name == "arithmetic_v2":
        # v1's two-line window was format-biased (multi-line solutions
        # scored 0); score the whole segment before the next question.
        for it in tasks["arithmetic"]:
            text = generate(hf, tok, it["prompt"], 64)
            seg = text.split("\nQ:")[0]
            scores.append(float(it["answer"] in re.findall(r"-?\d+", seg)))
    elif name == "sql":
        for it in tasks[name]:
            text = "SELECT" + generate(hf, tok, it["prompt"], 64)
            ok = all(re.search(c, text) for c in it["checks"])
            scores.append(float(ok))
    elif name == "prose_nll":
        for it in tasks[name]:
            scores.append(seq_nll(model, tok, it["text"]))
    elif name == "grammar":
        for it in tasks[name]:
            scores.append(float(seq_nll(model, tok, it["good"])
                                < seq_nll(model, tok, it["bad"])))
    return scores


def main() -> None:
    ensure_dirs()
    seed_all()
    rng = np.random.default_rng(0)
    lens = JacobianLens.load(str(RUN_DIR / "lens" / "olmo32bthink_lens.pt"))
    if not (STATE_DIR / f"layer_{BAND[0]}.pt").exists():
        die("s5 layer state missing; run s5 first")
    tasks = build_tasks(rng)
    res = read_json(OUT) if OUT.exists() else {"band": BAND, "doses": DOSES,
                                               "conditions": {}}
    model, hf, tok = load_model("main")
    subs, dicts = build_subspaces(lens, hf)
    log(f"built subspaces for {len(BAND)} band layers")

    conds = (["none", "jspace_dyn10"]
             + [f"{g}_k{k}" for k in DOSES
                for g in ("jspace", "random", "nonJ_pca")])
    task_names = ["twohop", "onehop", "arithmetic", "sql", "prose_nll",
                  "grammar", "twohop_lp", "samples", "arithmetic_v2"]
    ab = Ablator(model.layers)
    with ab:
        for cond in conds:
            res["conditions"].setdefault(cond, {})
            for tname in task_names:
                if tname in res["conditions"][cond] and "--force" not in sys.argv:
                    continue
                t0 = time.time()
                if cond == "none":
                    ab.mode = None
                elif cond == "jspace_dyn10":
                    ab.mode = ("dyn", dicts, 10)
                else:
                    ab.mode = ("static", subs[cond])
                extra = {}
                scores = run_task(tname, cond, model, hf, tok, tasks, extra)
                ab.mode = None
                entry = boot_ci(scores)
                entry["seconds"] = round(time.time() - t0)
                entry.update(extra)
                res["conditions"][cond][tname] = entry
                atomic_write_json(res, OUT)
                log(f"{cond:>16} {tname:>10}: mean {entry['mean']:.3f} "
                    f"[{entry['ci_lo']:.3f},{entry['ci_hi']:.3f}] "
                    f"({entry['seconds']}s)")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
