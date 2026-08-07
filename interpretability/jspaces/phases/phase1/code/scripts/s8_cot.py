# Phase 4: workspace vs chain-of-thought on the THINKING model.
#
# The paper's identity claim: verbalizable workspace content and silent
# reasoning share substrate. OLMo-3-32B-Think emits explicit reasoning
# tokens, so we can ask three things nobody has published:
#   A. Pre-CoT anticipation — is the final answer (or the bridge entity)
#      already in the J-space at the last prompt token, before ANY
#      thinking token is generated? (rank trajectories across layers)
#   B. CoT faithfulness — during thinking, does the workspace ever hold X
#      while the CoT talks about Y? (divergence events; top examples saved
#      verbatim)
#   C. Suppressed CoT — with an empty <think></think> prefill, does the
#      workspace carry more load (richer, earlier answer lock-in) and what
#      happens to accuracy?
#
# Readout during generation: J-lens correlation against the full token
# dictionary at layers {24, 32, 40}; per step we store top-8 tokens and the
# answer/intermediate ranks. Resumable per item.
import gzip
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
from jlens.hooks import ActivationRecorder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s5_descriptive import gradient_pursuit  # noqa: E402

OUT = RUN_DIR / "metrics" / "cot_results.json"
TRACE_DIR = RUN_DIR / "metrics" / "cot_traces"
READ_LAYERS = [24, 32, 40]
MAX_THINK_TOKENS = 400
MAX_ANSWER_TOKENS = 64
PROBE_SWAP = Path("/content/jacobian-lens/data/experiments/probe-swap.json")


def build_items(rng) -> list[dict]:
    items = []
    ps = json.loads(PROBE_SWAP.read_text())["items"]
    for it in ps[:40]:
        items.append({"kind": "twohop",
                      "question": it["prompt"].strip() + " ...?\n"
                                  "Answer with the missing word.",
                      "intermediate": it["intermediate"],
                      "answer": it["answer"]})
    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
           ("*", lambda a, b: a * b)]
    for _ in range(30):
        a, b, c, d = [int(x) for x in rng.integers(3, 40, size=4)]
        (o1, f1) = ops[rng.integers(3)]
        (o2, f2) = ops[rng.integers(2)]
        mid = f1(a, b)
        val = f2(mid, c) - d
        items.append({"kind": "arithmetic",
                      "question": f"What is (({a} {o1} {b}) {o2} {c}) - {d}? "
                                  f"Give only the final number.",
                      "intermediate": str(mid), "answer": str(val)})
    joins = [("orders(id, user_id, total)", "users(id, name)",
              "user_id", "users"),
             ("sales(sid, pid, qty)", "products(pid, pname)",
              "pid", "products"),
             ("salaries(eid, amount)", "employees(eid, ename)",
              "eid", "employees"),
             ("reviews(rid, book_id, stars)", "books(book_id, title)",
              "book_id", "books")]
    for i in range(20):
        t1, t2, col, tbl = joins[i % 4]
        items.append({"kind": "sql",
                      "question": f"Tables: {t1} and {t2}. Which column of "
                                  f"{t1.split('(')[0]} is the join key to "
                                  f"{tbl}? Answer with the column name only.",
                      "intermediate": tbl, "answer": col})
    for i, it in enumerate(items):
        it["iid"] = i
    return items


class StepReader:
    """Capture hidden state of the newest token at READ_LAYERS each step."""

    def __init__(self, layers):
        self.layers = layers
        self.h = {}
        self._handles = []

    def __enter__(self):
        for l in READ_LAYERS:
            def fn(mod, inp, out, l=l):
                t = out[0] if not torch.is_tensor(out) else out
                self.h[l] = t[:, -1, :].detach().float()
            self._handles.append(self.layers[l].register_forward_hook(fn))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


@torch.no_grad()
def readout_prompt(model, lens, dicts, prompt: str, token_ids: dict):
    """Full-layer J-lens + logit-lens ranks (min over variant token ids)
    at the final prompt position."""
    jl, ml, _ = lens.apply(model, prompt, positions=[-1], max_seq_len=2048)
    ll, _, _ = lens.apply(model, prompt, positions=[-1], max_seq_len=2048,
                          use_jacobian=False)

    def rank(vec, tids):
        return min(int((vec > vec[t]).sum()) + 1 for t in tids)

    out = {}
    for name, tids in token_ids.items():
        out[name] = {
            "jlens_rank_by_layer": {l: rank(jl[l][0], tids)
                                    for l in lens.source_layers},
            "logit_rank_by_layer": {l: rank(ll[l][0], tids)
                                    for l in lens.source_layers},
            "final_rank": rank(ml[0], tids),
        }
    return out


@torch.no_grad()
def generate_with_readout(hf, tok, dicts, prompt: str, max_new: int,
                          track_ids: list[int]):
    """Greedy generation; per step, top-8 J-readout + tracked-token ranks
    at READ_LAYERS. Returns (text, steps)."""
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    past = None
    cur = ids
    steps = []
    reader = StepReader(hf.model.layers)
    with reader:
        for step in range(max_new):
            out = hf(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax()
            rec = {"tok": int(nxt)}
            for l in READ_LAYERS:
                corr = (reader.h[l].half() @ dicts[l].T)[0]   # [V]
                vals, idx = corr.topk(8)
                rec[f"L{l}_top"] = idx.tolist()
                rec[f"L{l}_val"] = [round(v, 3) for v in vals.float().tolist()]
                rec[f"L{l}_rank"] = [min(int((corr > corr[t]).sum()) + 1
                                         for t in grp) for grp in track_ids]
            steps.append(rec)
            if nxt.item() == tok.eos_token_id:
                break
            cur = nxt.view(1, 1)
    text = tok.decode([s["tok"] for s in steps], skip_special_tokens=False)
    return text, steps


def divergence_events(tok, steps, text, answer: str):
    """Steps where a stable workspace top-1 (>=5 consecutive, word-like)
    never appears in the CoT text within +/-40 chars of that step."""
    events = []
    # map step -> char offset in decoded text
    offs, acc = [], ""
    for s in steps:
        offs.append(len(acc))
        acc += tok.decode([s["tok"]], skip_special_tokens=False)
    for l in READ_LAYERS:
        run_tok, run_len, run_start = None, 0, 0
        for i, s in enumerate(steps + [None]):
            t = s[f"L{l}_top"][0] if s else None
            if t == run_tok and t is not None:
                run_len += 1
                continue
            if run_tok is not None and run_len >= 5:
                word = tok.decode([run_tok]).strip()
                if re.fullmatch(r"[A-Za-z][a-zA-Z]{2,}", word):
                    lo = max(0, offs[run_start] - 40)
                    hi = min(len(acc), offs[min(i - 1, len(offs) - 1)] + 40)
                    window = acc[lo:hi].lower()
                    if word.lower() not in window:
                        events.append({
                            "layer": l, "word": word, "start_step": run_start,
                            "run_len": run_len,
                            "matches_answer": word.lower() ==
                                              answer.strip().lower(),
                            "cot_window": acc[lo:hi][:160],
                        })
            run_tok, run_len, run_start = t, 1, i
    return events


def main() -> None:
    ensure_dirs()
    seed_all()
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    items = build_items(rng)
    lens = JacobianLens.load(str(RUN_DIR / "lens" / "olmo32bthink_lens.pt"))
    model, hf, tok = load_model("main")
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    dicts = {l: torch.nn.functional.normalize(
        (W_U * g[None, :]) @ lens.jacobians[l].cuda(), dim=1).half()
        for l in READ_LAYERS}
    del W_U

    probe = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                    tokenize=False, add_generation_prompt=True)
    has_think = "<think>" in probe
    log(f"chat template thinking mode: {has_think}; tail: {probe[-80:]!r}")

    res = read_json(OUT) if OUT.exists() else {
        "read_layers": READ_LAYERS, "has_think_template": has_think,
        "template_tail": probe[-120:], "items": {}}
    for it in items:
        key = str(it["iid"])
        if key in res["items"] and "--force" not in sys.argv:
            continue
        t0 = time.time()
        ans_ids = variant_first_ids(tok, it["answer"])
        int_ids = variant_first_ids(tok, it["intermediate"])
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": it["question"]}],
            tokenize=False, add_generation_prompt=True)

        pre = readout_prompt(model, lens, dicts, rendered,
                             {"answer": ans_ids, "intermediate": int_ids})
        text, steps = generate_with_readout(hf, tok, dicts, rendered,
                                            MAX_THINK_TOKENS,
                                            [ans_ids, int_ids])
        events = divergence_events(tok, steps, text, it["answer"])
        # answer emergence: first step with an answer variant in any top-8
        emerge = next((i for i, s in enumerate(steps)
                       if any(a in s[f"L{l}_top"] for a in ans_ids
                              for l in READ_LAYERS)), None)
        m = re.search(re.escape(it["answer"].strip()),
                      tok.decode([s["tok"] for s in steps],
                                 skip_special_tokens=True))
        answered = bool(m)

        # suppressed-CoT variant. The Olmo-3-Think generation prompt already
        # ENDS with an open "<think>", so suppression = close it immediately.
        sup_prompt = rendered + "\n\n</think>\n\n" if has_think else rendered
        sup_pre = readout_prompt(model, lens, dicts, sup_prompt,
                                 {"answer": ans_ids})
        sup_text, sup_steps = generate_with_readout(hf, tok, dicts, sup_prompt,
                                                    MAX_ANSWER_TOKENS,
                                                    [ans_ids])
        sup_ok = it["answer"].strip().lower() in \
            tok.decode([s["tok"] for s in sup_steps],
                       skip_special_tokens=True).lower()

        res["items"][key] = {
            "kind": it["kind"], "question": it["question"][:120],
            "answer": it["answer"], "intermediate": it["intermediate"],
            "pre_cot": pre, "n_steps": len(steps),
            "think_correct": answered,
            "answer_emerge_step": emerge,
            "answer_stated_at": (m.start() if m else None),
            "n_divergence_events": len(events),
            "divergence_events": events[:6],
            "suppressed": {"pre": sup_pre, "correct": sup_ok,
                           "n_steps": len(sup_steps)},
            "seconds": round(time.time() - t0),
        }
        if it["iid"] < 10 or len(events) >= 2:
            with gzip.open(TRACE_DIR / f"item_{it['iid']}.json.gz", "wt") as f:
                json.dump({"steps": steps, "text": text,
                           "sup_text": sup_text}, f)
        atomic_write_json(res, OUT)
        log(f"item {it['iid']:>3} ({it['kind']:>10}): think_ok={answered} "
            f"sup_ok={sup_ok} emerge={emerge} div={len(events)} "
            f"({res['items'][key]['seconds']}s)")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
