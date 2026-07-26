# v2 Priority 4 (CPU): false-positive floor for the v1 cot-lead headline.
#
# v1 claim: "workspace holds the answer a median 46 steps before the CoT
# states it (91% of traced items)". The detector (any variant first-token in
# any of 3 read layers' top-8, across hundreds of steps) gives every word
# many chances to fire. Here the IDENTICAL detector runs on matched foils:
#   family : same-item-family wrong answers (other bridge entities / other
#            join-key columns / nearby numbers), up to 5 per item
#   freq   : 5 corpus-frequency-decile-matched content words per item
#            (excluded: words occurring in the item's question/answer/
#            intermediate)
# Separation stats reported per foil class: detection rate (ever fires),
# median first-detection step, lead where text mentions exist, and the
# within-item ROC-style stat: how often is the ANSWER detected earlier
# than every foil of that item, and the answer's detection rank among its
# foils. The v1 claim survives only if answer detection separates cleanly.
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (RUN_DIR, RUN_DIR_V2, atomic_write_json, log,
                        variant_first_ids)

import numpy as np

COT = RUN_DIR / "metrics" / "cot_results.json"
TRACE_DIR = RUN_DIR / "metrics" / "cot_traces"
CORPUS = RUN_DIR / "config" / "prompts" / "fitting_corpus.jsonl"
OUT = RUN_DIR_V2 / "metrics" / "cot_foils.json"
READ_LAYERS = (24, 32, 40)
SQL_KEYS = ["user_id", "pid", "eid", "book_id"]


def freq_table():
    cnt = Counter()
    for line in CORPUS.read_text().splitlines():
        rec = json.loads(line)
        text = rec.get("text") or rec.get("prompt") or ""
        cnt.update(w.lower() for w in re.findall(r"[A-Za-z]{3,}", text))
    words = list(cnt)
    freqs = np.array([cnt[w] for w in words], dtype=float)
    order = np.argsort(freqs)
    deciles = np.empty(len(words), dtype=int)
    deciles[order] = np.arange(len(words)) * 10 // len(words)
    return words, dict(zip(words, deciles)), cnt


def item_foils(it, all_items, words, decile_of, rng):
    kind, ans = it["kind"], it["answer"].strip()
    banned = set(re.findall(r"[A-Za-z0-9_]+", (it["question"] + " " + ans
                 + " " + it["intermediate"]).lower()))
    fam = []
    if kind == "arithmetic":
        v = int(ans)
        fam = [str(v + d) for d in (-3, -2, -1, 1, 2, 3)]
    elif kind == "sql":
        fam = [c for c in SQL_KEYS if c != ans] + ["id", "name"]
    else:
        pool = [o["answer"].strip() for o in all_items
                if o["kind"] == kind and o["answer"].strip().lower()
                not in banned and o["answer"].strip() != ans]
        pool = sorted(set(pool))
        fam = list(rng.choice(pool, size=min(5, len(pool)), replace=False))
    fam = [f for f in fam if f.lower() not in banned][:5]

    tgt_dec = decile_of.get(ans.lower(), 0)
    cand = [w for w in words
            if abs(decile_of[w] - tgt_dec) <= 1 and w not in banned
            and len(w) >= 3]
    freq = list(rng.choice(cand, size=min(5, len(cand)), replace=False))
    return fam, freq


def detect(tok, steps, cum_texts, word):
    """Same detector as v1: (first step any variant first-token in any read
    layer's top-8, first step word appears in cumulative decoded text)."""
    try:
        vids = set(variant_first_ids(tok, word))
    except SystemExit:
        return None, None
    ws = next((i for i, s in enumerate(steps)
               if any(v in s[f"L{l}_top"] for v in vids
                      for l in READ_LAYERS)), None)
    w = word.strip().lower().replace(" ", "")
    tx = next((i for i, c in enumerate(cum_texts) if w in c), None)
    return ws, tx


def main() -> None:
    import transformers
    (RUN_DIR_V2 / "metrics").mkdir(parents=True, exist_ok=True)
    res_v1 = json.loads(COT.read_text())
    items = res_v1["items"]
    tok = transformers.AutoTokenizer.from_pretrained(
        "allenai/Olmo-3-32B-Think")
    words, decile_of, _ = freq_table()
    rng = np.random.default_rng(2)

    per_item, agg = [], {"answer": [], "family": [], "freq": []}
    for f in sorted(TRACE_DIR.glob("item_*.json.gz")):
        iid = f.stem.split("_")[1].split(".")[0]
        it = items.get(iid)
        if it is None:
            continue
        steps = json.loads(gzip.open(f, "rt").read())["steps"]
        cum, cum_texts = "", []
        for s in steps:
            cum += tok.decode([s["tok"]], skip_special_tokens=False)
            cum_texts.append(cum.lower().replace(" ", ""))
        fam, frq = item_foils(it, list(items.values()), words, decile_of, rng)
        row = {"iid": int(iid), "kind": it["kind"], "answer": it["answer"],
               "n_steps": len(steps), "words": {}}
        for cls, wl in (("answer", [it["answer"]]), ("family", fam),
                        ("freq", frq)):
            for w in wl:
                ws, tx = detect(tok, steps, cum_texts, w)
                row["words"][f"{cls}:{w}"] = {"ws": ws, "text": tx}
                agg[cls].append({"iid": int(iid), "word": w, "ws": ws,
                                 "text": tx, "n_steps": len(steps)})
        ans_ws = row["words"][f"answer:{it['answer']}"]["ws"]
        foil_ws = [v["ws"] for k, v in row["words"].items()
                   if not k.startswith("answer:")]
        det_foil_ws = [w for w in foil_ws if w is not None]
        row["answer_earlier_than_all_foils"] = (
            ans_ws is not None
            and all(ans_ws < w for w in det_foil_ws))
        row["answer_det_rank"] = (
            None if ans_ws is None
            else 1 + sum(w < ans_ws for w in det_foil_ws))
        per_item.append(row)

    def stats(cls):
        rows = agg[cls]
        det = [r for r in rows if r["ws"] is not None]
        both = [r for r in rows if r["ws"] is not None
                and r["text"] is not None]
        return {
            "n_words": len(rows),
            "det_rate": len(det) / max(len(rows), 1),
            "med_first_det_step": (float(np.median([r["ws"] for r in det]))
                                   if det else None),
            "text_mention_rate": np.mean([r["text"] is not None
                                          for r in rows]),
            "n_with_both": len(both),
            "med_lead": (float(np.median([r["text"] - r["ws"]
                                          for r in both])) if both else None),
            "frac_ws_leads": (float(np.mean([r["text"] > r["ws"]
                                             for r in both]))
                              if both else None),
        }

    out = {
        "read_layers": list(READ_LAYERS), "n_items": len(per_item),
        "detector": "identical to v1 s8/s8b (variant first-token in any "
                    "read layer top-8; text = space-stripped substring)",
        "classes": {c: stats(c) for c in ("answer", "family", "freq")},
        "answer_earlier_than_all_foils_frac": float(np.mean(
            [r["answer_earlier_than_all_foils"] for r in per_item])),
        "answer_det_rank_median": float(np.median(
            [r["answer_det_rank"] for r in per_item
             if r["answer_det_rank"] is not None])),
        "per_item": per_item,
    }
    atomic_write_json(out, OUT)
    for c in ("answer", "family", "freq"):
        s = out["classes"][c]
        log(f"{c:>7}: det {s['det_rate']:.2f} med_first "
            f"{s['med_first_det_step']} lead {s['med_lead']} "
            f"(n={s['n_words']})")
    log(f"answer earlier than ALL its foils: "
        f"{out['answer_earlier_than_all_foils_frac']:.2f}; "
        f"median answer det rank {out['answer_det_rank_median']}")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
