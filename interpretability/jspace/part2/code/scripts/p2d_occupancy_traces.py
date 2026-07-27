# D (phase 1, CPU) — output-occupancy index on part-1 saved think-mode
# traces (Olmo-3-32B-Think, s8: 38 items × ≤400 steps, layers {24,32,40},
# top-8 J-readout per step from the state that produced that step's token).
#
# Occupancy = how much of the live verbalizable readout is simply the output
# stream itself (H2): per step, is the imminently-emitted token in the
# readout top-k? Reported per layer: occ@1, occ@8, lead-1 variants (is the
# NEXT step's token already in this step's top-8 — readout running ahead of
# emission), split inside-vs-after </think> when the tokenizer can mark the
# boundary. Chance floor ≈ 8/100278 ≈ 8e-5, so percentages are the signal.
#
# Phase 2 (ride-along, per matrix model) adds full-vocab emitted-token rank
# + cosine-to-top-k-span during Core Battery generation; this script is the
# saved-trace anchor for the Think column.
#
# Usage: python scripts/p2d_occupancy_traces.py [--force]
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import RUN_DIR, atomic_write_json, log, p2_metrics_dir

TRACES = RUN_DIR / "metrics" / "cot_traces"
LAYERS = (24, 32, 40)
OUT = p2_metrics_dir("olmo3-think") / "d_occupancy_traces.json"


def think_close_id():
    """Token id of '</think>' via the local Think tokenizer copy; None if
    unavailable (segmentation then skipped, overall stats still computed)."""
    try:
        import transformers
        for base in (Path("/content/models/olmo3-think"),
                     *sorted(Path("/content/drive/MyDrive/hf_cache/hub/"
                                  "models--allenai--Olmo-3-32B-Think/snapshots").glob("*"))):
            if (base / "tokenizer_config.json").exists():
                tok = transformers.AutoTokenizer.from_pretrained(str(base))
                ids = tok("</think>", add_special_tokens=False).input_ids
                return ids[0] if len(ids) == 1 else None
    except Exception as e:
        log(f"tokenizer unavailable ({e!r}); no think-segmentation")
    return None


def item_stats(steps, close_id):
    n = len(steps)
    cut = next((i for i, s in enumerate(steps) if s["tok"] == close_id), n) \
        if close_id is not None else None
    segs = {"all": range(n)}
    if cut is not None and 0 < cut < n - 1:
        segs["think"] = range(cut)
        segs["post"] = range(cut + 1, n)
    out = {"n_steps": n, "think_cut": cut}
    for seg, idx in segs.items():
        idx = list(idx)
        if not idx:
            continue
        for l in LAYERS:
            top = [steps[i].get(f"L{l}_top") for i in idx]
            tok = [steps[i]["tok"] for i in idx]
            nxt = [steps[i + 1]["tok"] if i + 1 < n else None for i in idx]
            occ1 = sum(t and x == t[0] for x, t in zip(tok, top)) / len(idx)
            occ8 = sum(t and x in t for x, t in zip(tok, top)) / len(idx)
            lead8 = sum(t and y is not None and y in t
                        for y, t in zip(nxt, top)) / len(idx)
            out[f"{seg}_L{l}"] = {"occ1": round(occ1, 4), "occ8": round(occ8, 4),
                                  "lead8": round(lead8, 4)}
    return out


def main() -> None:
    if OUT.exists() and "--force" not in sys.argv:
        log(f"{OUT} exists; skipping")
        return
    close_id = think_close_id()
    log(f"</think> id: {close_id}")
    rows = []
    for f in sorted(TRACES.glob("item_*.json.gz")):
        o = json.loads(gzip.open(f, "rt").read())
        rows.append({"item": f.stem.replace(".json", "")}
                    | item_stats(o["steps"], close_id))
    agg = {}
    for seg in ("all", "think", "post"):
        for l in LAYERS:
            key = f"{seg}_L{l}"
            vals = [r[key] for r in rows if key in r]
            if not vals:
                continue
            med = lambda k: sorted(v[k] for v in vals)[len(vals) // 2]  # noqa: E731
            agg[key] = {"n_items": len(vals), "occ1_med": med("occ1"),
                        "occ8_med": med("occ8"), "lead8_med": med("lead8")}
    out = {"model_slug": "olmo3-think", "source": "v1 s8 think-mode traces",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "layers": list(LAYERS), "chance_floor_occ8": 8 / 100278,
           "aggregate": agg, "items": rows}
    atomic_write_json(out, OUT)
    log(f"wrote {OUT}")
    for k, v in agg.items():
        log(f"  {k}: occ@1 {v['occ1_med']:.3f} occ@8 {v['occ8_med']:.3f} "
            f"lead8 {v['lead8_med']:.3f} (n={v['n_items']})")


if __name__ == "__main__":
    main()
