# Build the Part-2 master dataset: one row per banked cell, every detail a
# figure or table could need, regenerable at any time from metrics alone.
# Output: report/matrix_master.csv + .json (list of row dicts).
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import RUN_DIR_P2, atomic_write_json, log

ROWS = []


def add(model, ws, instrument, metric, value, **kw):
    ROWS.append({"model": model, "ws": ws, "instrument": instrument,
                 "metric": metric, "value": value} | kw)


def a0_rows():
    p = RUN_DIR_P2 / "metrics" / "olmo31-instruct" / "a0_transfer_gate.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    src = str(p.relative_to(RUN_DIR_P2))
    base = dict(n=d["multihop"]["n_scored"], seed=0, source=src,
                lens="donor-think-120")
    for k in (1, 5, 20):
        add(d["model_slug"], "A0", "readout", f"multihop_jlens_pass@{k}",
            d["multihop"][f"jlens_pass@{k}"], **base)
        add(d["model_slug"], "A0", "readout", f"multihop_logit_pass@{k}",
            d["multihop"][f"logit_pass@{k}"], **base)
    add(d["model_slug"], "A0", "readout", "probe_hits_at20",
        d["probe_jlens_hits_at_20"], n=d["n_probes"], seed=0, source=src,
        lens="donor-think-120")
    drift = d.get("unembed_drift", {})
    if "head_row_cos" in drift:
        add(d["model_slug"], "A0", "drift", "unembed_row_cos_median",
            drift["head_row_cos"]["median"], source=src)
        add(d["model_slug"], "A0", "drift", "final_norm_gain_cos",
            drift.get("final_norm_gain_cos"), source=src)
    add(d["model_slug"], "A0", "gate", "verdict", d["verdict"], source=src)


def causal_grid_rows(model_slug: str, fname: str, ws: str):
    """Generic extractor for s7-format condition grids
    (conditions.<cond>.<task> = {mean, ci_lo, ci_hi, seconds, ...})."""
    p = RUN_DIR_P2 / "metrics" / model_slug / fname
    if not p.exists():
        return
    d = json.loads(p.read_text())
    src = str(p.relative_to(RUN_DIR_P2))
    for cond, tasks in d.get("conditions", {}).items():
        for task, e in tasks.items():
            if not isinstance(e, dict) or "mean" not in e:
                continue
            add(model_slug, ws, "causal", task, e["mean"], condition=cond,
                ci_lo=e.get("ci_lo"), ci_hi=e.get("ci_hi"),
                n=e.get("n"), seed=d.get("seed", 0),
                dose_k=d.get("k"), decoding="greedy", source=src)


def d_trace_rows():
    p = RUN_DIR_P2 / "metrics" / "olmo3-think" / "d_occupancy_traces.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    src = str(p.relative_to(RUN_DIR_P2))
    for key, v in d["aggregate"].items():
        seg, layer = key.rsplit("_L", 1)
        for m in ("occ1_med", "occ8_med", "lead8_med"):
            add(d["model_slug"], "D", "occupancy", f"{seg}_{m}", v[m],
                layer=int(layer), n=v["n_items"], regime="think-mode",
                source=src)


def main():
    a0_rows()
    causal_grid_rows("olmo3-think", "b3_frozen_logit.json", "B3")
    d_trace_rows()
    # future extractors register above this line as workstreams land
    outdir = RUN_DIR_P2 / "report"
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "n_rows": len(ROWS)}
    atomic_write_json({"meta": meta, "rows": ROWS}, outdir / "matrix_master.json")
    cols = sorted({k for r in ROWS for k in r})
    with open(outdir / "matrix_master.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(ROWS)
    log(f"matrix_master: {len(ROWS)} rows -> {outdir}")


if __name__ == "__main__":
    main()
