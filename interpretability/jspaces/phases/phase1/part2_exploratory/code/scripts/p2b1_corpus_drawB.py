# B1/G2: independent fitting-corpus draw B — 120 fresh WikiText-103 records,
# seed 1, explicitly disjoint from ALL 200 draw-A rows (fit set + the
# descriptive/prose eval spares), so lens fits stay independent of every
# evaluation text. Deterministic + sha-recorded. Output lives in the PART-2
# run dir (v1 config is frozen).
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import RUN_DIR, RUN_DIR_P2, atomic_write_json, die, log, seed_all

import numpy as np

DRAW_A = RUN_DIR / "config" / "prompts" / "fitting_corpus.jsonl"
OUT = RUN_DIR_P2 / "config" / "prompts" / "fitting_corpus_drawB.jsonl"
N_B, MIN_CHARS, SEED_B = 120, 600, 1


def main() -> None:
    seed_all()
    if OUT.exists() and "--force" not in sys.argv:
        log(f"{OUT} exists; skipping")
        return
    a_rows = [json.loads(l) for l in DRAW_A.read_text().splitlines()]
    a_idx = {r["idx"] for r in a_rows}
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    keep = [i for i, t in enumerate(ds["text"]) if len(t.strip()) >= MIN_CHARS]
    log(f"pool {len(keep)}; excluding {len(a_idx)} draw-A rows")
    rng = np.random.default_rng(SEED_B)
    order = rng.permutation(len(keep))
    rows, skipped = [], 0
    for j in order:
        gi = int(keep[int(j)])
        if gi in a_idx:
            skipped += 1
            continue
        rows.append({"idx": gi, "text": ds[gi]["text"]})
        if len(rows) == N_B:
            break
    if len(rows) != N_B:
        die(f"only {len(rows)} disjoint rows found")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(OUT)
    sha = hashlib.sha256("".join(r["text"] for r in rows).encode()).hexdigest()
    atomic_write_json(
        {"dataset": "Salesforce/wikitext:wikitext-103-raw-v1:train",
         "n": N_B, "min_chars": MIN_CHARS, "seed": SEED_B,
         "disjoint_from": "draw A (all 200 rows: fit + eval spares)",
         "collisions_skipped": skipped, "sha256": sha},
        RUN_DIR_P2 / "config" / "fit_corpus_drawB_meta.json")
    log(f"wrote {OUT} (sha {sha[:16]}…, {skipped} collisions skipped)")


if __name__ == "__main__":
    main()
