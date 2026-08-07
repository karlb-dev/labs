# Phase 1a: deterministic fitting corpus.
#
# WikiText-103 (cached on Drive), the corpus family of the neuronpedia
# reference lenses ("Salesforce-wikitext/..._n1000.pt"), so our recipe is
# comparable to the published Qwen lenses. 200 records >= 600 chars, sampled
# with seed 0: first 120 are the fit set (4 slices x 30), rest spares.
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (RUN_DIR, SOURCE_LAYERS_32B, atomic_write_json,
                        die, ensure_dirs, log, seed_all)

import json
import numpy as np

OUT = RUN_DIR / "config" / "prompts" / "fitting_corpus.jsonl"
N_TOTAL, N_FIT, MIN_CHARS = 200, 120, 600


def main() -> None:
    ensure_dirs()
    seed_all()
    if OUT.exists() and "--force" not in sys.argv:
        log(f"{OUT} exists; skipping")
        return
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    log(f"wikitext-103 train: {len(ds)} records")
    keep = [i for i, t in enumerate(ds["text"]) if len(t.strip()) >= MIN_CHARS]
    log(f"{len(keep)} records >= {MIN_CHARS} chars")
    rng = np.random.default_rng(0)
    picked = rng.choice(len(keep), size=N_TOTAL, replace=False)
    rows = [{"idx": int(keep[j]), "text": ds[int(keep[j])]["text"]} for j in picked]

    if len(rows) != N_TOTAL:
        die(f"expected {N_TOTAL} rows, got {len(rows)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(OUT)

    sha = hashlib.sha256("".join(r["text"] for r in rows).encode()).hexdigest()
    atomic_write_json(
        {
            "dataset": "Salesforce/wikitext:wikitext-103-raw-v1:train",
            "n_total": N_TOTAL, "n_fit": N_FIT, "min_chars": MIN_CHARS,
            "seed": 0, "sha256": sha,
            "fit_geometry": {
                "model": "allenai/Olmo-3-32B-Think",
                "source_layers": SOURCE_LAYERS_32B,
                "target_layer": 63, "dim_batch": 8,
                "max_seq_len": 128, "skip_first": 16,
                "slices": [[k * 30, (k + 1) * 30] for k in range(4)],
            },
        },
        RUN_DIR / "config" / "fit_config.json",
    )
    log(f"wrote {OUT} (sha256 {sha[:16]}...) and config/fit_config.json")


if __name__ == "__main__":
    main()
