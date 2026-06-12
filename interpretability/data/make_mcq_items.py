"""Generate the frozen MCQ set for Lab 10 (data/mcq_items.csv).

Run ONCE at authoring time (network required); labs never download data at
runtime. Items are sampled from the MMLU test split (cais/mmlu on the Hub),
filtered to short, clean 4-option questions across a fixed subject mix, with
a fixed seed. The answer keys are MMLU's; the lab's baseline condition
measures the model's own accuracy against them, and the small known rate of
MMLU key errors is one more reason every claim is scoped to "on this
dataset".

Columns: id, domain, question, option_a..option_d, answer_key (A-D).
"""

from __future__ import annotations

import csv
import pathlib
import random

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = pathlib.Path(__file__).parent / "mcq_items.csv"
SEED = 0
PER_SUBJECT = 20
SUBJECTS = [
    "elementary_mathematics",
    "high_school_geography",
    "high_school_biology",
    "miscellaneous",
    "nutrition",
    "us_foreign_policy",
    "astronomy",
    "marketing",
]
MAX_Q_CHARS = 220          # short questions keep CoTs (and runtimes) bounded
MAX_OPT_CHARS = 60


def main() -> None:
    path = hf_hub_download("cais/mmlu", "all/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    rng = random.Random(SEED)
    rows = []
    for subject in SUBJECTS:
        sub = df[df["subject"] == subject]
        candidates = []
        for _, r in sub.iterrows():
            q, choices, answer = str(r["question"]), list(r["choices"]), int(r["answer"])
            if len(choices) != 4 or len(q) > MAX_Q_CHARS:
                continue
            if any(len(str(c)) > MAX_OPT_CHARS for c in choices):
                continue
            if "\n" in q or any("\n" in str(c) for c in choices):
                continue
            candidates.append((q, [str(c) for c in choices], answer))
        rng.shuffle(candidates)
        for i, (q, choices, answer) in enumerate(candidates[:PER_SUBJECT]):
            # Full subject in the id: truncation collapsed high_school_geography
            # and high_school_biology into duplicate ids (caught by Lab 10's
            # dataset validator in the run-3 sweep).
            rows.append({
                "id": f"{subject}_{i:02d}",
                "domain": subject,
                "question": q,
                "option_a": choices[0], "option_b": choices[1],
                "option_c": choices[2], "option_d": choices[3],
                "answer_key": "ABCD"[answer],
            })
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} items to {OUT}")


if __name__ == "__main__":
    main()
