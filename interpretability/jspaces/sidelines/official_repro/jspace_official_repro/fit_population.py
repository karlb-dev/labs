"""WikiText fit-population materialization (OLMO_FIT_CONTRACT §2).

Upstream `load_wikitext_prompts` semantics verbatim: stream
`Salesforce/wikitext` / `wikitext-103-raw-v1` / `train`, accept a record
iff ``len(record["text"].strip()) >= 600`` (leading/trailing strip only),
keep the **raw, unstripped** text. First 1,000 qualifying records are the
fit population (halves: even stream-order index = A, odd = B). Records
after the first 1,000 feed timing/runtime sentinels only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .manifests import file_sha256, text_sha256, write_json
from .paths import CONFIGS, FIT_DATA

MIN_CHARS = 600
N_FIT = 1000
N_SENTINEL = 8  # extra rows beyond the fit population, for timing/sentinels

FIT_JSONL = FIT_DATA / "wikitext_first1000_min600.jsonl"
SENTINEL_JSONL = FIT_DATA / "wikitext_sentinels_after1000.jsonl"
MANIFEST = CONFIGS / "fit_population_manifest.json"


def materialize() -> dict:
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
        streaming=True,
    )
    fingerprint = {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
    }
    # Resolve the exact hub revision for the manifest.
    from huggingface_hub import HfApi

    info = HfApi().dataset_info("Salesforce/wikitext")
    fingerprint["hub_revision"] = info.sha

    rows: list[dict] = []
    sentinels: list[dict] = []
    for stream_index, record in enumerate(dataset):
        text = record["text"]
        if len(text.strip()) >= MIN_CHARS:
            row = {
                "fit_index": len(rows) + len(sentinels),
                "stream_index": stream_index,
                "text": text,
                "raw_chars": len(text),
                "stripped_chars": len(text.strip()),
                "raw_sha256": text_sha256(text),
            }
            if len(rows) < N_FIT:
                row["fit_index"] = len(rows)
                row["half"] = "A" if len(rows) % 2 == 0 else "B"
                rows.append(row)
            else:
                row["fit_index"] = None
                row["role"] = "timing_sentinel_pool"
                sentinels.append(row)
                if len(sentinels) >= N_SENTINEL:
                    break
    assert len(rows) == N_FIT, len(rows)

    FIT_DATA.mkdir(parents=True, exist_ok=True)
    with FIT_JSONL.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with SENTINEL_JSONL.open("w") as handle:
        for row in sentinels:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    concat = hashlib.sha256()
    for row in rows:
        concat.update(row["text"].encode("utf-8"))
        concat.update(b"\x00")
    manifest = {
        "criterion": "len(record['text'].strip()) >= 600  (upstream verbatim)",
        "text_form": "raw unstripped record text",
        "n_fit": N_FIT,
        "halves": {"A": "even fit_index", "B": "odd fit_index"},
        "n_sentinel_pool": len(sentinels),
        "source": fingerprint,
        "fit_jsonl_sha256": file_sha256(FIT_JSONL),
        "sentinel_jsonl_sha256": file_sha256(SENTINEL_JSONL),
        "canonical_text_concat_sha256": concat.hexdigest(),
        "stream_index_range": [rows[0]["stream_index"], rows[-1]["stream_index"]],
        "char_stats": {
            "raw_min": min(r["raw_chars"] for r in rows),
            "raw_max": max(r["raw_chars"] for r in rows),
            "raw_mean": sum(r["raw_chars"] for r in rows) / N_FIT,
        },
    }
    write_json(MANIFEST, manifest)
    return manifest


if __name__ == "__main__":
    result = materialize()
    print(json.dumps({k: v for k, v in result.items() if k != "char_stats"},
                     indent=2)[:1200])
