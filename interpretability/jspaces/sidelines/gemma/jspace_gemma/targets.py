"""Frozen target helpers."""
from __future__ import annotations

import hashlib
from typing import Iterable


def token_subset_manifest(tokenizer, texts: Iterable[str], *, extras: Iterable[int] = ()) -> dict:
    token_ids = set(int(value) for value in extras)
    rows = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_ids.update(int(value) for value in encoded)
        rows.append(
            {
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "token_ids": [int(value) for value in encoded],
            }
        )
    return {"selected_token_ids": sorted(token_ids), "source_rows": rows}
