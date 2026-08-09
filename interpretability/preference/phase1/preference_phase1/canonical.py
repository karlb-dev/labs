"""Canonical serialization and scientific-content hashing (plan §3.4).

One serializer for everything whose hash is scientific identity: stable key
order, UTF-8, ``\n`` newlines, no timestamps, no output paths, explicit
schema version injected by the caller. jspaces convention:
``json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def canonical_json(value: Any) -> str:
    """Deterministic JSON for hashing and append-only registries."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    """sha256 of the canonical JSON of ``value``.

    Callers must pass only scientific content: no timestamps, no absolute
    paths, no machine identifiers. String fields should already be
    newline-normalized (`normalize_newlines`); this function does not walk
    the structure to normalize, it serializes exactly what it is given.
    """
    return sha256_text(canonical_json(value))


def stable_seed(*parts: Any, base: int = 0) -> int:
    """Content-derived 32-bit seed (course pattern: seeds from hashes, not
    wall clocks)."""
    text = "|".join(str(p) for p in parts)
    return (base + int(sha256_text(text)[:8], 16)) % (2**32)
