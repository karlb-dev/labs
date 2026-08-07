"""Deterministic scientific seeds.

Python's built-in ``hash`` is deliberately process-randomized. It must never
define an experimental realization.
"""
from __future__ import annotations

import hashlib

SEED_CONTRACT = "sha256-v1"


def stable_seed(namespace: str, item_id: str, base_seed: int = 0) -> int:
    payload = f"{namespace}\0{item_id}\0{base_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)

