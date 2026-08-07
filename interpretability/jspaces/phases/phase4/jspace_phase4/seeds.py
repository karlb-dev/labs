"""Stable Phase 4 scientific seeds."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

SEED_CONTRACT = "sha256-canonical-components-v1"
STUDY_ID = "jspace-phase4"


@dataclass(frozen=True)
class SeedComponents:
    experiment_id: str
    item_id: str
    condition: str
    layer: int | None = None
    position: int | None = None
    base_seed: int = 0
    study_id: str = STUDY_ID

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def value(self) -> int:
        digest = hashlib.sha256(self.canonical_bytes()).digest()
        return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def stable_seed(*, experiment_id: str, item_id: str, condition: str,
                layer: int | None = None, position: int | None = None,
                base_seed: int = 0) -> int:
    return SeedComponents(
        experiment_id=experiment_id,
        item_id=item_id,
        condition=condition,
        layer=layer,
        position=position,
        base_seed=base_seed,
    ).value()


def stable_rng(**components) -> np.random.Generator:
    return np.random.default_rng(stable_seed(**components))
