"""Atomic resumable state with an immutable compatibility header."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .manifests import object_sha256


@dataclass(frozen=True)
class StateHeader:
    evidence_id: str
    config_sha256: str
    code_commit: str
    model_id: str
    model_revision: str
    environment_sha256: str
    schema_version: int = 1


class StateStore:
    def __init__(self, path: str | Path, header: StateHeader):
        self.path = Path(path)
        self.header = header

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        envelope = json.loads(self.path.read_text())
        current = asdict(self.header)
        saved = envelope.get("header", {})
        mismatches = {
            key: {"saved": saved.get(key), "current": value}
            for key, value in current.items()
            if saved.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "refusing incompatible checkpoint: "
                + json.dumps(mismatches, sort_keys=True)
            )
        payload = envelope.get("payload")
        if object_sha256(payload) != envelope.get("payload_sha256"):
            raise RuntimeError("checkpoint payload hash mismatch")
        return payload

    def write(self, payload: Mapping) -> None:
        value = dict(payload)
        envelope = {
            "schema_version": 1,
            "header": asdict(self.header),
            "payload": value,
            "payload_sha256": object_sha256(value),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}")
        temporary.write_text(json.dumps(envelope, indent=1, sort_keys=True) + "\n")
        os.replace(temporary, self.path)
