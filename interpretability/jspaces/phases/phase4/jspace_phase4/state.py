"""Checkpoint state that refuses incompatible input manifests."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .manifests import object_sha256


def validate_state_header(saved: Mapping, current: Mapping) -> None:
    mismatches = {
        key: {"saved": saved.get(key), "current": value}
        for key, value in current.items()
        if saved.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Refusing to resume incompatible state: "
            + json.dumps(mismatches, sort_keys=True))


@dataclass(frozen=True)
class StateHeader:
    evidence_id: str
    input_manifest_sha256: str
    config_sha256: str
    model_revision: str
    bank_sha256: str
    partition_sha256: str
    schema_version: int = 1

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "input_manifest_sha256": self.input_manifest_sha256,
            "config_sha256": self.config_sha256,
            "model_revision": self.model_revision,
            "bank_sha256": self.bank_sha256,
            "partition_sha256": self.partition_sha256,
        }


class StateStore:
    def __init__(self, path: str | Path, header: StateHeader):
        self.path = Path(path)
        self.header = header

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        envelope = json.loads(self.path.read_text())
        validate_state_header(
            envelope.get("header", {}), self.header.as_dict())
        payload = envelope.get("payload")
        if object_sha256(payload) != envelope.get("payload_sha256"):
            raise RuntimeError("checkpoint payload hash mismatch")
        return payload

    def write(self, payload: Mapping) -> None:
        envelope = {
            "schema_version": 1,
            "header": self.header.as_dict(),
            "payload": dict(payload),
            "payload_sha256": object_sha256(dict(payload)),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(
            self.path.suffix + f".tmp{os.getpid()}")
        temporary.write_text(
            json.dumps(envelope, indent=1, sort_keys=True) + "\n")
        os.replace(temporary, self.path)
