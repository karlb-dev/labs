"""Deterministic result envelopes for the Gemma side track."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .manifests import atomic_json, git_info, object_sha256


@dataclass(frozen=True)
class Provenance:
    evidence_id: str
    tier: str
    command: str
    inputs: dict
    input_manifest_sha256: str
    model: dict | None = None
    seed_contract: str | None = None

    def block(self) -> dict:
        from . import __version__

        return {
            "study_id": "jspace-gemma-transport",
            "package_version": f"jspace_gemma {__version__}",
            "evidence_id": self.evidence_id,
            "tier": self.tier,
            "command": self.command,
            "inputs": self.inputs,
            "input_manifest_sha256": self.input_manifest_sha256,
            "model": self.model,
            "seed_contract": self.seed_contract,
            **git_info(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def write_result(payload: dict, path: str | Path, provenance: Provenance) -> dict:
    envelope = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
        "provenance": provenance.block(),
    }
    atomic_json(path, envelope)
    return envelope
