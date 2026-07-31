# Phase 3 provenance + event-sourced registry.
#
# Same event schema and hashing as jspace_part2.registry (canonical JSON,
# payload_sha256, evidence_created/superseded/withdrawn/reproduced, flock
# + fsync, duplicate-creation guard) but with the PHASE 3 tier vocabulary
# and study_id, and its own event log under jspace_phase3/reports/.
# Deliberately vendored rather than parameterised so that the Phase 2
# registry can never accept a phase3 tier and vice versa.
from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from jspace_part2.lib import sha256_file
from jspace_part2.provenance import git_info, require_clean_tree, resolve_model  # noqa: F401
from jspace_part2.registry import RegistryError, canonical_json, payload_sha256

PKG_ROOT = Path(__file__).resolve().parents[1]           # .../jspace_phase3
EVENTS = PKG_ROOT / "reports" / "evidence_events.jsonl"
STUDY_ID = "jspace-phase3"
SCHEMA_VERSION = 2

VALID_EVENTS = {"evidence_created", "evidence_superseded",
                "evidence_withdrawn", "evidence_reproduced",
                "evidence_corrected"}
VALID_TIERS = {"phase2-confirmatory", "phase3-development",
               "phase3-confirmatory", "phase3-replication", "methods"}
CREATE_REQUIRED = ("evidence_id", "tier", "what", "command", "code_commit")


def _validate(ev: dict) -> None:
    if ev.get("event") not in VALID_EVENTS:
        raise RegistryError(f"unknown event type {ev.get('event')!r}")
    if not ev.get("evidence_id"):
        raise RegistryError("event lacks evidence_id")
    if ev["event"] == "evidence_created":
        missing = [k for k in CREATE_REQUIRED if not ev.get(k)]
        if missing:
            raise RegistryError(f"evidence_created missing {missing}")
        if ev["tier"] not in VALID_TIERS:
            raise RegistryError(
                f"tier {ev['tier']!r} not in {sorted(VALID_TIERS)}")
        for o in ev.get("outputs", []) or []:
            if not isinstance(o, dict) or "path" not in o:
                raise RegistryError("outputs must be [{path, sha256?, ...}]")
    if ev["event"] == "evidence_corrected":
        fields = ev.get("corrected_fields")
        if not isinstance(fields, dict) or not fields:
            raise RegistryError(
                "evidence_corrected requires nonempty corrected_fields")
        if "tier" in fields and fields["tier"] not in VALID_TIERS:
            raise RegistryError(
                f"corrected tier {fields['tier']!r} not in "
                f"{sorted(VALID_TIERS)}")
        if not ev.get("reason"):
            raise RegistryError("evidence_corrected requires reason")


def append_event(ev: dict, *, path: Path = EVENTS) -> dict:
    ev = {"schema_version": SCHEMA_VERSION, "study_id": STUDY_ID,
          "event_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} | ev
    _validate(ev)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            known = {e["evidence_id"] for e in read_events(path)
                     if e.get("event") == "evidence_created"}
            if ev["event"] == "evidence_created":
                if ev["evidence_id"] in known:
                    raise RegistryError(
                        f"evidence_id {ev['evidence_id']!r} already created "
                        f"— supersede it instead")
            else:
                if ev["evidence_id"] not in known:
                    raise RegistryError(
                        f"{ev['event']} references unknown evidence_id "
                        f"{ev['evidence_id']!r}")
                nxt = ev.get("superseded_by")
                if nxt and nxt not in known:
                    raise RegistryError(
                        f"superseded_by {nxt!r} is not a created evidence id")
            f.write(canonical_json(ev) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return ev


def read_events(path: Path = EVENTS) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def create(evidence_id: str, *, tier: str, what: str, command: str,
           code_commit: str | None = None, outputs: list[dict] | None = None,
           inputs: dict | None = None, rerun: str = "auto",
           repro_notes: str = "", supersedes: str | None = None,
           **extra) -> dict:
    ev = append_event({
        "event": "evidence_created", "evidence_id": evidence_id,
        "tier": tier, "what": what, "command": command,
        "code_commit": code_commit or git_info()["code_commit"],
        "outputs": outputs or [], "inputs": inputs or {}, "rerun": rerun,
        "repro_notes": repro_notes, **extra})
    if supersedes:
        supersede(supersedes, evidence_id,
                  reason=f"superseded by {evidence_id}")
    return ev


def supersede(old_id: str, new_id: str, *, reason: str = "") -> dict:
    return append_event({"event": "evidence_superseded",
                         "evidence_id": old_id, "superseded_by": new_id,
                         "reason": reason})


def withdraw(evidence_id: str, *, reason: str) -> dict:
    return append_event({"event": "evidence_withdrawn",
                         "evidence_id": evidence_id, "reason": reason})


def correct(evidence_id: str, *, corrected_fields: dict,
            reason: str) -> dict:
    """Append a metadata correction while preserving the creation row."""
    return append_event({
        "event": "evidence_corrected",
        "evidence_id": evidence_id,
        "corrected_fields": corrected_fields,
        "reason": reason,
    })


def resolve(evidence_id: str, *, path: Path = EVENTS) -> dict:
    events = read_events(path)
    created = [e for e in events if e.get("event") == "evidence_created"
               and e["evidence_id"] == evidence_id]
    if len(created) != 1:
        raise RegistryError(
            f"{len(created)} creation events for {evidence_id!r}")
    rec = dict(created[0])
    rec["status_events"] = [e for e in events
                            if e.get("evidence_id") == evidence_id
                            and e.get("event") != "evidence_created"]
    rec["superseded_by"] = next(
        (e["superseded_by"] for e in reversed(rec["status_events"])
         if e["event"] == "evidence_superseded"), None)
    rec["withdrawn"] = any(e["event"] == "evidence_withdrawn"
                           for e in rec["status_events"])
    rec["live"] = not rec["withdrawn"] and rec["superseded_by"] is None
    effective = {
        key: value for key, value in rec.items()
        if key not in {"status_events"}
    }
    for event in rec["status_events"]:
        if event["event"] == "evidence_corrected":
            effective.update(event["corrected_fields"])
    rec["effective_metadata"] = effective
    rec["effective_tier"] = effective.get("tier", rec.get("tier"))
    return rec


def resolve_all(*, path: Path = EVENTS) -> list[dict]:
    ids = [e["evidence_id"] for e in read_events(path)
           if e.get("event") == "evidence_created"]
    return [resolve(i, path=path) for i in ids]


# ------------------------------------------------------------ envelopes
@dataclass
class Provenance3:
    evidence_id: str
    tier: str
    command: str
    config_path: str | None = None
    inputs: dict | None = None
    model: dict | None = None
    seed: int | None = None
    jlens_commit: str | None = None

    def block(self) -> dict:
        from . import __version__
        g = git_info()
        cfg_sha = (sha256_file(self.config_path)
                   if self.config_path and Path(self.config_path).exists()
                   else None)
        return {
            "study_id": STUDY_ID, "evidence_id": self.evidence_id,
            "tier": self.tier, **g,
            "package_version": f"jspace_phase3 {__version__}",
            "command": self.command, "config_path": self.config_path,
            "config_sha256": cfg_sha, "inputs": self.inputs or {},
            "model": self.model, "jlens_commit": self.jlens_commit,
            "seed": self.seed,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def write_result3(payload: dict, path: Path, prov: Provenance3) -> dict:
    """Deterministic payload + volatile provenance in separate envelopes
    (the jspace_part2 v2 convention: rerun -> identical payload_sha256)."""
    env = {"schema_version": 2, "payload": payload,
           "payload_sha256": payload_sha256(payload),
           "provenance": prov.block()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(env, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return env


def register(evidence_id: str, *, tier: str, what: str, command: str,
             outputs: list[Path | str], inputs: dict | None = None,
             supersedes: str | None = None, **extra) -> dict:
    """Create + hash-pin outputs in one call (the common producer tail)."""
    rows = [{"path": str(p), "sha256": sha256_file(p)} for p in outputs]
    return create(evidence_id, tier=tier, what=what, command=command,
                  outputs=rows, inputs=inputs, supersedes=supersedes, **extra)
