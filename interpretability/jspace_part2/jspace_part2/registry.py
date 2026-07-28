# N1.2 — event-sourced evidence registry (schema v2), nextsteps_2_2 §2.4/§8.5.
#
# THE DEFECTS THIS REPAIRS (all confirmed in the v1 registry):
#  1. Supersession appended a PARTIAL row under the same evidence_id, and
#     lookup was last-row-wins — so superseding an item ERASED its command,
#     outputs and tier from every reader. `registry-list` then crashed or
#     printed `tier=None` for those rows (9 of 63 rows have tier None:
#     they are supersede events being read as evidence).
#  2. `created_utc` was embedded in the very file whose sha256 was
#     registered, so an exact rerun could never reproduce the hash — the
#     contract's central promise was unverifiable by construction.
#  3. No schema validation: a typo'd evidence_id (`olmo3think` vs
#     `olmo3-think`) silently created a dangling supersede link (it did,
#     three times; fixed by hand at d4f2c69 — v2 makes it impossible).
#  4. No write lock.
#
# DESIGN
#   events.jsonl is append-only with typed events:
#     {"event":"evidence_created",   "evidence_id":..., "schema_version":2, ...}
#     {"event":"evidence_superseded","evidence_id":old, "superseded_by":new, ...}
#     {"event":"evidence_withdrawn", "evidence_id":..., "reason":...}
#     {"event":"evidence_reproduced","evidence_id":..., "runner":..., "ok":bool}
#   resolve() reconstructs an item from its creation event and ATTACHES
#   status events; a status event can never replace creation metadata.
#
#   Result files split deterministic science from volatile provenance:
#     {"payload": {...}, "payload_sha256": "...", "provenance": {...}}
#   `payload_sha256` is verified EXACTLY; the envelope only by schema.
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
EVENTS = PKG_ROOT / "reports" / "evidence_events.jsonl"
LEGACY = PKG_ROOT / "reports" / "evidence_registry.jsonl"
SCHEMA_VERSION = 2

VALID_EVENTS = {"evidence_created", "evidence_superseded",
                "evidence_withdrawn", "evidence_reproduced"}
VALID_TIERS = {"dev", "exploratory", "exploratory-pilot", "pilot",
               "methods", "confirmatory"}
CREATE_REQUIRED = ("evidence_id", "tier", "what", "command", "code_commit")


class RegistryError(RuntimeError):
    pass


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def payload_sha256(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


# ------------------------------------------------------------------ I/O
def _validate(ev: dict) -> None:
    et = ev.get("event")
    if et not in VALID_EVENTS:
        raise RegistryError(f"unknown event type {et!r}")
    if not ev.get("evidence_id"):
        raise RegistryError("event lacks evidence_id")
    if et == "evidence_created":
        missing = [k for k in CREATE_REQUIRED if not ev.get(k)]
        if missing:
            raise RegistryError(f"evidence_created missing {missing}")
        if ev["tier"] not in VALID_TIERS:
            raise RegistryError(f"tier {ev['tier']!r} not in {sorted(VALID_TIERS)}")
        for o in ev.get("outputs", []) or []:
            if not isinstance(o, dict) or "path" not in o:
                raise RegistryError("outputs must be [{path, sha256?, ...}]")


def append_event(ev: dict, *, path: Path = EVENTS) -> dict:
    ev = {"schema_version": SCHEMA_VERSION,
          "event_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} | ev
    _validate(ev)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if ev["event"] == "evidence_created":
                # duplicate-creation guard must run INSIDE the lock
                for e in read_events(path):
                    if (e.get("event") == "evidence_created"
                            and e["evidence_id"] == ev["evidence_id"]):
                        raise RegistryError(
                            f"evidence_id {ev['evidence_id']!r} already created "
                            f"at {e.get('event_utc')} — supersede it instead")
            else:
                known = {e["evidence_id"] for e in read_events(path)
                         if e.get("event") == "evidence_created"}
                if ev["evidence_id"] not in known:
                    raise RegistryError(
                        f"{ev['event']} references unknown evidence_id "
                        f"{ev['evidence_id']!r} (typo'd supersede links are "
                        f"exactly the v1 failure)")
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
           code_commit: str, outputs: list[dict] | None = None,
           inputs: dict | None = None, rerun: str = "auto",
           repro_notes: str = "", supersedes: str | None = None,
           **extra) -> dict:
    ev = append_event({"event": "evidence_created", "evidence_id": evidence_id,
                       "tier": tier, "what": what, "command": command,
                       "code_commit": code_commit, "outputs": outputs or [],
                       "inputs": inputs or {}, "rerun": rerun,
                       "repro_notes": repro_notes, **extra})
    if supersedes:
        supersede(supersedes, evidence_id,
                  reason=f"superseded by {evidence_id}")
    return ev


def supersede(old_id: str, new_id: str, *, reason: str = "") -> dict:
    return append_event({"event": "evidence_superseded", "evidence_id": old_id,
                         "superseded_by": new_id, "reason": reason})


def withdraw(evidence_id: str, *, reason: str) -> dict:
    return append_event({"event": "evidence_withdrawn",
                         "evidence_id": evidence_id, "reason": reason})


def record_reproduction(evidence_id: str, *, runner: str, ok: bool,
                        detail: dict | None = None) -> dict:
    return append_event({"event": "evidence_reproduced",
                         "evidence_id": evidence_id, "runner": runner,
                         "ok": bool(ok), "detail": detail or {}})


# ------------------------------------------------------------- resolve
def resolve(evidence_id: str, *, path: Path = EVENTS) -> dict:
    events = read_events(path)
    created = [e for e in events if e.get("event") == "evidence_created"
               and e["evidence_id"] == evidence_id]
    if len(created) != 1:
        raise RegistryError(
            f"{len(created)} creation events for {evidence_id!r} (expected 1)")
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
    return rec


def all_ids(*, path: Path = EVENTS) -> list[str]:
    return [e["evidence_id"] for e in read_events(path)
            if e.get("event") == "evidence_created"]


def resolve_all(*, path: Path = EVENTS) -> list[dict]:
    return [resolve(i, path=path) for i in all_ids(path=path)]


# ----------------------------------------------------------- migration
#
# AUDITED LINK REPAIRS. The v1 registry contains supersede rows whose
# TARGET id was never created — v1 accepted them silently, so the
# supersession never attached and the superseded item kept reading as
# live. Two of them matter scientifically: both are Gemma analyses whose
# conclusions were REVERSED by their successors, and both were still
# listed as live evidence at 53532f8. The remaining three name a
# never-created evidence_id on the LEFT (the `olmo3think` vs `olmo3-think`
# slug typo); those were already re-filed correctly at d4f2c69, so the
# stale rows are dropped rather than repaired.
LINK_REPAIRS = {
    ("linearization-faithfulness-gemma4-31b-v1", "linearization-faithfulness-v2"):
        "linearization-faithfulness-gemma4-31b-v2",
    ("local-linearity-gemma4-31b-v1", "local-linearity-v3-extended-eps"):
        "local-linearity-v3-gemma4-31b",
}


def migrate_legacy(*, legacy: Path = LEGACY, out: Path = EVENTS,
                   dry_run: bool = False) -> dict:
    """v1 rows -> v2 events. A v1 row is a CREATION if it carries a tier
    (v1 supersede rows carried only evidence_id+superseded_by, which is
    the bug: they overwrote creation metadata on lookup)."""
    rows = [json.loads(l) for l in Path(legacy).read_text().splitlines()
            if l.strip()]
    creations, links, skipped = [], [], []
    for r in rows:
        if r.get("tier") and r.get("command"):
            creations.append(r)
        elif r.get("superseded_by") or r.get("withdrawn"):
            links.append(r)
        else:
            skipped.append(r)
    # a v1 creation row could ALSO carry superseded_by inline
    inline = [(r["evidence_id"], r["superseded_by"]) for r in creations
              if r.get("superseded_by")]
    summary = {"legacy_rows": len(rows), "creations": len(creations),
               "link_rows": len(links), "inline_links": len(inline),
               "unclassified": len(skipped)}
    if dry_run:
        summary["skipped_examples"] = skipped[:3]
        return summary
    if Path(out).exists():
        raise RegistryError(f"{out} already exists; migration is one-shot")
    # v1 allowed the same evidence_id to be created twice (the runner wrote a
    # generic row, then a curated row followed). Merge them into ONE creation
    # event, preferring the later (curated) values, keeping both `what`
    # strings so nothing is lost.
    merged: dict[str, dict] = {}
    for r in creations:
        eid = r["evidence_id"]
        if eid in merged:
            summary.setdefault("duplicate_creations_merged", []).append(eid)
            prior_what = merged[eid].get("what")
            merged[eid] = merged[eid] | {k: v for k, v in r.items() if v not in (None, "")}
            merged[eid]["v1_duplicate_whats"] = [prior_what, r.get("what")]
        else:
            merged[eid] = dict(r)
    creations = list(merged.values())

    seen = set()
    for r in creations:
        eid = r["evidence_id"]
        seen.add(eid)
        ev = {k: v for k, v in r.items()
              if k not in ("superseded_by", "registered_utc", "withdrawn")}
        ev["event"] = "evidence_created"
        ev["migrated_from_v1"] = True
        ev["v1_registered_utc"] = r.get("registered_utc")
        if not ev.get("code_commit"):
            # v1 did not enforce this; record the absence rather than
            # inventing a commit or silently dropping the item.
            ev["code_commit"] = "UNRECORDED_IN_V1"
            summary.setdefault("creations_without_code_commit", []).append(eid)
        append_event(ev, path=out)
    n_link = 0
    for old, new in inline:
        if new in seen:
            supersede(old, new, reason="migrated from v1 inline link")
            n_link += 1
        else:
            summary.setdefault("dangling_links", []).append([old, new])
    for r in links:
        old, tgt = r["evidence_id"], r.get("superseded_by")
        if old not in seen:
            # left-hand id never existed (the olmo3think slug typo); the
            # correct link was re-filed separately, so this row is stale.
            summary.setdefault("v1_rows_naming_nonexistent_evidence", []).append([old, tgt])
            continue
        if r.get("withdrawn"):
            withdraw(old, reason=r.get("reason", "migrated v1 withdrawal"))
            n_link += 1
            continue
        if tgt in seen:
            supersede(old, tgt, reason="migrated from v1")
            n_link += 1
        elif (old, tgt) in LINK_REPAIRS:
            fixed = LINK_REPAIRS[(old, tgt)]
            supersede(old, fixed,
                      reason=(f"v1 link named a never-created target {tgt!r}; "
                              f"repaired to {fixed!r} during v2 migration — "
                              f"this supersession had NEVER attached, so the "
                              f"item was still reading as live evidence"))
            n_link += 1
            summary.setdefault("links_repaired", []).append([old, tgt, fixed])
        else:
            summary.setdefault("dangling_links_unrepaired", []).append([old, tgt])
    summary["events_written"] = len(seen) + n_link
    return summary
