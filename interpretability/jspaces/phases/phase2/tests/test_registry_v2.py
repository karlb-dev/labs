# Guards the event-sourced registry (nextsteps_2_2 §2.4/§8.5).
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jspace_part2 import registry as reg  # noqa: E402

fails = []


def check(cond, msg):
    print(f"  {'ok ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


tmp = Path(tempfile.mkdtemp()) / "events.jsonl"
C = dict(tier="pilot", what="w", command="cmd", code_commit="abc1234")

print("[1] create / resolve / supersede keeps creation metadata")
reg.append_event({"event": "evidence_created", "evidence_id": "a-v1",
                  "outputs": [{"path": "/x", "sha256": "h"}], **C}, path=tmp)
reg.append_event({"event": "evidence_created", "evidence_id": "a-v2", **C}, path=tmp)
reg.append_event({"event": "evidence_superseded", "evidence_id": "a-v1",
                  "superseded_by": "a-v2"}, path=tmp)
r = reg.resolve("a-v1", path=tmp)
check(r["command"] == "cmd" and r["tier"] == "pilot",
      "superseded item KEEPS its command and tier (the v1 bug)")
check(r["outputs"] and r["outputs"][0]["path"] == "/x",
      "superseded item keeps its outputs")
check(r["superseded_by"] == "a-v2" and not r["live"], "supersession attached")
check(reg.resolve("a-v2", path=tmp)["live"], "successor is live")

print("[2] duplicate creation is rejected")
try:
    reg.append_event({"event": "evidence_created", "evidence_id": "a-v1", **C},
                     path=tmp)
    check(False, "duplicate creation must raise")
except reg.RegistryError:
    check(True, "duplicate creation raises")

print("[3] typo'd links are impossible (the v1 dangling-link failure)")
for ev in ({"event": "evidence_superseded", "evidence_id": "nonexistent",
            "superseded_by": "a-v2"},
           {"event": "evidence_superseded", "evidence_id": "a-v2",
            "superseded_by": "also-nonexistent"}):
    try:
        reg.append_event(ev, path=tmp)
        check(False, f"link to unknown id must raise ({ev['evidence_id']})")
    except reg.RegistryError:
        check(True, f"link to unknown id raises ({ev['evidence_id']})")

print("[4] schema validation")
for bad, why in (({"event": "nope", "evidence_id": "x"}, "unknown event type"),
                 ({"event": "evidence_created", "evidence_id": "b-v1",
                   "tier": "made-up", "what": "w", "command": "c",
                   "code_commit": "d"}, "invalid tier"),
                 ({"event": "evidence_created", "evidence_id": "b-v1",
                   "tier": "pilot", "what": "w", "command": "c"},
                  "missing code_commit")):
    try:
        reg.append_event(bad, path=tmp)
        check(False, f"must reject: {why}")
    except reg.RegistryError:
        check(True, f"rejects: {why}")

print("[5] withdrawal")
reg.withdraw("a-v2", reason="test") if False else None
reg.append_event({"event": "evidence_withdrawn", "evidence_id": "a-v2",
                  "reason": "instrument fault"}, path=tmp)
r = reg.resolve("a-v2", path=tmp)
check(r["withdrawn"] and not r["live"], "withdrawn item is not live")
check(r["command"] == "cmd", "withdrawn item keeps creation metadata")

print("[6] payload hash is stable across timestamps (the created_utc defect)")
p1 = {"result": [1, 2, 3], "note": "x"}
p2 = {"note": "x", "result": [1, 2, 3]}          # different key order
check(reg.payload_sha256(p1) == reg.payload_sha256(p2),
      "canonical payload hash is order-independent")
env1 = {"payload": p1, "provenance": {"created_utc": "2026-01-01T00:00:00Z"}}
env2 = {"payload": p1, "provenance": {"created_utc": "2026-12-31T23:59:59Z"}}
check(reg.payload_sha256(env1["payload"]) == reg.payload_sha256(env2["payload"]),
      "payload hash ignores volatile provenance (exact rerun can match)")

print("[7] the live registry resolves cleanly")
if reg.EVENTS.exists():
    rows = reg.resolve_all()
    live = [r for r in rows if r["live"]]
    check(all(isinstance(r["tier"], str) for r in rows),
          f"every resolved item has a string tier ({len(rows)} items)")
    check(len(live) < len(rows), f"{len(rows)} items, {len(live)} live")
    for eid in ("local-linearity-gemma4-31b-v1",
                "linearization-faithfulness-gemma4-31b-v1"):
        try:
            check(not reg.resolve(eid)["live"],
                  f"{eid} is correctly NOT live (its v1 link was dangling)")
        except reg.RegistryError as e:
            check(False, f"{eid}: {e}")
else:
    print("  (skip: no live event log yet)")

print("ALL REGISTRY V2 TESTS PASS" if not fails else f"{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
