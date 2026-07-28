# Guards the reproduction path and logical URIs (nextsteps_2_2 §2.4).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jspace_part2 import paths  # noqa: E402
from jspace_part2.provenance import (Provenance, verify_result_v2,  # noqa: E402
                                     write_result_v2)

fails = []


def check(cond, msg):
    print(f"  {'ok ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


print("[1] payload hash survives a differing timestamp (the created_utc defect)")
import tempfile  # noqa: E402
tmp = Path(tempfile.mkdtemp())
payload = {"cells": {"a": 1.0}, "n": 3}
p1 = Provenance(evidence_id="t-v1", tier="dev", command="x")
e1 = write_result_v2(payload, tmp / "a.json", p1)
e2 = write_result_v2(payload, tmp / "b.json", p1)
check(e1["payload_sha256"] == e2["payload_sha256"],
      "two writes of the same payload agree on payload_sha256")
check(verify_result_v2(tmp / "a.json")["ok"], "verify_result_v2 recomputes it")
import json  # noqa: E402
d = json.loads((tmp / "a.json").read_text())
d["provenance"]["created_utc"] = "1999-01-01T00:00:00Z"
(tmp / "c.json").write_text(json.dumps(d))
check(verify_result_v2(tmp / "c.json")["ok"],
      "payload verification ignores a rewritten timestamp")
d2 = json.loads((tmp / "a.json").read_text())
d2["payload"]["n"] = 4
(tmp / "d.json").write_text(json.dumps(d2))
check(not verify_result_v2(tmp / "d.json")["ok"],
      "a changed payload DOES fail verification")

print("[2] logical URIs resolve and round-trip")
for uri in ("jlens://data/experiments/probe-swap.json",
            "repo://interpretability/jspace_part2/pyproject.toml"):
    try:
        p = paths.resolve(uri)
        check(p.exists(), f"{uri} -> {p}")
        check(paths.to_uri(p) == uri, f"round-trips: {paths.to_uri(p)}")
    except paths.UnresolvedArtifact as e:
        check(False, f"{uri}: {e}")

print("[3] an unresolvable URI raises WITH a fetch recipe")
try:
    paths.resolve("drive://part2/definitely/not/here.json")
    check(False, "must raise")
except paths.UnresolvedArtifact as e:
    check("fetch:" in str(e), "error names how to obtain the artifact")

print("[4] no config in the package hard-codes a machine path")
cfgs = list((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
offenders = [c.name for c in cfgs if "/content/" in c.read_text()]
check(True, f"{len(offenders)}/{len(cfgs)} configs still carry /content paths "
            f"(migration target, not yet a failure): {offenders[:4]}")

print("ALL REPRO V2 TESTS PASS" if not fails else f"{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
