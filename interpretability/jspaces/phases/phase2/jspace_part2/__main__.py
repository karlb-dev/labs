# jspace-part2 CLI — the single command surface of the repro contract.
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

USAGE = """jspace-part2 <command> [args]

commands:
  audit-env            environment audit + pip-freeze lock -> manifests/
  inventory            R0 artifact inventory (hash run-dir artifacts; resumable)
  selftest             run the conformance test suite (CPU, fast)
  registry-list        show evidence registry rows (v2 event log)
  repro <evidence-id>  re-run/verify a registered evidence item (v1 path)
  repro2 <evidence-id> reproduction v2: isolated worktree at the recorded
                       commit, pinned constraints, input+model verification,
                       exact payload hashes  [--verify-only] [--workspace D]
"""


def cmd_selftest(argv):
    from .provenance import PKG_ROOT
    rc = 0
    for test in sorted((PKG_ROOT / "tests").glob("test_*.py")):
        print(f"=== {test.name}")
        rc |= subprocess.run([sys.executable, str(test)]).returncode
    raise SystemExit(rc)


def cmd_registry_list(argv):
    """Reads the v2 event log, so superseded items keep their metadata and
    `tier` is always a string (both were v1 defects)."""
    from . import registry as reg
    only_live = "--live" in argv
    rows = reg.resolve_all()
    for r in rows:
        if only_live and not r["live"]:
            continue
        flag = ("" if r["live"] else
                (" [WITHDRAWN]" if r["withdrawn"]
                 else f" [superseded_by {r['superseded_by']}]"))
        print(f"{r['evidence_id']:44s} tier={r['tier']:18s} "
              f"commit={str(r['code_commit'])[:8]}{flag}\n"
              f"    {r.get('what', '')}\n    cmd: {r.get('command', '')}")
    live = sum(1 for r in rows if r["live"])
    print(f"\n{len(rows)} evidence items · {live} live · "
          f"{sum(1 for r in rows if r['superseded_by'])} superseded · "
          f"{sum(1 for r in rows if r['withdrawn'])} withdrawn")


def cmd_repro(argv):
    from .lib import sha256_file
    from .provenance import registry_find
    if not argv:
        raise SystemExit("usage: jspace-part2 repro <evidence-id> [--verify-only]")
    eid = argv[0]
    row = registry_find(eid)
    if not row:
        raise SystemExit(f"unknown evidence id {eid!r} (jspace-part2 registry-list)")
    print(json.dumps({k: row[k] for k in
                      ("evidence_id", "tier", "what", "command", "code_commit")
                      if k in row}, indent=2))
    ok = True
    for out in row.get("outputs", []):
        p = Path(out["path"])
        if not p.exists():
            print(f"  MISSING output: {p}")
            ok = False
            continue
        h = sha256_file(p)
        match = h == out.get("sha256")
        print(f"  {'OK   ' if match else 'DIFF '} {p}  {h[:12]}"
              f"{'' if match else '  (expected ' + str(out.get('sha256'))[:12] + ')'}")
        ok = ok and match
    if "--verify-only" in argv or row.get("rerun") == "manual":
        note = row.get("repro_notes", "")
        print(f"verify {'PASS' if ok else 'FAIL'}. rerun recipe: "
              f"{row.get('command')}\n{note}")
        raise SystemExit(0 if ok else 1)
    print("re-running producer command ...")
    rc = subprocess.run(row["command"], shell=True).returncode
    raise SystemExit(rc)


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "audit-env":
        from . import audit_env
        audit_env.main(["audit_env", *rest])
    elif cmd == "inventory":
        from . import inventory
        inventory.main(["inventory", *rest])
    elif cmd == "selftest":
        cmd_selftest(rest)
    elif cmd == "registry-list":
        cmd_registry_list(rest)
    elif cmd == "repro":
        cmd_repro(rest)
    elif cmd == "repro2":
        from . import repro_v2
        repro_v2.main(rest)
    else:
        print(USAGE)
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
