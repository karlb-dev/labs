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
  registry-list        show evidence registry rows
  repro <evidence-id>  re-run/verify a registered evidence item
"""


def cmd_selftest(argv):
    from .provenance import PKG_ROOT
    test = PKG_ROOT / "tests" / "test_lib.py"
    raise SystemExit(subprocess.run([sys.executable, str(test)]).returncode)


def cmd_registry_list(argv):
    from .provenance import registry_rows
    for r in registry_rows():
        sup = f"  [superseded_by {r['superseded_by']}]" if r.get("superseded_by") else ""
        print(f"{r.get('evidence_id'):40s} tier={r.get('tier'):12s} "
              f"commit={str(r.get('code_commit'))[:8]}{sup}\n"
              f"    {r.get('what', '')}\n    cmd: {r.get('command', '')}")


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
    else:
        print(USAGE)
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
