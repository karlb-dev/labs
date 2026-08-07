# CLI: jspace-phase3 <subcommand>
from __future__ import annotations

import json
import sys

from . import provenance3 as p3


def cmd_registry_list():
    rows = p3.resolve_all()
    for r in rows:
        flags = []
        if r["superseded_by"]:
            flags.append(f"superseded_by {r['superseded_by']}")
        if r["withdrawn"]:
            flags.append("WITHDRAWN")
        flag = f" [{'; '.join(flags)}]" if flags else ""
        print(f"{r['evidence_id']:<44} tier={r['tier']:<22} "
              f"commit={r['code_commit'][:8]}{flag}")
        print(f"    {r['what']}")
        print(f"    cmd: {r['command']}")
    live = sum(1 for r in rows if r["live"])
    print(f"\n{len(rows)} evidence items · {live} live")


def cmd_run_root():
    from .paths3 import run_root
    print(run_root(create=False))


def main():
    cmds = {"registry-list": cmd_registry_list, "run-root": cmd_run_root}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(f"usage: jspace-phase3 {{{'|'.join(cmds)}}}")
        return 1
    cmds[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
