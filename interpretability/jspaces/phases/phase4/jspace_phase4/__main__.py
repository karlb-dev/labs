from __future__ import annotations

import json
import sys

from . import registry4


def registry_list() -> None:
    for event in registry4.resolve_all():
        status = "live" if event["live"] else "non-live"
        print(
            f"{event['evidence_id']:<48} "
            f"tier={event['effective_tier']:<28} {status}")


def run_root() -> None:
    from .paths4 import run_root as resolve
    print(resolve(create=False))


def verify() -> None:
    from .repro4 import verify_live_evidence
    result = verify_live_evidence()
    print(json.dumps(result, indent=1))
    if not result["ok"]:
        raise SystemExit(1)


def environment() -> None:
    from .manifests import environment_payload
    print(json.dumps(environment_payload(), indent=1))


def main() -> int:
    commands = {
        "registry-list": registry_list,
        "run-root": run_root,
        "verify": verify,
        "environment": environment,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        print(f"usage: jspace-phase4 {{{'|'.join(commands)}}}")
        return 1
    commands[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
