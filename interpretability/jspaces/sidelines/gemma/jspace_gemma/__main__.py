from __future__ import annotations

import json
import sys


def registry_list() -> None:
    from .registry import resolve_all

    for row in resolve_all():
        status = "live" if row["live"] else "non-live"
        print(
            f"{row['evidence_id']:<48} "
            f"tier={row['effective_tier']:<24} {status}"
        )


def run_root() -> None:
    from .paths import run_root as resolve

    print(resolve(create=False))


def verify() -> None:
    from .repro import verify_all

    result = verify_all()
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
        print(f"usage: jspace-gemma {{{'|'.join(commands)}}}")
        return 1
    commands[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
