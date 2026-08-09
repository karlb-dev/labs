"""``jspace-or1`` — study CLI.

Subcommands appear as their stages land; every model-backed producer
requires CUDA and a clean tree, and registers through the study registry.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jspace-or1", description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("registry-list", help="list registry events, one line each")
    sub.add_parser("verify", help="rehash every live registered output")
    sub.add_parser("selftest", help="run the CPU conformance suite via pytest")
    arguments = parser.parse_args(argv)

    if arguments.command == "registry-list":
        from .registry import read_events

        for row in read_events():
            print(
                f"{row['event_utc']}  {row['event']:<20} {row['evidence_id']}"
            )
        return 0
    if arguments.command == "verify":
        from .registry import verify_outputs

        result = verify_outputs()
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    if arguments.command == "selftest":
        import pytest

        from .paths import STUDY_ROOT

        return pytest.main([str(STUDY_ROOT / "tests"), "-q"])
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
