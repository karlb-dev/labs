from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="jspace-olmo-lineage")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify")
    arguments = parser.parse_args()
    if arguments.command == "verify":
        from .repro import verify_live_evidence

        result = verify_live_evidence()
        print(json.dumps(result, indent=1))
        if not result["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
