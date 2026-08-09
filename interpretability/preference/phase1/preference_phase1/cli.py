"""``pref1`` — the campaign CLI (HARNESS_DECISION.md).

Stages honor gates: model stages refuse provisional codebooks; the frozen
stage refuses dirty trees and requires the freeze record; mechanism is not
implemented pre-freeze by design (plan §10 is conditional on graduation).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from . import BANK_VERSION
from . import artifacts, paths
from .models import PINS
from .provenance import append_session_log, utc_now


def cmd_bank_audit(args: argparse.Namespace) -> int:
    from . import bank, lexical
    from .models import PRIMARY, load_tokenizer
    from .schema import Codebook
    from .canonical import canonical_hash

    manifest = json.loads((paths.data_root() / "lab38_codebook.json").read_text())
    codebook = Codebook(
        codebook_id=manifest["codebook_id"],
        tokenizer_ref=manifest["tokenizer_ref"],
        ar_pair=tuple(manifest["ar_pair"]),
        ro_pair=tuple(manifest["ro_pair"]),
        leading_space_policy=manifest["leading_space_policy"],
        selection_manifest_hash=canonical_hash(manifest),
    )
    items = bank.build_bank(codebook)
    audit = bank.audit_bank(items)
    regenerated_hash = bank.bank_content_hash(items)
    meta = json.loads((paths.data_root() / "lab38_preference_bank.meta.json").read_text())
    match = regenerated_hash == meta["bank_content_hash"]
    print(json.dumps({"audit_passed": audit["passed"],
                      "failures": audit["failures"],
                      "counts": audit["counts"],
                      "regeneration_matches_frozen": match}, indent=2))
    return 0 if audit["passed"] and match else 2


def cmd_smoke(args: argparse.Namespace) -> int:
    from .runner import execute_battery, load_bank_records
    from .smoke import microtask_plumbing_probe, select_smoke_items, smoke_report

    pin = PINS[args.model_tier]
    items = select_smoke_items(load_bank_records("full"))
    run_dir = execute_battery(
        pin=pin, stage="smoke", subset="full",
        run_dir=pathlib.Path(args.run_dir) if args.run_dir else None,
        batch_size=args.batch_size, capture=False,
        max_items=0, require_final_codebook=True,
        item_filter=[it["item_id"] for it in items],
    )
    report = smoke_report(run_dir)
    if not report["summary"]["microtask_attempted"]:
        probe = microtask_plumbing_probe(run_dir, pin)
        report["summary"]["microtask_plumbing_probe_ok"] = probe["probe_ok"]
        report["summary"]["instrument_ok"] = bool(
            report["summary"]["instrument_ok"] and probe["probe_ok"])
    print(json.dumps(report["summary"], indent=2))
    append_session_log(f"smoke complete on tier {args.model_tier}: {run_dir.name}")
    return 0 if report["summary"]["instrument_ok"] else 2


def cmd_behavioral(args: argparse.Namespace) -> int:
    from .runner import execute_battery

    pin = PINS[args.model_tier]
    stage = args.stage
    if stage == "behavioral_frozen":
        freeze_record = (paths.phase1_root() / "preregistration"
                         / "PREFERENCE_PHASE1_FREEZE_RECORD.md")
        if not freeze_record.exists():
            print("REFUSED: behavioral_frozen requires the freeze record "
                  "(single human gate, addendum §I). Run behavioral_dev.",
                  file=sys.stderr)
            return 3
    run_dir = execute_battery(
        pin=pin, stage=stage, subset=args.subset,
        run_dir=pathlib.Path(args.run_dir) if args.run_dir else None,
        batch_size=args.batch_size, capture=args.capture,
        max_items=args.max_items)
    append_session_log(f"{stage} run complete: {run_dir.name}")
    print(run_dir)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .reporting import build_report

    run_dir = pathlib.Path(args.run_dir)
    out = build_report(run_dir, make_figures=not args.no_plots)
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_registry_list(args: argparse.Namespace) -> int:
    from . import registry

    for ev in registry.live_events():
        eid = ev.get("event_id")
        tier = ev.get("scientific_tier")
        status = ev.get("status")
        live = "live" if ev.get("live") else "superseded/withdrawn"
        print(f"{eid:44s} {tier:16s} {status:24s} {live}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pref1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bank-audit", help="no-model bank audit + regen check")
    p.set_defaults(fn=cmd_bank_audit)

    p = sub.add_parser("smoke", help="tier-a instrument smoke")
    p.add_argument("--model-tier", default="a", choices=sorted(PINS))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--run-dir", default="")
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("behavioral", help="behavioral battery")
    p.add_argument("--model-tier", default="b", choices=sorted(PINS))
    p.add_argument("--stage", default="behavioral_dev",
                   choices=["behavioral_dev", "behavioral_frozen"])
    p.add_argument("--subset", default="dev", choices=["dev", "full"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--capture", action="store_true")
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--run-dir", default="",
                   help="existing run dir to resume (same-command resume)")
    p.set_defaults(fn=cmd_behavioral)

    p = sub.add_parser("report", help="analysis tables + figures for a run")
    p.add_argument("run_dir")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("registry-list", help="live evidence events")
    p.set_defaults(fn=cmd_registry_list)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
