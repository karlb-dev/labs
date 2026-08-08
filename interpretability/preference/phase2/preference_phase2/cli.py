"""pref2 CLI — thin wrappers over the package stages (plan Part X).

The GPU campaign drivers call the same functions programmatically; this
CLI exists for reproduction and handoff. No subcommand invents behavior:
each maps 1:1 onto a package function documented in the protocol
contracts.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pref2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("bank-audit", help="rebuild + audit the frozen bank")
    p = sub.add_parser("registry-list", help="print live registry events")
    p = sub.add_parser("freeze-verify", help="verify the freeze tag exists")
    p = sub.add_parser("power-simulate", help="rerun the power simulation")
    p = sub.add_parser("language-wall", help="raising campaign scan")
    p = sub.add_parser("port-audit", help="tokenizer/render audit")
    p.add_argument("--model", required=True)
    p = sub.add_parser("run", help="execute a battery stage")
    p.add_argument("--model", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--banks", required=True,
                   help="comma-separated bank ids")
    p.add_argument("--capture", action="store_true")
    p.add_argument("--pcmech-variant", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p = sub.add_parser("analyze", help="behavioral adjudication of a run")
    p.add_argument("run_dir")

    args = ap.parse_args(argv)

    if args.cmd == "bank-audit":
        from . import banks, codebooks, paths, scenarios
        cb = json.loads((paths.data_root() / "pref2_codebooks.json")
                        .read_text())
        fam = codebooks.families_from_manifest(cb)
        scenarios.self_check()
        items = banks.build_bank(fam)
        audit = banks.audit_bank(items, fam)
        meta = json.loads((paths.data_root() / "pref2_bank.meta.json")
                          .read_text())
        match = banks.bank_content_hash(items) == meta["bank_content_hash"]
        print(json.dumps({"audit_passed": audit["passed"],
                          "hash_matches_frozen": match,
                          "failures": audit["failures"][:5]}, indent=1))
        return 0 if audit["passed"] and match else 2

    if args.cmd == "registry-list":
        from . import registry
        for e in registry.live_events():
            flag = "live" if e["live"] else "superseded"
            print(f"{e['event_id']:44s} {e['scientific_tier']:18s} "
                  f"{e['status']:10s} {flag}")
        return 0

    if args.cmd == "freeze-verify":
        import subprocess

        from . import paths
        tag = subprocess.check_output(
            ["git", "-C", str(paths.repo_root()), "tag", "--list",
             "preference-phase2-freeze-v1"], text=True).strip()
        amend = (paths.prereg_root()
                 / "PREFERENCE_PHASE2_FREEZE_AMENDMENT_E2.md").exists()
        print(json.dumps({"tag": bool(tag), "e2_amendment": amend}))
        return 0 if tag else 2

    if args.cmd == "power-simulate":
        from .power import run_power_simulation
        res = run_power_simulation()
        print(json.dumps(res["gates"], indent=1))
        return 0

    if args.cmd == "language-wall":
        from .language_wall import main as lw_main
        return lw_main()

    if args.cmd == "port-audit":
        from . import paths, ports
        from .models import PINS
        rows = [json.loads(l) for l in
                open(paths.data_root() / "pref2_bank.jsonl",
                     encoding="utf-8")]
        res = ports.port_audit(PINS[args.model], rows)
        print(json.dumps({"passed": res["passed"],
                          "failures": res["failures"][:5]}, indent=1))
        return 0 if res["passed"] else 2

    if args.cmd == "run":
        from . import runner
        from .models import PINS
        run_dir = runner.execute_battery(
            pin=PINS[args.model], stage=args.stage,
            banks=args.banks.split(","), capture=args.capture,
            pcmech_variant=args.pcmech_variant, run_dir=args.run_dir,
            batch_size=args.batch_size)
        print(run_dir)
        return 0

    if args.cmd == "analyze":
        import pathlib

        from . import behavioral_analysis as ba
        from .artifacts import read_jsonl
        rows = read_jsonl(pathlib.Path(args.run_dir) / "results.jsonl")
        res = ba.analyze_behavioral(rows)
        print(json.dumps({"statuses": res["statuses"],
                          "pc_gate_pass": res["pc_gate"]["pass"],
                          "nc_alarm": res["nc_alarm"]["alarm"]}, indent=1))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
