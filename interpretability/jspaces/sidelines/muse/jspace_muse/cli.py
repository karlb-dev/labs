"""jspace-muse CLI."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jspace-muse")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="CPU self-tests")
    sub.add_parser("stage", help="Download/hash Muse snapshot")
    sub.add_parser("admit", help="Pre-fit geometry admission")
    p_fit = sub.add_parser("fit", help="120-prompt WikiText J-lens fit")
    p_fit.add_argument("--dim-batch", type=int, default=None)
    p_fit.add_argument("--slices", type=str, default=None)
    sub.add_parser("battery", help="Run compact promising battery")
    sub.add_parser("figures", help="Regenerate PNG figures from metrics")
    sub.add_parser("report", help="Build TeX/PDF handout + state of record")
    sub.add_parser("status", help="Print Drive metrics summary")

    args = parser.parse_args(argv)

    if args.cmd == "selftest":
        from . import selftest
        return selftest.run()
    if args.cmd == "stage":
        from .experiments.stage import stage
        stage()
        return 0
    if args.cmd == "admit":
        from .experiments.admission import run_pre_fit
        r = run_pre_fit()
        print(json.dumps(r["gates"], indent=2))
        return 0 if r["gates"]["all_pass"] else 2
    if args.cmd == "fit":
        from .experiments.fit import run
        slices = [int(x) for x in args.slices.split(",")] if args.slices else None
        run(dim_batch=args.dim_batch, slices=slices)
        return 0
    if args.cmd == "battery":
        from .experiments.battery import run_all
        run_all()
        return 0
    if args.cmd == "figures":
        from .figures import generate_all
        generate_all()
        return 0
    if args.cmd == "report":
        from .report import build_all
        build_all()
        return 0
    if args.cmd == "status":
        from .paths import DRIVE_ROOT
        print("DRIVE_ROOT", DRIVE_ROOT)
        for p in sorted((DRIVE_ROOT / "metrics").glob("*.json")):
            print(" ", p.name, p.stat().st_size)
        lens = DRIVE_ROOT / "lens" / "muse_glimmer_lens.pt"
        print("lens", "OK" if lens.exists() else "MISSING", lens)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
