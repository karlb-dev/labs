# R0: artifact inventory — hash every referenced run-dir artifact so results
# can point at immutable inputs and reviewers can verify byte identity.
# Weights are pinned by HF revision (provenance.resolve_model), not hashed.
#
# Usage: jspace-part2 inventory [--roots <dir,dir,...>] [--max-mb 4096]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .lib import sha256_file
from .provenance import PKG_ROOT, git_info

DEFAULT_ROOTS = [
    "/content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726",
    "/content/drive/MyDrive/interpret/special-lab-1/2026-07-26_v2",
    "/content/drive/MyDrive/interpret/special-lab-1/part2_20260727",
]
SKIP_SUFFIXES = {".pyc"}
SKIP_PARTS = {"__pycache__"}


def arg(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def classify(path: Path) -> str:
    parts = set(path.parts)
    if "lens" in parts:
        return "lens"
    if "metrics" in parts:
        return "metrics"
    if "figures" in parts:
        return "figure"
    if "logs" in parts:
        return "log"
    if "report" in parts:
        return "report"
    if "code" in parts:
        return "code-mirror"
    if "config" in parts:
        return "config"
    return "other"


def main(argv: list[str]) -> None:
    roots = arg(argv, "--roots", ",".join(DEFAULT_ROOTS)).split(",")
    max_mb = float(arg(argv, "--max-mb", "4096"))
    out_dir = Path(arg(argv, "--out",
        "/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/manifests"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "artifact_inventory.jsonl"

    done = set()
    if out_path.exists():  # resumable: skip already-hashed paths
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["path"])
            except Exception:
                continue
    n_new, n_skip_big, t0 = 0, 0, time.time()
    with out_path.open("a") as f:
        for root in roots:
            rootp = Path(root)
            if not rootp.exists():
                print(f"  (missing root {root})")
                continue
            for p in sorted(rootp.rglob("*")):
                if not p.is_file() or p.suffix in SKIP_SUFFIXES \
                        or SKIP_PARTS & set(p.parts):
                    continue
                sp = str(p)
                if sp in done:
                    continue
                size = p.stat().st_size
                row = {"path": sp, "root": root, "bytes": size,
                       "class": classify(p.relative_to(rootp)),
                       "mtime": int(p.stat().st_mtime)}
                if size > max_mb * 1e6:
                    row["sha256"] = None
                    row["skipped"] = f">{max_mb}MB"
                    n_skip_big += 1
                else:
                    row["sha256"] = sha256_file(p)
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_new += 1
                if n_new % 100 == 0:
                    print(f"  {n_new} hashed ({time.time()-t0:.0f}s)")
    summary = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "git": git_info(), "roots": roots, "n_new": n_new,
               "n_total": len(done) + n_new, "n_skipped_big": n_skip_big,
               "inventory": str(out_path)}
    (out_dir / "inventory_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv)
