"""Durable Git/Drive mirrors for the OLMo-lineage recovery documents."""
from __future__ import annotations

import argparse
from pathlib import Path

from .manifests import atomic_json, atomic_text, file_sha256, git_info
from .paths import reports_dir

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORTS = (
    "reports/INPROGRESS_OLMO_LINEAGE.md",
    "reports/OLMO_LINEAGE_RESUME.md",
    "reports/OLMO_LINEAGE_DEVELOPMENT_REPORT.md",
)


def mirror_reports(*, require_clean: bool = True) -> dict:
    information = git_info()
    if require_clean and information["dirty_tree"]:
        raise RuntimeError(
            "refusing to publish recovery mirrors from a dirty Git tree")
    destination = reports_dir()
    rows = []
    for relative in SOURCE_REPORTS:
        source = PACKAGE_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / source.name
        atomic_text(target, source.read_text())
        rows.append({
            "source": str(source),
            "destination": str(target),
            "sha256": file_sha256(target),
            "bytes": int(target.stat().st_size),
        })
    payload = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "source_git": information,
        "mirrors": rows,
        "recovery_entrypoint": str(
            destination / "OLMO_LINEAGE_RESUME.md"),
    }
    index = destination / "olmo_lineage_recovery_index.json"
    atomic_json(index, payload)
    payload["index"] = {
        "path": str(index),
        "sha256": file_sha256(index),
        "bytes": int(index.stat().st_size),
    }
    return payload


def verify_mirrors() -> dict:
    destination = reports_dir()
    rows = []
    for relative in SOURCE_REPORTS:
        source = PACKAGE_ROOT / relative
        target = destination / source.name
        rows.append({
            "name": source.name,
            "source_exists": source.is_file(),
            "mirror_exists": target.is_file(),
            "source_sha256": file_sha256(source) if source.is_file() else None,
            "mirror_sha256": file_sha256(target) if target.is_file() else None,
        })
    ok = all(
        row["source_exists"] and row["mirror_exists"]
        and row["source_sha256"] == row["mirror_sha256"]
        for row in rows
    )
    return {"ok": ok, "mirrors": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="emergency recovery only; normal checkpoints must be committed",
    )
    arguments = parser.parse_args()
    if arguments.verify:
        result = verify_mirrors()
    else:
        result = mirror_reports(require_clean=not arguments.allow_dirty)
    import json

    print(json.dumps(result, indent=1))
    if arguments.verify and not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
