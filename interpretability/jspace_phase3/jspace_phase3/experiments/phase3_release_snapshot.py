"""Stage-0 immutable inventory for the Phase 3 release audit.

Hashes the frozen and current release inputs without modifying any Phase 3
outcome. The registry hash and live inventory describe the state immediately
before this command appends its own methods event.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from jspace_part2.lib import sha256_file
from jspace_part2.paths import resolve as resolve_uri

from ..paths3 import manifests_dir, run_root
from ..provenance3 import EVENTS, register, require_clean_tree, resolve_all

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
PRE_AUDIT_TAG = "jspace-phase3-pre-release-audit-v1"
FREEZE_TAG = "jspace-phase3-freeze-v1"
QWEN_LENS_URI = (
    "model://local/models--neuronpedia--jacobian-lens/snapshots/"
    "a4114d7752d11eb546e6cf372213d7e75526d3a1/qwen3.6-27b/jlens/"
    "Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
)
QWEN_LENS_HISTORICAL_SHA256 = (
    "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True).strip()


def artifact(label: str, path: Path, *, logical_uri: str | None = None,
             expected_sha256: str | None = None) -> dict:
    row = {"label": label, "path": str(path), "logical_uri": logical_uri}
    if not path.exists():
        row |= {"available": False, "expected_sha256": expected_sha256}
        return row
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(
            f"{label}: {digest} != expected {expected_sha256}")
    return row | {
        "available": True,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "expected_sha256": expected_sha256,
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    require_clean_tree("--allow-dirty" in sys.argv)
    root = run_root()
    metrics = root / "metrics"
    report = PACKAGE_ROOT / "reports" / "REPORT_PHASE3.md"
    handout = (
        PACKAGE_ROOT / "reports" / "handout" / "jspace_phase3_handout.pdf"
    )
    partition = PACKAGE_ROOT / "preregistration" / "partition_phase3.json"
    prereg = (
        PACKAGE_ROOT
        / "preregistration"
        / "SCIENTIFIC_PREREGISTRATION_PHASE3.md"
    )

    files: list[dict] = [
        artifact("frozen partition", partition),
        artifact("frozen preregistration", prereg),
        artifact("current report", report),
        artifact("current handout", handout),
        artifact("pre-audit evidence registry", EVENTS),
    ]
    for slug in ("olmo31-think", "olmo31-instruct", "qwen36-27b"):
        for side, dirname, stem in (
                ("confirmatory", "p3_grid", "p3_grid"),
                ("replication", "p3_grid_replication",
                 "p3_grid_replication")):
            files.append(artifact(
                f"{slug} {side} raw parquet",
                metrics / slug / dirname / f"{stem}_{slug}.parquet"))

    for slug, uri in (
        ("olmo31-think", "drive://lens/olmo31think_lens.pt"),
        ("olmo31-instruct", "drive://lens/olmo31instruct_lens.pt"),
    ):
        files.append(artifact(
            f"{slug} primary lens", Path(resolve_uri(uri)), logical_uri=uri))

    try:
        qwen_path = Path(resolve_uri(QWEN_LENS_URI))
    except Exception:
        qwen_path = Path("/content/hf_local") / QWEN_LENS_URI.removeprefix(
            "model://local/")
    files.append(artifact(
        "qwen36-27b primary lens", qwen_path, logical_uri=QWEN_LENS_URI,
        expected_sha256=QWEN_LENS_HISTORICAL_SHA256))
    files.append(artifact(
        "olmo3-base lineage lens",
        root / "lens" / "olmo3-base_lens.pt",
        logical_uri="phase3-run://lens/olmo3-base_lens.pt",
        expected_sha256=(
            "92f32e38dc4dffc45dda4e0c34a75f5433238f2046ae00046a4fe3fe1226b696"
        )))

    part = json.loads(partition.read_text())
    live = sorted(resolve_all(), key=lambda r: r["evidence_id"])
    inventory_payload = {
        "schema_version": 1,
        "snapshot_kind": "pre-release-audit-live-evidence",
        "registry_sha256": sha256_file(EVENTS),
        "n_created": len(live),
        "n_live": sum(bool(r["live"]) for r in live),
        "records": live,
    }
    out_dir = manifests_dir()
    inventory_path = out_dir / "phase3_pre_audit_live_evidence.json"
    snapshot_path = out_dir / "phase3_pre_audit_snapshot.json"
    atomic_json(inventory_path, inventory_payload)
    snapshot_payload = {
        "schema_version": 1,
        "snapshot_kind": "phase3-pre-release-audit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "current_head": git("rev-parse", "HEAD"),
            "pre_audit_tag": PRE_AUDIT_TAG,
            "pre_audit_head": git("rev-list", "-n", "1", PRE_AUDIT_TAG),
            "freeze_tag": FREEZE_TAG,
            "freeze_head": git("rev-list", "-n", "1", FREEZE_TAG),
            "freeze_parent": git("rev-parse", f"{FREEZE_TAG}^"),
        },
        "freeze": {
            "gate_event_id": "p3-partition-freeze-v1",
            "seed": part["payload"]["seed"],
            "p3p3_model": part["payload"]["p3p3_model"],
            "families": [
                part["payload"]["balance_report"]["n_confirmatory"],
                part["payload"]["balance_report"]["n_replication"],
            ],
            "intersection": part["payload"]["balance_report"]["intersection"],
            "partition_payload_sha256": part["payload_sha256"],
        },
        "artifacts": files,
        "live_evidence_inventory": {
            "path": str(inventory_path),
            "sha256": sha256_file(inventory_path),
        },
        "immutability": (
            "No Phase 3 outcome file was modified; unavailable artifacts are "
            "recorded as such with their historical expected hash."
        ),
    }
    atomic_json(snapshot_path, snapshot_payload)
    cmd = "python -m jspace_phase3.experiments.phase3_release_snapshot"
    register(
        "p3-release-pre-audit-snapshot-v1",
        tier="methods",
        what=(
            "Immutable Phase 3 pre-release-audit artifact hashes and resolved "
            "live-evidence inventory"
        ),
        command=cmd,
        outputs=[snapshot_path, inventory_path],
        inputs={"pre_audit_tag": PRE_AUDIT_TAG, "freeze_tag": FREEZE_TAG},
    )
    print(json.dumps({
        "snapshot": str(snapshot_path),
        "inventory": str(inventory_path),
        "n_artifacts": len(files),
        "unavailable": [r["label"] for r in files
                        if not r.get("available")],
    }, indent=1))


if __name__ == "__main__":
    main()
