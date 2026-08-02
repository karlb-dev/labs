"""Freeze and register the isolated OLMo-lineage study foundation."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from ..gpu import require_cuda_gpu
from ..imports import build_import_manifest
from ..manifests import (
    atomic_json,
    environment_payload,
    file_sha256,
    object_sha256,
    require_clean_tree,
    verify_constraints,
)
from ..paths import manifests_dir, resolve_uri, run_root
from ..recovery import mirror_reports
from ..registry import create

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]


def _load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("foundation config must be a mapping")
    return value


def _tracked_source_inventory() -> list[dict]:
    relative_root = PACKAGE_ROOT.relative_to(REPO_ROOT)
    names = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files", str(relative_root)],
        text=True,
    ).splitlines()
    rows = []
    for name in sorted(names):
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        rows.append({
            "path": name,
            "sha256": file_sha256(path),
            "bytes": int(path.stat().st_size),
        })
    return rows


def _run_tests() -> dict:
    command = [
        "python", "-m", "pytest", str(PACKAGE_ROOT / "tests"), "-q",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    payload = {
        "command": command,
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode:
        raise RuntimeError(
            "foundation conformance tests failed:\n" + result.stdout
            + result.stderr)
    return payload


def run(config_path: str | Path) -> dict:
    source = Path(config_path).resolve()
    config = _load_config(source)
    git = require_clean_tree(expected_branch=config["branch"])
    root = run_root()
    if root.resolve() != Path(config["run_root"]).resolve():
        raise RuntimeError("configured and environment run roots disagree")

    registry_path = PACKAGE_ROOT / "reports/evidence_events.jsonl"
    if registry_path.exists():
        raise FileExistsError(
            "foundation registry already exists; evidence is append-only")
    manifest_paths = {
        "environment": manifests_dir() / "ol_environment_lock.json",
        "imports": manifests_dir() / "ol_import_manifest.json",
        "conformance": manifests_dir() / "ol_foundation_conformance.json",
        "foundation": manifests_dir() / "ol_foundation_manifest.json",
    }
    collisions = [str(path) for path in manifest_paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "foundation outputs already exist and cannot be overwritten: "
            + ", ".join(collisions))

    tests = _run_tests()
    constraints = verify_constraints(
        PACKAGE_ROOT / "constraints.txt",
        package_names={str(name) for name in config["runtime_packages"]},
    )
    if not constraints["ok"]:
        raise RuntimeError(
            "runtime package lock mismatch: "
            + json.dumps(constraints["mismatches"], sort_keys=True))
    cuda = require_cuda_gpu()
    imports = build_import_manifest(config)
    environment = environment_payload(require_gpu=True)
    atomic_json(manifest_paths["environment"], environment)
    atomic_json(manifest_paths["imports"], imports)

    conformance = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "tier": config["tier"],
        "git": git,
        "config": {
            "path": str(source),
            "sha256": file_sha256(source),
        },
        "scientific_import_boundary": config["scientific_import_boundary"],
        "run_root": str(root),
        "tests": tests,
        "constraints": constraints,
        "cuda_gate": cuda,
        "source_inventory": _tracked_source_inventory(),
        "prohibitions": config["prohibitions"],
        "native_tiers": ["development", "methods"],
        "all_imports_hash_verified": True,
        "intervention_outcomes_opened": False,
    }
    conformance["content_sha256"] = object_sha256(conformance)
    atomic_json(manifest_paths["conformance"], conformance)

    mirrors = mirror_reports(require_clean=True)
    foundation = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "status": "frozen",
        "scientific_import_boundary": config["scientific_import_boundary"],
        "package_parent_commit": config["package_parent_commit"],
        "code_commit": git["code_commit"],
        "branch": git["branch"],
        "run_root": str(root),
        "models": config["models"],
        "six_axes": [
            "coordinate availability", "sparse capacity",
            "causal utilization", "downstream consumption",
            "external-state substitution", "temporal organization",
        ],
        "preregistration": next(
            row for row in imports["governance_documents"]
            if row["id"] == "side-preregistration"),
        "manifests": {
            key: {
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
            for key, path in manifest_paths.items()
            if key != "foundation"
        },
        "initial_recovery_mirror_snapshot": mirrors,
        "first_service_obligation": (
            "OLMo-3.1 Think and Instruct Bank-W baseline capability; "
            "no interventions"),
        "concurrent_track_policy": (
            "Phase 4, Gemma, and OLMo remain separate until a later "
            "integration phase"),
    }
    foundation["content_sha256"] = object_sha256(foundation)
    atomic_json(manifest_paths["foundation"], foundation)

    outputs = list(manifest_paths.values())
    event = create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "Frozen OLMo-lineage side-study foundation, imports, isolation "
            "contract, environment, tests, and recovery mirrors."),
        command=(
            "python -m jspace_olmo_lineage.experiments.foundation "
            f"--config {source.relative_to(REPO_ROOT)}"),
        outputs=outputs,
        inputs={
            "config": file_sha256(source),
            "import_manifest": file_sha256(manifest_paths["imports"]),
            "preregistration": next(
                row["sha256"] for row in imports["governance_documents"]
                if row["id"] == "side-preregistration"),
        },
        isolation={
            "repo_namespace": str(PACKAGE_ROOT.relative_to(REPO_ROOT)),
            "drive_root": str(root),
            "foreign_registry_writes": False,
        },
    )
    return {"foundation": foundation, "registry_event": event}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config), indent=1))


if __name__ == "__main__":
    main()
