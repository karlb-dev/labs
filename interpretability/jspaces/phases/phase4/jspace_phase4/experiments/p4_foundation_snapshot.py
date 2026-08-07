"""Bank the tested Phase 4 foundation and GPU-aware environment lock."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ..gpu import require_cuda_gpu
from ..imports3 import phase3_release_manifest
from ..manifests import (
    atomic_json,
    environment_payload,
    file_sha256,
    git_info,
    object_sha256,
    require_clean_tree,
    verify_constraints,
)
from ..paths4 import manifests_dir
from ..registry4 import create, resolve

EVIDENCE_ID = "p4-foundation-scaffold-v1"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = PACKAGE_ROOT / "jspace_phase4"


def source_inventory() -> dict:
    paths = [
        PACKAGE_ROOT / "pyproject.toml",
        PACKAGE_ROOT / "constraints.txt",
        PACKAGE_ROOT / "repro.sh",
        PACKAGE_ROOT / "README.md",
        PACKAGE_ROOT / "protocol/REPRO_CONTRACT_PHASE4.md",
        PACKAGE_ROOT / "preregistration/"
        "SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md",
        PACKAGE_ROOT / "reviews/jspace_lab_nextsteps_4_1.md",
        PACKAGE_ROOT / "reviews/jspace_lab_nextsteps_4_1_addendum.md",
        *sorted(MODULE_ROOT.rglob("*.py")),
        *sorted((PACKAGE_ROOT / "tests").glob("*.py")),
    ]
    return {
        str(path.relative_to(PACKAGE_ROOT)): {
            "sha256": file_sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for path in paths
    }


def main() -> None:
    clean = require_clean_tree()
    imported = resolve("p4-import-phase3-release-v1")
    if not imported["live"]:
        raise RuntimeError("immutable Phase 3 import is not live")

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
    )
    if test.returncode:
        raise RuntimeError(
            "Phase 4 conformance failed:\n" + test.stdout + test.stderr)
    constraints = verify_constraints(
        PACKAGE_ROOT / "constraints.txt",
        package_names={
            "huggingface_hub", "matplotlib", "numpy", "pandas",
            "pyarrow", "pyyaml", "scipy", "tokenizers", "torch",
            "transformers",
        },
    )
    if not constraints["ok"]:
        raise RuntimeError(
            "dependency lock mismatch: "
            + json.dumps(constraints["mismatches"], sort_keys=True))
    cuda_gate = require_cuda_gpu()
    environment = environment_payload(require_gpu=True)
    environment["cuda_hard_gate"] = cuda_gate
    environment["constraints_verification"] = constraints

    destination = manifests_dir()
    environment_path = destination / "p4_environment_lock.json"
    conformance_path = destination / "p4_foundation_conformance.json"
    manifest_path = destination / "p4_foundation_manifest.json"
    atomic_json(environment_path, environment)
    conformance = {
        "schema_version": 1,
        "command": "bash interpretability/jspaces/phases/phase4/repro.sh",
        "producer_test_command": (
            f"{sys.executable} -m pytest tests -q"),
        "returncode": test.returncode,
        "stdout": test.stdout.strip(),
        "stderr": test.stderr.strip(),
        "constraints_ok": True,
        "cuda_gate_passed": True,
    }
    atomic_json(conformance_path, conformance)
    payload = {
        "schema_version": 1,
        "status": (
            "foundation/development authorized; confirmatory and "
            "replication forbidden before PI sign-off and freeze"),
        "git": clean,
        "phase3_import": {
            "evidence_id": imported["evidence_id"],
            "source_commit": imported["source_commit"],
            "source_tag": imported["source_tag"],
            "source_registry_sha256":
                imported["source_registry_sha256"],
            "source_outputs": imported["source_outputs"],
            "release_manifest": phase3_release_manifest(),
        },
        "source_inventory": source_inventory(),
        "environment_lock_sha256": file_sha256(environment_path),
        "conformance_sha256": file_sha256(conformance_path),
        "contracts": {
            "seed": "sha256-canonical-components-v1",
            "scoring": "p4-scoring-v1",
            "phase_parser": "p4-phase-parser-v1",
            "primary_alias_aggregation":
                "prefix-disjoint-logsumexp",
            "cpu_model_fallback": "forbidden",
        },
    }
    envelope = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
        "provenance": {
            **git_info(),
            "evidence_id": EVIDENCE_ID,
            "tier": "methods",
            "command": (
                "python -m jspace_phase4.experiments."
                "p4_foundation_snapshot"),
        },
    }
    atomic_json(manifest_path, envelope)
    inputs = {
        "phase3_source_registry":
            imported["source_registry_sha256"],
        "phase3_release_manifest":
            phase3_release_manifest()["source_sha256"],
        "constraints": constraints["constraints_sha256"],
    }
    create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "Phase 4 foundation: 29-test conformance suite, pinned "
            "dependency audit, same-process RTX CUDA gate, immutable "
            "Phase 3 import, source inventory, and explicit no-"
            "confirmatory-before-freeze boundary."),
        command=(
            "python -m "
            "jspace_phase4.experiments.p4_foundation_snapshot"),
        outputs=[manifest_path, environment_path, conformance_path],
        inputs=inputs,
    )
    print(json.dumps({
        "evidence_id": EVIDENCE_ID,
        "manifest": str(manifest_path),
        "environment": str(environment_path),
        "conformance": str(conformance_path),
        "tests": test.stdout.strip(),
        "gpu": cuda_gate["name"],
    }, indent=1))


if __name__ == "__main__":
    main()
