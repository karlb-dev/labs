"""Freeze the Gemma study-2 foundation before G2.1 model data exist."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from jspace_gemma.manifests import (
    atomic_json,
    environment_payload,
    file_sha256,
    inventory,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import PACKAGE_ROOT, REPO_ROOT, run_root
from jspace_gemma.registry import create, read_events, resolve


EVIDENCE_ID = "gm2-foundation-v1"
BRANCH = "interp_jspace_gemma_transport_2"
SHARED_PARENT = "901fb4fc7578a913088c7947a2e6240f7fc45aeb"
CONFIG = PACKAGE_ROOT / "configs/gm2_backend_parity_calibration.yaml"
RELICENSE = PACKAGE_ROOT / "configs/gm2_stage1_relicense.yaml"
DESIGN = PACKAGE_ROOT / "preregistration/G2_STUDY2_FROZEN_DESIGN.md"
SENTENCES = PACKAGE_ROOT / "protocol/G2_STAGE1_CANDIDATE_SENTENCES.md"
SOURCE_REGISTRY = PACKAGE_ROOT / "reports/evidence_events.jsonl"
PHASE4_REGISTRY = REPO_ROOT / "interpretability/jspace_phase4/reports/evidence_events.jsonl"

GOVERNING = (
    Path("/content/drive/MyDrive/interpret/jspace_lab_sidelines_2.md"),
    Path("/content/drive/MyDrive/interpret/jspace_lab_sidelines_2_addendum.md"),
    Path("/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_gemma_1.md"),
    Path("/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_gemma_1_addendum.md"),
    PACKAGE_ROOT / "release/GEMMA_TRANSPORT_STATE_OF_RECORD.md",
    PACKAGE_ROOT / "release/gemma_transport_claim_ledger.md",
    PACKAGE_ROOT / "release/TRANSPORT_GATE_PROTOCOL.md",
)

IMPORT_EVENTS = (
    "gm-jvp-olmo-positive-control-v1",
    "gm-jvp-gemma-stage1-v1",
    "gm-jvp-gemma-backend-parity-v1",
    "gm-state-of-record-v1",
)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True
    ).strip()


def _run_tests() -> dict:
    command = ["python", "-m", "pytest", str(PACKAGE_ROOT / "tests"), "-q"]
    result = subprocess.run(
        command, cwd=REPO_ROOT, text=True, capture_output=True, check=True
    )
    return {
        "command": command,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "passed": True,
    }


def _verify_imports() -> list[dict]:
    rows = []
    for evidence_id in IMPORT_EVENTS:
        event = resolve(evidence_id)
        if not event["live"]:
            raise RuntimeError(f"study-1 import is not live: {evidence_id}")
        outputs = []
        for output in event.get("outputs", []):
            path = Path(output["path"])
            observed = file_sha256(path)
            if observed != output["sha256"]:
                raise RuntimeError(f"study-1 output hash drift: {path}")
            outputs.append({
                "path": str(path),
                "sha256": observed,
                "bytes": int(path.stat().st_size),
            })
        rows.append({
            "evidence_id": evidence_id,
            "code_commit": event.get("code_commit"),
            "tier": event.get("tier"),
            "outputs": outputs,
        })
    return rows


def main() -> None:
    git = require_clean_tree(branch=BRANCH)
    config = yaml.safe_load(CONFIG.read_text())
    if config["status"] != "FROZEN_PRE_G2_1" or config["evidence_id"] != "gm2-backend-parity-calibration-v1":
        raise RuntimeError("G2.1 calibration config is not frozen")
    if config["shared_parent_commit"] != SHARED_PARENT:
        raise RuntimeError("G2.1 shared parent drift")
    root = run_root()
    if root.resolve() != Path(config["run_root"]).resolve():
        raise RuntimeError("JSPACE_GEMMA_RUN_ROOT does not match frozen study-2 root")
    if _git("merge-base", SHARED_PARENT, "HEAD") != SHARED_PARENT:
        raise RuntimeError("Gemma study-2 branch is not descended from the shared parent")
    if any(
        row.get("evidence_id", "").startswith("gm2-")
        and row["event"] in {"evidence_created", "evidence_imported"}
        for row in read_events()
    ):
        raise RuntimeError("a Gemma study-2 origin event already exists")

    prefix = SOURCE_REGISTRY.read_bytes()
    source_prefix = {
        "path": str(SOURCE_REGISTRY),
        "bytes": len(prefix),
        "sha256": file_sha256(SOURCE_REGISTRY),
        "last_event": read_events()[-1]["evidence_id"],
    }
    imports = _verify_imports()
    tests = _run_tests()
    documents = inventory((*GOVERNING, CONFIG, RELICENSE, DESIGN, SENTENCES))
    phase4_before = {
        "path": str(PHASE4_REGISTRY),
        "bytes": int(PHASE4_REGISTRY.stat().st_size),
        "sha256": file_sha256(PHASE4_REGISTRY),
        "write_authorized": False,
    }
    payload = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "tier": "methods",
        "status": "frozen_before_g2_1",
        "branch": git["branch"],
        "code_commit": git["code_commit"],
        "shared_parent_commit": SHARED_PARENT,
        "run_root": str(root),
        "registry_prefix": source_prefix,
        "phase4_registry_immutable_boundary": phase4_before,
        "governing_and_frozen_documents": documents,
        "study1_imports": imports,
        "calibration_contract": config,
        "g2_2_target_file": {
            "path": str(RELICENSE.relative_to(REPO_ROOT)),
            "sha256": file_sha256(RELICENSE),
            "forbidden_to_g2_1_run_and_freeze_process": True,
        },
        "candidate_sentences": {
            "path": str(SENTENCES.relative_to(REPO_ROOT)),
            "sha256": file_sha256(SENTENCES),
            "committed_before_g2_1": True,
        },
        "gpu_serialization": {
            "gemma_lane_first": True,
            "one_model_resident": True,
            "olmo_wedge_may_not_begin_before_g2_2": True,
        },
        "forbidden_write_paths": [
            "interpretability/jspace_phase4/",
            "interpretability/jspace_olmo_lineage/",
        ],
        "tests": tests,
        "environment": environment_payload(require_gpu=True),
        "model_outcome_opened": False,
        "stage1_target_opened_by_foundation": False,
        "claim_tier": "methods/development",
    }
    payload["payload_sha256"] = object_sha256(payload)
    output = root / "manifests/gm2_foundation_v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    atomic_json(output, payload)
    create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "Frozen Gemma study-2 isolation, imports, G2.1 calibration design, "
            "G2.2 router, and pre-written outcome sentences before model data."
        ),
        command="python -m jspace_gemma.experiments.gm2_foundation",
        outputs=[output],
        inputs={
            "registry_prefix_sha256": source_prefix["sha256"],
            "phase4_registry_sha256": phase4_before["sha256"],
            "calibration_config_sha256": file_sha256(CONFIG),
            "relicense_config_sha256": file_sha256(RELICENSE),
            "design_sha256": file_sha256(DESIGN),
            "candidate_sentences_sha256": file_sha256(SENTENCES),
        },
        model_outcome_opened=False,
        target_threshold_frozen=False,
    )
    print(json.dumps({
        "foundation": str(output),
        "sha256": file_sha256(output),
        "shared_parent_commit": SHARED_PARENT,
        "study1_import_events": list(IMPORT_EVENTS),
    }, indent=1))


if __name__ == "__main__":
    main()
