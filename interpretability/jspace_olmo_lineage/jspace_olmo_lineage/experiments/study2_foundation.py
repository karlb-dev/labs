"""Register OLMo study-2 ancestry first, then the frozen wedge foundation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from ..manifests import (
    atomic_json,
    atomic_text,
    environment_payload,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths import REPO_ROOT, manifests_dir, run_root
from ..registry import create, read_events, resolve


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BRANCH = "interp_jspace_olmo_lineage_2"
SHARED_PARENT = "901fb4fc7578a913088c7947a2e6240f7fc45aeb"
CONFIG = PACKAGE_ROOT / "configs/ol2_stage_wedge.yaml"
TRANSPORT = PACKAGE_ROOT / "configs/ol2_transport_validation.yaml"
PREREG = PACKAGE_ROOT / "preregistration/OLMO_LINEAGE_STUDY2_PREREGISTRATION.md"
INPROGRESS = PACKAGE_ROOT / "reports/INPROGRESS_OLMO_LINEAGE_2.md"
REGISTRY = PACKAGE_ROOT / "reports/evidence_events.jsonl"
PHASE4_REGISTRY = REPO_ROOT / "interpretability/jspace_phase4/reports/evidence_events.jsonl"

GOVERNING = (
    Path("/content/drive/MyDrive/interpret/jspace_lab_sidelines_2.md"),
    Path("/content/drive/MyDrive/interpret/jspace_lab_sidelines_2_addendum.md"),
    Path("/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_olmo_lineage_1.md"),
    Path("/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_olmo_lineage_1_addendum.md"),
    PACKAGE_ROOT / "reports/OLMO_LINEAGE_STATE_OF_RECORD.md",
    PACKAGE_ROOT / "reports/OLMO_LINEAGE_CLAIMS_TABLE.md",
)

IMPORT_EVENTS = (
    "ol-checkpoint-inventory-v2",
    "ol-capacity-joint-dev-v1",
    "ol-lens-provenance-audit-v1",
    "ol-geometry-joint-dev-v1",
    "ol-independent-reconstruction-v1",
    "ol-phase4-final-import-bundle-v1",
)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True
    ).strip()


def _config() -> dict:
    value = yaml.safe_load(CONFIG.read_text())
    if value["status"] != "FROZEN_PRE_WEDGE_MODEL_LOAD":
        raise RuntimeError("OLMo wedge config is not frozen")
    if value["shared_parent_commit"] != SHARED_PARENT:
        raise RuntimeError("OLMo shared parent drift")
    if Path(value["run_root"]).resolve() != run_root().resolve():
        raise RuntimeError("JSPACE_OLMO_RUN_ROOT does not match frozen study-2 root")
    return value


def _study2_origins() -> list[str]:
    return [
        row["evidence_id"] for row in read_events()
        if row["event"] in {"evidence_created", "evidence_imported"}
        and row.get("evidence_id", "").startswith("ol2-")
    ]


def _verify_branch() -> dict:
    git = require_clean_tree(expected_branch=BRANCH)
    if _git("merge-base", SHARED_PARENT, "HEAD") != SHARED_PARENT:
        raise RuntimeError("OLMo study-2 branch is not descended from shared parent")
    return git


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


def _verify_event(evidence_id: str) -> dict:
    event = resolve(evidence_id)
    if not event["live"]:
        raise RuntimeError(f"source event is not live: {evidence_id}")
    outputs = []
    for output in event.get("outputs", []):
        path = Path(output["path"])
        observed = file_sha256(path)
        if observed != output["sha256"]:
            raise RuntimeError(f"source output hash drift: {path}")
        outputs.append({
            "path": str(path),
            "sha256": observed,
            "bytes": int(path.stat().st_size),
        })
    return {
        "evidence_id": evidence_id,
        "code_commit": event.get("code_commit"),
        "tier": event.get("tier"),
        "outputs": outputs,
    }


def ancestry() -> dict:
    git = _verify_branch()
    config = _config()
    if _study2_origins():
        raise RuntimeError("ancestry must be the first OLMo study-2 origin")
    source_event = _verify_event(config["ancestry"]["source_evidence_id"])
    source = Path(config["ancestry"]["source_json"])
    if file_sha256(source) != config["ancestry"]["source_json_sha256"]:
        raise RuntimeError("checkpoint inventory v2 source hash drift")
    inventory = json.loads(source.read_text())
    selected = {
        row["slug"]: row for row in inventory["artifacts"]
        if row["slug"] in {"olmo3-think-sft", "olmo3-think-dpo"}
    }
    if set(selected) != {"olmo3-think-sft", "olmo3-think-dpo"}:
        raise RuntimeError("inventory lacks the frozen SFT/DPO pair")
    expected = {
        spec["slug"]: spec
        for key, spec in config["checkpoints"].items()
        if key in {"think_sft", "think_dpo"}
    }
    for slug, spec in expected.items():
        row = selected[slug]
        if row["revision"] != spec["revision"] or row["repository"] != spec["model_id"]:
            raise RuntimeError(f"ancestry inventory drift for {slug}")
        if not row["intermediate_eligible"] or not row["model_contract"]["passes"]:
            raise RuntimeError(f"frozen intermediate is no longer eligible: {slug}")

    artifacts = {}
    for slug, row in selected.items():
        artifacts[slug] = {
            "repository": row["repository"],
            "revision": row["revision"],
            "stage": row["stage"],
            "declared_base_models": row["declared_base_models"],
            "expected_base_models": row["expected_base_models"],
            "declared_ancestry_matches": row["declared_ancestry_matches"],
            "ancestry_revision_qualification": row["ancestry_revision_qualification"],
            "model_contract": row["model_contract"],
            "metadata_files": row["metadata_files"],
            "tokenizer_semantics": row["tokenizer_semantics"],
            "weights": row["weights"],
            "recipe_card_statement_source": {
                "readme_sha256": row["metadata_files"]["README.md"]["sha256"],
                "base_model_field": row["declared_base_models"],
                "immutable_parent_revision_stated": False,
            },
        }
    payload = {
        "schema_version": 1,
        "evidence_id": "ol2-checkpoint-ancestry-v1",
        "tier": "methods",
        "status": "ancestry_qualified_with_explicit_revision_gap",
        "branch": git["branch"],
        "code_commit": git["code_commit"],
        "shared_parent_commit": SHARED_PARENT,
        "source_event": source_event,
        "source_inventory_sha256": file_sha256(source),
        "route": inventory["route"],
        "tokenizer_semantic_audit": inventory["tokenizer_semantic_audit"],
        "artifacts": artifacts,
        "qualification": config["ancestry"]["qualification"],
        "model_weights_downloaded": False,
        "model_outcome_opened": False,
    }
    payload["payload_sha256"] = object_sha256(payload)
    json_path = manifests_dir() / "ol2_checkpoint_ancestry_v1.json"
    markdown_path = manifests_dir() / "ol2_checkpoint_ancestry_v1.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("refusing to overwrite ancestry outputs")
    atomic_json(json_path, payload)
    lines = [
        "# OLMo study-2 checkpoint ancestry evidence",
        "",
        "Status: ancestry-qualified official SFT/DPO wedge with an explicit immutable-parent-revision gap.",
        "",
    ]
    for slug, row in artifacts.items():
        lines.extend([
            f"- `{slug}`: `{row['repository']}@{row['revision']}`; declared base `{', '.join(row['declared_base_models'])}`; config `{row['metadata_files']['config.json']['sha256']}`; tokenizer semantic fingerprint `{row['tokenizer_semantics']['semantic_fingerprint_sha256']}`; weight manifest `{row['weights']['manifest_sha256']}`.",
        ])
    lines.extend(["", config["ancestry"]["qualification"], ""])
    atomic_text(markdown_path, "\n".join(lines))
    create(
        "ol2-checkpoint-ancestry-v1",
        tier="methods",
        what=(
            "Exact official SFT/DPO config, tokenizer, architecture, weight, "
            "and model-card ancestry qualification before model load."
        ),
        command=(
            "python -m jspace_olmo_lineage.experiments.study2_foundation "
            "--stage ancestry"
        ),
        outputs=[json_path, markdown_path],
        inputs={
            "inventory_v2_sha256": file_sha256(source),
            "stage_wedge_config_sha256": file_sha256(CONFIG),
        },
        model_outcome_opened=False,
        weights_downloaded=False,
    )
    return {"json": str(json_path), "markdown": str(markdown_path)}


def foundation() -> dict:
    git = _verify_branch()
    config = _config()
    if _study2_origins() != ["ol2-checkpoint-ancestry-v1"]:
        raise RuntimeError("foundation requires ancestry as the sole prior study-2 origin")
    ancestry_event = _verify_event("ol2-checkpoint-ancestry-v1")
    raw = REGISTRY.read_bytes()
    prefix_contract = config["study1_registry_prefix"]
    prefix = raw[: int(prefix_contract["bytes"])]
    if len(prefix) != int(prefix_contract["bytes"]):
        raise RuntimeError("OLMo study-1 registry prefix is truncated")
    observed_prefix = hashlib.sha256(prefix).hexdigest()
    if observed_prefix != prefix_contract["sha256"]:
        raise RuntimeError("OLMo study-1 registry prefix hash drift")
    imports = [_verify_event(evidence_id) for evidence_id in IMPORT_EVENTS]
    tests = _run_tests()
    documents = []
    for path in (*GOVERNING, CONFIG, TRANSPORT, PREREG, INPROGRESS):
        documents.append({
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": int(path.stat().st_size),
        })
    phase4 = {
        "path": str(PHASE4_REGISTRY),
        "sha256": file_sha256(PHASE4_REGISTRY),
        "bytes": int(PHASE4_REGISTRY.stat().st_size),
        "write_authorized": False,
    }
    payload = {
        "schema_version": 1,
        "evidence_id": "ol2-foundation-v1",
        "tier": "methods",
        "status": "frozen_before_sft_or_dpo_weight_download",
        "branch": git["branch"],
        "code_commit": git["code_commit"],
        "shared_parent_commit": SHARED_PARENT,
        "run_root": str(run_root()),
        "study1_registry_prefix": {
            **prefix_contract,
            "observed_sha256": observed_prefix,
        },
        "ancestry_event": ancestry_event,
        "study1_imports": imports,
        "governing_and_frozen_documents": documents,
        "phase4_registry_immutable_boundary": phase4,
        "stage_wedge_contract": config,
        "transport_contract": yaml.safe_load(TRANSPORT.read_text()),
        "predictions_frozen": True,
        "two_registered_frames_required": True,
        "diffuse_outcome_predeclared_informative": True,
        "no_posthoc_prompt_repair": True,
        "tests": tests,
        "environment": environment_payload(require_gpu=True),
        "weights_downloaded_by_foundation": False,
        "model_outcome_opened": False,
        "claim_tier": "development/methods",
        "forbidden_write_paths": [
            "interpretability/jspace_phase4/",
            "interpretability/jspace_gemma/",
        ],
    }
    payload["payload_sha256"] = object_sha256(payload)
    output = manifests_dir() / "ol2_foundation_v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    atomic_json(output, payload)
    create(
        "ol2-foundation-v1",
        tier="methods",
        what=(
            "Frozen OLMo study-2 imports, two-frame SFT/DPO predictions, "
            "capability/router thresholds, transport gates, and isolation."
        ),
        command=(
            "python -m jspace_olmo_lineage.experiments.study2_foundation "
            "--stage foundation"
        ),
        outputs=[output],
        inputs={
            "study1_registry_prefix_sha256": observed_prefix,
            "ancestry_event_output_sha256": ancestry_event["outputs"][0]["sha256"],
            "stage_wedge_config_sha256": file_sha256(CONFIG),
            "transport_config_sha256": file_sha256(TRANSPORT),
            "preregistration_sha256": file_sha256(PREREG),
            "phase4_registry_sha256": phase4["sha256"],
        },
        model_outcome_opened=False,
        predictions_frozen=True,
    )
    return {"foundation": str(output), "sha256": file_sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("ancestry", "foundation"), required=True)
    arguments = parser.parse_args()
    result = ancestry() if arguments.stage == "ancestry" else foundation()
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
