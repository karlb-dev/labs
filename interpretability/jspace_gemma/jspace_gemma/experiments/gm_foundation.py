"""Create the isolated foundation and immutable historical import boundary."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from jspace_gemma.architecture import manifest_from_config
from jspace_gemma.imports import verify_source_event
from jspace_gemma.manifests import (
    atomic_json,
    environment_payload,
    file_sha256,
    git_info,
    hf_remote_inventory,
    inventory,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import PACKAGE_ROOT, REPO_ROOT, directory, run_root
from jspace_gemma.registry import create, import_evidence

FORK_COMMIT = "3b041735d8b842de46a9c0a474fccd0c44e0841a"
GEMMA_ID = "google/gemma-4-31B-it"
GEMMA_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
OLMO_ID = "allenai/Olmo-3-32B-Think"
OLMO_REVISION = "ebd033e4f0b284d5973b82c0ccb62ad0dbe877d7"
HANDOUT_SOURCE_COMMIT = "4ea7a9ba7a534daa61e0d8c9960763b921a1b80b"
HANDOUT_SOURCE_PATH = "interpretability/jspace_paper/gemma4_nonlinear_jacobian_handout.tex"
GEMMA_METADATA = Path(
    "/content/hf_local/models--google--gemma-4-31B-it/snapshots/"
    + GEMMA_REVISION
)
OLMO_DRIVE_SNAPSHOT = Path(
    "/content/drive/MyDrive/hf_cache/models--allenai--Olmo-3-32B-Think/"
    "snapshots/" + OLMO_REVISION
)
SOURCE_REGISTRY = REPO_ROOT / "interpretability/jspace_part2/reports/evidence_events.jsonl"

IMPORTS = {
    "a3-gemma-fullfit-v1": "a3_gemma_fit.py",
    "a3-gemma-identification-v1": "a3_gemma_identification.py",
    "a3-gemma-readout-verdict-v1": "a3_gemma_readout.py",
    "a3-gemma-deepband-logit-v1": "a3_gemma_deepband.py",
    "local-linearity-v3-gemma4-31b": "local_linearity.py",
    "linearization-faithfulness-gemma4-31b-v2": "linearization_faithfulness.py",
    "readout-control-olmo3think-v1": "readout_control.py",
    "local-linearity-v3-olmo3-think": "local_linearity.py",
    "linearization-faithfulness-olmo3-think-v2": "linearization_faithfulness.py",
}


def _snapshot_availability(snapshot: Path, remote: dict) -> dict:
    rows = []
    for expected in remote["files"]:
        if not (
            expected["path"].endswith(".safetensors")
            or expected["path"] in {
                "config.json", "generation_config.json", "model.safetensors.index.json",
                "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "vocab.json", "merges.txt", "special_tokens_map.json",
            }
        ):
            continue
        path = snapshot / expected["path"]
        exists = path.exists()
        size = path.stat().st_size if exists else None
        symlink_target = str(path.readlink()) if path.is_symlink() else None
        content_address_matches = (
            expected["lfs_sha256"] is None
            or (
                symlink_target is not None
                and Path(symlink_target).name == expected["lfs_sha256"]
            )
        )
        rows.append(
            {
                "path": expected["path"],
                "expected_size_bytes": expected["size_bytes"],
                "expected_lfs_sha256": expected["lfs_sha256"],
                "present": exists,
                "actual_size_bytes": size,
                "size_matches": size == expected["size_bytes"],
                "symlink_target": symlink_target,
                "content_address_matches": content_address_matches,
            }
        )
    weights = [row for row in rows if row["path"].endswith(".safetensors")]
    return {
        "snapshot": str(snapshot),
        "files": rows,
        "weight_files_complete_by_presence_and_size": sum(
            row["present"] and row["size_matches"] for row in weights
        ),
        "weight_files_expected": len(weights),
        "ready_for_model_load": all(
            row["present"] and row["size_matches"] and row["content_address_matches"]
            for row in rows
        ),
        "note": "Full content hashes are mandatory after local NVMe staging and before model load.",
    }


def _git_artifact_inventory(commitish: str, relative: str) -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", commitish], text=True
    ).strip()
    blob = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{relative}"]
    )
    blob_id = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", f"{commit}:{relative}"],
        text=True,
    ).strip()
    return {
        "path": relative,
        "source_commit": commit,
        "git_blob_id": blob_id,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
        "materialization": "immutable git object; file need not exist in the side fork worktree",
    }


def _source_code_inventory(imports: list[dict]) -> dict[str, list[dict]]:
    result = {}
    for verified in imports:
        evidence_id = verified["source_evidence_id"]
        filename = IMPORTS[evidence_id]
        relative = f"interpretability/jspace_part2/jspace_part2/experiments/{filename}"
        result[evidence_id] = [
            _git_artifact_inventory(verified["source_commit"], relative)
        ]
    return result


def _jlens_identity() -> dict:
    checkout = Path("/content/jacobian-lens")
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"path": str(checkout), "commit": commit, "dirty": dirty}


def main() -> None:
    git = require_clean_tree()
    if git["code_commit"] == FORK_COMMIT:
        raise RuntimeError("foundation producer must run from a committed Gemma scaffold")
    if not GEMMA_METADATA.exists():
        raise RuntimeError("pinned Gemma metadata snapshot is absent")
    if not OLMO_DRIVE_SNAPSHOT.exists():
        raise RuntimeError("historical OLMo cache snapshot is absent")

    tests = subprocess.run(
        ["python", "-m", "pytest", str(PACKAGE_ROOT / "tests"), "-q"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    gemma_remote = hf_remote_inventory(GEMMA_ID, GEMMA_REVISION)
    olmo_remote = hf_remote_inventory(OLMO_ID, OLMO_REVISION)
    architecture = manifest_from_config(GEMMA_METADATA / "config.json")
    committed_architecture = PACKAGE_ROOT / "configs/gemma4_31b_architecture_manifest.json"
    imports = []
    for evidence_id in IMPORTS:
        verified = verify_source_event(SOURCE_REGISTRY, evidence_id)
        if not verified["ok"]:
            raise RuntimeError(f"historical import failed: {verified}")
        imports.append(verified)
    source_code = _source_code_inventory(imports)

    governing = inventory(
        [
            Path("/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_gemma_1.md"),
            Path("/content/drive/MyDrive/interpret/special-lab-1/jspace_lab_gemma_1_addendum.md"),
            Path("/content/drive/MyDrive/interpret/special-lab-1/gemma4_nonlinear_jacobian_handout.pdf"),
        ]
    )
    governing.append(_git_artifact_inventory(HANDOUT_SOURCE_COMMIT, HANDOUT_SOURCE_PATH))
    model_inventory = {
        "gemma_remote": gemma_remote,
        "gemma_local_metadata": _snapshot_availability(GEMMA_METADATA, gemma_remote),
        "olmo_remote": olmo_remote,
        "olmo_drive_cache": _snapshot_availability(OLMO_DRIVE_SNAPSHOT, olmo_remote),
    }
    env = environment_payload(require_gpu=True)
    package_files = [
        path for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and not any(part.endswith(".egg-info") for part in path.parts)
    ]
    package_inventory = inventory(package_files, base=REPO_ROOT)
    payload = {
        "schema_version": 1,
        "evidence_id": "gm-foundation-v1",
        "tier": "methods",
        "branch": git["branch"],
        "code_commit": git["code_commit"],
        "fork_commit": FORK_COMMIT,
        "run_root": str(run_root()),
        "registry": str(PACKAGE_ROOT / "reports/evidence_events.jsonl"),
        "architecture": architecture,
        "committed_architecture_manifest": {
            "path": str(committed_architecture),
            "sha256": file_sha256(committed_architecture),
        },
        "models": model_inventory,
        "historical_imports": imports,
        "historical_source_code": source_code,
        "historical_registry": {
            "path": str(SOURCE_REGISTRY),
            "sha256": file_sha256(SOURCE_REGISTRY),
        },
        "governing_documents": governing,
        "jacobian_lens": _jlens_identity(),
        "environment": env,
        "tests": {
            "command": f"python -m pytest {PACKAGE_ROOT / 'tests'} -q",
            "stdout": tests.stdout.strip(),
            "stderr": tests.stderr.strip(),
            "passed": True,
        },
        "package_source_inventory": package_inventory,
        "claim_boundary": {
            "allowed_tiers": ["development", "methods"],
            "forbidden": [
                "Phase 4 confirmatory model cell",
                "workspace absence from transport failure",
                "information absence from readout opacity",
                "finite difference labeled exact JVP",
                "Gemma interpretation before OLMo control and committed thresholds",
            ],
        },
    }
    payload["foundation_payload_sha256"] = object_sha256(payload)
    manifests = directory("manifests")
    foundation_path = manifests / "gm_foundation_v1.json"
    models_path = manifests / "model_remote_and_staging_inventory_v1.json"
    imports_path = manifests / "historical_imports_v1.json"
    environment_path = manifests / "gemma_transport_environment_lock_v1.json"
    architecture_path = manifests / "gemma4_31b_architecture_runtime_v1.json"
    atomic_json(foundation_path, payload)
    atomic_json(models_path, model_inventory)
    atomic_json(imports_path, {"imports": imports, "source_code": source_code})
    atomic_json(environment_path, env)
    atomic_json(architecture_path, architecture)

    create(
        "gm-foundation-v1",
        tier="methods",
        what=(
            "isolated Gemma transport package, exact architecture/runtime audit, "
            "model staging inventory, and immutable historical import manifest"
        ),
        command="python -m jspace_gemma.experiments.gm_foundation",
        outputs=[foundation_path, models_path, imports_path, environment_path, architecture_path],
        inputs={
            "fork_commit": FORK_COMMIT,
            "governing_documents": governing,
            "source_registry_sha256": file_sha256(SOURCE_REGISTRY),
        },
    )
    for verified in imports:
        source = verified["source_event"]
        source_id = verified["source_evidence_id"]
        safe_slug = source_id.replace("_", "-")
        import_evidence(
            f"gm-import-{safe_slug}",
            tier="historical-development-import",
            what=f"read-only hash-pinned import of historical {source_id}",
            source_study="jspace-part2",
            source_evidence_id=source_id,
            source_commit=verified["source_commit"],
            source_registry=SOURCE_REGISTRY,
            source_outputs=source.get("outputs", []),
            source_code_files=source_code[source_id],
            import_code_commit=git["code_commit"],
        )
    print(json.dumps({
        "foundation": str(foundation_path),
        "foundation_sha256": file_sha256(foundation_path),
        "imports": len(imports),
        "olmo_drive_weight_files_complete": model_inventory["olmo_drive_cache"]["weight_files_complete_by_presence_and_size"],
        "olmo_drive_weight_files_expected": model_inventory["olmo_drive_cache"]["weight_files_expected"],
        "gemma_weights_present": model_inventory["gemma_local_metadata"]["weight_files_complete_by_presence_and_size"],
    }, indent=1))


if __name__ == "__main__":
    main()
