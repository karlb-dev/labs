"""Build and verify the final Phase 3 release manifest.

The producer refuses any missing or hash-mismatched output from a live
evidence event.  Superseded and withdrawn events remain in the event log but
are not release inputs.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from jspace_part2.lib import sha256_file
from jspace_part2.paths import resolve as resolve_uri
from ..bank import load_bank
from ..paths3 import manifests_dir, run_root
from ..provenance3 import (Provenance3, EVENTS, register, require_clean_tree,
                           resolve, resolve_all, write_result3)
from ..seeds import SEED_CONTRACT
from .p3_protected_answer_audit import canonical_hash, text_hash

EVIDENCE_ID = "p3-release-manifest-v1"
TIER = "methods"
RELEASE_TAG = "jspace-phase3-complete-v1"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = PACKAGE_ROOT / "data"
CONFIG_ROOT = PACKAGE_ROOT / "configs"
SLUGS = ("olmo31-think", "olmo31-instruct", "qwen36-27b")


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def model_ref(uri: str) -> dict:
    if not uri.startswith("model://") or "@" not in uri:
        raise ValueError(f"model URI is not revision-pinned: {uri}")
    ref, revision = uri.removeprefix("model://").rsplit("@", 1)
    return {"hub_id": ref, "revision": revision}


def file_record(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"release artifact is missing: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def prompt_alias_payload() -> dict:
    rows = []
    bank_hashes = {}
    for name in ("bank_f_v7.jsonl", "bank_s_v3.jsonl"):
        path = DATA_ROOT / name
        bank_hashes[name] = sha256_file(path)
        for bundle in load_bank(path):
            aliases_hash = canonical_hash(bundle.accepted_answers)
            for variant, prompt in sorted(bundle.prompts.items()):
                rows.append({
                    "fact_id": bundle.fact_id,
                    "canonical_family": bundle.canonical_family,
                    "bank": bundle.bank,
                    "variant": variant,
                    "prompt_text_sha256": text_hash(prompt),
                    "canonical_answer": bundle.answer,
                    "accepted_aliases": bundle.accepted_answers,
                    "accepted_aliases_sha256": aliases_hash,
                })
    rows.sort(key=lambda row: (row["fact_id"], row["variant"]))
    return {
        "schema_version": 1,
        "contract": (
            "exact prompt text hash plus frozen accepted alias text; "
            "tokenizer-specific token IDs are in protected-answer and "
            "accepted-alias audit artifacts"),
        "banks_sha256": bank_hashes,
        "n_fact_variants": len(rows),
        "n_facts": len({row["fact_id"] for row in rows}),
        "manifest_sha256": canonical_hash(rows),
        "rows": rows,
    }


def environment_payload() -> dict:
    packages = sorted(
        {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }.items())
    jlens_root = Path("/tmp/jacobian-lens")
    jlens_commit = None
    if (jlens_root / ".git").exists():
        jlens_commit = subprocess.check_output(
            ["git", "-C", str(jlens_root), "rev-parse", "HEAD"],
            text=True).strip()
    return {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "jlens_commit": jlens_commit,
        "packages": [
            {"name": name, "version": version}
            for name, version in packages
        ],
    }


def verify_live_registry() -> tuple[list[dict], dict]:
    inventory = []
    failures = []
    events = resolve_all()
    for event in events:
        if not event["live"]:
            continue
        effective = event["effective_metadata"]
        outputs = []
        for output in event.get("outputs", []) or []:
            path = Path(output["path"])
            expected = output.get("sha256")
            exists = path.exists()
            actual = sha256_file(path) if exists else None
            ok = bool(exists and (expected is None or actual == expected))
            row = {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "exists": exists,
                "ok": ok,
            }
            outputs.append(row)
            if not ok:
                failures.append({
                    "evidence_id": event["evidence_id"], **row})
        inventory.append({
            "evidence_id": event["evidence_id"],
            "effective_tier": event["effective_tier"],
            "code_commit": effective.get("code_commit"),
            "outputs": outputs,
            "all_outputs_verified": all(
                output["ok"] for output in outputs),
        })
    if failures:
        raise RuntimeError(
            "live registry verification failed: "
            + json.dumps(failures, sort_keys=True))
    status = {
        "n_created_events": len(events),
        "n_live_events": len(inventory),
        "n_verified_outputs": sum(
            len(event["outputs"]) for event in inventory),
        "n_failures": 0,
        "superseded_ids": sorted(
            event["evidence_id"] for event in events
            if event["superseded_by"] is not None),
        "withdrawn_ids": sorted(
            event["evidence_id"] for event in events
            if event["withdrawn"]),
    }
    return inventory, status


def core_paths(root: Path) -> dict[str, dict[str, Path]]:
    raw = {}
    for slug in SLUGS:
        raw[f"{slug}:confirmatory"] = (
            root / f"metrics/{slug}/p3_grid/p3_grid_{slug}.parquet")
        raw[f"{slug}:replication"] = (
            root / f"metrics/{slug}/p3_grid_replication/"
            f"p3_grid_replication_{slug}.parquet")
    results = {
        "inference": (
            root / "metrics/cross_model/release_audit/"
            "p3_inference_audit.json"),
        "protected_answer": (
            root / "metrics/qwen36-27b/release_audit/protected_answer/"
            "p3_protected_answer_audit.json"),
        "control_seed": (
            root / "metrics/qwen36-27b/release_audit/control_seed/"
            "p3_control_seed_audit_full.json"),
        "n8_phase3_level1": (
            root / "metrics/cross_model/release_audit/"
            "p3_n8_phase3_level1_comparison.json"),
        "n8_phase3_level3_qwen": (
            root / "metrics/cross_model/release_audit/n8_phase3/"
            "n8_p3_l3_qwen36-27b_v1.json"),
        "boundary_cohort": (
            root / "metrics/cross_model/release_audit/"
            "alias_cohort_sensitivity_v2/"
            "p3_alias_cohort_sensitivity.json"),
        "alias_endpoint": (
            root / "metrics/cross_model/release_audit/alias_endpoint/"
            "p3_alias_endpoint_cross_model.json"),
        "bridge_geometry": (
            root / "metrics/qwen36-27b/release_audit/"
            "bridge_geometry_v2/p3_bridge_geometry_qwen36-27b.json"),
        "bridge_swap": (
            root / "metrics/qwen36-27b/release_audit/"
            "bridge_swap_endpoint/p3_bridge_swap_endpoint_qwen36-27b.json"),
        "release_summary_figure": (
            root / "metrics/cross_model/release_audit/"
            "p3_release_summary_figure.json"),
        "publication": (
            root / "reports/p3_state_of_record_release.json"),
    }
    figures = {
        "release_summary": (
            root / "figures/p3f06_phase3_release_audit.png"),
        "boundary_cohort": (
            root / "metrics/cross_model/release_audit/"
            "alias_cohort_sensitivity_v2/"
            "p3_alias_cohort_sensitivity.png"),
        "alias_endpoint": (
            root / "metrics/cross_model/release_audit/alias_endpoint/"
            "p3_alias_endpoint_cross_model.png"),
        "bridge_geometry": (
            root / "metrics/qwen36-27b/release_audit/"
            "bridge_geometry_v2/p3_bridge_geometry_qwen36-27b.png"),
        "bridge_swap": (
            root / "metrics/qwen36-27b/release_audit/"
            "bridge_swap_endpoint/p3_bridge_swap_endpoint_qwen36-27b.png"),
    }
    return {"raw_parquets": raw, "result_envelopes": results,
            "figures": figures}


def model_manifest() -> dict:
    alias_evidence = {
        "olmo31-think": "p3-alias-endpoint-olmo31-think-v1",
        "olmo31-instruct": "p3-alias-endpoint-olmo31-instruct-v1",
        "qwen36-27b": "p3-alias-endpoint-qwen36-27b-v1",
    }
    models = {}
    for slug in SLUGS:
        config_path = CONFIG_ROOT / f"p3_grid_{slug}.yaml"
        config = yaml.safe_load(config_path.read_text())
        lens_path = Path(resolve_uri(config["lens_uri"]))
        event = resolve(alias_evidence[slug])
        models[slug] = {
            **model_ref(config["model_uri"]),
            "config": file_record(config_path),
            "tokenizer_manifest_sha256": event["inputs"][
                "tokenizer_manifest"],
            "lens": file_record(lens_path),
            "band": config["band"],
            "k": int(config["k"]),
            "protect_top_k": int(config["protect_top_k"]),
        }
    return models


def figure_manifest(paths: dict[str, Path]) -> dict:
    producers = {
        "release_summary": (
            PACKAGE_ROOT / "jspace_phase3/experiments/"
            "p3_release_summary_figure.py",
            "p3-release-summary-figure-v1"),
        "boundary_cohort": (
            PACKAGE_ROOT / "jspace_phase3/experiments/"
            "p3_alias_and_cohort_sensitivity.py",
            "p3-boundary-cohort-sensitivity-v2"),
        "alias_endpoint": (
            PACKAGE_ROOT / "jspace_phase3/experiments/"
            "p3_alias_endpoint_audit.py",
            "p3-alias-endpoint-cross-model-v1"),
        "bridge_geometry": (
            PACKAGE_ROOT / "jspace_phase3/experiments/"
            "p3_bridge_geometry_audit.py",
            "p3-bridge-geometry-qwen36-27b-v2"),
        "bridge_swap": (
            PACKAGE_ROOT / "jspace_phase3/experiments/"
            "p3_bridge_swap_endpoint_audit.py",
            "p3-bridge-swap-endpoint-qwen36-27b-v1"),
    }
    output = {}
    for name, path in paths.items():
        producer, evidence_id = producers[name]
        event = resolve(evidence_id)
        output[name] = {
            "artifact": file_record(path),
            "producer": file_record(producer),
            "evidence_id": evidence_id,
            "source_sha256": event.get("inputs", {}),
        }
    return output


def main() -> None:  # noqa: C901
    git_info = require_clean_tree(False)
    root = run_root()
    out_dir = manifests_dir()
    prompt_path = out_dir / "phase3_prompt_alias_manifest.json"
    environment_path = out_dir / "phase3_environment_lock.json"
    inventory_path = out_dir / "phase3_live_evidence_inventory.json"
    manifest_path = out_dir / "phase3_release_manifest.json"
    markdown_path = out_dir / "phase3_release_manifest.md"

    prompt_payload = prompt_alias_payload()
    atomic_json(prompt_path, prompt_payload)
    environment = environment_payload()
    atomic_json(environment_path, environment)
    inventory, registry_status = verify_live_registry()
    inventory_payload = {
        "schema_version": 1,
        "registry_sha256_before_manifest_event": sha256_file(EVENTS),
        "status": registry_status,
        "events": inventory,
    }
    atomic_json(inventory_path, inventory_payload)

    paths = core_paths(root)
    raw = {
        name: file_record(path)
        for name, path in paths["raw_parquets"].items()}
    results = {
        name: file_record(path)
        for name, path in paths["result_envelopes"].items()}
    figures = figure_manifest(paths["figures"])
    reports = {
        "state_of_record_markdown": file_record(
            PACKAGE_ROOT / "reports/PHASE3_STATE_OF_RECORD.md"),
        "living_report": file_record(
            PACKAGE_ROOT / "reports/REPORT_PHASE3.md"),
        "final_tex": file_record(
            PACKAGE_ROOT / "reports/handout/jspace_phase3_final.tex"),
        "final_pdf": file_record(
            PACKAGE_ROOT / "reports/handout/jspace_phase3_final.pdf"),
    }
    partition = (
        PACKAGE_ROOT / "preregistration/partition_phase3.json")
    selection = (
        PACKAGE_ROOT / "preregistration/"
        "p3_alias_sensitivity_selection_v1.json")
    tag_present = bool(subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "-q", "--verify",
         f"refs/tags/{RELEASE_TAG}"],
        capture_output=True).returncode == 0)
    manifest = {
        "schema_version": 1,
        "release": {
            "study_id": "jspace-phase3",
            "branch": git_info["branch"],
            "producer_code_commit": git_info["code_commit"],
            "release_tag": RELEASE_TAG,
            "tag_present_at_generation": tag_present,
            "tag_contract": (
                "Create and push this immutable tag only after this "
                "manifest event is committed and final verification passes."),
            "freeze_tag": "jspace-phase3-freeze-v1",
            "freeze_commit": "df4d45a",
            "pre_release_audit_tag":
                "jspace-phase3-pre-release-audit-v1",
            "pre_release_audit_commit": "660047d",
        },
        "models": model_manifest(),
        "banks": {
            name: file_record(DATA_ROOT / name)
            for name in ("bank_f_v7.jsonl", "bank_s_v3.jsonl")
        },
        "partition": file_record(partition),
        "prompt_alias_manifest": file_record(prompt_path),
        "alias_sensitivity_selection": file_record(selection),
        "environment_lock": file_record(environment_path),
        "seed_contract": {
            "algorithm": SEED_CONTRACT,
            "state_of_record_qwen_control_seed": 31337,
            "control_audit_seeds": [11, 101, 1009, 4242, 31337],
            "analysis_seed": 4242,
            "frozen_partition_seed": 85670,
            "historical_limitation": (
                "Frozen matched controls used Python hash(item_id) without "
                "a recorded PYTHONHASHSEED."),
        },
        "raw_parquets": raw,
        "result_envelopes": results,
        "figures": figures,
        "reports": reports,
        "registry_verification": {
            **registry_status,
            "registry_sha256_before_manifest_event": sha256_file(EVENTS),
            "inventory": file_record(inventory_path),
            "scope_note": (
                "The inventory necessarily precedes creation of the "
                "manifest's own evidence event; verify that event after "
                "registration before tagging."),
        },
    }
    command = (
        "python -m "
        "jspace_phase3.experiments.p3_release_manifest")
    inputs = {
        "registry_before_manifest_event": sha256_file(EVENTS),
        "state_of_record": reports[
            "state_of_record_markdown"]["sha256"],
        "final_pdf": reports["final_pdf"]["sha256"],
        "prompt_alias_manifest": sha256_file(prompt_path),
        "environment_lock": sha256_file(environment_path),
    }
    write_result3(
        manifest, manifest_path,
        Provenance3(
            evidence_id=EVIDENCE_ID, tier=TIER,
            command=command, inputs=inputs))
    lines = [
        "# Phase 3 release manifest",
        "",
        f"- Producer commit: `{git_info['code_commit']}`.",
        f"- Planned immutable tag: `{RELEASE_TAG}`.",
        f"- Live evidence events verified: "
        f"{registry_status['n_live_events']}.",
        f"- Registered outputs hash-verified: "
        f"{registry_status['n_verified_outputs']}.",
        "- Missing or mismatched live outputs: **0**.",
        f"- Prompt/alias fact-variants: "
        f"{prompt_payload['n_fact_variants']}.",
        f"- Environment packages pinned: "
        f"{len(environment['packages'])}.",
        "",
        "The JSON envelope contains model revisions, tokenizer and lens "
        "hashes, banks, partition, seed contract, six raw outcome parquet "
        "hashes, result envelopes, figure producers/sources, reports, and "
        "the complete verified live-evidence inventory.",
        "",
    ]
    atomic_text(markdown_path, "\n".join(lines))
    register(
        EVIDENCE_ID,
        tier=TIER,
        command=command,
        what=(
            "Final Phase 3 release manifest with zero-failure verification "
            "of every live registered output plus model, tokenizer, lens, "
            "bank, partition, environment, seed, raw-data, result, figure, "
            "and report hash pins."),
        outputs=[
            manifest_path, markdown_path, prompt_path,
            environment_path, inventory_path,
        ],
        inputs=inputs)
    print(json.dumps({
        "manifest": str(manifest_path),
        "live_events": registry_status["n_live_events"],
        "verified_outputs": registry_status["n_verified_outputs"],
        "failures": 0,
    }, indent=1))


if __name__ == "__main__":
    main()
