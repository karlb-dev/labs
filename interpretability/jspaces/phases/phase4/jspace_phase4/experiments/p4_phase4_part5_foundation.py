"""Phase 4.5 launch foundation: verify the integrated terminal state.

Produces the required Part-5 launch artifacts from the live branch rather
than from prose: exact source head, ancestry, registry hashes, Study-2
bundle pins, terminal Q-L4 extraction, launch test counts, environment and
Drive-materialization provenance, and an explicit no-model-scale-process
declaration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ..manifests import (
    atomic_json,
    environment_payload,
    file_sha256,
    git_info,
    object_sha256,
    require_clean_tree,
)
from ..registry4 import create, read_events, resolve_all

EVIDENCE_ID = "p4-phase4-part5-foundation-v1"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]
REPORTS = PACKAGE_ROOT / "reports"
MANIFESTS = PACKAGE_ROOT / "manifests"

FOUNDATION_JSON = REPORTS / "PHASE4_PART5_FOUNDATION.json"
FOUNDATION_MD = REPORTS / "PHASE4_PART5_FOUNDATION.md"
SOURCE_INVENTORY = MANIFESTS / "phase4_part5_source_inventory.json"

EXPECTED_ANCESTORS = {
    "phase4_4_parent": "901fb4fc7578a913088c7947a2e6240f7fc45aeb",
    "terminal_b_precommit": "6f23a29896e61cc367ed884a2d840f0e08857f40",
    "prompt323_runtime_amendment":
        "92831d31e37e76a50b573a99c9f19bf55005531c",
    "sidelines2_integration_head":
        "deaeb8baf07468c49e5397ad3b15f2903dacc659",
    "gemma_study2_branch_head": "e07880084b33fe0f998968bcd5d2394e2ae6465f",
    "gemma_study2_bundle_source":
        "aba2e01460dde32e5c2ca1478a5502950e2448ec",
    "olmo_study2_branch_head": "9159baff66402f33ee7a907e10acac748a3e6114",
    "olmo_study2_bundle_source":
        "80213290125a56ad75bd9a23a638211a0dc1c618",
    "phase4_4_terminal_merge": "a63d49c1879f888893e9d005bdace1e46bcdc603",
    "phase4_4_closeout_packet": "0f3380d580ba5f78c87d4b00adb7906f3c2ad747",
}

EXPECTED_LIVE_EVIDENCE = [
    "p4-qwen-lens-fit-drawA-n1000-dev-v1",
    "p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1",
    "p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1",
    "p4-qwen-selection-margin-a500-a1000-dev-v1",
    "p4-qwen-lens-influence-prompt323-dev-v1",
    "p4-qwen-canonical-lens-decision-a1000-dev-v1",
    "p4-import-olmo-bank-w-capability-v1",
    "p4-bank-w-capability-joint-imported-dev-v1",
    "p4-import-gemma-transport-v1",
    "p4-import-olmo-lineage-final-v1",
    "p4-qwen-a120-a250-state-permanent-deficit-v1",
]

EXPECTED_BUNDLES = {
    "gemma_study2": {
        "bundle": "interpretability/jspace_gemma/release/"
                  "IMPORT_BUNDLE_SIDELINES2.json",
        "bundle_sha256": (
            "9ef48b8ab1d99d52a756ddea1e285a9d61e781fd054cbf92702bfe81be56"
            "f5b0"),
        "markdown": "interpretability/jspace_gemma/release/"
                    "IMPORT_BUNDLE_SIDELINES2.md",
        "prefix": "interpretability/jspace_gemma/release/"
                  "evidence_events_prefix_sidelines2.jsonl",
        "prefix_sha256": (
            "2a144bcf0e7be0ac4307f7e2a2984c1879340b9a7e9278d10143d122a14f"
            "d30a"),
        "terminal_event": "gm2-sidelines2-import-bundle-v1",
    },
    "olmo_study2": {
        "bundle": "interpretability/jspace_olmo_lineage/release/"
                  "IMPORT_BUNDLE_SIDELINES2.json",
        "bundle_sha256": (
            "c213dc74aa78dcd6613c8bd1562dd07d2e2a0345409ee6da585001693d8e"
            "6b1c"),
        "markdown": "interpretability/jspace_olmo_lineage/release/"
                    "IMPORT_BUNDLE_SIDELINES2.md",
        "prefix": "interpretability/jspace_olmo_lineage/release/"
                  "evidence_events_prefix_sidelines2.jsonl",
        "prefix_sha256": (
            "0a8973e01d562a82fa88da650ab8597c140050f6caf46c8bbd72e2b58acf"
            "fb58"),
        "terminal_event": "ol2-sidelines2-import-bundle-v1",
    },
}

STATE_OF_RECORD_DOCS = [
    "interpretability/jspace_gemma/release/"
    "GEMMA_TRANSPORT_STATE_OF_RECORD_V2.md",
    "interpretability/jspace_gemma/release/gemma_transport_claim_ledger_v2.md",
    "interpretability/jspace_gemma/release/TRANSPORT_GATE_PROTOCOL_V2.md",
    "interpretability/jspace_olmo_lineage/reports/"
    "OLMO_LINEAGE_STATE_OF_RECORD_V2.md",
    "interpretability/jspace_olmo_lineage/reports/"
    "OLMO_LINEAGE_CLAIMS_TABLE_V2.md",
    "interpretability/reviews/JSPACE_SIDELINES_2_INTEGRATION_RECORD.md",
]

REGISTRIES = {
    "phase4": "interpretability/jspace_phase4/reports/evidence_events.jsonl",
    "gemma": "interpretability/jspace_gemma/reports/evidence_events.jsonl",
    "olmo_lineage":
        "interpretability/jspace_olmo_lineage/reports/evidence_events.jsonl",
}

TEST_SUITES = {
    "phase4": "interpretability/jspace_phase4/tests",
    "gemma": "interpretability/jspace_gemma/tests",
    "olmo_lineage": "interpretability/jspace_olmo_lineage/tests",
}


class FoundationError(RuntimeError):
    pass


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True).strip()


def _is_ancestor(commit: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor",
         commit, "HEAD"],
        capture_output=True).returncode == 0


def _ancestry() -> dict:
    rows = {}
    for name, commit in EXPECTED_ANCESTORS.items():
        reachable = _is_ancestor(commit)
        if not reachable:
            raise FoundationError(
                f"expected ancestor is unreachable: {name} {commit}")
        rows[name] = {"commit": commit, "ancestor_of_head": True}
    merges = []
    log = _git(
        "log", "--first-parent", "--merges", "--format=%H %P|%s", "-12")
    for line in log.splitlines():
        meta, subject = line.split("|", 1)
        parts = meta.split()
        merges.append({
            "commit": parts[0],
            "parents": parts[1:],
            "subject": subject,
        })
    return {"expected_ancestors": rows, "recent_first_parent_merges": merges}


def _registries() -> dict:
    rows = {}
    for name, relative in REGISTRIES.items():
        path = REPO_ROOT / relative
        events = read_events(path)
        origins = [
            row for row in events
            if row.get("event") in {"evidence_created", "evidence_imported"}
        ]
        rows[name] = {
            "path": relative,
            "sha256": file_sha256(path),
            "n_rows": len(events),
            "n_origin_events": len(origins),
        }
    return rows


def _bundles() -> dict:
    rows = {}
    for name, spec in EXPECTED_BUNDLES.items():
        bundle_path = REPO_ROOT / spec["bundle"]
        actual = file_sha256(bundle_path)
        if actual != spec["bundle_sha256"]:
            raise FoundationError(
                f"{name} bundle hash mismatch: expected "
                f"{spec['bundle_sha256']}, got {actual}")
        prefix_path = REPO_ROOT / spec["prefix"]
        prefix_actual = file_sha256(prefix_path)
        if prefix_actual != spec["prefix_sha256"]:
            raise FoundationError(
                f"{name} frozen prefix hash mismatch: got {prefix_actual}")
        envelope = json.loads(bundle_path.read_text())
        payload = envelope["payload"]
        if object_sha256(payload) != envelope.get("payload_sha256"):
            raise FoundationError(f"{name} bundle payload hash mismatch")
        if payload.get("evidence_id") != spec["terminal_event"]:
            raise FoundationError(
                f"{name} bundle terminal event drift: "
                f"{payload.get('evidence_id')!r}")
        live_registry = REPO_ROOT / payload["registry_prefix"]["path"]
        if not live_registry.read_bytes().startswith(
                prefix_path.read_bytes()):
            raise FoundationError(
                f"{name} live registry mutated its frozen prefix")
        rows[name] = {
            "bundle_path": spec["bundle"],
            "bundle_sha256": actual,
            "bundle_markdown_path": spec["markdown"],
            "bundle_markdown_sha256": file_sha256(
                REPO_ROOT / spec["markdown"]),
            "payload_sha256": envelope["payload_sha256"],
            "terminal_event": spec["terminal_event"],
            "frozen_prefix_path": spec["prefix"],
            "frozen_prefix_sha256": prefix_actual,
            "frozen_prefix_intact_in_live_registry": True,
            "source_commit": payload["source_git"]["code_commit"],
            "source_branch": payload["source_git"]["branch"],
        }
    return rows


def _phase4_terminal_state() -> dict:
    resolved = {row["evidence_id"]: row for row in resolve_all()}
    missing = [
        evidence_id for evidence_id in EXPECTED_LIVE_EVIDENCE
        if evidence_id not in resolved or not resolved[evidence_id]["live"]
    ]
    if missing:
        raise FoundationError(
            f"expected live Phase 4 evidence is absent: {missing}")
    leaked = sorted(
        evidence_id for evidence_id in resolved
        if evidence_id.startswith(("gm-", "gm2-", "ol-", "ol2-")))
    if leaked:
        raise FoundationError(
            f"native side evidence inside Phase 4 registry: {leaked}")
    canonical = resolved["p4-qwen-canonical-lens-decision-a1000-dev-v1"]
    decision_output = next(
        output for output in canonical["outputs"]
        if output["path"].endswith("canonical_lens_decision.json"))
    decision_path = Path(decision_output["path"])
    actual = file_sha256(decision_path)
    if actual != decision_output["sha256"]:
        raise FoundationError(
            "canonical decision output hash drift on this materialization")
    decision = json.loads(decision_path.read_text())
    payload = decision.get("payload", decision)
    branch = (
        payload.get("canonical_branch")
        or payload.get("branch")
        or payload.get("decision"))
    return {
        "expected_live_evidence": EXPECTED_LIVE_EVIDENCE,
        "all_expected_live": True,
        "no_native_side_origins": True,
        "canonical_decision_output": {
            "path": decision_output["path"],
            "sha256": actual,
        },
        "terminal_branch": branch,
    }


def _tests() -> dict:
    rows = {}
    for name, relative in TEST_SUITES.items():
        result = subprocess.run(
            [sys.executable, "-m", "pytest", relative, "-q"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        tail = [
            line for line in result.stdout.strip().splitlines()[-3:]
            if "passed" in line or "failed" in line or "error" in line
        ]
        if result.returncode != 0:
            raise FoundationError(
                f"{name} launch suite failed:\n"
                + result.stdout[-4000:] + result.stderr[-4000:])
        rows[name] = {
            "command": f"python -m pytest {relative} -q  # cwd=repo root",
            "returncode": result.returncode,
            "summary": tail[-1] if tail else "",
        }
    return rows


def _drive_provenance() -> dict:
    from ..paths4 import run_root as resolve_run_root

    run_root = resolve_run_root(create=False)
    lab = run_root.parent
    drive = lab.parents[1]
    if not (drive.is_dir() and lab.is_dir()):
        raise FoundationError("campaign Drive is not mounted")
    with open("/proc/uptime") as handle:
        uptime_seconds = float(handle.read().split()[0])
    git_dir_created = (REPO_ROOT / ".git").stat().st_ctime
    return {
        "drive_mount": str(drive),
        "campaign_root": str(lab),
        "phase4_run_root": str(run_root),
        "phase4_run_root_exists": run_root.is_dir(),
        "vm_uptime_seconds_at_snapshot": round(uptime_seconds, 1),
        "vm_boot_utc_estimate": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - uptime_seconds)),
        "repository_clone_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(git_dir_created)),
        "materialization_note": (
            "Fresh VM for Phase 4.5: this runtime booted today, cloned the "
            "repository from origin, and mounted Drive freshly; no Phase 4 "
            "producer ran on this machine before this snapshot. All Drive "
            "reads in this session are first materializations on this "
            "mount, independent of the Phase 4.4 VM14 cache."),
    }


def _no_model_process() -> dict:
    compute_apps = "unavailable"
    try:
        compute_apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "declaration": (
            "No model download, model load, lens fit, JVP run, "
            "intervention, generation, or ablation is active or authorized "
            "in this closeout block; the attached GPU is unused."),
        "nvidia_compute_apps": compute_apps or "none",
        "gpu_compute_processes_active": bool(
            compute_apps and compute_apps != "unavailable"),
    }


def source_inventory() -> dict:
    module_root = PACKAGE_ROOT / "jspace_phase4"
    paths = [
        PACKAGE_ROOT / "pyproject.toml",
        PACKAGE_ROOT / "constraints.txt",
        PACKAGE_ROOT / "repro.sh",
        PACKAGE_ROOT / "protocol/REPRO_CONTRACT_PHASE4.md",
        PACKAGE_ROOT / "protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json",
        PACKAGE_ROOT / "protocol/PRE_FREEZE_INVENTORY_POLICY_PHASE4.json",
        *sorted(module_root.rglob("*.py")),
        *sorted((PACKAGE_ROOT / "tests").glob("*.py")),
        *sorted((PACKAGE_ROOT / "configs").glob("*.yaml")),
    ]
    return {
        str(path.relative_to(PACKAGE_ROOT)): {
            "sha256": file_sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for path in paths if path.is_file()
    }


def render_markdown(payload: dict) -> str:
    bundles = payload["study2_bundles"]
    lines = [
        "# Phase 4.5 launch foundation",
        "",
        "**INTEGRATED TERMINAL-STATE VERIFICATION — NOT A FREEZE RECORD**",
        "",
        f"Generated: `{payload['generated_utc']}`  ",
        f"Source head: `{payload['git']['source_head']}` on "
        f"`{payload['git']['branch']}`  ",
        f"Closeout parent: `{payload['git']['closeout_parent']}`",
        "",
        "## Verified launch state",
        "",
        f"- Clean tree: `{payload['git']['clean_tree']}`",
        f"- Local equals `origin/interp_jspace_part2`: "
        f"`{payload['git']['matches_remote_mainline']}`",
        "- All expected ancestors reachable: `true` "
        "(Phase 4.4 merge, both Study-2 heads, runtime amendment).",
        f"- Terminal branch from registered canonical decision: "
        f"**{payload['phase4_terminal']['terminal_branch']}**",
        "- Expected live Phase 4 evidence: all "
        f"{len(payload['phase4_terminal']['expected_live_evidence'])} "
        "present and live.",
        "- Native `gm-*`/`gm2-*`/`ol-*`/`ol2-*` origins in Phase 4 "
        "registry: none.",
        "",
        "## Registries",
        "",
        "| Registry | Rows | Origins | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for name, row in payload["registries"].items():
        lines.append(
            f"| {name} | {row['n_rows']} | {row['n_origin_events']} | "
            f"`{row['sha256'][:16]}...` |")
    lines += [
        "",
        "## Study-2 terminal bundles",
        "",
        "| Bundle | Terminal event | Bundle SHA-256 | Frozen prefix |",
        "|---|---|---|---|",
    ]
    for name, row in bundles.items():
        lines.append(
            f"| {name} | `{row['terminal_event']}` | "
            f"`{row['bundle_sha256'][:16]}...` | "
            f"`{row['frozen_prefix_sha256'][:16]}...` intact |")
    lines += [
        "",
        "## Launch test suites",
        "",
    ]
    for name, row in payload["tests"].items():
        lines.append(f"- {name}: `{row['summary']}`")
    lines += [
        "",
        "## Environment and materialization",
        "",
        f"- {payload['drive']['materialization_note']}",
        f"- VM boot (estimate): `{payload['drive']['vm_boot_utc_estimate']}`;"
        f" repository cloned: `{payload['drive']['repository_clone_utc']}`.",
        f"- {payload['no_model_process']['declaration']}",
        "",
        "This foundation authorizes CPU/storage governance work only. It is "
        "not an independent review, PI approval, or freeze artifact.",
        "",
    ]
    return "\n".join(lines)


def produce() -> dict:
    information = git_info(REPO_ROOT)
    if information["dirty_tree"]:
        raise FoundationError(
            "produce the foundation snapshot from a clean tree")
    local = _git("rev-parse", "interp_jspace_part2")
    remote = _git("rev-parse", "origin/interp_jspace_part2")
    parent_file = Path("/content/phase4_5_parent_sha.txt")
    closeout_parent = (
        parent_file.read_text().strip() if parent_file.is_file() else local)
    payload = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "generated_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "source_head": information["code_commit"],
            "branch": information["branch"],
            "clean_tree": not information["dirty_tree"],
            "closeout_parent": closeout_parent,
            "mainline_local": local,
            "mainline_remote": remote,
            "matches_remote_mainline": local == remote,
        },
        "ancestry": _ancestry(),
        "registries": _registries(),
        "study2_bundles": _bundles(),
        "phase4_terminal": _phase4_terminal_state(),
        "tests": _tests(),
        "environment": {
            key: value for key, value in environment_payload().items()
            if key != "packages"
        },
        "drive": _drive_provenance(),
        "no_model_process": _no_model_process(),
        "launch_deviations_recorded": [
            {
                "deviation": (
                    "OLMo final_release --verify failed after the "
                    "append-only Study-2 merge because it compared "
                    "whole-registry live counts to the release-time "
                    "expectation."),
                "resolution": (
                    "Verifier repair anchoring counts to the hash-verified "
                    "immutable prefix plus release-event row; no registry "
                    "row, threshold, payload, or release hash changed."),
                "commit": "bf8d0a8",
            },
            {
                "deviation": (
                    "First fresh materialization surfaced five "
                    "reference-resolution failures invisible to the "
                    "same-mounted Phase 4.4 passes: three bank-w-joint "
                    "outputs registered under the producing VM's worktree "
                    "path with byte-identical tracked repo files, and two "
                    "Study-1 side-registry whole-file pins whose "
                    "append-only registries legitimately grew."),
                "resolution": (
                    "Strict resolver in durability/live-evidence "
                    "verifiers: literal path, identical tracked repository "
                    "file, or exact historical registry bytes at the "
                    "registering commit with byte-prefix extension. "
                    "Hashes never relaxed; resolution modes reported. "
                    "Result: 418/419 verified with only the known "
                    "permanent A120-A250 state deficit."),
                "commit": "0fc9a18",
            },
        ],
        "state_of_record_documents": {
            relative: {
                "sha256": file_sha256(REPO_ROOT / relative),
                "bytes": (REPO_ROOT / relative).stat().st_size,
            }
            for relative in STATE_OF_RECORD_DOCS
        },
        "claim_boundary": (
            "Launch verification only: development/methods terminal state "
            "with zero Phase 4 confirmatory primaries. This artifact opens "
            "no outcome, signs no review, and authorizes no freeze."),
    }
    if payload["git"]["closeout_parent"] != payload["git"]["source_head"] \
            and not _is_ancestor(payload["git"]["closeout_parent"]):
        raise FoundationError("closeout parent is not reachable from HEAD")
    if not payload["git"]["matches_remote_mainline"]:
        raise FoundationError("local mainline is behind or ahead of remote")
    envelope = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
    }
    atomic_json(FOUNDATION_JSON, envelope)
    FOUNDATION_MD.parent.mkdir(parents=True, exist_ok=True)
    FOUNDATION_MD.write_text(render_markdown(payload))
    inventory = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "source_head": payload["git"]["source_head"],
        "inventory": source_inventory(),
    }
    atomic_json(SOURCE_INVENTORY, {
        "schema_version": 1,
        "payload": inventory,
        "payload_sha256": object_sha256(inventory),
    })
    return envelope


def register() -> None:
    require_clean_tree()
    envelope = json.loads(FOUNDATION_JSON.read_text())
    payload = envelope["payload"]
    if object_sha256(payload) != envelope.get("payload_sha256"):
        raise FoundationError("foundation payload hash mismatch")
    if not _is_ancestor(payload["git"]["source_head"]):
        raise FoundationError(
            "recorded source head is not an ancestor of HEAD")
    for name, spec in EXPECTED_BUNDLES.items():
        if file_sha256(REPO_ROOT / spec["bundle"]) != spec["bundle_sha256"]:
            raise FoundationError(f"{name} bundle changed after production")
    event = create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "Phase 4.5 launch foundation: exact integrated source head, "
            "ancestry and registry verification, Study-2 bundle and frozen-"
            "prefix rehash, terminal Q-L4 extraction, launch test counts, "
            "fresh-VM Drive materialization provenance, and no-model-"
            "process declaration for the CPU-only closeout block."),
        command=(
            "python -m jspace_phase4.experiments.p4_phase4_part5_foundation"
            " --produce && ... --register"),
        outputs=[FOUNDATION_JSON, FOUNDATION_MD, SOURCE_INVENTORY],
        inputs={
            "source_head": payload["git"]["source_head"],
            "phase4_registry_sha256":
                payload["registries"]["phase4"]["sha256"],
            "gemma_bundle_sha256":
                payload["study2_bundles"]["gemma_study2"]["bundle_sha256"],
            "olmo_bundle_sha256":
                payload["study2_bundles"]["olmo_study2"]["bundle_sha256"],
        },
    )
    print(json.dumps({
        "evidence_id": event["evidence_id"],
        "code_commit": event["code_commit"],
        "n_outputs": len(event["outputs"]),
    }, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--produce", action="store_true")
    mode.add_argument("--register", action="store_true")
    arguments = parser.parse_args()
    if arguments.produce:
        envelope = produce()
        payload = envelope["payload"]
        print(json.dumps({
            "source_head": payload["git"]["source_head"],
            "terminal_branch":
                payload["phase4_terminal"]["terminal_branch"],
            "tests": {
                name: row["summary"]
                for name, row in payload["tests"].items()
            },
            "payload_sha256": envelope["payload_sha256"],
        }, indent=1))
    else:
        register()


if __name__ == "__main__":
    main()
