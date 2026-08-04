"""Produce the Phase 4 freeze verification record and release manifest.

Runs only inside the Phase 4.5 closeout after the external gates are
signed. Performs no model-scale work: suites, verifiers, hashing, and
manifest assembly only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ..durability import load_known_deficits, verify_registry_durability
from ..manifests import atomic_json, file_sha256, git_info, object_sha256
from ..registry4 import EVENTS, read_events, resolve_all

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parents[1]
RELEASE = PACKAGE_ROOT / "release"
REPORTS = PACKAGE_ROOT / "reports"

VERIFICATION_JSON = REPORTS / "PHASE4_FREEZE_VERIFICATION.json"
VERIFICATION_MD = REPORTS / "PHASE4_FREEZE_VERIFICATION.md"
POST_MERGE_JSON = REPORTS / "PHASE4_POST_MERGE_VERIFICATION.json"
POST_MERGE_MD = REPORTS / "PHASE4_POST_MERGE_VERIFICATION.md"
MANIFEST_JSON = RELEASE / "PHASE4_RELEASE_MANIFEST.json"
MANIFEST_MD = RELEASE / "PHASE4_RELEASE_MANIFEST.md"

SUITES = {
    "phase4": "interpretability/jspace_phase4/tests",
    "gemma": "interpretability/jspace_gemma/tests",
    "olmo_lineage": "interpretability/jspace_olmo_lineage/tests",
}
KNOWN_DEFICIT_SHA = (
    "361bda08e9ffbe1d333fd3cfaf3c7b9545e6a3504246a16dd8b0c07ad26f45e8")

RELEASE_DOCUMENTS = {
    "state_of_record": "reports/PHASE4_STATE_OF_RECORD.md",
    "claim_ledger": "release/PHASE4_CLAIM_LEDGER.md",
    "known_limitations": "release/PHASE4_KNOWN_LIMITATIONS.md",
    "reproduction_guide": "release/PHASE4_REPRODUCTION_GUIDE.md",
}
GOVERNANCE_DOCUMENTS = {
    "independent_review":
        "reviews/PHASE4_INDEPENDENT_REVIEW_20260804.md",
    "untouched_data_audit": "reviews/PHASE4_UNTOUCHED_DATA_AUDIT.md",
    "pi_disposition": "reviews/PHASE4_PI_DISPOSITION_20260804.md",
    "permanent_deficit_packet":
        "reviews/PHASE4_PERMANENT_DEFICIT_REVIEW_PACKET.md",
}
PAPER_SOURCES = {
    "conclusion_skeleton": "paper/PAPER_CONCLUSION_SKELETON.md",
    "methods_decision_record": "paper/PHASE4_METHODS_DECISION_RECORD.md",
    "runtime_identity_synthesis":
        "reports/PHASE4_RUNTIME_IDENTITY_SYNTHESIS.md",
}
FORBIDDEN_CLAIMS = [
    "a canonical sparse Qwen A1000 lens exists",
    "Qwen has no J-space",
    "Gemma is nondifferentiable or lacks a workspace",
    "SFT/DPO show no effect (effects are missing, not zero)",
    "OLMo transport fails (H6 bounds the lens-as-predictor only)",
    "Bank W is negative (planning closure only)",
    "Phase 4 found no effect / Phase 4 was confirmatory",
    "the historical A120-A250 state was recovered",
    "all artifacts verify (one permanent deficit is accepted)",
    "Phase 5 is approved",
]


class FreezeError(RuntimeError):
    pass


def _run(command: list[str], *, cwd: Path = REPO_ROOT) -> tuple[int, str]:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True)
    return result.returncode, (result.stdout + result.stderr)


def _suite_results() -> dict:
    rows = {}
    for name, relative in SUITES.items():
        code, output = _run(
            [sys.executable, "-m", "pytest", relative, "-q"])
        summary = [
            line for line in output.strip().splitlines()
            if "passed" in line or "failed" in line
        ]
        if code != 0:
            raise FreezeError(f"{name} suite failed:\n{output[-3000:]}")
        rows[name] = {"returncode": code, "summary": summary[-1]}
    return rows


def _verifier_results() -> dict:
    rows = {}
    for name, module in (
            ("gemma", "jspace_gemma"),
            ("olmo_lineage", "jspace_olmo_lineage")):
        code, output = _run([sys.executable, "-m", module, "verify"])
        if code != 0:
            raise FreezeError(f"{name} verifier failed:\n{output[-3000:]}")
        payload = json.loads(output[output.index("{"):])
        live = payload.get("live_evidence", payload)
        rows[name] = {
            "ok": live["ok"],
            "n_live_events": live["n_live_events"],
            "n_checked_outputs": live["n_checked_outputs"],
        }
    code, output = _run([sys.executable, "-m", "jspace_phase4", "verify"])
    payload = json.loads(output[output.index("{"):])
    failures = payload.get("failures", [])
    if code == 0 or len(failures) != 1 \
            or failures[0].get("expected") != KNOWN_DEFICIT_SHA:
        raise FreezeError(
            "phase4 verifier must fail with exactly the known permanent "
            f"deficit; got returncode {code} and failures {failures}")
    rows["phase4"] = {
        "ok": False,
        "raw_all_live_outputs_verified": False,
        "only_failure_is_known_deficit": True,
        "n_live_events": payload["n_live_events"],
        "n_checked_outputs": payload["n_checked_outputs"],
        "n_verified_by_resolution": payload.get(
            "n_verified_by_resolution", {}),
    }
    return rows


def _durability() -> dict:
    deficits = load_known_deficits(
        PACKAGE_ROOT / "protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json")
    result = verify_registry_durability(
        known_deficits=deficits, pass_label="phase4-freeze-verification")
    if result["n_unexpected_failures"] or result["path_pin_conflicts"]:
        raise FreezeError("durability found unexpected discrepancies")
    if not result["only_known_deficits"]:
        raise FreezeError("durability must show only the known deficit")
    return {
        key: result[key] for key in (
            "pass_label", "registry_sha256", "n_live_events",
            "n_output_references", "n_verified", "n_verified_by_resolution",
            "n_failures", "n_known_deficits", "n_unexpected_failures")
    }


def _repository_checks() -> dict:
    code, output = _run(["git", "diff", "--check"])
    if code != 0:
        raise FreezeError(f"git diff --check failed:\n{output}")
    code, status = _run(["git", "status", "--porcelain"])
    untracked_large = []
    for line in status.splitlines():
        if not line.startswith("??"):
            continue
        path = REPO_ROOT / line[3:].strip()
        if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
            untracked_large.append(str(path))
    if untracked_large:
        raise FreezeError(
            f"large untracked artifacts present: {untracked_large}")
    information = git_info(REPO_ROOT)
    return {
        "git_diff_check": "clean",
        "large_untracked_artifacts": [],
        "branch": information["branch"],
        "head": information["code_commit"],
        "dirty_tree": information["dirty_tree"],
    }


def _document_rows(documents: dict, *, require: bool = True) -> dict:
    rows = {}
    for name, relative in documents.items():
        path = PACKAGE_ROOT / relative
        if not path.is_file():
            if require:
                raise FreezeError(f"required release document absent: "
                                  f"{relative}")
            continue
        rows[name] = {
            "path": f"interpretability/jspace_phase4/{relative}",
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return rows


def verify_stage(*, post_merge: bool = False) -> dict:
    json_target = POST_MERGE_JSON if post_merge else VERIFICATION_JSON
    md_target = POST_MERGE_MD if post_merge else VERIFICATION_MD
    payload = {
        "schema_version": 1,
        "stage": "post-merge" if post_merge else "pre-merge",
        "generated_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository": _repository_checks(),
        "suites": _suite_results(),
        "live_evidence_verifiers": _verifier_results(),
        "durability": _durability(),
        "handout_policy": (
            "Per the accepted Phase 4.5 addendum, the governed "
            "pre-canonical handout boundary lifts at the tag; handout "
            "regeneration belongs to the paper-analysis phase. Registered "
            "TeX/PDF/figure bytes are hash-verified by the durability "
            "pass rather than rebuilt here."),
        "claim_boundary": (
            "Mechanical freeze verification only; approval lives in the "
            "signed review and PI records."),
    }
    envelope = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
    }
    atomic_json(json_target, envelope)
    lines = [
        "# Phase 4 freeze verification"
        + (" (post-merge)" if post_merge else ""),
        "",
        f"Generated: `{payload['generated_utc']}` at "
        f"`{payload['repository']['head']}` on "
        f"`{payload['repository']['branch']}`.",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, row in payload["suites"].items():
        lines.append(f"| {name} suite | `{row['summary']}` |")
    for name, row in payload["live_evidence_verifiers"].items():
        if name == "phase4":
            lines.append(
                "| phase4 live evidence | 520/521; only failure is the "
                "known permanent deficit |")
        else:
            lines.append(
                f"| {name} live evidence | ok "
                f"({row['n_live_events']} events / "
                f"{row['n_checked_outputs']} outputs) |")
    d = payload["durability"]
    lines += [
        f"| durability | {d['n_verified']}/{d['n_output_references']} "
        f"verified; {d['n_known_deficits']} known deficit; "
        f"0 unexpected |",
        "| git diff --check | clean |",
        "| large untracked artifacts | none |",
        "",
        payload["handout_policy"],
        "",
    ]
    md_target.write_text("\n".join(lines))
    return envelope


def manifest_stage() -> dict:
    information = git_info(REPO_ROOT)
    if information["dirty_tree"]:
        raise FreezeError("manifest requires a clean tree")
    verification = json.loads(VERIFICATION_JSON.read_text())
    events = read_events(EVENTS)
    resolved = resolve_all()
    imports = []
    for event in resolved:
        if event["event"] != "evidence_imported" or not event["live"]:
            continue
        imports.append({
            "evidence_id": event["evidence_id"],
            "tier": event["effective_tier"],
            "source_study": event.get("source_study"),
            "source_commit": event.get("source_commit"),
            "bundle_sha256": event.get("bundle_sha256"),
            "source_registry_sha256": event.get("source_registry_sha256"),
        })
    durability = verification["payload"]["durability"]
    payload = {
        "schema_version": 1,
        "study": "jspace-phase4",
        "terminal": "Q-L4-Terminal-C",
        "tier": "development-methods",
        "source_commit_at_manifest": information["code_commit"],
        "bound_by_tag": "jspace-phase4-frozen-v1",
        "phase3_import_tag": "jspace-phase3-complete-v1",
        "registry": {
            "path": "interpretability/jspace_phase4/reports/"
                    "evidence_events.jsonl",
            "sha256": file_sha256(EVENTS),
            "rows": len(events),
            "origin_events": len(resolved),
            "live_events": sum(event["live"] for event in resolved),
        },
        "imports": imports,
        "live_outputs": {
            "expected": durability["n_output_references"],
            "verified": durability["n_verified"],
            "known_deficits": durability["n_known_deficits"],
            "unexpected_failures": durability["n_unexpected_failures"],
            "verified_by_resolution": durability[
                "n_verified_by_resolution"],
        },
        "fresh_materialization": {
            "artifact": "interpretability/jspace_phase4/reports/"
                        "PHASE4_PART5_DURABILITY_FRESH.json",
            "sha256": file_sha256(
                REPORTS / "PHASE4_PART5_DURABILITY_FRESH.json"),
        },
        "freeze_verification": {
            "artifact": "interpretability/jspace_phase4/reports/"
                        "PHASE4_FREEZE_VERIFICATION.json",
            "sha256": file_sha256(VERIFICATION_JSON),
        },
        "governance": _document_rows(GOVERNANCE_DOCUMENTS),
        "release_documents": _document_rows(RELEASE_DOCUMENTS),
        "papers": _document_rows(PAPER_SOURCES),
        "tests": {
            name: row["summary"]
            for name, row in verification["payload"]["suites"].items()
        },
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "phase5_authorized": False,
    }
    envelope = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
    }
    atomic_json(MANIFEST_JSON, envelope)
    lines = [
        "# Phase 4 release manifest",
        "",
        "**FROZEN DEVELOPMENT/METHODS RELEASE — Q-L4 TERMINAL C**",
        "",
        f"Manifest payload SHA-256: `{envelope['payload_sha256']}`  ",
        f"Source commit at manifest: "
        f"`{payload['source_commit_at_manifest']}`  ",
        f"Bound by tag: `{payload['bound_by_tag']}` "
        "(the annotated tag object binds the final merged head)  ",
        f"Registry: `{payload['registry']['sha256']}` "
        f"({payload['registry']['rows']} rows, "
        f"{payload['registry']['live_events']} live events)",
        "",
        "## Live outputs",
        "",
        f"- expected {payload['live_outputs']['expected']}, verified "
        f"{payload['live_outputs']['verified']}, known deficits "
        f"{payload['live_outputs']['known_deficits']}, unexpected 0",
        f"- resolution modes: "
        f"`{payload['live_outputs']['verified_by_resolution']}`",
        "",
        "## Imports",
        "",
        "| Event | Source study | Source commit |",
        "|---|---|---|",
    ]
    for row in imports:
        lines.append(
            f"| `{row['evidence_id']}` | {row['source_study']} | "
            f"`{(row['source_commit'] or '')[:12]}` |")
    lines += [
        "",
        "## Governance and release documents",
        "",
        "| Document | SHA-256 |",
        "|---|---|",
    ]
    for section in ("governance", "release_documents", "papers"):
        for name, row in payload[section].items():
            lines.append(f"| {row['path']} | `{row['sha256'][:16]}...` |")
    lines += [
        "",
        f"Phase 5 authorized: **{payload['phase5_authorized']}**.",
        "",
    ]
    MANIFEST_MD.write_text("\n".join(lines))
    return envelope


def main() -> None:
    parser = argparse.ArgumentParser()
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--verify", action="store_true")
    stage.add_argument("--manifest", action="store_true")
    parser.add_argument("--post-merge", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify:
        envelope = verify_stage(post_merge=arguments.post_merge)
        print(json.dumps({
            "payload_sha256": envelope["payload_sha256"],
            "suites": {
                name: row["summary"]
                for name, row in envelope["payload"]["suites"].items()
            },
            "durability": envelope["payload"]["durability"],
        }, indent=1))
    else:
        envelope = manifest_stage()
        print(json.dumps({
            "payload_sha256": envelope["payload_sha256"],
            "registry_sha256":
                envelope["payload"]["registry"]["sha256"],
            "n_imports": len(envelope["payload"]["imports"]),
        }, indent=1))


if __name__ == "__main__":
    main()
