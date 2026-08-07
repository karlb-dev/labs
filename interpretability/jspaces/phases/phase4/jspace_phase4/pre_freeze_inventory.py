"""Generate the Phase 4 pre-freeze registry and durability inventory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

def _find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError("cannot locate git repository root")

import subprocess
import time
from typing import Callable, Iterable, Mapping

from .durability import load_known_deficits, verify_registry_durability
from .manifests import atomic_json, file_sha256, git_info, object_sha256
from .registry4 import EVENTS, resolve_all


PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # interpretability/jspaces/phases/phase4
REPO_ROOT = _find_repo_root()
DEFAULT_POLICY = (
    PACKAGE_ROOT / "protocol/PRE_FREEZE_INVENTORY_POLICY_PHASE4.json")
DEFAULT_KNOWN_DEFICITS = (
    PACKAGE_ROOT / "protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json")
DEFAULT_JSON = PACKAGE_ROOT / "manifests/phase4_pre_freeze_inventory.json"
DEFAULT_MARKDOWN = PACKAGE_ROOT / "manifests/phase4_pre_freeze_inventory.md"

CommitCheck = Callable[[str], bool]


def load_policy(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != 1:
        raise RuntimeError("unsupported pre-freeze inventory policy schema")
    required = {
        "allowed_registered_recovery_outputs",
        "forbidden_path_fragments",
        "native_side_event_prefixes",
    }
    if not required <= set(value):
        raise RuntimeError("pre-freeze inventory policy is incomplete")
    identities = set()
    for row in value["allowed_registered_recovery_outputs"]:
        identity = (
            row.get("evidence_id"), row.get("path_suffix"),
            row.get("expected_sha256"),
        )
        if None in identity or identity in identities or not row.get("reason"):
            raise RuntimeError("invalid registered-recovery exception")
        identities.add(identity)
    return value


def commit_is_ancestor(commit: str, *, repo_root: Path = REPO_ROOT) -> bool:
    if len(commit) != 40:
        return False
    exists = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False).returncode == 0
    if not exists:
        return False
    return subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor",
         commit, "HEAD"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False).returncode == 0


def _matches_recovery_exception(
        reference: Mapping, exceptions: Iterable[Mapping]) -> dict | None:
    for row in exceptions:
        if (
            reference["evidence_id"] == row["evidence_id"]
            and reference["path"].endswith(row["path_suffix"])
            and reference["expected_sha256"] == row["expected_sha256"]
        ):
            return dict(row)
    return None


def _event_commit(event: Mapping) -> str | None:
    if event["event"] == "evidence_imported":
        return event.get("source_commit")
    return event.get("code_commit")


def build_inventory(
        *, events_path: str | Path = EVENTS,
        policy: Mapping,
        known_deficits: Iterable[Mapping] = (),
        hash_file: Callable[[Path], str] = file_sha256,
        commit_check: CommitCheck = commit_is_ancestor,
        repository: Mapping | None = None,
        pass_label: str = "phase4-pre-freeze") -> dict:
    """Build a review inventory; no gate is weakened for known deficits."""
    source = Path(events_path)
    repository = dict(repository or git_info(REPO_ROOT))
    durability = verify_registry_durability(
        events_path=source, known_deficits=known_deficits,
        hash_file=hash_file, pass_label=pass_label)
    resolved = resolve_all(path=source)

    event_rows = []
    unreachable = []
    for event in resolved:
        commit = _event_commit(event)
        reachable = bool(commit and commit_check(str(commit)))
        row = {
            "evidence_id": event["evidence_id"],
            "origin_event": event["event"],
            "effective_tier": event["effective_tier"],
            "live": bool(event["live"]),
            "commit": commit,
            "commit_reachable_from_inventory_head": reachable,
            "n_status_events": len(event["status_events"]),
            "n_outputs": len(event.get(
                "source_outputs" if event["event"] == "evidence_imported"
                else "outputs", []) or []),
        }
        event_rows.append(row)
        if event["live"] and not reachable:
            unreachable.append(row)

    native_prefixes = tuple(policy["native_side_event_prefixes"])
    namespace_violations = [
        row["evidence_id"] for row in event_rows
        if row["evidence_id"].startswith(native_prefixes)
    ]
    exceptions = policy["allowed_registered_recovery_outputs"]
    forbidden_fragments = tuple(policy["forbidden_path_fragments"])
    recovery_exceptions = []
    path_violations = []
    for reference in durability["references"]:
        exception = _matches_recovery_exception(reference, exceptions)
        if exception is not None:
            recovery_exceptions.append({
                "evidence_id": reference["evidence_id"],
                "path": reference["path"],
                "expected_sha256": reference["expected_sha256"],
                "status": reference["status"],
                "reason": exception["reason"],
            })
            continue
        reasons = [
            fragment for fragment in forbidden_fragments
            if fragment in reference["path"]
        ]
        if "/recovery/" in reference["path"]:
            reasons.append("unreviewed-/recovery/-path")
        if reasons:
            path_violations.append({
                "evidence_id": reference["evidence_id"],
                "path": reference["path"],
                "reasons": sorted(set(reasons)),
            })

    policy_exception_identities = {
        (row["evidence_id"], row["path_suffix"], row["expected_sha256"])
        for row in exceptions
    }
    observed_exception_identities = {
        (row["evidence_id"], next(
            policy_row["path_suffix"] for policy_row in exceptions
            if row["evidence_id"] == policy_row["evidence_id"]
            and row["path"].endswith(policy_row["path_suffix"])
            and row["expected_sha256"] == policy_row["expected_sha256"]),
         row["expected_sha256"])
        for row in recovery_exceptions
    }
    missing_policy_exceptions = sorted(
        policy_exception_identities - observed_exception_identities)

    gates = {
        "repository_clean": not bool(repository.get("dirty_tree")),
        "all_live_outputs_verified": bool(durability["ok"]),
        "all_live_event_commits_reachable": not unreachable,
        "no_native_side_event_ids": not namespace_violations,
        "no_unreviewed_temporary_or_recovery_paths": not path_violations,
        "registered_recovery_policy_resolves_exactly": (
            not missing_policy_exceptions
            and len(recovery_exceptions) == len(exceptions)),
        "no_live_path_pin_conflicts": not durability["path_pin_conflicts"],
    }
    review_ready = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "REVIEW_READY" if review_ready else "NOT_REVIEW_READY",
        "generated_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository": repository,
        "registry": {
            "path": str(source),
            "sha256": durability["registry_sha256"],
            "n_rows": durability["n_registry_rows"],
            "n_origin_events": durability["n_origin_events"],
            "n_live_events": durability["n_live_events"],
        },
        "gates": gates,
        "review_ready": review_ready,
        "events": event_rows,
        "unreachable_live_events": unreachable,
        "native_side_namespace_violations": namespace_violations,
        "registered_recovery_exceptions": recovery_exceptions,
        "missing_registered_recovery_policy_rows": [
            list(row) for row in missing_policy_exceptions],
        "temporary_or_recovery_path_violations": path_violations,
        "durability": durability,
        "claim_boundary": (
            "Mechanical pre-freeze inventory only. REVIEW_READY does not "
            "constitute independent review, PI approval, a freeze commit, "
            "a freeze tag, or authorization to open untouched outcomes."),
    }
    return {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
    }


def render_markdown(inventory: Mapping) -> str:
    payload = inventory["payload"]
    durability = payload["durability"]
    lines = [
        "# Phase 4 pre-freeze inventory",
        "",
        f"**{payload['status']} — NOT A FREEZE OR APPROVAL RECORD**",
        "",
        f"Generated: `{payload['generated_utc']}`  ",
        f"Repository commit: `{payload['repository']['code_commit']}`  ",
        f"Registry SHA-256: `{payload['registry']['sha256']}`",
        "",
        "## Mechanical gates",
        "",
        "| Gate | State |",
        "|---|---|",
    ]
    for name, passed in payload["gates"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend([
        "",
        "## Registry and durability",
        "",
        f"- Origin events: {payload['registry']['n_origin_events']}",
        f"- Live events: {payload['registry']['n_live_events']}",
        f"- Output references: {durability['n_output_references']}",
        f"- Verified outputs: {durability['n_verified']}",
        f"- Failures: {durability['n_failures']}",
        f"- Known deficits among failures: {durability['n_known_deficits']}",
        "",
        "## Explicit exceptions and violations",
        "",
        f"- Reviewed immutable recovery-path exceptions: "
        f"{len(payload['registered_recovery_exceptions'])}",
        f"- Unreviewed temporary/recovery paths: "
        f"{len(payload['temporary_or_recovery_path_violations'])}",
        f"- Unreachable live commits: "
        f"{len(payload['unreachable_live_events'])}",
        f"- Native side-namespace violations: "
        f"{len(payload['native_side_namespace_violations'])}",
        "",
        payload["claim_boundary"],
        "",
    ])
    return "\n".join(lines)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(EVENTS))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--known-deficits", default=str(DEFAULT_KNOWN_DEFICITS))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--pass-label", default="phase4-pre-freeze")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    repository = git_info(REPO_ROOT)
    if repository["dirty_tree"]:
        raise SystemExit(
            "pre-freeze inventory refused: repository is not clean")
    inventory = build_inventory(
        events_path=arguments.registry,
        policy=load_policy(arguments.policy),
        known_deficits=load_known_deficits(arguments.known_deficits),
        repository=repository,
        pass_label=arguments.pass_label)
    atomic_json(arguments.json_output, inventory)
    _atomic_text(Path(arguments.markdown_output), render_markdown(inventory))
    print(json.dumps({
        "status": inventory["payload"]["status"],
        "payload_sha256": inventory["payload_sha256"],
        "registry_sha256": inventory["payload"]["registry"]["sha256"],
        "n_live_events": inventory["payload"]["registry"]["n_live_events"],
        "n_output_references": inventory["payload"]["durability"][
            "n_output_references"],
    }, indent=1))
    if not inventory["payload"]["review_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
