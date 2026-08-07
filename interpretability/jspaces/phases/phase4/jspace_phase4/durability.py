"""Whole-registry durability snapshots for Phase 4 artifacts.

The ordinary reproduction verifier answers whether live outputs hash today.
This module additionally records every reference, distinguishes known
historical deficits from new failures, detects conflicting live path pins,
and compares independent verification passes without weakening the clean
criterion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

from .manifests import atomic_json, file_sha256
from .paths4 import _rewrite_repo_relative
from .registry4 import EVENTS, read_events, resolve_all


HashFile = Callable[[Path], str]
GitShow = Callable[[str, str], bytes]
REPO_ROOT = _find_repo_root()


def _git_show(commit: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{relative}"],
        stderr=subprocess.DEVNULL)


def _repository_relative(path: Path) -> Path | None:
    """Map a producer-worktree absolute path onto this repository.

    Only a suffix beginning at the literal ``interpretability`` directory is
    eligible; Drive and other external paths never remap.
    """
    if not path.is_absolute():
        return path
    try:
        marker = path.parts.index("interpretability")
    except ValueError:
        return None
    return Path(*path.parts[marker:])


def resolve_output_reference(
        path_str: str, expected_sha256: str | None, *,
        event: Mapping, expected_bytes: int | None = None,
        repo_root: Path = REPO_ROOT,
        hash_file: HashFile = file_sha256,
        git_show: GitShow = _git_show) -> dict:
    """Strictly resolve one registered output reference.

    Hashing is never weakened: a reference resolves only to bytes whose
    SHA-256 equals the registered pin. Three lawful materializations exist:

    1. the literal registered path;
    2. the identical tracked file in the current repository, when the
       registered absolute path came from another VM's worktree; and
    3. for append-only registry files, the exact historical bytes at the
       registering commit, provided the live file is a byte-prefix
       extension of those bytes.
    """
    row = {
        "exists": False,
        "actual_sha256": None,
        "actual_bytes": None,
        "status": "missing",
        "resolution": None,
        "resolved_path": None,
        "error": None,
    }
    path = Path(path_str)
    try:
        if path.is_file():
            row["exists"] = True
            row["actual_bytes"] = int(path.stat().st_size)
            row["actual_sha256"] = hash_file(path)
            if (
                expected_bytes is not None
                and row["actual_bytes"] != expected_bytes
            ):
                row["status"] = "byte_count_mismatch"
            elif row["actual_sha256"] != expected_sha256:
                row["status"] = "hash_mismatch"
            else:
                row["status"] = "verified"
                row["resolution"] = "literal-path"
                return row
    except (OSError, RuntimeError) as error:
        row["status"] = "read_error"
        row["error"] = f"{type(error).__name__}: {error}"
        return row

    relative = _repository_relative(path)
    if relative is None:
        return row
    rewritten = Path(_rewrite_repo_relative(relative.as_posix()))
    candidate = repo_root / rewritten
    if not candidate.is_file():
        candidate = repo_root / relative

    if not row["exists"] and candidate.is_file():
        try:
            actual_bytes = int(candidate.stat().st_size)
            actual = hash_file(candidate)
        except (OSError, RuntimeError) as error:
            row["status"] = "read_error"
            row["error"] = f"{type(error).__name__}: {error}"
            return row
        if actual == expected_sha256 and (
                expected_bytes is None or actual_bytes == expected_bytes):
            row.update({
                "exists": True,
                "actual_bytes": actual_bytes,
                "actual_sha256": actual,
                "status": "verified",
                "resolution": "repository-materialization",
                "resolved_path": str(candidate),
            })
        return row

    if (
        row["status"] == "hash_mismatch"
        and path.name == "evidence_events.jsonl"
        and candidate.is_file()
    ):
        commit = str(
            event.get("import_code_commit")
            or event.get("code_commit") or "")
        if len(commit) == 40:
            historical = None
            for spec in dict.fromkeys(
                    (relative.as_posix(), rewritten.as_posix())):
                try:
                    historical = git_show(commit, spec)
                    break
                except (OSError, subprocess.CalledProcessError):
                    continue
            if historical is None:
                return row
            if (
                hashlib.sha256(historical).hexdigest() == expected_sha256
                and candidate.read_bytes().startswith(historical)
                and (expected_bytes is None
                     or len(historical) == expected_bytes)
            ):
                row.update({
                    "actual_bytes": len(historical),
                    "actual_sha256": expected_sha256,
                    "status": "verified",
                    "resolution":
                        f"append-only-registry-prefix@{commit}",
                    "resolved_path": str(candidate),
                })
    return row


def _output_field(event: Mapping) -> str:
    return (
        "source_outputs"
        if event.get("event") == "evidence_imported" else "outputs")


def load_known_deficits(path: str | Path | None) -> list[dict]:
    if path is None:
        return []
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != 1:
        raise RuntimeError("unsupported durability-deficit schema")
    deficits = list(value.get("deficits", []))
    identities = set()
    for row in deficits:
        required = {"evidence_id", "path_suffix", "expected_sha256"}
        if not required <= set(row):
            raise RuntimeError("known durability deficit lacks required fields")
        identity = (
            str(row["evidence_id"]), str(row["path_suffix"]),
            str(row["expected_sha256"]),
        )
        if identity in identities:
            raise RuntimeError("duplicate known durability deficit")
        identities.add(identity)
    return deficits


def _known_deficit(reference: Mapping, deficits: Iterable[Mapping]) -> bool:
    return any(
        reference["evidence_id"] == row["evidence_id"]
        and reference["path"].endswith(str(row["path_suffix"]))
        and reference["expected_sha256"] == row["expected_sha256"]
        for row in deficits
    )


def _reference_identity(row: Mapping) -> tuple[str, str, str, int]:
    return (
        str(row["evidence_id"]), str(row["path"]),
        str(row["expected_sha256"]), int(row["ordinal"]),
    )


def verify_registry_durability(
        *, events_path: str | Path = EVENTS,
        known_deficits: Iterable[Mapping] = (),
        hash_file: HashFile = file_sha256,
        pass_label: str = "manual",
        repo_root: Path = REPO_ROOT,
        git_show: GitShow = _git_show) -> dict:
    """Hash every live registry output and return a durable plain-JSON rowset."""
    source = Path(events_path)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    events = resolve_all(path=source)
    events_by_id = {event["evidence_id"]: event for event in events}
    live = [event for event in events if event["live"]]
    references = []
    for event in live:
        field = _output_field(event)
        for ordinal, output in enumerate(event.get(field, []) or []):
            references.append({
                "evidence_id": event["evidence_id"],
                "effective_tier": event["effective_tier"],
                "event_type": event["event"],
                "output_field": field,
                "ordinal": ordinal,
                "path": str(output["path"]),
                "expected_sha256": output.get("sha256"),
                "expected_bytes": output.get("bytes"),
            })

    pins_by_path: dict[str, set[str | None]] = {}
    for row in references:
        pins_by_path.setdefault(row["path"], set()).add(
            row["expected_sha256"])
    conflicts = [
        {"path": path, "expected_sha256_values": sorted(values)}
        for path, values in sorted(pins_by_path.items())
        if len(values) > 1
    ]

    rows = []
    deficits = list(known_deficits)
    for reference in references:
        row = dict(reference)
        row.update(resolve_output_reference(
            reference["path"], reference["expected_sha256"],
            event=events_by_id[reference["evidence_id"]],
            expected_bytes=reference["expected_bytes"],
            repo_root=repo_root, hash_file=hash_file, git_show=git_show))
        row["known_deficit"] = bool(
            row["status"] != "verified"
            and _known_deficit(reference, deficits))
        rows.append(row)

    failures = [row for row in rows if row["status"] != "verified"]
    unexpected = [row for row in failures if not row["known_deficit"]]
    resolved_known = []
    for deficit in deficits:
        matching = [
            row for row in rows
            if row["evidence_id"] == deficit["evidence_id"]
            and row["path"].endswith(str(deficit["path_suffix"]))
            and row["expected_sha256"] == deficit["expected_sha256"]
        ]
        if len(matching) != 1:
            raise RuntimeError(
                "known deficit does not identify exactly one live output: "
                + json.dumps(deficit, sort_keys=True))
        if matching[0]["status"] == "verified":
            resolved_known.append(dict(deficit))

    resolution_counts: dict[str, int] = {}
    for row in rows:
        if row["status"] == "verified":
            mode = str(row.get("resolution") or "literal-path")
            if mode.startswith("append-only-registry-prefix"):
                mode = "append-only-registry-prefix"
            resolution_counts[mode] = resolution_counts.get(mode, 0) + 1

    completed = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "schema_version": 1,
        "pass_label": pass_label,
        "started_utc": started,
        "completed_utc": completed,
        "registry_path": str(source),
        "registry_sha256": file_sha256(source),
        "n_registry_rows": len(read_events(source)),
        "n_origin_events": len(events),
        "n_live_events": len(live),
        "n_non_live_events": len(events) - len(live),
        "n_output_references": len(rows),
        "n_unique_output_paths": len(pins_by_path),
        "n_verified": len(rows) - len(failures),
        "n_verified_by_resolution": resolution_counts,
        "n_failures": len(failures),
        "n_known_deficits": sum(row["known_deficit"] for row in failures),
        "n_unexpected_failures": len(unexpected),
        "path_pin_conflicts": conflicts,
        "resolved_known_deficits": resolved_known,
        "only_known_deficits": bool(
            failures and not unexpected and not conflicts),
        "ok": not failures and not conflicts,
        "references": rows,
    }


def compare_durability_passes(first: Mapping, second: Mapping) -> dict:
    """Compare two independently materialized whole-registry snapshots."""
    registry_matches = (
        first.get("registry_sha256") == second.get("registry_sha256"))
    first_rows = {
        _reference_identity(row): row for row in first.get("references", [])}
    second_rows = {
        _reference_identity(row): row for row in second.get("references", [])}
    reference_set_matches = set(first_rows) == set(second_rows)
    drifts = []
    for identity in sorted(set(first_rows) & set(second_rows)):
        left, right = first_rows[identity], second_rows[identity]
        fields = ("status", "actual_sha256", "actual_bytes")
        changed = {
            field: [left.get(field), right.get(field)]
            for field in fields if left.get(field) != right.get(field)
        }
        if changed:
            drifts.append({
                "evidence_id": identity[0],
                "path": identity[1],
                "changed": changed,
            })
    consistent = bool(registry_matches and reference_set_matches and not drifts)
    return {
        "schema_version": 1,
        "first_pass_label": first.get("pass_label"),
        "second_pass_label": second.get("pass_label"),
        "registry_matches": registry_matches,
        "reference_set_matches": reference_set_matches,
        "drifts": drifts,
        "consistent": consistent,
        "clean_both": bool(consistent and first.get("ok") and second.get("ok")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(EVENTS))
    parser.add_argument("--known-deficits")
    parser.add_argument("--pass-label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--previous")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    deficits = load_known_deficits(arguments.known_deficits)
    result = verify_registry_durability(
        events_path=arguments.registry,
        known_deficits=deficits,
        pass_label=arguments.pass_label,
    )
    if arguments.previous:
        previous = json.loads(Path(arguments.previous).read_text())
        result["comparison_to_previous"] = compare_durability_passes(
            previous, result)
    atomic_json(Path(arguments.output), result)
    print(json.dumps({
        key: result[key] for key in (
            "pass_label", "registry_sha256", "n_live_events",
            "n_output_references", "n_verified", "n_failures",
            "n_known_deficits", "n_unexpected_failures", "ok")
    }, indent=1))
    if not result["ok"]:
        raise SystemExit(1)
    comparison = result.get("comparison_to_previous")
    if comparison is not None and not comparison["clean_both"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
