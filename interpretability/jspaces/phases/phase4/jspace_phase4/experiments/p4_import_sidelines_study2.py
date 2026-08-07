"""Admit a terminal Study-2 sidelines bundle into the Phase 4 closeout record.

This importer is the dedicated Phase 4.5 second import layer. It admits the
already-merged Gemma/OLMo Study-2 terminal bundles at their native
methods/development tiers without copying any native ``gm2-*``/``ol2-*`` event
into the Phase 4 registry, without touching the earlier Study-1 imports, and
without promoting any tier or opening any intervention meaning.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

def _find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError("cannot locate git repository root")

from typing import Callable, Mapping

import yaml

from ..import_bundle import (
    _repository_materialization,
    git_commit_reachable,
    git_is_ancestor,
    materialize_import_output,
)
from ..manifests import (
    atomic_json,
    file_sha256,
    git_info,
    object_sha256,
    require_clean_tree,
)
from ..registry4 import EVENTS, import_evidence, read_events

REPOSITORY = _find_repo_root()
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_SOURCE_TIERS = {"development", "methods"}
ALLOWED_PHASE4_USES = {"methods-development-boundary-update"}
FORBIDDEN_PHASE4_ORIGIN_PREFIXES = ("gm-", "gm2-", "ol-", "ol2-")
REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "evidence_id",
    "expected_payload_study_id",
    "expected_bundle_id",
    "expected_terminal_event",
    "expected_source_branch",
    "expected_source_commit",
    "allowed_admitted_prefixes",
    "required_partial_statuses",
    "bundle",
    "bundle_markdown",
    "frozen_prefix",
    "live_registry_repo_path",
    "required_dependency_imports",
    "phase4_use",
    "imported_meaning",
    "forbidden_phase4_uses",
    "validation_output",
}


class Sidelines2ImportError(RuntimeError):
    pass


def load_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise Sidelines2ImportError("unsupported sidelines2 import config")
    missing = REQUIRED_CONFIG_KEYS - set(value)
    if missing:
        raise Sidelines2ImportError(
            f"sidelines2 import config lacks keys: {sorted(missing)}")
    if value["phase4_use"] not in ALLOWED_PHASE4_USES:
        raise Sidelines2ImportError(
            f"forbidden phase4_use {value['phase4_use']!r}; allowed: "
            f"{sorted(ALLOWED_PHASE4_USES)}")
    if not str(value["imported_meaning"]).strip():
        raise Sidelines2ImportError("imported_meaning must be non-empty")
    if not value["forbidden_phase4_uses"]:
        raise Sidelines2ImportError("forbidden_phase4_uses must be non-empty")
    if not str(value["evidence_id"]).startswith("p4-import-"):
        raise Sidelines2ImportError("import evidence ID must be p4-import-*")
    if not value["required_dependency_imports"]:
        raise Sidelines2ImportError(
            "a Study-2 admission requires its Study-1 dependency imports")
    return value


def _pinned_file(spec: Mapping, *, repository: Path, label: str) -> dict:
    path = repository / str(spec["repo_path"])
    if not path.is_file():
        raise Sidelines2ImportError(f"{label} is absent: {path}")
    actual = file_sha256(path)
    if actual != spec["sha256"]:
        raise Sidelines2ImportError(
            f"{label} hash drift: expected {spec['sha256']}, got {actual}")
    return {
        "repo_path": str(spec["repo_path"]),
        "sha256": actual,
        "bytes": int(path.stat().st_size),
    }


def _effective_prefix_event(rows: list[dict], evidence_id: str) -> dict:
    origins = [
        row for row in rows
        if row.get("evidence_id") == evidence_id
        and row.get("event") in {"evidence_created", "evidence_imported"}
    ]
    if len(origins) != 1:
        raise Sidelines2ImportError(
            f"frozen prefix lacks exactly one origin for {evidence_id!r}")
    statuses = [
        row for row in rows
        if row.get("evidence_id") == evidence_id
        and row.get("event") not in {"evidence_created", "evidence_imported"}
    ]
    if any(row.get("event") == "evidence_withdrawn" for row in statuses):
        raise Sidelines2ImportError(
            f"admitted event is withdrawn in frozen prefix: {evidence_id}")
    if any(row.get("event") == "evidence_superseded" for row in statuses):
        raise Sidelines2ImportError(
            f"admitted event is superseded in frozen prefix: {evidence_id}")
    effective = dict(origins[0])
    for row in statuses:
        if row.get("event") == "evidence_corrected":
            effective.update(row.get("corrected_fields", {}))
    return effective


def _verify_admitted_output(
        output: Mapping, *, repository: Path) -> dict:
    path = Path(str(output.get("path", "")))
    expected = str(output.get("sha256", ""))
    if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected):
        raise Sidelines2ImportError(
            f"admitted output lacks a full SHA-256: {path}")
    materialized = _repository_materialization(path, repository)
    if not materialized.is_file():
        raise Sidelines2ImportError(f"admitted output is absent: {path}")
    actual_bytes = int(materialized.stat().st_size)
    expected_bytes = output.get("bytes")
    if expected_bytes is not None and actual_bytes != int(expected_bytes):
        raise Sidelines2ImportError(
            f"admitted output byte-count drift: {path}; expected "
            f"{expected_bytes}, got {actual_bytes}")
    actual = file_sha256(materialized)
    if actual != expected:
        raise Sidelines2ImportError(
            f"admitted output hash drift: {path}; expected {expected}, "
            f"got {actual}")
    return {"path": str(path), "sha256": actual, "bytes": actual_bytes}


def validate_sidelines2_bundle(
        config: Mapping, *, repository: Path = REPOSITORY,
        main_events_path: str | Path = EVENTS,
        commit_reachable: Callable[[str], bool] = git_commit_reachable,
        is_ancestor: Callable[[str, str], bool] = git_is_ancestor,
        head_commit: str | None = None) -> dict:
    """Validate one Study-2 terminal bundle for methods-only admission."""
    bundle_row = _pinned_file(
        config["bundle"], repository=repository, label="sidelines2 bundle")
    markdown_row = _pinned_file(
        config["bundle_markdown"], repository=repository,
        label="sidelines2 bundle markdown")
    prefix_row = _pinned_file(
        config["frozen_prefix"], repository=repository,
        label="frozen registry prefix snapshot")

    envelope = json.loads(
        (repository / bundle_row["repo_path"]).read_text())
    if envelope.get("schema_version") != 1 or "payload" not in envelope:
        raise Sidelines2ImportError("invalid sidelines2 bundle envelope")
    payload = envelope["payload"]
    if envelope.get("payload_sha256") != object_sha256(payload):
        raise Sidelines2ImportError("sidelines2 bundle payload hash mismatch")
    if payload.get("schema_version") != 1:
        raise Sidelines2ImportError("unsupported sidelines2 payload schema")
    if payload.get("study_id") != config["expected_payload_study_id"]:
        raise Sidelines2ImportError(
            f"unexpected payload study {payload.get('study_id')!r}")
    if payload.get("bundle_id") != config["expected_bundle_id"]:
        raise Sidelines2ImportError(
            f"unexpected bundle_id {payload.get('bundle_id')!r}")
    if payload.get("evidence_id") != config["expected_terminal_event"]:
        raise Sidelines2ImportError(
            f"unexpected terminal event {payload.get('evidence_id')!r}")

    source_git = payload.get("source_git", {})
    source_commit = str(source_git.get("code_commit", ""))
    if source_git.get("branch") != config["expected_source_branch"]:
        raise Sidelines2ImportError(
            f"unexpected source branch {source_git.get('branch')!r}")
    if source_commit != config["expected_source_commit"]:
        raise Sidelines2ImportError(
            f"unexpected source commit {source_commit!r}")
    if source_git.get("dirty_tree") is not False:
        raise Sidelines2ImportError(
            "bundle was not produced from a clean source tree")
    if len(source_commit) != 40 or not commit_reachable(source_commit):
        raise Sidelines2ImportError(
            f"source commit is not reachable: {source_commit}")
    resolved_head = head_commit or git_info(repository)["code_commit"]
    if not is_ancestor(source_commit, resolved_head):
        raise Sidelines2ImportError(
            "source commit is not an ancestor of the merged head")

    for key, expected in dict(config["required_partial_statuses"]).items():
        actual = payload.get("partial_statuses", {}).get(key)
        if actual != expected:
            raise Sidelines2ImportError(
                f"partial status {key!r} is {actual!r}, requires {expected!r}")
    if not payload.get("forbidden_uses"):
        raise Sidelines2ImportError("bundle lacks forbidden_uses")
    allowed_tiers = set(payload.get("native_tiers", []))
    if not allowed_tiers <= ALLOWED_SOURCE_TIERS:
        raise Sidelines2ImportError(
            f"bundle declares forbidden native tiers: {sorted(allowed_tiers)}")

    prefix_spec = payload.get("registry_prefix", {})
    snapshot_spec = payload.get("registry_snapshot", {})
    if prefix_row["sha256"] != prefix_spec.get("prefix_sha256"):
        raise Sidelines2ImportError(
            "frozen prefix hash does not match bundle registry_prefix")
    if prefix_row["sha256"] != snapshot_spec.get("sha256"):
        raise Sidelines2ImportError(
            "frozen prefix hash does not match bundle registry_snapshot")
    if prefix_row["bytes"] != int(prefix_spec.get("prefix_bytes", -1)):
        raise Sidelines2ImportError("frozen prefix byte count drift")
    prefix_bytes = (repository / prefix_row["repo_path"]).read_bytes()
    prefix_rows = [
        json.loads(line)
        for line in prefix_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    if len(prefix_rows) != int(prefix_spec.get("line_count", -1)):
        raise Sidelines2ImportError("frozen prefix line count drift")

    live_registry = repository / str(config["live_registry_repo_path"])
    if not live_registry.is_file():
        raise Sidelines2ImportError(
            f"live source registry is absent: {live_registry}")
    live_bytes = live_registry.read_bytes()
    if not live_bytes.startswith(prefix_bytes):
        raise Sidelines2ImportError(
            "live source registry has mutated the frozen prefix")
    live_rows = [
        json.loads(line)
        for line in live_bytes.decode("utf-8").splitlines() if line.strip()
    ]

    allowed_prefixes = tuple(config["allowed_admitted_prefixes"])
    admitted = list(payload.get("admitted_evidence", []))
    if not admitted:
        raise Sidelines2ImportError("bundle admits no evidence")
    admitted_ids = [str(row.get("evidence_id", "")) for row in admitted]
    if len(admitted_ids) != len(set(admitted_ids)):
        raise Sidelines2ImportError("duplicate admitted evidence IDs")
    admitted_rows = []
    all_outputs: list[dict] = []
    seen_output_identity: set[tuple[str, str]] = set()

    def _collect(row: dict) -> None:
        identity = (row["path"], row["sha256"])
        if identity not in seen_output_identity:
            seen_output_identity.add(identity)
            all_outputs.append(row)

    for event in admitted:
        evidence_id = str(event.get("evidence_id", ""))
        if not evidence_id.startswith(allowed_prefixes):
            raise Sidelines2ImportError(
                f"foreign admitted evidence namespace: {evidence_id}")
        if evidence_id.startswith("p4-"):
            raise Sidelines2ImportError(
                f"admitted evidence collides with Phase 4 namespace: "
                f"{evidence_id}")
        tier = event.get("tier")
        if tier not in ALLOWED_SOURCE_TIERS:
            raise Sidelines2ImportError(
                f"forbidden source tier {tier!r} on {evidence_id}")
        effective = _effective_prefix_event(prefix_rows, evidence_id)
        if effective.get("tier") != tier:
            raise Sidelines2ImportError(
                f"bundle tier does not match frozen prefix for {evidence_id}")
        if effective.get("code_commit") != event.get("code_commit"):
            raise Sidelines2ImportError(
                f"bundle code commit does not match frozen prefix for "
                f"{evidence_id}")
        # The live registry may append later events but may not restate the
        # admitted event: its effective view must stay identical.
        live_effective = _effective_prefix_event(live_rows, evidence_id)
        if live_effective != effective:
            raise Sidelines2ImportError(
                f"live registry restates admitted event {evidence_id}")
        outputs = [
            _verify_admitted_output(row, repository=repository)
            for row in event.get("outputs", [])
        ]
        if not outputs:
            raise Sidelines2ImportError(
                f"admitted event has no outputs: {evidence_id}")
        expected_outputs = [
            {"path": str(row.get("path")), "sha256": str(row.get("sha256"))}
            for row in effective.get("outputs", [])
        ]
        if [
            {"path": row["path"], "sha256": row["sha256"]} for row in outputs
        ] != expected_outputs:
            raise Sidelines2ImportError(
                f"bundle outputs do not match frozen prefix outputs for "
                f"{evidence_id}")
        for row in outputs:
            _collect(dict(row))
        admitted_rows.append({
            "evidence_id": evidence_id,
            "tier": tier,
            "category": event.get("category"),
            "origin_event": event.get("origin_event"),
            "code_commit": event.get("code_commit"),
            "n_status_events": len(event.get("status_events", [])),
            "outputs": outputs,
        })

    release_rows = []
    for artifact in payload.get("release_artifacts", []):
        row = _pinned_file(
            {"repo_path": artifact["repo_path"], "sha256": artifact["sha256"]},
            repository=repository,
            label=f"release artifact {artifact.get('role')}")
        if int(artifact.get("bytes", -1)) != row["bytes"]:
            raise Sidelines2ImportError(
                f"release artifact byte drift: {artifact['repo_path']}")
        release_rows.append({"role": artifact.get("role"), **row})
        _collect({
            "path": row["repo_path"],
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        })

    main_events = read_events(main_events_path)
    main_origins = {
        row["evidence_id"]: row for row in main_events
        if row.get("event") in {"evidence_created", "evidence_imported"}
    }
    leaked = sorted(
        evidence_id for evidence_id in main_origins
        if evidence_id.startswith(FORBIDDEN_PHASE4_ORIGIN_PREFIXES))
    if leaked:
        raise Sidelines2ImportError(
            f"native side evidence leaked into Phase 4 registry: {leaked}")
    target_id = str(config["evidence_id"])
    if target_id in main_origins:
        raise Sidelines2ImportError(
            f"Phase 4 import evidence ID already exists: {target_id}")
    dependency_rows = []
    for dependency in config["required_dependency_imports"]:
        origin = main_origins.get(str(dependency))
        if origin is None:
            raise Sidelines2ImportError(
                f"required dependency import is absent: {dependency}")
        status = [
            row for row in main_events
            if row.get("evidence_id") == dependency
            and row.get("event") in {
                "evidence_superseded", "evidence_withdrawn"}
        ]
        if status:
            raise Sidelines2ImportError(
                f"required dependency import is not live: {dependency}")
        dependency_rows.append({
            "evidence_id": str(dependency),
            "tier": origin.get("tier"),
            "import_code_commit": origin.get("import_code_commit"),
        })

    materialized_outputs = []
    seen_materialized: set[tuple[str, str]] = set()
    for row in all_outputs:
        materialized = {
            "path": str(materialize_import_output(
                row["path"], repository=repository)),
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        identity = (materialized["path"], materialized["sha256"])
        if identity not in seen_materialized:
            seen_materialized.add(identity)
            materialized_outputs.append(materialized)

    validation = {
        "schema_version": 1,
        "ok": True,
        "import_evidence_id": target_id,
        "phase4_use": config["phase4_use"],
        "imported_meaning": str(config["imported_meaning"]).strip(),
        "forbidden_phase4_uses": list(config["forbidden_phase4_uses"]),
        "bundle": {
            **bundle_row,
            "payload_sha256": envelope["payload_sha256"],
            "bundle_id": payload["bundle_id"],
            "terminal_event": payload["evidence_id"],
            "study_id": payload["study_id"],
            "markdown": markdown_row,
        },
        "source_git": {
            "branch": source_git.get("branch"),
            "code_commit": source_commit,
            "dirty_tree": False,
        },
        "shared_parent": payload.get("shared_parent"),
        "frozen_prefix": {
            **prefix_row,
            "line_count": len(prefix_rows),
            "through_evidence_id": prefix_spec.get("through_evidence_id"),
        },
        "live_registry": {
            "repo_path": str(config["live_registry_repo_path"]),
            "sha256": file_sha256(live_registry),
            "n_rows": len(live_rows),
            "frozen_prefix_intact": True,
        },
        "dependency_imports": dependency_rows,
        "admitted_events": [
            {key: value for key, value in row.items() if key != "outputs"}
            for row in admitted_rows
        ],
        "release_artifacts": release_rows,
        "selected_event_ids": admitted_ids,
        "selected_event_ids_sha256": object_sha256(admitted_ids),
        "source_outputs": materialized_outputs,
        "n_source_outputs": len(materialized_outputs),
        "source_output_inventory_sha256": object_sha256(
            materialized_outputs),
        "result_summary": payload.get("result_summary", {}),
        "claim_boundary": payload.get("claim_boundary"),
        "bundle_forbidden_uses": list(payload["forbidden_uses"]),
        "native_tier_preserved": True,
        "no_confirmatory_or_replication_intervention_outcome": True,
    }
    return validation


def register_sidelines2_import(
        config_path: str | Path, validation_path: str | Path,
        *, backup_roots: list[Path] | None = None) -> dict:
    require_clean_tree()
    if Path.cwd().resolve() != REPOSITORY.resolve():
        raise Sidelines2ImportError(
            "register from the repository root so repo-relative output "
            "paths hash correctly")
    config = load_config(config_path)
    recorded = json.loads(Path(validation_path).read_text())
    current = validate_sidelines2_bundle(config)
    if current != recorded:
        raise Sidelines2ImportError(
            "saved sidelines2 validation does not match a fresh validation")

    output_paths = [row["path"] for row in current["source_outputs"]]
    event = import_evidence(
        current["import_evidence_id"],
        tier="side-development-import",
        what=(
            "Terminal Study-2 methods/development admission from "
            f"{current['bundle']['study_id']}: hash-verified sidelines-2 "
            "bundle updating only the Phase 4 terminal methods boundary; "
            "native tiers preserved; no confirmatory or replication "
            "intervention outcome; no native side event created."),
        source_study=current["bundle"]["study_id"],
        source_evidence_id=current["bundle"]["terminal_event"],
        source_commit=current["source_git"]["code_commit"],
        source_registry=current["frozen_prefix"]["repo_path"],
        source_outputs=output_paths,
        source_branch=current["source_git"]["branch"],
        source_evidence_ids=current["selected_event_ids"],
        source_selected_event_ids_sha256=current[
            "selected_event_ids_sha256"],
        source_output_inventory_sha256=current[
            "source_output_inventory_sha256"],
        bundle_id=current["bundle"]["bundle_id"],
        bundle_sha256=current["bundle"]["sha256"],
        bundle_markdown_sha256=current["bundle"]["markdown"]["sha256"],
        bundle_payload_sha256=current["bundle"]["payload_sha256"],
        frozen_prefix_sha256=current["frozen_prefix"]["sha256"],
        frozen_prefix_path=current["frozen_prefix"]["repo_path"],
        dependency_imports=[
            row["evidence_id"] for row in current["dependency_imports"]],
        phase4_use=current["phase4_use"],
        imported_meaning=current["imported_meaning"],
        forbidden_phase4_uses=current["forbidden_phase4_uses"],
        bundle_forbidden_uses=current["bundle_forbidden_uses"],
        result_summary=current["result_summary"],
        native_tier_preserved=True,
        no_confirmatory_or_replication_intervention_outcome=True,
    )
    backups = []
    for root in backup_roots or []:
        backups.append(backup_registered_outputs(
            event, root, repository=REPOSITORY))
    return {"event": event, "backups": backups}


def backup_registered_outputs(
        event: Mapping, backup_root: Path, *,
        repository: Path = REPOSITORY) -> dict:
    """Copy every registered output byte-exactly under one backup root."""
    destination_root = Path(backup_root) / str(event["evidence_id"])
    rows = []
    for output in event.get("source_outputs", []):
        source = Path(output["path"])
        resolved = source if source.is_absolute() else repository / source
        relative = (
            source.as_posix().lstrip("/") if source.is_absolute()
            else source.as_posix())
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolved, destination)
        actual = file_sha256(destination)
        if actual != output["sha256"]:
            raise Sidelines2ImportError(
                f"backup hash drift for {source}: {actual}")
        rows.append({
            "path": output["path"],
            "backup_path": str(destination),
            "sha256": actual,
            "bytes": int(destination.stat().st_size),
        })
    manifest = {
        "schema_version": 1,
        "evidence_id": event["evidence_id"],
        "n_outputs": len(rows),
        "outputs": rows,
    }
    manifest_path = destination_root / "backup_manifest.json"
    atomic_json(manifest_path, manifest)
    return {
        "backup_root": str(destination_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "n_outputs": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--register", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--validation")
    parser.add_argument("--backup-root", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = load_config(arguments.config)
    if arguments.validate:
        if not arguments.output:
            raise SystemExit("--validate requires --output")
        validation = validate_sidelines2_bundle(config)
        atomic_json(Path(arguments.output), validation)
        print(json.dumps({
            "ok": validation["ok"],
            "import_evidence_id": validation["import_evidence_id"],
            "terminal_event": validation["bundle"]["terminal_event"],
            "bundle_sha256": validation["bundle"]["sha256"],
            "frozen_prefix_sha256": validation["frozen_prefix"]["sha256"],
            "n_admitted_events": len(validation["admitted_events"]),
            "n_source_outputs": validation["n_source_outputs"],
        }, indent=1))
        return
    validation_path = arguments.validation or config["validation_output"]
    result = register_sidelines2_import(
        arguments.config, validation_path,
        backup_roots=[Path(root) for root in arguments.backup_root])
    print(json.dumps({
        "event": result["event"]["event"],
        "evidence_id": result["event"]["evidence_id"],
        "source_study": result["event"]["source_study"],
        "n_source_outputs": len(result["event"]["source_outputs"]),
        "backups": result["backups"],
    }, indent=1))


if __name__ == "__main__":
    main()
