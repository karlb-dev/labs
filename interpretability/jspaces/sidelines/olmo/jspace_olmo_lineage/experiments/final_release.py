"""Emit and verify the final isolated OLMo Phase 4 handoff bundle.

The bundle is a methods artifact.  It inventories already registered evidence,
copies the isolated paper and claim reports, and records the exact registry
prefix that Phase 5 may import.  It does not open another scientific cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

def _find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError("cannot locate git repository root")

from typing import Mapping, Sequence

import yaml

from ..manifests import (
    atomic_json,
    atomic_text,
    commit_reachable,
    environment_payload,
    file_sha256,
    object_sha256,
    require_clean_tree,
    verify_constraints,
)
from ..paths import local_work, resolve_uri, run_root
from ..registry import EVENTS, create, read_events, resolve, resolve_all
from ..repro import verify_live_evidence

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = _find_repo_root()
NATIVE_TIERS = {"development", "methods"}
CONSTRAINT_PACKAGES = {
    "huggingface_hub", "matplotlib", "numpy", "pandas", "pyarrow",
    "pyyaml", "scipy", "tokenizers", "torch", "transformers",
}


def load_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("final-release config must be a mapping")
    return value


def _envelope(payload: Mapping) -> dict:
    body = dict(payload)
    return {
        "schema_version": 1,
        "payload": body,
        "payload_sha256": object_sha256(body),
    }


def _load_envelope(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or not isinstance(value.get("payload"), dict):
        raise ValueError(f"expected payload envelope: {path}")
    actual = object_sha256(value["payload"])
    if actual != value.get("payload_sha256"):
        raise ValueError(f"payload-envelope hash drift: {path}")
    return value


def _repo_path(path: str | Path) -> str:
    source = Path(path)
    try:
        return str(source.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(source)


def _registry_prefix_record(path: str | Path = EVENTS) -> dict:
    source = Path(path)
    raw = source.read_bytes()
    events = read_events(source)
    records = resolve_all(path=source)
    live = [record for record in records if record["live"]]
    if not events:
        raise ValueError("cannot release an empty evidence registry")
    return {
        "path": _repo_path(source),
        "prefix_bytes": len(raw),
        "prefix_sha256": hashlib.sha256(raw).hexdigest(),
        "line_count": len(events),
        "origin_events": len(records),
        "live_events": len(live),
        "live_outputs": sum(
            len(record["effective_metadata"].get("outputs") or [])
            for record in live
        ),
        "through_evidence_id": events[-1]["evidence_id"],
        "through_event": events[-1]["event"],
        "note": "The byte prefix excludes the final-release event itself.",
    }


def _verify_registry_prefix(
    record: Mapping, path: str | Path = EVENTS,
) -> dict:
    source = Path(path)
    raw = source.read_bytes()
    length = int(record["prefix_bytes"])
    if len(raw) < length:
        raise ValueError("registry is shorter than the released prefix")
    prefix = raw[:length]
    actual = hashlib.sha256(prefix).hexdigest()
    if actual != record["prefix_sha256"]:
        raise ValueError("released registry prefix hash drift")
    lines = [line for line in prefix.decode("utf-8").splitlines()
             if line.strip()]
    if len(lines) != int(record["line_count"]):
        raise ValueError("released registry prefix line-count drift")
    last = json.loads(lines[-1])
    if (last.get("evidence_id") != record["through_evidence_id"]
            or last.get("event") != record["through_event"]):
        raise ValueError("released registry-prefix endpoint drift")
    return {
        "ok": True,
        "prefix_bytes": length,
        "prefix_sha256": actual,
        "current_registry_bytes": len(raw),
    }


def _release_boundary_state(
    record: Mapping, release_evidence_id: str, path: str | Path = EVENTS,
) -> dict:
    """Recompute the release-boundary registry state after append-only merges.

    The release contract fixed the live-event and live-output counts at the
    moment the final-release event landed.  A later ancestry-preserving merge
    may legitimately append new native events after that boundary, so the
    counts are recomputed over the hash-verified immutable prefix plus the
    release event row itself, never over the growing tail.  Rows after the
    boundary may only add new evidence; any supersession, withdrawal, or
    correction that restates boundary-era evidence is reported and fails the
    caller's gate.
    """
    raw = Path(path).read_bytes()
    length = int(record["prefix_bytes"])
    prefix = raw[:length]
    if hashlib.sha256(prefix).hexdigest() != record["prefix_sha256"]:
        raise ValueError("released registry prefix hash drift")
    rows = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines() if line.strip()
    ]
    boundary_lines = [
        json.loads(line)
        for line in prefix.decode("utf-8").splitlines() if line.strip()
    ]
    release_row = rows[len(boundary_lines)]
    if (release_row.get("evidence_id") != release_evidence_id
            or release_row.get("event") != "evidence_created"):
        raise ValueError(
            "row after the released prefix is not the release event")
    boundary_rows = boundary_lines + [release_row]
    origin_ids = {
        row["evidence_id"] for row in boundary_rows
        if row.get("event") in {"evidence_created", "evidence_imported"}
    }
    dead = set()
    for row in boundary_rows:
        if row.get("event") in {"evidence_superseded", "evidence_withdrawn"}:
            dead.add(row["evidence_id"])
    live_outputs = 0
    for row in boundary_rows:
        if (row.get("event") in {"evidence_created", "evidence_imported"}
                and row["evidence_id"] not in dead):
            field = (
                "source_outputs" if row["event"] == "evidence_imported"
                else "outputs")
            live_outputs += len(row.get(field, []) or [])
    restatements = [
        {"event": row.get("event"), "evidence_id": row.get("evidence_id")}
        for row in rows[len(boundary_rows):]
        if row.get("event") not in {"evidence_created", "evidence_imported"}
        and row.get("evidence_id") in origin_ids
    ]
    return {
        "n_live_events": len(origin_ids - dead),
        "n_live_outputs": live_outputs,
        "post_boundary_restatements": restatements,
    }


def _validate_partition(
    categories: Mapping[str, Sequence[str]], live_evidence_ids: Sequence[str],
) -> dict:
    expected = set(live_evidence_ids)
    assigned: dict[str, str] = {}
    duplicates = {}
    for category, identifiers in categories.items():
        for evidence_id in identifiers:
            if evidence_id in assigned:
                duplicates.setdefault(evidence_id, [assigned[evidence_id]])
                duplicates[evidence_id].append(category)
            else:
                assigned[evidence_id] = category
    missing = sorted(expected - set(assigned))
    extra = sorted(set(assigned) - expected)
    if duplicates or missing or extra:
        raise ValueError(
            "evidence-category partition drift: "
            f"duplicates={duplicates}, missing={missing}, extra={extra}")
    return {
        "ok": True,
        "n_categories": len(categories),
        "n_evidence_ids": len(assigned),
        "category_counts": {
            name: len(identifiers)
            for name, identifiers in categories.items()
        },
    }


def _hash_record(
    path: str | Path, *, role: str, target_path: str | Path | None = None,
    logical_uri: str | None = None,
) -> dict:
    source = Path(path)
    record = {
        "role": role,
        "sha256": file_sha256(source),
        "bytes": int(source.stat().st_size),
    }
    if target_path is not None:
        record["path"] = str(Path(target_path))
    else:
        record["path"] = str(source)
    if logical_uri is not None:
        record["uri"] = logical_uri
    return record


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(
            f"state/report template drift for {label}: found {text.count(old)}")
    return text.replace(old, new, 1)


def _canonical_claim_text(value: str) -> str:
    """Normalize prose wrapping, including a YAML fold after a hyphen."""
    compact = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"(?<=\w)-\s+(?=\w)", "-", compact)


def _released_state_text(
    source: str,
    *,
    config: Mapping,
    source_commit: str,
    release_utc: str,
    source_sha256: str,
) -> str:
    expected = config["expected_registry_before_release"]
    final_origins = int(expected["origin_events"]) + 1
    final_live = int(expected["live_events"]) + 1
    final_outputs = int(expected["live_outputs"]) + 13
    tests = int(expected["package_tests"])
    source = re.sub(
        r"State date: [^\n]+",
        f"State date: {release_utc}", source, count=1)
    source = _replace_once(
        source,
        "Status: OLMo parallel-phase scientific execution and the isolated "
        "run-specific\n"
        "paper are complete at the first release boundary. The final import/restart\n"
        "bundle is being assembled.",
        "Status: complete at the first OLMo parallel-phase release boundary. The\n"
        "self-verifying import/restart bundle is emitted and registered; this side\n"
        "track is stopped pending the single Phase 5 integration router.",
        label="release status")
    registry_pattern = re.compile(
        r"At this state date the append-only registry contains 24 origin events,.*?"
        r"`12f21ad5badeac980c11f0817906ad18c6c1d52d`\.\n",
        flags=re.DOTALL,
    )
    replacement = (
        "The frozen pre-release registry prefix contains "
        f"{expected['origin_events']} origin events, of which "
        f"{expected['live_events']} are live, and "
        f"{expected['live_outputs']} live immutable outputs. "
        "`ol-checkpoint-inventory-v1` remains immutable but is explicitly "
        "superseded by version 2. The final methods event adds 13 immutable "
        f"handoff outputs, giving {final_origins} origins, {final_live} live "
        f"events, and {final_outputs} live outputs. All pass byte/hash "
        f"verification. {tests} package tests and the exact dependency lock "
        "pass.\n\n"
        f"The latest event is `{config['evidence_id']}`, emitted from clean "
        f"source commit `{source_commit}`. Its embedded prefix ends through "
        "`ol-independent-reconstruction-v1`.\n"
    )
    source, count = registry_pattern.subn(replacement, source, count=1)
    if count != 1:
        raise ValueError("state/report template drift for registry summary")
    source = _replace_once(
        source,
        "`reports/OLMO_LINEAGE_CLAIMS_TABLE.md`; the final bundle will contain a\n"
        "hash-pinned copy.",
        "`reports/OLMO_LINEAGE_CLAIMS_TABLE.md`; the final bundle contains its\n"
        "hash-pinned release copy.",
        label="claims bundle tense")
    source = _replace_once(
        source,
        "Required to finish this release artifact layer:\n\n"
        "1. emit and register the self-verifying final OLMo import/restart bundle;\n"
        "2. stop this side track and hand its queues to Phase 5.",
        "This release artifact layer is complete. The side track is stopped; all\n"
        "unopened work remains queued for an explicit Phase 5 decision.",
        label="remaining release work")
    source = _replace_once(
        source,
        "- [ ] final import/restart bundle emitted and registered.",
        "- [x] final import/restart bundle emitted and registered.",
        label="release checklist")
    source = _replace_once(
        source,
        "When the final two boxes are complete, this workstream stops and joins the\n"
        "single Phase 5 router only through its hash-pinned handoff.",
        "This workstream is stopped and may join the single Phase 5 router only\n"
        "through this hash-pinned handoff.",
        label="phase-5 stop rule")
    source += (
        "\n\n## 11. Final release attestation\n\n"
        f"- Evidence ID: `{config['evidence_id']}`.\n"
        f"- Bundle ID: `{config['bundle_id']}`.\n"
        f"- Clean source commit: `{source_commit}`.\n"
        f"- Source state-report SHA-256: `{source_sha256}`.\n"
        "- The embedded registry prefix excludes only the final release event; "
        "the event registers this state, claims ledger, inventory, environment "
        "lock, bundle pair, isolated paper, and five paper figures.\n"
        "- No scientific outcome, intervention, Phase 4 result, Gemma result, "
        "confirmatory claim, or replication claim is added by the release event.\n"
    )
    return source


def _released_claims_text(
    source: str,
    *,
    config: Mapping,
    source_commit: str,
    release_utc: str,
    source_sha256: str,
) -> str:
    source = re.sub(
        r"State date: [^\n]+",
        f"State date: {release_utc}", source, count=1)
    source = _replace_once(
        source,
        "This mutable ledger is a recovery/reporting aid until copied into\n"
        "the hash-pinned final release bundle; the append-only evidence registry and\n"
        "registered Drive outputs remain authoritative.",
        "This ledger is copied into the hash-pinned final release bundle; the\n"
        "append-only evidence registry and registered Drive outputs remain\n"
        "authoritative.",
        label="claims release status")
    source += (
        "\n\n## Final release attestation\n\n"
        f"This copy is an output of `{config['evidence_id']}` from clean source "
        f"commit `{source_commit}`. The mutable source ledger used to produce it "
        f"has SHA-256 `{source_sha256}`. Sentence 2 remains "
        f"**{config['claim_resolution']['sentence_2']['status']}** and sentence 4 "
        f"remains **{config['claim_resolution']['sentence_4']['status']}**. "
        "Queued H5/H6/O5 work is not an outcome.\n"
    )
    return source


def _version(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "unavailable"
    text = (result.stdout or result.stderr).strip().splitlines()
    return text[0] if text else f"exit-{result.returncode}"


def _source_file(config_value: str) -> Path:
    path = PACKAGE_ROOT / config_value
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_source_artifacts(config: Mapping) -> dict:
    source = config["source_artifacts"]
    state = _source_file(source["state_report"])
    claims = _source_file(source["claims_report"])
    records = {
        "state_report": _hash_record(state, role="mutable-source-state"),
        "claims_report": _hash_record(claims, role="mutable-source-claims"),
    }
    for key in ("paper_tex", "paper_pdf"):
        specification = source[key]
        path = _source_file(specification["path"])
        actual = file_sha256(path)
        if actual != specification["sha256"]:
            raise ValueError(f"source {key} hash drift")
        records[key] = _hash_record(path, role=f"source-{key}")
    figures = []
    for specification in source["paper_figures"]:
        path = _source_file(specification["path"])
        actual = file_sha256(path)
        if actual != specification["sha256"]:
            raise ValueError(f"source paper figure hash drift: {path.name}")
        figures.append(_hash_record(path, role="source-paper-figure"))
    records["paper_figures"] = figures
    for key in (
        "foundation_environment_lock", "early_bundle_json",
        "early_bundle_markdown",
    ):
        specification = source[key]
        path = resolve_uri(specification["uri"])
        if file_sha256(path) != specification["sha256"]:
            raise ValueError(f"source {key} hash drift")
        records[key] = _hash_record(path, role=f"source-{key}")
    return records


def _collect_live_inventory(
    config: Mapping, *, release_utc: str, source_commit: str,
) -> tuple[dict, dict]:
    expected = config["expected_registry_before_release"]
    records = resolve_all()
    live = [record for record in records if record["live"]]
    counts = {
        "origin_events": len(records),
        "live_events": len(live),
        "live_outputs": sum(
            len(record["effective_metadata"].get("outputs") or [])
            for record in live
        ),
    }
    for key in ("origin_events", "live_events", "live_outputs"):
        if counts[key] != int(expected[key]):
            raise ValueError(
                f"pre-release registry {key} drift: "
                f"{counts[key]} != {expected[key]}")
    live_ids = [record["evidence_id"] for record in live]
    partition = _validate_partition(config["evidence_categories"], live_ids)
    verification = verify_live_evidence()
    if not verification["ok"]:
        raise ValueError("live evidence does not hash-verify")
    root = run_root(create=False).resolve()
    rows = []
    for record in live:
        effective = record["effective_metadata"]
        tier = effective["tier"]
        if tier not in NATIVE_TIERS:
            raise ValueError(f"non-native effective tier: {record['evidence_id']}")
        outputs = []
        for registered in effective.get("outputs") or []:
            path = Path(registered["path"])
            try:
                path.resolve().relative_to(root)
            except ValueError as error:
                raise ValueError(f"output escapes OLMo root: {path}") from error
            actual = _hash_record(path, role="registered-live-output")
            if (actual["sha256"] != registered["sha256"]
                    or actual["bytes"] != int(registered["bytes"])):
                raise ValueError(f"registered output drift: {path}")
            outputs.append(actual)
        rows.append({
            "evidence_id": record["evidence_id"],
            "category": next(
                category for category, identifiers
                in config["evidence_categories"].items()
                if record["evidence_id"] in identifiers),
            "origin_event": record["event"],
            "event_utc": record["event_utc"],
            "tier": tier,
            "code_commit": effective.get("code_commit"),
            "what": effective.get("what"),
            "command": effective.get("command"),
            "verdict": effective.get("verdict"),
            "claim_boundary": effective.get("claim_boundary"),
            "outputs": outputs,
        })
    superseded = []
    for record in records:
        if record["superseded_by"] is not None:
            superseded.append({
                "evidence_id": record["evidence_id"],
                "superseded_by": record["superseded_by"],
                "status_events": record["status_events"],
            })
    expected_superseded = dict(expected["superseded"])
    actual_superseded = {
        row["evidence_id"]: row["superseded_by"] for row in superseded
    }
    if actual_superseded != expected_superseded:
        raise ValueError("supersession map drift")
    prefix = _registry_prefix_record()
    payload = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "generated_utc": release_utc,
        "source_commit": source_commit,
        "scientific_import_boundary": config["scientific_import_boundary"],
        "registry_prefix": prefix,
        "counts": counts,
        "partition_validation": partition,
        "evidence_categories": config["evidence_categories"],
        "live_evidence": rows,
        "superseded_evidence": superseded,
        "native_tiers": sorted(NATIVE_TIERS),
    }
    return _envelope(payload), prefix


def _environment_lock(
    config: Mapping, source_records: Mapping, *, release_utc: str,
    source_commit: str,
) -> dict:
    foundation_path = Path(
        source_records["foundation_environment_lock"]["path"])
    foundation = json.loads(foundation_path.read_text())
    constraints = verify_constraints(
        PACKAGE_ROOT / "constraints.txt", package_names=CONSTRAINT_PACKAGES)
    if not constraints["ok"]:
        raise ValueError("release environment violates the exact constraints")
    payload = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "generated_utc": release_utc,
        "source_commit": source_commit,
        "foundation_environment_lock": {
            "path": str(foundation_path),
            "sha256": source_records["foundation_environment_lock"]["sha256"],
            "bytes": source_records["foundation_environment_lock"]["bytes"],
            "content": foundation,
        },
        "release_runtime": environment_payload(require_gpu=False),
        "constraint_verification": constraints,
        "paper_toolchain": {
            "source_date_epoch": int(config["paper_source_date_epoch"]),
            "pdflatex": _version(["pdflatex", "--version"]),
            "pdfinfo": _version(["pdfinfo", "-v"]),
            "git": _version(["git", "--version"]),
        },
    }
    return _envelope(payload)


def _completion_payload() -> dict:
    return {
        "O1": "complete; Bank-W v1 service gate blocked at 16/20",
        "O2": "complete; broadly-conserved-capacity-recruitment-consistent",
        "O3": "complete; dictionary-formation-pattern",
        "O4": "resolved-gate-blocked; no intervention opened",
        "H5_inventory": "complete; bounded official SFT/DPO wedge queued",
        "O5": "resolved-no-identifiable-estimand; no proxy substitution",
        "independent_reconstruction": "pass",
        "claims_and_state": "complete",
        "isolated_paper": "complete",
        "parallel_phase": "stopped-at-first-release-boundary",
    }


def _render_markdown(payload: Mapping) -> str:
    prefix = payload["registry_prefix"]
    claims = payload["claim_resolution"]
    lines = [
        "# OLMo lineage Phase 4 final import bundle",
        "",
        f"Bundle ID: `{payload['bundle_id']}`  ",
        f"Evidence ID: `{payload['evidence_id']}`  ",
        f"Generated: `{payload['generated_utc']}`  ",
        f"Clean source commit: `{payload['source_git']['code_commit']}`  ",
        f"Scientific import boundary: `{payload['scientific_import_boundary']}`",
        "",
        "## Import boundary",
        "",
        f"The exact pre-release registry prefix is {prefix['prefix_bytes']:,} "
        f"bytes / {prefix['line_count']} events, SHA-256 "
        f"`{prefix['prefix_sha256']}`. It ends at "
        f"`{prefix['through_evidence_id']}` and contains "
        f"{prefix['origin_events']} origins, {prefix['live_events']} live events, "
        f"and {prefix['live_outputs']} live outputs. Inventory v1 remains "
        "immutable and superseded by v2.",
        "",
        "This is an OLMo-only methods handoff. It neither imports into nor writes "
        "the concurrent Phase 4 or Gemma namespaces. Phase 5 must verify the "
        "registry prefix and every artifact hash before importing it.",
        "",
        "## Disposition",
        "",
    ]
    for name, value in payload["completion"].items():
        lines.append(f"- **{name}:** {value}.")
    lines.extend([
        "",
        "## Licensed conclusion wording",
        "",
        f"- **Sentence 2 — {claims['sentence_2']['status']}:** "
        f"{claims['sentence_2']['wording']}",
        "",
        f"- **Sentence 4 — {claims['sentence_4']['status']}:** "
        f"{claims['sentence_4']['wording']}",
        "",
        "## Phase 5 queues",
        "",
    ])
    for name, value in payload["phase5_queues"].items():
        lines.append(f"- `{name}`: `{value}`.")
    lines.extend(["", "## Forbidden uses", ""])
    for value in payload["forbidden_uses"]:
        lines.append(f"- {value}.")
    lines.extend([
        "",
        "## Companion artifacts",
        "",
        "| Role | Logical URI | Bytes | SHA-256 |",
        "|---|---|---:|---|",
    ])
    for row in payload["artifacts"]:
        lines.append(
            f"| {row['role']} | `{row.get('uri', row['path'])}` | "
            f"{row['bytes']} | `{row['sha256']}` |")
    lines.extend([
        "",
        "The JSON companion contains the machine-readable inventory, queue, "
        "claim, environment, artifact, and importer-check records. This Markdown "
        "file intentionally does not hash itself; the JSON companion and final "
        "registry event do.",
        "",
        "## Claim boundary",
        "",
        payload["claim_boundary"],
        "",
    ])
    return "\n".join(lines)


def _atomic_copy(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp{os.getpid()}")
    with source_path.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, target)


def _target_paths(config: Mapping) -> tuple[dict[str, Path], Path]:
    outputs = config["outputs"]
    targets = {
        name: resolve_uri(uri, must_exist=False)
        for name, uri in outputs.items()
        if name != "paper_figure_directory"
    }
    figure_directory = resolve_uri(
        outputs["paper_figure_directory"], must_exist=False)
    root = run_root(create=False).resolve()
    values = list(targets.values()) + [figure_directory]
    if len({str(path) for path in values}) != len(values):
        raise ValueError("release output paths are not unique")
    for path in values:
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"release target escapes OLMo root: {path}") from error
    return targets, figure_directory


def _ensure_absent(targets: Mapping[str, Path], figure_directory: Path) -> None:
    occupied = [str(path) for path in targets.values() if path.exists()]
    if figure_directory.exists():
        occupied.append(str(figure_directory))
    if occupied:
        raise FileExistsError(
            "immutable final-release targets already exist; audit/verify rather "
            f"than overwrite: {occupied}")


def emit(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    information = require_clean_tree(expected_branch=config["branch"])
    if not commit_reachable(config["scientific_import_boundary"]):
        raise ValueError("scientific import boundary is not reachable")
    source_records = _validate_source_artifacts(config)
    targets, figure_directory = _target_paths(config)
    _ensure_absent(targets, figure_directory)
    release_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    inventory, prefix = _collect_live_inventory(
        config, release_utc=release_utc,
        source_commit=information["code_commit"])
    environment = _environment_lock(
        config, source_records, release_utc=release_utc,
        source_commit=information["code_commit"])

    stage = Path(tempfile.mkdtemp(
        prefix="olmo_final_release_", dir=local_work()))
    try:
        staged = {
            name: stage / target.name for name, target in targets.items()
        }
        staged_figures = stage / "paper_figures"
        staged_figures.mkdir()
        state_source = _source_file(
            config["source_artifacts"]["state_report"])
        claims_source = _source_file(
            config["source_artifacts"]["claims_report"])
        atomic_text(staged["state_report"], _released_state_text(
            state_source.read_text(), config=config,
            source_commit=information["code_commit"],
            release_utc=release_utc,
            source_sha256=file_sha256(state_source)))
        atomic_text(staged["claims_report"], _released_claims_text(
            claims_source.read_text(), config=config,
            source_commit=information["code_commit"],
            release_utc=release_utc,
            source_sha256=file_sha256(claims_source)))
        atomic_json(staged["live_inventory"], inventory)
        atomic_json(staged["environment_lock"], environment)
        for key in ("paper_tex", "paper_pdf"):
            source_path = _source_file(
                config["source_artifacts"][key]["path"])
            shutil.copyfile(source_path, staged[key])
        figure_pairs = []
        for specification in config["source_artifacts"]["paper_figures"]:
            source_path = _source_file(specification["path"])
            stage_path = staged_figures / source_path.name
            shutil.copyfile(source_path, stage_path)
            target = figure_directory / source_path.name
            uri = (
                config["outputs"]["paper_figure_directory"].rstrip("/")
                + "/" + source_path.name)
            figure_pairs.append((stage_path, target, uri))

        artifact_specs = [
            ("state_report", "release-state-of-record"),
            ("claims_report", "release-claims-ledger"),
            ("live_inventory", "live-evidence-inventory"),
            ("environment_lock", "release-environment-lock"),
            ("paper_tex", "isolated-paper-source"),
            ("paper_pdf", "isolated-paper-pdf"),
        ]
        artifact_records = [
            _hash_record(
                staged[name], role=role, target_path=targets[name],
                logical_uri=config["outputs"][name])
            for name, role in artifact_specs
        ]
        artifact_records.extend(
            _hash_record(
                stage_path, role="isolated-paper-figure",
                target_path=target, logical_uri=uri)
            for stage_path, target, uri in figure_pairs
        )
        bundle_payload = {
            "schema_version": 1,
            "study_id": "jspace-olmo-lineage",
            "bundle_id": config["bundle_id"],
            "evidence_id": config["evidence_id"],
            "generated_utc": release_utc,
            "source_git": information,
            "scientific_import_boundary": config[
                "scientific_import_boundary"],
            "registry_prefix": prefix,
            "early_bundle": {
                "json": source_records["early_bundle_json"],
                "markdown": source_records["early_bundle_markdown"],
            },
            "evidence_categories": config["evidence_categories"],
            "completion": _completion_payload(),
            "claim_resolution": config["claim_resolution"],
            "phase5_queues": config["phase5_queues"],
            "forbidden_uses": config["forbidden_uses"],
            "artifacts": artifact_records,
            "importer_checks": [
                "source commit is reachable",
                "registry byte prefix matches",
                "all pre-release live outputs rehash",
                "all artifact paths remain inside the OLMo run root",
                "native tiers remain development/methods",
                "inventory v1 remains superseded by v2",
                "sentence 2 stays narrowed and sentence 4 stays pending",
                "H5/H6/O5 queues stay unopened",
                "forbidden confirmatory and causal uses remain attached",
            ],
            "claim_boundary": config["claim_boundary"],
        }
        atomic_text(
            staged["bundle_markdown"], _render_markdown(bundle_payload))
        artifact_records.append(_hash_record(
            staged["bundle_markdown"], role="human-readable-import-bundle",
            target_path=targets["bundle_markdown"],
            logical_uri=config["outputs"]["bundle_markdown"]))
        atomic_json(staged["bundle_json"], _envelope(bundle_payload))

        publish_pairs = [
            (staged[name], targets[name]) for name in (
                "state_report", "claims_report", "bundle_markdown",
                "live_inventory", "environment_lock", "paper_tex",
                "paper_pdf",
            )
        ]
        publish_pairs.extend(
            (stage_path, target) for stage_path, target, _ in figure_pairs)
        publish_pairs.append((staged["bundle_json"], targets["bundle_json"]))
        for source_path, target in publish_pairs:
            _atomic_copy(source_path, target)

        outputs = [
            targets["state_report"], targets["claims_report"],
            targets["bundle_json"], targets["bundle_markdown"],
            targets["live_inventory"], targets["environment_lock"],
            targets["paper_tex"], targets["paper_pdf"],
            *[target for _, target, _ in figure_pairs],
        ]
        event = create(
            config["evidence_id"], tier=config["tier"],
            what=("Final self-verifying OLMo parallel-phase import/restart "
                  "handoff; no new scientific cell."),
            command=(
                "python -m jspace_olmo_lineage.experiments.final_release "
                f"--config {_repo_path(config_path)} --emit"),
            outputs=outputs,
            inputs={
                "config_sha256": file_sha256(config_path),
                "registry_prefix_sha256": prefix["prefix_sha256"],
                "state_source_sha256": source_records[
                    "state_report"]["sha256"],
                "claims_source_sha256": source_records[
                    "claims_report"]["sha256"],
                "paper_tex_sha256": source_records["paper_tex"]["sha256"],
                "paper_pdf_sha256": source_records["paper_pdf"]["sha256"],
                "foundation_environment_lock_sha256": source_records[
                    "foundation_environment_lock"]["sha256"],
            },
            verdict="complete-first-release-boundary",
            phase4_service_ready=False,
            interventions_opened=False,
            scientific_outcomes_opened=False,
            sentence_2_status=config["claim_resolution"][
                "sentence_2"]["status"],
            sentence_4_status=config["claim_resolution"][
                "sentence_4"]["status"],
            claim_boundary=config["claim_boundary"],
        )
        verification = verify(config_path)
        return {
            "ok": True,
            "event": event,
            "verification": verification,
        }
    finally:
        shutil.rmtree(stage)


def verify(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    targets, figure_directory = _target_paths(config)
    bundle = _load_envelope(targets["bundle_json"])
    payload = bundle["payload"]
    if payload["bundle_id"] != config["bundle_id"]:
        raise ValueError("bundle ID drift")
    if payload["evidence_id"] != config["evidence_id"]:
        raise ValueError("bundle evidence ID drift")
    if payload["scientific_import_boundary"] != config[
            "scientific_import_boundary"]:
        raise ValueError("scientific import boundary drift")
    source_commit = payload["source_git"]["code_commit"]
    if not commit_reachable(source_commit):
        raise ValueError("bundle source commit is not reachable")
    _verify_registry_prefix(payload["registry_prefix"])

    artifact_paths = set()
    for row in payload["artifacts"]:
        path = Path(row["path"])
        artifact_paths.add(str(path))
        if not path.is_file():
            raise FileNotFoundError(path)
        if (file_sha256(path) != row["sha256"]
                or path.stat().st_size != int(row["bytes"])):
            raise ValueError(f"release artifact drift: {path}")
    inventory = _load_envelope(targets["live_inventory"])["payload"]
    environment = _load_envelope(targets["environment_lock"])["payload"]
    if inventory["registry_prefix"] != payload["registry_prefix"]:
        raise ValueError("inventory registry-prefix drift")
    live_ids = [row["evidence_id"] for row in inventory["live_evidence"]]
    _validate_partition(config["evidence_categories"], live_ids)
    if environment["constraint_verification"]["ok"] is not True:
        raise ValueError("stored environment constraint check failed")
    foundation = config["source_artifacts"]["foundation_environment_lock"]
    if environment["foundation_environment_lock"]["sha256"] != foundation[
            "sha256"]:
        raise ValueError("foundation environment-lock identity drift")
    if payload["claim_resolution"] != config["claim_resolution"]:
        raise ValueError("claim-resolution drift")
    if payload["phase5_queues"] != config["phase5_queues"]:
        raise ValueError("Phase 5 queue drift")
    if payload["forbidden_uses"] != config["forbidden_uses"]:
        raise ValueError("forbidden-use drift")

    state = targets["state_report"].read_text()
    claims = targets["claims_report"].read_text()
    if "- [x] final import/restart bundle emitted and registered." not in state:
        raise ValueError("release state does not close the final checklist")
    if "bundle is being assembled" in state:
        raise ValueError("release state retains an in-progress status")
    canonical_claims = _canonical_claim_text(claims)
    for sentence in config["claim_resolution"].values():
        if _canonical_claim_text(sentence["wording"]) not in canonical_claims:
            raise ValueError("released claims ledger loses licensed wording")

    event = resolve(config["evidence_id"])
    if not event["live"] or event["effective_tier"] != config["tier"]:
        raise ValueError("final-release event is not live methods evidence")
    effective = event["effective_metadata"]
    if effective.get("verdict") != "complete-first-release-boundary":
        raise ValueError("final-release verdict drift")
    if effective.get("interventions_opened") is not False:
        raise ValueError("final release improperly opens interventions")
    if effective["code_commit"] != source_commit:
        raise ValueError("final event/source commit drift")
    event_rows = effective.get("outputs") or []
    expected_paths = artifact_paths | {str(targets["bundle_json"])}
    if {row["path"] for row in event_rows} != expected_paths:
        raise ValueError("final event output set drift")
    if len(event_rows) != 13:
        raise ValueError("final event must register exactly 13 outputs")
    root = run_root(create=False).resolve()
    for row in event_rows:
        path = Path(row["path"])
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"final output escapes OLMo root: {path}") from error
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"final registered output hash drift: {path}")
    if not figure_directory.is_dir() or len(list(
            figure_directory.glob("*.pdf"))) != 5:
        raise ValueError("release paper-figure directory drift")
    live_verification = verify_live_evidence()
    expected = config["expected_registry_before_release"]
    boundary = _release_boundary_state(
        payload["registry_prefix"], config["evidence_id"])
    if (not live_verification["ok"]
            or boundary["n_live_events"] != int(
                expected["live_events"]) + 1
            or boundary["n_live_outputs"] != int(
                expected["live_outputs"]) + 13
            or boundary["post_boundary_restatements"]):
        raise ValueError("post-release live-evidence verification drift")
    return {
        "ok": True,
        "bundle_id": payload["bundle_id"],
        "evidence_id": event["evidence_id"],
        "source_commit": source_commit,
        "registry_prefix_sha256": payload[
            "registry_prefix"]["prefix_sha256"],
        "artifact_count": len(event_rows),
        "live_events": live_verification["n_live_events"],
        "live_outputs": live_verification["n_checked_outputs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--emit", action="store_true")
    action.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    result = emit(arguments.config) if arguments.emit else verify(
        arguments.config)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
