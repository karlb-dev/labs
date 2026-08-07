"""Render, publish, and verify the partial-safe Gemma Study-2 handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

def _find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError("cannot locate git repository root")


import yaml

from jspace_gemma.manifests import (
    atomic_json,
    atomic_text,
    file_sha256,
    git_info,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import _REPO_PATH_ALIASES, resolve_uri, run_root
from jspace_gemma.registry import EVENTS, create, read_events, resolve, resolve_all
from jspace_gemma.repro import _repository_materialization, verify_live_evidence

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = _find_repo_root()
NATIVE_TIERS = {"development", "methods"}
DECISION_KEYS = (
    "backend_parity_pass",
    "methods_blocker",
    "terminal_status",
    "route",
    "selected_branch",
    "calibration_route",
    "all_five_layers_local_tangent_mismatch_licensed",
    "model_compute_performed",
)


def load_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("Study-2 release config must be a mapping")
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
        raise TypeError(f"expected payload envelope: {path}")
    if value.get("payload_sha256") != object_sha256(value["payload"]):
        raise ValueError(f"payload-envelope hash drift: {path}")
    return value


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return _repository_materialization(value)


def _output_rows(record: Mapping) -> list[dict]:
    field = "source_outputs" if record["event"] == "evidence_imported" else "outputs"
    rows = []
    for output in record.get(field, []) or []:
        materialized = _repository_materialization(Path(output["path"]))
        if not materialized.is_file() \
                or file_sha256(materialized) != output["sha256"]:
            raise ValueError(
                f"live output hash drift: {record['evidence_id']}: "
                f"{output['path']}")
        rows.append(
            {
                "path": str(output["path"]),
                "sha256": output["sha256"],
                "bytes": materialized.stat().st_size,
            }
        )
    return rows


def registry_prefix_record(path: str | Path = EVENTS) -> dict:
    source = Path(path)
    raw = source.read_bytes()
    events = read_events(source)
    records = resolve_all(path=source)
    live = [record for record in records if record["live"]]
    if not events:
        raise ValueError("cannot release an empty evidence registry")
    try:
        label = str(source.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        label = str(source)
    return {
        "path": label,
        "prefix_bytes": len(raw),
        "prefix_sha256": hashlib.sha256(raw).hexdigest(),
        "line_count": len(events),
        "origin_events": len(records),
        "live_events": len(live),
        "live_outputs": sum(len(_output_rows(record)) for record in live),
        "through_evidence_id": events[-1]["evidence_id"],
        "through_event": events[-1]["event"],
        "note": "The prefix excludes the Study-2 release event itself.",
    }


def verify_registry_prefix(record: Mapping, path: str | Path = EVENTS) -> dict:
    raw = Path(path).read_bytes()
    length = int(record["prefix_bytes"])
    if len(raw) < length:
        raise ValueError("registry is shorter than the released prefix")
    prefix = raw[:length]
    actual = hashlib.sha256(prefix).hexdigest()
    if actual != record["prefix_sha256"]:
        raise ValueError("released registry prefix hash drift")
    lines = [line for line in prefix.decode().splitlines() if line.strip()]
    if len(lines) != int(record["line_count"]):
        raise ValueError("released registry prefix line-count drift")
    last = json.loads(lines[-1])
    if (
        last.get("evidence_id") != record["through_evidence_id"]
        or last.get("event") != record["through_event"]
    ):
        raise ValueError("released registry-prefix endpoint drift")
    return {
        "ok": True,
        "prefix_bytes": length,
        "prefix_sha256": actual,
        "current_registry_bytes": len(raw),
    }


def _validate_expected_prefix(config: Mapping, prefix: Mapping) -> None:
    expected = config["expected_registry_before_release"]
    for key in ("line_count", "origin_events", "live_events", "live_outputs"):
        if int(prefix[key]) != int(expected[key]):
            raise ValueError(
                f"pre-release registry {key} drift: {prefix[key]} != {expected[key]}"
            )
    if prefix["through_evidence_id"] != expected["through_evidence_id"]:
        raise ValueError("pre-release registry endpoint drift")


def source_artifact_records(config: Mapping) -> list[dict]:
    records = []
    sources = set()
    targets = set()
    for specification in config["source_artifacts"]:
        source = _repo_path(specification["repo_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = file_sha256(source)
        if actual != specification["sha256"]:
            raise ValueError(f"release source hash drift: {source}")
        if str(source) in sources:
            raise ValueError(f"duplicate release source: {source}")
        if specification["target_uri"] in targets:
            raise ValueError(f"duplicate release target: {specification['target_uri']}")
        sources.add(str(source))
        targets.add(specification["target_uri"])
        records.append(
            {
                "role": specification["role"],
                "repo_path": specification["repo_path"],
                "target_uri": specification["target_uri"],
                "sha256": actual,
                "bytes": source.stat().st_size,
            }
        )
    return records


def _flatten_admissions(categories: Mapping[str, Sequence[str]]) -> list[str]:
    values = [item for rows in categories.values() for item in rows]
    if len(values) != len(set(values)):
        raise ValueError("admitted evidence categories contain duplicates")
    return values


def admitted_evidence_records(config: Mapping) -> list[dict]:
    categories = config["admitted_evidence"]
    identifiers = _flatten_admissions(categories)
    category_by_id = {
        evidence_id: category
        for category, rows in categories.items()
        for evidence_id in rows
    }
    records = []
    for evidence_id in identifiers:
        record = resolve(evidence_id)
        if not record["live"]:
            raise ValueError(f"admitted evidence is not live: {evidence_id}")
        tier = record["effective_tier"]
        if tier not in NATIVE_TIERS:
            raise ValueError(f"admitted evidence has invalid tier: {evidence_id}")
        records.append(
            {
                "evidence_id": evidence_id,
                "category": category_by_id[evidence_id],
                "origin_event": record["event"],
                "event_utc": record["event_utc"],
                "tier": tier,
                "code_commit": record.get("code_commit"),
                "what": record.get("what"),
                "decisions": {
                    key: record[key] for key in DECISION_KEYS if key in record
                },
                "status_events": record["status_events"],
                "outputs": _output_rows(record),
            }
        )
    return records


def _render_markdown(envelope: Mapping) -> str:
    payload = envelope["payload"]
    prefix = payload["registry_prefix"]
    result = payload["result_summary"]
    lines = [
        "# Gemma transport Study-2 import bundle",
        "",
        f"Bundle ID: `{payload['bundle_id']}`  ",
        f"Evidence ID: `{payload['evidence_id']}`  ",
        f"Generated: `{payload['generated_utc']}`  ",
        f"Clean render commit: `{payload['source_git']['code_commit']}`  ",
        f"JSON payload SHA-256: `{envelope['payload_sha256']}`",
        "",
        "## Import boundary",
        "",
        (
            f"The exact pre-release registry prefix is "
            f"{prefix['prefix_bytes']:,} bytes / {prefix['line_count']} events, "
            f"SHA-256 `{prefix['prefix_sha256']}`. It ends at "
            f"`{prefix['through_evidence_id']}` and contains "
            f"{prefix['origin_events']} origins, {prefix['live_events']} live "
            f"events, and {prefix['live_outputs']} live outputs."
        ),
        "",
        (
            "This bundle is partial-safe: mandatory G2.1 and G2.2 are complete; "
            "mechanism, intervention, and confirmatory items remain explicitly "
            "unopened. Development/methods tiers are preserved."
        ),
        "",
        "## Scientific disposition",
        "",
        f"- Calibration route: `{result['calibration_route']}`.",
        (
            f"- Pooled all-frozen-batches ceiling: "
            f"`{result['pooled_ceiling']}` from {result['full_backend_pairs']} "
            "full pairs."
        ),
        (
            f"- Historical all-slot relative error: "
            f"`{result['historical_all_slot_relative_error_exact']}`; selected slot "
            "remains bit-identical."
        ),
        f"- License route: `{result['license_branch']}`; no G2.2 model compute.",
        (
            "- L22/L30/L37/L44/L52 remain `local_tangent_mismatch`, now a "
            "closed methods result over the tested finite-scale scope."
        ),
        "- No mechanism, workspace, intervention, or confirmatory claim opens.",
        "",
        "## Partial and terminal statuses",
        "",
    ]
    for key, value in payload["partial_statuses"].items():
        lines.append(f"- `{key}`: `{value}`.")
    lines.extend(["", "## Admitted evidence", ""])
    for record in payload["admitted_evidence"]:
        lines.append(
            f"- `{record['evidence_id']}` — {record['tier']} / "
            f"{record['category']}; {len(record['outputs'])} outputs."
        )
    lines.extend(
        [
            "",
            "## Release artifacts",
            "",
            "| Role | Repo source | Drive target | Bytes | SHA-256 |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in payload["release_artifacts"]:
        lines.append(
            f"| {row['role']} | `{row['repo_path']}` | "
            f"`{row['target_uri']}` | {row['bytes']} | `{row['sha256']}` |"
        )
    lines.extend(["", "## Forbidden uses", ""])
    for value in payload["forbidden_uses"]:
        lines.append(f"- {value}.")
    lines.extend(["", "## Importer checks", ""])
    for value in payload["importer_checks"]:
        lines.append(f"- [ ] {value}.")
    lines.extend(["", "## Claim boundary", "", payload["claim_boundary"], ""])
    return "\n".join(lines)


def _repo_outputs(config: Mapping) -> dict[str, Path]:
    outputs = config["outputs"]
    return {
        "bundle_json": _repo_path(outputs["repo_bundle_json"]),
        "bundle_markdown": _repo_path(outputs["repo_bundle_markdown"]),
        "registry_prefix": _repo_path(outputs["repo_registry_prefix"]),
    }


REORG_BOUNDARY_TAG = "pre-jspaces-reorg-v1"


def _pre_reorg_relative(relative: Path) -> Path:
    """Map a current repo-relative path back to its pre-reorg location."""
    text = relative.as_posix()
    for old, new in _REPO_PATH_ALIASES:
        if old.startswith("interpretability/") and text.startswith(new):
            return Path(old + text[len(new):])
    return relative


def _generation_protocol_unchanged(generation_commit: str, config_path: Path) -> bool:
    module = Path(__file__).resolve().relative_to(REPO_ROOT)
    config = config_path.resolve().relative_to(REPO_ROOT)
    ancestor = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", generation_commit, "HEAD"],
        check=False,
    ).returncode == 0
    generation_is_post_reorg = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor",
         REORG_BOUNDARY_TAG, generation_commit],
        check=False,
    ).returncode == 0
    if generation_is_post_reorg:
        unchanged = subprocess.run(
            [
                "git", "-C", str(REPO_ROOT), "diff", "--quiet", generation_commit,
                "HEAD", "--", str(module), str(config),
            ],
            check=False,
        ).returncode == 0
        return ancestor and unchanged
    # Pre-reorg bundle: the protocol files moved (and the renderer gained
    # historical-path resolution) at the reorg boundary. Require them
    # untouched from generation to the boundary tag at their old paths, and
    # the frozen config byte-identical on the current tree.
    old_module = _pre_reorg_relative(module)
    old_config = _pre_reorg_relative(config)
    unchanged_to_boundary = subprocess.run(
        [
            "git", "-C", str(REPO_ROOT), "diff", "--quiet", generation_commit,
            REORG_BOUNDARY_TAG, "--", str(old_module), str(old_config),
        ],
        check=False,
    ).returncode == 0
    try:
        frozen_config = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show",
             f"{generation_commit}:{old_config.as_posix()}"],
            stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return False
    config_exact = frozen_config == config_path.read_bytes()
    return ancestor and unchanged_to_boundary and config_exact


def render(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    information = require_clean_tree(branch=config["branch"])
    outputs = _repo_outputs(config)
    occupied = [str(path) for path in outputs.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"immutable release source already exists: {occupied}")
    prefix = registry_prefix_record()
    _validate_expected_prefix(config, prefix)
    verification = verify_live_evidence()
    if (
        not verification["ok"]
        or verification["n_live_events"] != prefix["live_events"]
        or verification["n_checked_outputs"] != prefix["live_outputs"]
    ):
        raise ValueError("pre-release live-evidence verification drift")
    payload = {
        "schema_version": 1,
        "study_id": "jspace-gemma-transport-study2",
        "bundle_id": config["bundle_id"],
        "evidence_id": config["evidence_id"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_git": information,
        "shared_parent": config["shared_parent"],
        "native_tiers": sorted(NATIVE_TIERS),
        "registry_prefix": prefix,
        "registry_snapshot": {
            "repo_path": config["outputs"]["repo_registry_prefix"],
            "target_uri": config["outputs"]["drive_registry_prefix"],
            "sha256": prefix["prefix_sha256"],
            "bytes": prefix["prefix_bytes"],
        },
        "pre_release_live_verification": {
            "ok": True,
            "n_live_events": verification["n_live_events"],
            "n_checked_outputs": verification["n_checked_outputs"],
        },
        "admitted_evidence_categories": config["admitted_evidence"],
        "admitted_evidence": admitted_evidence_records(config),
        "partial_statuses": config["partial_statuses"],
        "result_summary": config["result_summary"],
        "release_artifacts": source_artifact_records(config),
        "forbidden_uses": config["forbidden_uses"],
        "importer_checks": [
            "source commit is reachable",
            "registry snapshot and current registry share the exact released prefix",
            "all admitted events remain live at their native tier",
            "all admitted output paths, sizes, and SHA-256 values verify",
            "all V2 reports, TeX/PDF bytes, figure, designs, and configs verify",
            "Study-1 failed-gate evidence remains immutable",
            "G2.2 remains a no-recompute license decision",
            "no mechanism, workspace, intervention, or confirmatory tier is inferred",
        ],
        "claim_boundary": config["claim_boundary"],
    }
    envelope = _envelope(payload)
    outputs["registry_prefix"].parent.mkdir(parents=True, exist_ok=True)
    outputs["registry_prefix"].write_bytes(EVENTS.read_bytes())
    atomic_json(outputs["bundle_json"], envelope)
    atomic_text(outputs["bundle_markdown"], _render_markdown(envelope))
    return {
        "status": "rendered-unregistered",
        "bundle_id": config["bundle_id"],
        "payload_sha256": envelope["payload_sha256"],
        "outputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in outputs.items()
        },
    }


def verify_bundle_source(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    outputs = _repo_outputs(config)
    envelope = _load_envelope(outputs["bundle_json"])
    payload = envelope["payload"]
    if payload["bundle_id"] != config["bundle_id"]:
        raise ValueError("bundle ID drift")
    if payload["evidence_id"] != config["evidence_id"]:
        raise ValueError("bundle evidence ID drift")
    if payload["shared_parent"] != config["shared_parent"]:
        raise ValueError("shared-parent drift")
    source_commit = payload["source_git"]["code_commit"]
    if not _generation_protocol_unchanged(source_commit, config_path):
        raise ValueError("release protocol is unreachable or changed after rendering")
    verify_registry_prefix(payload["registry_prefix"])
    snapshot = outputs["registry_prefix"]
    if (
        file_sha256(snapshot) != payload["registry_snapshot"]["sha256"]
        or snapshot.stat().st_size != int(payload["registry_snapshot"]["bytes"])
    ):
        raise ValueError("registry snapshot drift")
    if payload["release_artifacts"] != source_artifact_records(config):
        raise ValueError("release source artifact drift")
    if payload["admitted_evidence"] != admitted_evidence_records(config):
        raise ValueError("admitted evidence drift")
    if payload["partial_statuses"] != config["partial_statuses"]:
        raise ValueError("partial-status drift")
    if payload["result_summary"] != config["result_summary"]:
        raise ValueError("result-summary drift")
    if payload["forbidden_uses"] != config["forbidden_uses"]:
        raise ValueError("forbidden-use drift")
    markdown = outputs["bundle_markdown"].read_text()
    for required in (
        "no G2.2 model compute",
        "Study-1 failed-gate evidence remains immutable",
        "No mechanism, workspace, intervention, or confirmatory claim opens",
    ):
        if required not in markdown:
            raise ValueError(f"bundle Markdown loses boundary: {required}")
    return {
        "ok": True,
        "bundle_id": payload["bundle_id"],
        "source_commit": source_commit,
        "payload_sha256": envelope["payload_sha256"],
        "registry_prefix_sha256": payload["registry_prefix"]["prefix_sha256"],
        "admitted_events": len(payload["admitted_evidence"]),
        "release_artifacts": len(payload["release_artifacts"]),
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            destination.is_file()
            and file_sha256(destination) == file_sha256(source)
            and destination.stat().st_size == source.stat().st_size
        ):
            return
        raise FileExistsError(f"immutable release target differs: {destination}")
    temporary = destination.with_name(destination.name + f".tmp{os.getpid()}")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)


def _require_study2_root(config: Mapping) -> None:
    root = run_root(create=False)
    if root.name != config["expected_run_root_name"]:
        raise ValueError(
            f"expected Study-2 root {config['expected_run_root_name']}, got {root}"
        )


def _publish_pairs(config: Mapping) -> list[tuple[Path, Path]]:
    pairs = [
        (_repo_path(row["repo_path"]), resolve_uri(row["target_uri"], must_exist=False))
        for row in config["source_artifacts"]
    ]
    repo = _repo_outputs(config)
    pairs.extend(
        [
            (
                repo["registry_prefix"],
                resolve_uri(config["outputs"]["drive_registry_prefix"], must_exist=False),
            ),
            (
                repo["bundle_json"],
                resolve_uri(config["outputs"]["drive_bundle_json"], must_exist=False),
            ),
            (
                repo["bundle_markdown"],
                resolve_uri(config["outputs"]["drive_bundle_markdown"], must_exist=False),
            ),
        ]
    )
    targets = [str(target) for _, target in pairs]
    if len(targets) != len(set(targets)):
        raise ValueError("release publish targets are not unique")
    return pairs


def publish(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    require_clean_tree(branch=config["branch"])
    _require_study2_root(config)
    verification = verify_bundle_source(config_path)
    pairs = _publish_pairs(config)
    for source, target in pairs:
        _atomic_copy(source, target)
    repo_outputs = list(_repo_outputs(config).values())
    drive_outputs = [target for _, target in pairs]
    output_paths = [*repo_outputs, *drive_outputs]
    if len({str(path) for path in output_paths}) != len(output_paths):
        raise ValueError("release event output paths are not unique")
    bundle_path = _repo_outputs(config)["bundle_json"]
    bundle = _load_envelope(bundle_path)
    payload = bundle["payload"]
    event = create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "Partial-safe terminal Gemma Study-2 import bundle: target-blind "
            "G2.1 calibration and mechanical G2.2 license complete; mechanism, "
            "workspace, intervention, and confirmatory claims remain unopened."
        ),
        command=(
            "python -m jspace_gemma.experiments.gm2_study2_release "
            f"--config {config_path} --publish"
        ),
        outputs=output_paths,
        inputs={
            "config": file_sha256(config_path),
            "bundle_json": file_sha256(bundle_path),
            "bundle_payload": bundle["payload_sha256"],
            "registry_prefix": payload["registry_prefix"]["prefix_sha256"],
            **{
                f"artifact:{row['role']}": row["sha256"]
                for row in payload["release_artifacts"]
            },
        },
        verdict="complete-mandatory-partial-conditional",
        mandatory_stages_complete=True,
        partial_bundle=True,
        interventions_opened=False,
        calibration_route=config["result_summary"]["calibration_route"],
        selected_branch=config["result_summary"]["license_branch"],
        five_layer_classifier_licensed=True,
        mechanism_claim_opened=False,
        workspace_claim_opened=False,
        confirmatory_cell_opened=False,
        claim_boundary=config["claim_boundary"],
    )
    return {
        "ok": True,
        "event": event,
        "source_verification": verification,
        "post_publish": verify_published(config_path),
    }


def verify_published(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    _require_study2_root(config)
    source = verify_bundle_source(config_path)
    pairs = _publish_pairs(config)
    for source_path, target in pairs:
        if not target.is_file():
            raise FileNotFoundError(target)
        if (
            file_sha256(target) != file_sha256(source_path)
            or target.stat().st_size != source_path.stat().st_size
        ):
            raise ValueError(f"published release artifact drift: {target}")
    event = resolve(config["evidence_id"])
    if not event["live"] or event["effective_tier"] != config["tier"]:
        raise ValueError("Study-2 release event is not live methods evidence")
    if event.get("verdict") != "complete-mandatory-partial-conditional":
        raise ValueError("Study-2 release verdict drift")
    if event.get("interventions_opened") is not False:
        raise ValueError("Study-2 release improperly opens an intervention")
    expected_paths = {str(path) for path in _repo_outputs(config).values()} | {
        str(target) for _, target in pairs
    }
    rows = event.get("outputs") or []
    if {row["path"] for row in rows} != expected_paths:
        raise ValueError("Study-2 release event output set drift")
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise ValueError(f"registered Study-2 release output drift: {path}")
    outputs = _repo_outputs(config)
    return {
        "ok": True,
        "bundle_id": config["bundle_id"],
        "evidence_id": config["evidence_id"],
        "release_event_commit": event["code_commit"],
        "bundle_source_commit": source["source_commit"],
        "bundle_json_sha256": file_sha256(outputs["bundle_json"]),
        "bundle_markdown_sha256": file_sha256(outputs["bundle_markdown"]),
        "registry_prefix_sha256": source["registry_prefix_sha256"],
        "registered_outputs": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--render", action="store_true")
    action.add_argument("--verify-source", action="store_true")
    action.add_argument("--publish", action="store_true")
    action.add_argument("--verify-published", action="store_true")
    arguments = parser.parse_args()
    if arguments.render:
        result = render(arguments.config)
    elif arguments.verify_source:
        result = verify_bundle_source(arguments.config)
    elif arguments.publish:
        result = publish(arguments.config)
    else:
        result = verify_published(arguments.config)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
