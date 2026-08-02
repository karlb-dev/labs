"""Register a validated isolated side-track bundle in Phase 4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..import_bundle import validate_import_bundle
from ..manifests import file_sha256, require_clean_tree
from ..registry4 import EVENTS, import_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--validation", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    require_clean_tree()
    bundle_path = Path(arguments.bundle)
    validation_path = Path(arguments.validation)
    recorded = json.loads(validation_path.read_text())
    current = validate_import_bundle(
        bundle_path, main_events_path=EVENTS,
        allow_existing_target=False)
    if current != recorded:
        raise RuntimeError(
            "side-bundle validation file does not match a fresh validation")
    # Preserve the exact registry bytes used for validation as a live import
    # output. A side branch can legitimately append later events, so a
    # normalized early bundle points at an immutable Git-stored snapshot.
    outputs = [
        bundle_path,
        validation_path,
        Path(current["source_registry"]["path"]),
    ]
    outputs.extend(Path(row["path"]) for row in current["outputs"])
    event = import_evidence(
        current["target_import_evidence_id"],
        tier="side-development-import",
        what=(
            "Hash-verified development/methods import bundle from isolated "
            f"side study {current['source_study']}; no confirmatory or "
            "replication intervention outcome is included."),
        source_study=current["source_study"],
        source_evidence_id=str(current["bundle_id"]),
        source_commit=current["source_commit"],
        source_registry=current["source_registry"]["path"],
        source_outputs=outputs,
        source_branch=current["source_branch"],
        source_evidence_ids=[
            row["evidence_id"] for row in current["selected_events"]],
        source_selected_event_ids_sha256=current[
            "selected_event_ids_sha256"],
        source_output_inventory_sha256=current[
            "output_inventory_sha256"],
        bundle_sha256=file_sha256(bundle_path),
        bundle_payload_sha256=current["bundle_payload_sha256"],
        no_confirmatory_or_replication_intervention_outcome=True,
    )
    print(json.dumps({
        "event": event["event"],
        "evidence_id": event["evidence_id"],
        "source_study": current["source_study"],
        "source_commit": current["source_commit"],
        "n_source_events": len(current["selected_events"]),
        "n_source_outputs": len(current["outputs"]),
    }, indent=1))


if __name__ == "__main__":
    main()
