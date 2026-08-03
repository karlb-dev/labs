"""Render and register the paper-facing H6 figure from joint evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..manifests import file_sha256, object_sha256, require_clean_tree
from ..paths import figures_dir
from ..registry import RegistryError, create, resolve
from .stage_wedge import configure_run_root
from .transport_validation_analysis import EVIDENCE_ID as JOINT_EVIDENCE_ID
from .transport_validation_analysis import make_figure

EVIDENCE_ID = "ol2-transport-validation-figure-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def registered_replay() -> dict | None:
    try:
        event = resolve(EVIDENCE_ID)
    except RegistryError:
        return None
    for row in event["outputs"]:
        path = Path(row["path"])
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"registered H6 paper figure drift: {path}")
    return {"already_registered": True, "n_outputs_verified": len(event["outputs"])}


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    configure_run_root(config)
    clean = require_clean_tree(expected_branch=config["branch"])
    replay = registered_replay()
    if replay is not None:
        print(json.dumps(replay, indent=1))
        return
    joint = resolve(JOINT_EVIDENCE_ID)
    if not joint["live"]:
        raise RuntimeError("joint H6 evidence is not live")
    result_rows = [
        row for row in joint["outputs"]
        if Path(row["path"]).name == "transport_joint_result.json"
    ]
    if len(result_rows) != 1:
        raise RuntimeError("joint H6 result output is ambiguous")
    source = result_rows[0]
    source_path = Path(source["path"])
    if not source_path.is_file() or file_sha256(source_path) != source["sha256"]:
        raise RuntimeError("joint H6 result hash mismatch")
    envelope = json.loads(source_path.read_text())
    if object_sha256(envelope["payload"]) != envelope["payload_sha256"]:
        raise RuntimeError("joint H6 result payload mismatch")
    payload = envelope["payload"]
    png_path = figures_dir() / "ol2_transport_validation_joint_paper.png"
    pdf_path = figures_dir() / "ol2_transport_validation_joint_paper.pdf"
    make_figure(
        payload["checkpoint_summaries"],
        payload["dose_audit"],
        config,
        png_path,
        pdf_path,
    )
    command = (
        "python -m jspace_olmo_lineage.experiments.transport_validation_figure "
        f"--config {config_path}"
    )
    event = create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "Paper-facing PNG/PDF rendering of the registered joint H6 passage "
            "table; no new scientific analysis."
        ),
        command=command,
        outputs=[png_path, pdf_path],
        inputs={
            "joint_evidence_id": JOINT_EVIDENCE_ID,
            "joint_result_sha256": source["sha256"],
            "config_sha256": file_sha256(config_path),
        },
        scientific_result_changed=False,
        source_code_commit=clean["code_commit"],
    )
    print(json.dumps(event, indent=1))


if __name__ == "__main__":
    main()
