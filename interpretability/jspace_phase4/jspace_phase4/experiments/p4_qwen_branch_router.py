"""Read-only router for the frozen post-A500 Qwen continuation branch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import yaml

from ..manifests import file_sha256, object_sha256
from ..registry4 import resolve


ROUTES = {
    "A": "draw_b_n120",
    "B": "draw_a_n1000",
    "C": "draw_b_n120",
}


def route_from_payload(payload: Mapping,
                       interpretations: Mapping[str, str]) -> dict:
    branch = str(payload.get("branch", ""))
    if branch == "PENDING_STRUCTURAL":
        raise RuntimeError("A500 branch remains pending structural evidence")
    if branch not in ROUTES:
        raise RuntimeError(f"unknown frozen A500 branch: {branch!r}")
    expected = str(interpretations[branch])
    actual = str(payload.get("branch_interpretation", ""))
    if actual != expected:
        raise RuntimeError("registered branch interpretation drift")
    return {
        "branch": branch,
        "branch_interpretation": actual,
        "continuation": ROUTES[branch],
    }


def registered_route(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    evidence_id = str(config["evidence_id"])
    event = resolve(evidence_id)
    if not event["live"]:
        raise RuntimeError("A500 functional gate evidence is not live")
    candidates = [
        row for row in event["outputs"]
        if Path(row["path"]).name == "functional_gate_result.json"
    ]
    if len(candidates) != 1:
        raise RuntimeError("A500 functional event lacks one result envelope")
    registered = candidates[0]
    result_path = Path(registered["path"])
    if file_sha256(result_path) != registered["sha256"]:
        raise RuntimeError("A500 functional result output hash drift")
    envelope = json.loads(result_path.read_text())
    if envelope.get("provenance", {}).get("evidence_id") != evidence_id:
        raise RuntimeError("A500 functional result provenance drift")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("A500 functional result lacks a payload")
    if object_sha256(payload) != envelope.get("payload_sha256"):
        raise RuntimeError("A500 functional result payload hash drift")
    route = route_from_payload(
        payload, config["analysis"]["branch_interpretations"])
    return {
        "schema_version": 1,
        "evidence_id": evidence_id,
        "result_path": str(result_path),
        "result_sha256": registered["sha256"],
        **route,
        "decision_boundary": (
            "Mechanical application of the prospectively frozen A/B/C map; "
            "no outcome reinterpretation or canonical-lens nomination."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    print(json.dumps(registered_route(Path(arguments.config)), indent=1))


if __name__ == "__main__":
    main()
