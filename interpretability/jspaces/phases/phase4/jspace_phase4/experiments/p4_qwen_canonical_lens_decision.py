"""Mechanical A1000 canonical-lens decision after all successor audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import yaml

from ..import_bundle import REPOSITORY, _repository_materialization
from ..manifests import atomic_json, file_sha256, object_sha256, require_clean_tree
from ..paths4 import metrics_dir
from ..provenance4 import Provenance4, write_result4
from ..registry4 import RegistryError, create, resolve
from .p4_qwen_multilens_functional_gate import ql_branch_from_gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _output(event: Mapping, name: str) -> Path:
    rows = [row for row in event["outputs"]
            if Path(row["path"]).name == name]
    if len(rows) != 1:
        raise RuntimeError(
            f"evidence {event['evidence_id']} lacks exactly one {name}")
    path = _repository_materialization(Path(rows[0]["path"]), REPOSITORY)
    if file_sha256(path) != rows[0]["sha256"]:
        raise RuntimeError(f"registered output hash mismatch: {name}")
    return path


def _payload(event: Mapping, name: str) -> tuple[dict, Path]:
    path = _output(event, name)
    envelope = json.loads(path.read_text())
    return envelope.get("payload", envelope), path


def decide(*, functional: Mapping, margin: Mapping, influence: Mapping,
           config: Mapping) -> dict:
    if functional.get("branch") != "PENDING_SELECTION_MARGIN_AUDIT":
        raise RuntimeError("functional gate is not pending margin audit")
    structural = functional["structural_gate"]
    if structural.get("status") != "verified-live":
        raise RuntimeError("canonical decision lacks verified structural gate")
    recomputed = ql_branch_from_gates(
        functional["functional_gates"],
        structural_stable=bool(structural["all_structural_gates_pass"]),
    )
    if recomputed != functional.get("branch_candidate"):
        raise RuntimeError("functional Q-L branch candidate does not reproduce")
    verdicts = margin["contract_verdict"]
    for name, expected in config["contract"][
            "selection_margin_required_verdicts"].items():
        if verdicts.get(name) is not expected:
            raise RuntimeError(f"selection-margin verdict failed: {name}")
    if margin.get("functional_branch_candidate") != recomputed:
        raise RuntimeError("selection-margin source branch does not match")
    influence_decision = influence["decision"]
    if influence_decision not in config["contract"][
            "influence_allowed_decisions"]:
        raise RuntimeError("retained-prompt influence decision is not licensed")
    if influence.get("prompt_retained_unconditionally") is not True:
        raise RuntimeError("prompt-323 influence attempted trimming/refit")
    action = dict(config["actions"][recomputed])
    amendment = config["contract"]["ql2_amendment"]
    amendment_path = _repository_materialization(
        Path(amendment["path"]), REPOSITORY)
    if file_sha256(amendment_path) != amendment["sha256"]:
        raise RuntimeError("prospective Q-L2 amendment hash mismatch")
    return {
        "schema_version": 1,
        "tier": config["tier"],
        "branch": recomputed,
        "action": action,
        "a1000_is_last_automatic_fit_size_escalation": True,
        "selection_margin_audit_complete": True,
        "selection_margin_stratum_counts": margin["stratum_counts"],
        "manual_lexical_review_required_rows": margin["lexical_audit"][
            "manual_review_required_rows"],
        "manual_lexical_review_is_nondecisional": True,
        "prompt323_influence_decision": influence_decision,
        "prompt323_retained": True,
        "ql2_amendment": {
            **dict(amendment),
            "activated": recomputed == amendment["activation_branch"],
            "discarded_unused": recomputed != amendment["activation_branch"],
        },
        "canonical_lens_nominated": action["canonical_lens"] is not None,
        "canonical_lens": action["canonical_lens"],
        "p4_p2_status": action["p4_p2_status"],
        "confirmatory_or_replication_outcomes_opened": False,
        "independent_review_complete": False,
        "pi_signoff_complete": False,
    }


def _markdown(result: Mapping, source_hashes: Mapping) -> str:
    action = result["action"]
    canonical = action["canonical_lens"] or "none"
    return "\n".join([
        "# Qwen A1000 canonical-lens decision",
        "",
        "**PHASE 4 DEVELOPMENT — NOT INDEPENDENT REVIEW OR PI SIGN-OFF**",
        "",
        f"Mechanical branch: **{result['branch']}**.",
        f"Canonical lens: **{canonical}**.",
        f"P4-P2 status: `{action['p4_p2_status']}`.",
        f"Scientific estimand: `{action['estimand']}`.",
        "",
        "The decision combines the registered structural, functional, "
        "all-position selection-margin, and retained prompt-323 influence "
        "audits. Prompt 323 remains in both fits regardless of its influence "
        "classification. No untouched outcome was opened.",
        "",
        "Source SHA-256 bindings:",
        "",
        *[f"- `{name}`: `{digest}`"
          for name, digest in sorted(source_hashes.items())],
        "",
        "Independent protocol review and PI sign-off remain pending.",
        "",
    ])


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("tier") != "phase4-development":
        raise RuntimeError("canonical-lens decision is development only")
    clean = require_clean_tree()
    try:
        existing = resolve(config["evidence_id"])
    except RegistryError as error:
        if "found 0" not in str(error):
            raise
    else:
        if not existing["live"]:
            raise RuntimeError("existing canonical decision is not live")
        for output in existing["outputs"]:
            if file_sha256(output["path"]) != output["sha256"]:
                raise RuntimeError("registered canonical-decision output drift")
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return

    events = {}
    for key, evidence_id in config["sources"].items():
        event = resolve(evidence_id)
        if not event["live"]:
            raise RuntimeError(f"canonical source is not live: {evidence_id}")
        events[key] = event
    functional, functional_path = _payload(
        events["functional_evidence_id"], "functional_gate_result.json")
    functional_manifest, functional_manifest_path = _payload(
        events["functional_evidence_id"], "input_manifest.json")
    margin, margin_path = _payload(
        events["selection_margin_evidence_id"],
        "selection_margin_result.json")
    influence, influence_path = _payload(
        events["retained_prompt_influence_evidence_id"],
        "influence_result.json")
    structural_path = _output(
        events["structural_evidence_id"], "convergence_result.json")
    a1000_outputs = {
        row["sha256"] for row in events["fit_evidence_id"]["outputs"]
        if Path(row["path"]).name == "qwen36-27b_jlens_drawA_n1000.pt"
    }
    if len(a1000_outputs) != 1:
        raise RuntimeError("registered A1000 fit lacks one final lens")
    a1000_sha256 = next(iter(a1000_outputs))
    if functional["structural_gate"].get("evidence_id") != config[
            "sources"]["structural_evidence_id"]:
        raise RuntimeError("functional result references another structural event")
    if functional_manifest["lenses"]["a1000"][
            "lens_sha256"] != a1000_sha256:
        raise RuntimeError("functional result is not bound to registered A1000")
    if influence.get("lens_hashes", {}).get("a1000") != a1000_sha256:
        raise RuntimeError("prompt influence is not bound to registered A1000")
    result = decide(
        functional=functional, margin=margin, influence=influence,
        config=config)
    source_hashes = {
        "a1000_lens": a1000_sha256,
        "structural_result": file_sha256(structural_path),
        "functional_result": file_sha256(functional_path),
        "functional_manifest": file_sha256(functional_manifest_path),
        "selection_margin_result": file_sha256(margin_path),
        "prompt323_influence_result": file_sha256(influence_path),
        "ql2_amendment": config["contract"]["ql2_amendment"]["sha256"],
    }
    result["source_hashes"] = source_hashes
    output_dir = (
        metrics_dir(config["slug"]) / "canonical_lens_decision"
        / config["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "canonical_lens_decision.json"
    markdown_path = output_dir / "CANONICAL_LENS_DECISION.md"
    manifest_path = output_dir / "input_manifest.json"
    command = (
        "python -m "
        "jspace_phase4.experiments.p4_qwen_canonical_lens_decision "
        f"--config {arguments.config}")
    manifest_payload = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "source_hashes": source_hashes,
        "contract": dict(config["contract"]),
        "actions": dict(config["actions"]),
    }
    manifest = {
        "schema_version": 1, "payload": manifest_payload,
        "payload_sha256": object_sha256(manifest_payload),
    }
    atomic_json(manifest_path, manifest)
    inputs = {**source_hashes, "input_manifest": manifest["payload_sha256"]}
    write_result4(
        result, result_path,
        Provenance4(
            evidence_id=config["evidence_id"], tier=config["tier"],
            command=command, inputs=inputs,
            input_manifest_sha256=manifest["payload_sha256"],
            seed_contract="deterministic frozen Q-L1--Q-L5 truth table",
        ))
    markdown_path.write_text(_markdown(result, source_hashes))
    create(
        config["evidence_id"], tier=config["tier"],
        what=config["registry_what"], command=command,
        outputs=[result_path, manifest_path, markdown_path], inputs=inputs)
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "branch": result["branch"],
        "canonical_lens": result["canonical_lens"],
        "p4_p2_status": result["p4_p2_status"],
        "result": str(result_path),
    }, indent=1))


if __name__ == "__main__":
    main()
