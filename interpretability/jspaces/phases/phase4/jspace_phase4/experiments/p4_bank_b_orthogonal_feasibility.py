"""One-shot consumed-development answer-orthogonal bridge feasibility.

The geometry stage never evaluates a model response.  The outcome stage is
available only after that geometry is registered and an independent reviewer
has approved the final bound producer.  No Bank-B candidate, confirmatory, or
replication row is ever loaded by this module.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jspace_part2.dictionaries import effective_gain
from jspace_phase3.bank import FactBundle, load_bank
from jspace_phase3.experiments.p3_bridge_swap_endpoint_audit import (
    boundary_generation_category,
    greedy_from_prefill,
    piece_ids,
)
from jspace_phase3.scoring import DEFAULT_SPEC, ScoringSession

from ..bank_b_orthogonal_analysis import analyze_orthogonal_outcomes
from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..orthogonal_bridge import (
    OrthogonalBridgeAblator,
    OrthogonalBridgeGeometry,
    geometry_gate,
    geometry_match_gate,
    orthogonal_bridge_direction,
    partial_j_dictionary_rows,
    select_unrelated_geometry_match,
    stable_random_answer_orthogonal_direction,
)
from ..paths4 import (
    figures_dir,
    materialize_local_file,
    metrics_dir,
    resolve_uri,
)
from ..provenance4 import Provenance4, write_result4
from ..registry4 import RegistryError, create, resolve
from ..seeds import SEED_CONTRACT, stable_seed
from .p4_qwen_lens_structural_stability import load_lens_checkpoint
from .p4_qwen_nested_lens_fit import (
    model_reference,
    qwen_fused_kernel_contract,
    verify_model_fused_bindings,
    verify_package_versions,
    verify_snapshot,
)


class OrthogonalExecutionBlocked(RuntimeError):
    """A prospective governance gate has not licensed execution."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--geometry", action="store_true")
    action.add_argument("--run", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _load_envelope(path: Path) -> tuple[dict, dict]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"result envelope lacks payload: {path}")
    if envelope.get("payload_sha256") != object_sha256(payload):
        raise RuntimeError(f"result payload hash drift: {path}")
    return envelope, payload


def _registered_basename(evidence_id: str, basename: str,
                         expected_sha256: str | None = None) -> tuple[
                             dict, Path, str]:
    event = resolve(evidence_id)
    if not event["live"]:
        raise RuntimeError(f"registered evidence is not live: {evidence_id}")
    matches = [
        row for row in event.get("outputs", [])
        if Path(row["path"]).name == basename
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{evidence_id} lacks exactly one output named {basename}")
    path = Path(matches[0]["path"])
    digest = file_sha256(path)
    if digest != matches[0]["sha256"]:
        raise RuntimeError(f"registered output hash drift: {path}")
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"expected output hash mismatch: {path}")
    return event, path, digest


def _registered_output_check(evidence_id: str) -> dict | None:
    try:
        event = resolve(evidence_id)
    except RegistryError as error:
        if "found 0" in str(error):
            return None
        raise
    if not event["live"]:
        raise RuntimeError(f"existing evidence is not live: {evidence_id}")
    for output in event.get("outputs", []):
        if file_sha256(output["path"]) != output["sha256"]:
            raise RuntimeError(
                f"registered output drift for {evidence_id}: "
                f"{output['path']}")
    return event


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef"
                                   for character in text)


def canonical_lens_binding(config: Mapping, *, require_bound: bool) -> dict:
    specification = config["canonical_lens"]
    decision_digest = str(specification["decision_result_sha256"])
    lens_digest = str(specification["lens_sha256"])
    placeholders = [value for value in (decision_digest, lens_digest)
                    if not _is_sha256(value)]
    if placeholders:
        if require_bound:
            raise OrthogonalExecutionBlocked(
                "canonical A1000 decision/lens hashes are not bound")
        return {"bound": False, "placeholders": placeholders}
    decision_event, decision_path, observed = _registered_basename(
        specification["decision_evidence_id"],
        "canonical_lens_decision.json", decision_digest)
    _, decision = _load_envelope(decision_path)
    branch = str(decision.get("branch"))
    if branch not in specification["permitted_branches"]:
        raise OrthogonalExecutionBlocked(
            f"canonical branch {branch} blocks the Bank-B rescue")
    if decision.get("canonical_lens") != specification[
            "required_canonical_lens"]:
        raise OrthogonalExecutionBlocked(
            "canonical decision did not nominate A1000")
    if decision.get("source_hashes", {}).get("a1000_lens") != lens_digest:
        raise RuntimeError("canonical decision/lens hash binding drift")
    lens_event = resolve(specification["lens_evidence_id"])
    if not lens_event["live"]:
        raise OrthogonalExecutionBlocked("registered A1000 lens is not live")
    source = resolve_uri(specification["lens_uri"], must_exist=False)
    matches = [
        row for row in lens_event.get("outputs", [])
        if Path(row["path"]).resolve() == source.resolve()
        and row["sha256"] == lens_digest
    ]
    if len(matches) != 1:
        raise RuntimeError("A1000 path/hash is absent from its fit event")
    local = materialize_local_file(
        specification["lens_uri"], expected_sha256=lens_digest)
    return {
        "bound": True,
        "decision_evidence_id": decision_event["evidence_id"],
        "decision_result_path": str(decision_path),
        "decision_result_sha256": observed,
        "branch": branch,
        "lens_evidence_id": lens_event["evidence_id"],
        "lens_path": str(local),
        "lens_sha256": lens_digest,
    }


def independent_review_binding(
        config: Mapping, *, config_path: Path, require_review: bool) -> dict:
    specification = config["review_contract"]
    evidence_id = specification["independent_review_evidence_id"]
    try:
        event, path, digest = _registered_basename(
            evidence_id,
            specification["independent_review_output_basename"])
    except (RegistryError, RuntimeError):
        if require_review:
            raise OrthogonalExecutionBlocked(
                "independent Bank-B orthogonal producer review is absent") \
                from None
        return {"complete": False, "evidence_id": evidence_id}
    envelope, review = _load_envelope(path)
    package_root = Path(__file__).resolve().parents[2]
    expected = {
        "reviewed_config_sha256": file_sha256(config_path),
        "reviewed_producer_sha256": file_sha256(Path(__file__)),
        "reviewed_geometry_core_sha256": file_sha256(
            package_root / "jspace_phase4/orthogonal_bridge.py"),
        "reviewed_analysis_sha256": file_sha256(
            package_root / "jspace_phase4/bank_b_orthogonal_analysis.py"),
        "reviewed_core_test_sha256": file_sha256(
            package_root / "tests/test_orthogonal_bridge.py"),
        "reviewed_analysis_test_sha256": file_sha256(
            package_root / "tests/test_bank_b_orthogonal_analysis.py"),
        "reviewed_producer_test_sha256": file_sha256(
            package_root / "tests/test_bank_b_orthogonal_feasibility.py"),
        "reviewer_independent": True,
        "verdict": specification["required_verdict"],
        "intervention_outcome_opened": False,
        "bank_b_outcome_opened": False,
        "confirmatory_or_replication_outcome_opened": False,
    }
    expected.update({str(field): True for field in
                     specification["required_boolean_findings"]})
    drift = {key: [review.get(key), value]
             for key, value in expected.items()
             if review.get(key) != value}
    if drift:
        raise OrthogonalExecutionBlocked(
            "independent Bank-B review contract drift: "
            + json.dumps(drift, sort_keys=True))
    if not str(review.get("reviewer_identity", "")).strip():
        raise OrthogonalExecutionBlocked("independent review lacks identity")
    if not str(review.get("review_completed_utc", "")).endswith("Z"):
        raise OrthogonalExecutionBlocked("independent review lacks UTC stamp")
    if envelope.get("provenance", {}).get("dirty_tree") is not False:
        raise OrthogonalExecutionBlocked(
            "independent review came from a dirty tree")
    return {
        "complete": True, "evidence_id": event["evidence_id"],
        "path": str(path), "sha256": digest,
        "review_payload_sha256": envelope["payload_sha256"],
    }


def _source_cohort(config: Mapping) -> tuple[
        list[FactBundle], dict, pd.DataFrame]:
    specification = config["consumed_cohort"]
    registry_path = resolve_uri(specification["source_registry"])
    if file_sha256(registry_path) != specification[
            "source_registry_sha256"]:
        raise RuntimeError("Phase-3 source registry hash drift")
    events = [json.loads(line) for line in registry_path.read_text().splitlines()
              if line.strip()]
    matches = [row for row in events
               if row.get("event") == "evidence_created"
               and row.get("evidence_id") == specification[
                   "source_evidence_id"]]
    if len(matches) != 1:
        raise RuntimeError("expected one Phase-3 bridge endpoint event")
    event = matches[0]
    paired_path = resolve_uri(specification["paired_uri"])
    paired_digest = file_sha256(paired_path)
    if paired_digest != specification["paired_sha256"]:
        raise RuntimeError("consumed Phase-3 cohort file hash drift")
    registered = {row["sha256"] for row in event.get("outputs", [])
                  if Path(row["path"]).name == paired_path.name}
    if registered != {paired_digest}:
        raise RuntimeError("consumed cohort is absent from source event")
    # The cohort firewall intentionally reads only identifiers, never an old
    # language-model outcome column.
    identifiers = pd.read_parquet(
        paired_path, columns=["fact_id", "canonical_family"])
    if identifiers.fact_id.duplicated().any():
        raise RuntimeError("consumed cohort contains duplicate fact IDs")
    fact_ids = sorted(str(value) for value in identifiers.fact_id)
    family_ids = sorted(set(str(value) for value in
                            identifiers.canonical_family))
    if object_sha256(fact_ids) != specification["fact_ids_sha256"]:
        raise RuntimeError("consumed fact-ID set hash drift")
    if object_sha256(family_ids) != specification["family_ids_sha256"]:
        raise RuntimeError("consumed family-ID set hash drift")
    if len(fact_ids) != int(specification["expected_items"]) \
            or len(family_ids) != int(specification["expected_families"]):
        raise RuntimeError("consumed cohort size drift")

    banks = {}
    all_bundles = []
    for bank_specification in config["task_banks"]:
        path = resolve_uri(bank_specification["uri"])
        digest = file_sha256(path)
        if digest != bank_specification["sha256"]:
            raise RuntimeError(f"task-bank hash drift: {path.name}")
        banks[path.name] = digest
        all_bundles.extend(load_bank(path))
    by_id = {bundle.fact_id: bundle for bundle in all_bundles}
    if len(by_id) != len(all_bundles):
        raise RuntimeError("task banks contain duplicate fact IDs")
    missing = set(fact_ids) - set(by_id)
    if missing:
        raise RuntimeError(f"consumed facts absent from task banks: {missing}")
    families = identifiers.set_index("fact_id").canonical_family.astype(str)
    cohort = [by_id[fact_id] for fact_id in fact_ids]
    for bundle in cohort:
        if bundle.canonical_family != families.loc[bundle.fact_id]:
            raise RuntimeError("cohort family differs from source event")
        if bundle.bank not in {"F", "S"} \
                or bundle.fact_id.startswith("bank-b"):
            raise RuntimeError("Bank-B row leaked into consumed cohort")
        if not (bundle.counterfactual_bridge
                and bundle.counterfactual_answer
                and bundle.counterfactual_accepted):
            raise RuntimeError(
                f"incomplete counterfactual bundle: {bundle.fact_id}")
    source = {
        "source_registry_path": str(registry_path),
        "source_registry_sha256": file_sha256(registry_path),
        "source_evidence_id": event["evidence_id"],
        "source_code_commit": event["code_commit"],
        "paired_path": str(paired_path),
        "paired_sha256": paired_digest,
        "fact_ids_sha256": object_sha256(fact_ids),
        "family_ids_sha256": object_sha256(family_ids),
        "task_banks": banks,
        "task_banks_sha256": object_sha256(banks),
        "columns_read": ["fact_id", "canonical_family"],
        "outcome_columns_read": [],
        "bank_b_rows_read": False,
    }
    return cohort, source, identifiers.sort_values("fact_id")


def _answer_piece_ids(tokenizer, bundle: FactBundle) -> torch.Tensor:
    identifiers = set()
    aliases = [*bundle.accepted_answers, *bundle.counterfactual_accepted]
    for alias in aliases:
        identifiers.update(int(value) for value in tokenizer(
            alias, add_special_tokens=False).input_ids)
    if not identifiers:
        raise RuntimeError(f"empty answer span for {bundle.fact_id}")
    return torch.tensor(sorted(identifiers), dtype=torch.long)


def _counterfactual_answer_piece_ids(
        tokenizer, bundle: FactBundle) -> torch.Tensor:
    identifiers = set()
    for alias in bundle.counterfactual_accepted:
        identifiers.update(int(value) for value in tokenizer(
            alias, add_special_tokens=False).input_ids)
    if not identifiers:
        raise RuntimeError(
            f"empty counterfactual answer direction for {bundle.fact_id}")
    return torch.tensor(sorted(identifiers), dtype=torch.long)


def _semantic_ids(tokenizer, cohort: Sequence[FactBundle]) -> dict:
    rows = {}
    all_ids = set()
    for bundle in cohort:
        true_bridge = piece_ids(tokenizer, bundle.bridge)
        counterfactual_bridge = piece_ids(
            tokenizer, str(bundle.counterfactual_bridge))
        answer = _answer_piece_ids(tokenizer, bundle)
        counterfactual_answer = _counterfactual_answer_piece_ids(
            tokenizer, bundle)
        record = {
            "true_bridge": true_bridge,
            "counterfactual_bridge": counterfactual_bridge,
            "answer_span": answer,
            "counterfactual_answer": counterfactual_answer,
        }
        rows[bundle.fact_id] = record
        for value in record.values():
            all_ids.update(int(item) for item in value)
    return {"by_fact": rows, "all_ids": sorted(all_ids)}


def _runtime(config: Mapping, lens: Mapping, *, token_ids: Sequence[int]):
    gpu = require_cuda_gpu()
    packages = verify_package_versions(config["runtime"]["packages"])
    fused_runtime = qwen_fused_kernel_contract(config["runtime"])
    model_path = resolve_uri(config["model_uri"])
    snapshot_path = resolve_uri(config["model_snapshot_manifest_uri"])
    if file_sha256(snapshot_path) != config[
            "model_snapshot_manifest_sha256"]:
        raise RuntimeError("Qwen model snapshot manifest hash drift")
    snapshot = verify_snapshot(
        model_path, json.loads(snapshot_path.read_text()))
    import jlens
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
    torch.cuda.reset_peak_memory_stats()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.bfloat16).to("cuda").eval()
    assert_model_on_cuda(hf)
    embeddings = hf.get_output_embeddings().weight
    if list(embeddings.shape) != [
            int(config["runtime"]["expected_vocab_size"]),
            int(config["runtime"]["expected_d_model"])]:
        raise RuntimeError("Qwen output-embedding shape drift")
    fused_model = verify_model_fused_bindings(hf, config["runtime"])
    wrapped = jlens.from_hf(hf, tokenizer)
    checkpoint = load_lens_checkpoint(
        Path(lens["lens_path"]), config["canonical_lens"], config["runtime"])
    gain = effective_gain(hf).to("cuda", torch.float32)
    contract = {
        "gpu": gpu, "packages": packages,
        "fused_runtime": fused_runtime, "fused_model": fused_model,
        "snapshot_inventory_sha256": snapshot["inventory_sha256"],
        "snapshot_manifest_sha256": file_sha256(snapshot_path),
        "partial_dictionary_token_ids_sha256": object_sha256(
            [int(value) for value in token_ids]),
        "partial_dictionary_token_count": len(token_ids),
    }
    return hf, wrapped, tokenizer, checkpoint, gain, contract


def _geometry_output_dir(config: Mapping) -> Path:
    return (metrics_dir(config["slug"]) / "orthogonal_bridge_geometry"
            / config["geometry_evidence_id"])


def _outcome_output_dir(config: Mapping) -> Path:
    return (metrics_dir(config["slug"]) / "orthogonal_bridge_feasibility"
            / config["evidence_id"])


def _geometry_row(*, target: FactBundle, candidate: FactBundle | None,
                  layer: int, role: str,
                  geometry: OrthogonalBridgeGeometry, gate: Mapping) -> dict:
    return {
        "target_fact_id": target.fact_id,
        "target_canonical_family": target.canonical_family,
        "candidate_fact_id": (
            candidate.fact_id if candidate is not None else None),
        "candidate_canonical_family": (
            candidate.canonical_family if candidate is not None else None),
        "layer": int(layer), "role": role,
        **asdict(geometry),
        "gate_passed": bool(gate["passed"]),
        "gate_checks_json": json.dumps(gate["checks"], sort_keys=True),
    }


def _plot_geometry(frame: pd.DataFrame, *, selection: Mapping,
                   png: Path, pdf: Path) -> None:
    selected = {
        fact_id: row["selected_fact_id"]
        for fact_id, row in selection.items()
    }
    target = frame[frame.role == "target"].copy()
    candidates = frame[frame.role == "unrelated_candidate"].copy()
    candidates = candidates[candidates.apply(
        lambda row: selected.get(row.target_fact_id)
        == row.candidate_fact_id, axis=1)]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for values, label, color in (
            (target, "counterfactual bridge", "#0072B2"),
            (candidates, "selected unrelated", "#E69F00")):
        grouped = values.groupby("layer")
        x = sorted(grouped.groups)
        median = grouped.retained_fraction.median().reindex(x)
        low = grouped.retained_fraction.quantile(0.1).reindex(x)
        high = grouped.retained_fraction.quantile(0.9).reindex(x)
        axes[0].plot(x, median, marker="o", label=label, color=color)
        axes[0].fill_between(x, low, high, color=color, alpha=0.15)
        readout = grouped.self_readout_cosine_mean.median().reindex(x)
        axes[1].plot(x, readout, marker="o", label=label, color=color)
        overlap = grouped.maximum_answer_span_cosine.max().reindex(x)
        axes[2].plot(x, overlap, marker="o", label=label, color=color)
    axes[0].set(ylabel="retained norm fraction",
                title="A · Answer-orthogonal retention")
    axes[1].set(ylabel="mean bridge-row cosine",
                title="B · Positive semantic readout")
    axes[2].set(ylabel="maximum answer-span cosine",
                title="C · Numerical orthogonality", yscale="log")
    for axis in axes:
        axis.set_xlabel("source layer")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Consumed-development Bank-B rescue: outcome-blind geometry")
    figure.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def run_geometry(config_path: Path, config: Mapping) -> None:  # noqa: C901
    if _registered_output_check(config["geometry_evidence_id"]) is not None:
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["geometry_evidence_id"],
        }, indent=1))
        return
    clean = require_clean_tree()
    lens = canonical_lens_binding(config, require_bound=True)
    cohort, source, _ = _source_cohort(config)
    # Load the tokenizer first without a model response; it fixes the exact
    # token rows needed by the partial dictionary.
    import transformers
    model_path = resolve_uri(config["model_uri"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
    semantic = _semantic_ids(tokenizer, cohort)
    maximum_pieces = max(
        len(semantic["by_fact"][bundle.fact_id]["true_bridge"])
        for bundle in cohort)
    if maximum_pieces > int(config["intervention"]["k"]):
        raise RuntimeError(
            "a true bridge exceeds frozen k; cross-arm rank would vary")
    hf, _wrapped, tokenizer, checkpoint, gain, runtime = _runtime(
        config, lens, token_ids=semantic["all_ids"])
    band = [int(value) for value in config["intervention"]["band"]]
    offsets = {token_id: ordinal for ordinal, token_id in enumerate(
        semantic["all_ids"])}
    geometry_thresholds = config["geometry_gate"]
    profiles: dict[str, dict[str, list[OrthogonalBridgeGeometry]]] = {
        bundle.fact_id: {"target": [], **{
            candidate.fact_id: [] for candidate in cohort
            if candidate.fact_id != bundle.fact_id
            and candidate.canonical_family != bundle.canonical_family}}
        for bundle in cohort
    }
    random_reports: dict[str, list[dict]] = {
        bundle.fact_id: [] for bundle in cohort}
    rows = []
    for layer in band:
        table = partial_j_dictionary_rows(
            hf, gain, checkpoint["J"][layer], semantic["all_ids"],
            dtype=torch.float16)
        for target in cohort:
            target_tokens = semantic["by_fact"][target.fact_id]
            answer_local = torch.tensor([
                offsets[int(value)] for value in target_tokens["answer_span"]
            ], device=table.device, dtype=torch.long)
            answer_rows = table.index_select(0, answer_local)
            bridge_local = torch.tensor([
                offsets[int(value)] for value in
                target_tokens["counterfactual_bridge"]
            ], device=table.device, dtype=torch.long)
            direction, geometry = orthogonal_bridge_direction(
                table.index_select(0, bridge_local), answer_rows,
                relative_tolerance=float(config["answer_span"][
                    "relative_rank_tolerance"]),
                absolute_tolerance=float(config["answer_span"][
                    "absolute_rank_tolerance"]))
            gate = geometry_gate(geometry, geometry_thresholds)
            profiles[target.fact_id]["target"].append(geometry)
            rows.append(_geometry_row(
                target=target, candidate=None, layer=layer, role="target",
                geometry=geometry, gate=gate))
            random_seed = stable_seed(
                experiment_id=config["geometry_evidence_id"],
                item_id=target.fact_id,
                condition=config["intervention"][
                    "random_direction_seed_namespace"],
                layer=layer,
                base_seed=int(config["intervention"][
                    "random_direction_seed"]))
            _, random_report = stable_random_answer_orthogonal_direction(
                answer_rows, direction, seed=random_seed)
            random_report.update({"layer": layer})
            random_reports[target.fact_id].append(random_report)
            for candidate in cohort:
                if candidate.fact_id == target.fact_id \
                        or candidate.canonical_family \
                        == target.canonical_family:
                    continue
                candidate_tokens = semantic["by_fact"][candidate.fact_id][
                    "true_bridge"]
                candidate_local = torch.tensor([
                    offsets[int(value)] for value in candidate_tokens
                ], device=table.device, dtype=torch.long)
                _, candidate_geometry = orthogonal_bridge_direction(
                    table.index_select(0, candidate_local), answer_rows,
                    relative_tolerance=float(config["answer_span"][
                        "relative_rank_tolerance"]),
                    absolute_tolerance=float(config["answer_span"][
                        "absolute_rank_tolerance"]))
                candidate_gate = geometry_gate(
                    candidate_geometry, geometry_thresholds)
                profiles[target.fact_id][candidate.fact_id].append(
                    candidate_geometry)
                rows.append(_geometry_row(
                    target=target, candidate=candidate, layer=layer,
                    role="unrelated_candidate", geometry=candidate_geometry,
                    gate=candidate_gate))
        del table
        torch.cuda.empty_cache()
        log(f"geometry layer L{layer} complete")

    selections = {}
    all_pass = True
    by_id = {bundle.fact_id: bundle for bundle in cohort}
    for target in cohort:
        candidates = [{
            "fact_id": candidate_id,
            "canonical_family": by_id[candidate_id].canonical_family,
            "profile": profile,
        } for candidate_id, profile in profiles[target.fact_id].items()
            if candidate_id != "target"]
        selected, match = select_unrelated_geometry_match(
            target_profile=profiles[target.fact_id]["target"],
            target_fact_id=target.fact_id,
            target_family=target.canonical_family,
            candidates=candidates)
        target_gates = [geometry_gate(value, geometry_thresholds)
                        for value in profiles[target.fact_id]["target"]]
        selected_gates = [geometry_gate(value, geometry_thresholds)
                          for value in selected["profile"]]
        match_gate = geometry_match_gate(match, geometry_thresholds)
        random_gate = all(
            float(row["maximum_anchor_cosine"])
            <= float(geometry_thresholds["maximum_random_anchor_cosine"])
            for row in random_reports[target.fact_id])
        passed = bool(
            all(row["passed"] for row in target_gates)
            and all(row["passed"] for row in selected_gates)
            and match_gate["passed"] and random_gate)
        selections[target.fact_id] = {
            **match, "target_canonical_family": target.canonical_family,
            "target_profile": [asdict(value) for value in
                               profiles[target.fact_id]["target"]],
            "selected_profile": [asdict(value) for value in
                                  selected["profile"]],
            "target_profile_sha256": object_sha256([
                asdict(value) for value in
                profiles[target.fact_id]["target"]]),
            "selected_profile_sha256": object_sha256([
                asdict(value) for value in selected["profile"]]),
            "target_all_layer_gates_pass": all(
                row["passed"] for row in target_gates),
            "selected_all_layer_gates_pass": all(
                row["passed"] for row in selected_gates),
            "match_gate": match_gate,
            "random_anchor_gate_pass": random_gate,
            "passed": passed,
        }
        all_pass = all_pass and passed

    frame = pd.DataFrame(rows).sort_values([
        "target_fact_id", "role", "candidate_fact_id", "layer"],
        na_position="first")
    output_dir = _geometry_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "orthogonal_geometry_rows.parquet"
    selection_path = output_dir / "unrelated_selection_manifest.json"
    manifest_path = output_dir / "input_manifest.json"
    result_path = output_dir / "orthogonal_geometry_result.json"
    _atomic_parquet(rows_path, frame)
    atomic_json(selection_path, {
        "schema_version": 1, "selection": selections,
        "selection_sha256": object_sha256(selections),
        "selection_used_outcomes": False,
        "bank_b_rows_opened": False,
    })
    geometry_summary = {
        "schema_version": 1,
        "geometry_evidence_id": config["geometry_evidence_id"],
        "n_items": len(cohort), "n_families": len({
            bundle.canonical_family for bundle in cohort}),
        "n_layers": len(band), "n_geometry_rows": len(frame),
        "maximum_true_bridge_piece_count": maximum_pieces,
        "all_target_selected_match_and_random_gates_pass": bool(all_pass),
        "geometry_admitted": bool(all_pass),
        "selection_used_outcomes": False,
        "causal_language_model_forward_calls": 0,
        "final_norm_gain_probe_calls": 1,
        "intervention_outcomes_opened": False,
        "bank_b_outcomes_opened": False,
        "confirmatory_or_replication_outcomes_opened": False,
        "claim_boundary": config["claim_boundary"],
    }
    runtime["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    manifest_payload = {
        "schema_version": 1,
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "geometry_core_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "orthogonal_bridge.py"),
        "source": source, "canonical_lens": lens,
        "runtime": runtime, "semantic_token_ids": semantic["all_ids"],
        "semantic_token_ids_sha256": object_sha256(semantic["all_ids"]),
        "band": band, "answer_span": config["answer_span"],
        "geometry_gate": geometry_thresholds,
        "random_reports": random_reports,
        "geometry_rows_sha256": file_sha256(rows_path),
        "selection_manifest_sha256": file_sha256(selection_path),
    }
    manifest = {
        "schema_version": 1, "payload": manifest_payload,
        "payload_sha256": object_sha256(manifest_payload),
    }
    atomic_json(manifest_path, manifest)
    command = (
        "python -m jspace_phase4.experiments."
        "p4_bank_b_orthogonal_feasibility "
        f"--config {config_path} --geometry")
    inputs = {
        "input_manifest": manifest["payload_sha256"],
        "canonical_decision": lens["decision_result_sha256"],
        "canonical_lens": lens["lens_sha256"],
        "consumed_cohort": source["paired_sha256"],
    }
    write_result4(
        geometry_summary, result_path,
        Provenance4(
            evidence_id=config["geometry_evidence_id"],
            tier=config["geometry_tier"], command=command, inputs=inputs,
            input_manifest_sha256=manifest["payload_sha256"],
            model=model_reference(config["model_uri"]),
            seed_contract=SEED_CONTRACT))
    png = figures_dir() / (
        config["outputs"]["geometry_figure_stem"] + ".png")
    pdf = figures_dir() / (
        config["outputs"]["geometry_figure_stem"] + ".pdf")
    _plot_geometry(frame, selection=selections, png=png, pdf=pdf)
    create(
        config["geometry_evidence_id"], tier=config["geometry_tier"],
        what=(
            "Outcome-blind answer-span orthogonal bridge geometry, fixed "
            "unrelated matches, and random-control anchors on the consumed "
            "Phase-3 cohort; zero model response calls."),
        command=command,
        outputs=[result_path, manifest_path, rows_path, selection_path,
                 png, pdf], inputs=inputs,
        causal_language_model_forward_calls=0,
        final_norm_gain_probe_calls=1, selection_used_outcomes=False,
        intervention_outcomes_opened=False, bank_b_outcomes_opened=False,
        confirmatory_or_replication_outcomes_opened=False)
    del checkpoint, hf
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({
        "evidence_id": config["geometry_evidence_id"],
        "geometry_admitted": all_pass,
        "n_geometry_rows": len(frame),
        "result": str(result_path),
    }, indent=1))


def preflight(config_path: Path, config: Mapping) -> dict:
    cohort, source, _ = _source_cohort(config)
    lens = canonical_lens_binding(config, require_bound=False)
    review = independent_review_binding(
        config, config_path=config_path, require_review=False)
    try:
        geometry_event = _registered_output_check(
            config["geometry_evidence_id"])
    except RuntimeError as error:
        geometry = {"complete": False, "error": str(error)}
    else:
        geometry = {
            "complete": geometry_event is not None,
            "evidence_id": config["geometry_evidence_id"],
        }
        if geometry_event is not None:
            _, path, digest = _registered_basename(
                config["geometry_evidence_id"],
                "orthogonal_geometry_result.json")
            _, payload = _load_envelope(path)
            geometry.update({
                "path": str(path), "sha256": digest,
                "geometry_admitted": payload.get("geometry_admitted"),
            })
    return {
        "schema_version": 1,
        "consumed_cohort": {
            "n_items": len(cohort),
            "n_families": len({bundle.canonical_family for bundle in cohort}),
            **source,
        },
        "canonical_lens": lens,
        "geometry": geometry,
        "independent_review": review,
        "geometry_execution_ready": bool(lens.get("bound")),
        "outcome_execution_ready": bool(
            lens.get("bound") and geometry.get("complete")
            and geometry.get("geometry_admitted")
            and review.get("complete")),
        "bank_b_rows_opened": False,
        "confirmatory_or_replication_outcomes_opened": False,
    }


def _geometry_binding(config_path: Path, config: Mapping) -> dict:
    event, result_path, result_digest = _registered_basename(
        config["geometry_evidence_id"], "orthogonal_geometry_result.json")
    _, result = _load_envelope(result_path)
    if result.get("geometry_admitted") is not True:
        raise OrthogonalExecutionBlocked(
            "outcome-blind orthogonal geometry gate did not pass")
    _, manifest_path, manifest_digest = _registered_basename(
        config["geometry_evidence_id"], "input_manifest.json")
    _, selection_path, selection_digest = _registered_basename(
        config["geometry_evidence_id"],
        "unrelated_selection_manifest.json")
    _, rows_path, rows_digest = _registered_basename(
        config["geometry_evidence_id"], "orthogonal_geometry_rows.parquet")
    manifest_envelope = json.loads(manifest_path.read_text())
    manifest = manifest_envelope.get("payload")
    if not isinstance(manifest, dict) or manifest_envelope.get(
            "payload_sha256") != object_sha256(manifest):
        raise RuntimeError("geometry input manifest payload drift")
    expected = {
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "geometry_core_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "orthogonal_bridge.py"),
        "geometry_rows_sha256": rows_digest,
        "selection_manifest_sha256": selection_digest,
    }
    drift = {key: [manifest.get(key), value]
             for key, value in expected.items()
             if manifest.get(key) != value}
    if drift:
        raise RuntimeError(
            "registered geometry/code binding drift: "
            + json.dumps(drift, sort_keys=True))
    selection_envelope = json.loads(selection_path.read_text())
    selection = selection_envelope.get("selection")
    if not isinstance(selection, dict) or selection_envelope.get(
            "selection_sha256") != object_sha256(selection):
        raise RuntimeError("unrelated selection manifest drift")
    if selection_envelope.get("selection_used_outcomes") is not False \
            or selection_envelope.get("bank_b_rows_opened") is not False:
        raise RuntimeError("geometry selection violated outcome firewall")
    return {
        "event": event, "result_path": str(result_path),
        "result_sha256": result_digest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "manifest_payload_sha256": manifest_envelope["payload_sha256"],
        "selection_path": str(selection_path),
        "selection_sha256": selection_digest,
        "rows_path": str(rows_path), "rows_sha256": rows_digest,
        "selection": selection, "manifest": manifest,
    }


def _new_state(header: Mapping) -> dict:
    return {
        "schema_version": 1, "header": dict(header),
        "protections": {}, "protection_complete": False,
        "runtime": None, "done": {},
        "started_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _load_state(path: Path, header: Mapping) -> dict:
    if not path.exists():
        state = _new_state(header)
        atomic_json(path, state)
        return state
    state = json.loads(path.read_text())
    if state.get("header") != dict(header):
        mismatch = {
            key: [state.get("header", {}).get(key), value]
            for key, value in header.items()
            if state.get("header", {}).get(key) != value
        }
        raise RuntimeError(
            "refusing incompatible orthogonal rescue resume: "
            + json.dumps(mismatch, sort_keys=True))
    return state


@torch.no_grad()
def _collect_protections(
        hf, session: ScoringSession, cohort: Sequence[FactBundle], *,
        state: dict, state_path: Path, top_k: int) -> None:
    for ordinal, bundle in enumerate(cohort, start=1):
        if bundle.fact_id in state["protections"]:
            continue
        prompt = bundle.prompts["composed"]
        prompt_ids = session.prompt_ids(prompt)
        output = hf(input_ids=prompt_ids, use_cache=False)
        identifiers = output.logits[0].topk(
            int(top_k), dim=-1).indices.detach().cpu().tolist()
        record = {
            "prompt_length": int(prompt_ids.shape[1]),
            "prompt_ids_sha256": hashlib.sha256(
                np.asarray(prompt_ids[0].detach().cpu(), dtype="<i8")
                .tobytes()).hexdigest(),
            "top_k": int(top_k), "protect_ids": identifiers,
            "protect_ids_sha256": object_sha256(identifiers),
            "role": "mechanical clean per-position protection only",
            "language_model_outcome_retained": False,
        }
        state["protections"][bundle.fact_id] = record
        atomic_json(state_path, state)
        log(f"clean protection {ordinal}/{len(cohort)} {bundle.fact_id}")
        del output
    if set(state["protections"]) != {
            bundle.fact_id for bundle in cohort}:
        raise RuntimeError("clean protection grid ended incomplete")
    state["protection_complete"] = True
    atomic_json(state_path, state)


def _protection_ids(state: Mapping) -> list[int]:
    identifiers = set()
    for record in state["protections"].values():
        rows = record["protect_ids"]
        if object_sha256(rows) != record["protect_ids_sha256"]:
            raise RuntimeError("clean protection state hash drift")
        identifiers.update(int(value) for row in rows for value in row)
    return sorted(identifiers)


def _rows_for_ids(table: torch.Tensor, offsets: Mapping[int, int],
                  identifiers: torch.Tensor) -> torch.Tensor:
    indices = torch.tensor([
        int(offsets[int(value)]) for value in identifiers
    ], device=table.device, dtype=torch.long)
    return table.index_select(0, indices)


def _unit_mean(rows: torch.Tensor, *, label: str) -> torch.Tensor:
    value = rows.float().mean(dim=0)
    norm = float(value.norm())
    if not math.isfinite(norm) or norm <= 1e-8:
        raise RuntimeError(f"{label} direction is null")
    return value / norm


def _profiles_close(actual: Sequence[Mapping], expected: Sequence[Mapping],
                    *, relative_tolerance: float,
                    absolute_tolerance: float) -> bool:
    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected, strict=True):
        if set(left) != set(right):
            return False
        for key in left:
            if isinstance(left[key], int) and isinstance(right[key], int):
                if left[key] != right[key]:
                    return False
            elif not math.isclose(
                    float(left[key]), float(right[key]),
                    rel_tol=relative_tolerance,
                    abs_tol=absolute_tolerance):
                return False
    return True


def _build_directions(
        *, tables: Mapping[int, torch.Tensor], offsets: Mapping[int, int],
        semantic: Mapping, cohort: Sequence[FactBundle],
        geometry: Mapping, config: Mapping) -> dict:
    by_id = {bundle.fact_id: bundle for bundle in cohort}
    directions = {}
    for target in cohort:
        target_tokens = semantic["by_fact"][target.fact_id]
        selected_id = geometry["selection"][target.fact_id][
            "selected_fact_id"]
        if selected_id not in by_id \
                or by_id[selected_id].canonical_family \
                == target.canonical_family:
            raise RuntimeError("registered unrelated selection is invalid")
        selected_tokens = semantic["by_fact"][selected_id]["true_bridge"]
        target_directions = {arm: {} for arm in
                             config["intervention"]["arm_order"]}
        target_profile = []
        selected_profile = []
        for layer in config["intervention"]["band"]:
            layer = int(layer)
            table = tables[layer]
            answer_rows = _rows_for_ids(
                table, offsets, target_tokens["answer_span"])
            bridge_rows = _rows_for_ids(
                table, offsets, target_tokens["counterfactual_bridge"])
            orthogonal, target_geometry = orthogonal_bridge_direction(
                bridge_rows, answer_rows,
                relative_tolerance=float(config["answer_span"][
                    "relative_rank_tolerance"]),
                absolute_tolerance=float(config["answer_span"][
                    "absolute_rank_tolerance"]))
            unrelated, unrelated_geometry = orthogonal_bridge_direction(
                _rows_for_ids(table, offsets, selected_tokens), answer_rows,
                relative_tolerance=float(config["answer_span"][
                    "relative_rank_tolerance"]),
                absolute_tolerance=float(config["answer_span"][
                    "absolute_rank_tolerance"]))
            target_profile.append(asdict(target_geometry))
            selected_profile.append(asdict(unrelated_geometry))
            random_seed = stable_seed(
                experiment_id=config["geometry_evidence_id"],
                item_id=target.fact_id,
                condition=config["intervention"][
                    "random_direction_seed_namespace"],
                layer=layer,
                base_seed=int(config["intervention"][
                    "random_direction_seed"]))
            random_direction, random_report = (
                stable_random_answer_orthogonal_direction(
                    answer_rows, orthogonal, seed=random_seed))
            if float(random_report["maximum_anchor_cosine"]) > float(
                    config["geometry_gate"][
                        "maximum_random_anchor_cosine"]):
                raise RuntimeError("random orthogonal control gate drift")
            target_directions[
                "counterfactual_bridge_answer_orthogonal"][layer] = orthogonal
            target_directions[
                "unrelated_bridge_answer_orthogonal"][layer] = unrelated
            target_directions[
                "counterfactual_answer_direction"][layer] = _unit_mean(
                    _rows_for_ids(table, offsets,
                                  target_tokens["counterfactual_answer"]),
                    label="counterfactual answer")
            target_directions["counterfactual_bridge_full"][layer] = (
                _unit_mean(bridge_rows, label="full counterfactual bridge"))
            target_directions[
                "random_answer_and_bridge_orthogonal"][layer] = (
                    random_direction)
        registered = geometry["selection"][target.fact_id]
        if object_sha256(registered["target_profile"]) != registered[
                "target_profile_sha256"]:
            raise RuntimeError("registered target geometry profile drift")
        if object_sha256(registered["selected_profile"]) != registered[
                "selected_profile_sha256"]:
            raise RuntimeError("registered selected geometry profile drift")
        if not _profiles_close(
                target_profile, registered["target_profile"],
                relative_tolerance=float(config["geometry_gate"][
                    "replay_relative_tolerance"]),
                absolute_tolerance=float(config["geometry_gate"][
                    "replay_absolute_tolerance"])):
            raise RuntimeError("target geometry profile replay drift")
        if not _profiles_close(
                selected_profile, registered["selected_profile"],
                relative_tolerance=float(config["geometry_gate"][
                    "replay_relative_tolerance"]),
                absolute_tolerance=float(config["geometry_gate"][
                    "replay_absolute_tolerance"])):
            raise RuntimeError("selected geometry profile replay drift")
        target_directions["no_injection"] = None
        directions[target.fact_id] = target_directions
    return directions


def _mode(*, arm: str, protect_ids: torch.Tensor,
          restrict_ids: torch.Tensor, inject_dir,
          prompt_length: int, config: Mapping) -> dict:
    intervention = config["intervention"]
    return {
        "arm": arm, "k": int(intervention["k"]),
        "selection_sign_rule": intervention["selection_sign_rule"],
        "protect_ids": protect_ids, "restrict_ids": restrict_ids,
        "inject_dir": inject_dir, "active_phases": {"prefill"},
        "active_position_limit": int(prompt_length),
        "maximum_injection_direction_norm_error": float(
            intervention["maximum_injection_direction_norm_error"]),
        "maximum_injection_dose_relative_error": float(
            intervention["maximum_injection_dose_relative_error"]),
        "maximum_injection_dose_absolute_error": float(
            intervention["maximum_injection_dose_absolute_error"]),
    }


def _profile_rows(records: Sequence, *, fact_id: str,
                  canonical_family: str, call_key: str) -> list[dict]:
    output = []
    for record in records:
        row = asdict(record)
        row["selected_ids_json"] = json.dumps(list(row.pop("selected_ids")))
        output.append({
            "fact_id": fact_id,
            "canonical_family": canonical_family,
            "call_key": call_key,
            **row,
        })
    return output


@torch.no_grad()
def _forward(
        hf, ablator: OrthogonalBridgeAblator, input_ids: torch.Tensor, *,
        mode: Mapping, call_key: str, bundle: FactBundle,
        use_cache: bool, expected_layers: int) -> tuple[object, list[dict]]:
    ablator.reset_log()
    ablator.phase = "prefill"
    ablator.forward_index = 0
    ablator.mode = dict(mode)
    try:
        output = hf(input_ids=input_ids, use_cache=use_cache)
    finally:
        ablator.mode = None
    expected_positions = int(mode["active_position_limit"]) * expected_layers
    if ablator.log.hook_fires != {"prefill": expected_layers, "decode": 0}:
        raise RuntimeError(
            f"phase hook ownership failed for {bundle.fact_id}/{call_key}")
    if len(ablator.log.positions) != expected_positions:
        raise RuntimeError(
            f"mechanical profile row count failed for "
            f"{bundle.fact_id}/{call_key}")
    rows = _profile_rows(
        ablator.log.positions, fact_id=bundle.fact_id,
        canonical_family=bundle.canonical_family, call_key=call_key)
    return output, rows


@torch.no_grad()
def _run_fact(
        hf, tokenizer, session: ScoringSession,
        ablator: OrthogonalBridgeAblator, *, bundle: FactBundle,
        protection: Mapping, semantic: Mapping, directions: Mapping,
        config: Mapping) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt = bundle.prompts[config["intervention"]["prompt_variant"]]
    prompt_ids = session.prompt_ids(prompt)
    if int(prompt_ids.shape[1]) != int(protection["prompt_length"]):
        raise RuntimeError("clean protection prompt length drift")
    if hashlib.sha256(np.asarray(
            prompt_ids[0].detach().cpu(), dtype="<i8").tobytes()
            ).hexdigest() != protection["prompt_ids_sha256"]:
        raise RuntimeError("clean protection prompt IDs drift")
    prompt_protect = torch.tensor(
        protection["protect_ids"], device=prompt_ids.device,
        dtype=torch.long)
    restrict = semantic["by_fact"][bundle.fact_id]["true_bridge"].to(
        prompt_ids.device)
    arms = list(config["intervention"]["arm_order"])
    order_seed = stable_seed(
        experiment_id=config["evidence_id"], item_id=bundle.fact_id,
        condition=config["intervention"]["arm_order_seed_namespace"],
        base_seed=int(config["intervention"]["arm_order_seed"]))
    permutation = np.random.default_rng(order_seed).permutation(len(arms))
    order = [arms[int(index)] for index in permutation]
    rows, profiles = [], []
    expected_layers = len(config["intervention"]["band"])
    for arm in order:
        started = time.time()
        injection = directions[bundle.fact_id][arm]
        prompt_mode = _mode(
            arm=arm, protect_ids=prompt_protect,
            restrict_ids=restrict, inject_dir=injection,
            prompt_length=int(prompt_ids.shape[1]), config=config)
        prefill, record = _forward(
            hf, ablator, prompt_ids, mode=prompt_mode,
            call_key="generation_prefill", bundle=bundle,
            use_cache=True, expected_layers=expected_layers)
        profiles.extend(record)
        original_lps = {}
        for ordinal, alias in enumerate(bundle.accepted_answers):
            full, prompt_length = session.full_ids(prompt, alias)
            suffix = prompt_protect[-1:].expand(
                full.shape[1] - prompt_length, -1)
            full_protect = torch.cat([prompt_protect, suffix], dim=0)
            full_mode = _mode(
                arm=arm, protect_ids=full_protect,
                restrict_ids=restrict, inject_dir=injection,
                prompt_length=prompt_length, config=config)
            output, record = _forward(
                hf, ablator, full, mode=full_mode,
                call_key=f"original_alias_{ordinal}", bundle=bundle,
                use_cache=False, expected_layers=expected_layers)
            original_lps[alias] = session.answer_seq_lp(
                full, output.logits[0], prompt_length)
            profiles.extend(record)
            del output
        counterfactual_lps = {}
        for ordinal, alias in enumerate(bundle.counterfactual_accepted):
            full, prompt_length = session.full_ids(prompt, alias)
            suffix = prompt_protect[-1:].expand(
                full.shape[1] - prompt_length, -1)
            full_protect = torch.cat([prompt_protect, suffix], dim=0)
            full_mode = _mode(
                arm=arm, protect_ids=full_protect,
                restrict_ids=restrict, inject_dir=injection,
                prompt_length=prompt_length, config=config)
            output, record = _forward(
                hf, ablator, full, mode=full_mode,
                call_key=f"counterfactual_alias_{ordinal}", bundle=bundle,
                use_cache=False, expected_layers=expected_layers)
            counterfactual_lps[alias] = session.answer_seq_lp(
                full, output.logits[0], prompt_length)
            profiles.extend(record)
            del output
        # Hooks are inert during decoding because ablator.mode is already None.
        generated, generated_ids = greedy_from_prefill(
            hf, tokenizer, prefill.logits[0, -1].float(),
            prefill.past_key_values,
            prompt_length=int(prompt_ids.shape[1]),
            max_new_tokens=int(config["intervention"]["max_new_tokens"]))
        grading = boundary_generation_category(
            generated, bundle.accepted_answers,
            bundle.counterfactual_accepted)
        original_canonical = original_lps[bundle.accepted_answers[0]]
        counterfactual_canonical = counterfactual_lps[
            bundle.counterfactual_accepted[0]]
        original_max = max(original_lps.values())
        counterfactual_max = max(counterfactual_lps.values())
        rows.append({
            "fact_id": bundle.fact_id,
            "canonical_family": bundle.canonical_family,
            "relation_group": bundle.relation_group,
            "bank": bundle.bank, "arm": arm,
            "lp_original_canonical": original_canonical,
            "lp_counterfactual_canonical": counterfactual_canonical,
            "preference_canonical": (
                counterfactual_canonical - original_canonical),
            "lp_original_max_alias": original_max,
            "lp_counterfactual_max_alias": counterfactual_max,
            "preference_max_alias": counterfactual_max - original_max,
            "original_alias_lps_json": json.dumps(
                original_lps, sort_keys=True),
            "counterfactual_alias_lps_json": json.dumps(
                counterfactual_lps, sort_keys=True),
            "greedy_text": generated,
            "greedy_token_ids_json": json.dumps(generated_ids),
            "greedy_category": grading["category"],
            "greedy_normalized": grading["normalized"],
            "greedy_original_hits_json": json.dumps(
                grading["original_hits"]),
            "greedy_counterfactual_hits_json": json.dumps(
                grading["counterfactual_hits"]),
            "arm_order_seed": int(order_seed),
            "arm_order_json": json.dumps(order),
            "elapsed_seconds": round(time.time() - started, 3),
        })
        del prefill
    return pd.DataFrame(rows), pd.DataFrame(profiles)


def _verify_mechanical_profiles(
        profiles: pd.DataFrame, outcomes: pd.DataFrame, *,
        config: Mapping) -> dict:
    required = {
        "fact_id", "canonical_family", "call_key", "arm", "layer",
        "phase", "position", "selected_ids_json", "selected_rank",
        "effective_rank", "lost_rank", "removed_norm",
        "injection_direction_norm", "delivered_injection_norm",
        "injection_dose_relative_error", "injection_dose_absolute_error",
    }
    missing = required - set(profiles.columns)
    if missing:
        raise RuntimeError(f"mechanical profiles lack {sorted(missing)}")
    arms = list(config["intervention"]["arm_order"])
    if set(profiles.arm) != set(arms):
        raise RuntimeError("mechanical profile arm set drift")
    if set(profiles.phase) != {"prefill"}:
        raise RuntimeError("wrong-phase intervention profile present")
    key = ["fact_id", "call_key", "layer", "position"]
    grouped = profiles.groupby(key, sort=False)
    arm_counts = grouped.arm.nunique()
    selected_ids = grouped.selected_ids_json.nunique()
    selected_rank = grouped.selected_rank.nunique()
    effective_rank = grouped.effective_rank.nunique()
    exact_rank = bool(
        (arm_counts == len(arms)).all()
        and (selected_ids == 1).all()
        and (selected_rank == 1).all()
        and (effective_rank == 1).all())
    injected = profiles[profiles.arm != "no_injection"]
    direction_error = np.abs(
        injected.injection_direction_norm.to_numpy(dtype=float) - 1.0)
    maximum_direction_error = float(direction_error.max())
    maximum_relative_error = float(
        injected.injection_dose_relative_error.max())
    maximum_absolute_error = float(
        injected.injection_dose_absolute_error.max())
    intervention = config["intervention"]
    dose_pass = bool(
        maximum_direction_error <= float(intervention[
            "maximum_injection_direction_norm_error"])
        and np.all(
            (injected.injection_dose_relative_error.to_numpy(dtype=float)
             <= float(intervention[
                 "maximum_injection_dose_relative_error"]))
            | (injected.injection_dose_absolute_error.to_numpy(dtype=float)
               <= float(intervention[
                   "maximum_injection_dose_absolute_error"]))))
    complete_outcome_grid = bool(
        len(outcomes) == int(config["consumed_cohort"]["expected_items"])
        * len(arms)
        and outcomes.groupby("fact_id").arm.nunique().eq(len(arms)).all())
    checks = {
        "complete_outcome_grid": complete_outcome_grid,
        "prefill_only": set(profiles.phase) == {"prefill"},
        "exact_selected_and_effective_rank_match_across_arms": exact_rank,
        "unit_direction_and_removed_norm_dose": dose_pass,
        "every_profile_value_finite": bool(np.isfinite(profiles[[
            "removed_norm", "delivered_injection_norm",
            "injection_dose_relative_error",
            "injection_dose_absolute_error",
        ]].to_numpy(dtype=float)).all()),
    }
    return {
        "checks": checks, "passed": bool(all(checks.values())),
        "n_profile_rows": int(len(profiles)),
        "n_profile_cells_compared_across_arms": int(len(grouped)),
        "maximum_injection_direction_norm_error": maximum_direction_error,
        "maximum_injection_dose_relative_error": maximum_relative_error,
        "maximum_injection_dose_absolute_error": maximum_absolute_error,
    }


def _plot_outcome(result: Mapping, paired: pd.DataFrame, *,
                  png: Path, pdf: Path) -> None:
    components = ["semantic_component", "bridge_specific_component"]
    labels = ["semantic", "bridge-specific"]
    family = paired.groupby("canonical_family")[components].mean()
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    axes[0].boxplot(
        [family[column] for column in components], tick_labels=labels,
        showfliers=True)
    axes[0].axhline(
        result["fixed_sesoi_nats"], color="#D55E00", linestyle="--",
        label="fixed SESOI")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(ylabel="family contrast (nats)",
                title="A · Orthogonal components")
    axes[0].legend(fontsize=8)
    sample = [result["components"][name]["sample_sd_nats"]
              for name in components]
    planning = [result["components"][name]["planning_sd_nats"]
                for name in components]
    x = np.arange(2)
    axes[1].bar(x - 0.18, sample, 0.36, label="sample SD", color="#56B4E9")
    axes[1].bar(x + 0.18, planning, 0.36, label="planning SD",
                color="#E69F00")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="family SD (nats)",
                title="B · Prospective variance gate")
    axes[1].legend(fontsize=8)
    curve = result["power"]["power_curve"]
    axes[2].plot(
        [row["n_families"] for row in curve],
        [row["rejection_rate"] for row in curve], marker="o")
    axes[2].axhline(
        result["power"]["target_power"], color="#D55E00",
        linestyle="--")
    axes[2].set(xlabel="prospective families", ylabel="joint IUT power",
                ylim=(0, 1.02), title="C · Fixed-SESOI planning")
    figure.suptitle(
        "One-shot consumed-development answer-orthogonal bridge rescue")
    figure.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def _disposition_markdown(result: Mapping) -> str:
    components = result["components"]
    return "\n".join([
        "# Bank-B answer-orthogonal development disposition",
        "",
        "**PHASE 4 DEVELOPMENT — NO BANK-B OUTCOME OPENED**",
        "",
        f"Mechanical disposition: **{result['p4_p1_disposition']}**.",
        "",
        "The single permitted consumed-Phase-3 development shot used the "
        "precommitted answer-span-orthogonal bridge estimand. The two "
        "equal-family components were:",
        "",
        f"- semantic: {components['semantic_component']['equal_family_mean_nats']:+.4f} "
        f"nats; planning SD "
        f"{components['semantic_component']['planning_sd_nats']:.4f};",
        f"- bridge-specific: "
        f"{components['bridge_specific_component']['equal_family_mean_nats']:+.4f} "
        f"nats; planning SD "
        f"{components['bridge_specific_component']['planning_sd_nats']:.4f}.",
        "",
        f"The fixed 0.25-nat joint SESOI has planning power "
        f"{result['power']['available_confirmatory']['rejection_rate']:.4f} "
        f"at the available confirmatory family count. The observed mean did "
        "not alter that SESOI.",
        "",
        "No candidate, confirmatory, or replication Bank-B intervention "
        "outcome was read. This development result cannot emit a reject "
        "verdict. Independent protocol review and PI sign-off remain "
        "separate governance gates.",
        "",
    ])


@torch.no_grad()
def run_outcomes(config_path: Path, config: Mapping) -> None:  # noqa: C901
    if _registered_output_check(config["evidence_id"]) is not None:
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return
    clean = require_clean_tree()
    lens = canonical_lens_binding(config, require_bound=True)
    geometry = _geometry_binding(config_path, config)
    review = independent_review_binding(
        config, config_path=config_path, require_review=True)
    cohort, source, _ = _source_cohort(config)
    output_dir = _outcome_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    parts_dir = output_dir / "parts"
    profile_parts_dir = output_dir / "profile_parts"
    parts_dir.mkdir(exist_ok=True)
    profile_parts_dir.mkdir(exist_ok=True)
    header = {
        "schema_version": 1, "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "geometry_core_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "orthogonal_bridge.py"),
        "analysis_sha256": file_sha256(
            Path(__file__).resolve().parents[1]
            / "bank_b_orthogonal_analysis.py"),
        "canonical_lens": lens, "geometry": {
            key: geometry[key] for key in (
                "result_sha256", "manifest_sha256",
                "selection_sha256", "rows_sha256")},
        "independent_review": review, "source": source,
        "intervention_sha256": object_sha256(config["intervention"]),
        "variance_gate_sha256": object_sha256(config["variance_gate"]),
        "power_gate_sha256": object_sha256(config["power_gate"]),
    }
    state = _load_state(state_path, header)
    for fact_id, record in state["done"].items():
        for prefix in ("outcome", "profile"):
            path = Path(record[f"{prefix}_part_path"])
            if file_sha256(path) != record[f"{prefix}_part_sha256"]:
                raise RuntimeError(
                    f"resume {prefix} part hash drift: {fact_id}")
    expected_ids = [bundle.fact_id for bundle in cohort]
    unexpected = set(state["done"]) - set(expected_ids)
    if unexpected:
        raise RuntimeError(f"orthogonal state has unexpected facts: {unexpected}")
    incomplete = [bundle for bundle in cohort
                  if bundle.fact_id not in state["done"]]
    if incomplete or not state["protection_complete"]:
        import transformers
        tokenizer_probe = transformers.AutoTokenizer.from_pretrained(
            str(resolve_uri(config["model_uri"])))
        semantic = _semantic_ids(tokenizer_probe, cohort)
        hf, wrapped, tokenizer, checkpoint, gain, runtime = _runtime(
            config, lens, token_ids=semantic["all_ids"])
        session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
        if not state["protection_complete"]:
            _collect_protections(
                hf, session, cohort, state=state, state_path=state_path,
                top_k=int(config["intervention"]["clean_protect_top_k"]))
        full_token_ids = sorted(set(semantic["all_ids"])
                                | set(_protection_ids(state)))
        offsets = {token_id: ordinal for ordinal, token_id in enumerate(
            full_token_ids)}
        runtime["partial_dictionary_token_ids_sha256"] = object_sha256(
            full_token_ids)
        runtime["partial_dictionary_token_count"] = len(full_token_ids)
        tables = {}
        for layer in config["intervention"]["band"]:
            layer = int(layer)
            tables[layer] = partial_j_dictionary_rows(
                hf, gain, checkpoint["J"][layer], full_token_ids,
                dtype=torch.float16)
            log(f"outcome partial dictionary L{layer} complete")
        directions = _build_directions(
            tables=tables, offsets=offsets, semantic=semantic,
            cohort=cohort, geometry=geometry, config=config)
        ablator = OrthogonalBridgeAblator(
            wrapped.layers, config["intervention"]["band"],
            dictionaries=tables, offsets=offsets)
        with ablator:
            for ordinal, bundle in enumerate(cohort, start=1):
                if bundle.fact_id in state["done"]:
                    continue
                outcome_part, profile_part = _run_fact(
                    hf, tokenizer, session, ablator, bundle=bundle,
                    protection=state["protections"][bundle.fact_id],
                    semantic=semantic, directions=directions,
                    config=config)
                digest = hashlib.sha256(bundle.fact_id.encode()).hexdigest()[:16]
                outcome_path = parts_dir / f"{ordinal:04d}_{digest}.parquet"
                profile_path = (
                    profile_parts_dir / f"{ordinal:04d}_{digest}.parquet")
                _atomic_parquet(outcome_path, outcome_part)
                _atomic_parquet(profile_path, profile_part)
                state["done"][bundle.fact_id] = {
                    "ordinal": ordinal,
                    "canonical_family": bundle.canonical_family,
                    "outcome_part_path": str(outcome_path),
                    "outcome_part_sha256": file_sha256(outcome_path),
                    "outcome_rows": len(outcome_part),
                    "profile_part_path": str(profile_path),
                    "profile_part_sha256": file_sha256(profile_path),
                    "profile_rows": len(profile_part),
                }
                state["runtime"] = runtime
                atomic_json(state_path, state)
                log(f"outcome {len(state['done'])}/{len(cohort)} "
                    f"{bundle.fact_id}")
        runtime["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        state["runtime"] = runtime
        atomic_json(state_path, state)
        del directions, tables, checkpoint, hf
        gc.collect()
        torch.cuda.empty_cache()
    elif state.get("runtime") is None:
        raise RuntimeError("complete orthogonal state lacks runtime provenance")
    if set(state["done"]) != set(expected_ids):
        raise RuntimeError("orthogonal consumed-development grid is incomplete")

    outcomes = pd.concat([
        pd.read_parquet(state["done"][fact_id]["outcome_part_path"])
        for fact_id in expected_ids], ignore_index=True)
    profiles = pd.concat([
        pd.read_parquet(state["done"][fact_id]["profile_part_path"])
        for fact_id in expected_ids], ignore_index=True)
    mechanical = _verify_mechanical_profiles(
        profiles, outcomes, config=config)
    if not mechanical["passed"]:
        raise RuntimeError(
            "orthogonal mechanical gates failed before analysis: "
            + json.dumps(mechanical, sort_keys=True))
    analysis, paired = analyze_orthogonal_outcomes(
        outcomes, config=config, mechanical_gate=mechanical)
    analysis.update({
        "evidence_id": config["evidence_id"],
        "geometry_evidence_id": config["geometry_evidence_id"],
        "canonical_lens": lens, "independent_review": review,
        "mechanical_gate": mechanical,
        "claim_boundary": config["claim_boundary"],
        "consumed_phase3_development_only": True,
        "bank_b_candidate_rows_opened": False,
        "bank_b_confirmatory_rows_opened": False,
        "bank_b_replication_rows_opened": False,
    })
    rows_path = output_dir / "orthogonal_feasibility_rows.parquet"
    profiles_path = output_dir / "orthogonal_mechanical_profiles.parquet"
    paired_path = output_dir / "orthogonal_feasibility_paired.parquet"
    parts_manifest_path = output_dir / "part_manifest.json"
    input_manifest_path = output_dir / "input_manifest.json"
    result_path = output_dir / "orthogonal_feasibility_result.json"
    disposition_path = output_dir / "BANK_B_ORTHOGONAL_DISPOSITION.md"
    _atomic_parquet(rows_path, outcomes)
    _atomic_parquet(profiles_path, profiles)
    _atomic_parquet(paired_path, paired)
    parts = [{
        "fact_id": fact_id, **state["done"][fact_id]
    } for fact_id in expected_ids]
    atomic_json(parts_manifest_path, {
        "schema_version": 1, "parts": parts,
        "parts_sha256": object_sha256(parts),
    })
    manifest_payload = {
        "schema_version": 1, "header": header,
        "runtime": state["runtime"],
        "protection_manifest_sha256": object_sha256(state["protections"]),
        "part_manifest_sha256": file_sha256(parts_manifest_path),
        "outcome_rows_sha256": file_sha256(rows_path),
        "profile_rows_sha256": file_sha256(profiles_path),
        "paired_rows_sha256": file_sha256(paired_path),
        "geometry_gate": config["geometry_gate"],
        "variance_gate": config["variance_gate"],
        "power_gate": config["power_gate"],
        "mechanical_gate": mechanical,
        "bank_b_rows_opened": False,
    }
    manifest = {
        "schema_version": 1, "payload": manifest_payload,
        "payload_sha256": object_sha256(manifest_payload),
    }
    atomic_json(input_manifest_path, manifest)
    command = (
        "python -m jspace_phase4.experiments."
        "p4_bank_b_orthogonal_feasibility "
        f"--config {config_path} --run")
    inputs = {
        "input_manifest": manifest["payload_sha256"],
        "canonical_decision": lens["decision_result_sha256"],
        "canonical_lens": lens["lens_sha256"],
        "geometry": geometry["result_sha256"],
        "independent_review": review["sha256"],
        "consumed_cohort": source["paired_sha256"],
    }
    write_result4(
        analysis, result_path,
        Provenance4(
            evidence_id=config["evidence_id"], tier=config["tier"],
            command=command, inputs=inputs,
            input_manifest_sha256=manifest["payload_sha256"],
            model=model_reference(config["model_uri"]),
            seed_contract=SEED_CONTRACT))
    _atomic_text(disposition_path, _disposition_markdown(analysis))
    png = figures_dir() / (
        config["outputs"]["outcome_figure_stem"] + ".png")
    pdf = figures_dir() / (
        config["outputs"]["outcome_figure_stem"] + ".pdf")
    _plot_outcome(analysis, paired, png=png, pdf=pdf)
    outputs = [
        result_path, input_manifest_path, rows_path, profiles_path,
        paired_path, parts_manifest_path, state_path, disposition_path,
        png, pdf,
    ]
    create(
        config["evidence_id"], tier=config["tier"],
        what=(
            "Exactly one consumed-Phase-3 development shot of the "
            "answer-span-orthogonal bridge estimand with six shared-lesion "
            "arms, fixed variance gate, and fixed-SESOI power disposition."),
        command=command, outputs=outputs, inputs=inputs,
        interventions_opened=True,
        intervention_tier="consumed-phase3-development-only",
        bank_b_candidate_rows_opened=False,
        bank_b_confirmatory_rows_opened=False,
        bank_b_replication_rows_opened=False,
        confirmatory_or_replication_outcomes_opened=False,
        reject_language_licensed=False,
        p4_p1_disposition=analysis["p4_p1_disposition"])
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "p4_p1_disposition": analysis["p4_p1_disposition"],
        "variance_gate_pass": analysis["variance_gate_pass"],
        "power_gate_pass": analysis["power_gate_pass"],
        "result": str(result_path),
    }, indent=1))


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("tier") != "phase4-development":
        raise RuntimeError("orthogonal Bank-B shot must be development tier")
    if arguments.preflight:
        print(json.dumps(preflight(config_path, config), indent=1))
    elif arguments.geometry:
        run_geometry(config_path, config)
    else:
        run_outcomes(config_path, config)


if __name__ == "__main__":
    main()
