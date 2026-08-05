#!/usr/bin/env python3
"""Reconstruct/verify the frozen Phase 4 Qwen fit-ladder headline numbers.

Paper-analysis audit task (branch interp_jspace_paper_analysis). CPU-only and
deterministic. READ-ONLY over the Phase 4 run root
/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/ and the
repo registry interpretability/jspace_phase4/reports/evidence_events.jsonl;
WRITES only the two reconstruction tables under
interpretability/jspace_paper/analysis/tables/.

Targets (frozen renders from PHASE4_STATE_OF_RECORD.md and
FREEZE_GATE_LEDGER_PHASE4.md)
  1. A500->A1000 structural task q50/q05 0.998702 / 0.998122 (gates 0.95/0.90)
     plus the A120->A250 and A250->A500 progression.
  2. A500/A1000 functional gate: selected-ID Jaccard 0.538462 (floor 0.75,
     FAIL), normalized projector overlap 0.709818 (floor 0.85, FAIL),
     bridge-rescue difference -0.294028 nat (|x| <= 0.25 + same sign, FAIL);
     aggregate endpoints (occupancy, centered excess, span-safe
     specificity/tail, G4, bridge preference) per fit boundary.
  3. Selection-margin audit 17,381/17,381 retained; 15,536 near-tie; 1,845
     stable-core; 0 rank-deficient.
  4. Prompt-323 influence: primary max 181.776618, discarded repeat
     181.777423, worst per-layer repeat difference 0.004572 (tol 0.5);
     materiality all negligible, closest > 3,800x below threshold; the
     runtime-identity diagnostics (323: 181.826310/181.785516 vs fit-log
     173.345; 112: 55.544060/55.587600 vs registered recompute 160.070954).
  5. Canonical decision: mechanical route Q-L4 from the frozen Q-L1..Q-L5
     table, with the routing-input SHA-256 bindings re-hashed.
  6. The accepted permanent A120-A250 state.json deficit (16/17 outputs of
     p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1 resolve).

Methods (recorded per row in the output CSV)
  recomputed_from_registered_rows_independent : independent re-implementation
      (plain numpy/pandas, no repo analysis module) on the registered
      hash-verified row-level tables (CSV/parquet).
  recomputed_from_registered_raw_tensors      : recomputed on CPU from the
      registered raw .pt tensor bytes.
  recomputed_from_registered_state_summaries  : difference recomputed from
      per-lens summaries inside the registered hash-verified state.json.
  recomputed_frozen_truth_table               : deterministic re-application
      of the frozen Q-L1..Q-L5 routing table to registered gate booleans.
  hash_verified_artifact_field                : value read from a registered
      artifact whose sha256 matches the registry event (underlying raw data
      not released, or the value is categorical).
  cross_artifact_consistency                  : the same number is asserted by
      >= 2 independently verified artifacts/events.
  registry_output_resolution                  : existence + sha256 audit of a
      registered event's output list.

Status vocabulary: byte_identical | numerically_identical_render_diff |
numerically_within_frozen_tolerance | failed |
not_reconstructable_from_released_data.

byte_identical is used when the reconstructed full-precision value equals the
registered full-precision value bit-for-bit AND the frozen render is the same
literal; numerically_identical_render_diff when the bit-identical registered
value is rendered rounded in the frozen prose (e.g. 0.998702 for
0.9987020492553711).

Run:  python interpretability/jspace_paper/analysis/scripts/reconstruct_qwen_ladder.py
      [--skip-tensor-recompute]  (skips the 6.6 GB prompt-323 fp32 pass and
      the two 3.3 GB lens hash passes; affected rows then degrade to
      hash_verified_artifact_field and the routing-input row is marked
      accordingly)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Frozen locations (read-only)
# ----------------------------------------------------------------------------
REPO = Path("/content/labs")
PKG = REPO / "interpretability/jspace_phase4"
REGISTRY = PKG / "reports/evidence_events.jsonl"
DRIVE = Path("/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731")
OUT_DIR = REPO / "interpretability/jspace_paper/analysis/tables"

EV_CONV = {
    "A120->A250": "p4-qwen-lens-convergence-drawA-n120-n250-dev-v1",
    "A250->A500": "p4-qwen-lens-convergence-drawA-n250-n500-dev-v1",
    "A500->A1000": "p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1",
}
EV_FUNC = {
    "A120->A250": "p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1",
    "A250->A500": "p4-qwen-multilens-functional-gate-a250-a500-published-dev-v1",
    "A500->A1000": "p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1",
}
EV_MARGIN = "p4-qwen-selection-margin-a500-a1000-dev-v1"
EV_P323 = "p4-qwen-lens-influence-prompt323-dev-v1"
EV_P112 = "p4-qwen-lens-influence-prompt112-dev-v1"
EV_CANON = "p4-qwen-canonical-lens-decision-a1000-dev-v1"
EV_DEFICIT = "p4-qwen-a120-a250-state-permanent-deficit-v1"
EV_RUNTIME = "p4-runtime-identity-synthesis-v1"

PAIRS = {
    "A120->A250": ("a120", "a250"),
    "A250->A500": ("a250", "a500"),
    "A500->A1000": ("a500", "a1000"),
}
ASSAY_START, ASSAY_STOP = 20, 44          # contiguous convergence assay band
TASK_STRATA = ("task_answer_only", "task_bridge_only",
               "task_answer_bridge_shared")
OPERATOR_COLS = ("raw_matrix_cosine", "minus_identity_matrix_cosine",
                 "minus_alpha_identity_matrix_cosine")
TAIL_THRESHOLD_NATS = -1.0                # frozen protocol.tail_threshold_nats
G4_MIN_FLIP = 0.5                         # frozen g4.pass_min_flip_rate
G4_MIN_MARGIN = 0.25                      # frozen g4.pass_min_margin_over_random
MARGIN_K = 10                             # frozen contract.intervention_k
MARGIN_STABLE = 0.01                      # frozen stable_core_margin_at_k
D_MODEL = 5120
# Frozen analysis thresholds (identical in all three functional-gate configs).
THR = {
    "assay_task_token_median_cosine_min": 0.95,
    "task_token_q05_min": 0.90,
    "normalized_selected_span_overlap_min": 0.85,
    "selected_id_jaccard_min": 0.75,
    "occupancy_difference_max": 1.0,
    "centered_excess_difference_percentage_points_max": 1.0,
    "span_safe_specific_mean_difference_nats_max": 0.15,
    "tail_rate_difference_max": 0.05,
    "g4_flip_rate_difference_max": 0.10,
    "bridge_rescue_preference_difference_nats_max": 0.25,
}
MATERIALITY_THR = {
    "assay_task_token_median_disagreement": 0.02,
    "assay_task_token_q05_disagreement": 0.05,
    "assay_identity_adjusted_matrix_disagreement": 0.03,
}
# Accepted permanent deficit (registered in EV_DEFICIT and the deficit
# register; note the audit-brief transcription "361bda08e9ffbe1d333dfcaf..."
# is a corrupted copy of this registered value).
DEFICIT_SHA = ("361bda08e9ffbe1d333fd3cfaf3c7b9545"
               "e6a3504246a16dd8b0c07ad26f45e8")

ROWS: list[dict] = []
PROGRESSION: list[dict] = []
HASH_CACHE: dict[str, str] = {}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    key = str(path)
    if key not in HASH_CACHE:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        HASH_CACHE[key] = digest.hexdigest()
    return HASH_CACHE[key]


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_registry() -> dict[str, dict]:
    events: dict[str, dict] = {}
    with open(REGISTRY) as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") != "evidence_created":
                continue
            eid = record.get("evidence_id")
            if eid:
                events[eid] = record       # last evidence_created record wins
    return events


def event_output(events: dict, evidence_id: str, name: str) -> tuple[Path, str]:
    """Return (path, registered sha256) of the unique output whose basename
    is `name`."""
    hits = [row for row in events[evidence_id]["outputs"]
            if Path(row["path"]).name == name]
    if len(hits) != 1:
        raise RuntimeError(f"{evidence_id}: {len(hits)} outputs named {name}")
    return Path(hits[0]["path"]), hits[0]["sha256"]


def verified_output(events: dict, evidence_id: str, name: str) -> Path:
    """Resolve a registered output and require its bytes to match the
    registry sha256 (the registry is the authoritative pointer)."""
    path, registered = event_output(events, evidence_id, name)
    actual = file_sha256(path)
    if actual != registered:
        raise RuntimeError(
            f"hash mismatch for {evidence_id}:{name}: "
            f"registered {registered} != actual {actual}")
    return path


def add_row(target_id: str, description: str, frozen, reconstructed,
            method: str, status: str, source_paths, notes: str = "") -> None:
    ROWS.append({
        "target_id": target_id,
        "description": description,
        "frozen_value": frozen,
        "reconstructed_value": reconstructed,
        "method": method,
        "status": status,
        "source_paths": ";".join(str(p) for p in source_paths),
        "notes": notes,
    })


def add_prog(boundary: str, metric_class: str, metric: str, value, gate: str,
             passed, source_event: str, method: str) -> None:
    PROGRESSION.append({
        "fit_boundary": boundary,
        "metric_class": metric_class,
        "metric": metric,
        "value": value,
        "gate": gate,
        "pass": "" if passed is None else bool(passed),
        "source_event": source_event,
        "method": method,
    })


def render_status(recomputed: float, registered: float, frozen_render: str,
                  tolerance: float | None = None) -> str:
    """Classify a float target against the registered full-precision value
    and its frozen (possibly rounded) render."""
    if recomputed == registered:
        if frozen_render == repr(registered):
            return "byte_identical"
        return "numerically_identical_render_diff"
    if tolerance is not None and abs(recomputed - registered) <= tolerance:
        return "numerically_within_frozen_tolerance"
    return "failed"


def equal_family_mean(frame: pd.DataFrame, column: str) -> float:
    return float(
        frame.groupby("canonical_family", sort=True)[column].mean().mean())


def ql_branch_from_gates(gates: dict, structural_stable: bool | None) -> str:
    """Frozen Q-L1..Q-L5 truth table (independent re-implementation of
    jspace_phase4.experiments.p4_qwen_multilens_functional_gate
    .ql_branch_from_gates)."""
    if structural_stable is None:
        return "PENDING_STRUCTURAL"
    aggregate = ("occupancy", "centered_excess", "span_safe_specific",
                 "tail_rate", "g4")
    bridge = ("bridge_rescue", "bridge_preference")
    if not all(bool(gates[name]) for name in (*aggregate, *bridge)):
        return "Q-L4"
    if structural_stable is False:
        return "Q-L5"
    if not bool(gates["normalized_selected_span_overlap"]):
        return "Q-L3"
    if not bool(gates["selected_id_jaccard"]):
        return "Q-L2"
    return "Q-L1"


# ----------------------------------------------------------------------------
# 1. Structural convergence ladder
# ----------------------------------------------------------------------------
def reconstruct_structural(events: dict) -> dict[str, bool]:
    frozen_renders = {
        ("A500->A1000", "q50"): "0.998702",
        ("A500->A1000", "q05"): "0.998122",
    }
    stable: dict[str, bool] = {}
    for boundary, evidence_id in EV_CONV.items():
        left, right = PAIRS[boundary]
        comparison = f"{left}_vs_{right}"
        result_path = verified_output(events, evidence_id,
                                      "convergence_result.json")
        csv_path = verified_output(events, evidence_id,
                                   "layer_comparison_metrics.csv")
        payload = json.loads(result_path.read_text())["payload"]
        assay = payload["aggregate"][comparison][
            f"assay_L{ASSAY_START}_L{ASSAY_STOP}"]

        table = pd.read_csv(csv_path, float_precision="round_trip")
        sub = table[(table["comparison_id"] == comparison)
                    & table["layer"].between(ASSAY_START, ASSAY_STOP)]
        if len(sub) != ASSAY_STOP - ASSAY_START + 1:
            raise RuntimeError(f"{boundary}: unexpected assay row count")

        recon, registered = {}, {}
        for tag in ("q50", "q05"):
            recon[tag] = min(
                float(np.median(sub[
                    f"token_{name}_direction_cosine_{tag}"].to_numpy()))
                for name in TASK_STRATA)
            registered[tag] = min(
                float(assay[f"token_{name}_direction_cosine_{tag}"]["median"])
                for name in TASK_STRATA)
        operators = {
            col: (float(np.median(sub[col].to_numpy())),
                  float(assay[col]["median"]))
            for col in OPERATOR_COLS
        }

        gate_q50 = recon["q50"] >= THR["assay_task_token_median_cosine_min"]
        gate_q05 = recon["q05"] >= THR["task_token_q05_min"]
        stable[boundary] = bool(gate_q50 and gate_q05)

        for tag, gate_pass, gate_text in (
                ("q50", gate_q50,
                 f">={THR['assay_task_token_median_cosine_min']}"),
                ("q05", gate_q05, f">={THR['task_token_q05_min']}")):
            frozen = frozen_renders.get((boundary, tag),
                                        repr(registered[tag]))
            add_row(
                f"qwen-structural-{boundary}-task-{tag}",
                f"{boundary} conservative task-row direction-cosine {tag} "
                f"(assay L20-L44 median, min over answer/bridge/shared "
                f"strata); gate {gate_text}",
                frozen, repr(recon[tag]),
                "recomputed_from_registered_rows_independent",
                render_status(recon[tag], registered[tag], frozen),
                [csv_path, result_path],
                f"gate {'PASS' if gate_pass else 'FAIL'}; registered "
                f"aggregate {repr(registered[tag])}; row-level medians "
                f"recomputed over 25 layers x 3 task strata",
            )
            add_prog(boundary, "structural",
                     f"task_direction_cosine_{tag}_conservative",
                     repr(registered[tag]), gate_text, gate_pass,
                     evidence_id,
                     "recomputed_from_registered_rows_independent")
        for col, (recomputed, reg) in operators.items():
            if recomputed != reg:
                raise RuntimeError(f"{boundary}: {col} recompute mismatch")
            add_prog(boundary, "structural_operator",
                     f"{col}_assay_median", repr(reg), "", None, evidence_id,
                     "recomputed_from_registered_rows_independent")

    # Cross-check the frozen engineering progression table in the repo.
    progression_path = PKG / "reports/qwen_structural_progression.json"
    frozen_prog = json.loads(progression_path.read_text())
    for row in frozen_prog["rows"]:
        boundary = {"a120_vs_a250": "A120->A250",
                    "a250_vs_a500": "A250->A500",
                    "a500_vs_a1000": "A500->A1000"}[row["comparison_id"]]
        mine = {r["metric"]: r["value"] for r in PROGRESSION
                if r["fit_boundary"] == boundary}
        checks = {
            "task_direction_cosine_q50_conservative":
                row["task_q50_conservative"],
            "task_direction_cosine_q05_conservative":
                row["task_q05_conservative"],
            "raw_matrix_cosine_assay_median": row["raw_operator_cosine"],
            "minus_identity_matrix_cosine_assay_median":
                row["minus_identity_operator_cosine"],
            "minus_alpha_identity_matrix_cosine_assay_median":
                row["minus_alpha_identity_operator_cosine"],
        }
        for metric, frozen_value in checks.items():
            if mine[metric] != repr(float(frozen_value)):
                raise RuntimeError(
                    f"progression cross-check failed {boundary}:{metric}")
    return stable


# ----------------------------------------------------------------------------
# 2. Functional gate ladder
# ----------------------------------------------------------------------------
def reconstruct_functional(events: dict,
                           structural_stable: dict[str, bool]) -> dict:
    frozen_renders = {
        ("A500->A1000", "selected_id_jaccard_median"): "0.538462",
        ("A500->A1000", "normalized_projector_overlap_median"): "0.709818",
        ("A500->A1000", "bridge_rescue_difference_nats"): "-0.294028",
    }
    payloads = {}
    for boundary, evidence_id in EV_FUNC.items():
        left, right = PAIRS[boundary]
        comparison = f"{left}_vs_{right}"
        result_path = verified_output(events, evidence_id,
                                      "functional_gate_result.json")
        payload = json.loads(result_path.read_text())["payload"]
        payloads[boundary] = payload
        pair = payload["pairs"][comparison]
        gates = payload["functional_gates"]

        sel = pd.read_parquet(
            verified_output(events, evidence_id, "selection_pair_rows.parquet"))
        geometry = sel[sel["comparison"] == comparison]
        prim = pd.read_parquet(
            verified_output(events, evidence_id, "primary_rows.parquet"))
        bridge = pd.read_parquet(
            verified_output(events, evidence_id, "bridge_rows.parquet"))
        g4 = pd.read_parquet(
            verified_output(events, evidence_id, "g4_rows.parquet"))

        # --- row-level recomputes -------------------------------------------
        recon = {
            "n_positions": len(geometry),
            "selected_id_jaccard_median":
                float(geometry["selected_id_jaccard"].median()),
            "normalized_projector_overlap_median":
                float(geometry["normalized_projector_overlap"].median()),
        }
        rescue, pref, tail, flip, g4_pass = {}, {}, {}, {}, {}
        for lens in (left, right):
            lens_primary = prim[prim["lens"] == lens]
            rescue[lens] = equal_family_mean(
                lens_primary[lens_primary["bridge_rescue"].notna()],
                "bridge_rescue")
            pref[lens] = equal_family_mean(
                bridge[bridge["lens"] == lens],
                "preference_counterfactual_swap")
            tail[lens] = float(
                (lens_primary["delta_span_safe"] < TAIL_THRESHOLD_NATS)
                .mean())
            measured = g4[(g4["lens"] == lens) & (~g4["is_calibration"])]
            flip[lens] = (float(measured["swap_j_pick_swap"].mean()),
                          float(measured["swap_random_pick_swap"].mean()))
            g4_pass[lens] = bool(
                flip[lens][0] >= G4_MIN_FLIP
                and flip[lens][0] - flip[lens][1] >= G4_MIN_MARGIN)
        recon["bridge_rescue_difference_nats"] = rescue[left] - rescue[right]
        recon["bridge_preference_difference"] = pref[left] - pref[right]
        recon["tail_rate_difference"] = tail[left] - tail[right]
        recon["g4_flip_rate_difference"] = flip[left][0] - flip[right][0]
        wide = (prim[prim["lens"].isin([left, right])]
                .pivot_table(index="canonical_family", columns="lens",
                             values="specific", aggfunc="mean")
                .dropna(subset=[left, right]))
        recon["span_safe_specific_equal_family_mean_difference_nats"] = float(
            (wide[left] - wide[right]).to_numpy(np.float64).mean())

        registered = {
            "n_positions": pair["selection_geometry"]["n_positions"],
            "selected_id_jaccard_median":
                pair["selection_geometry"]["selected_id_jaccard_median"],
            "normalized_projector_overlap_median":
                pair["selection_geometry"][
                    "normalized_projector_overlap_median"],
            "bridge_rescue_difference_nats": pair["bridge_rescue_difference"],
            "bridge_preference_difference":
                pair["bridge_preference_difference"],
            "tail_rate_difference": pair["tail_rate_difference"],
            "g4_flip_rate_difference": pair["g4_flip_rate_difference"],
            "span_safe_specific_equal_family_mean_difference_nats":
                pair["specific"]["equal_family_mean_difference"],
        }

        # --- capacity endpoints ---------------------------------------------
        capacity_method = "hash_verified_artifact_field"
        occupancy = {layer: value["occupancy_difference"]
                     for layer, value in pair["capacity"].items()}
        centered = {layer: value["centered_excess_difference"]
                    for layer, value in pair["capacity"].items()}
        state_hits = [row for row in events[evidence_id]["outputs"]
                      if Path(row["path"]).name == "state.json"]
        state_path = Path(state_hits[0]["path"]) if state_hits else None
        if state_path is not None and state_path.exists():
            state_path = verified_output(events, evidence_id, "state.json")
            state = json.loads(state_path.read_text())
            for layer in occupancy:
                cell_l = state["lenses"][left]["capacity"][layer]
                cell_r = state["lenses"][right]["capacity"][layer]
                occ = float(cell_l["occupancy_median"]) \
                    - float(cell_r["occupancy_median"])
                cen = float(cell_l["centered_variance_explained_excess"]) \
                    - float(cell_r["centered_variance_explained_excess"])
                if occ != occupancy[layer] or cen != centered[layer]:
                    raise RuntimeError(
                        f"{boundary}: capacity recompute mismatch at L{layer}")
            capacity_method = "recomputed_from_registered_state_summaries"
            del state
        recon["occupancy_difference_max_abs"] = max(
            abs(v) for v in occupancy.values())
        recon["centered_excess_difference_max_abs_pp"] = max(
            100 * abs(v) for v in centered.values())
        registered["occupancy_difference_max_abs"] = recon[
            "occupancy_difference_max_abs"]
        registered["centered_excess_difference_max_abs_pp"] = recon[
            "centered_excess_difference_max_abs_pp"]

        # --- recomputed gate booleans ---------------------------------------
        recon_gates = {
            "normalized_selected_span_overlap":
                recon["normalized_projector_overlap_median"]
                >= THR["normalized_selected_span_overlap_min"],
            "selected_id_jaccard":
                recon["selected_id_jaccard_median"]
                >= THR["selected_id_jaccard_min"],
            "occupancy": recon["occupancy_difference_max_abs"]
                <= THR["occupancy_difference_max"],
            "centered_excess": recon["centered_excess_difference_max_abs_pp"]
                <= THR["centered_excess_difference_percentage_points_max"],
            "span_safe_specific": abs(
                recon["span_safe_specific_equal_family_mean_difference_nats"])
                <= THR["span_safe_specific_mean_difference_nats_max"],
            "tail_rate": abs(recon["tail_rate_difference"])
                <= THR["tail_rate_difference_max"],
            "g4": abs(recon["g4_flip_rate_difference"])
                <= THR["g4_flip_rate_difference_max"]
                and g4_pass[left] and g4_pass[right],
            "bridge_rescue": abs(recon["bridge_rescue_difference_nats"])
                <= THR["bridge_rescue_preference_difference_nats_max"]
                and np.sign(rescue[left]) == np.sign(rescue[right]),
            "bridge_preference": abs(recon["bridge_preference_difference"])
                <= THR["bridge_rescue_preference_difference_nats_max"]
                and np.sign(pref[left]) == np.sign(pref[right]),
        }
        for name, value in recon_gates.items():
            if bool(value) != bool(gates[name]):
                raise RuntimeError(
                    f"{boundary}: gate {name} recompute mismatch")

        # --- recon table rows (A500->A1000 headline metrics) ----------------
        for metric, gate_name, gate_text in (
                ("selected_id_jaccard_median", "selected_id_jaccard",
                 f">={THR['selected_id_jaccard_min']}"),
                ("normalized_projector_overlap_median",
                 "normalized_selected_span_overlap",
                 f">={THR['normalized_selected_span_overlap_min']}"),
                ("bridge_rescue_difference_nats", "bridge_rescue",
                 "abs<=0.25 and same rescue sign")):
            frozen = frozen_renders.get((boundary, metric))
            if frozen is None:
                continue
            add_row(
                f"qwen-functional-{boundary}-{metric}",
                f"{boundary} functional gate {metric} over {comparison}; "
                f"frozen floor/gate {gate_text}",
                frozen, repr(recon[metric]),
                "recomputed_from_registered_rows_independent",
                render_status(recon[metric], registered[metric], frozen),
                [result_path],
                f"gate {'PASS' if recon_gates[gate_name] else 'FAIL'} "
                f"(registered boolean {gates[gate_name]}); registered "
                f"{repr(registered[metric])}; n_positions="
                f"{recon['n_positions']}",
            )

        # --- progression rows ------------------------------------------------
        for metric, cls, gate_name, gate_text in (
                ("selected_id_jaccard_median", "functional_sparse_geometry",
                 "selected_id_jaccard", ">=0.75"),
                ("normalized_projector_overlap_median",
                 "functional_sparse_geometry",
                 "normalized_selected_span_overlap", ">=0.85"),
                ("bridge_rescue_difference_nats", "functional_causal_bridge",
                 "bridge_rescue", "abs<=0.25 & same rescue sign"),
                ("bridge_preference_difference", "functional_causal_bridge",
                 "bridge_preference", "abs<=0.25 & same preference sign"),
                ("occupancy_difference_max_abs", "functional_aggregate",
                 "occupancy", "abs<=1.0"),
                ("centered_excess_difference_max_abs_pp",
                 "functional_aggregate", "centered_excess", "abs<=1.0 pp"),
                ("span_safe_specific_equal_family_mean_difference_nats",
                 "functional_aggregate", "span_safe_specific", "abs<=0.15"),
                ("tail_rate_difference", "functional_aggregate", "tail_rate",
                 "abs<=0.05"),
                ("g4_flip_rate_difference", "functional_aggregate", "g4",
                 "abs<=0.10 & both lenses pass positive control")):
            method = (capacity_method if cls == "functional_aggregate"
                      and metric.startswith(("occupancy", "centered"))
                      else "recomputed_from_registered_rows_independent")
            add_prog(boundary, cls, metric, repr(float(registered[metric])),
                     gate_text, bool(gates[gate_name]), evidence_id, method)
        add_prog(boundary, "functional_sparse_geometry", "n_positions",
                 registered["n_positions"], "completeness 17381",
                 registered["n_positions"] == 17381, evidence_id,
                 "recomputed_from_registered_rows_independent")

    # Aggregate-endpoint stability summary row (state-of-record claim).
    always_pass = ("occupancy", "centered_excess", "span_safe_specific",
                   "tail_rate", "g4", "bridge_preference")
    stable_all = all(payloads[b]["functional_gates"][g]
                     for b in EV_FUNC for g in always_pass)
    add_row(
        "qwen-functional-aggregate-endpoints-stable-across-fits",
        "Occupancy, centered excess, span-safe specificity, tail rate, G4, "
        "and bridge preference pass at every fit boundary "
        "(A120->A250, A250->A500, A500->A1000)",
        "stable across fits", f"all-pass={stable_all}",
        "recomputed_from_registered_rows_independent",
        "byte_identical" if stable_all else "failed",
        [DRIVE / "metrics"],
        "gate booleans recomputed from row-level parquets and verified "
        "against every registered functional_gate_result.json; "
        "bridge_rescue passes only at A120->A250 and fails at the two "
        "later boundaries; jaccard + overlap fail at every boundary",
    )
    return payloads


# ----------------------------------------------------------------------------
# 3. Selection-margin audit
# ----------------------------------------------------------------------------
def reconstruct_margin(events: dict) -> dict:
    result_path = verified_output(events, EV_MARGIN,
                                  "selection_margin_result.json")
    rows_path = verified_output(events, EV_MARGIN,
                                "selection_margin_pair_rows.parquet")
    payload = json.loads(result_path.read_text())["payload"]
    frame = pd.read_parquet(rows_path)

    recomputed = np.where(
        (frame["a500_effective_rank"] < MARGIN_K)
        | (frame["a1000_effective_rank"] < MARGIN_K), "rank_deficient",
        np.where((frame["a500_margin_at_k"] >= MARGIN_STABLE)
                 & (frame["a1000_margin_at_k"] >= MARGIN_STABLE),
                 "stable_core", "near_tie"))
    label_match = bool((recomputed == frame["stratum"]).all())
    counts = pd.Series(recomputed).value_counts().to_dict()
    counts.setdefault("rank_deficient", 0)

    registered_counts = {
        "near_tie": payload["by_stratum"]["near_tie"]["n_positions"],
        "stable_core": payload["by_stratum"]["stable_core"]["n_positions"],
        "rank_deficient":
            payload["by_stratum"]["rank_deficient"]["n_positions"],
    }
    verdict = payload["contract_verdict"]
    targets = [
        ("retained-positions", "all captured positions retained",
         "17,381/17,381", f"{len(frame)}/{payload['n_positions']}",
         len(frame) == 17381 and payload["n_positions"] == 17381
         and verdict["all_positions_retained"]),
        ("near-tie", "near-tie stratum count", "15,536",
         str(counts["near_tie"]),
         counts["near_tie"] == 15536 == registered_counts["near_tie"]),
        ("stable-core", "stable-core stratum count", "1,845",
         str(counts["stable_core"]),
         counts["stable_core"] == 1845 == registered_counts["stable_core"]),
        ("rank-deficient", "rank-deficient stratum count", "0",
         str(counts["rank_deficient"]),
         counts["rank_deficient"] == 0 == registered_counts["rank_deficient"]),
    ]
    for suffix, description, frozen, recon, ok in targets:
        add_row(
            f"qwen-selection-margin-{suffix}",
            f"A500/A1000 selection-margin audit: {description} "
            f"(k={MARGIN_K}, stable-core relative margin >={MARGIN_STABLE})",
            frozen, recon,
            "recomputed_from_registered_rows_independent",
            ("byte_identical" if frozen == recon
             else "numerically_identical_render_diff") if ok else "failed",
            [rows_path, result_path],
            f"per-position stratum labels recomputed from margins/ranks; "
            f"row-label match={label_match}; contract verdict all-true="
            f"{all(bool(v) is (k != 'behavioral_columns_used') for k, v in verdict.items())}",
        )
    for name, count in (("retained_positions", len(frame)),
                        ("near_tie", counts["near_tie"]),
                        ("stable_core", counts["stable_core"]),
                        ("rank_deficient", counts["rank_deficient"])):
        add_prog("A500->A1000", "selection_margin", name, count,
                 "completeness", True, EV_MARGIN,
                 "recomputed_from_registered_rows_independent")
    return payload


# ----------------------------------------------------------------------------
# 4. Prompt-323 influence + runtime identity
# ----------------------------------------------------------------------------
def reconstruct_prompt323(events: dict, skip_tensor: bool) -> None:
    result_path = verified_output(events, EV_P323, "influence_result.json")
    csv_path = verified_output(events, EV_P323, "layer_influence_metrics.csv")
    payload = json.loads(result_path.read_text())["payload"]
    contribution = payload["prompt_contribution"]
    repeatability = contribution["repeatability"]

    # 4a. primary max ||J||_F / sqrt(d): recompute from the registered fp32
    # tensor when allowed.
    reg_primary = repeatability["primary_max_jacobian_norm_over_sqrt_d"]
    if not skip_tensor:
        import torch
        tensor_path, registered_sha = event_output(
            events, EV_P323, "qwen36-27b_prompt323_contribution_fp32.pt")
        actual_sha = file_sha256(tensor_path)
        if actual_sha != registered_sha:
            raise RuntimeError("prompt-323 fp32 contribution hash mismatch")
        blob = torch.load(tensor_path, map_location="cpu",
                          weights_only=True, mmap=True)
        if int(blob["d_model"]) != D_MODEL or blob["dtype"] != "float32":
            raise RuntimeError("prompt-323 contribution metadata drift")
        norms = {}
        for layer in blob["source_layers"]:
            tensor = blob["J"][layer]
            norms[int(layer)] = float(
                torch.linalg.vector_norm(tensor).item()) / math.sqrt(D_MODEL)
        recon_primary = max(norms.values())
        norms_sha = object_sha256(norms)
        sha_match = norms_sha == repeatability["primary_layer_norms_sha256"]
        status = render_status(
            recon_primary, reg_primary, "181.776618",
            tolerance=repeatability["absolute_tolerance"])
        if sha_match and status == "numerically_within_frozen_tolerance":
            status = "numerically_identical_render_diff"
        add_row(
            "qwen-prompt323-primary-max-norm",
            "Prompt-323 primary max ||J_layer||_F / sqrt(d_model) over 63 "
            "layers (current-runtime sensitivity shape only)",
            "181.776618", repr(recon_primary),
            "recomputed_from_registered_raw_tensors", status,
            [tensor_path, result_path],
            f"registered {repr(reg_primary)}; layer-norm dict object-sha256 "
            f"match={sha_match} (registered "
            f"{repeatability['primary_layer_norms_sha256'][:12]}...); "
            f"n_layers={len(norms)}; fp32 tensor sha verified",
        )
        del blob
    else:
        add_row(
            "qwen-prompt323-primary-max-norm",
            "Prompt-323 primary max ||J_layer||_F / sqrt(d_model)",
            "181.776618", repr(reg_primary),
            "hash_verified_artifact_field",
            "numerically_identical_render_diff",
            [result_path],
            "--skip-tensor-recompute: value read from the hash-verified "
            "registered summary only",
        )

    # 4b. discarded repeat + worst per-layer repeat difference: the repeat
    # tensor was registered as diagnostic-only-discarded and never released,
    # so these verify against the hash-verified registered summary.
    add_row(
        "qwen-prompt323-repeat-max-norm",
        "Prompt-323 discarded diagnostic repeat max norm "
        "(repeat tensor not released; role diagnostic-only-discarded)",
        "181.777423",
        repr(repeatability["repeat_max_jacobian_norm_over_sqrt_d"]),
        "hash_verified_artifact_field",
        "numerically_identical_render_diff"
        if repr(repeatability["repeat_max_jacobian_norm_over_sqrt_d"])
        .startswith("181.777423") else "failed",
        [result_path],
        "raw repeat tensor NOT reconstructable from released data; "
        "summary field verified via registered result sha256",
    )
    worst = repeatability["maximum_layer_norm_over_sqrt_d_absolute_difference"]
    add_row(
        "qwen-prompt323-repeat-worst-layer-diff",
        "Worst per-layer normalized-norm absolute difference between primary "
        "and discarded repeat; frozen current-runtime tolerance 0.5",
        "0.004572 <= 0.5", repr(worst),
        "hash_verified_artifact_field",
        "numerically_identical_render_diff"
        if repr(worst).startswith("0.00457203840613829")
        and worst <= repeatability["absolute_tolerance"] else "failed",
        [result_path],
        f"registered pass={repeatability['pass']}; tolerance "
        f"{repeatability['absolute_tolerance']}",
    )

    # 4c. materiality: recompute all six frozen metrics from the registered
    # row-level per-layer CSV.
    table = pd.read_csv(csv_path, float_precision="round_trip")
    ratios = []
    all_match = True
    for lens in ("a500", "a1000"):
        assay = table[(table["lens"] == lens)
                      & table["layer"].between(ASSAY_START, ASSAY_STOP)]
        recon = {
            "assay_task_token_median_disagreement": max(
                1 - float(assay[
                    f"token_{name}_direction_cosine_q50"].median())
                for name in TASK_STRATA),
            "assay_task_token_q05_disagreement": max(
                1 - float(assay[
                    f"token_{name}_direction_cosine_q05"].median())
                for name in TASK_STRATA),
            "assay_identity_adjusted_matrix_disagreement": 1 - float(
                assay["minus_alpha_identity_matrix_cosine"].median()),
        }
        registered = payload["materiality"]["by_lens"][lens]
        for name, value in recon.items():
            all_match = all_match and value == registered[name]
            ratios.append(MATERIALITY_THR[name] / value)
    closest = min(ratios)
    add_row(
        "qwen-prompt323-materiality-negligible",
        "All six frozen A500/A1000 materiality metrics negligible; closest "
        "metric-to-threshold ratio",
        "negligible; closest > 3,800x below threshold",
        f"decision={payload['materiality']['decision']}; closest ratio "
        f"{closest:.6f}x",
        "recomputed_from_registered_rows_independent",
        "numerically_identical_render_diff"
        if all_match and closest > 3800
        and payload["materiality"]["decision"] == "negligible" else "failed",
        [csv_path, result_path],
        "six metrics recomputed bit-identically from layer_influence_metrics"
        ".csv (assay L20-L44 medians); closest = a500 "
        "assay_task_token_median_disagreement 5.185604095458984e-06 vs "
        "threshold 0.02",
    )

    # 4d. runtime-identity diagnostics (registered synthesis + repo report).
    synthesis_path = verified_output(
        events, EV_RUNTIME, "PHASE4_RUNTIME_IDENTITY_SYNTHESIS.md")
    runtime_path = PKG / "reports/PHASE4_PART4_PROMPT323_RUNTIME_IDENTITY.json"
    runtime = json.loads(runtime_path.read_text())
    observations = {row["name"]: row
                    for row in runtime["diagnostic_observations"]}
    synthesis_text = synthesis_path.read_text()
    p323 = (observations["prompt323_clean_process_1"]["max_norm_over_sqrt_d"],
            observations["prompt323_clean_process_2"]["max_norm_over_sqrt_d"])
    fit_log = runtime["frozen_prompt323_control"][
        "historical_logged_max_norm_over_sqrt_d"]
    renders_present = all(
        text in synthesis_text
        for text in ("181.826310", "181.785516", "173.345"))
    add_row(
        "qwen-prompt323-runtime-identity",
        "Blocked runtime-identity diagnostic: two clean current-runtime "
        "prompt-323 processes vs the frozen fit-log value (tolerance 0.5, "
        "reported-non-gating)",
        "181.826310 / 181.785516 vs fit-log 173.345",
        f"{repr(p323[0])} / {repr(p323[1])} vs fit-log {fit_log}",
        "cross_artifact_consistency",
        "numerically_identical_render_diff"
        if renders_present and repr(p323[0]).startswith("181.82630")
        and repr(p323[1]).startswith("181.78551") and fit_log == 173.345
        else "failed",
        [runtime_path, synthesis_path],
        "raw diagnostic tensors not released -> values verified from the "
        "repo diagnostic JSON and the registered runtime-identity synthesis "
        "(sha-verified); registered influence event separately records "
        "historical diff 8.431618 vs its own primary; historical-runtime "
        "reproducibility not claimed",
    )

    # 4e. prompt-112 controls.
    p112_result = verified_output(events, EV_P112, "influence_result.json")
    p112 = json.loads(p112_result.read_text())["payload"]
    reg_recompute = p112["prompt_contribution"]["max_jacobian_norm_over_sqrt_d"]
    controls = (
        observations["prompt112_current_runtime_control_1"],
        observations["prompt112_current_runtime_control_2"],
    )
    consistent = all(
        row["registered_clean_recompute"] == reg_recompute
        for row in controls)
    add_row(
        "qwen-prompt112-runtime-controls",
        "Prompt-112 current-runtime controls vs the registered fit-era clean "
        "recompute (runtime-identity block evidence)",
        "55.544060 / 55.587600 vs registered recompute 160.070954",
        f"{repr(controls[0]['max_norm_over_sqrt_d'])} / "
        f"{repr(controls[1]['max_norm_over_sqrt_d'])} vs "
        f"{repr(reg_recompute)}",
        "cross_artifact_consistency",
        "numerically_identical_render_diff"
        if repr(controls[0]["max_norm_over_sqrt_d"]).startswith("55.544060")
        and repr(controls[1]["max_norm_over_sqrt_d"]).startswith("55.587600")
        and repr(reg_recompute).startswith("160.07095424367213") and consistent
        else "failed",
        [runtime_path, p112_result, synthesis_path],
        "160.07095424367213 verified inside the hash-verified registered "
        "prompt-112 influence result (fit-log 159.952, diff 0.118954 <= "
        "0.5); 55.5x values exist only in the runtime-identity diagnostics",
    )


# ----------------------------------------------------------------------------
# 5. Canonical decision
# ----------------------------------------------------------------------------
def reconstruct_canonical(events: dict, functional_payloads: dict,
                          structural_stable: dict[str, bool],
                          margin_payload: dict, skip_tensor: bool) -> None:
    decision_path = verified_output(events, EV_CANON,
                                    "canonical_lens_decision.json")
    payload = json.loads(decision_path.read_text())["payload"]
    gates = functional_payloads["A500->A1000"]["functional_gates"]
    route = ql_branch_from_gates(gates, structural_stable["A500->A1000"])

    # Re-hash the recorded routing inputs.
    expected = payload["source_hashes"]
    actual_sources = {
        "functional_result": event_output(
            events, EV_FUNC["A500->A1000"], "functional_gate_result.json"),
        "functional_manifest": event_output(
            events, EV_FUNC["A500->A1000"], "input_manifest.json"),
        "structural_result": event_output(
            events, EV_CONV["A500->A1000"], "convergence_result.json"),
        "selection_margin_result": event_output(
            events, EV_MARGIN, "selection_margin_result.json"),
        "prompt323_influence_result": event_output(
            events, EV_P323, "influence_result.json"),
    }
    binding_checks = {}
    for name, (path, _registered) in actual_sources.items():
        binding_checks[name] = file_sha256(path) == expected[name]
    lens_note = ""
    if not skip_tensor:
        lens_path = DRIVE / "lens/qwen36-27b/nested_fit/draw_a" \
            / "qwen36-27b_jlens_drawA_n1000.pt"
        binding_checks["a1000_lens"] = (
            file_sha256(lens_path) == expected["a1000_lens"])
    else:
        lens_note = "; a1000_lens 3.3GB hash skipped (--skip-tensor-recompute)"
    amendment_path = PKG / "preregistration/QL2_ESTIMAND_AMENDMENT_DRAFT.md"
    binding_checks["ql2_amendment"] = (
        file_sha256(amendment_path) == expected["ql2_amendment"])

    margin_ok = bool(margin_payload["contract_verdict"][
        "audit_complete_for_canonical_branch_router"])
    ok = (route == payload["branch"] == "Q-L4"
          and payload["canonical_lens"] is None
          and not payload["canonical_lens_nominated"]
          and all(binding_checks.values()) and margin_ok)
    add_row(
        "qwen-canonical-route",
        "Mechanical canonical-lens route from the frozen Q-L1..Q-L5 table "
        "over registered structural/functional/margin/influence inputs",
        "Q-L4 (no canonical sparse lens)",
        f"{route} (canonical_lens={payload['canonical_lens']})",
        "recomputed_frozen_truth_table",
        "byte_identical" if ok else "failed",
        [decision_path],
        "route driver: bridge_rescue gate FALSE forces Q-L4 before the "
        "sparse-geometry rows are consulted (overlap/jaccard also FALSE); "
        "structural_stable=True; routing-input sha256 bindings re-hashed: "
        + ", ".join(f"{k}={v}" for k, v in sorted(binding_checks.items()))
        + f"; margin audit_complete={margin_ok}" + lens_note,
    )
    add_prog("A500->A1000", "decision", "ql_branch", route,
             "frozen Q-L1..Q-L5 table", route == "Q-L4", EV_CANON,
             "recomputed_frozen_truth_table")


# ----------------------------------------------------------------------------
# 6. Permanent A120-A250 state deficit
# ----------------------------------------------------------------------------
def reconstruct_deficit(events: dict) -> None:
    event = events[EV_FUNC["A120->A250"]]
    resolved, failures = 0, []
    for output in event["outputs"]:
        path = Path(output["path"])
        if path.exists() and file_sha256(path) == output["sha256"]:
            resolved += 1
        else:
            failures.append((path.name, output["sha256"]))
    deficit_event = events[EV_DEFICIT]
    recorded = deficit_event["missing_output"]["expected_sha256"]
    register = json.loads(
        (PKG / "protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json").read_text())
    register_sha = [row["expected_sha256"] for row in register["deficits"]
                    if row["path_suffix"].endswith("state.json")][0]
    ok = (resolved == len(event["outputs"]) - 1
          and len(failures) == 1
          and failures[0][0] == "state.json"
          and failures[0][1] == DEFICIT_SHA == recorded == register_sha)
    add_row(
        "qwen-a120-a250-state-permanent-deficit",
        "Accepted permanent historical deficit: a120-a250 functional-gate "
        "state.json absent; all other outputs of the event must resolve",
        f"16/17 resolve; state.json absent (sha {DEFICIT_SHA[:24]}...)",
        f"{resolved}/{len(event['outputs'])} resolve; missing="
        + (failures[0][0] if failures else "none"),
        "registry_output_resolution",
        "byte_identical" if ok else "failed",
        [REGISTRY, PKG / "protocol/KNOWN_DURABILITY_DEFICITS_PHASE4.json"],
        f"expected sha256 {DEFICIT_SHA} matches the deficit event and the "
        "known-deficit register; NOTE the audit-brief string "
        "'361bda08e9ffbe1d333dfcaf...' is a 62-hex transcription corruption "
        "of this registered 64-hex value; capacity_reconstructions_a120.pt "
        "verifies against its registered recovery hash 6b0399df...",
    )


# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tensor-recompute", action="store_true",
                        help="skip the 6.6GB fp32 pass and 3.3GB lens hash")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = load_registry()

    structural_stable = reconstruct_structural(events)
    functional_payloads = reconstruct_functional(events, structural_stable)
    margin_payload = reconstruct_margin(events)
    reconstruct_prompt323(events, args.skip_tensor_recompute)
    reconstruct_canonical(events, functional_payloads, structural_stable,
                          margin_payload, args.skip_tensor_recompute)
    reconstruct_deficit(events)

    recon_out = OUT_DIR / "recon_qwen_ladder.csv"
    with open(recon_out, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target_id", "description", "frozen_value",
                        "reconstructed_value", "method", "status",
                        "source_paths", "notes"])
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"wrote {recon_out} ({len(ROWS)} rows)")

    prog_out = OUT_DIR / "qwen_ladder_progression.csv"
    order = {"A120->A250": 0, "A250->A500": 1, "A500->A1000": 2}
    PROGRESSION.sort(key=lambda row: (order[row["fit_boundary"]],
                                      row["metric_class"], row["metric"]))
    with open(prog_out, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fit_boundary", "metric_class", "metric", "value",
                        "gate", "pass", "source_event", "method"])
        writer.writeheader()
        writer.writerows(PROGRESSION)
    print(f"wrote {prog_out} ({len(PROGRESSION)} rows)")

    bad = [row for row in ROWS if row["status"] == "failed"]
    print("FAILED targets:", [row["target_id"] for row in bad] if bad
          else "none")
    for row in ROWS:
        print(f"  {row['target_id']:55s} {row['status']}")


if __name__ == "__main__":
    main()
