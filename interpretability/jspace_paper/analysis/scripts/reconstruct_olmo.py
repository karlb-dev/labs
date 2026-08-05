# reconstruct_olmo.py — deterministic reconstruction of the frozen OLMo
# lineage headline numbers (Phase 4 lineage study + Study 1 + Study 2)
# for the paper-analysis audit.
#
# Reads (read-only):
#   /content/drive/MyDrive/interpret/special-lab-1/phase4_20260731
#   /content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801
#   /content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_2_20260803
#   interpretability/jspace_phase4/reports/evidence_events.jsonl
#   interpretability/jspace_olmo_lineage/reports/evidence_events.jsonl
# Writes:
#   interpretability/jspace_paper/analysis/tables/recon_olmo.csv
#   interpretability/jspace_paper/analysis/tables/olmo_lineage_matrix_inputs.csv
#
# Reconstruction methods
#   recomputed_from_items          — recomputed from registered row-level data
#                                    (npz error curves, parquet item rows,
#                                    safetensors readout rows) or a full
#                                    deterministic rerun of the registered
#                                    simulation with its frozen seeds.
#   verified_registered_summary    — value cross-checked against the frozen
#                                    registered summary artifact only (no
#                                    row-level data exists for it).
# Statuses: byte_identical | numerically_identical_render_diff |
#   numerically_within_frozen_tolerance | failed |
#   not_reconstructable_from_released_data
#
# CPU-only, no network. Rerunnable: output is deterministic.
# Usage: python reconstruct_olmo.py [--skip-power-rerun]
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

REPO = Path("/content/labs")
LAB = Path("/content/drive/MyDrive/interpret/special-lab-1")
P4_RUN = LAB / "phase4_20260731"
OL_RUN = LAB / "olmo_lineage_20260801"
OL2_RUN = LAB / "olmo_lineage_2_20260803"
P4_REGISTRY = REPO / "interpretability/jspace_phase4/reports/evidence_events.jsonl"
OL_REGISTRY = (REPO / "interpretability/jspace_olmo_lineage/reports/"
               "evidence_events.jsonl")
OUT_DIR = REPO / "interpretability/jspace_paper/analysis/tables"
OL_PKG = REPO / "interpretability/jspace_olmo_lineage"

sys.path.insert(0, str(OL_PKG))
from jspace_olmo_lineage.capacity import (  # noqa: E402
    curve_summary,
    lower_median,
    percentile_interval,
)
from jspace_olmo_lineage.geometry import (  # noqa: E402
    neighbor_overlap,
    quantile_summary,
    row_cosines,
)
from jspace_olmo_lineage.manifests import object_sha256  # noqa: E402
from jspace_olmo_lineage.experiments.bank_w_pair_power import (  # noqa: E402
    _random_signs,
    simulate_rejection_rate,
    stable_seed,
)

import torch  # noqa: E402  (CPU wheel)

SLUGS = ["olmo3-base", "olmo3-think", "olmo31-think", "olmo31-instruct"]
LAYERS = [24, 32, 40]
FRAMES = {"own": "own", "base_common": "base_common"}
CKPT_LABEL = {
    "olmo3-base": "Base",
    "olmo3-think": "3.0 Think",
    "olmo31-think": "3.1 Think",
    "olmo31-instruct": "3.1 Instruct",
}

RECON_ROWS: list[dict] = []
MATRIX_ROWS: list[dict] = []
HASH_FAILURES: list[str] = []


# ----------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


P4_EVENTS = load_registry(P4_REGISTRY)
OL_EVENTS = load_registry(OL_REGISTRY)


def event(evidence_id: str, events: list[dict]) -> dict:
    hits = [e for e in events
            if e.get("evidence_id") == evidence_id
            and e.get("event") in ("evidence_created", "evidence_imported")]
    if len(hits) != 1:
        raise SystemExit(f"expected one creation event for {evidence_id}, "
                         f"found {len(hits)}")
    return hits[0]


def registered_path(evidence_id: str, name: str, events: list[dict],
                    verify: bool = True) -> Path:
    """Return the registered output path ending in `name`, sha-verified."""
    ev = event(evidence_id, events)
    outs = ev.get("outputs") or ev.get("source_outputs") or []
    matches = [o for o in outs if str(o.get("path", "")).endswith(name)]
    if len(matches) != 1:
        raise SystemExit(f"{evidence_id}: {len(matches)} outputs end in {name}")
    path = Path(matches[0]["path"])
    if not path.exists() and not str(path).startswith("/content/drive"):
        # repo-relative registration
        alt = REPO / matches[0]["path"]
        if alt.exists():
            path = alt
    if verify:
        observed = sha256_file(path)
        if observed != matches[0]["sha256"]:
            HASH_FAILURES.append(f"{evidence_id}:{name}")
    return path


def add_recon(target_id: str, description: str, frozen, recon, method: str,
              status: str, source_paths: str, notes: str = "") -> None:
    RECON_ROWS.append({
        "target_id": target_id,
        "description": description,
        "frozen_value": frozen,
        "reconstructed_value": recon,
        "method": method,
        "status": status,
        "source_paths": source_paths,
        "notes": notes,
    })


def add_matrix(checkpoint: str, metric_class: str, metric: str, frame: str,
               cohort: str, value, tier: str, source_evidence_id: str) -> None:
    MATRIX_ROWS.append({
        "checkpoint": checkpoint,
        "metric_class": metric_class,
        "metric": metric,
        "frame": frame,
        "cohort": cohort,
        "value": value,
        "tier": tier,
        "source_evidence_id": source_evidence_id,
    })


def status_number(frozen: float, recon: float, exact_tol: float = 0.0,
                  render_tol: float = 5e-7) -> str:
    if frozen == recon:
        return "byte_identical"
    if abs(frozen - recon) <= max(exact_tol, 0.0):
        return "numerically_within_frozen_tolerance"
    if abs(frozen - recon) <= render_tol * max(1.0, abs(frozen)):
        return "numerically_identical_render_diff"
    return "failed"


def envelope(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data.get("payload", data)


# ================================================================ 1. CAPACITY
def reconstruct_capacity() -> None:
    joint_path = registered_path("ol-capacity-joint-dev-v1",
                                 "ol-capacity-joint-dev-v1.json", OL_EVENTS)
    joint = envelope(joint_path)
    table = joint["table"]
    model_rows = {
        (r["left"], r["frame"], r["layer"]): r
        for r in table if r["row_type"] == "model_estimate"
    }
    recomputed = {}
    for slug in SLUGS:
        result_path = registered_path(f"ol-capacity-{slug}-dev-v1",
                                      "capacity_result.json", OL_EVENTS)
        result = envelope(result_path)
        for layer in LAYERS:
            npz_path = registered_path(f"ol-capacity-{slug}-dev-v1",
                                       f"capacity_layer_{layer}.npz", OL_EVENTS)
            with np.load(npz_path, allow_pickle=False) as data:
                errors = {
                    "own": np.array(data["own_centered_errors"]),
                    "base_common": np.array(data["common_centered_errors"]),
                }
                random_errors = np.array(data["random_centered_errors"])
                boot = {
                    "own": np.array(data["own_bootstrap_centered_excess"]),
                    "base_common": np.array(
                        data["common_bootstrap_centered_excess"]),
                }
            for frame in FRAMES:
                summary = curve_summary(errors[frame], random_errors,
                                        persistence=2)
                frozen_block = result["per_layer"][str(layer)][
                    frame]["primary_centered"]
                jrow = model_rows[(slug, frame, str(layer))]
                interval = percentile_interval(boot[frame], 0.9)
                boot_ci = (interval["low"], interval["high"])
                recomputed[(slug, frame, layer)] = {
                    "occ": summary["occupancy_median"],
                    "excess": summary["excess_share"],
                    "frozen_occ": frozen_block["occupancy_median"],
                    "frozen_excess": frozen_block["excess_share"],
                    "joint_occ": jrow["occupancy_median"],
                    "joint_excess": jrow["centered_excess"],
                    "boot_lo": float(boot_ci[0]),
                    "boot_hi": float(boot_ci[1]),
                    "frozen_lo": frozen_block["prompt_bootstrap"][
                        "excess_share"]["low"],
                    "frozen_hi": frozen_block["prompt_bootstrap"][
                        "excess_share"]["high"],
                    "n_positions": summary["n_positions"],
                }
    # recon rows: per checkpoint x layer, own + base_common
    ci_matches = 0
    ci_total = 0
    for slug in SLUGS:
        for layer in LAYERS:
            for frame in FRAMES:
                cell = recomputed[(slug, frame, layer)]
                src = (f"{OL_RUN}/metrics/{slug}/capacity/"
                       f"ol-capacity-{slug}-dev-v1/capacity_layer_{layer}.npz")
                joint_note = ("joint table agrees" if
                              (cell["joint_occ"] == cell["frozen_occ"]
                               and cell["joint_excess"] == cell["frozen_excess"])
                              else "JOINT TABLE DISAGREES WITH MODEL RESULT")
                add_recon(
                    f"cap_occupancy_{slug}_L{layer}_{frame}",
                    f"{CKPT_LABEL[slug]} L{layer} {frame} lower-median "
                    "crossing occupancy",
                    cell["frozen_occ"], cell["occ"], "recomputed_from_items",
                    ("byte_identical" if cell["occ"] == cell["frozen_occ"]
                     else "failed"),
                    src, joint_note)
                add_recon(
                    f"cap_excess_{slug}_L{layer}_{frame}",
                    f"{CKPT_LABEL[slug]} L{layer} {frame} centered excess "
                    "share (J minus matched-random)",
                    cell["frozen_excess"], cell["excess"],
                    "recomputed_from_items",
                    status_number(cell["frozen_excess"], cell["excess"]),
                    src, joint_note)
                ci_total += 1
                if (cell["boot_lo"] == cell["frozen_lo"]
                        and cell["boot_hi"] == cell["frozen_hi"]):
                    ci_matches += 1
                for name, value in (
                        ("occupancy_median", cell["occ"]),
                        ("centered_excess", cell["excess"]),
                        ("centered_excess_ci90_low", cell["frozen_lo"]),
                        ("centered_excess_ci90_high", cell["frozen_hi"])):
                    add_matrix(slug, "capacity", f"{name}_L{layer}", frame,
                               f"dev_corpus_{cell['n_positions']}pos",
                               value, "development",
                               f"ol-capacity-{slug}-dev-v1")
    add_recon(
        "cap_bootstrap_ci_cells",
        "capacity 90% prompt-bootstrap CI recompute from stored draw arrays "
        "(24 model/frame/layer cells)",
        f"{ci_total}/{ci_total}", f"{ci_matches}/{ci_total}",
        "recomputed_from_items",
        "byte_identical" if ci_matches == ci_total else "failed",
        str(OL_RUN / "metrics/*/capacity/*/capacity_layer_*.npz"),
        "np.quantile([0.05,0.95]) over registered 2000-draw arrays")

    # headline Base -> 3.0 Think equal-layer own-frame contrast
    headline = joint["headline_base_to_3_0_think"]
    diff = float(np.mean([
        recomputed[("olmo3-think", "own", layer)]["excess"]
        - recomputed[("olmo3-base", "own", layer)]["excess"]
        for layer in LAYERS]))
    occ_diff = float(np.mean([
        recomputed[("olmo3-think", "own", layer)]["occ"]
        - recomputed[("olmo3-base", "own", layer)]["occ"]
        for layer in LAYERS]))
    boot_npz = registered_path("ol-capacity-joint-dev-v1",
                               "ol-capacity-joint-dev-v1_bootstrap.npz",
                               OL_EVENTS)
    with np.load(boot_npz, allow_pickle=False) as data:
        eq = np.array(data["pair__olmo3-base__olmo3-think__own__equal_layer"
                           "__centered_difference"])
        eq_occ = np.array(data["pair__olmo3-base__olmo3-think__own"
                               "__equal_layer__occupancy_difference"])
    eq_ci = np.quantile(eq, [0.05, 0.95])
    eq_occ_ci = np.quantile(eq_occ, [0.05, 0.95])
    src = str(joint_path)
    add_recon("cap_headline_base_think_centered_diff",
              "Base->3.0 Think equal-layer own-frame centered-excess "
              "difference",
              headline["centered_difference"], diff, "recomputed_from_items",
              status_number(headline["centered_difference"], diff), src, "")
    add_recon("cap_headline_base_think_occupancy_diff",
              "Base->3.0 Think equal-layer own-frame occupancy difference",
              headline["occupancy_difference"], occ_diff,
              "recomputed_from_items",
              status_number(float(headline["occupancy_difference"]), occ_diff),
              src, "")
    ci_ok = (float(eq_ci[0]) == headline["centered_ci_low"]
             and float(eq_ci[1]) == headline["centered_ci_high"]
             and float(eq_occ_ci[0]) == headline["occupancy_ci_low"]
             and float(eq_occ_ci[1]) == headline["occupancy_ci_high"])
    add_recon("cap_headline_ci90",
              "headline contrast 90% CI from registered joint bootstrap "
              "distributions",
              f"[{headline['centered_ci_low']},{headline['centered_ci_high']}]",
              f"[{float(eq_ci[0])},{float(eq_ci[1])}]",
              "recomputed_from_items",
              "byte_identical" if ci_ok else "failed", str(boot_npz),
              "occupancy CI also checked")
    # classification per frozen decision rules
    rules = joint["decision_rules"]
    margin = rules["centered_excess_equivalence_margin"]
    classification = (
        "stable" if (abs(diff) < margin and occ_diff == 0
                     and eq_ci[0] >= -margin and eq_ci[1] <= margin)
        else "not-stable-by-recompute")
    add_recon("cap_headline_classification",
              "headline contrast classification under frozen decision rules",
              headline["classification"], classification,
              "recomputed_from_items",
              "byte_identical" if classification == headline["classification"]
              else "failed", src,
              f"stable requires |diff|<{margin}, occ unchanged, CI within "
              f"[-{margin},{margin}]")
    # verdict: aggregate stable + no positive material individual layer
    pair_rows = [r for r in table
                 if r["row_type"] == "pair_contrast" and r["frame"] == "own"
                 and r["left"] == "olmo3-base" and r["right"] == "olmo3-think"]
    material = rules["centered_excess_material_margin"]
    positive_material = [
        r["layer"] for r in pair_rows
        if r["centered_ci_low"] > material
        or (r["occupancy_ci_low"] is not None and r["occupancy_ci_low"] > 1.0)]
    verdict = ("broadly_conserved_capacity_recruitment_consistent"
               if classification in ("stable", "small_shift")
               and not positive_material else "other")
    add_recon("cap_lineage_verdict",
              "capacity lineage router verdict ('measured sparse capacity "
              "broadly conserved across the tested lineage')",
              joint["lineage_verdict"], verdict, "recomputed_from_items",
              "byte_identical" if verdict == joint["lineage_verdict"]
              else "failed", src,
              f"positive material layers recomputed: {positive_material}; "
              f"registered: {joint['positive_material_individual_layers']}")
    add_matrix("lineage", "capacity", "lineage_verdict", "own",
               "equal_layer_headline", joint["lineage_verdict"],
               "development", "ol-capacity-joint-dev-v1")
    add_matrix("olmo3-base__olmo3-think", "capacity",
               "centered_excess_equal_layer_difference", "own",
               "equal_layer_headline", headline["centered_difference"],
               "development", "ol-capacity-joint-dev-v1")
    add_matrix("olmo3-base__olmo3-think", "capacity",
               "occupancy_equal_layer_difference", "own",
               "equal_layer_headline", headline["occupancy_difference"],
               "development", "ol-capacity-joint-dev-v1")


# ================================================================ 2. GEOMETRY
def reconstruct_geometry() -> None:
    joint_path = registered_path("ol-geometry-joint-dev-v1",
                                 "ol-geometry-joint-dev-v1.json", OL_EVENTS)
    joint = json.loads(joint_path.read_text())
    layers = pd.read_parquet(registered_path(
        "ol-geometry-joint-dev-v1", "ol-geometry-joint-dev-v1_layers.parquet",
        OL_EVENTS))
    selection = pd.read_parquet(registered_path(
        "ol-geometry-joint-dev-v1",
        "ol-geometry-joint-dev-v1_selection.parquet", OL_EVENTS))
    layer_metrics = ["raw_matrix_cosine", "symmetric_relative_frobenius_delta",
                     "j_minus_identity_cosine", "j_minus_alpha_identity_cosine",
                     "probe_transport_cosine_q50", "mapped_token_cosine_q50",
                     "mapped_token_cosine_q05", "mapped_token_centered_linear_cka",
                     "mapped_neighbor_overlap_fraction"]
    sel_metrics = ["selected_id_jaccard_q50", "rank_biased_overlap_q50",
                   "projector_overlap_q50", "principal_angle_median_degrees_q50",
                   "persistent_direction_jaccard"]
    total = matched = 0
    recomputed_layer = {}
    for pair_id, group in layers.groupby("pair_id"):
        recomputed_layer[pair_id] = {}
        for metric in layer_metrics:
            agg = quantile_summary(group[metric].to_numpy(), (0.05, 0.5, 0.95))
            recomputed_layer[pair_id][metric] = agg
            frozen = joint["pairwise_layer_aggregate"][pair_id][metric]
            total += 1
            matched += int(all(agg[k] == frozen[k] for k in agg))
    recomputed_sel = {}
    for pair_id, group in selection.groupby("pair_id"):
        recomputed_sel[pair_id] = {}
        for metric in sel_metrics:
            agg = quantile_summary(group[metric].to_numpy(), (0.05, 0.5, 0.95))
            recomputed_sel[pair_id][metric] = agg
            frozen = joint["pairwise_selection_aggregate"][pair_id][metric]
            total += 1
            matched += int(all(agg[k] == frozen[k] for k in agg))
    add_recon("geo_aggregate_cells",
              "geometry pairwise layer+selection aggregates recomputed from "
              "per-layer parquet rows (6 pairs x 14 metrics x 7 stats)",
              f"{total}/{total}", f"{matched}/{total}",
              "recomputed_from_items",
              "byte_identical" if matched == total else "failed",
              str(OL_RUN / "metrics/geometry/ol-geometry-joint-dev-v1_"
                           "{layers,selection}.parquet"),
              "quantile_summary((0.05,0.5,0.95)) over 21 layer rows / 3 "
              "selection layers per pair")

    base_pairs = {"olmo3-think": "olmo3-base__olmo3-think",
                  "olmo31-think": "olmo3-base__olmo31-think",
                  "olmo31-instruct": "olmo3-base__olmo31-instruct"}
    for slug, pair_id in base_pairs.items():
        for tid, metric, table, desc in (
                ("opsim", "raw_matrix_cosine", recomputed_layer,
                 "operator (lens matrix) cosine to Base, median over 21 "
                 "layers"),
                ("maptok", "mapped_token_cosine_q50", recomputed_layer,
                 "mapped-token row cosine q50 to Base, median over 21 layers"),
                ("selid", "selected_id_jaccard_q50", recomputed_sel,
                 "selected-ID Jaccard q50 vs Base, median over 3 assay "
                 "layers"),
                ("projover", "projector_overlap_q50", recomputed_sel,
                 "selection projector overlap q50 vs Base, median over 3 "
                 "assay layers")):
            frozen_agg = (joint["pairwise_layer_aggregate"]
                          if table is recomputed_layer
                          else joint["pairwise_selection_aggregate"])
            frozen = frozen_agg[pair_id][metric]["q50"]
            recon = table[pair_id][metric]["q50"]
            add_recon(f"geo_{tid}_{slug}", f"{CKPT_LABEL[slug]}: {desc}",
                      frozen, recon, "recomputed_from_items",
                      status_number(frozen, recon), str(joint_path), "")
    # matrix rows: all six pairs
    for pair_id in recomputed_layer:
        for metric in ("raw_matrix_cosine", "mapped_token_cosine_q50",
                       "mapped_token_centered_linear_cka"):
            add_matrix(pair_id, "geometry", f"{metric}_q50_layers", "pair",
                       "lens_layers_21",
                       joint["pairwise_layer_aggregate"][pair_id][metric]["q50"],
                       "development", "ol-geometry-joint-dev-v1")
        for metric in ("selected_id_jaccard_q50", "projector_overlap_q50",
                       "persistent_direction_jaccard"):
            add_matrix(pair_id, "geometry", f"{metric}_q50_layers", "pair",
                       "assay_layers_3",
                       joint["pairwise_selection_aggregate"][pair_id][metric]["q50"],
                       "development", "ol-geometry-joint-dev-v1")

    # readout rows: genuine row-level recompute from safetensors
    from safetensors import safe_open
    manifest = json.loads(
        (OL_RUN / "manifests/ol_geometry_row_manifest_v1.json").read_text())
    stable_ids = manifest["stable_sample_token_ids"]
    lookup = {int(t): i for i, t in enumerate(
        manifest["all_extracted_token_ids"])}
    stable_idx = torch.tensor([lookup[int(t)] for t in stable_ids])
    eps = 1e-6
    readouts = {}
    for slug in SLUGS:
        tensor_path = registered_path(f"ol-geometry-readout-{slug}-v1",
                                      "readout_rows.safetensors", OL_EVENTS)
        with safe_open(tensor_path, framework="pt") as handle:
            rows = handle.get_tensor("lm_head_rows")
            norm = handle.get_tensor("final_norm_weight")
            token_ids = handle.get_tensor("token_ids")
        if token_ids.tolist() != manifest["all_extracted_token_ids"]:
            raise SystemExit(f"{slug}: readout token order != row manifest")
        readouts[slug] = {"rows": rows, "norm": norm}
    readout_pairs = {r["pair_id"]: r for r in joint["readout_pairs"]}
    readout_total = readout_matched = 0
    for pair_id, frozen in readout_pairs.items():
        left, right = pair_id.split("__")
        lraw = readouts[left]["rows"][stable_idx].float()
        rraw = readouts[right]["rows"][stable_idx].float()
        lgain = readouts[left]["norm"].float() / float(np.sqrt(1.0 + eps))
        rgain = readouts[right]["norm"].float() / float(np.sqrt(1.0 + eps))
        leff = readouts[left]["rows"].float() * lgain[None, :]
        reff = readouts[right]["rows"].float() * rgain[None, :]
        leff, reff = leff[stable_idx], reff[stable_idx]
        raw_q = quantile_summary(row_cosines(lraw, rraw),
                                 (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
        eff_q = quantile_summary(row_cosines(leff, reff),
                                 (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
        neighbors = neighbor_overlap(
            torch.nn.functional.normalize(leff, dim=1),
            torch.nn.functional.normalize(reff, dim=1), k=20)
        checks = [
            ("raw_unembedding_row_cosine_q50", raw_q["q50"]),
            ("effective_unembedding_row_cosine_q50", eff_q["q50"]),
            ("raw_unembedding_row_cosine_q05", raw_q["q05"]),
            ("effective_unembedding_row_cosine_q05", eff_q["q05"]),
            ("plain_neighbor_overlap_fraction",
             neighbors["overlap_fraction_mean"]),
        ]
        agree = all(
            abs(frozen[name] - value) <= 2e-6 * max(1.0, abs(frozen[name]))
            for name, value in checks)
        exact = all(frozen[name] == value for name, value in checks)
        readout_total += 1
        readout_matched += int(agree)
        readout_exact = globals().setdefault("_READOUT_EXACT", [0])
        readout_exact[0] += int(exact)
        if pair_id in ("olmo3-base__olmo3-think", "olmo3-base__olmo31-think",
                       "olmo3-base__olmo31-instruct"):
            slug = pair_id.split("__")[1]
            frozen_value = frozen["effective_unembedding_row_cosine_q50"]
            add_recon(f"geo_readout_eff_{slug}",
                      f"{CKPT_LABEL[slug]}: effective unembedding row cosine "
                      "q50 vs Base (1024 stable rows)",
                      frozen_value, eff_q["q50"], "recomputed_from_items",
                      status_number(frozen_value, eff_q["q50"],
                                    render_tol=2e-6),
                      str(OL_RUN / f"metrics/{slug}/geometry_readout/"
                          f"ol-geometry-readout-{slug}-v1/"
                          "readout_rows.safetensors"),
                      "recomputed from lm_head rows x final_norm gain, "
                      "eps=1e-6; original computed on GPU floats")
        add_matrix(pair_id, "geometry", "readout_effective_row_cosine_q50",
                   "pair", "stable_vocab_1024",
                   frozen["effective_unembedding_row_cosine_q50"],
                   "development", "ol-geometry-joint-dev-v1")
    readout_exact = globals().get("_READOUT_EXACT", [0])[0]
    add_recon("geo_readout_pairs",
              "readout pair blocks recomputed from registered safetensors "
              "(6 pairs; raw/effective q05+q50, neighbor overlap)",
              f"{readout_total}/{readout_total}",
              f"{readout_matched}/{readout_total}", "recomputed_from_items",
              "byte_identical" if readout_exact == readout_total
              else ("numerically_within_frozen_tolerance"
                    if readout_matched == readout_total else "failed"),
              str(OL_RUN / "metrics/*/geometry_readout/*/"
                           "readout_rows.safetensors"),
              f"{readout_exact}/{readout_total} pairs exact to the last bit "
              "on CPU float32")
    # router verdict
    router = joint["router"]
    primary = layers[layers.edge_type == "primary"]
    base_movement = 1.0 - float(np.median(primary.mapped_token_cosine_q50))
    traj = layers[(layers.left == "olmo3-think")
                  & (layers.right == "olmo31-think")]
    later_movement = 1.0 - float(np.median(traj.mapped_token_cosine_q50))
    add_recon("geo_router_base_to_30_movement",
              "router: Base->3.0 Think mapped-token movement "
              "(1 - median q50 cosine)",
              router["base_to_3_0_mapped_movement"], base_movement,
              "recomputed_from_items",
              status_number(router["base_to_3_0_mapped_movement"],
                            base_movement), str(joint_path), "")
    add_recon("geo_router_30_to_31_movement",
              "router: 3.0->3.1 Think mapped-token movement",
              router["3_0_to_3_1_mapped_movement"], later_movement,
              "recomputed_from_items",
              status_number(router["3_0_to_3_1_mapped_movement"],
                            later_movement), str(joint_path), "")
    formation = (base_movement >= 1.5 * max(later_movement, 1e-12)
                 and base_movement - later_movement >= 0.03)
    verdict = ("dictionary-formation-pattern" if formation
               else "not-formation-by-recompute")
    add_recon("geo_router_verdict", "geometry router verdict",
              router["verdict"], verdict, "recomputed_from_items",
              "byte_identical" if verdict == router["verdict"] else "failed",
              str(joint_path),
              f"dictionary_formation_pattern={formation}; registered "
              f"{router['dictionary_formation_pattern']}")
    add_matrix("lineage", "geometry", "router_verdict", "pair", "all_pairs",
               router["verdict"], "development", "ol-geometry-joint-dev-v1")

    # lens-frame agreement per non-base checkpoint (p4 grids, row-level)
    frame_events = {
        "olmo3-think": ("p4-lens-frame-analysis-olmo3-think-dev-v2",
                        "p4-lineage-grid-olmo3-think-dev-v1",
                        "lineage_grid_olmo3-think.parquet",
                        "p4-lineage-grid-olmo3-think-common-base-lens-dev-v4",
                        "lineage_grid_olmo3-think-common-base-lens.parquet"),
        "olmo31-think": ("p4-lens-frame-analysis-olmo31-think-dev-v1",
                         "p4-lineage-grid-olmo31-think-dev-v1",
                         "lineage_grid_olmo31-think.parquet",
                         "p4-lineage-grid-olmo31-think-common-base-lens-dev-v1",
                         "lineage_grid_olmo31-think-common-base-lens.parquet"),
        "olmo31-instruct": (
            "p4-lens-frame-analysis-olmo31-instruct-dev-v1",
            "p4-lineage-grid-olmo31-instruct-dev-v1",
            "lineage_grid_olmo31-instruct.parquet",
            "p4-lineage-grid-olmo31-instruct-common-base-lens-dev-v1",
            "lineage_grid_olmo31-instruct-common-base-lens.parquet"),
    }
    for slug, (fa_id, own_id, own_name, common_id, common_name) in (
            frame_events.items()):
        fa_path = registered_path(fa_id, f"lens_frame_analysis_"
                                  f"{slug}-lens-frame.json", P4_EVENTS)
        fa = envelope(fa_path)
        own = pd.read_parquet(registered_path(own_id, own_name, P4_EVENTS))
        common = pd.read_parquet(registered_path(common_id, common_name,
                                                 P4_EVENTS))
        for frame_df in (own, common):
            frame_df["specific"] = (frame_df.lp_meanJ_span_safe
                                    - frame_df.lp_ss_matched)
        keys = ["item_id", "bank", "fact_id", "canonical_family", "variant"]
        paired = own[keys + ["specific"]].merge(
            common[keys + ["specific"]], on=keys, suffixes=("_own", "_common"),
            validate="one_to_one")
        item_pearson = float(np.corrcoef(paired.specific_own,
                                         paired.specific_common)[0, 1])
        fam = paired.groupby(["bank", "variant", "canonical_family"],
                             as_index=False)[
            ["specific_own", "specific_common"]].mean()
        family_pearson = float(np.corrcoef(fam.specific_own,
                                           fam.specific_common)[0, 1])
        frozen = fa["specific_frame_agreement"]
        add_recon(f"geo_frame_agreement_{slug}",
                  f"{CKPT_LABEL[slug]}: own-vs-common-lens specific-effect "
                  "item Pearson r",
                  frozen["item_pearson"], item_pearson,
                  "recomputed_from_items",
                  status_number(frozen["item_pearson"], item_pearson),
                  str(fa_path),
                  f"family_pearson recon {family_pearson:.9f} vs frozen "
                  f"{frozen['family_pearson']:.9f}")
        add_matrix(slug, "geometry_frame_agreement", "item_pearson",
                   "own_vs_base_common", f"paired_items_{len(paired)}",
                   frozen["item_pearson"], "development", fa_id)
        add_matrix(slug, "geometry_frame_agreement", "family_pearson",
                   "own_vs_base_common", f"families_{len(fam)}",
                   frozen["family_pearson"], "development", fa_id)


# ==================================================== 3. COMMON-COHORT TRAJECTORY
def reconstruct_common_cohort() -> None:
    cc_id = "p4-lineage-common-cohort-analysis-olmo-dev-v1"
    cc_path = registered_path(
        cc_id, "common_cohort_analysis_olmo-lineage-common-cohort.json",
        P4_EVENTS)
    cc = envelope(cc_path)
    estimates = pd.DataFrame(cc["checkpoint_estimates"])
    grids = {}
    grid_specs = {
        ("olmo3-base", "own"): ("p4-lineage-grid-olmo3-base-dev-v1",
                                "lineage_grid_olmo3-base.parquet"),
        ("olmo3-base", "common"): ("p4-lineage-grid-olmo3-base-dev-v1",
                                   "lineage_grid_olmo3-base.parquet"),
        ("olmo3-think", "own"): ("p4-lineage-grid-olmo3-think-dev-v1",
                                 "lineage_grid_olmo3-think.parquet"),
        ("olmo3-think", "common"): (
            "p4-lineage-grid-olmo3-think-common-base-lens-dev-v4",
            "lineage_grid_olmo3-think-common-base-lens.parquet"),
        ("olmo31-think", "own"): ("p4-lineage-grid-olmo31-think-dev-v1",
                                  "lineage_grid_olmo31-think.parquet"),
        ("olmo31-think", "common"): (
            "p4-lineage-grid-olmo31-think-common-base-lens-dev-v1",
            "lineage_grid_olmo31-think-common-base-lens.parquet"),
        ("olmo31-instruct", "own"): (
            "p4-lineage-grid-olmo31-instruct-dev-v1",
            "lineage_grid_olmo31-instruct.parquet"),
        ("olmo31-instruct", "common"): (
            "p4-lineage-grid-olmo31-instruct-common-base-lens-dev-v1",
            "lineage_grid_olmo31-instruct-common-base-lens.parquet"),
    }
    for key, (eid, name) in grid_specs.items():
        frame = pd.read_parquet(registered_path(eid, name, P4_EVENTS))
        frame["specific"] = frame.lp_meanJ_span_safe - frame.lp_ss_matched
        grids[key] = frame

    def complete_facts(frame: pd.DataFrame) -> set:
        counts = frame.groupby("fact_id").variant.agg(
            lambda values: tuple(sorted(values)))
        return set(counts[counts == ("composed", "direct")].index)

    populations = {p["key"]: p for p in cc["populations"]}
    # membership derived from the *common*-frame grids (frozen population)
    pop_sets = {}
    for pop_key, pop in populations.items():
        sets = [complete_facts(grids[(ck, "common")])
                for ck in pop["checkpoints"]]
        fact_ids = sorted(set.intersection(*sets))
        pop_sets[pop_key] = fact_ids
        if pop_key == "all_four":
            reference = grids[("olmo3-base", "common")]
            reference = reference[reference.fact_id.isin(fact_ids)]
            meta = reference[["fact_id", "canonical_family", "bank"]
                             ].drop_duplicates()
            recon_pop = {
                "n_facts": len(fact_ids),
                "n_families": int(meta.canonical_family.nunique()),
                "s_facts": int((meta.bank == "S").sum()),
                "s_families": int(
                    meta[meta.bank == "S"].canonical_family.nunique()),
                "sha": object_sha256(fact_ids),
            }
            frozen_pop = {
                "n_facts": pop["n_facts"],
                "n_families": pop["n_families"],
                "s_facts": pop["facts_by_bank"]["S"],
                "s_families": pop["families_by_bank"]["S"],
                "sha": pop["fact_id_set_sha256"],
            }
            add_recon(
                "cc_population_all_four",
                "all-four common cohort membership (facts/families/Bank-S "
                "facts/Bank-S families/fact-set sha256)",
                json.dumps(frozen_pop, sort_keys=True),
                json.dumps(recon_pop, sort_keys=True),
                "recomputed_from_items",
                "byte_identical" if recon_pop == frozen_pop else "failed",
                str(P4_RUN / "metrics/*/lineage_grid/*"), "")

    def fact_rows(frame: pd.DataFrame, fact_ids, bank: str, metric: str
                  ) -> pd.DataFrame:
        subset = frame[frame.fact_id.isin(fact_ids)
                       & (frame.bank == bank)
                       & frame.variant.isin(["direct", "composed"])]
        if metric in ("direct", "composed"):
            rows = subset[subset.variant == metric][
                ["fact_id", "canonical_family", "specific"]].copy()
            return rows.rename(columns={"specific": "value"})
        pivot = subset.pivot(index=["fact_id", "canonical_family"],
                             columns="variant", values="specific")
        out = pivot.reset_index()
        out["value"] = out["composed"] - out["direct"]
        return out

    checked = exact = 0
    ci_exact = 0
    recon_values = {}
    for row in estimates.itertuples():
        frame_key = (row.checkpoint, row.frame)
        rows = fact_rows(grids[frame_key], pop_sets[row.population],
                         row.bank, row.metric)
        if row.weighting == "equal_family":
            vector = rows.groupby("canonical_family", sort=True)[
                "value"].mean().to_numpy(dtype=float)
        else:
            vector = rows["value"].to_numpy(dtype=float)
        estimate = float(vector.mean())
        generator = np.random.Generator(np.random.PCG64(
            int(row.bootstrap_seed)))
        indices = generator.integers(0, len(vector),
                                     size=(int(row.n_bootstrap), len(vector)))
        distribution = vector[indices].mean(axis=1)
        low, high = np.quantile(distribution, [0.025, 0.975])
        digest = hashlib.sha256(
            np.asarray(distribution, dtype="<f8").tobytes()).hexdigest()
        checked += 1
        exact += int(estimate == row.estimate and len(vector) == row.n_units)
        ci_exact += int(float(low) == row.ci95_low
                        and float(high) == row.ci95_high
                        and digest == row.distribution_sha256)
        recon_values[(row.population, row.frame, row.bank, row.metric,
                      row.weighting, row.checkpoint)] = (
            estimate, float(low), float(high))
    add_recon("cc_point_estimates_all",
              "common-cohort checkpoint estimates (240 rows: population x "
              "frame x bank x metric x weighting x checkpoint)",
              f"{checked}/{checked}", f"{exact}/{checked}",
              "recomputed_from_items",
              "byte_identical" if exact == checked else "failed",
              str(cc_path), "point estimates + n_units from grid rows")
    add_recon("cc_bootstrap_ci_all",
              "common-cohort 95% bootstrap CIs + distribution sha256 "
              "(240 rows, PCG64 registered seeds, 100000 draws)",
              f"{checked}/{checked}", f"{ci_exact}/{checked}",
              "recomputed_from_items",
              "byte_identical" if ci_exact == checked else "failed",
              str(cc_path), "exact digest match of bootstrap distributions")
    # headline Bank-S rows on the all-four cohort
    for ckpt in SLUGS:
        for frame in ("own", "common"):
            for metric in ("direct", "composed", "composition"):
                for bank in ("S", "F"):
                    key = ("all_four", frame, bank, metric, "equal_family",
                           ckpt)
                    frozen_row = estimates[
                        (estimates.population == "all_four")
                        & (estimates.frame == frame)
                        & (estimates.bank == bank)
                        & (estimates.metric == metric)
                        & (estimates.weighting == "equal_family")
                        & (estimates.checkpoint == ckpt)].iloc[0]
                    est, low, high = recon_values[key]
                    if bank == "S" and metric in ("direct", "composition"):
                        add_recon(
                            f"cc_S_{metric}_{ckpt}_{frame}",
                            f"{CKPT_LABEL[ckpt]} Bank-S {metric} specific "
                            f"effect, {frame} frame, all-four cohort, "
                            "equal-family",
                            frozen_row.estimate, est, "recomputed_from_items",
                            status_number(frozen_row.estimate, est),
                            str(cc_path),
                            f"ci95 recon [{low:.6f},{high:.6f}] frozen "
                            f"[{frozen_row.ci95_low:.6f},"
                            f"{frozen_row.ci95_high:.6f}]")
                    add_matrix(ckpt, "causal_trajectory",
                               f"bank_{bank}_{metric}_specific", frame,
                               "all_four_common_cohort_equal_family",
                               frozen_row.estimate, "development", cc_id)
                    if bank == "S":
                        add_matrix(ckpt, "causal_trajectory",
                                   f"bank_{bank}_{metric}_specific_ci95_low",
                                   frame,
                                   "all_four_common_cohort_equal_family",
                                   frozen_row.ci95_low, "development", cc_id)
                        add_matrix(ckpt, "causal_trajectory",
                                   f"bank_{bank}_{metric}_specific_ci95_high",
                                   frame,
                                   "all_four_common_cohort_equal_family",
                                   frozen_row.ci95_high, "development", cc_id)
    # registered CSV vs JSON consistency
    csv_path = registered_path(cc_id, "checkpoint_estimates.csv", P4_EVENTS)
    csv_df = pd.read_csv(csv_path)
    join_cols = ["population", "frame", "bank", "metric", "weighting",
                 "checkpoint"]
    merged = estimates.merge(csv_df, on=join_cols, suffixes=("_json", "_csv"))
    exact_csv = (len(merged) == len(estimates)
                 and (merged.estimate_json == merged.estimate_csv).all()
                 and (merged.distribution_sha256_json
                      == merged.distribution_sha256_csv).all())
    render_csv = (len(merged) == len(estimates)
                  and (merged.distribution_sha256_json
                       == merged.distribution_sha256_csv).all()
                  and (merged.bootstrap_seed_json
                       == merged.bootstrap_seed_csv).all()
                  and np.allclose(merged.estimate_json, merged.estimate_csv,
                                  rtol=0, atol=1e-12))
    add_recon("cc_csv_vs_json",
              "registered checkpoint_estimates.csv consistent with JSON "
              "payload", "consistent",
              "consistent" if render_csv else "inconsistent",
              "verified_registered_summary",
              "byte_identical" if exact_csv else (
                  "numerically_identical_render_diff" if render_csv
                  else "failed"),
              str(csv_path),
              f"{len(merged)} joined rows; CSV float rendering differs from "
              "JSON at <=1e-16; seeds and distribution sha256 identical")
    # per-checkpoint own-cohort trajectory (p4-lineage-trajectory) — matrix
    traj_id = "p4-lineage-trajectory-analysis-olmo-dev-v1"
    traj = pd.read_csv(registered_path(
        traj_id, "trajectory_table_olmo-lineage-trajectory.csv", P4_EVENTS))
    for row in traj.itertuples():
        add_matrix(row.checkpoint_key, "causal_trajectory",
                   f"bank_{row.bank}_{row.variant}_specific", row.frame,
                   "per_checkpoint_capability_cohort_equal_family",
                   row.estimate, "development", traj_id)


# ================================================================ 4. STAGE WEDGE
def reconstruct_stage_wedge() -> None:
    stages = {
        "think_sft": ("ol2-stage-wedge-think-sft-tier1-v1", 0.006172839506172839,
                      0.008333333333333333),
        "think_dpo": ("ol2-stage-wedge-think-dpo-tier1-v1",
                      0.0030864197530864196, 0.002777777777777778),
    }
    frames = {}
    for stage, (eid, _, _) in stages.items():
        result = envelope(registered_path(eid, "stage_result.json", OL_EVENTS))
        rows = pd.read_parquet(registered_path(eid, "g5_capability.parquet",
                                               OL_EVENTS))
        frames[stage] = rows
        capability = result["capability"]
        rate = float(rows.capable_generation.mean())
        s_rate = float(rows[rows.bank == "S"].capable_generation.mean())
        s_facts = rows[(rows.bank == "S")
                       & rows.variant.isin(["direct", "composed"])]
        fully = int(s_facts.groupby("fact_id").capable_generation.all().sum())
        src = str(OL2_RUN / f"metrics/stage-wedge/olmo3-{stage.replace('_','-')}"
                  f"/{eid}/g5_capability.parquet")
        add_recon(f"wedge_{stage}_capable_rate",
                  f"{stage} overall exact-generation capability rate "
                  "(972-item battery)",
                  capability["capable_rate"], rate, "recomputed_from_items",
                  status_number(capability["capable_rate"], rate), src,
                  f"{int(rows.capable_generation.sum())}/972 items")
        add_recon(f"wedge_{stage}_bank_s_rate",
                  f"{stage} Bank-S capability rate",
                  capability["capable_by_bank"]["S"], s_rate,
                  "recomputed_from_items",
                  status_number(capability["capable_by_bank"]["S"], s_rate),
                  src, "")
        add_recon(f"wedge_{stage}_bank_s_fully_capable_facts",
                  f"{stage} Bank-S facts capable on both direct and composed",
                  capability["fully_capable_direct_composed_facts_by_bank"]["S"],
                  fully, "recomputed_from_items",
                  "byte_identical"
                  if fully == capability[
                      "fully_capable_direct_composed_facts_by_bank"]["S"]
                  else "failed", src, "")
        add_matrix(f"olmo3-{stage.replace('_', '-')}", "stage_wedge",
                   "capable_rate", "generation", "battery_972",
                   capability["capable_rate"], "development", eid)
        add_matrix(f"olmo3-{stage.replace('_', '-')}", "stage_wedge",
                   "bank_s_capable_rate", "generation", "battery_972",
                   capability["capable_by_bank"]["S"], "development", eid)
        add_matrix(f"olmo3-{stage.replace('_', '-')}", "stage_wedge",
                   "bank_s_direct_composed_capable_facts", "generation",
                   "battery_972",
                   capability["fully_capable_direct_composed_facts_by_bank"]["S"],
                   "development", eid)
        add_matrix(f"olmo3-{stage.replace('_', '-')}", "stage_wedge",
                   "bank_s_intervention_effect", "own",
                   "stage_cohort", "capability_gated_missing", "development",
                   eid)
    joint_id = "ol2-stage-wedge-joint-analysis-v1"
    joint = envelope(registered_path(joint_id,
                                     "stage_wedge_joint_analysis.json",
                                     OL_EVENTS))
    transition = joint["capability_transition"]
    sft = frames["think_sft"].set_index("item_id").capable_generation
    dpo = frames["think_dpo"].set_index("item_id").capable_generation
    if not sft.index.equals(dpo.index):
        dpo = dpo.reindex(sft.index)
    counts = {
        "incapable_both": int((~sft & ~dpo).sum()),
        "capable_both": int((sft & dpo).sum()),
        "lost_at_dpo": int((sft & ~dpo).sum()),
        "onset_at_dpo": int((~sft & dpo).sum()),
    }
    frozen_counts = transition["overall_transition_counts"]
    frozen_full = {"incapable_both": frozen_counts.get("incapable_both", 0),
                   "capable_both": frozen_counts.get("capable_both", 0),
                   "lost_at_dpo": frozen_counts.get("lost_at_dpo", 0),
                   "onset_at_dpo": frozen_counts.get("onset_at_dpo", 0)}
    src = str(OL2_RUN / "metrics/stage-wedge/*/*/g5_capability.parquet")
    for name, frozen in frozen_full.items():
        add_recon(f"wedge_joint_{name}",
                  f"SFT/DPO paired transition count: {name} "
                  "(lost_at_dpo == capable-SFT-not-DPO)",
                  frozen, counts[name], "recomputed_from_items",
                  "byte_identical" if counts[name] == frozen else "failed",
                  src, "paired on item_id across 972-item batteries")
        add_matrix("think_sft__think_dpo", "stage_wedge",
                   f"transition_{name}", "generation", "battery_972_paired",
                   frozen, "development", joint_id)
    onset = frames["think_dpo"][~sft.reindex(
        frames["think_dpo"].item_id).to_numpy()
        & frames["think_dpo"].capable_generation.to_numpy()]
    onset_desc = ";".join(sorted(onset.bank + ":" + onset.variant))
    add_recon("wedge_joint_onset_detail",
              "the single onset-at-DPO item is Bank-F bridge_supplied",
              "F:bridge_supplied", onset_desc, "recomputed_from_items",
              "byte_identical" if onset_desc == "F:bridge_supplied"
              else "failed", src,
              "frozen claim: 1 Bank-F onset at DPO")
    add_recon("wedge_route", "stage-wedge joint router route",
              "null_or_unresolved", joint["router"]["route"],
              "verified_registered_summary",
              "byte_identical"
              if joint["router"]["route"] == "null_or_unresolved"
              else "failed",
              str(OL2_RUN / f"metrics/stage-wedge-joint/{joint_id}/"
                  "stage_wedge_joint_analysis.json"),
              "effects blocked: " + joint["effects"]["reason"][:80])
    add_matrix("think_sft__think_dpo", "stage_wedge", "route", "generation",
               "battery_972_paired", joint["router"]["route"], "development",
               joint_id)
    add_matrix("think_sft__think_dpo", "stage_wedge",
               "stage_effect_contrasts", "own", "stage_cohort",
               "capability_gated_missing", "development", joint_id)


# ================================================================ 5. H6 TRANSPORT
def reconstruct_transport() -> None:
    joint_id = "ol2-transport-validation-joint-v1"
    joint = envelope(registered_path(joint_id, "transport_joint_result.json",
                                     OL_EVENTS))
    summaries = joint["checkpoint_summaries"]
    frozen_claims = {
        "base": {"eid": "ol2-transport-validation-base-v1",
                 "meas": 34, "dec": 13, "pass": 9,
                 "maxerr": 0.02216238880688507,
                 "route": "no_common_licensed_regime_measured",
                 "l56_frac": 0.75, "l56_gate": False},
        "olmo31_think": {"eid": "ol2-transport-validation-olmo31-think-v1",
                         "meas": 77, "dec": 37, "pass": 12,
                         "maxerr": 0.025295454600777884,
                         "route": "late_only_regime_measured",
                         "l56_frac": 1.0, "l56_gate": True},
    }
    ceiling = 0.07870368901355948
    all_under = True
    total_rows = 0
    for key, claim in frozen_claims.items():
        summary = summaries[key]
        rows = pd.read_parquet(registered_path(claim["eid"],
                                               "transport_rows.parquet",
                                               OL_EVENTS))
        result = envelope(registered_path(claim["eid"],
                                          "transport_result.json", OL_EVENTS))
        total_rows += len(rows)
        all_under &= bool((rows.backend_tangent_relative_error
                           <= ceiling).all())
        meas = int(rows.measurement_eligible.sum())
        dec = int(rows.decision_eligible.sum())
        passing = int(rows.transport_row_passed.sum())
        maxerr = float(rows.backend_tangent_relative_error.max())
        l56 = rows[(rows.source_layer == 56)
                   & (np.isclose(rows.desired_relative_epsilon, 0.10))]
        l56_frac = float(l56.transport_row_passed.mean())
        l56_gate = bool(l56_frac >= 0.90)
        src = str(OL2_RUN / f"metrics/transport-validation/*/{claim['eid']}/"
                  "transport_rows.parquet")
        checkpoint = "olmo3-base" if key == "base" else "olmo31-think"
        for tid, desc, frozen, recon in (
                ("measurement_eligible", "measurement-eligible rows of 336",
                 claim["meas"], meas),
                ("decision_eligible", "decision-eligible rows of 336",
                 claim["dec"], dec),
                ("passing_rows", "transport rows passing of 336",
                 claim["pass"], passing)):
            registered = summary[{"measurement_eligible": "measurement_eligible_rows",
                                  "decision_eligible": "decision_eligible_rows",
                                  "passing_rows": "passing_rows"}[tid]]
            add_recon(f"h6_{key}_{tid}", f"{checkpoint} H6: {desc}",
                      frozen, recon, "recomputed_from_items",
                      "byte_identical"
                      if recon == frozen == registered else "failed",
                      src, f"joint summary {registered}")
            add_matrix(checkpoint, "h6_transport", tid, "residual_stream",
                       "frozen_ladder_336rows", frozen, "methods",
                       claim["eid"])
        add_recon(f"h6_{key}_max_backend_relative_error",
                  f"{checkpoint} H6: max backend tangent relative error",
                  claim["maxerr"], maxerr, "recomputed_from_items",
                  status_number(claim["maxerr"], maxerr), src,
                  "claim text rounds to "
                  + ("0.02216" if key == "base" else "0.02530"))
        l56_status = ("byte_identical"
                      if l56_frac == claim["l56_frac"]
                      and l56_gate == claim["l56_gate"]
                      and len(l56) == 12 else "failed")
        add_recon(f"h6_{key}_l56_eps010",
                  f"{checkpoint} H6: L56 eps-0.10 passage fraction "
                  "(12-row cell) vs 0.90 floor",
                  f"{claim['l56_frac']} (gate_passed={claim['l56_gate']})",
                  f"{l56_frac} (gate_passed={l56_gate}; "
                  f"{int(l56.transport_row_passed.sum())}/{len(l56)})",
                  "recomputed_from_items", l56_status, src,
                  "row_passage_floor=0.90 from frozen config")
        add_recon(f"h6_{key}_route", f"{checkpoint} intrinsic transport route",
                  claim["route"], result["intrinsic_transport_route"],
                  "verified_registered_summary",
                  "byte_identical"
                  if result["intrinsic_transport_route"] == claim["route"]
                  else "failed",
                  src.replace("transport_rows.parquet",
                              "transport_result.json"),
                  f"late_anchor_valid_epsilons={result['late_anchor_valid_epsilons']}")
        add_matrix(checkpoint, "h6_transport",
                   "max_backend_tangent_relative_error", "residual_stream",
                   "frozen_ladder_336rows", claim["maxerr"], "methods",
                   claim["eid"])
        add_matrix(checkpoint, "h6_transport", "l56_eps0p10_passage_fraction",
                   "residual_stream", "cell_12rows", l56_frac, "methods",
                   claim["eid"])
        add_matrix(checkpoint, "h6_transport", "route", "residual_stream",
                   "frozen_ladder_336rows",
                   result["intrinsic_transport_route"], "methods",
                   claim["eid"])
        add_matrix(checkpoint, "h6_transport", "common_assay_valid_epsilons",
                   "residual_stream", "frozen_ladder_336rows",
                   ("none_measured" if not result["common_assay_valid_epsilons"]
                    else str(result["common_assay_valid_epsilons"])),
                   "methods", claim["eid"])
        add_matrix(checkpoint, "h6_transport", "late_anchor_valid_epsilons",
                   "residual_stream", "frozen_ladder_336rows",
                   ("none_measured" if not result["late_anchor_valid_epsilons"]
                    else str(result["late_anchor_valid_epsilons"])),
                   "methods", claim["eid"])
    add_recon("h6_all_rows_under_imported_ceiling",
              "all 672 transport rows under imported backend ceiling "
              "0.07870368901355948",
              f"True (672 rows)", f"{all_under} ({total_rows} rows)",
              "recomputed_from_items",
              "byte_identical" if all_under and total_rows == 672
              else "failed",
              str(OL2_RUN / "metrics/transport-validation/*/*/"
                  "transport_rows.parquet"),
              "ceiling imported from gm2-backend-parity-calibration-v1")
    add_recon("h6_joint_route", "H6 joint route",
              "h6_fail_in_band_with_checkpoint_specific_late_anchor",
              joint["router"]["intrinsic_joint_route"],
              "verified_registered_summary",
              "byte_identical"
              if joint["router"]["intrinsic_joint_route"]
              == "h6_fail_in_band_with_checkpoint_specific_late_anchor"
              else "failed",
              str(OL2_RUN / f"metrics/transport-validation-joint/{joint_id}/"
                  "transport_joint_result.json"), "")
    add_matrix("joint", "h6_transport", "intrinsic_joint_route",
               "residual_stream", "both_checkpoints_672rows",
               joint["router"]["intrinsic_joint_route"], "methods", joint_id)
    add_matrix("joint", "h6_transport", "relevant_dose_route",
               "residual_stream", "registered_dose_tables",
               joint["router"]["relevant_dose_route"], "methods", joint_id)
    add_matrix("joint", "h6_transport", "backend_relative_error_ceiling",
               "residual_stream", "imported_gemma_calibration", ceiling,
               "methods", "ol2-gemma-backend-calibration-import-v1")
    add_matrix("joint", "h6_transport",
               "intervention_distribution_coverage", "residual_stream",
               "registered_dose_tables", "unresolved_missing_registered_"
               "site_dose_records", "methods", joint_id)


# ================================================================ 6. PAIR POWER
def reconstruct_pair_power(skip_rerun: bool) -> None:
    eid = "ol2-bank-w-olmo-pair-power-v1"
    result_path = registered_path(eid, "ol2_bank_w_olmo_pair_power_v1.json",
                                  OL_EVENTS)
    result = json.loads(result_path.read_text())
    decision = result["decision"]
    support = result["capability_support"]
    sesoi = result["sesoi"]
    simulation = result["simulation"]
    src = str(result_path)
    # frozen summary verifications
    add_recon("bw2_shared_capable_families",
              "OLMo Think/Instruct shared capable Bank-W families",
              16, support["n_shared_capable_families"],
              "verified_registered_summary",
              "byte_identical" if support["n_shared_capable_families"] == 16
              else "failed", src, "")
    sesoi_recon = 0.1 * math.log2(6.0 / 2.0)
    add_recon("bw2_sesoi_nats",
              "SESOI endpoint (0.10 nat/doubling, load 2->6)",
              sesoi["endpoint_high_minus_low_nats"], sesoi_recon,
              "recomputed_from_items",
              status_number(sesoi["endpoint_high_minus_low_nats"],
                            sesoi_recon), src,
              "task quotes 0.15849625 nat")
    add_recon("bw2_family_sd",
              "conservative family SD (nats) from registered variance ruler",
              0.23, result["registered_variance_ruler"][
                  "conservative_common_sd_nats"],
              "verified_registered_summary",
              "byte_identical"
              if result["registered_variance_ruler"][
                  "conservative_common_sd_nats"] == 0.23 else "failed",
              src, "imported from p4-bank-w-power-holm2-dev-v1")
    sim_desc = (f"t(df={simulation['heavy_tail_df']}), "
                f"{simulation['n_simulations_per_scenario']} sims, "
                f"{simulation['permutation_draws_per_simulation']} sign-flip "
                "draws")
    add_recon("bw2_simulation_contract",
              "simulation contract: Student-t(5), 5000 sims, 2048 sign-flip "
              "draws", "t(df=5), 5000 sims, 2048 sign-flip draws", sim_desc,
              "verified_registered_summary",
              "byte_identical"
              if sim_desc == "t(df=5), 5000 sims, 2048 sign-flip draws"
              else "failed", src, f"seed {simulation['seed']}")
    power_by_n = {row["n_families"]: row["rejection_rate"]
                  for row in result["power_at_sesoi"]}
    type_i = result["type_i_calibration"]
    frozen_min = decision["minimum_common_families_for_power_target"]
    if skip_rerun:
        add_recon("bw2_power_at_16",
                  "conservative one-active-model power at 16 families",
                  0.7788, power_by_n[16], "verified_registered_summary",
                  "byte_identical" if power_by_n[16] == 0.7788 else "failed",
                  src, "rerun skipped (--skip-power-rerun)")
        add_recon("bw2_min_families_for_target",
                  "first family count with power >= 0.80", 18, frozen_min,
                  "verified_registered_summary",
                  "byte_identical" if frozen_min == 18 else "failed", src,
                  "rerun skipped")
        add_recon("bw2_type_i_all_pass",
                  "all four type-I scenarios inside acceptance interval",
                  True, all(r["type_i_pass"] for r in type_i),
                  "verified_registered_summary",
                  "byte_identical"
                  if all(r["type_i_pass"] for r in type_i) else "failed",
                  src, "rerun skipped")
    else:
        root_seed = int(simulation["seed"])
        family_sd = float(result["registered_variance_ruler"][
            "conservative_common_sd_nats"])
        labels = ["olmo31-think", "olmo31-instruct"]
        correlation_rows = result["registered_variance_ruler"][
            "family_correlation_by_source"]
        empirical = np.asarray([[correlation_rows[a][b] for b in labels]
                                for a in labels])
        independent = np.eye(2)
        signs_cache = {}

        def signs_for(count: int) -> np.ndarray:
            if count not in signs_cache:
                signs_cache[count] = _random_signs(
                    int(simulation["permutation_draws_per_simulation"]),
                    count, stable_seed(root_seed, "signs", count))
            return signs_cache[count]

        def rerun(name: str, n_families: int, effects, correlation_name: str,
                  distribution: str) -> dict:
            correlation = (independent if correlation_name == "independent"
                           else empirical)
            return simulate_rejection_rate(
                n_simulations=int(
                    simulation["n_simulations_per_scenario"]),
                n_families=n_families,
                effects=np.asarray(effects),
                family_sd=family_sd,
                correlation=correlation,
                distribution=distribution,
                heavy_tail_df=int(simulation["heavy_tail_df"]),
                signs=signs_for(n_families),
                seed=stable_seed(root_seed, "scenario", name),
                alpha=0.05,
                simulation_batch_size=50,
                permutation_chunk_size=512,
            )
        # type-I: four scenarios at observed support
        rerun_type_i = {}
        for correlation_name in ("independent", "empirical"):
            for distribution in ("normal", "student_t"):
                name = f"null-{correlation_name}-{distribution}"
                rerun_type_i[name] = rerun(name, 16, [0.0, 0.0],
                                           correlation_name, distribution)
        bounds = simulation["type_i_acceptance_interval"]
        type_i_match = all(
            rerun_type_i[row["scenario"]]["rejection_rate"]
            == row["rejection_rate"] for row in type_i)
        type_i_pass = all(
            bounds[0] <= rerun_type_i[row["scenario"]]["rejection_rate"]
            <= bounds[1] for row in type_i)
        add_recon("bw2_type_i_all_pass",
                  "all four type-I scenarios inside acceptance interval "
                  "[0.025,0.075] (deterministic rerun, seeds registered)",
                  True, bool(type_i_pass), "recomputed_from_items",
                  "byte_identical" if type_i_pass and type_i_match
                  else ("failed" if not type_i_pass else
                        "numerically_identical_render_diff"), src,
                  "rerun rates "
                  + json.dumps({k: v["rejection_rate"]
                                for k, v in rerun_type_i.items()})
                  + f"; exact match with frozen: {type_i_match}")
        # power at SESOI: full frozen curve (21 family counts)
        endpoint = float(sesoi["endpoint_high_minus_low_nats"])
        rerun_power = {}
        for count in sorted(power_by_n):
            name = f"sesoi-one-active-model-n{count}"
            rerun_power[count] = rerun(name, count, [endpoint, 0.0],
                                       "independent", "student_t")
        p16 = rerun_power[16]["rejection_rate"]
        add_recon("bw2_power_at_16",
                  "conservative one-active-model power at 16 shared families "
                  "(deterministic rerun of frozen simulation)",
                  0.7788, p16, "recomputed_from_items",
                  "byte_identical" if p16 == 0.7788 else
                  status_number(0.7788, p16, render_tol=0.0), src,
                  f"registered {power_by_n[16]}; "
                  f"{rerun_power[16]['rejections']}/5000 rejections")
        rerun_min = next((count for count in sorted(rerun_power)
                          if rerun_power[count]["rejection_rate"] >= 0.8),
                         None)
        all_match = all(rerun_power[c]["rejection_rate"] == power_by_n[c]
                        for c in rerun_power)
        add_recon("bw2_min_families_for_target",
                  "first family count with rerun power >= 0.80 target",
                  18, rerun_min, "recomputed_from_items",
                  "byte_identical" if rerun_min == 18 == frozen_min
                  else "failed", src,
                  "rerun powers "
                  + json.dumps({c: rerun_power[c]["rejection_rate"]
                                for c in sorted(rerun_power)})
                  + f"; all equal frozen curve: {all_match}")
    add_recon("bw2_route", "pair-power planning route",
              "not-powered-at-current-support", decision["route"],
              "verified_registered_summary",
              "byte_identical"
              if decision["route"] == "not-powered-at-current-support"
              else "failed", src,
              f"worthwhile={decision['future_pair_worthwhile_at_current_support']}")
    pair = "olmo31-think__olmo31-instruct"
    add_matrix(pair, "bank_w_planning", "n_shared_capable_families", "pair",
               "bank_w_24_families", support["n_shared_capable_families"],
               "methods", eid)
    add_matrix(pair, "bank_w_planning", "power_at_shared_support", "pair",
               "sesoi_0.1585nat_sd0.23_t5",
               decision["power_at_shared_capable_support"], "methods", eid)
    add_matrix(pair, "bank_w_planning", "minimum_families_for_power_target",
               "pair", "sesoi_0.1585nat_sd0.23_t5", frozen_min, "methods",
               eid)
    add_matrix(pair, "bank_w_planning", "power_target", "pair",
               "sesoi_0.1585nat_sd0.23_t5", decision["power_target"],
               "methods", eid)
    add_matrix(pair, "bank_w_planning", "route", "pair",
               "sesoi_0.1585nat_sd0.23_t5", decision["route"], "methods", eid)
    add_matrix(pair, "bank_w_planning", "type_i_all_pass", "pair",
               "null_calibration_4_scenarios",
               all(r["type_i_pass"] for r in type_i), "methods", eid)
    add_matrix(pair, "bank_w_planning", "sesoi_endpoint_nats", "pair",
               "load_2_to_6", sesoi["endpoint_high_minus_low_nats"],
               "methods", eid)
    add_matrix(pair, "bank_w_planning", "family_sd_nats", "pair",
               "registered_variance_ruler", 0.23, "methods", eid)


# ==================================================== 7. BANK-W CAPABILITY JOINT
def reconstruct_bank_w_capability() -> None:
    joint_id = "ol-bank-w-capability-joint-dev-v1"
    joint = envelope(registered_path(joint_id,
                                     "ol-bank-w-capability-joint-dev-v1.json",
                                     OL_EVENTS))
    floor = 0.70
    capable_sets = {}
    row_sources = {
        "olmo31-think": registered_path(
            "ol-bank-w-capability-olmo31-think-dev-v1",
            "bank_w_capability_rows.parquet", OL_EVENTS),
        "olmo31-instruct": registered_path(
            "ol-bank-w-capability-olmo31-instruct-dev-v1",
            "bank_w_capability_rows.parquet", OL_EVENTS),
        "qwen36-27b": registered_path(
            "p4-bank-w-capability-qwen36-27b-dev-v1",
            "bank_w_capability_rows.parquet", P4_EVENTS),
    }
    for slug, rows_path in row_sources.items():
        rows = pd.read_parquet(rows_path)
        accuracy = rows.groupby(["canonical_family", "load"]).correct.mean()
        by_family = accuracy.unstack("load")
        capable = sorted(by_family[(by_family >= floor).all(axis=1)].index)
        capable_sets[slug] = capable
        frozen_ids = joint["model_analyses"][slug]["capable_family_ids"]
        add_recon(f"s1_capable_families_{slug}",
                  f"{slug} Bank-W capable families (accuracy >= 0.70 at both "
                  "loads, recomputed from 384-row battery)",
                  len(frozen_ids), len(capable), "recomputed_from_items",
                  "byte_identical"
                  if capable == sorted(frozen_ids) else "failed",
                  str(rows_path),
                  "full ID set match" if capable == sorted(frozen_ids)
                  else "ID SET MISMATCH")
        add_matrix(slug, "bank_w_capability", "n_capable_families",
                   "generation", "bank_w_24_families", len(frozen_ids),
                   "development",
                   f"ol-bank-w-capability-{slug}-dev-v1"
                   if slug != "qwen36-27b"
                   else "p4-bank-w-capability-qwen36-27b-dev-v1")
    pair_intersection = sorted(set(capable_sets["olmo31-think"])
                               & set(capable_sets["olmo31-instruct"]))
    joint_intersection = sorted(set(pair_intersection)
                                & set(capable_sets["qwen36-27b"]))
    frozen_joint = sorted(joint["joint_common_capable_family_ids"])
    add_recon("s1_joint_common_capable",
              "three-model joint common capable families (16) vs minimum 20",
              f"{len(frozen_joint)}/20",
              f"{len(joint_intersection)}/"
              f"{joint['minimum_joint_common_families']}",
              "recomputed_from_items",
              "byte_identical" if joint_intersection == frozen_joint
              else "failed",
              str(OL_RUN / "metrics/bank-w-capability/"
                  "ol-bank-w-capability-joint-dev-v1.json"),
              f"sha256(sorted ids) recon {object_sha256(joint_intersection)[:12]} "
              f"frozen {joint['joint_common_capable_family_ids_sha256'][:12]}")
    add_recon("s1_pair_intersection_16",
              "OLMo Think/Instruct pairwise intersection equals the 16 "
              "shared families used by Study-2 pair planning",
              16, len(pair_intersection), "recomputed_from_items",
              "byte_identical" if len(pair_intersection) == 16 else "failed",
              str(OL_RUN / "metrics/*/bank_w_capability/*/"
                  "bank_w_capability_rows.parquet"),
              "matches ol2-bank-w-olmo-pair-power-v1 support sha "
              if object_sha256(pair_intersection)
              == "b24741a742f0f22992c386b53ca60b9aa8a832b5f03b5d586da18aa3822e3df7"
              else "SHA MISMATCH with pair-power support")
    add_recon("s1_joint_decision_blocked",
              "Study-1 joint capability decision blocks Bank-W intervention",
              "BLOCKED", joint["decision"].split(":")[0],
              "verified_registered_summary",
              "byte_identical"
              if joint["decision"].startswith("BLOCKED") else "failed",
              str(OL_RUN / "metrics/bank-w-capability/"
                  "ol-bank-w-capability-joint-dev-v1.json"),
              f"freeze_ready={joint['freeze_ready']}")
    # phase4 import of the joint block
    p4_path = REPO / ("interpretability/jspace_phase4/reports/"
                      "bank_w_capability_joint_imported_dev_v1.json")
    p4_event = event("p4-bank-w-capability-joint-imported-dev-v1", P4_EVENTS)
    registered_sha = [o["sha256"] for o in p4_event["outputs"]
                      if o["path"].endswith(".json")][0]
    observed_sha = sha256_file(p4_path)
    p4_joint = envelope(p4_path)["joint"] if "joint" in envelope(p4_path) \
        else envelope(p4_path)
    p4_ids = sorted(p4_joint["joint_common_capable_family_ids"])
    add_recon("s1_p4_import_16of20",
              "Phase-4 imported joint block repeats 16/20 common capable "
              "families",
              "16/20 (sha " + registered_sha[:12] + ")",
              f"{len(p4_ids)}/{p4_joint.get('minimum_joint_common_families', 20)}"
              f" (sha {observed_sha[:12]})",
              "verified_registered_summary",
              "byte_identical"
              if observed_sha == registered_sha and p4_ids == frozen_joint
              else "failed", str(p4_path),
              "registered path /content/labs_phase4_4/... no longer exists; "
              "repo copy verified against registered sha256")
    add_matrix("joint_think_instruct_qwen", "bank_w_capability",
               "n_joint_common_capable_families", "generation",
               "bank_w_24_families", len(frozen_joint), "development",
               joint_id)
    add_matrix("joint_think_instruct_qwen", "bank_w_capability",
               "minimum_joint_common_families", "generation",
               "bank_w_24_families", joint["minimum_joint_common_families"],
               "development", joint_id)
    add_matrix("joint_think_instruct_qwen", "bank_w_capability",
               "bank_w_intervention", "generation", "bank_w_24_families",
               "blocked_insufficient_common_support", "development", joint_id)


# ------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-power-rerun", action="store_true")
    arguments = parser.parse_args()
    reconstruct_capacity()
    reconstruct_geometry()
    reconstruct_common_cohort()
    reconstruct_stage_wedge()
    reconstruct_transport()
    reconstruct_pair_power(arguments.skip_power_rerun)
    reconstruct_bank_w_capability()
    if HASH_FAILURES:
        add_recon("archive_sha256_integrity",
                  "registered source files whose sha256 no longer matches",
                  "0 mismatches", f"{len(HASH_FAILURES)} mismatches",
                  "verified_registered_summary", "failed", "",
                  ";".join(HASH_FAILURES))
    else:
        add_recon("archive_sha256_integrity",
                  "every archive file read verified against its registered "
                  "sha256", "all match", "all match",
                  "verified_registered_summary", "byte_identical",
                  "both registries", "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "recon_olmo.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "target_id", "description", "frozen_value",
            "reconstructed_value", "method", "status", "source_paths",
            "notes"])
        writer.writeheader()
        writer.writerows(RECON_ROWS)
    with open(OUT_DIR / "olmo_lineage_matrix_inputs.csv", "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "checkpoint", "metric_class", "metric", "frame", "cohort",
            "value", "tier", "source_evidence_id"])
        writer.writeheader()
        writer.writerows(MATRIX_ROWS)
    statuses = {}
    for row in RECON_ROWS:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    print(json.dumps({"recon_rows": len(RECON_ROWS),
                      "matrix_rows": len(MATRIX_ROWS),
                      "statuses": statuses}, indent=1))
    failures = [r["target_id"] for r in RECON_ROWS if r["status"] == "failed"]
    if failures:
        print("FAILED:", failures)


if __name__ == "__main__":
    main()
