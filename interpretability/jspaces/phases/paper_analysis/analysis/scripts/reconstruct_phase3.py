# reconstruct_phase3.py — deterministic reconstruction of the frozen
# Phase 3 headline numbers for the paper-analysis audit.
#
# Reads (read-only):
#   RUN      = /content/drive/MyDrive/interpret/special-lab-1/phase3_20260729
#   REPO     = /content/labs (frozen jspace_phase3 analysis code, imported)
#   REGISTRY = interpretability/jspaces/phases/phase3/reports/evidence_events.jsonl
# Writes:
#   interpretability/jspaces/phases/paper_analysis/analysis/tables/recon_phase3.csv
#
# Reconstruction policy (task contract):
#   * prefer recomputation from item-level rows ("recomputed_from_items")
#     over registered summaries ("verified_registered_summary");
#   * P3-P1 exact p-values are full 2^17 family sign-flip enumerations,
#     recomputed here from item rows;
#   * Monte Carlo p-values (plus-one rule) and family bootstrap CIs are
#     recomputed with the frozen seeds found in the analysis code
#     (SEED=4242 everywhere) using the frozen modules' own functions;
#     null/bootstrap distribution sha256s are compared where registered;
#   * frozen_value below is the full-precision registered artifact value;
#     the rounded state-of-record quote is carried in `notes`.
#
# Environment parity with the frozen run (phase3_environment_lock.json):
# python 3.12.13 / numpy 2.0.2 / pandas 2.2.2 / pyarrow 18.1.0.
# CPU only. No network. Deterministic and rerunnable.
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/content/labs")
RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/phase3_20260729")
PKG3 = REPO / "interpretability/jspaces/phases/phase3"
PKG2 = REPO / "interpretability/jspaces/phases/phase2"
OUT_CSV = (REPO / "interpretability/jspaces/phases/paper_analysis" /
           "analysis" / "tables" / "recon_phase3.csv")

sys.path.insert(0, str(PKG3))
sys.path.insert(0, str(PKG2))

# Frozen analysis primitives, imported from the release-audit code itself.
from jspace_phase3.stats import (  # noqa: E402
    exact_signflip_test, signflip_confidence_set,
    wild_cluster_percentile_t_ci, within_fact_composition,
    within_fact_model_diff, within_item_exchange_mean,
    within_item_label_exchange_tail)
from jspace_phase3.experiments.p3_inference_audit import (  # noqa: E402
    effect_bootstrap, family_weighted_randomization)
from jspace_phase3.experiments.p3_bridge_geometry_audit import (  # noqa: E402
    _family_inference as geo_family_inference, nested_family_ridge)
from jspace_phase3.experiments.p3_bridge_swap_endpoint_audit import (  # noqa: E402
    _family_inference as swap_family_inference)
from jspace_phase3.experiments.p3_alias_and_cohort_sensitivity import (  # noqa: E402
    _capability_sets)

SLUGS = ("olmo31-think", "olmo31-instruct", "qwen36-27b")

# Diagnosis for the one non-byte-identical auxiliary artifact: the exact
# 2^17 sign-flip *distribution array* is produced by `signs @ v` (BLAS
# dgemv). Its last-ulp accumulation pattern is CPU-kernel dependent; on
# this host it is stable across OMP/OPENBLAS thread counts but differs
# from the frozen GPU-VM host (local reorderings move entries by
# <= 5.6e-17, i.e. 1 ulp). Every decision-bearing scalar derived from it
# (estimate, extreme_patterns under the 1e-15 tolerance, p, inverted CI
# endpoints on the 4001-point grid) reproduces bit-identically.
BLAS_ULP_NOTE = (
    "sha mismatch is BLAS-matvec ulp provenance (<=1 ulp, "
    "CPU-kernel dependent); all derived scalars bit-identical")

# Registered frozen artifacts (read-only references).
INF_JSON = RUN / "metrics/cross_model/release_audit/p3_inference_audit.json"
INF_FAMILY = (RUN / "metrics/cross_model/release_audit/"
              "p3_inference_audit_family_values.parquet")
CSF_JSON = (RUN / "metrics/qwen36-27b/release_audit/control_seed/"
            "p3_control_seed_audit_full.json")
CSF_ROWS = (RUN / "metrics/qwen36-27b/release_audit/control_seed/"
            "p3_control_seed_audit_full_rows.parquet")
PROT_JSON = (RUN / "metrics/qwen36-27b/release_audit/protected_answer/"
             "p3_protected_answer_audit.json")
PROT_JOINED = (RUN / "metrics/qwen36-27b/release_audit/protected_answer/"
               "p3_protected_answer_joined_qwen.parquet")
GEO_DIR = RUN / "metrics/qwen36-27b/release_audit/bridge_geometry_v2"
GEO_JSON = GEO_DIR / "p3_bridge_geometry_qwen36-27b.json"
SWAP_DIR = RUN / "metrics/qwen36-27b/release_audit/bridge_swap_endpoint"
SWAP_JSON = SWAP_DIR / "p3_bridge_swap_endpoint_qwen36-27b.json"
SENS_DIR = RUN / "metrics/cross_model/release_audit/alias_cohort_sensitivity_v2"
SENS_JSON = SENS_DIR / "p3_alias_cohort_sensitivity.json"
PARTITION = PKG3 / "preregistration" / "partition_phase3.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def grid_path(slug: str, side: str) -> Path:
    suffix = "" if side == "confirmatory" else "_replication"
    return (RUN / "metrics" / slug / f"p3_grid{suffix}"
            / f"p3_grid{suffix}_{slug}.parquet")


def load_grid(slug: str, side: str) -> pd.DataFrame:
    # Mirrors p3_inference_audit.load_effects / sensitivity _load_grid.
    df = pd.read_parquet(grid_path(slug, side))
    df["model"] = slug
    df["J_eff"] = df["lp_meanJ_span_safe"] - df["lp_baseline"]
    df["C_eff"] = df["lp_ss_matched"] - df["lp_baseline"]
    df["specific"] = df["J_eff"] - df["C_eff"]
    return df


ROWS: list[dict] = []


def fmt(v) -> str:
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return str(v)


def status_of(frozen, recon) -> str:
    def pair(a, b):
        a, b = float(a), float(b)
        if a == b:
            return "byte_identical"
        if b != 0 and f"{a:.12g}" == f"{b:.12g}":
            return "numerically_identical_render_diff"
        if abs(a - b) <= max(1e-6 * abs(a), 1e-9):
            return "numerically_within_frozen_tolerance"
        return "failed"

    if isinstance(frozen, (list, tuple)):
        stats = [pair(a, b) for a, b in zip(frozen, recon)]
        if len(frozen) != len(recon):
            return "failed"
        order = ["failed", "numerically_within_frozen_tolerance",
                 "numerically_identical_render_diff", "byte_identical"]
        return min(stats, key=order.index)
    return pair(frozen, recon)


def add(tid, desc, frozen, recon, method, sources, notes="", status=None):
    ROWS.append({
        "target_id": tid,
        "description": desc,
        "frozen_value": fmt(frozen),
        "reconstructed_value": fmt(recon),
        "method": method,
        "status": status or status_of(frozen, recon),
        "source_paths": "; ".join(str(s) for s in sources),
        "notes": notes,
    })


def main() -> None:
    # ------------------------------------------------------------------
    # 0. Integrity gate: the six frozen p3_grid parquets must hash to the
    # inputs registered by p3-inference-audit-v1.
    # ------------------------------------------------------------------
    inf = jload(INF_JSON)
    registered_inputs = inf["provenance"]["inputs"]
    hash_fail = []
    for path_str, want in registered_inputs.items():
        got = sha256_file(Path(path_str))
        if got != want:
            hash_fail.append(path_str)
    if hash_fail:
        raise SystemExit(f"frozen parquet hash mismatch: {hash_fail}")
    print("input gate: all 6 p3_grid parquet sha256 match "
          "p3-inference-audit-v1 registered inputs")

    conf = {s: load_grid(s, "confirmatory") for s in SLUGS}
    repl = {s: load_grid(s, "replication") for s in SLUGS}
    effects_conf = pd.concat(conf.values(), ignore_index=True)
    effects_repl = pd.concat(repl.values(), ignore_index=True)
    grid_sources = [grid_path(s, "confirmatory") for s in SLUGS]
    grid_sources_repl = [grid_path(s, "replication") for s in SLUGS]

    # ==================================================================
    # 1. P3-P1 (frozen module pipeline over item rows)
    # ==================================================================
    def p3p1(effects: pd.DataFrame) -> dict:
        comp = within_fact_composition(effects, value_col="specific")
        diff = within_fact_model_diff(
            comp, model_a="qwen36-27b",
            model_b=["olmo31-think", "olmo31-instruct"])
        family = diff.groupby("canonical_family", sort=True)["diff"].mean()
        vals = family.to_numpy()
        return {
            "family": family,
            "estimate": float(vals.mean()),
            "exact": exact_signflip_test(vals),
            "ci_set": signflip_confidence_set(vals),
            "pct_t": wild_cluster_percentile_t_ci(
                diff.rename(columns={"diff": "d"}), "d"),
        }

    r1c = p3p1(effects_conf)
    r1r = p3p1(effects_repl)
    fc = inf["payload"]["confirmatory"]["P3-P1"]
    fr = inf["payload"]["replication"]["P3-P1"]

    # Cross-check recomputed family deltas against the registered family
    # values parquet (17 x 2 rows).
    famtab = pd.read_parquet(INF_FAMILY)
    fam_conf = famtab[famtab["side"] == "confirmatory"].set_index(
        "canonical_family")["family_mean"]
    fam_repl = famtab[famtab["side"] == "replication"].set_index(
        "canonical_family")["family_mean"]
    fam_dev_c = float((r1c["family"] - fam_conf).abs().max())
    fam_dev_r = float((r1r["family"] - fam_repl).abs().max())
    fam_note = (f"17 family deltas vs registered parquet: max|diff| "
                f"conf={fam_dev_c:.3g}, repl={fam_dev_r:.3g}")

    sha_c = r1c["exact"]["distribution_sha256"] == \
        fc["exact_randomization"]["distribution_sha256"]
    add("p3p1_confirm_hist_estimate",
        "P3-P1 confirmatory, historical Qwen control realization, "
        "equal-family estimate (nats)",
        fc["estimate_family_weighted"], r1c["estimate"],
        "recomputed_from_items", grid_sources,
        f"SOR quote -0.260954; 17 families / 52 facts. {fam_note}")
    add("p3p1_confirm_hist_exact_p",
        "P3-P1 confirmatory historical: exact 2^17 family sign-flip p "
        "(two-sided)",
        fc["exact_randomization"]["p"], r1c["exact"]["p"],
        "recomputed_from_items", grid_sources,
        f"SOR quote p=0.062332; extreme_patterns frozen="
        f"{fc['exact_randomization']['extreme_patterns']} recon="
        f"{r1c['exact']['extreme_patterns']} of 131072; "
        f"null distribution sha256 match={sha_c}"
        + ("" if sha_c else f"; {BLAS_ULP_NOTE}"))
    add("p3p1_confirm_randomization_ci",
        "P3-P1 confirmatory: randomization-compatible 95% confidence set "
        "(exact shifted sign-flip inversion)",
        fc["randomization_compatible_confidence_set"]["confidence_set"],
        r1c["ci_set"]["confidence_set"],
        "recomputed_from_items", grid_sources,
        "SOR quote [-0.535135, +0.014565]; 4001-point grid, deterministic")
    sha_t = r1c["pct_t"]["t_distribution_sha256"] == \
        fc["wild_cluster_percentile_t_interval"]["t_distribution_sha256"]
    add("p3p1_confirm_percentile_t_ci",
        "P3-P1 confirmatory: wild-cluster percentile-t 95% CI "
        "(exact 2^17 Rademacher enumeration)",
        fc["wild_cluster_percentile_t_interval"]["ci"], r1c["pct_t"]["ci"],
        "recomputed_from_items", grid_sources,
        f"SOR quote [-0.537109, +0.015201]; t-distribution sha256 "
        f"match={sha_t}")
    add("p3p1_repl_estimate",
        "P3-P1 replication equal-family estimate (nats)",
        fr["estimate_family_weighted"], r1r["estimate"],
        "recomputed_from_items", grid_sources_repl,
        "SOR quote -0.197020; 17 families")
    sha_r = r1r["exact"]["distribution_sha256"] == \
        fr["exact_randomization"]["distribution_sha256"]
    add("p3p1_repl_exact_p",
        "P3-P1 replication: exact 2^17 family sign-flip p (two-sided)",
        fr["exact_randomization"]["p"], r1r["exact"]["p"],
        "recomputed_from_items", grid_sources_repl,
        f"SOR quote p=0.219482; extreme_patterns frozen="
        f"{fr['exact_randomization']['extreme_patterns']} recon="
        f"{r1r['exact']['extreme_patterns']}; null sha match={sha_r}"
        + ("" if sha_r else f"; {BLAS_ULP_NOTE}"))

    # -- state-of-record realization: sha256-v1 control at seed 31337 ---
    # Mirrors p3_control_seed_audit.seed_summary over the full-cohort
    # audit rows (frozen GPU outputs) + immutable OLMo grids.
    csf = jload(CSF_JSON)
    frozen_31337 = csf["payload"]["sides"]["confirmatory"]["per_seed"][
        "31337"]["P3-P1_subset"]
    rows_all = pd.read_parquet(CSF_ROWS)
    side_rows = rows_all[rows_all["side"] == "confirmatory"]
    frame = side_rows[side_rows["audit_seed"] == 31337].copy()
    frame["delta_J"] = frame["lp_meanJ_span_safe"] - frame["lp_baseline"]
    frame["delta_C"] = frame["lp_ss_matched"] - frame["lp_baseline"]
    frame["specific"] = frame["delta_J"] - frame["delta_C"]
    olmo_parts = []
    fact_ids = set(frame["fact_id"])
    for slug in ("olmo31-think", "olmo31-instruct"):
        f = pd.read_parquet(
            grid_path(slug, "confirmatory"),
            columns=["fact_id", "variant", "canonical_family",
                     "lp_meanJ_span_safe", "lp_ss_matched"])
        f = f[f["fact_id"].isin(fact_ids)].copy()
        f["specific"] = f["lp_meanJ_span_safe"] - f["lp_ss_matched"]
        piv = f.pivot(index=["fact_id", "canonical_family"],
                      columns="variant", values="specific").reset_index()
        piv[f"{slug}_composition"] = piv["composed"] - piv["direct"]
        olmo_parts.append(
            piv[["fact_id", "canonical_family", f"{slug}_composition"]])
    olmo = olmo_parts[0].merge(
        olmo_parts[1], on=["fact_id", "canonical_family"],
        validate="one_to_one")
    qwen_comp = frame.pivot(
        index=["fact_id", "canonical_family"],
        columns="variant", values="specific").reset_index()
    qwen_comp["qwen_composition"] = (
        qwen_comp["composed"] - qwen_comp["direct"])
    p1 = qwen_comp.merge(olmo, on=["fact_id", "canonical_family"],
                         validate="one_to_one")
    p1["diff"] = p1["qwen_composition"] - 0.5 * (
        p1["olmo31-think_composition"] + p1["olmo31-instruct_composition"])
    family_vals = p1.groupby("canonical_family", sort=True)["diff"].mean()
    sor_exact = exact_signflip_test(family_vals.to_numpy())
    sor_sha = (sor_exact["distribution_sha256"]
               == frozen_31337["distribution_sha256"])
    sor_sources = [CSF_ROWS, grid_path("olmo31-think", "confirmatory"),
                   grid_path("olmo31-instruct", "confirmatory")]
    add("p3p1_confirm_sor_estimate",
        "P3-P1 confirmatory STATE OF RECORD: sha256-v1 Qwen control at "
        "seed 31337, equal-family estimate (nats)",
        frozen_31337["estimate"], sor_exact["estimate"],
        "recomputed_from_items", sor_sources,
        "SOR quote -0.271183; Qwen lp rows are frozen GPU outputs at the "
        "explicit seed-31337 realization; statistic recomputed from them")
    add("p3p1_confirm_sor_exact_p",
        "P3-P1 confirmatory STATE OF RECORD: exact 2^17 sign-flip p at "
        "seed 31337",
        frozen_31337["p"], sor_exact["p"],
        "recomputed_from_items", sor_sources,
        f"SOR quote p=0.057892; extreme_patterns frozen="
        f"{frozen_31337['extreme_patterns']} recon="
        f"{sor_exact['extreme_patterns']}; null sha match={sor_sha}"
        + ("" if sor_sha else f"; {BLAS_ULP_NOTE}"))

    # ==================================================================
    # 2. P3-P2 tail excess at -1 nat (frozen MC seed 4242)
    # ==================================================================
    def p3p2(effects: pd.DataFrame, threshold: float = -1.0) -> dict:
        q = effects[effects["model"] == "qwen36-27b"].copy()
        hd = ((q["J_eff"].to_numpy() < threshold).astype(float)
              - (q["C_eff"].to_numpy() < threshold).astype(float))
        rand = family_weighted_randomization(
            hd, q["canonical_family"].to_numpy())
        boot = effect_bootstrap(hd, q["canonical_family"].to_numpy())
        return {"rand": rand, "boot": boot}

    r2c = p3p2(effects_conf)
    r2r = p3p2(effects_repl)
    f2c = inf["payload"]["confirmatory"][
        "P3-P2_all_items_threshold_curve"]["-1.0"]
    f2r = inf["payload"]["replication"][
        "P3-P2_all_items_threshold_curve"]["-1.0"]
    qsrc = [grid_path("qwen36-27b", "confirmatory")]
    qsrc_r = [grid_path("qwen36-27b", "replication")]

    sha2c = (r2c["rand"]["null_distribution_sha256"]
             == f2c["null_distribution_sha256"])
    add("p3p2_confirm_estimate",
        "P3-P2 confirmatory: Qwen span-safe tail excess at -1 nat, "
        "equal-family estimate",
        f2c["estimate"], r2c["rand"]["estimate"],
        "recomputed_from_items", qsrc,
        f"SOR quote +0.095833; {r2c['rand']['n_items']} items / "
        f"{r2c['rand']['n_families']} families (frozen 188/26); "
        f"bootstrap-route estimate recon={r2c['boot']['estimate']!r} vs "
        f"frozen={f2c['effect_size_interval']['estimate']!r}")
    add("p3p2_confirm_p_plus_one",
        "P3-P2 confirmatory: plus-one MC p (100000 item sign flips, "
        "seed 4242)",
        f2c["p_plus_one"], r2c["rand"]["p_plus_one"],
        "recomputed_from_items", qsrc,
        f"SOR quote p=1/100001; null distribution sha256 match={sha2c}")
    sha2r = (r2r["rand"]["null_distribution_sha256"]
             == f2r["null_distribution_sha256"])
    add("p3p2_repl_estimate",
        "P3-P2 replication: tail excess at -1 nat, equal-family estimate",
        f2r["estimate"], r2r["rand"]["estimate"],
        "recomputed_from_items", qsrc_r,
        f"SOR quote +0.102083; {r2r['rand']['n_items']} items / "
        f"{r2r['rand']['n_families']} families (frozen 190/28); "
        f"bootstrap-route estimate recon={r2r['boot']['estimate']!r} vs "
        f"frozen={f2r['effect_size_interval']['estimate']!r} "
        "(the two frozen routes differ by 1 ulp; both reproduced)")
    add("p3p2_repl_p_plus_one",
        "P3-P2 replication: plus-one MC p (seed 4242)",
        f2r["p_plus_one"], r2r["rand"]["p_plus_one"],
        "recomputed_from_items", qsrc_r,
        f"SOR quote p=1/100001; null sha match={sha2r}")

    # -- protected-answer strata (joined item rows + frozen functions) --
    prot = jload(PROT_JSON)
    joined = pd.read_parquet(PROT_JOINED)
    protect_k = prot["payload"]["protocol"]["protect_k"]

    def protected_view(side: str, rank_field: str | None) -> dict:
        f = joined[joined["partition_side"] == side]
        sub = f if rank_field is None else f[f[rank_field] <= protect_k]
        hd = ((sub["delta_J"].to_numpy() < -1.0).astype(float)
              - (sub["delta_C"].to_numpy() < -1.0).astype(float))
        rand = family_weighted_randomization(
            hd, sub["canonical_family"].to_numpy())
        return rand

    views = {
        ("confirmatory", "rank_exact_scored_alias",
         "exact_scored_alias_protected"),
        ("confirmatory", "rank_min_accepted_alias",
         "any_accepted_alias_protected"),
        ("replication", "rank_exact_scored_alias",
         "exact_scored_alias_protected"),
    }
    for side, rank_field, view_name in sorted(views):
        rv = protected_view(side, rank_field)
        fv = prot["payload"]["sides"][side][view_name][
            "threshold_curve"]["-1.0"]
        sha_v = (rv["null_distribution_sha256"]
                 == fv["null_distribution_sha256"])
        quote = ("+0.095833" if side == "confirmatory" else "+0.098895")
        add(f"p3p2_{side[:4]}_protected_{rank_field.split('rank_')[1]}",
            f"P3-P2 {side}: protected-answer stratum "
            f"({view_name}, rank<={protect_k}), tail excess at -1 nat",
            fv["estimate"], rv["estimate"],
            "recomputed_from_items", [PROT_JOINED],
            f"SOR quote {quote}; {rv['n_items']} items (frozen "
            f"{fv['n_items']}); plus-one p frozen={fv['p_plus_one']!r} "
            f"recon={rv['p_plus_one']!r}; null sha match={sha_v}")

    # ==================================================================
    # 3. P3-P3 bridge rescue
    # ==================================================================
    q = effects_conf[
        (effects_conf["model"] == "qwen36-27b")
        & effects_conf["lp_true_bridge"].notna()].copy()
    d = (q["lp_true_bridge"] - q["lp_distractor_bridge"]).to_numpy()
    rand3 = family_weighted_randomization(
        d, q["canonical_family"].to_numpy())
    boot3 = effect_bootstrap(d, q["canonical_family"].to_numpy())
    f3 = inf["payload"]["confirmatory"]["P3-P3"]
    sha3 = (rand3["null_distribution_sha256"]
            == f3["null_distribution_sha256"])
    add("p3p3_estimate",
        "P3-P3 confirmatory: true-minus-distractor bridge rescue, "
        "equal-family estimate (nats)",
        f3["estimate"], rand3["estimate"],
        "recomputed_from_items", qsrc,
        f"SOR quote +0.431367; {rand3['n_items']} items / "
        f"{rand3['n_families']} families (frozen 94/26)")
    add("p3p3_p_plus_one",
        "P3-P3 confirmatory: plus-one MC p (100000 item sign flips, "
        "seed 4242)",
        f3["p_plus_one"], rand3["p_plus_one"],
        "recomputed_from_items", qsrc,
        f"SOR quote p=0.009180; null sha match={sha3}; "
        f"inference-audit 20k family bootstrap CI recon="
        f"{boot3['ci95']} vs frozen={f3['effect_size_interval']['ci95']}")

    # -- bridge geometry v2: raw/residualized rescue, cross-fit ridge --
    geo = jload(GEO_JSON)["payload"]
    item = pd.read_parquet(GEO_DIR / "p3_bridge_geometry_items_qwen36-27b.parquet")
    site = pd.read_parquet(GEO_DIR / "p3_bridge_geometry_sites_qwen36-27b.parquet")
    geo_sources = [GEO_DIR / "p3_bridge_geometry_items_qwen36-27b.parquet",
                   GEO_DIR / "p3_bridge_geometry_sites_qwen36-27b.parquet"]
    # Transcription of p3_bridge_geometry_audit.analyze (analysis half).
    metrics = [
        "protected_rank_before", "protected_rank_after", "added_rank",
        "added_selected_overlap", "rank_selected_before",
        "rank_selected", "removed_energy_l2_sq", "removed_energy_frac",
        "lost_rank", "answer_dir_survival_mean",
        "diagnostic_dir_survival_mean", "diagnostic_base_overlap",
        "diagnostic_activation_score_mean",
        "diagnostic_activation_score_max",
        "diagnostic_answer_cosine_mean",
    ]
    arm = site.groupby(["fact_id", "canonical_family", "arm"],
                       sort=True, as_index=False).agg(
        {m: "mean" for m in metrics})
    wide = arm.pivot(index=["fact_id", "canonical_family"],
                     columns="arm", values=metrics)
    paired = item[[
        "fact_id", "canonical_family", "n_tokens", "lp_baseline",
        "lp_span_safe", "lp_true_bridge", "lp_distractor_bridge",
        "true_piece_count", "distractor_piece_count"]].copy()
    paired["rescue"] = (
        paired["lp_true_bridge"] - paired["lp_distractor_bridge"])
    idx = pd.MultiIndex.from_frame(paired[["fact_id", "canonical_family"]])
    for m in metrics:
        paired[f"true_{m}_mean"] = wide[(m, "true")].reindex(idx).to_numpy()
        paired[f"distractor_{m}_mean"] = (
            wide[(m, "distractor")].reindex(idx).to_numpy())
        paired[f"diff_{m}_mean"] = (
            paired[f"true_{m}_mean"] - paired[f"distractor_{m}_mean"])
    paired["diff_piece_count"] = (
        paired.true_piece_count - paired.distractor_piece_count)
    rank_profiles = {}
    for (fact_id, arm_name), sub in site.groupby(["fact_id", "arm"]):
        ordered = sub.sort_values(["layer", "position"])
        rank_profiles[(fact_id, arm_name)] = tuple(
            int(v) for v in ordered.added_rank)
    paired["exact_piece_count_match"] = (
        paired.true_piece_count == paired.distractor_piece_count)
    paired["exact_added_rank_profile_match"] = [
        rank_profiles[(fid, "true")] == rank_profiles[(fid, "distractor")]
        for fid in paired.fact_id]
    paired["exact_geometry_match"] = (
        paired.exact_piece_count_match
        & paired.exact_added_rank_profile_match)
    feature_names = [
        "diff_piece_count", "diff_added_rank_mean",
        "diff_added_selected_overlap_mean",
        "diff_protected_rank_after_mean", "diff_rank_selected_mean",
        "diff_removed_energy_frac_mean", "diff_lost_rank_mean",
        "diff_answer_dir_survival_mean_mean",
        "diff_diagnostic_dir_survival_mean_mean",
        "diff_diagnostic_base_overlap_mean",
        "diff_diagnostic_activation_score_mean_mean",
        "diff_diagnostic_answer_cosine_mean_mean",
    ]
    x = paired[feature_names].to_numpy(dtype=float)
    y = paired.rescue.to_numpy(dtype=float)
    families = paired.canonical_family.to_numpy()
    prediction, _choices = nested_family_ridge(x, y, families)
    paired["geometry_prediction_crossfit"] = prediction
    paired["semantic_residual_crossfit"] = y - prediction
    total = float(np.sum((y - y.mean()) ** 2))
    crossfit_r2 = float(1 - np.sum((y - prediction) ** 2) / total)

    stored_paired = pd.read_parquet(
        GEO_DIR / "p3_bridge_geometry_paired_qwen36-27b.parquet")
    pred_dev = float(np.max(np.abs(
        stored_paired["geometry_prediction_crossfit"].to_numpy()
        - prediction)))

    raw = geo_family_inference(paired, "rescue")
    resid = geo_family_inference(paired, "semantic_residual_crossfit")
    exact_sub = paired[paired.exact_geometry_match].copy()
    exact_inf = geo_family_inference(exact_sub, "rescue")

    fraw = geo["raw_rescue"]
    add("p3p3_family_bootstrap_ci",
        "P3-P3: 95% family bootstrap CI of the raw rescue "
        "(bridge-geometry-v2 inference, seed 4242, 100000 draws)",
        fraw["ci95_family_bootstrap"], raw["ci95_family_bootstrap"],
        "recomputed_from_items", geo_sources,
        f"SOR quote [+0.132018, +0.763437]; point estimate recon="
        f"{raw['estimate_equal_family']!r} frozen="
        f"{fraw['estimate_equal_family']!r}")
    fres = geo["residualized_semantic_contrast"]
    add("p3p3_geom_residualized_estimate",
        "P3-P3: geometry-residualized rescue (outcome minus cross-fitted "
        "geometry prediction), equal-family estimate",
        fres["estimate_equal_family"], resid["estimate_equal_family"],
        "recomputed_from_items", geo_sources,
        f"SOR quote +0.403816; cross-fit ridge prediction vs stored "
        f"paired parquet: max|diff|={pred_dev:.3g}")
    add("p3p3_geom_residualized_ci",
        "P3-P3: geometry-residualized rescue 95% family bootstrap CI",
        fres["ci95_family_bootstrap"], resid["ci95_family_bootstrap"],
        "recomputed_from_items", geo_sources,
        "SOR quote [+0.105071, +0.734497]")
    add("p3p3_geom_residualized_p",
        "P3-P3: geometry-residualized rescue family sign-flip p "
        "(MC 100000, seed 4242, plus-one)",
        fres["family_signflip"]["p"], resid["family_signflip"]["p"],
        "recomputed_from_items", geo_sources,
        "SOR quote p=0.01854")
    add("p3p3_crossfit_r2",
        "P3-P3: leave-one-family-out geometry prediction cross-fit R^2",
        geo["geometry_only_prediction"]["crossfit_r2"], crossfit_r2,
        "recomputed_from_items", geo_sources,
        "SOR quote -0.0947; nested family ridge, lambda grid 1e-4..1e4")
    fex = geo["exact_geometry_matched_subset"]
    add("p3p3_exact_subset_estimate",
        "P3-P3: exact all-site geometry-matched subset rescue "
        "(equal-family estimate)",
        fex["rescue"]["estimate_equal_family"],
        exact_inf["estimate_equal_family"],
        "recomputed_from_items", geo_sources,
        f"SOR quote +0.450282; {exact_inf['n_items']} facts / "
        f"{exact_inf['n_families']} families (frozen {fex['n_items']}/"
        f"{fex['n_families']})")
    add("p3p3_exact_subset_p",
        "P3-P3: exact subset sign-flip p (exact 2^5 enumeration)",
        fex["rescue"]["family_signflip"]["p"],
        exact_inf["family_signflip"]["p"],
        "recomputed_from_items", geo_sources,
        "SOR quote p=0.1875 (6/32 patterns)")

    # ==================================================================
    # 4. Bridge swap endpoint (40 facts / 13 families)
    # ==================================================================
    swap = jload(SWAP_JSON)["payload"]
    frame_sw = pd.read_parquet(
        SWAP_DIR / "p3_bridge_swap_endpoint_qwen36-27b.parquet")
    swap_src = [SWAP_DIR / "p3_bridge_swap_endpoint_qwen36-27b.parquet"]
    # Transcription of p3_bridge_swap_endpoint_audit.analyze (paired part).
    id_cols = ["fact_id", "canonical_family"]
    values = [
        "lp_original_canonical", "lp_counterfactual_canonical",
        "preference_canonical", "lp_original_max_alias",
        "lp_counterfactual_max_alias", "preference_max_alias"]
    required_arms = {
        "baseline", "span_safe", "true_protect", "distractor_protect",
        "bridge_only", "cf_swap", "true_reinject", "unrelated_swap",
        "random_orthogonal_swap", "cf_answer_swap"}
    wide_sw = frame_sw.pivot(index=id_cols, columns="arm", values=values)
    paired_sw = frame_sw[frame_sw.arm == "baseline"][id_cols].copy()
    pair_index = pd.MultiIndex.from_frame(paired_sw[id_cols])
    expanded = {}
    for value in values:
        baseline = wide_sw[(value, "baseline")].reindex(
            pair_index).to_numpy()
        for arm_name in sorted(required_arms):
            arm_value = wide_sw[(value, arm_name)].reindex(
                pair_index).to_numpy()
            expanded[f"{value}__{arm_name}"] = arm_value
            expanded[f"{value}_shift__{arm_name}"] = arm_value - baseline
    paired_sw = pd.concat(
        [paired_sw.reset_index(drop=True), pd.DataFrame(expanded)], axis=1)
    paired_sw["primary_cf_preference_shift"] = (
        paired_sw["preference_canonical_shift__cf_swap"])
    paired_sw["primary_cf_vs_unrelated"] = (
        paired_sw["preference_canonical__cf_swap"]
        - paired_sw["preference_canonical__unrelated_swap"])
    paired_sw["cf_vs_cf_answer"] = (
        paired_sw["preference_canonical__cf_swap"]
        - paired_sw["preference_canonical__cf_answer_swap"])

    primary = swap_family_inference(paired_sw, "primary_cf_preference_shift")
    vs_unrel = swap_family_inference(paired_sw, "primary_cf_vs_unrelated")
    vs_answer = swap_family_inference(paired_sw, "cf_vs_cf_answer")
    # cf_answer_swap arm: preference shift from baseline.
    sub = frame_sw[frame_sw.arm == "cf_answer_swap"].copy()
    sub["preference_shift"] = (
        sub.preference_canonical.to_numpy()
        - paired_sw.set_index("fact_id").loc[
            sub.fact_id, "preference_canonical__baseline"].to_numpy())
    answer_dir = swap_family_inference(sub, "preference_shift")

    fprim = swap["primary_cf_preference_shift"]
    add("swap_cf_pref_shift_estimate",
        "Bridge swap: counterfactual-bridge preference shift "
        "LP(cf)-LP(orig) vs baseline, equal-family estimate (nats)",
        fprim["estimate_equal_family"], primary["estimate_equal_family"],
        "recomputed_from_items", swap_src,
        f"SOR quote +8.582031; {primary['n_items']} facts / "
        f"{primary['n_families']} families (frozen 40/13)")
    add("swap_cf_pref_shift_ci",
        "Bridge swap: preference shift 95% family bootstrap CI "
        "(seed 4242, 100000 draws)",
        fprim["ci95_family_bootstrap"], primary["ci95_family_bootstrap"],
        "recomputed_from_items", swap_src,
        "SOR quote [+5.077661, +12.122991]")
    add("swap_cf_pref_shift_exact_p",
        "Bridge swap: preference shift exact 2^13 family sign-flip p",
        fprim["family_signflip"]["p"], primary["family_signflip"]["p"],
        "recomputed_from_items", swap_src,
        "SOR quote p=0.000488 (4/8192, two-sided)")
    add("swap_cf_vs_unrelated_estimate",
        "Bridge swap: cf-bridge minus geometry-selected unrelated "
        "injection, equal-family estimate (nats)",
        swap["primary_cf_vs_unrelated_matched_injection"][
            "estimate_equal_family"],
        vs_unrel["estimate_equal_family"],
        "recomputed_from_items", swap_src, "SOR quote +4.765142")
    add("swap_cf_answer_direction_shift",
        "Bridge swap: direct counterfactual answer-direction injection "
        "preference shift, equal-family estimate (nats)",
        swap["arm_results"]["cf_answer_swap"][
            "preference_shift_from_baseline"]["estimate_equal_family"],
        answer_dir["estimate_equal_family"],
        "recomputed_from_items", swap_src,
        "SOR quote +7.240 (rounded in report)")
    fctrl = swap["control_contrasts"]["cf_swap_minus_cf_answer_direction"]
    add("swap_bridge_minus_answer_estimate",
        "Bridge swap: cf-bridge minus cf-answer-direction contrast, "
        "equal-family estimate (nats)",
        fctrl["estimate_equal_family"], vs_answer["estimate_equal_family"],
        "recomputed_from_items", swap_src, "SOR quote +1.342254")
    add("swap_bridge_minus_answer_ci",
        "Bridge swap: cf-bridge minus cf-answer-direction 95% family "
        "bootstrap CI",
        fctrl["ci95_family_bootstrap"], vs_answer["ci95_family_bootstrap"],
        "recomputed_from_items", swap_src,
        "SOR quote [-1.593275, +4.482051]")
    add("swap_bridge_minus_answer_p",
        "Bridge swap: cf-bridge minus cf-answer-direction exact "
        "sign-flip p",
        fctrl["family_signflip"]["p"], vs_answer["family_signflip"]["p"],
        "recomputed_from_items", swap_src, "SOR quote p=0.419")

    def greedy_counts(arm_name: str) -> tuple[int, int]:
        s = frame_sw[frame_sw.arm == arm_name]
        return (int((s.greedy_category == "original").sum()),
                int((s.greedy_category == "counterfactual").sum()))

    gb = greedy_counts("baseline")
    gs = greedy_counts("cf_swap")
    fb = (swap["arm_results"]["baseline"]["greedy_generation"]["original"]
          ["n"],
          swap["arm_results"]["baseline"]["greedy_generation"]
          ["counterfactual"]["n"])
    fs = (swap["arm_results"]["cf_swap"]["greedy_generation"]["original"]
          ["n"],
          swap["arm_results"]["cf_swap"]["greedy_generation"]
          ["counterfactual"]["n"])
    add("swap_greedy_baseline",
        "Bridge swap: greedy generation hits at baseline "
        "(original/counterfactual of 40)",
        f"{fb[0]}/{fb[1]}", f"{gb[0]}/{gb[1]}",
        "recomputed_from_items", swap_src,
        "counted from per-item greedy_category rows",
        status=("byte_identical" if (fb == gb) else "failed"))
    add("swap_greedy_cf_swap",
        "Bridge swap: greedy generation hits under cf-bridge swap "
        "(original/counterfactual of 40)",
        f"{fs[0]}/{fs[1]}", f"{gs[0]}/{gs[1]}",
        "recomputed_from_items", swap_src,
        "counted from per-item greedy_category rows",
        status=("byte_identical" if (fs == gs) else "failed"))

    # ==================================================================
    # 5. Boundary-safe sensitivity (v2)
    # ==================================================================
    sens = jload(SENS_JSON)["payload"]
    partition = jload(PARTITION)["payload"]
    allowed = set(partition["confirmatory"])
    gsets = {}
    for slug in SLUGS:
        g5 = pd.read_parquet(SENS_DIR / f"g5_boundary_safe_{slug}.parquet")
        gsets[slug] = _capability_sets(g5, allowed)
    sens_src = [SENS_DIR / f"g5_boundary_safe_{s}.parquet" for s in SLUGS]

    def filter_grids(grids: dict, population: str) -> dict:
        return {s: grids[s][grids[s].fact_id.isin(
            gsets[s][population])].copy() for s in SLUGS}

    # -- boundary-safe strict P3-P2 / P3-P3 at the historical control --
    filt = filter_grids(conf, "boundary_safe_strict")
    qwen_b = filt["qwen36-27b"].copy()
    qwen_b["delta_J"] = qwen_b.J_eff
    qwen_b["delta_C"] = qwen_b.C_eff
    qwen_b["tail_difference"] = (
        (qwen_b.delta_J < -1.0).astype(float)
        - (qwen_b.delta_C < -1.0).astype(float))
    p2_family = qwen_b.groupby("canonical_family")["tail_difference"].mean()
    p2_est = float(p2_family.mean())
    label_ex = within_item_label_exchange_tail(
        qwen_b, draws=100_000, threshold=-1.0, seed=4242)
    bridge_b = qwen_b[qwen_b["lp_true_bridge"].notna()].copy()
    bridge_b["rescue"] = (
        bridge_b.lp_true_bridge - bridge_b.lp_distractor_bridge)
    p3_est = float(
        bridge_b.groupby("canonical_family")["rescue"].mean().mean())
    item_ex = within_item_exchange_mean(
        bridge_b, a_col="lp_true_bridge", b_col="lp_distractor_bridge",
        draws=100_000, seed=4242, alternative="greater")

    fb2 = sens["cohort_sensitivity"]["confirmatory"][
        "boundary_safe_strict"]["P3-P2"]
    fb3 = sens["cohort_sensitivity"]["confirmatory"][
        "boundary_safe_strict"]["P3-P3"]
    bsrc = sens_src + [PARTITION] + qsrc
    add("boundary_p3p2_estimate",
        "Boundary-safe strict cohort: P3-P2 tail excess at -1 nat, "
        "equal-family estimate",
        fb2["estimate_equal_family"], p2_est,
        "recomputed_from_items", bsrc,
        f"SOR quote +0.095833 on 186 items; recon n={len(qwen_b)} "
        f"(frozen {fb2['n_items']})")
    add("boundary_p3p2_p",
        "Boundary-safe strict: P3-P2 within-item label-exchange plus-one "
        "p (MC 100000, seed 4242)",
        fb2["label_exchange"]["p"], label_ex["p"],
        "recomputed_from_items", bsrc,
        "SOR quote p=2/100001")
    add("boundary_p3p3_estimate",
        "Boundary-safe strict cohort: P3-P3 rescue equal-family estimate "
        "(nats)",
        fb3["estimate_equal_family"], p3_est,
        "recomputed_from_items", bsrc,
        f"SOR quote +0.428230 on 93 items; recon n={len(bridge_b)} "
        f"(frozen {fb3['n_items']}); item-exchange rounded estimate "
        f"recon={item_ex['estimate']!r} frozen="
        f"{fb3['item_exchange']['estimate']!r}")
    add("boundary_p3p3_p",
        "Boundary-safe strict: P3-P3 within-item exchange plus-one p "
        "(MC 100000, seed 4242, one-sided greater)",
        fb3["item_exchange"]["p"], item_ex["p"],
        "recomputed_from_items", bsrc,
        "SOR quote p=0.009420")

    # -- boundary-safe strict P3-P1 at sha256-v1 seed 31337 -------------
    qwen_seed = side_rows[side_rows["audit_seed"] == 31337].copy()
    bridge_cols = conf["qwen36-27b"][
        ["item_id", "lp_true_bridge", "lp_distractor_bridge"]
    ].drop_duplicates("item_id")
    qwen_seed = qwen_seed.merge(
        bridge_cols, on="item_id", how="left", validate="one_to_one")
    qwen_seed["model"] = "qwen36-27b"
    qwen_seed["J_eff"] = (
        qwen_seed.lp_meanJ_span_safe - qwen_seed.lp_baseline)
    qwen_seed["C_eff"] = qwen_seed.lp_ss_matched - qwen_seed.lp_baseline
    qwen_seed["specific"] = qwen_seed.J_eff - qwen_seed.C_eff
    stable = dict(conf)
    stable["qwen36-27b"] = qwen_seed
    filt_seed = {s: stable[s][stable[s].fact_id.isin(
        gsets[s]["boundary_safe_strict"])].copy() for s in SLUGS}
    combined = pd.concat(filt_seed.values(), ignore_index=True)
    comp_seed = within_fact_composition(combined, value_col="specific")
    diff_seed = within_fact_model_diff(
        comp_seed, model_a="qwen36-27b",
        model_b=["olmo31-think", "olmo31-instruct"])
    fam_seed = diff_seed.groupby(
        "canonical_family", sort=True)["diff"].mean()
    exact_seed = exact_signflip_test(fam_seed.to_numpy())
    fseed = sens["control_seed_cohort_sensitivity"]["confirmatory"][
        "31337"]["boundary_safe_strict"]["P3-P1"]
    sha_seed = (exact_seed["distribution_sha256"]
                == fseed["exact_family_signflip"]["distribution_sha256"])
    add("boundary_p3p1_seed31337",
        "Boundary-safe strict cohort: P3-P1 at sha256-v1 control seed "
        "31337, equal-family estimate (nats)",
        fseed["estimate_equal_family"], exact_seed["estimate"],
        "recomputed_from_items",
        bsrc + [CSF_ROWS],
        f"SOR quote -0.270160; {exact_seed['n_families']} families / "
        f"{len(diff_seed)} facts (frozen {fseed['n_families']}/"
        f"{fseed['n_facts']}); exact p frozen="
        f"{fseed['exact_family_signflip']['p']!r} recon="
        f"{exact_seed['p']!r}; null sha match={sha_seed}"
        + ("" if sha_seed else f"; {BLAS_ULP_NOTE}"))

    # ------------------------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "target_id", "description", "frozen_value",
            "reconstructed_value", "method", "status", "source_paths",
            "notes"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nwrote {OUT_CSV} ({len(ROWS)} targets)")
    width = max(len(r["target_id"]) for r in ROWS)
    counts: dict[str, int] = {}
    for r in ROWS:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"  {r['target_id']:<{width}}  {r['status']}")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
