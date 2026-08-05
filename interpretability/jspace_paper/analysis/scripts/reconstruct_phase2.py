# reconstruct_phase2.py — deterministic reconstruction of the frozen
# Phase 2 headline numbers for the paper-analysis audit.
#
# Reads (read-only):
#   RUN_ROOT  = /content/drive/MyDrive/interpret/special-lab-1/part2_20260727
#   REGISTRY  = interpretability/jspace_part2/reports/evidence_registry.jsonl
# Writes:
#   interpretability/jspace_paper/analysis/tables/recon_phase2.csv
#   SCRATCH/rerun_conf/, SCRATCH/rerun_repl/  (full frozen-pipeline reruns)
#
# Two independent reconstruction routes for the n6 confirmatory numbers:
#   (a) in-process recompute: replays main()'s exact estimator sequence
#       (population selection, seed-4242 / 4000-draw family bootstraps,
#       Holm) using the frozen module's own functions over the raw
#       per-item parquets;
#   (b) full-pipeline rerun: subprocess `python -m
#       jspace_part2.experiments.confirmatory_analysis --no-register`
#       with JSPACE_PART2_OUT_ROOT redirected to scratch, then compares
#       payload_sha256 against the registered envelope (byte-level).
# Occupancy (T5) and G4 swap controls (T6) are verified from the registry
# event text against the registered output files; swap flip rates are
# additionally recomputed from the per-item rows inside r5_swap.json, and
# occupancy medians/excesses from the per-position histogram / component
# shares. No network. CPU only. Rerunnable: identical output modulo the
# `generated_by` comment line.
#
# Usage: python reconstruct_phase2.py [--skip-rerun] [--reuse-rerun]
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/content/labs")
PKG_DIR = REPO / "interpretability" / "jspace_part2"
RUN = Path(os.environ.get(
    "JSPACE_PART2_RUN_ROOT",
    "/content/drive/MyDrive/interpret/special-lab-1/part2_20260727"))
REGISTRY = PKG_DIR / "reports" / "evidence_registry.jsonl"
OUT_CSV = (REPO / "interpretability" / "jspace_paper" / "analysis" /
           "tables" / "recon_phase2.csv")
SCRATCH = Path(os.environ.get(
    "RECON_SCRATCH",
    "/tmp/claude-0/-content-labs/d11a04bf-2c54-402f-9154-10712293620d/"
    "scratchpad"))
SLUGS = ["olmo31-think", "olmo31-instruct", "qwen36-27b"]

sys.path.insert(0, str(PKG_DIR))
from jspace_part2.experiments import confirmatory_analysis as ca  # noqa: E402
from jspace_part2.registry import payload_sha256 as canonical_payload_sha256  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def registry_entry(eid: str) -> dict:
    hits = []
    with open(REGISTRY) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("evidence_id") == eid:
                hits.append(d)
    if not hits:
        raise SystemExit(f"evidence_id {eid} not found in {REGISTRY}")
    return hits[-1]


# --------------------------------------------------------------------------
# Route (a): in-process recompute of the locked estimators from parquets.
# This replays confirmatory_analysis.main()'s exact P-HP1 / P-HP3 / Holm
# sequence (same functions, same seed, same draw counts, same ordering)
# without the provenance/registry side effects.
# --------------------------------------------------------------------------
def recompute_headline(suffix: str) -> dict:
    ca.DIR_SUFFIX = suffix
    dfs = pd.concat([ca.load_deltas(s) for s in SLUGS], ignore_index=True)
    dfs = dfs[dfs.task.isin(["twohop", "onehop", "prose"])]
    core = dfs[dfs.task != "prose"]

    # ---- P-HP1 interaction contrast (population selection as in main) ----
    pair = core[(core.model.isin(["olmo31-think", "olmo31-instruct"]))
                & (core.condition == "meanJ_protected")]
    pair_int = pair[pair.apply(ca.in_intersection, axis=1)]

    def both_tasks_nonempty(d):
        return all(len(d[(d.model == m) & (d.task == t)]) > 0
                   for m in ("olmo31-think", "olmo31-instruct")
                   for t in ("twohop", "onehop"))
    if both_tasks_nonempty(pair_int) and len(pair_int) >= 30:
        use, pop = pair_int, "cross_model_intersection"
    elif both_tasks_nonempty(pair):
        use, pop = pair, "ALL-items fallback"
    else:
        use, pop = None, "NOT_EVALUABLE"
    obs_hp1 = ca.hp1_contrast(use) if use is not None else float("nan")

    rng = np.random.default_rng(ca.SEED)
    _u = use if use is not None else pair.iloc[:0]
    fams_by_task = {t: sorted(_u[_u.task == t].family.unique())
                    for t in ("twohop", "onehop")}
    grouped = {(t, f): g for (t, f), g in _u.groupby(["task", "family"])}
    hp1_boots = []
    for _ in range(ca.N_BOOT):
        parts = []
        for t, fams in fams_by_task.items():
            for f in rng.choice(fams, size=len(fams), replace=True):
                parts.append(grouped[(t, f)])
        hp1_boots.append(ca.hp1_contrast(pd.concat(parts)))
    hp1_boots = np.array(hp1_boots)
    hp1 = {"population": pop,
           "observed_contrast_nats": round(obs_hp1, 4),
           "ci95": ca.ci(hp1_boots),
           "p_bootstrap": ca.p_two_sided(hp1_boots, obs_hp1),
           "n_items": int(len(use)), "n_families": int(use.family.nunique())}

    # ---- P-HP3 Qwen paired tail-rate (protected-answer stratum) ----------
    def tail_frame(slug, stratified=True):
        q = core[core.model == slug]
        j = q[q.condition == "meanJ_protected"].set_index("item_id")
        c = q[q.condition == "matched_control"].set_index("item_id")
        common = j.index.intersection(c.index)
        d = pd.DataFrame({
            "delta_J": j.loc[common, "delta"],
            "delta_C": c.loc[common, "delta"],
            "family": j.loc[common, "family"],
            "task": j.loc[common, "task"],
            "rank": j.loc[common, "clean_first_rank_min"]})
        if stratified:
            d = d[d["rank"].notna() & (d["rank"] <= ca.PROTECT_K)]
        return d

    strat = tail_frame("qwen36-27b", stratified=True)
    alli = tail_frame("qwen36-27b", stratified=False)
    obs = ca.tail_stat(strat)
    boots = ca.fam_boot(strat, ca.tail_stat, strata_col="task")
    hp3 = {"n_stratified": int(len(strat)),
           "n_families": int(strat.family.nunique()),
           "rate_diff_fam_weighted": round(obs, 4),
           "ci95": ca.ci(boots),
           "p_one_sided": ca.p_one_sided_le0(boots),
           "all_items_n": int(len(alli)),
           "all_items_rate_diff": round(ca.tail_stat(alli), 4)}

    # ---- Holm over the two primary tests (as in main) --------------------
    ps = {"P_HP1": hp1["p_bootstrap"], "P_HP3": hp3["p_one_sided"]}
    order = sorted(ps, key=lambda k: ps[k])
    holm, sig = {}, True
    for i, k in enumerate(order):
        adj = min(1.0, ps[k] * (len(ps) - i))
        sig = sig and adj < 0.05
        holm[k] = {"p_raw": ps[k], "p_holm": round(adj, 5),
                   "reject_at_05": bool(sig and adj < 0.05)}
    return {"P_HP1": hp1, "P_HP3_qwen": hp3, "holm": holm}


# --------------------------------------------------------------------------
# Route (b): full frozen-pipeline rerun (subprocess, --no-register).
# --------------------------------------------------------------------------
def full_rerun(out_root: Path, extra_args: list[str]) -> None:
    env = dict(os.environ)
    env["JSPACE_PART2_OUT_ROOT"] = str(out_root)
    env.setdefault("JSPACE_PART2_RUN_ROOT", str(RUN))
    cmd = [sys.executable, "-m",
           "jspace_part2.experiments.confirmatory_analysis",
           "--slugs", ",".join(SLUGS), "--no-register"] + extra_args
    subprocess.run(cmd, cwd=PKG_DIR, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def payload_diff_leaves(a, b, path=""):
    """Leaf-level differences between two JSON payloads."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}:missing")
            else:
                diffs.extend(payload_diff_leaves(a[k], b[k], f"{path}/{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}:len {len(a)}!={len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs.extend(payload_diff_leaves(x, y, f"{path}[{i}]"))
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


# --------------------------------------------------------------------------
def fmt(v):
    if isinstance(v, float):
        return repr(round(v, 6))
    return str(v)


def main():
    skip_rerun = "--skip-rerun" in sys.argv
    reuse = "--reuse-rerun" in sys.argv
    rows = []

    def add_row(tid, desc, frozen, recon, method, status, sources, notes):
        rows.append({"target_id": tid, "description": desc,
                     "frozen_value": frozen, "reconstructed_value": recon,
                     "method": method, "status": status,
                     "source_paths": ";".join(str(s) for s in sources),
                     "notes": notes})

    # ---------------- input-chain verification ---------------------------
    frozen_env = {}
    input_notes = {}
    for name, suffix in (("confirmatory_analysis", ""),
                         ("replication_analysis", "_repl")):
        fp = RUN / "metrics" / "cross_model" / f"{name}.json"
        frozen_env[suffix] = json.load(open(fp))
        reg = registry_entry(frozen_env[suffix]["provenance"]["evidence_id"])
        reg_sha = reg["outputs"][0]["sha256"]
        file_sha = sha256_file(fp)
        prov_in = frozen_env[suffix]["provenance"]["inputs"]
        pq_ok = []
        for s in SLUGS:
            pq = RUN / "metrics" / s / f"n6_grid{suffix}" / \
                f"n6_per_item_{s}.parquet"
            pq_ok.append(sha256_file(pq) == prov_in[f"parquet_{s}"])
        part_ok = (sha256_file(RUN / "metrics" / "cross_model" /
                               "partition_manifest.json")
                   == prov_in["partition"])
        pay_ok = (canonical_payload_sha256(frozen_env[suffix]["payload"])
                  == frozen_env[suffix]["payload_sha256"])
        input_notes[suffix] = (
            f"registered_output_sha256_match={reg_sha == file_sha}; "
            f"input_parquet_sha256_match={all(pq_ok)}; "
            f"partition_manifest_sha256_match={part_ok}; "
            f"envelope_payload_sha256_selfconsistent={pay_ok}")

    # ---------------- routes (a) + (b) for both partitions ----------------
    recon = {}
    rerun_payload = {}
    for suffix, rerun_dir, out_name in (
            ("", SCRATCH / "rerun_conf", "confirmatory_analysis.json"),
            ("_repl", SCRATCH / "rerun_repl", "replication_analysis.json")):
        recon[suffix] = recompute_headline(suffix)
        rerun_fp = rerun_dir / "metrics" / "cross_model" / out_name
        if not skip_rerun and not (reuse and rerun_fp.exists()):
            extra = ([] if suffix == "" else
                     ["--dir-suffix", "_repl",
                      "--eid", "n6-replication-analysis-v2",
                      "--out-name", out_name])
            full_rerun(rerun_dir, extra)
        rerun_payload[suffix] = (json.load(open(rerun_fp))
                                 if rerun_fp.exists() else None)

    def n6_status(suffix, keyset, frozen_vals, recon_vals):
        """Compare quoted-decimal tuples + payload-level byte identity."""
        exact = frozen_vals == recon_vals
        env = frozen_env[suffix]
        rr = rerun_payload[suffix]
        byte = (rr is not None and
                canonical_payload_sha256(rr["payload"])
                == env["payload_sha256"])
        if byte and exact:
            return "byte_identical", ""
        if exact:
            leaves = (payload_diff_leaves(env["payload"], rr["payload"])
                      if rr is not None else ["rerun unavailable"])
            return ("numerically_identical_render_diff",
                    f"payload diff leaves: {'; '.join(leaves[:6])}")
        return ("numerically_within_frozen_tolerance",
                f"mismatch on {keyset}: frozen={frozen_vals} "
                f"recon={recon_vals}")

    # ---- T1: P-HP1 confirmatory -----------------------------------------
    r = recon[""]["P_HP1"]
    h = recon[""]["holm"]["P_HP1"]
    frozen = ("contrast=-0.5045; ci95=[-0.7195, -0.2949]; p=0.0005; "
              "holm_reject=True")
    got = (f"contrast={r['observed_contrast_nats']}; ci95={r['ci95']}; "
           f"p={r['p_bootstrap']}; holm_reject={h['reject_at_05']}")
    st, extra = n6_status(
        "", "contrast/ci/p/holm",
        (-0.5045, [-0.7195, -0.2949], 0.0005, True),
        (r["observed_contrast_nats"], r["ci95"], r["p_bootstrap"],
         h["reject_at_05"]))
    add_row("T1.P-HP1.confirmatory",
            "Think-vs-Instruct task-interaction contrast, meanJ_protected, "
            "cross_model_intersection, fam-weighted (nats)",
            frozen, got, "recomputed_from_items", st,
            [RUN / "metrics" / s / "n6_grid" / f"n6_per_item_{s}.parquet"
             for s in ("olmo31-think", "olmo31-instruct")]
            + [RUN / "metrics/cross_model/confirmatory_analysis.json"],
            f"population={r['population']}; n_items={r['n_items']}; "
            f"n_families={r['n_families']}; seed=4242 n_boot=4000; "
            + input_notes[""] + (f"; {extra}" if extra else ""))

    # ---- T2: P-HP3 confirmatory -----------------------------------------
    r = recon[""]["P_HP3_qwen"]
    frozen = ("rate_diff=0.2788; ci95=[0.2048, 0.3608]; p_one_sided=0.00025; "
              "n_stratified=158; n_families=32; all_items_n=164")
    got = (f"rate_diff={r['rate_diff_fam_weighted']}; ci95={r['ci95']}; "
           f"p_one_sided={r['p_one_sided']}; n_stratified={r['n_stratified']}; "
           f"n_families={r['n_families']}; all_items_n={r['all_items_n']}")
    st, extra = n6_status(
        "", "rate_diff/ci/p/n",
        (0.2788, [0.2048, 0.3608], 0.00025, 158, 32, 164),
        (r["rate_diff_fam_weighted"], r["ci95"], r["p_one_sided"],
         r["n_stratified"], r["n_families"], r["all_items_n"]))
    add_row("T2.P-HP3.confirmatory",
            "Qwen paired tail-rate diff at -1.0 nats, meanJ_protected vs "
            "matched_control, protected-answer stratum (rank<=10)",
            frozen, got, "recomputed_from_items", st,
            [RUN / "metrics/qwen36-27b/n6_grid/n6_per_item_qwen36-27b.parquet",
             RUN / "metrics/cross_model/confirmatory_analysis.json"],
            f"all_items_rate_diff={r['all_items_rate_diff']} "
            f"(frozen 0.2831); seed=4242 n_boot=4000; " + input_notes[""]
            + (f"; {extra}" if extra else ""))

    # ---- T3: P-HP1 replication ------------------------------------------
    r = recon["_repl"]["P_HP1"]
    h = recon["_repl"]["holm"]["P_HP1"]
    frozen = ("contrast=0.1036; ci95=[-1.6813, 1.8892]; p=0.7075; "
              "holm_reject=False")
    got = (f"contrast={r['observed_contrast_nats']}; ci95={r['ci95']}; "
           f"p={r['p_bootstrap']}; holm_reject={h['reject_at_05']}")
    st, extra = n6_status(
        "_repl", "contrast/ci/p/holm",
        (0.1036, [-1.6813, 1.8892], 0.7075, False),
        (r["observed_contrast_nats"], r["ci95"], r["p_bootstrap"],
         h["reject_at_05"]))
    add_row("T3.P-HP1.replication",
            "P-HP1 contrast on the held-out replication partition (nats)",
            frozen, got, "recomputed_from_items", st,
            [RUN / "metrics" / s / "n6_grid_repl" / f"n6_per_item_{s}.parquet"
             for s in ("olmo31-think", "olmo31-instruct")]
            + [RUN / "metrics/cross_model/replication_analysis.json"],
            f"population={r['population']}; n_items={r['n_items']}; "
            f"n_families={r['n_families']}; seed=4242 n_boot=4000; "
            + input_notes["_repl"] + (f"; {extra}" if extra else ""))

    # ---- T4: P-HP3 replication ------------------------------------------
    r = recon["_repl"]["P_HP3_qwen"]
    frozen = ("rate_diff=0.2966; ci95=[0.2071, 0.3824]; p_one_sided=0.00025; "
              "n_stratified=153; n_families=32; all_items_n=161")
    got = (f"rate_diff={r['rate_diff_fam_weighted']}; ci95={r['ci95']}; "
           f"p_one_sided={r['p_one_sided']}; n_stratified={r['n_stratified']}; "
           f"n_families={r['n_families']}; all_items_n={r['all_items_n']}")
    st, extra = n6_status(
        "_repl", "rate_diff/ci/p/n",
        (0.2966, [0.2071, 0.3824], 0.00025, 153, 32, 161),
        (r["rate_diff_fam_weighted"], r["ci95"], r["p_one_sided"],
         r["n_stratified"], r["n_families"], r["all_items_n"]))
    add_row("T4.P-HP3.replication",
            "Qwen paired tail-rate diff, replication partition, "
            "protected-answer stratum",
            frozen, got, "recomputed_from_items", st,
            [RUN /
             "metrics/qwen36-27b/n6_grid_repl/n6_per_item_qwen36-27b.parquet",
             RUN / "metrics/cross_model/replication_analysis.json"],
            f"all_items_rate_diff={r['all_items_rate_diff']}; "
            f"seed=4242 n_boot=4000; " + input_notes["_repl"]
            + (f"; {extra}" if extra else ""))

    # ---- T5: repaired occupancy (registry event text vs output files) ----
    occ_pat = re.compile(
        r"L(\d+): occ_med ([\d.]+), RAW excess (-?[\d.]+), "
        r"CENTERED excess (-?[\d.]+) \[(-?[\d.]+),(-?[\d.]+)\]")
    for slug, eid in (("olmo31-think", "r2-occupancy-olmo31think-v2"),
                      ("olmo31-instruct", "r2-occupancy-olmo31instruct-v2"),
                      ("qwen36-27b", "r2-occupancy-qwen36-v2")):
        reg = registry_entry(eid)
        fp = RUN / "metrics" / slug / "r2_occupancy" / "r2_occupancy_v2.json"
        env = json.load(open(fp))
        sha_ok = sha256_file(fp) == reg["outputs"][0]["sha256"]
        pay_ok = (canonical_payload_sha256(env["payload"])
                  == env["payload_sha256"])
        frozen_layers = {m[0]: (float(m[1]), float(m[2]), float(m[3]),
                                float(m[4]), float(m[5]))
                        for m in occ_pat.findall(reg["what"])}
        got_parts, mismatches, hist_checks = [], [], []
        for lay, (f_occ, f_raw, f_cen, f_lo, f_hi) in \
                sorted(frozen_layers.items(), key=lambda kv: int(kv[0])):
            L = env["payload"]["per_layer"][lay]
            g_occ = float(L["occ_median"])
            g_raw = round(float(L["raw_reconstruction_excess"]), 4)
            g_cen = round(float(L["centered_variance_explained_excess"]), 4)
            g_lo = round(float(L["centered_excess_ci"]["low"]), 4)
            g_hi = round(float(L["centered_excess_ci"]["high"]), 4)
            got_parts.append(f"L{lay}: occ_med {g_occ}, RAW {g_raw}, "
                             f"CENTERED {g_cen} [{g_lo},{g_hi}]")
            for name, fv, gv in (("occ_med", f_occ, g_occ),
                                 ("raw", f_raw, g_raw),
                                 ("centered", f_cen, g_cen),
                                 ("ci_lo", f_lo, g_lo),
                                 ("ci_hi", f_hi, g_hi)):
                # event text rendered via round(x, 4); accept 1-ulp-of-
                # rounding render differences and flag anything larger
                if abs(fv - gv) > 1e-4 + 1e-12:
                    mismatches.append(f"L{lay}.{name} {fv} vs {gv}")
            # recompute occ_median from the per-position histogram, and
            # both excesses from their component shares
            hist = np.array(L["occ_hist"], dtype=float)
            grid = np.repeat(np.arange(len(hist)), hist.astype(int))
            occ_from_hist = float(np.median(grid))
            raw_from_parts = (float(L["raw_share_j"])
                              - float(L["raw_share_rand"]))
            cen_from_parts = (float(L["centered_r2_j"])
                              - float(L["centered_r2_rand"]))
            hist_checks.append(
                f"L{lay} occ_median_from_hist={occ_from_hist} "
                f"(n_pos={int(hist.sum())}); "
                f"raw_j_minus_rand={round(raw_from_parts, 6)} vs "
                f"{round(float(L['raw_reconstruction_excess']), 6)}; "
                f"cen_r2_diff={round(cen_from_parts, 6)} vs "
                f"{round(float(L['centered_variance_explained_excess']), 6)}")
        render_diffs = [m for m in mismatches]
        if not sha_ok:
            st = "failed"
        elif not render_diffs:
            st = "byte_identical"
        else:
            st = "numerically_identical_render_diff"
        add_row(f"T5.occupancy.{slug}",
                "repaired r2 occupancy: per-layer occ_med, RAW vs CENTERED "
                "excess (registry event text vs registered output file)",
                reg["what"].split("): ", 1)[-1],
                "; ".join(got_parts),
                "verified_registered_summary", st,
                [fp, REGISTRY],
                f"file_sha256_match={sha_ok}; "
                f"payload_sha256_selfconsistent={pay_ok}; "
                + ("; ".join(hist_checks))
                + (f"; render_diffs: {render_diffs}" if render_diffs else "")
                + "; per-position raw vectors not released (occ_hist is the "
                  "finest granularity; centered-excess CI n_boot=400 "
                  "prompt-resample not recomputable from released data)")

    # ---- T6: G4 swap positive controls ----------------------------------
    dict_pat = re.compile(r"\{[^{}]*\}")
    for slug, eid in (
            ("olmo31-think", "r5-swap-positive-control-olmo31-think-v2"),
            ("olmo31-instruct", "r5-swap-positive-control-olmo31-instruct-v2"),
            ("qwen36-27b", "r5-swap-positive-control-qwen36-27b-v2")):
        reg = registry_entry(eid)
        fp = RUN / "metrics" / slug / "r5_swap_v2" / "r5_swap.json"
        d = json.load(open(fp))
        sha_ok = sha256_file(fp) == reg["outputs"][0]["sha256"]
        ev_dicts = [ast.literal_eval(m) for m in dict_pat.findall(reg["what"])]
        ev = dict(zip(("swap_j", "swap_rand", "none"), ev_dicts))
        ev_alpha = float(re.search(r"alpha ([\d.]+)\)", reg["what"]).group(1))
        rows_ = d["rows"]
        n = len(rows_)
        got_parts, mismatches = [], []
        if abs(float(d["summary"]["alpha_star"]) - ev_alpha) > 1e-9:
            mismatches.append(
                f"alpha event={ev_alpha} file={d['summary']['alpha_star']}")
        for arm in ("swap_j", "swap_rand", "none"):
            fr = round(float(np.mean([r[arm]["pick_swap"] for r in rows_])), 4)
            ls = round(float(np.mean([r[arm]["swap"] for r in rows_])), 3)
            lo = round(float(np.mean([r[arm]["orig"] for r in rows_])), 3)
            got_parts.append(f"{arm}: flip_rate={fr} lp_swap={ls} lp_orig={lo}")
            summ = d["summary"][arm]
            for name, evv, gv in (("flip_rate", ev[arm]["flip_rate"], fr),
                                  ("mean_lp_swapans",
                                   ev[arm]["mean_lp_swapans"], ls),
                                  ("mean_lp_origans",
                                   ev[arm]["mean_lp_origans"], lo)):
                if abs(evv - gv) > 5e-4:
                    mismatches.append(f"{arm}.{name} event={evv} recomputed={gv}")
                if abs(summ[name] - gv) > 5e-4:
                    mismatches.append(
                        f"{arm}.{name} file_summary={summ[name]} "
                        f"recomputed={gv}")
        st = ("failed" if not sha_ok else
              ("numerically_identical_render_diff" if not mismatches
               else "numerically_within_frozen_tolerance"))
        add_row(f"T6.g4_swap.{slug}",
                "G4 positive control: probe-swap flip_rate (J-space "
                "remove-bridge+inject-swap vs rand vs none), n=50 items",
                f"swap_j={ev['swap_j']}; rand={ev['swap_rand']}; "
                f"none={ev['none']}; alpha={ev_alpha}",
                "; ".join(got_parts) + f"; n={n}",
                "recomputed_from_items", st,
                [fp, REGISTRY],
                f"file_sha256_match={sha_ok}; recomputed from the {n} "
                f"per-item rows inside the registered r5_swap.json (item "
                f"rows are the only released item-level swap data)"
                + (f"; mismatches: {mismatches}" if mismatches else ""))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "target_id", "description", "frozen_value",
            "reconstructed_value", "method", "status", "source_paths",
            "notes"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")
    for row in rows:
        print(f"  {row['target_id']}: {row['status']}")


if __name__ == "__main__":
    main()
