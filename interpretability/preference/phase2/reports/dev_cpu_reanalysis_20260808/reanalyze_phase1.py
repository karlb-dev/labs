#!/usr/bin/env python3
"""Phase 2 pre-VM CPU re-analysis of the frozen Lab 38 Phase 1 record.

Development tier. Inputs are the frozen, hash-pinned Phase 1 artifacts
(read-only): frozen_{7b,32b}/results.jsonl, the frozen bank, and the
frozen graduation tables (used only to VALIDATE this pipeline, never to
source numbers).

Analyses (numbering follows preference_2_1.md §0.2):
  A  pipeline validation: reproduce graduation `effect` and
     `position_effect` per scenario from raw rows (must match exactly)
  F1 censoring identity: position + |content| vs 0.500 per scenario
  F2 folded-margin refold: per-scenario content terms on the teacher-
     forced margin, incidental-clustered t (df=4), sign census, NC floor,
     cross-model and cross-channel structure
  F3 label/position aliasing census from the frozen bank
  F4 mechanism retrodiction: replicate the frozen nuisance design
     (intercept + order + label_set + code_map + frame + drop-one train-
     incidental FEs) per scenario-channel; train R^2, residual sd, raw sd
     (MIN_MARGIN_STD check), rank; PC-vs-AR fittability
  F5 RO constant-code census + AR-only vs pooled matched-pair agreement

Outputs: reanalysis.json (all numbers) + reanalysis_summary.md (prose).
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

import argparse
import os


def _interp_root() -> pathlib.Path:
    """Portable root discovery (plan §6.3): --repo-root arg, else
    $PREF2_REPO_ROOT, else walk up from this file to `.git`."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--out", default=None)
    ns, _ = ap.parse_known_args()
    if ns.out:
        globals()["_OUT_OVERRIDE"] = pathlib.Path(ns.out).resolve()
    root = ns.repo_root or os.environ.get("PREF2_REPO_ROOT")
    if root:
        return pathlib.Path(root).resolve() / "interpretability"
    here = pathlib.Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            return parent / "interpretability"
    raise RuntimeError("repo root not found; pass --repo-root or set PREF2_REPO_ROOT")


_OUT_OVERRIDE = None
REPO = _interp_root()
PREF = REPO / "preference"
P1 = PREF / "phase1" / "reports"
BANK = PREF / "data" / "lab38_preference_bank.jsonl"
OUT = _OUT_OVERRIDE or pathlib.Path(__file__).resolve().parent

T90_DF4 = 2.131846786  # two-sided 90% critical t, df=4 (5 incidentals)


def load_rows(model: str) -> list[dict]:
    path = P1 / f"frozen_{model}" / "results.jsonl"
    return [json.loads(l) for l in open(path)]


def load_bank() -> list[dict]:
    return [json.loads(l) for l in open(BANK)]


def load_graduation(model: str) -> dict[str, dict]:
    path = P1 / f"frozen_{model}" / "tables" / "graduation_decisions.csv"
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["scenario_id"]] = r
    return out


def valid_choice(r: dict) -> bool:
    return r["parse_status"] == "valid" and r["parsed_pole"] in (0, 1)


def chose_first(r: dict) -> int:
    """1 if the parsed pole is the first-displayed pole.

    Convention calibrated in validate_pipeline(): order_index==0 means
    pole_0 is displayed first (asserted against the frozen tables).
    """
    first_pole = 0 if r["order_index"] == 0 else 1
    return int(r["parsed_pole"] == first_pole)


# ---------------------------------------------------------------- A: validate
def scenario_choice_stats(rows: list[dict]) -> dict[str, dict]:
    by_scen = defaultdict(list)
    for r in rows:
        if r["channel"] == "AR" or r["family"] in ("PC", "NC"):
            # graduation tables cover AR channel + PC + NC families
            if r["channel"] == "AR":
                by_scen[r["scenario_id"]].append(r)
    out = {}
    for scen, rs in by_scen.items():
        vs = [r for r in rs if valid_choice(r)]
        if not vs:
            continue
        p1 = np.mean([r["parsed_pole"] for r in vs])
        pf = np.mean([chose_first(r) for r in vs])
        out[scen] = {
            "n_rows": len(rs),
            "n_valid": len(vs),
            "content_effect": float(p1 - 0.5),
            "position_effect": float(pf - 0.5),
        }
    return out


def validate_pipeline(model: str, rows: list[dict]) -> dict:
    got = scenario_choice_stats(rows)
    grad = load_graduation(model)
    checks = []
    for scen, g in grad.items():
        if scen not in got:
            checks.append({"scenario": scen, "ok": False, "why": "missing"})
            continue
        eff_ok = math.isclose(got[scen]["content_effect"], float(g["effect"]),
                              abs_tol=1e-9)
        nuis = json.loads(g["nuisances"])
        pos_ok = math.isclose(got[scen]["position_effect"],
                              float(nuis["position_effect"]), abs_tol=1e-9)
        checks.append({"scenario": scen, "ok": eff_ok and pos_ok,
                       "effect_ok": eff_ok, "position_ok": pos_ok,
                       "got_effect": got[scen]["content_effect"],
                       "table_effect": float(g["effect"]),
                       "got_position": got[scen]["position_effect"],
                       "table_position": float(nuis["position_effect"])})
    n_ok = sum(c["ok"] for c in checks)
    return {"n_scenarios": len(checks), "n_exact_match": n_ok,
            "all_match": n_ok == len(checks), "checks": checks,
            "stats": got}


# --------------------------------------------------------------- F1: identity
def censoring_identity(stats: dict[str, dict], grad: dict[str, dict]) -> dict:
    per = {}
    for scen, s in stats.items():
        fam = grad[scen]["family"] if scen in grad else "?"
        total = s["position_effect"] + abs(s["content_effect"])
        per[scen] = {"family": fam,
                     "position": s["position_effect"],
                     "abs_content": abs(s["content_effect"]),
                     "sum": total, "dev_from_half": total - 0.5,
                     "n_invalid": s["n_rows"] - s["n_valid"]}
    devs = [abs(v["dev_from_half"]) for v in per.values()
            if v["family"] in ("AR", "NC")]
    exact = sum(1 for d in devs if d < 1e-9)
    return {"per_scenario": per, "n_ar_nc": len(devs),
            "n_exact": exact, "max_abs_dev": max(devs) if devs else None}


# ------------------------------------------------------------ F2: folded margin
def folded_margins(rows: list[dict]) -> dict:
    """Per (channel, scenario): incidental-level folded margin means.

    Fold = plain mean of margin_pole1_minus_pole0 over the full
    counterbalance within an incidental; surface terms cancel by the
    exact counterbalance, leaving content + any pole-authoring lexical
    term. Clustered t across incidentals (df = n_inc - 1).
    """
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not r["margin_finite"]:
            continue
        cells[(r["channel"], r["scenario_id"])][r["incidental_id"]].append(
            r["margin_pole1_minus_pole0"])
    out = {}
    for (ch, scen), inc in sorted(cells.items()):
        means = {k: float(np.mean(v)) for k, v in sorted(inc.items())}
        m = np.array(list(means.values()))
        n = len(m)
        mu = float(m.mean())
        sd = float(m.std(ddof=1)) if n > 1 else float("nan")
        se = sd / math.sqrt(n) if n > 1 else float("nan")
        t = mu / se if se and se > 0 else float("nan")
        out[f"{ch}:{scen}"] = {
            "channel": ch, "scenario": scen, "n_incidentals": n,
            "per_incidental": means, "mean_nats": mu, "sd_nats": sd,
            "t": t, "sig90": bool(abs(t) >= T90_DF4) if n == 5 else None,
            "sign": "pole_0" if mu < 0 else ("pole_1" if mu > 0 else "zero"),
        }
    return out


def f2_census(fold: dict, families: dict[str, str]) -> dict:
    def fam(scen):
        return families.get(scen, "?")
    ar = {k: v for k, v in fold.items()
          if fam(v["scenario"]) == "AR"}
    ar_ar = {k: v for k, v in ar.items() if v["channel"] == "AR"}
    ar_ro = {k: v for k, v in ar.items() if v["channel"] == "RO"}
    nc = {k: v for k, v in fold.items() if fam(v["scenario"]) == "NC"}
    pc = {k: v for k, v in fold.items()
          if fam(v["scenario"]) == "PC" and v["channel"] == "AR"}

    def census(d):
        sig = sum(1 for v in d.values() if v["sig90"])
        neg = sum(1 for v in d.values() if v["mean_nats"] < 0)
        ts = sorted(abs(v["t"]) for v in d.values() if not math.isnan(v["t"]))
        return {"n": len(d), "n_sig90": sig, "n_toward_pole0": neg,
                "abs_t_min": ts[0] if ts else None,
                "abs_t_max": ts[-1] if ts else None}

    return {"AR_enacted": census(ar_ar), "AR_report_only": census(ar_ro),
            "NC_all_channels": census(nc), "PC_enacted": census(pc),
            "nc_abs_mean_max": max((abs(v["mean_nats"]) for v in nc.values()),
                                   default=None),
            "pc_means": {k: v["mean_nats"] for k, v in pc.items()}}


# ---------------------------------------------------------------- F3: aliasing
def aliasing_census(bank: list[dict]) -> dict:
    n = len(bank)
    label_first_rank = 0
    reply_is_display_order = 0
    for r in bank:
        labels = r["display_labels"]
        lset = sorted(labels)
        if labels[0] == lset[0]:
            label_first_rank += 1
        codes_disp = r["valid_codes_in_display_order"]
        cb = r["response_code_by_pole"]
        order = r["order_index"]
        first_pole = "0" if order == 0 else "1"
        second_pole = "1" if order == 0 else "0"
        if codes_disp == [cb[first_pole], cb[second_pole]]:
            reply_is_display_order += 1
    return {"n_rows": n,
            "label_rank_aliased_rate": label_first_rank / n,
            "reply_order_aliased_rate": reply_is_display_order / n}


# ------------------------------------------------------------ F4: retrodiction
def nuisance_design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the frozen per-scenario design on TRAIN incidentals:
    intercept, order_index, label_set==letters, code_map_index,
    frame==enacted, drop-one FEs over the train incidentals.
    (prompt_token_count column: absorbed per declared simplification D8.)
    """
    train = [r for r in rows if r["incidental_split"] == "train"
             and r["margin_finite"]]
    incs = sorted({r["incidental_id"] for r in train})
    cols = []
    y = []
    for r in train:
        base = [1.0, float(r["order_index"]),
                1.0 if r["display_label_set"] == "letters" else 0.0,
                float(r["code_map_index"]),
                1.0 if r["consequence_frame"] == "enacted" else 0.0]
        fe = [1.0 if r["incidental_id"] == i else 0.0 for i in incs[1:]]
        cols.append(base + fe)
        y.append(r["margin_pole1_minus_pole0"])
    return np.array(cols), np.array(y)


def retrodiction(rows: list[dict], families: dict[str, str]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["channel"], r["scenario_id"])].append(r)
    out = {}
    for (ch, scen), rs in sorted(by.items()):
        X, y = nuisance_design(rs)
        if len(y) < 8:
            continue
        raw_sd = float(y.std(ddof=1))
        rank = int(np.linalg.matrix_rank(X))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        rss = float((resid ** 2).sum())
        tss = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        out[f"{ch}:{scen}"] = {
            "family": families.get(scen, "?"), "channel": ch,
            "n_train": len(y), "design_cols": X.shape[1],
            "design_rank": rank, "rank_deficient": rank < X.shape[1],
            "raw_margin_sd": raw_sd,
            "raw_gate_would_exclude": raw_sd < 0.10,
            "nuisance_R2": r2,
            "residual_margin_sd": float(resid.std(ddof=1)),
            "residual_gate_would_exclude": float(resid.std(ddof=1)) < 0.10,
        }
    return out


# ----------------------------------------------------------------- F5: RO/conc
def ro_constant_code(rows: list[dict], families: dict[str, str]) -> dict:
    by = defaultdict(list)
    for r in rows:
        if r["channel"] == "RO" and families.get(r["scenario_id"]) == "AR" \
                and valid_choice(r):
            by[r["scenario_id"]].append(r["parsed_response_code"])
    out = {}
    for scen, codes in sorted(by.items()):
        c = Counter(codes)
        top, ntop = c.most_common(1)[0]
        out[scen] = {"n_valid": len(codes), "n_distinct_codes": len(c),
                     "top_code": top, "top_share": ntop / len(codes),
                     "constant": len(c) == 1}
    n_const = sum(1 for v in out.values() if v["constant"])
    return {"per_scenario": out, "n_scenarios": len(out),
            "n_constant_code": n_const}


def concordance(rows: list[dict], families: dict[str, str]) -> dict:
    ar = {}
    ro = {}
    for r in rows:
        if not valid_choice(r):
            continue
        if r["channel"] == "AR":
            ar[r["pair_key"]] = r
        elif r["channel"] == "RO":
            ro[r["pair_key"]] = r
    agree_ar, n_ar, agree_all, n_all = 0, 0, 0, 0
    for k, ra in ar.items():
        rb = ro.get(k)
        if rb is None:
            continue
        same = int(ra["parsed_pole"] == rb["parsed_pole"])
        n_all += 1
        agree_all += same
        if families.get(ra["scenario_id"]) == "AR":
            n_ar += 1
            agree_ar += same
    return {"pooled_agreement": agree_all / n_all if n_all else None,
            "pooled_n": n_all,
            "ar_only_agreement": agree_ar / n_ar if n_ar else None,
            "ar_only_n": n_ar,
            "mechanical_floor": 0.5}


# ----------------------------------------------------------------------- main
def family_map(bank: list[dict]) -> dict[str, str]:
    fam = {}
    for r in bank:
        fam[r["scenario_id"]] = r["family"]
    return fam


def cross_structure(f7: dict, f32: dict, families: dict[str, str]) -> dict:
    scens = sorted({v["scenario"] for v in f7.values()
                    if families.get(v["scenario"]) == "AR"})
    def vec(fold, ch):
        return np.array([fold[f"{ch}:{s}"]["mean_nats"] for s in scens
                         if f"{ch}:{s}" in fold])
    a7, a32 = vec(f7, "AR"), vec(f32, "AR")
    r7, r32 = vec(f7, "RO"), vec(f32, "RO")
    def corr(a, b):
        if len(a) != len(b) or len(a) < 3:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    def signmatch(a, b):
        return int(np.sum(np.sign(a) == np.sign(b)))
    return {"scenarios": scens,
            "ar_7b_vs_32b_r": corr(a7, a32),
            "ar_7b_vs_32b_signmatch": signmatch(a7, a32),
            "ar_vs_ro_7b_r": corr(a7, r7),
            "ar_vs_ro_32b_r": corr(a32, r32),
            "ar_folded_7b": dict(zip(scens, a7.tolist())),
            "ar_folded_32b": dict(zip(scens, a32.tolist()))}


def main() -> None:
    bank = load_bank()
    fams = family_map(bank)
    # Repo-relative identities only (plan §6.3: no machine paths in
    # scientific artifacts).
    _repo = REPO.parent
    result = {"inputs": {
        "frozen_7b": str((P1 / "frozen_7b" / "results.jsonl").relative_to(_repo)),
        "frozen_32b": str((P1 / "frozen_32b" / "results.jsonl").relative_to(_repo)),
        "bank": str(BANK.relative_to(_repo)), "n_bank_rows": len(bank)}}

    folds = {}
    for model in ("7b", "32b"):
        rows = load_rows(model)
        val = validate_pipeline(model, rows)
        if not val["all_match"]:
            print(f"[FATAL] pipeline validation failed on {model}",
                  file=sys.stderr)
            for c in val["checks"]:
                if not c["ok"]:
                    print(" ", c, file=sys.stderr)
            sys.exit(1)
        grad = load_graduation(model)
        fold = folded_margins(rows)
        folds[model] = fold
        result[model] = {
            "pipeline_validation": {k: v for k, v in val.items()
                                    if k != "checks"},
            "f1_censoring_identity": censoring_identity(val["stats"], grad),
            "f2_folded_margins": fold,
            "f2_census": f2_census(fold, fams),
            "f4_retrodiction": retrodiction(rows, fams),
            "f5_ro_constant_code": ro_constant_code(rows, fams),
            "f5_concordance": concordance(rows, fams),
        }
    result["f3_aliasing"] = aliasing_census(bank)
    result["cross_structure"] = cross_structure(folds["7b"], folds["32b"],
                                                fams)

    out = OUT / "reanalysis.json"
    json.dump(result, open(out, "w"), indent=1, sort_keys=True)
    print(f"wrote {out}")

    # Compact console verdict
    for model in ("7b", "32b"):
        r = result[model]
        c = r["f2_census"]
        print(f"\n== {model} ==")
        print("pipeline exact-match:",
              r["pipeline_validation"]["n_exact_match"], "/",
              r["pipeline_validation"]["n_scenarios"])
        f1 = r["f1_censoring_identity"]
        print(f"F1 identity: {f1['n_exact']}/{f1['n_ar_nc']} exact, "
              f"max|dev|={f1['max_abs_dev']:.4f}")
        print("F2 AR enacted:", c["AR_enacted"])
        print("F2 AR report-only:", c["AR_report_only"])
        print("F2 NC:", c["NC_all_channels"],
              f"max|mean|={c['nc_abs_mean_max']:.4f}")
        ret = r["f4_retrodiction"]
        excl_raw = sum(v["raw_gate_would_exclude"] for v in ret.values())
        excl_res = sum(v["residual_gate_would_exclude"] for v in ret.values())
        print(f"F4: raw-gate excludes {excl_raw}/{len(ret)}; "
              f"residual-gate would exclude {excl_res}/{len(ret)}")
        print("F5 RO constant-code:",
              r["f5_ro_constant_code"]["n_constant_code"], "/",
              r["f5_ro_constant_code"]["n_scenarios"],
              "| concordance:", r["f5_concordance"])
    print("\nF3 aliasing:", result["f3_aliasing"])
    print("cross:", {k: v for k, v in result["cross_structure"].items()
                     if not k.startswith("ar_folded")})


if __name__ == "__main__":
    main()
