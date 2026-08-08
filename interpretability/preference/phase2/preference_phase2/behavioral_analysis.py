"""Behavioral analysis contract (plan §23-§31; addendum E pins).

Everything operates on a pandas DataFrame of per-item result records
(synthetic tables injectable — the §54.3 worlds run through these exact
code paths). Analysis works in authored semantic A/B space; the E15 sign
anchor is applied only at the reporting layer.

Endpoints per row: ``margin_full`` (full-target semantic margin, A-B),
``margin_first`` (first-token), ``chose_a`` (strict parsed semantic id;
NaN when invalid — invalid = missing is the primary convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .stats import (N_BOOT, bootstrap_ci, exact_sign_flip_p,
                    hierarchical_bootstrap, holm, incidental_ols_slope)


@dataclass(frozen=True)
class Thresholds:
    """Frozen at preregistration (plan §25-§27 + addendum E)."""

    margin_floor_nats: float = 0.15         # max(this, 2 x NC f1-f2 p95)
    strict_sesoi: float = 0.10
    ci_level: float = 0.95
    parse_rate_min: float = 0.98
    pc_aggregate_min: float = 0.90
    pc_scenario_min: float = 0.80
    slope_holdout_rank_corr_min: float = 0.70
    n_boot: int = N_BOOT


def results_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["valid"] = df["parse_status"] == "valid"
    sem = df["parsed_sem"].where(df["valid"], None)
    df["chose_a"] = np.where(sem == "a", 1.0,
                             np.where(sem == "b", 0.0, np.nan))
    first_sem = np.where(df["display_order"] == 0, "a", "b")
    df["chose_first"] = np.where(
        df["valid"], (sem == first_sem).astype(float), np.nan)
    df["margin_full"] = pd.to_numeric(df["margin_full_a_minus_b"],
                                      errors="coerce")
    df["margin_first"] = pd.to_numeric(df["margin_first_a_minus_b"],
                                       errors="coerce")
    return df


def _incidental_means(d: pd.DataFrame, col: str) -> pd.Series:
    return d.groupby("incidental_id")[col].mean()


def _strata_effects(d: pd.DataFrame, col: str,
                    factors: tuple[str, ...]) -> dict[str, float]:
    out = {}
    for f in factors:
        if f not in d.columns:
            continue
        for val, grp in d.groupby(f, dropna=False):
            out[f"{f}={val}"] = float(np.nanmean(grp[col]))
    return out


_PRIMARY_STRATA = ("codebook_pair_id", "paraphrase_id", "consequence_frame",
                   "display_order")


def scenario_endpoint(df: pd.DataFrame, scenario_id: str, *,
                      endpoint: str, channel: str = "AR",
                      center: float = 0.0,
                      exclude_reserved: bool = True) -> dict[str, Any]:
    """Design-based folded scenario estimate (plan §25 steps 1-5).

    ``endpoint``: margin_full | margin_first | chose_a. ``center`` is
    subtracted (0.5 for choice rates). Reserved-codebook rows are excluded
    from primary estimates (they exist for mechanism transfer)."""
    d = df[(df["scenario_id"] == scenario_id) & (df["channel"] == channel)]
    if exclude_reserved and "codebook_reserved" in d.columns:
        d = d[~d["codebook_reserved"].astype(bool)]
    if "context_strength" in d.columns:
        d = d[d["context_strength"].fillna(0) == 0]
    inc = _incidental_means(d, endpoint) - center
    values = inc.to_numpy(dtype=float)
    clusters = [
        (d[d["incidental_id"] == i][endpoint] - center).dropna().to_numpy()
        for i in inc.index]
    clusters = [c for c in clusters if len(c)]
    draws = (hierarchical_bootstrap(clusters,
                                    seed_key=f"scn-{scenario_id}-{endpoint}")
             if clusters else np.array([np.nan]))
    lo, hi = bootstrap_ci(draws)
    loio = {}
    for i in inc.index:
        rest = inc.drop(i)
        loio[str(i)] = float(rest.mean())
    n_valid = int(d["valid"].sum()) if "valid" in d.columns else len(d)
    n_invalid = int((~d["valid"]).sum()) if "valid" in d.columns else 0
    est = float(np.nanmean(values))
    # worst-case invalid bounds (choice endpoints only): every invalid row
    # assigned to A, then to B
    wc_lo, wc_hi = est, est
    if endpoint == "chose_a" and len(d):
        s = np.nansum(d["chose_a"].to_numpy(dtype=float))
        n = len(d)
        wc_lo = float(s / n) - 0.5
        wc_hi = float((s + n_invalid) / n) - 0.5
    return {
        "scenario_id": scenario_id, "channel": channel,
        "endpoint": endpoint, "estimate": est,
        "p_exact_signflip": exact_sign_flip_p(
            values, seed_key=f"sf-{scenario_id}-{endpoint}"),
        "ci_lo": lo, "ci_hi": hi,
        "n_rows": int(len(d)), "n_valid": n_valid,
        "n_incidentals": int(len(inc)),
        "valid_rate": (n_valid / len(d)) if len(d) else float("nan"),
        "incidental_means": {str(k): float(v) for k, v in inc.items()},
        "loio": loio,
        "strata": _strata_effects(d, endpoint, _PRIMARY_STRATA),
        "worst_case_lo": wc_lo, "worst_case_hi": wc_hi,
        "invalid_rate_diff_by_order": _invalid_rate_diff(d),
    }


def _invalid_rate_diff(d: pd.DataFrame) -> float:
    if "valid" not in d.columns or not len(d):
        return float("nan")
    rates = d.groupby("display_order")["valid"].mean()
    if len(rates) < 2:
        return float("nan")
    return float(abs((1 - rates).max() - (1 - rates).min()))


def _sign_stable(effects: dict[str, float], sign: float,
                 *, max_reversals: int = 0) -> bool:
    if sign == 0:
        return False
    reversals = sum(1 for v in effects.values()
                    if np.isfinite(v) and np.sign(v) == -np.sign(sign)
                    and abs(v) > 1e-12)
    return reversals <= max_reversals


def nc_floors(df: pd.DataFrame, th: Thresholds) -> dict[str, Any]:
    """Model-specific floors (plan §19; addendum E10 family mapping)."""
    out: dict[str, Any] = {}

    def p95_abs(scenarios: list[str], endpoint: str, center: float) -> float:
        draws = []
        for scn in scenarios:
            d = df[(df["scenario_id"] == scn) & (df["channel"] == "AR")]
            if "context_strength" in d.columns:
                d = d[d["context_strength"].fillna(0) == 0]
            if not len(d):
                continue
            clusters = [
                (d[d["incidental_id"] == i][endpoint] - center)
                .dropna().to_numpy()
                for i in d["incidental_id"].unique()]
            clusters = [c for c in clusters if len(c)]
            if clusters:
                draws.append(hierarchical_bootstrap(
                    clusters, seed_key=f"ncfloor-{scn}-{endpoint}"))
        if not draws:
            return float("nan")
        return float(np.nanquantile(np.abs(np.concatenate(draws)), 0.95))

    f12 = [s for s in df.loc[df["nc_family"].isin(
        ["nc_identical", "nc_paraphrase"]), "scenario_id"].unique()]
    f3 = [s for s in df.loc[df["nc_family"] == "nc_code_only",
                            "scenario_id"].unique()]
    out["margin_floor_p95"] = p95_abs(f12, "margin_full", 0.0)
    out["strict_floor_p95"] = p95_abs(f12, "chose_a", 0.5)
    out["carrier_floor_p95"] = p95_abs(f3, "margin_full", 0.0)
    out["margin_floor_effective"] = float(
        max(th.margin_floor_nats,
            2 * out["margin_floor_p95"]
            if np.isfinite(out["margin_floor_p95"]) else 0.0))
    # context-slope floor from the family-4 null ladder
    f4 = df[(df["nc_family"] == "nc_context_null") & (df["channel"] == "AR")]
    if len(f4):
        slopes = [
            incidental_ols_slope(g["context_strength"].to_numpy(),
                                 g["margin_full"].to_numpy())
            for _, g in f4.groupby("incidental_id")]
        slopes = np.array([s for s in slopes if np.isfinite(s)])
        out["null_slope_mean"] = float(np.mean(slopes)) if len(slopes) else float("nan")
        out["null_slope_p95_abs"] = (float(np.quantile(np.abs(slopes), 0.95))
                                     if len(slopes) else float("nan"))
        out["slope_floor_effective"] = float(2 * out["null_slope_p95_abs"]) \
            if np.isfinite(out.get("null_slope_p95_abs", np.nan)) else float("nan")
        out["null_slope_p"] = exact_sign_flip_p(slopes, seed_key="nullslope")
    return out


def semantic_margin_decision(df: pd.DataFrame, scenario_id: str,
                             th: Thresholds, *, pc_passed: bool,
                             floors: dict[str, Any],
                             p_holm: float | None = None) -> dict[str, Any]:
    """The ten SEMANTIC_MARGIN criteria (plan §25)."""
    full = scenario_endpoint(df, scenario_id, endpoint="margin_full")
    first = scenario_endpoint(df, scenario_id, endpoint="margin_first")
    est, sign = full["estimate"], np.sign(full["estimate"])
    floor = floors.get("margin_floor_effective", th.margin_floor_nats)
    strata = full["strata"]
    crit = {
        "c1_pc_nc_gates": bool(pc_passed),
        "c2_above_floor": bool(abs(est) > floor),
        "c3_ci_excludes_zero": bool(full["ci_lo"] * full["ci_hi"] > 0),
        "c4_holm_significant": bool((p_holm if p_holm is not None
                                     else full["p_exact_signflip"]) < 0.05),
        "c5_sign_all_codebooks": _sign_stable(
            {k: v for k, v in strata.items()
             if k.startswith("codebook_pair_id=")}, sign),
        "c6_sign_both_paraphrases": _sign_stable(
            {k: v for k, v in strata.items()
             if k.startswith("paraphrase_id=")}, sign),
        "c7_loio_sign_stable": _sign_stable(full["loio"], sign),
        "c8_no_reversing_interaction": _interaction_check(
            df, scenario_id, "margin_full", sign),
        "c9_worst_case_ok": True,   # margins have no invalid censoring
        "c10_first_token_reported": bool(np.isfinite(first["estimate"])),
    }
    return {
        "scenario_id": scenario_id, "full": full, "first": first,
        "criteria": crit, "floor": floor,
        "passes": all(crit.values()),
    }


def _interaction_check(df: pd.DataFrame, scenario_id: str, endpoint: str,
                       sign: float) -> bool:
    """No predeclared semantic-by-surface interaction reverses sign in
    more than one stratum (plan §25 criterion 8): within each level of
    each primary surface factor, the folded effect keeps the pooled sign
    (at most one reversal across all strata of a factor)."""
    d = df[(df["scenario_id"] == scenario_id) & (df["channel"] == "AR")]
    if "context_strength" in d.columns:
        d = d[d["context_strength"].fillna(0) == 0]
    if sign == 0 or not len(d):
        return False
    for f in _PRIMARY_STRATA:
        if f not in d.columns:
            continue
        effs = {str(v): float(np.nanmean(g[endpoint]))
                for v, g in d.groupby(f, dropna=False)}
        if not _sign_stable(effs, sign, max_reversals=1):
            return False
    return True


def enacted_choice_decision(df: pd.DataFrame, scenario_id: str,
                            th: Thresholds, *, margin_passes: bool,
                            margin_sign: float,
                            p_holm: float | None = None) -> dict[str, Any]:
    """The ten ENACTED_CHOICE criteria (plan §26)."""
    strict = scenario_endpoint(df, scenario_id, endpoint="chose_a",
                               center=0.5)
    est, sign = strict["estimate"], np.sign(strict["estimate"])
    d = df[(df["scenario_id"] == scenario_id) & (df["channel"] == "AR")]
    wrong = int(d.get("wrong_branch_free", pd.Series(dtype=bool))
                .eq(False).sum()) if len(d) else 0
    crit = {
        "c1_semantic_margin_passes": bool(margin_passes),
        "c2_sesoi": bool(abs(est) >= th.strict_sesoi),
        "c3_ci_excludes_zero": bool(strict["ci_lo"] * strict["ci_hi"] > 0),
        "c4_holm_significant": bool((p_holm if p_holm is not None
                                     else strict["p_exact_signflip"]) < 0.05),
        "c5_same_sign_as_margin": bool(sign == margin_sign and sign != 0),
        "c6_sign_stable_strata": _sign_stable(strict["strata"], sign,
                                              max_reversals=1),
        "c7_no_reversing_position_interaction": _interaction_check(
            df, scenario_id, "chose_a", sign),
        "c8_parse_rate": bool(strict["valid_rate"] >= th.parse_rate_min),
        "c9_wrong_branches_zero": bool(wrong == 0),
        "c10_worst_case_ok": bool(
            strict["worst_case_lo"] * strict["worst_case_hi"] > 0
            or abs(est) < th.strict_sesoi),
    }
    return {"scenario_id": scenario_id, "strict": strict, "criteria": crit,
            "passes": all(crit.values())}


def context_ladder_decision(df: pd.DataFrame, scenario_id: str,
                            th: Thresholds, *, floors: dict[str, Any],
                            p_holm: float | None = None) -> dict[str, Any]:
    """CONTEXTUAL_VALUE criteria (plan §27)."""
    d = df[(df["scenario_id"] == scenario_id) & (df["channel"] == "AR")]
    if "codebook_reserved" in d.columns:
        d = d[~d["codebook_reserved"].astype(bool)]
    slopes, orient = {}, {}
    for i, g in d.groupby("incidental_id"):
        slopes[str(i)] = incidental_ols_slope(
            g["context_strength"].to_numpy(), g["margin_full"].to_numpy())
        pos = g[g["context_strength"] >= 0]
        neg = g[g["context_strength"] <= 0]
        orient[str(i)] = (
            incidental_ols_slope(pos["context_strength"].to_numpy(),
                                 pos["margin_full"].to_numpy()),
            incidental_ols_slope(neg["context_strength"].to_numpy(),
                                 neg["margin_full"].to_numpy()))
    vals = np.array([v for v in slopes.values() if np.isfinite(v)])
    slope = float(np.mean(vals)) if len(vals) else float("nan")
    sign = np.sign(slope)
    p = exact_sign_flip_p(vals, seed_key=f"slope-{scenario_id}")
    clusters = [np.array([v]) for v in vals]
    draws = hierarchical_bootstrap(clusters, seed_key=f"slopeci-{scenario_id}")
    lo, hi = bootstrap_ci(draws)
    # holdout monotonicity: mean margin per strength rank on holdout incs
    hold = d[d["incidental_split"] == "holdout"]
    mono = float("nan")
    if len(hold):
        by_s = hold.groupby("context_strength")["margin_full"].mean()
        if len(by_s) >= 3:
            from scipy.stats import spearmanr
            mono = float(spearmanr(by_s.index.to_numpy(),
                                   by_s.to_numpy()).statistic)
    pos_slopes = np.array([o[0] for o in orient.values()
                           if np.isfinite(o[0])])
    neg_slopes = np.array([o[1] for o in orient.values()
                           if np.isfinite(o[1])])
    both_orient = (len(pos_slopes) > 0 and len(neg_slopes) > 0
                   and np.sign(np.mean(pos_slopes)) == sign
                   and np.sign(np.mean(neg_slopes)) == sign)
    # strata agreement (codebook, menu paraphrase)
    strata_ok = True
    for f in ("codebook_pair_id", "paraphrase_id"):
        if f not in d.columns:
            continue
        for v, g in d.groupby(f):
            ss = [incidental_ols_slope(gg["context_strength"].to_numpy(),
                                       gg["margin_full"].to_numpy())
                  for _, gg in g.groupby("incidental_id")]
            ss = [s for s in ss if np.isfinite(s)]
            if ss and np.sign(np.mean(ss)) != sign:
                strata_ok = False
    loio_ok = _sign_stable(
        {i: float(np.mean([v for j, v in slopes.items() if j != i and
                           np.isfinite(v)])) for i in slopes}, sign)
    floor = floors.get("slope_floor_effective", float("nan"))
    crit = {
        "c1_expected_sign": bool(sign > 0),   # ladder authored +: favors A
        "c2_ci_excludes_zero": bool(lo * hi > 0),
        "c3_holm_significant": bool((p_holm if p_holm is not None else p)
                                    < 0.05),
        "c4_holdout_monotonic": bool(np.isfinite(mono)
                                     and mono >= th.slope_holdout_rank_corr_min),
        "c5_both_orientations": bool(both_orient),
        "c6_strata_agree": bool(strata_ok),
        "c7_above_null_floor": bool(not np.isfinite(floor)
                                    or abs(slope) > floor),
        "c8_no_single_incidental": bool(loio_ok),
    }
    # neutral intercept + strict-choice crossing estimate
    neutral = d[d["context_strength"] == 0]
    intercept = float(np.nanmean(neutral["margin_full"])) if len(neutral) else float("nan")
    strict_by_s = d.groupby("context_strength")["chose_a"].mean()
    crossing = _crossing_estimate(strict_by_s)
    return {
        "scenario_id": scenario_id, "slope": slope, "p": p,
        "ci_lo": lo, "ci_hi": hi, "holdout_rank_corr": mono,
        "neutral_intercept": intercept,
        "strict_by_strength": {str(k): float(v)
                               for k, v in strict_by_s.items()},
        "estimated_choice_crossing": crossing,
        "incidental_slopes": slopes, "criteria": crit,
        "passes": all(crit.values()),
    }


def _crossing_estimate(strict_by_s: pd.Series) -> float:
    """First strength at which the strict semantic-choice rate crosses
    0.5 (linear interpolation; instrument characteristic, plan §27)."""
    s = strict_by_s.dropna()
    if len(s) < 2:
        return float("nan")
    xs, ys = s.index.to_numpy(dtype=float), s.to_numpy(dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    for i in range(len(xs) - 1):
        y0, y1 = ys[i] - 0.5, ys[i + 1] - 0.5
        if y0 == 0:
            return float(xs[i])
        if y0 * y1 < 0:
            return float(xs[i] - y0 * (xs[i + 1] - xs[i]) / (y1 - y0))
    return float("nan")


def pc_gate(df: pd.DataFrame, th: Thresholds) -> dict[str, Any]:
    d = df[(df["family"] == "PC") & (df["channel"] == "AR")]
    if not len(d):
        return {"pass": False, "reason": "no PC rows"}
    valid = d[d["valid"]]
    chose_expected = (valid["parsed_sem"] == "a")
    per_scn = {str(s): float((g["parsed_sem"] == "a").mean())
               for s, g in valid.groupby("scenario_id")}
    wrong = int(d.get("wrong_branch_free", pd.Series(dtype=bool))
                .eq(False).sum())
    parse_rate = float(d["valid"].mean())
    agg = float(chose_expected.mean()) if len(valid) else float("nan")
    ok = (parse_rate >= th.parse_rate_min
          and agg >= th.pc_aggregate_min
          and all(v >= th.pc_scenario_min for v in per_scn.values())
          and wrong == 0)
    return {"pass": bool(ok), "parse_rate": parse_rate,
            "expected_rate": agg, "per_scenario": per_scn,
            "wrong_branch_count": wrong}


def nc_alarm(df: pd.DataFrame, th: Thresholds,
             floors: dict[str, Any]) -> dict[str, Any]:
    """Any NC family clearing a semantic or slope floor is a stop-and-ask
    (addendum L item 4).

    An alarming NC scenario must not soften its own trigger, so each NC
    is compared against the STATIC floor components (0.15 nats semantic;
    0.05 nats/unit slope), never against the empirical p95 that its own
    rows would inflate."""
    alarms = []
    for scn in df.loc[df["family"] == "NC", "scenario_id"].unique():
        e = scenario_endpoint(df, str(scn), endpoint="margin_full")
        if (abs(e["estimate"]) > th.margin_floor_nats
                and e["p_exact_signflip"] < 0.05):
            alarms.append({"scenario_id": str(scn),
                           "estimate": e["estimate"],
                           "p": e["p_exact_signflip"]})
    if np.isfinite(floors.get("null_slope_p", np.nan)):
        if (floors["null_slope_p"] < 0.05
                and abs(floors.get("null_slope_mean", 0.0)) > 0.05):
            alarms.append({"scenario_id": "nc_ctxnull",
                           "slope": floors.get("null_slope_mean")})
    return {"alarm": bool(alarms), "details": alarms}


def analyze_behavioral(rows: list[dict[str, Any]],
                       th: Thresholds | None = None) -> dict[str, Any]:
    """Full behavioral adjudication for one model run (F1/F2/F3 families,
    Holm within each; plan §31)."""
    th = th or Thresholds()
    df = results_frame(rows)
    floors = nc_floors(df, th)
    pc = pc_gate(df, th)
    alarm = nc_alarm(df, th, floors)

    arb = sorted(df.loc[df["family"] == "ARB", "scenario_id"].unique())
    f1_p = {s: scenario_endpoint(df, s, endpoint="margin_full")
            ["p_exact_signflip"] for s in arb}
    f1_holm = holm(f1_p)
    margins = {
        s: semantic_margin_decision(
            df, s, th, pc_passed=pc["pass"], floors=floors,
            p_holm=f1_holm[s]["p_holm"]) for s in arb}

    f2_p = {s: scenario_endpoint(df, s, endpoint="chose_a", center=0.5)
            ["p_exact_signflip"] for s in arb if margins[s]["passes"]}
    f2_holm = holm(f2_p) if f2_p else {}
    choices = {}
    for s in arb:
        choices[s] = enacted_choice_decision(
            df, s, th, margin_passes=margins[s]["passes"],
            margin_sign=np.sign(margins[s]["full"]["estimate"]),
            p_holm=(f2_holm[s]["p_holm"] if s in f2_holm else None))

    mech = sorted(df.loc[df["family"] == "MECH", "scenario_id"].unique())
    f3_p = {}
    ladders = {}
    for s in mech:
        ladders[s] = context_ladder_decision(df, s, th, floors=floors)
        f3_p[s] = ladders[s]["p"]
    f3_holm = holm(f3_p) if f3_p else {}
    for s in mech:
        ladders[s]["criteria"]["c3_holm_significant"] = bool(
            f3_holm[s]["p_holm"] < 0.05) if s in f3_holm else False
        ladders[s]["passes"] = all(ladders[s]["criteria"].values())

    statuses = {}
    for s in arb:
        if not pc["pass"]:
            statuses[s] = "INSTRUMENT_FAILURE"
        elif choices[s]["passes"]:
            statuses[s] = "ENACTED_CHOICE"
        elif margins[s]["passes"]:
            statuses[s] = "SEMANTIC_MARGIN"
        else:
            statuses[s] = "CLEAN_NULL"
    for s in mech:
        statuses[s] = ("CONTEXTUAL_VALUE" if ladders[s]["passes"]
                       else ("INSTRUMENT_FAILURE" if not pc["pass"]
                             else "CLEAN_NULL"))

    return {
        "thresholds": th.__dict__, "pc_gate": pc, "nc_floors": floors,
        "nc_alarm": alarm,
        "f1_holm": f1_holm, "f2_holm": f2_holm, "f3_holm": f3_holm,
        "semantic_margins": margins, "enacted_choices": choices,
        "context_ladders": ladders, "statuses": statuses,
    }
