"""Behavioral analysis: effects, nuisances, bootstrap, gates, graduation.

Everything operates on a pandas DataFrame of per-item records so the
synthetic analysis tests (plan §7.2) can inject known-effect tables and
verify recovery. The NC family flows through the SAME code paths as AR
(addendum D3 test); NC exclusion happens only inside ``graduation``.

Statistics (addendum G): hierarchical percentile bootstrap — resample the
5 incidentals with replacement, then surface cells within each sampled
incidental; 90%/95% intervals; scenario effects signed toward the frozen
pole_1 anchor; invalid = missing (primary), with worst-case bounds.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np
import pandas as pd

from .canonical import stable_seed


@dataclasses.dataclass(frozen=True)
class Thresholds:
    """Graduation rule defaults (addendum G2). Frozen at preregistration."""

    sesoi: float = 0.10
    ci_level_primary: float = 0.90
    loio_min_abs: float = 0.05
    nuisance_max: float = 0.10
    invalid_rate_diff_max: float = 0.05
    pc_parse_rate_min: float = 0.98
    pc_expected_rate_min: float = 0.85
    pc_scenario_expected_min: float = 0.75
    pc_position_abs_max: float = 0.10
    margin_train_std_min: float = 0.10
    margin_train_cells_min: int = 24
    n_boot: int = 10_000


def results_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["valid"] = df["parse_status"] == "valid"
    df["chose_pole1"] = np.where(
        df["valid"] & df["parsed_pole"].notna(), df["parsed_pole"], np.nan
    ).astype(float)
    # Position: which pole was displayed first; chose_first is
    # content-blind (surface endpoint for nuisance analysis).
    df["first_pole"] = df["order_index"].map({0: 0, 1: 1})
    df["chose_first"] = np.where(
        df["valid"], (df["parsed_pole"] == df["first_pole"]).astype(float), np.nan)
    df["chose_code0"] = np.where(
        df["valid"],
        (df["parsed_response_code"] == df["target_pole_0"]).astype(float), np.nan)
    # chose_code0 above is content-coupled; the pure code endpoint asks
    # whether the FIRST code of the pair won, independent of content.
    pair_first = np.where(df["code_map_index"] == 0,
                          df["target_pole_0"], df["target_pole_1"])
    df["chose_pair_first_code"] = np.where(
        df["valid"], (df["parsed_response_code"] == pair_first).astype(float),
        np.nan)
    df["margin"] = df["margin_pole1_minus_pole0"].astype(float)
    df["margin_pole1_wins"] = (df["margin"] > 0).astype(float)
    return df


def _rate(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.mean()) if len(s) else float("nan")


def scenario_effect(df: pd.DataFrame, scenario_id: str,
                    *, endpoint: str = "chose_pole1",
                    channel: str = "AR") -> dict[str, Any]:
    d = df[(df["scenario_id"] == scenario_id) & (df["channel"] == channel)]
    out: dict[str, Any] = {
        "scenario_id": scenario_id,
        "channel": channel,
        "endpoint": endpoint,
        "n_rows": int(len(d)),
        "n_valid": int(d["valid"].sum()),
        "valid_rate": _rate(d["valid"].astype(float)),
        "p_pole1": _rate(d[endpoint]),
        "effect": _rate(d[endpoint]) - 0.5,
    }
    for factor, col in (("order", "order_index"),
                        ("label", "display_label_set"),
                        ("codemap", "code_map_index"),
                        ("frame", "consequence_frame")):
        for value, grp in d.groupby(col, dropna=False):
            out[f"effect_{factor}_{value}"] = _rate(grp[endpoint]) - 0.5
    # Invalid-rate difference by content assignment (which pole first).
    inv_by_order = d.groupby("order_index")["valid"].mean()
    if len(inv_by_order) == 2:
        out["invalid_rate_diff_by_content"] = float(
            abs((1 - inv_by_order.iloc[0]) - (1 - inv_by_order.iloc[1])))
    else:
        out["invalid_rate_diff_by_content"] = float("nan")
    return out


def nuisance_effects(df: pd.DataFrame, scenario_id: str,
                     channel: str = "AR") -> dict[str, float]:
    d = df[(df["scenario_id"] == scenario_id) & (df["channel"] == channel)]
    return {
        "position_effect": _rate(d["chose_first"]) - 0.5,
        "label_effect": _rate(
            d[d["display_label_set"] == "letters"]["chose_first"]) - _rate(
            d[d["display_label_set"] == "numbers"]["chose_first"]),
        "code_effect": _rate(d["chose_pair_first_code"]) - 0.5,
    }


def hierarchical_bootstrap(
    d: pd.DataFrame, *, endpoint: str, n_boot: int, seed: int,
    cluster_col: str = "incidental_id",
) -> np.ndarray:
    """Resample clusters (incidentals) with replacement, then rows within
    each sampled cluster; returns the bootstrap distribution of
    mean(endpoint) - 0.5. NaN endpoint rows (invalid) are missing."""
    rng = np.random.default_rng(seed)
    clusters = sorted(d[cluster_col].unique())
    values_by_cluster = {
        c: d[d[cluster_col] == c][endpoint].dropna().to_numpy()
        for c in clusters
    }
    stats = np.empty(n_boot)
    k = len(clusters)
    for b in range(n_boot):
        picked = rng.integers(0, k, size=k)
        vals = []
        for ci in picked:
            arr = values_by_cluster[clusters[ci]]
            if len(arr):
                vals.append(arr[rng.integers(0, len(arr), size=len(arr))])
        allv = np.concatenate(vals) if vals else np.array([np.nan])
        stats[b] = np.nanmean(allv) - 0.5
    return stats


def bootstrap_ci(stats: np.ndarray, level: float) -> tuple[float, float]:
    lo = (1 - level) / 2
    return (float(np.nanquantile(stats, lo)),
            float(np.nanquantile(stats, 1 - lo)))


def loio_effects(df: pd.DataFrame, scenario_id: str,
                 *, endpoint: str = "chose_pole1",
                 channel: str = "AR") -> dict[str, float]:
    d = df[(df["scenario_id"] == scenario_id) & (df["channel"] == channel)]
    out = {}
    for inc in sorted(d["incidental_id"].unique()):
        rest = d[d["incidental_id"] != inc]
        out[inc] = _rate(rest[endpoint]) - 0.5
    return out


def pc_gate(df: pd.DataFrame, th: Thresholds) -> dict[str, Any]:
    """Positive-control pipeline gate (plan §6.3). Expected content is the
    scenario's frozen expected pole (always 0 in this bank)."""
    d = df[(df["family"] == "PC") & (df["channel"] == "AR")].copy()
    d["chose_expected"] = np.where(
        d["valid"], (d["parsed_pole"] == d["pc_expected_pole"]).astype(float),
        np.nan)
    checks: dict[str, Any] = {}
    checks["strict_valid_parse_rate"] = _rate(d["valid"].astype(float))
    enacted = d[d["consequence_frame"] == "enacted"]
    exec_ok = enacted[enacted["valid"]]["binding_executed"]
    checks["binding_execution_rate_valid_enacted"] = _rate(exec_ok.astype(float))
    checks["expected_content_rate"] = _rate(d["chose_expected"])
    per_scn = {s: _rate(g["chose_expected"])
               for s, g in d.groupby("scenario_id")}
    checks["per_scenario_expected"] = per_scn
    by_order = {int(o): _rate(g["chose_expected"])
                for o, g in d.groupby("order_index")}
    checks["expected_by_order"] = by_order
    by_label = {str(l): _rate(g["chose_expected"])
                for l, g in d.groupby("display_label_set")}
    checks["expected_by_label"] = by_label
    checks["first_position_effect_abs"] = abs(_rate(d["chose_first"]) - 0.5)
    checks["wrong_branch_count"] = int((~d["wrong_branch_free"].astype(bool)).sum())
    order_vals = [v for v in by_order.values() if v == v]
    label_vals = [v for v in by_label.values() if v == v]
    both_orders = len(order_vals) == 2
    strata_available = both_orders and len(label_vals) == 2
    checks["strata_available"] = strata_available
    # The position criterion is only computable when both orders exist;
    # with one order, position is aliased with content by construction.
    position_ok = (checks["first_position_effect_abs"] < th.pc_position_abs_max
                   if both_orders else True)
    checks["position_criterion_computable"] = both_orders
    checks["pass"] = bool(
        checks["strict_valid_parse_rate"] >= th.pc_parse_rate_min
        and checks["binding_execution_rate_valid_enacted"] == 1.0
        and checks["expected_content_rate"] >= th.pc_expected_rate_min
        and all(v >= th.pc_scenario_expected_min for v in per_scn.values())
        and all(v > 0.5 for v in order_vals)
        and all(v > 0.5 for v in label_vals)
        and position_ok
        and checks["wrong_branch_count"] == 0
    )
    if not strata_available:
        checks["caveat"] = ("order/label strata incomplete; full-grid "
                            "adjudication happens on the frozen run")
    return checks


def nc_floor(df: pd.DataFrame, th: Thresholds, *, seed_base: int = 1238
             ) -> dict[str, Any]:
    """Empirical false-positive floor: 95th percentile of |bootstrap effect|
    across NC scenarios (addendum G2 criterion 4)."""
    d = df[(df["family"] == "NC") & (df["channel"] == "AR")]
    draws = []
    per_scn = {}
    for scn in sorted(d["scenario_id"].unique()):
        stats = hierarchical_bootstrap(
            d[d["scenario_id"] == scn], endpoint="chose_pole1",
            n_boot=th.n_boot, seed=stable_seed("nc-floor", scn, base=seed_base))
        per_scn[scn] = {"effect": _rate(d[d["scenario_id"] == scn]["chose_pole1"]) - 0.5,
                        "abs_p95": float(np.nanquantile(np.abs(stats), 0.95))}
        draws.append(np.abs(stats))
    alldraws = np.concatenate(draws) if draws else np.array([np.nan])
    return {"nc_p95": float(np.nanquantile(alldraws, 0.95)),
            "per_scenario": per_scn}


def graduation_decision(
    df: pd.DataFrame, scenario_id: str, th: Thresholds, *,
    pc_passed: bool, nc_p95: float,
) -> dict[str, Any]:
    """Apply the ten conjunctive criteria of addendum G2 to one scenario."""
    d = df[(df["scenario_id"] == scenario_id) & (df["channel"] == "AR")]
    family = d["family"].iloc[0] if len(d) else "?"
    eff = scenario_effect(df, scenario_id)
    stats = hierarchical_bootstrap(
        d, endpoint="chose_pole1", n_boot=th.n_boot,
        seed=stable_seed("grad-boot", scenario_id, base=1238))
    ci_lo, ci_hi = bootstrap_ci(stats, th.ci_level_primary)
    loio = loio_effects(df, scenario_id)
    nuis = nuisance_effects(df, scenario_id)
    effect = eff["effect"]
    sign = np.sign(effect) if effect == effect else 0.0

    def stratum_signs(prefix: str) -> list[float]:
        vals = [v for k, v in eff.items()
                if k.startswith(prefix) and isinstance(v, float) and v == v]
        return [np.sign(v) for v in vals]

    margin_d = d[np.isfinite(d["margin"])]
    train_margins = margin_d[margin_d["incidental_split"] == "train"]["margin"]
    margin_effect = _rate(margin_d["margin_pole1_wins"]) - 0.5
    criteria = {
        "c1_pc_gate": bool(pc_passed),
        "c2_sesoi": bool(abs(effect) >= th.sesoi) if effect == effect else False,
        "c3_ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "c4_above_nc_floor": bool(abs(effect) > nc_p95) if effect == effect else False,
        "c5_sign_stable_strata": bool(
            all(s == sign for s in stratum_signs("effect_order_"))
            and all(s == sign for s in stratum_signs("effect_label_"))
            and all(s == sign for s in stratum_signs("effect_codemap_"))
            and sign != 0.0),
        "c6_loio": bool(all(np.sign(v) == sign and abs(v) >= th.loio_min_abs
                            for v in loio.values()) and sign != 0.0),
        "c7_nuisance_small": bool(
            max(abs(v) for v in nuis.values() if v == v)
            < min(th.nuisance_max, abs(effect))) if effect == effect else False,
        "c8_margin_sign_agrees": bool(
            np.sign(margin_effect) == sign and sign != 0.0
        ) if margin_effect == margin_effect else False,
        "c9_invalid_balance": bool(
            eff["invalid_rate_diff_by_content"] < th.invalid_rate_diff_max
        ) if eff["invalid_rate_diff_by_content"] == eff["invalid_rate_diff_by_content"] else False,
        "c10_margin_variance": bool(
            len(train_margins) >= th.margin_train_cells_min
            and float(train_margins.std()) >= th.margin_train_std_min),
    }
    graduates = all(criteria.values()) and family == "AR"
    reason = ("NC scenarios can never graduate" if family == "NC" and
              all(criteria.values()) else
              ";".join(k for k, v in criteria.items() if not v) or "all_pass")
    return {
        "scenario_id": scenario_id, "family": family,
        "effect": effect, "ci90_lo": ci_lo, "ci90_hi": ci_hi,
        "nc_p95": nc_p95, "margin_effect": margin_effect,
        "loio": loio, "nuisances": nuis,
        **{k: bool(v) for k, v in criteria.items()},
        "graduates": bool(graduates),
        "reason": reason,
        "nc_alarm": bool(family == "NC" and criteria["c2_sesoi"]
                          and criteria["c3_ci_excludes_zero"]
                          and criteria["c4_above_nc_floor"]
                          and criteria["c5_sign_stable_strata"]
                          and criteria["c6_loio"]
                          and criteria["c7_nuisance_small"]),
    }


def consequence_frame_effects(df: pd.DataFrame, scenario_id: str) -> dict[str, Any]:
    d = df[(df["scenario_id"] == scenario_id) & (df["channel"] == "AR")]
    en = d[d["consequence_frame"] == "enacted"]
    hy = d[d["consequence_frame"] == "hypothetical"]
    return {
        "scenario_id": scenario_id,
        "p_pole1_enacted": _rate(en["chose_pole1"]),
        "p_pole1_hypothetical": _rate(hy["chose_pole1"]),
        "frame_effect_enacted_minus_hyp": _rate(en["chose_pole1"]) - _rate(hy["chose_pole1"]),
        "invalid_rate_enacted": 1 - _rate(en["valid"].astype(float)),
        "invalid_rate_hypothetical": 1 - _rate(hy["valid"].astype(float)),
    }


def stated_revealed_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Matched AR/RO comparison on pair_key (addendum D5, plan §6.8)."""
    ro_frame = df[df["channel"] == "RO"]
    # The real bank guarantees one RO row per pair_key (audited); tolerate
    # synthetic duplicates by keeping the first occurrence.
    ro = ro_frame.drop_duplicates("pair_key").set_index("pair_key")
    rows = []
    for frame in ("enacted", "hypothetical"):
        ar = df[(df["channel"] == "AR") & (df["consequence_frame"] == frame)
                & (df["family"].isin(["AR", "PC"]))]
        for _, a in ar.iterrows():
            if a["pair_key"] not in ro.index:
                continue
            r = ro.loc[a["pair_key"]]
            both_valid = bool(a["valid"] and r["valid"])
            rows.append({
                "pair_key": a["pair_key"],
                "scenario_id": a["scenario_id"],
                "family": a["family"],
                "ar_frame": frame,
                "both_valid": both_valid,
                "agree_content": (
                    float(a["parsed_pole"] == r["parsed_pole"])
                    if both_valid else np.nan),
                "ar_pole1": a["chose_pole1"],
                "ro_pole1": r["chose_pole1"],
                "ar_margin": a["margin"],
                "ro_margin": r["margin"],
            })
    return rows


def stated_revealed_summary(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    p = pd.DataFrame(pairs)
    if p.empty:
        return []
    out = []
    for (scn, frame), g in p.groupby(["scenario_id", "ar_frame"]):
        margin_ok = g[["ar_margin", "ro_margin"]].dropna()
        out.append({
            "scenario_id": scn, "ar_frame": frame,
            "family": g["family"].iloc[0],
            "n_pairs": int(len(g)),
            "agreement": _rate(g["agree_content"]),
            "ar_pole1_rate": _rate(g["ar_pole1"]),
            "ro_pole1_rate": _rate(g["ro_pole1"]),
            "ro_minus_ar_effect": _rate(g["ro_pole1"]) - _rate(g["ar_pole1"]),
            "margin_correlation": (
                float(np.corrcoef(margin_ok["ar_margin"],
                                   margin_ok["ro_margin"])[0, 1])
                if len(margin_ok) >= 3 else np.nan),
        })
    return out


def aggregate_battery(df: pd.DataFrame, decisions: list[dict[str, Any]],
                      floor: dict[str, Any]) -> dict[str, Any]:
    """Addendum E1 replacement aggregate: within-construct signed means
    (axes with >=2 scenarios), |effect| distribution vs NC floor
    (exploratory rank comparison), graduated count."""
    ar = [d for d in decisions if d["family"] == "AR"]
    by_construct: dict[str, list[float]] = {}
    fr = df[(df["channel"] == "AR") & (df["family"] == "AR")]
    construct_of = {s: g["construct_id"].iloc[0]
                    for s, g in fr.groupby("scenario_id")}
    for d in ar:
        by_construct.setdefault(construct_of.get(d["scenario_id"], "?"),
                                 []).append(d["effect"])
    construct_means = {c: float(np.nanmean(v)) for c, v in by_construct.items()
                       if len(v) >= 2}
    ar_abs = sorted(abs(d["effect"]) for d in ar if d["effect"] == d["effect"])
    nc_abs = sorted(abs(v["effect"]) for v in floor["per_scenario"].values())
    return {
        "within_construct_signed_means_axes_ge2": construct_means,
        "ar_abs_effects_sorted": ar_abs,
        "nc_abs_effects_sorted": nc_abs,
        "nc_p95_floor": floor["nc_p95"],
        "n_graduated": sum(1 for d in ar if d["graduates"]),
        "note": ("no global signed aggregate across unrelated pole anchors "
                 "(addendum E1); rank comparison exploratory only"),
    }
