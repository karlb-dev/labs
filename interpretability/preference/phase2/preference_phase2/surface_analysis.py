"""B-SURF surface-policy decomposition + Phase 1 reconstruction
(plan §10, §24; addendum E3 — per-format design ranks).

Endpoints are properties of the EMITTED CODE on semantically null twin
menus: position (first-displayed record), label rank (lower-rank label),
pair member (pair code 0), reply-list position (listed-first code), twin
identity (twin_x's record). Balanced contrasts are primary; a linear-
probability regression is the declared full-rank sensitivity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .stats import bootstrap_ci, exact_sign_flip_p, hierarchical_bootstrap


def surf_frame(rows: list[dict[str, Any]],
               bank_by_id: dict[str, dict] | None = None) -> pd.DataFrame:
    rows = [r for r in rows if r.get("bank") == "B-SURF"]
    if bank_by_id is not None:
        merged = []
        for r in rows:
            b = bank_by_id.get(r["item_id"], {})
            m = dict(r)
            for k in ("display_label_set", "label_assignment",
                      "inline_code_assignment", "reply_list_order"):
                m.setdefault(k, b.get(k))
            merged.append(m)
        rows = merged
    df = pd.DataFrame(rows)
    df["valid"] = df["parse_status"] == "valid"
    sem = df["parsed_sem"].where(df["valid"], None)
    first_sem = np.where(df["display_order"] == 0, "a", "b")
    df["chose_first"] = np.where(df["valid"],
                                 (sem == first_sem).astype(float), np.nan)
    df["chose_twin_x"] = np.where(df["valid"], (sem == "a").astype(float),
                                  np.nan)
    # pair member 0 = the code at index 0 of the rotated pair: sem "a"
    # holds pair[cmap], so chose pair0 iff (sem==a) == (cmap==0)
    df["chose_pair0"] = np.where(
        df["valid"],
        ((sem == "a") == (df["code_map_index"] == 0)).astype(float), np.nan)
    # F-P1 only: label rank + reply-list position of the emitted code
    la = df.get("label_assignment")
    rl = df.get("reply_list_order")
    if la is not None:
        chose_first = df["chose_first"] == 1.0
        # label_assignment 0: first-displayed record has the lower-rank
        # label => chose lower-rank iff chose_first == (la == 0)
        df["chose_lowrank_label"] = np.where(
            df["valid"] & df["format_id"].eq("F-P1"),
            (chose_first == (df["label_assignment"] == 0)).astype(float),
            np.nan)
        df["chose_replylist_first"] = np.where(
            df["valid"] & df["format_id"].eq("F-P1"),
            (chose_first == (df["reply_list_order"] == 0)).astype(float),
            np.nan)
    return df


_ENDPOINTS = {
    "position_first": "chose_first",
    "twin_identity": "chose_twin_x",
    "inline_code_pair0": "chose_pair0",
    "label_rank_low": "chose_lowrank_label",
    "reply_list_first": "chose_replylist_first",
}


def surface_coefficients(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-format balanced contrasts, clustered by skin (incidental)."""
    out = []
    for fmt, d in df.groupby("format_id"):
        for name, col in _ENDPOINTS.items():
            if col not in d.columns:
                continue
            vals = d[col].dropna()
            if not len(vals):
                continue
            inc_means = d.groupby("incidental_id")[col].mean().dropna() - 0.5
            clusters = [
                (g[col].dropna() - 0.5).to_numpy()
                for _, g in d.groupby("incidental_id")]
            clusters = [c for c in clusters if len(c)]
            draws = hierarchical_bootstrap(
                clusters, seed_key=f"surf-{fmt}-{name}")
            lo, hi = bootstrap_ci(draws)
            by_labelfam = {}
            if "display_label_set" in d.columns:
                by_labelfam = {
                    str(v): float(np.nanmean(g[col]) - 0.5)
                    for v, g in d.groupby("display_label_set", dropna=False)
                    if g[col].notna().any()}
            out.append({
                "format_id": fmt, "endpoint": name,
                "effect": float(inc_means.mean()),
                "p_exact_signflip": exact_sign_flip_p(
                    inc_means.to_numpy(), seed_key=f"surfp-{fmt}-{name}"),
                "ci_lo": lo, "ci_hi": hi,
                "n_rows": int(vals.shape[0]),
                "n_skins": int(len(inc_means)),
                "by_label_family": by_labelfam,
            })
    return out


def surface_interactions(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Predeclared two-way interactions (F-P1): position x label family,
    position x reply-list order, position x inline-code."""
    d = df[df["format_id"] == "F-P1"]
    out = []
    pairs = (("display_label_set", "letters"),
             ("reply_list_order", 0), ("inline_code_assignment", 0))
    for factor, ref in pairs:
        if factor not in d.columns or not len(d):
            continue
        eff = {}
        for v, g in d.groupby(factor, dropna=False):
            eff[str(v)] = float(np.nanmean(g["chose_first"]) - 0.5)
        vals = list(eff.values())
        out.append({
            "interaction": f"position_x_{factor}",
            "levels": eff,
            "difference": float(vals[0] - vals[1]) if len(vals) == 2
            else float("nan"),
        })
    return out


def lpm_sensitivity(df: pd.DataFrame) -> dict[str, Any]:
    """Linear-probability full-rank regression of chose_first on the
    per-format factor set (declared sensitivity, not primary)."""
    out = {}
    for fmt, d in df.groupby("format_id"):
        d = d.dropna(subset=["chose_first"])
        if not len(d):
            continue
        cols = {"intercept": np.ones(len(d))}
        if fmt == "F-P1":
            cols["label_assignment"] = d["label_assignment"].to_numpy(float)
            cols["inline_code"] = d["inline_code_assignment"].to_numpy(float)
            cols["reply_list"] = d["reply_list_order"].to_numpy(float)
            cols["label_family_letters"] = (
                d["display_label_set"] == "letters").to_numpy(float)
        else:
            cols["code_map"] = d["code_map_index"].to_numpy(float)
        cols["twin_x_first"] = (d["display_order"] == 0).to_numpy(float)
        X = np.column_stack(list(cols.values()))
        y = d["chose_first"].to_numpy(float)
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        out[fmt] = {
            "coefficients": {k: float(b) for k, b in zip(cols, beta)},
            "design_rank": int(rank), "design_cols": len(cols),
            "full_rank": bool(rank == len(cols)),
            "n": int(len(d)),
        }
    return out


def phase1_reconstruction(df: pd.DataFrame,
                          phase1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Out-of-sample reconstruction of the frozen Phase 1 per-cell
    first-choice rates from the B-SURF F-P1 fit (plan §24).

    Phase 1 F-P1 cells aliased label rank and reply-list order to display
    position; the B-SURF prediction for such a cell is the mean of
    P(first-displayed) over F-P1 rows in the aligned configuration
    (label_assignment=0, reply_list_order=0), by label family."""
    d = df[(df["format_id"] == "F-P1")
           & (df["label_assignment"] == 0)
           & (df["reply_list_order"] == 0)]
    pred_by_fam = {
        str(v): float(np.nanmean(g["chose_first"]))
        for v, g in d.groupby("display_label_set", dropna=False)}
    p1 = pd.DataFrame([r for r in phase1_rows
                       if r.get("family") in ("AR", "NC")
                       and r.get("channel") == "AR"
                       and r.get("parse_status") == "valid"])
    if not len(p1):
        return {"error": "no phase1 rows"}
    p1["chose_first"] = (p1["parsed_pole"].astype(float)
                         == p1["order_index"].astype(float)).astype(float)
    obs = p1.groupby(["scenario_id", "display_label_set"])["chose_first"] \
            .mean().reset_index()
    obs["predicted"] = obs["display_label_set"].map(pred_by_fam)
    obs["abs_err"] = (obs["chose_first"] - obs["predicted"]).abs()
    nc_mask = obs["scenario_id"].str.startswith("nc_")
    return {
        "prediction_by_label_family": pred_by_fam,
        "n_phase1_cells": int(len(obs)),
        "mae_all": float(obs["abs_err"].mean()),
        "mae_nc_only": float(obs.loc[nc_mask, "abs_err"].mean())
        if nc_mask.any() else float("nan"),
        "note": ("content-indifferent Phase 1 cells (NC) are the clean "
                 "reconstruction targets; content-pulled AR cells deviate "
                 "by their semantic pull, which is the point of the "
                 "decomposition"),
        "cells": obs.to_dict("records"),
    }
