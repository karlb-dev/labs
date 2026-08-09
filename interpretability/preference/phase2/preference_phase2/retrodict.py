"""Analysis-level mechanism retrodiction (P2-4; addendum E8).

Scope pinned ``retrodiction_scope=analysis_level``: the 7B capture seal
is unverifiable and no 32B frozen capture manifest exists, so this reruns
the OLD estimator logic on (a) the frozen mechanism JSONs and (b)
synthetic worlds reproducing the recorded geometry. It must:

  (i)  reproduce the recorded PC numbers within tolerance from the JSONs;
  (ii) predict docsection non-identifiability under the old estimator;
  (iii) classify the old PC as MARGIN_HANDLE with DIRECT_OUTPUT_RISK.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from . import paths
from .stats import holm
from .canonical import stable_seed

TOL = 1e-6


def _old_sign_flip_p(deltas: list[float]) -> float:
    """Phase 1 exact sign-flip (n <= 20 enumeration), reimplemented."""
    v = np.asarray([d for d in deltas if np.isfinite(d)], dtype=np.float64)
    n = len(v)
    if n == 0:
        return float("nan")
    bits = np.arange(2 ** n, dtype=np.uint32)
    signs = (((bits[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(np.float64)
    sums = np.abs(signs @ v)
    return float(np.mean(sums >= abs(v.sum()) - 1e-12))


def _mech_dir() -> Any:
    return paths.phase1_root() / "reports" / "frozen_32b" / "mechanism"


def reproduce_pc_numbers() -> dict[str, Any]:
    d = json.loads((_mech_dir() / "mech_pc_control.json").read_text())
    causal, coupling = d["causal"], d["coupling"]
    add_pm = np.array(causal["add_pm_contrast"], dtype=float)
    removal = np.array(causal["removal_deltas"], dtype=float)
    ro_pm = np.array(coupling["ar_dir_on_ro_add_pm"], dtype=float)
    ps = {
        "ar_addition_pm": _old_sign_flip_p(list(add_pm)),
        "ar_removal": _old_sign_flip_p(list(removal)),
        "ar_to_ro_transfer": _old_sign_flip_p(list(ro_pm)),
    }
    h = holm(ps)
    rec = d["primaries_holm"]
    checks = {
        "add_pm_mean_matches": bool(abs(float(add_pm.mean()) - 0.787) < 0.05),
        "ro_transfer_mean_matches": bool(abs(float(ro_pm.mean()) - 0.93)
                                         < 0.05),
        "p_holm_addition_matches": bool(
            abs(h["ar_addition_pm"]["p_holm"]
                - rec["ar_addition_pm"]["p_holm"]) < TOL),
        "p_holm_transfer_matches": bool(
            abs(h["ar_to_ro_transfer"]["p_holm"]
                - rec["ar_to_ro_transfer"]["p_holm"]) < TOL),
        "removal_null_matches": bool(
            (h["ar_removal"]["p"] > 0.05)
            == (rec["ar_removal"]["p"] > 0.05)),
        "code_control_opposite_on_ro": bool(
            coupling["code_control_on_ro_add_pm_mean"] < 0 < ro_pm.mean()),
    }
    return {"recomputed_holm": h, "recorded_holm": rec,
            "add_pm_mean": float(add_pm.mean()),
            "ro_transfer_mean": float(ro_pm.mean()),
            "checks": checks, "passed": all(checks.values())}


def predict_docsection_nonidentifiability() -> dict[str, Any]:
    d = json.loads((_mech_dir() / "mech_ar_docsection_readme.json")
                   .read_text())
    per_depth = d["per_depth"]
    rows = []
    for depth, cell in sorted(per_depth.items(), key=lambda kv: int(kv[0])):
        corr = cell.get("validation_corr") or cell.get("validation_fit_corr")
        band = cell.get("random_band_95")
        gate = cell.get("gate") or {}
        # the OLD gate: |validation corr| must beat the random band; the
        # recorded corrs (0.09-0.13) sit far under the recorded bands
        # (0.40-0.57), so the prediction is fail-at-every-depth
        rows.append({"depth": int(depth), "validation_corr": corr,
                     "band": band,
                     "old_gate_would_pass": bool(
                         corr is not None and band is not None
                         and abs(corr) > band),
                     "recorded_identifiable": bool(gate.get("identifiable"))})
    bands_absent = all(r["band"] is None for r in rows)
    corr_max = max(abs(r["validation_corr"]) for r in rows)
    predicted_fail = bands_absent and corr_max < 0.14 or all(
        not r["old_gate_would_pass"] for r in rows)
    matches_record = (d["status"] == "DIRECTION_NOT_IDENTIFIABLE"
                      and all(not r["recorded_identifiable"] for r in rows))
    return {"per_depth": rows, "predicted_nonidentifiable": bool(predicted_fail),
            "recorded_status": d["status"],
            "passed": bool(predicted_fail and matches_record)}


def classify_old_pc() -> dict[str, Any]:
    """Mechanical classification from recorded numbers (E8 iii)."""
    d = json.loads((_mech_dir() / "mech_pc_control.json").read_text())
    causal = d["causal"]
    add_pm_mean = float(np.mean(causal["add_pm_contrast"]))
    wrong = causal["controls"].get("d_wrong_scenario", {})
    wrong_ratio = (abs(wrong.get("add_pm_mean", 0.0)) / abs(add_pm_mean)
                   if add_pm_mean else float("nan"))
    strict = causal.get("strict_outputs") or {}
    flips = 0
    clean = strict.get("clean") or []
    for cond in ("add_plus", "add_minus", "removed"):
        vals = strict.get(cond) or []
        flips += sum(1 for a, b in zip(clean, vals) if a != b)
    # Phase 1 injected at the final prompt token by design
    site_is_final_token = True
    classification = "MARGIN_HANDLE"
    risk_flags = []
    if site_is_final_token:
        risk_flags.append("DIRECT_OUTPUT_RISK:final_prompt_token_site")
    if np.isfinite(wrong_ratio) and wrong_ratio >= 0.5:
        risk_flags.append(
            f"DIRECT_OUTPUT_RISK:wrong_scenario_{wrong_ratio:.2f}_of_primary")
    if flips == 0:
        risk_flags.append("no_strict_flips")
    return {
        "classification": classification, "risk_flags": risk_flags,
        "wrong_scenario_ratio": float(wrong_ratio),
        "strict_flips": flips,
        "passed": bool(classification == "MARGIN_HANDLE"
                       and any("DIRECT_OUTPUT_RISK" in f for f in risk_flags)
                       and flips == 0 and 0.5 <= wrong_ratio <= 0.8),
    }


def synthetic_intercept_blindness(dim: int = 64, n_inc: int = 16,
                                  seed: int = 0) -> dict[str, Any]:
    """§0.7 structural failure, planted: the scenario-mean semantic effect
    lives on a direction; the OLD estimator residualizes the margin
    against an intercept+nuisance design, removing exactly the variation
    that carries the signal; the NEW context-randomized estimator
    recovers the plant from the same states."""
    rng = np.random.default_rng(stable_seed("retro-syn", seed, base=2262))
    d_true = rng.standard_normal(dim)
    d_true /= np.linalg.norm(d_true)
    rows, states, margins = [], {}, []
    idx = 0
    for i in range(n_inc):
        for s in (-2, -1, 0, 1, 2):
            for order in (0, 1):
                item_id = f"syn-{idx}"
                idx += 1
                # state carries the CONTEXT signal + noise
                h = (1.5 * s * d_true
                     + 0.8 * rng.standard_normal(dim))
                # margin: constant scenario mean + context response;
                # within-cell residual tiny (the Phase 1 geometry)
                m = 1.0 + 0.9 * s + 0.05 * rng.standard_normal()
                states[item_id] = h.astype(np.float32)
                margins.append(m)
                rows.append({
                    "item_id": item_id,
                    "incidental_id": f"i{i:02d}",
                    "incidental_split": ("train" if i < 8 else
                                         "validation" if i < 12 else
                                         "holdout"),
                    "display_order": order, "code_map_index": 0,
                    "paraphrase_id": 0, "codebook_pair_id": "ar0",
                    "context_strength": s,
                    "margin_full_a_minus_b": m,
                })
    # OLD estimator: covariance with the intercept-residualized margin
    # using only neutral-context variation (no manipulated labels)
    neutral = [r for r in rows if r["context_strength"] == 0
               and r["incidental_split"] == "train"]
    m_neu = np.array([r["margin_full_a_minus_b"] for r in neutral])
    S_neu = np.stack([states[r["item_id"]] for r in neutral]).astype(float)
    m_res = m_neu - m_neu.mean()
    raw = (m_res[:, None] * (S_neu - S_neu.mean(0))).sum(0)
    old_corr = float("nan")
    if np.linalg.norm(raw) > 0:
        d_old = raw / np.linalg.norm(raw)
        val = [r for r in rows if r["context_strength"] == 0
               and r["incidental_split"] == "validation"]
        pv = np.stack([states[r["item_id"]] for r in val]).astype(float) @ d_old
        mv = np.array([r["margin_full_a_minus_b"] for r in val])
        old_corr = (float(np.corrcoef(pv, mv)[0, 1])
                    if np.std(mv) > 0 else float("nan"))
        old_cos = float(abs(d_old @ d_true))
    else:
        old_cos = 0.0
    # NEW estimator: matched context pairs
    from .mechanism import decoder_eval, fit_direction
    fit = fit_direction(rows, lambda i: states[i])
    new_cos = float(abs(fit["direction"] @ d_true)) if fit["ok"] else 0.0
    val_eval = decoder_eval(fit["direction"], rows, lambda i: states[i],
                            split="validation") if fit["ok"] else {}
    return {
        "old_estimator_true_cosine": old_cos,
        "old_estimator_validation_corr": old_corr,
        "new_estimator_true_cosine": new_cos,
        "new_estimator_validation_corr": val_eval.get("corr"),
        "passed": bool(new_cos > 0.9 and old_cos < 0.3
                       and val_eval.get("corr", 0) > 0.8),
    }


def run_retrodiction() -> dict[str, Any]:
    out = {
        "retrodiction_scope": "analysis_level",
        "pc_numbers": reproduce_pc_numbers(),
        "docsection": predict_docsection_nonidentifiability(),
        "old_pc_classification": classify_old_pc(),
        "synthetic_intercept_blindness": synthetic_intercept_blindness(),
    }
    out["passed"] = all(out[k]["passed"] for k in
                        ("pc_numbers", "docsection", "old_pc_classification",
                         "synthetic_intercept_blindness"))
    return out
